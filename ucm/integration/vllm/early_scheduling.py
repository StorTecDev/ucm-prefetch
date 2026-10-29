"""Bounded READY preference in the vLLM 0.28.0 waiting admission loop."""

from __future__ import annotations

import ast
import bisect
import functools
import heapq
import inspect
import textwrap
import weakref
from dataclasses import dataclass


class ReadyPriority:
    def __init__(self, config):
        self.max_candidates = config.get("early_ready_max_candidates", 64)
        self.max_bypasses = config.get("early_ready_max_bypasses", 8)
        for name in ("max_candidates", "max_bypasses"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"early_ready_{name} must be a positive integer")
        self.bypasses = weakref.WeakKeyDictionary()
        self.last_rejection = weakref.WeakKeyDictionary()
        self.round = 0


@dataclass
class _Candidate:
    request: object
    queue: object
    order: int

    def peek_request(self):
        return self.request

    def pop_request(self):
        queue = self.queue
        if queue.peek_request() is self.request:
            return queue.pop_request()
        queue.remove_request(self.request)
        return self.request


class WaitingPass:
    def __init__(self, scheduler, early):
        self.scheduler = scheduler
        self.early = early
        self.policy = early.scheduling
        self.candidates = None
        self.remaining = None
        self.selected = None
        self.attempted = set()
        self.admissions = []
        self.admitted_requests = set()

    def _rank(self, candidate):
        request = candidate.request
        arrival = getattr(request, "arrival_time", 0)
        rejected = self.policy.last_rejection.get(request, 0)
        age_due = self.policy.bypasses.get(request, 0) >= self.policy.max_bypasses
        job = self.early.probe(request)
        saved = 0
        if job is not None and job.state == "READY" and not job.dispatched:
            saved = min(
                job.hit * self.early.connector.hash_block_size,
                max(0, request.num_tokens - 1),
            )
        remaining = max(1, request.num_tokens - saved)
        if age_due:
            return (0, arrival, 0, 0, 0, candidate.order)
        if saved:
            return (1, rejected, remaining, arrival, 0, candidate.order)
        if getattr(self.scheduler.policy, "value", self.scheduler.policy) == "priority":
            return (
                2,
                rejected,
                getattr(request, "priority", 0),
                arrival,
                0,
                candidate.order,
            )
        return (2, rejected, 0, candidate.order, 0, candidate.order)

    def _prepare(self):
        self.policy.round += 1
        queues = [self.scheduler.waiting]
        skipped = getattr(self.scheduler, "skipped_waiting", None)
        if skipped is not None:
            queues.insert(0, skipped)
        self.candidates = []
        for queue in queues:
            for request in queue:
                self.candidates.append(_Candidate(request, queue, len(self.candidates)))
        # Ranking costs O(N log B) after native queue enumeration (O(N log N)
        # for priority queues). Bound matching/allocation to B candidates;
        # arbitrary native queue removal can still cost O(N).
        choices = heapq.nsmallest(
            self.policy.max_candidates, self.candidates, key=self._rank
        )
        self.remaining = iter(choices)
        self.left = len(choices)

    def can_scan(self):
        return self.remaining is None or self.left > 0

    def choose_queue(self):
        if self.remaining is None:
            self._prepare()
        self.selected = next(self.remaining)
        self.left -= 1
        self.attempted.add(self.selected.request)
        return self.selected

    def admitted(self, request):
        self.admissions.append(getattr(request, "arrival_time", 0))
        self.admitted_requests.add(request)

    def finish(self):
        if self.candidates is None:
            return
        admitted = sorted(self.admissions)
        for candidate in self.candidates:
            request = candidate.request
            count = (
                0 if request in self.attempted else self.policy.bypasses.get(request, 0)
            )
            # Count actual younger admissions, never polls, wakeups or failed
            # allocations. This grants an attempt, not an HBM reservation.
            count += len(admitted) - bisect.bisect_right(
                admitted, getattr(request, "arrival_time", 0)
            )
            self.policy.bypasses[request] = min(count, self.policy.max_bypasses)
            if request in self.admitted_requests:
                self.policy.last_rejection.pop(request, None)
            elif request in self.attempted:
                # With a bounded scan and no admissions, bypass counts cannot
                # advance. Rotate rejected candidates so later work is reached.
                self.policy.last_rejection[request] = self.policy.round


def _is_waiting_queue(node):
    return isinstance(node, ast.Attribute) and ast.unparse(node) == "self.waiting"


def _request_rejection(test):
    expression = ast.unparse(test)
    if expression in ("new_blocks is None", "num_new_tokens == 0"):
        return True
    if expression in (
        "not self.scheduler_config.enable_chunked_prefill and num_new_tokens > request_token_budget",
        "num_new_tokens > request_token_budget or num_computed_tokens + num_new_tokens > self.max_model_len",
    ):
        return True
    return False


class _WaitingLoop(ast.NodeTransformer):
    def __init__(self):
        self.selections = 0
        self.allocations = 0

    def visit_Call(self, node):
        function = node.func
        if isinstance(function, ast.Attribute):
            if ast.unparse(function) == "self._select_waiting_queue_for_scheduling":
                self.selections += 1
                return ast.copy_location(
                    ast.parse("_ucm_waiting.choose_queue()", mode="eval").body, node
                )
        return self.generic_visit(node)

    def visit_If(self, node):
        self.generic_visit(node)
        breaks = [
            index
            for index, statement in enumerate(node.body)
            if isinstance(statement, ast.Break)
        ]
        if not _request_rejection(node.test):
            if breaks and ast.unparse(node.test) not in (
                "input_budget <= draft_slots",
                "num_running >= self.max_num_running_reqs",
                "defer_prefills and num_computed_tokens < request.num_tokens - 1",
            ):
                raise ValueError("unrecognized native waiting-loop stop condition")
            return node
        if len(breaks) != 1:
            raise ValueError("unrecognized waiting-request rejection branch")
        # Keep native rollback statements before the exit. The rejected item
        # stays in its original queue; the pass never selects it twice.
        node.body[breaks[0]] = ast.copy_location(ast.Continue(), node.body[breaks[0]])
        if ast.unparse(node.test) == "num_new_tokens == 0":
            # Encoder admission can touch cached inputs before discovering that
            # a later input cannot fit. Undo those references before backfill.
            cleanup = ast.parse(
                "if request.has_encoder_inputs:\n"
                "    self.encoder_cache_manager.free(request)"
            ).body[0]
            node.body.insert(breaks[0], ast.copy_location(cleanup, node))
        if ast.unparse(node.test) == "new_blocks is None":
            self.allocations += 1
            admitted = ast.parse("_ucm_waiting.admitted(request)").body[0]
            return [node, ast.copy_location(admitted, node)]
        return node


def compile_schedule(original):
    """Adapt verified waiting-loop boundaries without copying a vLLM scheduler.

    Args:
        original: The native Scheduler.schedule method.

    Returns:
        A method accepting an additional private waiting-pass argument.

    Raises:
        ValueError: The method does not have the supported admission structure.
    """
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)
    function = tree.body[0]
    loops = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.While)
        and any(_is_waiting_queue(value) for value in ast.walk(node.test))
    ]
    if len(loops) != 1 or original.__closure__:
        raise ValueError("unrecognized native waiting loop or closure")
    loop = loops[0]
    transform = _WaitingLoop()
    transform.visit(loop)
    if transform.selections != 1 or transform.allocations != 1:
        raise ValueError("unrecognized native waiting selection/allocation boundaries")
    loop.test = ast.BoolOp(
        op=ast.And(),
        values=[loop.test, ast.parse("_ucm_waiting.can_scan()", mode="eval").body],
    )
    function.decorator_list = []
    function.args.kwonlyargs.append(ast.arg(arg="_ucm_waiting"))
    function.args.kw_defaults.append(None)
    ast.fix_missing_locations(tree)
    namespace = {}
    exec(compile(tree, "<ucm-early-schedule>", "exec"), original.__globals__, namespace)
    return namespace[function.name]


def patch_scheduler_class(scheduler_class):
    """Install READY admission only for schedulers using an Early connector.

    Args:
        scheduler_class: A native scheduler class with a supported waiting loop.
    """
    original = scheduler_class.schedule
    if getattr(original, "_ucm_ready_priority", False):
        return
    adapted = compile_schedule(original)

    @functools.wraps(original)
    def schedule(self, *args, **kwargs):
        from .early_prefetch import EarlyScheduler

        connector = getattr(self, "connector", None)
        early = None
        for _ in range(3):
            candidate = getattr(connector, "_early", None)
            if isinstance(candidate, EarlyScheduler):
                early = candidate
                break
            connector = getattr(connector, "connector", None)
        if early is None:
            return original(self, *args, **kwargs)
        waiting = WaitingPass(self, early)
        output = adapted(self, *args, _ucm_waiting=waiting, **kwargs)
        waiting.finish()
        return output

    schedule._ucm_ready_priority = True
    scheduler_class.schedule = schedule


def install_scheduler_policy(config):
    """Patch the configured scheduler while preserving native admission checks.

    Args:
        config: The vLLM scheduler configuration.
    """
    from vllm import __version__

    if __version__.split("+", 1)[0] != "0.28.0":
        return
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    from vllm.v1.core.sched.scheduler import Scheduler

    scheduler_class = config.get_scheduler_cls()
    if scheduler_class not in (Scheduler, AsyncScheduler):
        raise ValueError("Early READY priority requires the native vLLM scheduler")
    # AsyncScheduler inherits admission and retains its native output handling.
    patch_scheduler_class(Scheduler)
