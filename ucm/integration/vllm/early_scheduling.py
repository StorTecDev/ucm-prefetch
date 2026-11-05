"""Opt-in bounded Early admission in supported native vLLM waiting loops."""

from __future__ import annotations

import ast
import bisect
import functools
import heapq
import inspect
import textwrap
import time
import weakref
from dataclasses import dataclass


def hrrn_ratio(entered, now, work, minimum_work):
    """Compute HRRN from queue age and a consistent service-work estimate.

    Args:
        entered: Monotonic enqueue time of the logical request.
        now: Monotonic time shared by candidates in the scheduling round.
        work: Estimated service work in the caller's common work units.
        minimum_work: Positive floor in the same units, including drained work.

    Returns:
        A response ratio where larger values receive priority.
    """
    return 1 + (now - entered) / max(work, minimum_work)


class ReadyPriority:
    def __init__(self, config):
        self.enabled = config.get("early_ready_priority", False)
        if type(self.enabled) is not bool:
            raise ValueError("early_ready_priority must be a boolean")
        self.pending_enabled = config.get("early_pending_hrrn", self.enabled)
        if type(self.pending_enabled) is not bool:
            raise ValueError("early_pending_hrrn must be a boolean")
        self.max_candidates = config.get("early_ready_max_candidates", 64)
        self.max_bypasses = config.get("early_ready_max_bypasses", 8)
        for name in ("max_candidates", "max_bypasses"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"early_ready_{name} must be a positive integer")
        self.bypasses = weakref.WeakKeyDictionary()
        self.last_rejection = weakref.WeakKeyDictionary()
        self.round = 0

    @property
    def active(self):
        return self.enabled or self.pending_enabled


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
        self.now = None
        self.pending_requests = set()
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
        pending_rank = self._pending_rank(candidate, job)
        if pending_rank is not None:
            return (2, 1, pending_rank[0], 0, 0, candidate.order)
        if age_due:
            return (0, arrival, 0, 0, 0, candidate.order)
        if saved:
            return (1, rejected, remaining, arrival, 0, candidate.order)
        if job is not None and job.state == "READY" and not job.dispatched:
            return (2, 0, 0, 0, 0, candidate.order)
        if getattr(self.scheduler.policy, "value", self.scheduler.policy) == "priority":
            return (
                3,
                rejected,
                getattr(request, "priority", 0),
                arrival,
                0,
                candidate.order,
            )
        return (3, rejected, 0, candidate.order, 0, candidate.order)

    def _pending_rank(self, candidate, job):
        if (self.policy.pending_enabled and job is not None
                and job.state == "PENDING" and not job.dispatched
                and getattr(job, "pending_hit", None) is not None):
            ratio = hrrn_ratio(job.entered, self.now, job.remaining_l2_work, 1)
            return (-ratio, candidate.order)
        return None

    def _pending_choices(self, queues):
        groups = [[_Candidate(request, queue, 0) for request in queue]
                  for queue in queues]
        candidates = []
        if (len(groups) == 2 and
                getattr(self.scheduler.policy, "value", self.scheduler.policy) == "priority"):
            skipped, waiting = groups
            i = j = 0
            # Match the native two-head merge, including skipped-first ties.
            while i < len(skipped) and j < len(waiting):
                if waiting[j].request < skipped[i].request:
                    candidates.append(waiting[j])
                    j += 1
                else:
                    candidates.append(skipped[i])
                    i += 1
            candidates.extend(skipped[i:])
            candidates.extend(waiting[j:])
        else:
            candidates = [candidate for group in groups for candidate in group]
        ranked = []
        for order, candidate in enumerate(candidates):
            candidate.order = order
            rank = self._pending_rank(candidate, self.early.probe(candidate.request))
            if rank is not None:
                ranked.append((rank, candidate))
                self.pending_requests.add(candidate.request)
        self.candidates = candidates
        choices = candidates[:self.policy.max_candidates]
        slots = [index for index, candidate in enumerate(choices)
                 if candidate.request in self.pending_requests]
        # Reorder only eligible PENDING slots; other native positions stay fixed.
        best = heapq.nsmallest(len(slots), ranked, key=lambda item: item[0])
        for index, (_, candidate) in zip(slots, best):
            choices[index] = candidate
        return choices

    def can_backfill(self, request):
        return self.policy.enabled or request in self.pending_requests

    def _prepare(self):
        # One time snapshot keeps equal-score ties independent of scan order.
        self.now = time.monotonic()
        self.policy.round += 1
        queues = [self.scheduler.waiting]
        skipped = getattr(self.scheduler, "skipped_waiting", None)
        if skipped is not None:
            queues.insert(0, skipped)
        # Bound matching/allocation to B candidates after native enumeration.
        # Priority-queue iteration itself can still cost O(N log N).
        if self.policy.enabled:
            self.candidates = []
            for queue in queues:
                for request in queue:
                    self.candidates.append(_Candidate(request, queue, len(self.candidates)))
            choices = heapq.nsmallest(
                self.policy.max_candidates, self.candidates, key=self._rank
            )
        else:
            choices = self._pending_choices(queues)
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
        if self.candidates is None or not self.policy.enabled:
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
        "padded_num_tokens > request_token_budget",
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
        exit_branch = ast.parse(
            "if _ucm_waiting.can_backfill(request):\n"
            "    continue\n"
            "else:\n"
            "    break"
        ).body[0]
        node.body[breaks[0]] = ast.copy_location(exit_branch, node.body[breaks[0]])
        if ast.unparse(node.test) == "num_new_tokens == 0":
            # Encoder admission can touch cached inputs before discovering that
            # a later input cannot fit. Undo those references before backfill.
            cleanup = ast.parse(
                "if request.has_encoder_inputs:\n"
                "    self.encoder_cache_manager.free(request)"
            ).body[0]
            exit_branch.body.insert(0, ast.copy_location(cleanup, node))
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
    """Install admission only for schedulers using an enabled Early connector.

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
        # Patching is process-wide, but admission policy belongs to each
        # connector. An instance with both options off must bypass the entire adapter.
        if early is None or not early.scheduling.active:
            return original(self, *args, **kwargs)
        waiting = WaitingPass(self, early)
        output = adapted(self, *args, _ucm_waiting=waiting, **kwargs)
        waiting.finish()
        return output

    schedule._ucm_ready_priority = True
    scheduler_class.schedule = schedule


def _native_admission_scheduler(scheduler_class, vllm_config):
    """Resolve a verified Ascend pass-through to the native admission method.

    Args:
        scheduler_class: Scheduler exported by the installed vLLM module.
        vllm_config: Full configuration used to resolve Ascend balance settings.

    Returns:
        The native class whose admission loop must be adapted.

    Raises:
        ValueError: An Ascend wrapper enables balance or changes its fast path.
    """
    if (scheduler_class.__module__, scheduler_class.__name__) != (
        "vllm_ascend.patch.platform.patch_balance_schedule", "BalanceScheduler"
    ):
        return scheduler_class

    from vllm_ascend.patch.platform.patch_balance_schedule import (
        _balance_scheduling_enabled,
    )

    if vllm_config is None or _balance_scheduling_enabled(vllm_config):
        raise ValueError("Early admission requires Ascend balance scheduling disabled")
    bases = scheduler_class.__bases__
    if len(bases) != 1 or (bases[0].__module__, bases[0].__name__) != (
        "vllm.v1.core.sched.scheduler", "Scheduler"
    ):
        raise ValueError("unrecognized Ascend balance scheduler base")
    method = scheduler_class.__dict__.get("schedule")
    if method is None:
        raise ValueError("unrecognized Ascend balance scheduler fast path")
    function = ast.parse(textwrap.dedent(inspect.getsource(method))).body[0]
    expected = ast.parse(
        "if not self._balance_enabled:\n"
        "    return super().schedule(throttle_prefills)"
    ).body[0]
    if not function.body or ast.dump(function.body[0]) != ast.dump(expected):
        raise ValueError("unrecognized Ascend balance scheduler fast path")
    # Adapting the wrapper would lose its __class__ closure and miss the
    # disabled branch's delegation. Preserve it and adapt the actual callee.
    return bases[0]


def install_scheduler_policy(config, launch_config, *, vllm_config=None):
    """Install admission when either option is enabled for this connector.

    Args:
        config: The vLLM scheduler configuration.
        launch_config: The UCM connector configuration at the YAML root.
        vllm_config: Full configuration for installed platform scheduler wrappers.
    """
    if not ReadyPriority(launch_config).active:
        return
    from vllm import __version__

    if __version__.split("+", 1)[0] not in ("0.28.0", "0.30.0"):
        return
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    from vllm.v1.core.sched.scheduler import Scheduler

    scheduler_class = config.get_scheduler_cls()
    if scheduler_class not in (Scheduler, AsyncScheduler):
        raise ValueError("Early admission requires the native vLLM scheduler")
    # AsyncScheduler inherits admission and retains its native output handling.
    native = _native_admission_scheduler(Scheduler, vllm_config)
    patch_scheduler_class(native)
