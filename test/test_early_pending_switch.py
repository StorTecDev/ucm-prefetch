"""Independent PENDING HRRN admission without READY preference."""

import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_early_native_scheduler import attach, requests_of_sizes, scheduler_factory
from test_early_ready_reuse import connector
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.core.sched.scheduler import Scheduler

from ucm.integration.vllm import early_scheduling
from ucm.integration.vllm.early_scheduling import ReadyPriority, install_scheduler_policy


def configure(connector, **options):
    connector._early.scheduling = ReadyPriority(options)


def pending(connector, request_id, hit, work, age):
    job = connector._early.jobs[request_id]
    job.pending_hit, job.remaining_l2_work = hit, work
    job.entered = time.monotonic() - age


@pytest.mark.parametrize("ready,pending_flag,expected", [
    (False, None, (False, False, False)),
    (False, True, (False, True, True)),
    (True, None, (True, True, True)),
    (True, False, (True, False, True)),
    (True, True, (True, True, True)),
])
def test_independent_switch_defaults(ready, pending_flag, expected):
    options = {"early_ready_priority": ready}
    if pending_flag is not None:
        options["early_pending_hrrn"] = pending_flag
    policy = ReadyPriority(options)
    assert (policy.enabled, policy.pending_enabled, policy.active) == expected


@pytest.mark.parametrize("value", [None, 0, 1, "false", "true"])
def test_pending_switch_rejects_nonbooleans(value):
    with pytest.raises(ValueError, match="early_pending_hrrn must be a boolean"):
        ReadyPriority({"early_pending_hrrn": value})


def test_pending_only_installs_adapter_without_ready_priority(monkeypatch):
    patch = Mock()
    monkeypatch.setattr(early_scheduling, "patch_scheduler_class", patch)
    config = SimpleNamespace(get_scheduler_cls=lambda: Scheduler)
    install_scheduler_policy(config, {"early_pending_hrrn": True})
    patch.assert_called_once_with(Scheduler)


@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize("limit", [2, 64])
def test_pending_only_keeps_native_slots_and_selects_hrrn_across_queue(
    connector, scheduler_factory, async_scheduling, limit
):
    scheduler = scheduler_factory(max_num_seqs=4, max_num_batched_tokens=8192,
                                  async_scheduling=async_scheduling)
    requests = requests_of_sizes(*([8192] * 4))
    attach(connector, scheduler, requests,
           ("PENDING", "READY", "PENDING", "FAILED"), (48, 48, 48, 0),
           ready_priority=False)
    configure(connector, early_pending_hrrn=True, early_ready_max_candidates=limit)
    pending(connector, "0", 48, 1, 1)
    pending(connector, "2", 48, 8, 200)
    output = scheduler.schedule()
    assert list(output.num_scheduled_tokens) == (["2", "1"] if limit == 2 else ["2", "1", "0", "3"])
    assert output.total_num_scheduled_tokens <= 8192
    assert not connector._early.scheduling.enabled


def test_pending_only_does_not_promote_ready_over_native_fallback(
    connector, scheduler_factory
):
    scheduler = scheduler_factory(max_num_batched_tokens=4096)
    requests = requests_of_sizes(8192, 8192)
    adapter = attach(connector, scheduler, requests, ("FAILED", "READY"), (0, 48),
                     ready_priority=False)
    configure(connector, early_pending_hrrn=True)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"0": 4096}
    assert adapter.get_num_new_matched_tokens.call_count == 1


@pytest.mark.parametrize("is_pending", [False, True])
def test_pending_only_preserves_nonpending_native_allocation_stop(
    connector, scheduler_factory, is_pending
):
    scheduler = scheduler_factory(num_blocks=16)
    requests = requests_of_sizes(8192, 1024)
    adapter = attach(connector, scheduler, requests,
                     ("PENDING" if is_pending else "READY", "PENDING"), (60, 4),
                     ready_priority=False)
    configure(connector, early_pending_hrrn=True)
    if is_pending:
        pending(connector, "0", 60, 1, 100)
    pending(connector, "1", 4, 1, 1)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == ({"1": 512} if is_pending else {})
    assert adapter.get_num_new_matched_tokens.call_count == (2 if is_pending else 1)


def test_pending_only_merges_native_priority_queues_before_reordering_pending(
    connector, scheduler_factory
):
    scheduler = scheduler_factory(max_num_seqs=4, max_num_batched_tokens=16384)
    scheduler.policy = SchedulingPolicy.PRIORITY
    scheduler.waiting = create_request_queue(scheduler.policy)
    scheduler.skipped_waiting = create_request_queue(scheduler.policy)
    requests = requests_of_sizes(*([8192] * 4))
    for request, priority in zip(requests, (10, -5, 0, -10)):
        request.priority = priority
    attach(connector, scheduler, requests,
           ("PENDING", "READY", "PENDING", "FAILED"), (48, 48, 48, 0),
           ready_priority=False)
    for request in (requests[2], requests[3]):
        scheduler.waiting.remove_request(request)
        scheduler.skipped_waiting.add_request(request)
    configure(connector, early_pending_hrrn=True)
    pending(connector, "0", 48, 8, 200)
    pending(connector, "2", 48, 1, 1)
    output = scheduler.schedule()
    assert list(output.num_scheduled_tokens) == ["3", "1", "0", "2"]


@pytest.mark.parametrize("enabled", [False, True])
def test_pending_only_controls_all_pending_admission(connector, scheduler_factory, enabled):
    scheduler = scheduler_factory(max_num_seqs=2, max_num_batched_tokens=4096)
    requests = requests_of_sizes(8192, 8192)
    attach(connector, scheduler, requests, ("PENDING", "PENDING"), (48, 48),
           ready_priority=False)
    configure(connector, early_pending_hrrn=enabled)
    pending(connector, "0", 48, 1, 1)
    pending(connector, "1", 48, 8, 200)
    output = scheduler.schedule()
    assert list(output.num_scheduled_tokens) == (["1", "0"] if enabled else [])


def test_explicit_pending_disable_preserves_ready_only(connector, scheduler_factory):
    scheduler = scheduler_factory()
    requests = requests_of_sizes(8192, 8192)
    attach(connector, scheduler, requests, ("READY", "PENDING"), (48, 48))
    configure(connector, early_ready_priority=True, early_pending_hrrn=False)
    pending(connector, "1", 48, 1, 100)
    assert scheduler.schedule().num_scheduled_tokens == {"0": 2048}
