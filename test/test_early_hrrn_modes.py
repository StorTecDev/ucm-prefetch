"""Worker HRRN and mixed admission are independent of READY work ordering."""

import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_early_batch_pipeline import Capability, Control, early, enqueue
from test_early_native_scheduler import attach, requests_of_sizes, scheduler_factory
from test_early_ready_reuse import connector
from test_early_request_async import constructor_config
from vllm.v1.core.sched.scheduler import Scheduler

from ucm.integration.vllm import early_scheduling
from ucm.integration.vllm.early_connector import (
    EarlyDirectConnector,
    EarlyLayerWiseConnector,
)
from ucm.integration.vllm.early_scheduling import ReadyPriority, install_scheduler_policy
from ucm.integration.vllm.ucm_connector import KVConnectorRole


@pytest.fixture
def hrrn_config():
    return {
        "enable_early_prefetch": True,
        "early_prefetch_request_first": False,
        "early_ready_priority": False,
        "early_pending_hrrn": True,
    }


@pytest.mark.parametrize("cls", [EarlyDirectConnector, EarlyLayerWiseConnector])
def test_mixed_hrrn_configuration_installs_admission_without_ready_ordering(
    constructor_config, hrrn_config, monkeypatch, cls
):
    config, launch, _ = constructor_config
    launch.update(hrrn_config)
    config.scheduler_config.get_scheduler_cls = lambda: Scheduler
    patch = Mock()
    monkeypatch.setattr(early_scheduling, "patch_scheduler_class", patch)
    monkeypatch.setattr(
        early_scheduling, "install_scheduler_policy", install_scheduler_policy
    )
    cls(config, KVConnectorRole.SCHEDULER)
    patch.assert_called_once_with(Scheduler)
    policy = ReadyPriority(launch)
    assert policy.pending_enabled and policy.active and not policy.enabled


@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize("budget", [512, 8192])
@pytest.mark.parametrize("mixed", [False, True])
def test_hrrn_configuration_preserves_ready_positions_and_controls_pending_gate(
    connector, scheduler_factory, hrrn_config, async_scheduling, budget, mixed
):
    scheduler = scheduler_factory(
        max_num_seqs=4, max_num_batched_tokens=budget,
        async_scheduling=async_scheduling,
    )
    requests = requests_of_sizes(*([8192] * 4))
    attach(connector, scheduler, requests,
           ("READY", "PENDING", "READY", "PENDING"), (48, 48, 60, 48),
           ready_priority=False)
    hrrn_config["early_pending_hrrn"] = mixed
    policy = connector._early.scheduling = ReadyPriority(hrrn_config)
    now = time.monotonic()
    for request_id, work, age in (("1", 1, 1), ("3", 8, 200)):
        job = connector._early.jobs[request_id]
        job.pending_hit, job.remaining_l2_work = 48, work
        job.entered = now - age
    output = scheduler.schedule()
    if budget == 512:
        # A smaller READY suffix must not displace the first native candidate.
        expected = {"0": 512}
    elif mixed:
        expected = {"0": 2048, "3": 2048, "2": 512, "1": 2048}
    else:
        expected = {"0": 2048, "2": 512}
    assert output.num_scheduled_tokens == expected
    assert list(output.num_scheduled_tokens) == list(expected)
    assert output.total_num_scheduled_tokens <= budget
    assert not policy.bypasses and not policy.last_rejection
    assert all(connector._early.jobs[key].state == "PENDING" for key in ("1", "3"))


@pytest.mark.parametrize("mixed", [False, True])
def test_worker_hrrn_survives_both_admission_modes(hrrn_config, mixed):
    hrrn_config["early_pending_hrrn"] = mixed
    store = Mock()
    store.domain_info.return_value = dict(block_bytes=1, shard_bytes=1)
    connector = SimpleNamespace(
        store=store, launch_config=hrrn_config, use_request_async_load=False,
        tp_rank=0, tp_size=8, is_mla=True, unique_id="hrrn-modes-unit",
        request_hasher=Mock(),
        _vllm_config=SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_index=0)),
    )
    worker = early.EarlyWorker(connector, Control)
    try:
        assert not worker.request_first
        assert worker.pending_backfill is mixed
        # Hold the producer lock so the real worker cannot consume the test queue.
        with worker.condition:
            try:
                oldest, _ = enqueue(worker, "oldest", 1, Capability(100), age=100)
                preferred, part = enqueue(worker, "preferred", 2, Capability(10), age=50)
                smallest, _ = enqueue(worker, "smallest", 3, Capability(1), age=1)
                worker._choose()
                assert worker.selected == [(preferred, part)]
                assert preferred is not oldest and preferred is not smallest
                part.started = True
                part.stats = {"can_submit": False}
                worker._publish_progress(preferred)
                assert bool(worker.control.messages) is mixed
                if mixed:
                    assert worker.control.messages[-1]["op"] == "progress"
                    assert worker.control.messages[-1]["remaining_l2_work"] == 10
                assert not preferred.ready
            finally:
                worker.preparing.clear()
                worker.selected.clear()
                worker.hrrn_requests.clear()
    finally:
        worker.close()
    assert not worker.data_thread.is_alive()
