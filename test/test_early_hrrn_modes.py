"""Baseline READY gating with complete worker HRRN on native vLLM schedulers."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_early_batch_pipeline import (
    Capability,
    Control,
    early,
    enqueue,
    wait,
    wait_answer,
)
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
    }


@pytest.mark.parametrize("cls", [EarlyDirectConnector, EarlyLayerWiseConnector])
@pytest.mark.parametrize("obsolete_flag", [None, False, True])
def test_hrrn_configuration_does_not_install_compute_adapter(
    constructor_config, hrrn_config, monkeypatch, cls, obsolete_flag
):
    config, launch, _ = constructor_config
    launch.update(hrrn_config)
    if obsolete_flag is not None:
        launch["early_pending_hrrn"] = obsolete_flag
    config.scheduler_config.get_scheduler_cls = Mock(return_value=Scheduler)
    patch = Mock()
    monkeypatch.setattr(early_scheduling, "patch_scheduler_class", patch)
    monkeypatch.setattr(
        early_scheduling, "install_scheduler_policy", install_scheduler_policy
    )
    cls(config, KVConnectorRole.SCHEDULER)
    patch.assert_not_called()
    config.scheduler_config.get_scheduler_cls.assert_not_called()
    assert not ReadyPriority(launch).enabled


@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize("ready_priority", [False, True])
@pytest.mark.parametrize("obsolete_flag", [None, True])
def test_pinned_pending_waits_for_ready_before_native_admission(
    connector, scheduler_factory, hrrn_config, async_scheduling,
    ready_priority, obsolete_flag
):
    scheduler = scheduler_factory(async_scheduling=async_scheduling)
    requests = requests_of_sizes(8192)
    attach(connector, scheduler, requests, ("PENDING",), (48,),
           ready_priority=ready_priority)
    hrrn_config["early_ready_priority"] = ready_priority
    if obsolete_flag is not None:
        hrrn_config["early_pending_hrrn"] = obsolete_flag
    connector._early.scheduling = ReadyPriority(hrrn_config)
    connector._early.dp_rank = 0
    job = connector._early.jobs["0"]
    # Stale progress fields must never authorize an unfinished external prefix.
    job.pending_hit, job.remaining_l2_work = 48, 0
    assert scheduler.schedule().num_scheduled_tokens == {}
    assert job.state == "PENDING"
    connector._early.control.send.assert_not_called()
    connector._early.receive(dict(op="result", key=job.key, state="READY", hit=48))
    assert job.state == "READY"
    assert scheduler.schedule().num_scheduled_tokens == {"0": 2048}
    connector.store.store_.LookupOnPrefix.assert_not_called()


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_hrrn_loading_keeps_native_ready_order(
    connector, scheduler_factory, hrrn_config, monkeypatch, async_scheduling
):
    scheduler = scheduler_factory(
        max_num_batched_tokens=512, async_scheduling=async_scheduling
    )
    requests = requests_of_sizes(8192, 8192)
    attach(connector, scheduler, requests, ("READY", "READY"), (48, 60),
           ready_priority=False)
    connector._early.scheduling = ReadyPriority(hrrn_config)
    waiting = Mock(side_effect=AssertionError("disabled adapter was invoked"))
    monkeypatch.setattr(early_scheduling, "WaitingPass", waiting)
    assert scheduler.schedule().num_scheduled_tokens == {"0": 512}
    waiting.assert_not_called()


def make_worker(config):
    store = Mock()
    store.domain_info.return_value = dict(block_bytes=1, shard_bytes=1)
    connector = SimpleNamespace(
        store=store, launch_config=config, use_request_async_load=False,
        tp_rank=0, tp_size=8, is_mla=True, unique_id="hrrn-modes-unit",
        request_hasher=Mock(),
        _vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_index=0)
        ),
    )
    return early.EarlyWorker(connector, Control)


@pytest.mark.parametrize("obsolete_flag", [None, False, True])
def test_worker_retains_hrrn_without_compute_admission(hrrn_config, obsolete_flag):
    if obsolete_flag is not None:
        hrrn_config["early_pending_hrrn"] = obsolete_flag
    worker = make_worker(hrrn_config)
    try:
        assert not worker.request_first
        # Hold the producer lock so the real worker cannot consume the test queue.
        with worker.condition:
            try:
                enqueue(worker, "oldest", 1, Capability(100), age=100)
                preferred, part = enqueue(worker, "preferred", 2, Capability(10), age=50)
                enqueue(worker, "smallest", 3, Capability(1), age=1)
                worker._choose()
                assert worker.selected == [(preferred, part)]
                assert not preferred.ready
                assert not worker.control.messages
            finally:
                worker.preparing.clear()
                worker.selected.clear()
                worker.hrrn_requests.clear()
    finally:
        worker.close()
    assert not worker.data_thread.is_alive()


def test_background_loading_completes_without_early_pending_publication(hrrn_config):
    worker = make_worker(hrrn_config)
    gate = threading.Event()
    capability = Capability(2, gate=gate)
    capability.hit = 32
    try:
        with worker.condition:
            job, _ = enqueue(worker, "background", 1, capability)
            worker.condition.notify_all()
        wait(lambda: capability.pending == 2)
        with worker.condition:
            assert not job.ready
            assert not worker.control.messages
        gate.set()
        wait_answer(worker, job.key)
        assert job.ready and capability.reads == 2
        assert [message["op"] for message in worker.control.messages] == ["answer"]
    finally:
        gate.set()
        worker.close()
    assert not worker.data_thread.is_alive()
