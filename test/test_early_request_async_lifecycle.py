"""Exercise native async dispatch with real Early bookkeeping and simulated I/O."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from ucm.integration.vllm import ucm_connector
from ucm.integration.vllm.early_connector import (
    EarlyConnectorMetadata,
    EarlyDirectConnector,
)
from ucm.integration.vllm.early_owner_worker import OwnerWorker
from ucm.integration.vllm.early_prefetch import EarlyWorker, WorkerBatch, WorkerJob
from ucm.integration.vllm.rank_consistency import RankConsistencyManager
from ucm.integration.vllm.ucm_connector import (
    KVConnectorRole,
    RequestDispatchMeta,
    UCMWorkerMetadata,
)
from ucm.store.pipeline import early_connector as early_store


@pytest.fixture
def worker_factory(monkeypatch):
    """Keep submission, polling, Wait receipts and release logic intact."""
    monkeypatch.setattr(ucm_connector.ucmmetrics, "update_stats", Mock())

    def make(owner=False):
        connector = object.__new__(EarlyDirectConnector)
        connector._role = KVConnectorRole.WORKER
        connector.use_request_async_load = True
        connector.use_layerwise = False
        connector.tp_rank = 0
        connector.is_mla = True
        connector._skip_null_vllm_blocks = False
        connector.block_data_size = 16
        connector._invalid_block_ids = set()
        connector._pending_load_tasks = {}
        connector._finished_async_load_req_ids = set()
        connector._async_dump_req_ids = set()
        connector._pending_dump_tasks = []
        connector._connector_worker_meta = UCMWorkerMetadata()
        connector._rank_consistency = RankConsistencyManager(is_scheduler=False)
        connector.kv_cache_layout = SimpleNamespace(
            extract_block_addrs=lambda _: np.zeros((1, 1), dtype=np.uint64)
        )
        pipeline = Mock()
        pipeline.Check.return_value = False
        pipeline.Wait.return_value = None
        submit = Mock(return_value=77)
        monkeypatch.setattr(early_store.ucmpipelinestore, "EarlyLoad", submit)
        store = object.__new__(early_store.EarlyPipelineStore)
        store._native_store = pipeline
        store.store_ = early_store._ReservedLoad(pipeline)
        store.release_reservation = Mock()
        connector.store = store
        worker = object.__new__(EarlyWorker)
        connector._early = worker
        worker.connector = connector
        worker.store = store
        worker.condition = threading.Condition()
        worker.request_keys = {}
        worker.control = Mock()
        block = b"a" * 16
        job = WorkerJob("generation", [block], ready=True, request_id="r")
        job.batches = [WorkerBatch(0, 0, {"handle": 41})]
        worker.jobs = {job.key: job}
        capability = {"ready": True, "layers": [0]} if owner else {}
        worker.owner_protocol = OwnerWorker(worker, capability, "w0.0")
        store.set_io_failure_handler(worker.owner_protocol.poison)
        tickets = {}
        if owner:
            job.owner, job.attempt, job.grant = "w0.0", "attempt", 1
            tickets["r"] = dict(
                key=job.key,
                attempt=job.attempt,
                owner=job.owner,
                grant=job.grant,
                expected=[block],
            )
        metadata = EarlyConnectorMetadata(
            {"r": RequestDispatchMeta(([block], [3]), ([], []), load_async=True)},
            set(),
            early_keys={"r": job.key},
            early_owners=tickets,
        )
        connector.bind_connector_metadata(metadata)
        return SimpleNamespace(
            connector=connector,
            worker=worker,
            store=store,
            pipeline=pipeline,
            submit=submit,
            job=job,
        )

    return make


def messages(state, operation):
    return [
        c.args[0]
        for c in state.worker.control.send.call_args_list
        if c.args[0]["op"] == operation
    ]


@pytest.mark.parametrize("owner", [False, True])
def test_async_load_survives_empty_metadata_and_reports_completion_once(
    worker_factory, owner
):
    state = worker_factory(owner)
    c = state.connector
    c.start_load_kv(None)
    state.submit.assert_called_once()
    state.pipeline.Wait.assert_not_called()
    assert set(c._pending_load_tasks) == {"r"}
    c.start_load_kv(None)
    state.submit.assert_called_once()
    c.bind_connector_metadata(EarlyConnectorMetadata({}, set()))
    assert c.get_finished(set()) == (None, None)
    state.store.release_reservation.assert_not_called()
    assert not messages(state, "consumed")
    state.pipeline.Check.return_value = True
    assert c.get_finished(set()) == (None, {"r"})
    state.pipeline.Wait.assert_called_once_with(77)
    assert not c._pending_load_tasks
    assert not c._rank_consistency._load_task_contexts
    assert not state.store.store_.receipts
    if owner:
        assert len(messages(state, "consumed")) == 1
        state.store.release_reservation.assert_not_called()
    else:
        state.store.release_reservation.assert_called_once_with(41)
        assert not state.worker.jobs
        assert not state.worker.request_keys
    assert c.get_finished(set()) == (None, None)
    state.pipeline.Wait.assert_called_once_with(77)


@pytest.mark.parametrize("terminal", ["finish", "preempt"])
def test_terminal_callback_does_not_release_a_pending_load(worker_factory, terminal):
    state = worker_factory()
    c = state.connector
    c.start_load_kv(None)
    if terminal == "finish":
        assert c.get_finished({"r"}) == (None, None)
    else:
        c.handle_preemptions({"r"})
    state.store.release_reservation.assert_not_called()
    assert state.worker.request_keys == {"r": "generation"}
    state.pipeline.Check.return_value = True
    assert c.get_finished(set()) == (None, {"r"})
    state.store.release_reservation.assert_called_once_with(41)
    assert not state.worker.jobs
    assert not state.worker.request_keys
    c.get_finished({"r"})
    state.store.release_reservation.assert_called_once_with(41)


@pytest.mark.parametrize("owner", [False, True])
@pytest.mark.parametrize("operation", ["submit", "check", "wait"])
def test_unknown_io_completion_fences_without_releasing_protection(
    worker_factory, owner, operation
):
    state = worker_factory(owner)
    failure = RuntimeError("I/O completion unknown")
    if operation == "submit":
        state.submit.side_effect = failure
    elif operation == "check":
        state.pipeline.Check.side_effect = failure
    else:
        state.pipeline.Check.return_value = True
        state.pipeline.Wait.side_effect = failure
    state.connector.start_load_kv(None)
    assert state.connector.get_finished({"r"}) == (None, {"r"})
    assert state.worker.owner_protocol.fatal
    state.store.release_reservation.assert_not_called()
    assert state.job.key in state.worker.jobs
    assert state.connector._connector_worker_meta.load_failed_reqs == {"r"}
    assert state.connector.get_block_ids_with_load_errors() == {3}
    assert not state.connector._pending_load_tasks
    assert not state.connector._rank_consistency._load_task_contexts
    assert not messages(state, "consumed")
    assert not messages(state, "quiesced")
    with pytest.raises(RuntimeError, match="fenced"):
        state.connector.start_load_kv(None)


@pytest.mark.parametrize("owner", [False, True])
def test_capacity_rejection_preserves_native_failure_without_fencing(
    worker_factory, owner
):
    state = worker_factory(owner)
    state.submit.side_effect = early_store.ucmpipelinestore.EarlyCapacityRejected(
        "before I/O"
    )
    state.connector.start_load_kv(None)
    assert state.connector.get_finished(set()) == (None, {"r"})
    assert not state.worker.owner_protocol.fatal
    assert not state.connector._pending_load_tasks
    assert not state.connector._rank_consistency._load_task_contexts
    assert not state.store.store_.receipts
    assert state.connector._connector_worker_meta.load_failed_reqs == {"r"}
    assert state.connector._rank_consistency._missing_reqs == {"r"}
    state.pipeline.Wait.assert_not_called()
    if owner:
        assert len(messages(state, "worker_retire")) == 1
        assert not messages(state, "consumed")
        state.store.release_reservation.assert_not_called()
    else:
        state.store.release_reservation.assert_called_once_with(41)


@pytest.mark.parametrize("failed", [False, True])
def test_shutdown_waits_or_quarantines_before_closing_early(
    worker_factory, monkeypatch, failed
):
    state = worker_factory()
    state.connector.start_load_kv(None)
    events = []

    def wait(task):
        events.append("wait")
        if failed:
            raise RuntimeError("unknown shutdown drain")

    state.pipeline.Wait.side_effect = wait
    state.worker.close = Mock(side_effect=lambda: events.append("close"))
    native_shutdown = Mock(side_effect=lambda: events.append("native_shutdown"))
    monkeypatch.setattr(ucm_connector.UCMDirectConnector, "shutdown", native_shutdown)
    state.connector.shutdown()
    assert events[:2] == ["wait", "close"]
    if failed:
        assert state.worker.owner_protocol.fatal
        state.store.release_reservation.assert_not_called()
        native_shutdown.assert_not_called()
    else:
        assert not state.connector._pending_load_tasks
        state.store.release_reservation.assert_called_once_with(41)
        native_shutdown.assert_called_once()


@pytest.mark.parametrize("enabled", [False, True])
def test_real_worker_initialization_installs_async_failure_handler(enabled):
    store = Mock()
    store.config = {}
    connector = SimpleNamespace(
        store=store,
        tp_rank=0,
        tp_size=1,
        is_mla=True,
        unique_id="async-handler-unit",
        use_request_async_load=enabled,
        launch_config={"enable_early_owner": False},
        _vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_index=0)
        ),
    )
    worker = EarlyWorker(connector, control_factory=Mock(return_value=Mock()))
    try:
        assert store.set_io_failure_handler.call_count == int(enabled)
        if enabled:
            handler = store.set_io_failure_handler.call_args.args[0]
            assert handler.__self__ is worker.owner_protocol
    finally:
        worker.close()
