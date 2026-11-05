"""PENDING progress and owner retirement before whole-request READY."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_early_batch_pipeline import worker
from test_early_owner_protocol import Cluster, early, owner


def pending_worker(*, remaining=2, can_submit=False, waiting=False):
    worker = object.__new__(early.EarlyWorker)
    worker.control = Mock()
    worker.connector = SimpleNamespace(tp_rank=0)
    worker.pending_backfill = True
    part = early.WorkerBatch(0, 0, dict(total_hit_blocks=4, remaining_l2_work=remaining))
    part.started = True
    part.stats = dict(can_submit=can_submit, waiting_capacity=waiting)
    job = early.WorkerJob("job", [b"a"] * 4, reserve_done=True, lookup_finished=True)
    job.batches = [part]
    return worker, job


@pytest.mark.parametrize("blocked", ["capacity", "submission", "lookup", "reservation", "cancel", "discard"])
def test_progress_requires_complete_stable_pins(blocked):
    worker, job = pending_worker()
    if blocked == "capacity":
        job.batches[0].stats["waiting_capacity"] = True
    elif blocked == "submission":
        job.batches[0].stats["can_submit"] = True
    elif blocked == "lookup":
        job.lookup_finished = False
    elif blocked == "reservation":
        job.reserve_done = False
    elif blocked == "cancel":
        job.cancelled = True
    else:
        job.batches[0].discarded = True
    worker._publish_progress(job)
    worker.control.send.assert_not_called()


def test_progress_keeps_pending_and_reports_only_changes():
    worker, job = pending_worker()
    worker._publish_progress(job)
    worker._publish_progress(job)
    assert not job.ready
    assert job.pending_progress == (4, 2)
    assert worker.control.send.call_count == 1
    job.batches[0].reservation["remaining_l2_work"] = 1
    worker._publish_progress(job)
    assert worker.control.send.call_args.args[0]["remaining_l2_work"] == 1


def test_owner_progress_preserves_generation_and_accepts_early_consumption():
    cluster = Cluster()
    job = cluster.job()
    cluster.message(job, "owner_progress", hit=4, remaining_l2_work=2, capacity_truncated=False)
    progress = cluster.out[-1]
    assert progress[0] == "s0"
    assert progress[1]["op"] == "progress"
    assert progress[1]["grant"] == job["grant"]
    for peer in cluster.b.workers(0):
        cluster.message(job, "consumed", peer)
    assert not job.get("retiring")
    cluster.ready(job)
    assert job["retiring"]
    assert not any(m["op"] == "release_owner" for _, m in cluster.out)
    for peer in cluster.b.workers(0):
        cluster.message(job, "quiesced", peer)
    assert any(m["op"] == "release_owner" for _, m in cluster.out)


@pytest.mark.parametrize("field,value", [("hit", True), ("hit", 5), ("remaining_l2_work", -1)])
def test_invalid_owner_progress_fences_domain(field, value):
    cluster = Cluster()
    job = cluster.job()
    fields = dict(hit=4, remaining_l2_work=2, capacity_truncated=False)
    fields[field] = value
    cluster.message(job, "owner_progress", **fields)
    assert job["status"] == "FAILED"
    assert any(m["op"] == "domain_lost" for _, m in cluster.out)


def test_stale_or_nonowner_progress_cannot_authorize_consumption():
    cluster = Cluster()
    job = cluster.job()
    cluster.message(job, "owner_progress", "w0.1", hit=4, remaining_l2_work=1, capacity_truncated=False)
    cluster.b.receive(job["owner"], dict(op="owner_progress", key=job["key"], attempt="stale",
                                       hit=4, remaining_l2_work=1, capacity_truncated=False))
    assert "pending_hit" not in job


def test_retirement_of_published_pending_waits_for_consumer_quiescence():
    worker, job = pending_worker()
    job.owner, job.attempt = "w0.0", "attempt"
    worker._publish_progress(job)
    worker.condition = threading.Condition()
    worker.jobs = {job.key: job}
    worker._release = Mock()
    protocol = owner_worker(worker)
    protocol.receive(dict(op="retire", key=job.key, attempt=job.attempt))
    worker._release.assert_not_called()
    protocol.receive(dict(op="release_owner", key=job.key, attempt=job.attempt))
    worker._release.assert_called_once_with(job)


def owner_worker(worker):
    protocol_module = __import__(early.__package__ + ".early_owner_worker", fromlist=["OwnerWorker"])
    return protocol_module.OwnerWorker(worker, {}, "w0.0")


def test_broadcast_progress_waits_for_every_rank_and_uses_conservative_work():
    messages = []
    broker = owner.BrokerState(lambda peer, msg: messages.append((peer, msg)), 2)
    for peer in ("w0.0", "w0.1"):
        broker.receive(peer, dict(op="hello"))
    broker.receive("s0", dict(op="job", key="job", dp=0, blocks=[b"a"] * 4))
    messages.clear()
    broker.receive("w0.0", dict(op="progress", key="job", hit=4, remaining_l2_work=2))
    assert not messages
    broker.receive("w0.1", dict(op="progress", key="job", hit=3, remaining_l2_work=1))
    assert messages[-1][1] == dict(op="progress", key="job", hit=3, remaining_l2_work=3)


def test_scheduler_ignores_progress_from_previous_owner_generation():
    scheduler = object.__new__(early.EarlyScheduler)
    scheduler.lock = threading.RLock()
    request = type("Request", (), {})()
    job = early.Job(request, [], (), key="job", attempt="current", grant=2, owner="w0.0")
    scheduler.by_key = {job.key: job}
    scheduler.receive(dict(op="progress", key=job.key, attempt="old", grant=1,
                           hit=4, remaining_l2_work=1))
    assert not job.loadable
    scheduler.receive(dict(op="progress", key=job.key, attempt="current", grant=2,
                           hit=4, remaining_l2_work=1))
    assert job.loadable and job.state == "PENDING"
    assert job.matched_hit == 4


@pytest.mark.parametrize("limit,expected", [
    (5, [3, 0, 1, 2, 4]), (2, [3, 0])
])
def test_pending_hrrn_prefers_older_work_and_keeps_equal_score_queue_order(
    monkeypatch, limit, expected
):
    module = __import__(early.__package__ + ".early_scheduling", fromlist=["WaitingPass"])
    policy = module.ReadyPriority({"early_ready_priority": True})
    requests = [type("Request", (), {"num_tokens": 1024, "arrival_time": 0})() for _ in range(5)]
    jobs = [early.Job(request, [], ()) for request in requests]
    for job, work in zip(jobs, [8, 2, 2, 0, 1]):
        job.pending_hit, job.remaining_l2_work = 4, work
    for job, entered in zip(jobs, (0, 90, 90, 0, 0)):
        job.entered = entered
    jobs[3].state, jobs[3].hit = "READY", 4
    jobs[4].pending_hit = None
    service = type("Early", (), {})()
    service.scheduling = policy
    service.connector = type("Connector", (), {"hash_block_size": 128})()
    service.probe = lambda request: jobs[requests.index(request)]
    scheduler = type("Scheduler", (), {"waiting": requests, "policy": "fcfs"})()
    policy.bypasses[requests[0]] = policy.max_bypasses
    policy.last_rejection[requests[1]] = 1
    policy.max_candidates = limit
    monotonic = Mock(return_value=100)
    monkeypatch.setattr(module.time, "monotonic", monotonic)
    waiting = module.WaitingPass(scheduler, service)
    waiting._prepare()
    assert [candidate.request for candidate in waiting.remaining] == [
        requests[index] for index in expected
    ]
    monotonic.assert_called_once_with()


def test_published_pending_failure_stops_submission_without_releasing_pins():
    worker, job = pending_worker()
    worker._release = Mock()
    worker._publish_progress(job)
    protocol = owner_worker(worker)
    protocol.answer(job, "read failed")
    assert job.cancelled
    worker._release.assert_not_called()
    assert worker.control.send.call_args.args[0]["state"] == "FAILED"


def test_worker_publishes_pending_while_reads_are_still_in_flight(worker):
    from test_early_batch_pipeline import blocks, result, wait, wait_answer

    gate = threading.Event()
    reservation = result(1, 4, 2)
    reservation["batch"].gate = gate
    worker.pending_backfill = True
    worker.store.reserve_prefix.return_value = reservation
    try:
        worker.receive(dict(op="reserve", key="pending", blocks=blocks(4)))
        wait(lambda: any(m["op"] == "progress" for m in worker.control.messages))
        assert not worker.jobs["pending"].ready
        progress = next(m for m in worker.control.messages if m["op"] == "progress")
        assert progress["hit"] == 4 and progress["remaining_l2_work"] == 2
        gate.set()
        wait_answer(worker, "pending")
        assert worker.jobs["pending"].ready
    finally:
        gate.set()


def test_owner_retry_and_progress_preserve_pending_hrrn_age():
    scheduler = object.__new__(early.EarlyScheduler)
    scheduler.lock = threading.RLock()
    request = type("Request", (), {})()
    job = early.Job(request, [], (), key="job", entered=10)
    job.pending_hit, job.remaining_l2_work = 4, 8
    scheduler.by_key = {job.key: job}
    scheduler.receive(dict(op="owner_assigned", key=job.key, owner="w0.0",
                           attempt="retry", grant=1))
    assert job.entered == 10 and job.pending_hit is None
    scheduler.receive(dict(op="progress", key=job.key, attempt="retry", grant=1,
                           hit=4, remaining_l2_work=2))
    assert job.entered == 10 and job.remaining_l2_work == 2
