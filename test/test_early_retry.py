"""Typed retries keep logical age and wait for every old consumer to drain."""

import threading

import pytest
from test_early_batch_pipeline import blocks, result, wait
from test_early_owner_protocol import Cluster, early, make_worker, owner


@pytest.fixture
def clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(owner.time, "monotonic", lambda: now[0])
    return now


def reject(cluster, job, code=owner.EarlyLookupUnavailable.code):
    cluster.message(
        job,
        "owner_answer",
        state="FAILED",
        hit=0,
        error_code=code,
        reason="injected metadata failure",
    )


def ack_drain(cluster, job):
    for peer in cluster.b.workers(job["dp"]):
        cluster.message(job, "quiesced", peer)
    cluster.message(job, "owner_drained", read_bytes=0, submitted_read_bytes=0)


def results(cluster):
    return [message for _, message in cluster.out if message["op"] == "result"]


def test_retry_requires_full_drain_and_backoff_preserves_age_and_identity(clock):
    c = Cluster(retry_backoff_ms=100)
    job = c.job()
    first = job.copy()
    reject(c, job)
    assert job["retiring"] and job["retry_pending"]
    assert not results(c)
    c.message(job, "quiesced", "w0.0")
    c.message(job, "owner_drained")
    assert job["attempt"] == first["attempt"]
    assert job["domain"]["owners"][job["owner"]]["jobs"] == 1
    ack_drain(c, job)
    assert job["owner"] is None
    assert job["wait_reason"] == "retry_backoff"
    assert 99 <= c.b.poll_timeout_ms() <= 101
    c.b.advance_retries(clock[0] + 0.05)
    assert not job.get("retry_count")
    clock[0] += 0.1
    c.b.advance_retries(clock[0])
    assert job["attempt"] != first["attempt"]
    assert job["grant"] > first["grant"]
    assert job["retry_count"] == 1
    assert (job["key"], job["blocks"], job["created"]) == (
        first["key"],
        first["blocks"],
        first["created"],
    )
    assert c.b.poll_timeout_ms() == 1000
    c.message(first, "owner_drained")
    c.ready(first)
    assert job["status"] == "PENDING" and not results(c)
    c.ready(job)
    assert results(c)[-1]["attempt"] == job["attempt"]
    c.drain(job)
    assert not c.b.jobs and not c.b.admission_waiting
    assert c.b.waiting_candidates == 0


@pytest.mark.parametrize("budget", [0, 1, 2])
def test_budget_counts_new_executions_and_fallback_waits_for_drain(clock, budget):
    c = Cluster(max_retries=budget, retry_backoff_ms=0)
    job = c.job()
    attempts = set()
    for expected in range(budget + 1):
        attempts.add(job["attempt"])
        assert job.get("retry_count", 0) == expected
        reject(c, job)
        assert not results(c)
        ack_drain(c, job)
    assert len(attempts) == budget + 1
    assert len(results(c)) == 1 and results(c)[0]["state"] == "FAILED"
    assert results(c)[0]["attempt"] in attempts
    assert not c.b.jobs
    c.b.advance_retries(clock[0] + 100)
    assert len(results(c)) == 1


def test_retry_waits_for_owner_without_spending_budget_or_timeout(clock):
    c = Cluster(retry_backoff_ms=100, timeout=1)
    for cap in c.b.capabilities.values():
        cap["max_jobs"] = 1
    job = c.job("retry")
    reject(c, job)
    ack_drain(c, job)
    occupants = [c.job("busy-0"), c.job("busy-1")]
    for occupant in occupants:
        c.ready(occupant)
    clock[0] += 0.1
    c.b.advance_retries(clock[0])
    assert job["wait_reason"] == "owner_capacity"
    assert c.b.poll_timeout_ms() == 1000
    for _ in range(10):
        clock[0] += 10
        for peer in c.b.peers:
            c.b.peers[peer] = clock[0]
        c.b.expire(clock[0])
        c.b.wake()
        c.b._wake_admission()
        assert not job.get("retry_count")
        assert job["status"] == "PENDING" and job["owner"] is None
    c.drain(occupants[0])
    assert job["retry_count"] == 1 and job["owner"]
    assert job["created"] == 1000
    assert not c.b.pending_expired(job, clock[0] + 0.5)


@pytest.mark.parametrize("stage", ["drain", "backoff", "owner"])
@pytest.mark.parametrize("operation", ["cancel", "finish"])
def test_terminal_client_message_prevents_reexecution(clock, stage, operation):
    c = Cluster(retry_backoff_ms=100)
    for cap in c.b.capabilities.values():
        cap["max_jobs"] = 1
    job = c.job()
    reject(c, job)
    if stage != "drain":
        ack_drain(c, job)
    if stage == "owner":
        c.job("busy-0")
        c.job("busy-1")
        clock[0] += 0.1
        c.b.advance_retries(clock[0])
        assert job["wait_reason"] == "owner_capacity"
    c.b.receive("s0", dict(op=operation, key=job["key"]))
    if stage == "drain":
        ack_drain(c, job)
    clock[0] += 1
    c.b.advance_retries(clock[0])
    c.b._wake_admission()
    assert job["key"] not in c.b.jobs
    assert job["key"] not in c.b.admission_waiting
    assert not job.get("retry_count")
    assert not results(c)


@pytest.mark.parametrize(
    "stage", ["unknown", "pending_batch", "granted_batch", "lookup_done"]
)
def test_unknown_errors_or_any_published_work_do_not_retry(stage):
    c = Cluster(retry_backoff_ms=0)
    job = c.job()
    if stage == "pending_batch":
        job["execution_pending"][0] = {"count": 1}
    elif stage == "granted_batch":
        job["execution_grants"][0] = {"count": 1}
    elif stage == "lookup_done":
        job["storage_hit"] = 0
    reject(
        c,
        job,
        code=(
            "unclassified" if stage == "unknown" else owner.EarlyLookupUnavailable.code
        ),
    )
    assert not job.get("retry_pending")
    assert results(c)[-1]["state"] == "FAILED"
    ack_drain(c, job)
    assert not c.b.jobs


@pytest.mark.parametrize("failure", ["drain_timeout", "domain_fault", "heartbeat"])
def test_unknown_drain_or_shared_state_fences_without_retry(clock, failure):
    c = Cluster(retry_backoff_ms=0, timeout=1)
    job = c.job()
    domain, peer = job["domain"], job["owner"]
    reject(c, job)
    if failure == "domain_fault":
        c.b.receive(peer, dict(op="domain_fault", reason="unknown native state"))
    else:
        clock[0] += 2 if failure == "drain_timeout" else 61
        c.b.expire(clock[0])
    assert c.b.fenced and not job.get("retry_count")
    ack_drain(c, job)
    assert domain["owners"][peer]["jobs"] == 1


@pytest.mark.parametrize("name", ["max_retries", "retry_backoff_ms"])
@pytest.mark.parametrize("value", [-1, True, 1.5, "1"])
def test_invalid_retry_configuration_fails_before_admission(name, value):
    with pytest.raises(ValueError, match="nonnegative integer"):
        Cluster(**{name: value})


def test_scheduler_ignores_results_and_assignments_from_old_attempts():
    job = early.Job(object(), object(), (), key="generation")
    scheduler = early.EarlyScheduler.__new__(early.EarlyScheduler)
    scheduler.lock = threading.RLock()
    scheduler.by_key = {job.key: job}
    scheduler.dp_rank = 0
    first = dict(key=job.key, owner="w0.0", attempt="first", grant=1)
    second = dict(key=job.key, owner="w0.1", attempt="second", grant=2)
    scheduler.receive(dict(first, op="owner_assigned"))
    scheduler.receive(dict(second, op="owner_assigned"))
    scheduler.receive(dict(first, op="owner_assigned"))
    for state in ("READY", "FAILED"):
        scheduler.receive(dict(first, op="result", state=state, hit=1))
    assert job.state == "PENDING" and job.attempt == "second"
    scheduler.receive(dict(second, op="result", state="READY", hit=1))
    assert job.state == "READY" and job.hit == 1


@pytest.mark.parametrize(
    "typed,deferred,prior_batch,expected",
    [
        (True, True, False, "lookup_unavailable"),
        (False, True, False, None),
        (True, False, False, None),
        (True, True, True, None),
    ],
)
def test_worker_only_classifies_first_deferred_query(
    typed, deferred, prior_batch, expected
):
    worker = make_worker(0)
    calls = 0

    def reserve(ids, **kwargs):
        nonlocal calls
        calls += 1
        if prior_batch and calls == 1:
            reservation = result(0, len(ids), 0)
            reservation["batch"].snapshot = lambda: dict(
                l2_read_bytes=0, completed_read_bytes=0
            )
            return reservation
        error = early.EarlyLookupUnavailable if typed else RuntimeError
        raise error("same diagnostic text")

    worker.store.reserve_prefix.side_effect = reserve
    try:
        worker.receive(
            dict(
                op="owner_reserve",
                key="retry",
                attempt="a1",
                grant=1,
                owner="w0.0",
                request="r",
                blocks=blocks(96 if prior_batch else 1),
                io_quota=8,
                deferred=deferred,
                entered=10.0,
            )
        )
        wait(lambda: any(m["op"] == "owner_answer" for m in worker.control.messages))
        answers = [m for m in worker.control.messages if m["op"] == "owner_answer"]
        assert len(answers) == 1 and answers[0]["state"] == "FAILED"
        assert answers[0]["error_code"] == expected
        assert worker.jobs["retry"].entered == 10.0
        worker.receive(dict(op="retire", key="retry", attempt="a1"))
        worker.receive(dict(op="release_owner", key="retry", attempt="a1"))
        wait(lambda: any(m["op"] == "owner_drained" for m in worker.control.messages))
        worker.receive(dict(op="forget", key="retry", attempt="a1"))
    finally:
        worker.close()


def test_real_workers_drain_failed_attempt_then_retry_to_ready():
    from test_early_capacity_wait import DeferredCapability

    c = Cluster(retry_backoff_ms=0)
    workers = [make_worker(0), make_worker(1)]
    sent, received = [0], [0, 0]
    calls = []

    def reserve(ids, *, deferred=False):
        assert deferred
        calls.append(ids)
        if len(calls) == 1:
            raise early.EarlyLookupUnavailable("health breaker open")
        return dict(
            handle=1,
            total_hit_blocks=len(ids),
            remaining_l2_work=0,
            batch=DeferredCapability(len(ids), 0),
        )

    def pump():
        while sent[0] < len(c.out):
            peer, message = c.out[sent[0]]
            sent[0] += 1
            if peer.startswith("w"):
                workers[int(peer[-1])].receive(message)
        for rank, worker in enumerate(workers):
            while received[rank] < len(worker.control.messages):
                message = worker.control.messages[received[rank]]
                received[rank] += 1
                c.b.receive(f"w0.{rank}", message)
        c.b.advance_retries(owner.time.monotonic())

    for worker in workers:
        worker.store.reserve_prefix.side_effect = reserve
    try:
        job = c.job(count=4)
        first = job.copy()
        wait(lambda: (pump(), job["status"] == "READY")[1])
        assert len(calls) == 2 and calls[0] == calls[1]
        assert job["retry_count"] == 1 and job["attempt"] != first["attempt"]
        assert job["created"] == first["created"]
        assert [m["state"] for m in results(c)] == ["READY"]
        for worker in workers:
            assert worker.jobs[job["key"]].attempt == job["attempt"]
            assert worker.jobs[job["key"]].entered == first["created"]
            assert worker.owner_protocol.fatal is None
        c.b.receive("s0", dict(op="finish", key=job["key"]))
        wait(lambda: (pump(), not c.b.jobs and all(not w.jobs for w in workers))[1])
        assert all(load["jobs"] == 0 for load in job["domain"]["owners"].values())
        assert not job["domain"]["execution"].used
    finally:
        for worker in workers:
            worker.close()
