"""Request priority with real worker threads and controlled storage completion."""

import threading
import time
from unittest.mock import Mock

import pytest
from test_early_batch_pipeline import Control, blocks, early, result, wait, worker


def add_request(worker, key, work, *, age=10, submitted=None):
    entered = time.monotonic() - age
    batches = []
    begin = 0
    for index, count in enumerate(work):
        reservation = result((key, index), count, count)
        if submitted is not None:
            reservation["batch"].submit = lambda: submitted.append(key)
        batches.append(early.WorkerBatch(index, begin, reservation, entered=entered))
        begin += count
    job = early.WorkerJob(
        key,
        blocks(begin),
        entered=entered,
        batches=batches,
        reserve_done=True,
        lookup_finished=True,
    )
    worker.jobs[key] = job
    worker.preparing.extend((job, batch) for batch in batches)
    return job


@pytest.mark.parametrize("parts", [[12], [1, 11], [4, 4, 4]])
def test_request_score_is_independent_of_batch_partition(worker, parts):
    with worker.condition:
        job = add_request(worker, "partition", parts)
        job.entered = 0
        worker.shard_bytes = 4096
        assert worker.ratio(job, 12) == pytest.approx(1 + 1 / 4096)
        worker.preparing.clear()


def test_score_includes_draining_and_blocked_work_but_excludes_finished_tail(worker):
    with worker.condition:
        job = add_request(worker, "all-work", [1, 4, 8, 16])
        job.entered = 0
        job.batches[1].can_submit = False
        job.batches[1].inflight_bytes = 4
        job.batches[2].done = True
        job.batches[3].discarded = True
        assert worker.ratio(job, 10) == 3
        worker.preparing.clear()


@pytest.mark.parametrize("shard_bytes", [1024, 8192])
def test_unqueried_tail_uses_native_block_bytes_until_lookup_finishes(
    worker, shard_bytes
):
    with worker.condition:
        job = add_request(worker, "lookup", [1])
        job.entered = 0
        job.blocks = blocks(3)
        job.reserve_done = False
        worker.block_bytes, worker.shard_bytes = 8192, shard_bytes
        assert worker.ratio(job, 20) == pytest.approx(1 + 20 / (shard_bytes + 2 * 8192))
        job.reserve_done = True
        assert worker.ratio(job, 20) == pytest.approx(1 + 20 / shard_bytes)
        worker.preparing.clear()


def test_selection_uses_request_tail_instead_of_small_head(worker):
    with worker.condition:
        long = add_request(worker, "long", [1, 99])
        short = add_request(worker, "short", [4])
        long.entered = short.entered = time.monotonic() - 10
        worker._choose()
        assert worker.selected == [(short, short.batches[0])]
        worker.preparing.clear()
        worker.selected.clear()


def test_selected_request_finishes_all_batches_before_competing_request(worker):
    submitted = []
    with worker.condition:
        first = add_request(worker, "first", [4, 3], age=100, submitted=submitted)
        second = add_request(worker, "second", [2], age=20, submitted=submitted)
        worker.condition.notify_all()
    wait(lambda: first.ready and second.ready)
    assert submitted == ["first"] * 7 + ["second"] * 2
    with worker.condition:
        assert [message["key"] for message in worker.control.messages] == [
            "first",
            "second",
        ]
        assert [message["hit"] for message in worker.control.messages] == [7, 2]


@pytest.mark.parametrize("terminal", ["ready", "cancelled", "replaced"])
def test_terminal_request_does_not_retain_priority(worker, terminal):
    with worker.condition:
        first = add_request(worker, "first", [1], age=100)
        second = add_request(worker, "second", [4])
        worker._choose()
        assert worker.selected == [(first, first.batches[0])]
        worker.selected.clear()
        if terminal == "replaced":
            worker.jobs[first.key] = early.WorkerJob(first.key, [])
        else:
            setattr(first, terminal, True)
        worker._choose()
        assert worker.selected == [(second, second.batches[0])]
        assert first not in worker.hrrn_requests
        worker.preparing.clear()
        worker.selected.clear()


def test_blocked_request_regains_priority_after_backfill(worker):
    with worker.condition:
        first = add_request(worker, "first", [1, 4], age=100)
        second = add_request(worker, "second", [4])
        worker._choose()
        assert worker.selected == [(first, first.batches[0])]
        worker.selected.clear()
        first.batches[0].waiting_capacity = True
        first.batches[0].can_submit = False
        worker._choose()
        assert worker.selected == [(second, second.batches[0])]
        worker.preparing.extend(worker.selected)
        worker.selected.clear()
        first.batches[0].waiting_capacity = False
        first.batches[0].done = True
        worker._choose()
        assert worker.selected == [(first, first.batches[1])]
        worker.preparing.clear()
        worker.selected.clear()


def test_round_scores_each_request_once_and_defers_new_arrivals(worker, monkeypatch):
    score = Mock(wraps=worker.ratio)
    monkeypatch.setattr(worker, "ratio", score)
    with worker.condition:
        first = add_request(worker, "first", [2], age=100)
        second = add_request(worker, "second", [4], age=10)
        worker._choose()
        assert worker.selected == [(first, first.batches[0])]
        assert score.call_count == 2
        assert score.call_args_list[0].args[1] == score.call_args_list[1].args[1]
        worker.selected.clear()
        newcomer = add_request(worker, "new", [1], age=1000)
        first.ready = True
        worker._choose()
        assert worker.selected == [(second, second.batches[0])]
        assert score.call_count == 2
        worker.selected.clear()
        second.ready = True
        worker._choose()
        assert worker.selected == [(newcomer, newcomer.batches[0])]
        assert score.call_count == 3
        worker.selected.clear()
        worker.preparing.clear()


def test_round_order_survives_cost_updates(worker):
    with worker.condition:
        first = add_request(worker, "first", [1, 1], age=100)
        second = add_request(worker, "second", [4], age=10)
        third = add_request(worker, "third", [8], age=10)
        worker._choose()
        worker.selected.clear()
        first.ready = True
        third.batches[0].reservation["remaining_l2_work"] = 1
        worker._choose()
        assert worker.selected == [(second, second.batches[0])]
        worker.selected.clear()
        worker.preparing.clear()


def test_blocked_round_backfills_next_round_and_resumes_in_frozen_order(
    worker, monkeypatch
):
    score = Mock(wraps=worker.ratio)
    monkeypatch.setattr(worker, "ratio", score)
    with worker.condition:
        first = add_request(worker, "first", [1, 2], age=100)
        second = add_request(worker, "second", [4], age=10)
        worker._choose()
        worker.selected.clear()
        first.batches[0].waiting_capacity = True
        first.batches[0].can_submit = False
        second.batches[0].can_submit = False
        third = add_request(worker, "third", [4], age=10)
        fourth = add_request(worker, "fourth", [1], age=20)
        worker._choose()
        assert worker.selected == [(fourth, fourth.batches[0])]
        assert worker.hrrn_requests == [first, second, fourth, third]
        assert score.call_count == 4
        worker.preparing.extend(worker.selected)
        worker.selected.clear()
        second.batches[0].can_submit = True
        worker.preparing.append((second, second.batches[0]))
        worker._choose()
        assert worker.selected == [(second, second.batches[0])]
        worker.preparing.extend(worker.selected)
        worker.selected.clear()
        first.batches[0].waiting_capacity = False
        first.batches[0].done = True
        worker._choose()
        assert worker.selected == [(first, first.batches[1])]
        assert score.call_count == 4
        worker.selected.clear()
        worker.preparing.clear()


def test_equal_scores_use_request_age_then_key(worker):
    with worker.condition:
        second = add_request(worker, "b", [2])
        first = add_request(worker, "a", [2])
        first.entered = second.entered = 0
        worker._choose()
        assert worker.hrrn_requests == [first, second]
        worker.selected.clear()
        worker.preparing.clear()


def test_arrival_during_io_wait_cannot_bypass_runnable_current_round(worker):
    gate = threading.Event()
    submitted = []
    with worker.condition:
        first = add_request(worker, "first", [2, 2], age=100, submitted=submitted)
        second = add_request(worker, "second", [1], age=10, submitted=submitted)
        first.batches[0].reservation["batch"].gate = gate
        worker.condition.notify_all()
    try:
        wait(lambda: submitted == ["first", "first"])
        with worker.condition:
            last = add_request(worker, "last", [1], age=1000, submitted=submitted)
            worker.condition.notify_all()
        time.sleep(0.01)
        assert submitted == ["first", "first"]
    finally:
        gate.set()
    wait(lambda: first.ready and second.ready and last.ready)
    assert submitted == ["first"] * 4 + ["second", "last"]


@pytest.mark.parametrize(
    "layout, expected",
    [
        ({}, (8192, 1024)),
        ({"cache_full_block_l1": True}, (8192, 8192)),
        (
            {"cache_full_block_l1": True, "posix_packed_layer_bytes": [512] * 8},
            (4096, 4096),
        ),
    ],
)
def test_startup_health_rejection_preserves_cost_geometry_after_recovery(
    worker, layout, expected
):
    worker.store.domain_info.return_value = dict(block_bytes=0, shard_bytes=0)
    worker.store.config = dict(block_size=8192, shard_size=1024, **layout)
    recovered = early.EarlyWorker(worker.connector, Control)
    try:
        assert (recovered.block_bytes, recovered.shard_bytes) == expected
        with recovered.condition:
            job = add_request(recovered, "recovered", [2])
            recovered.condition.notify_all()
        wait(lambda: job.ready)
    finally:
        recovered.close()
