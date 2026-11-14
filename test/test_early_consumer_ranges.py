"""Consumer coverage reuse preserves exact successful native completion."""
import random

import numpy as np
import pytest

from test_early_owner_protocol import blocks, consumer, start_load, ticket


def make_consumer(tickets, layers):
    sent, faults = [], []
    observer = consumer.Consumer("worker", sent.append, faults.append)
    observer.bind(tickets, layers)
    return observer, sent, faults


def submit(observer, requests, block_ids, layers):
    scope = consumer.load_scope.set(observer.scope(requests))
    try:
        return observer.begin(
            np.frombuffer(b"".join(block_ids), dtype=np.uint8), layers
        )
    finally:
        consumer.load_scope.reset(scope)


def test_uniform_layers_reuse_coverage_without_mutating_prior_completions():
    t = ticket(expected=blocks(4))
    observer, sent, faults = make_consumer({"r": t}, [0, 1, 2])
    first = start_load(observer, "r", blocks(2), 0)
    second = start_load(observer, "r", blocks(2), 1)
    observer.complete(first, 1, True)
    observer.complete(second, 2, True)
    state = observer.generations[("g", "a-g")]
    before = state["completed"][0]
    rest = start_load(observer, "r", blocks(4)[1:], 1)
    observer.complete(rest, 3, True)
    observer.complete(rest, 3, True)
    assert state["completed"][0] == before == set(blocks(2))
    assert state["completed"][1] == set(blocks(4))
    assert state["remaining"] == 6
    assert not sent and not faults


def test_mixed_layers_and_shared_requests_keep_exact_owner_coverage():
    a, b = ticket(expected=blocks(2)), ticket("b", expected=blocks(4)[1:])
    observer, sent, faults = make_consumer({"a": a, "b": b}, [0, 1])
    ids = [blocks(1)[0], blocks(2)[1], blocks(2)[1], blocks(4)[3], b"x" * 16]
    receipt = submit(observer, ["a", "b"], ids, [0, 0, 1, 1, 0])
    observer.complete(receipt, 1, True)
    sa, sb = observer.generations[("g", "a-g")], observer.generations[("b", "a-b")]
    assert sa["completed"] == {0: set(blocks(2)), 1: {blocks(2)[1]}}
    assert sb["completed"] == {0: {blocks(2)[1]}, 1: {blocks(2)[1], blocks(4)[3]}}
    assert sa["remaining"] == 1 and sb["remaining"] == 3
    assert not sent and not faults


def test_input_mutation_and_rebinding_do_not_change_inflight_coverage():
    a = ticket(expected=blocks(2))
    observer, sent, faults = make_consumer({"r": a}, [0, 1])
    raw = bytearray(blocks(1)[0])
    scope = consumer.load_scope.set(observer.scope(["r"]))
    try:
        first = observer.begin(memoryview(raw), [0])
        raw[:] = blocks(2)[1]
        second = observer.begin(memoryview(raw), [1])
    finally:
        consumer.load_scope.reset(scope)
    b = ticket("new", expected=blocks(3)[2:])
    observer.bind({"r": b}, [0, 1])
    observer.complete(second, 2, True)
    observer.complete(first, 1, True)
    old = observer.generations[("g", "a-g")]
    assert old["completed"] == {0: {blocks(1)[0]}, 1: {blocks(2)[1]}}
    assert old["remaining"] == 2
    assert not observer.generations[("new", "a-new")]["completed"]
    assert not sent and not faults


def test_cached_coverage_never_bypasses_retirement_or_failed_wait():
    observer, sent, faults = make_consumer({"r": ticket()}, [0, 1])
    first = start_load(observer, "r", blocks(1), 0)
    second = start_load(observer, "r", blocks(1), 1)
    observer.retire("g", "a-g")
    with pytest.raises(RuntimeError, match="retired"):
        start_load(observer, "r", blocks(1), 0)
    observer.complete(first, 1, True)
    assert not sent
    observer.complete(second, 2, False)
    assert faults and not sent
    assert observer.generations[("g", "a-g")]["remaining"] == 3


def test_partial_plan_cache_is_bounded_and_bind_discards_metadata():
    observer, sent, faults = make_consumer({"r": ticket(expected=blocks(32))}, [0, 1])
    for i in range(31):
        receipt = start_load(observer, "r", blocks(32)[i:i + 1], 0)
        observer.complete(receipt, i, True)
        assert len(observer._range_cache) <= 1
    observer.bind({"r": ticket(expected=blocks(32))}, [0, 1])
    assert not observer._range_cache
    assert observer.generations[("g", "a-g")]["remaining"] == 33
    assert not sent and not faults


@pytest.mark.parametrize("seed", range(5))
def test_overlapping_out_of_order_receipts_match_reference_sets(seed):
    rng = random.Random(seed)
    expected = blocks(8)
    observer, sent, faults = make_consumer({"r": ticket(expected=expected)}, [0, 1, 2])
    pending = []
    for _ in range(30):
        layer = rng.randrange(3)
        ids = rng.choices(expected + [b"x" * 16], k=rng.randrange(1, 12))
        pending.append((start_load(observer, "r", ids, layer), layer, ids))
    for layer in range(3):
        pending.append((start_load(observer, "r", expected, layer), layer, expected))
    rng.shuffle(pending)
    reference = {layer: set() for layer in range(3)}
    state = observer.generations[("g", "a-g")]
    for index, (receipt, layer, ids) in enumerate(pending):
        observer.complete(receipt, index, True)
        observer.complete(receipt, index, True)
        reference[layer].update(set(ids).intersection(expected))
        assert state["remaining"] == 24 - sum(map(len, reference.values()))
        assert all(state["completed"].get(k, set()) == v for k, v in reference.items())
        assert len(sent) == int(index == len(pending) - 1)
    assert sent[0]["op"] == "consumed"
    assert not faults
