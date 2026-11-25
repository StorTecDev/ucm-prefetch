"""Check broker capacity shortcuts against an exhaustive completion-order oracle."""
import itertools
import random

import pytest

from test_early_owner_protocol import owner


def admits_in_some_order(window, key, proposed):
    claims = window.claims
    held = {
        name: claim.held | (proposed if name == key else set())
        for name, claim in claims.items()
    }
    for order in itertools.permutations(name for name, blocks in held.items() if blocks):
        for index, name in enumerate(order):
            live = set().union(*(held[other] for other in order[index:]))
            if len(live) > window.capacity:
                break
            if not claims[name].completed and len(live | claims[name].maximum) > window.capacity:
                break
        else:
            return True
    return False


@pytest.mark.parametrize("seed", range(16))
def test_capacity_and_occupancy_after_interleaved_updates(seed):
    rng = random.Random(seed)
    window = owner.ExecutionWindow(rng.randrange(1, 12))
    for _ in range(300):
        key = rng.randrange(5)
        action = rng.choice(("add", "acquire", "acquire", "narrow", "complete", "release"))
        blocks = {rng.randrange(12) for _ in range(rng.randrange(9))}
        if key not in window.claims or action == "add":
            window.add(key, blocks)
        elif action == "acquire":
            if rng.choice((False, True)):
                blocks &= window.claims[key].maximum
            expected = blocks <= window.claims[key].maximum and admits_in_some_order(window, key, blocks)
            baseline = window.claims[key].held.copy()
            assert window.acquire(key, blocks) == expected
            assert window.claims[key].held == (baseline | blocks if expected else baseline)
        elif action == "narrow":
            window.narrow(key, blocks & window.claims[key].maximum)
        elif action == "complete":
            window.complete(key)
        else:
            window.release(key)
        assert window.used == len(set().union(*(claim.held for claim in window.claims.values())))


@pytest.mark.parametrize("shared", [False, True])
def test_full_holders_prove_completion_without_rebuilding_large_unions(shared):
    window = owner.ExecutionWindow(120 * 115 + 13)
    for key in range(120):
        blocks = set(range(115)) if shared else set(range(key * 128, key * 128 + 115))
        window.add(key, blocks)
        assert window.acquire(key, blocks)
    if shared:
        window.capacity = 128
    before = dict(window.block_holders)
    assert window.safe(119, window.claims[119].held)
    assert dict(window.block_holders) == before
    assert all(not claim.completed for claim in window.claims.values())


def test_completion_proof_revisits_holders_after_shared_tail_drains():
    window = owner.ExecutionWindow(3)
    window.add("first", {0, 3})
    window.add("second", {1, 4})
    window.add("last", {2})
    for key, held in (("first", {0}), ("second", {1}), ("last", {2})):
        assert window.acquire(key, held)
    before = dict(window.block_holders)
    assert admits_in_some_order(window, "first", {0})
    assert window.safe("first", {0})
    assert dict(window.block_holders) == before


def test_completed_shared_holder_does_not_free_another_claims_pin():
    window = owner.ExecutionWindow(3)
    window.add("completed", {0, 1})
    window.add("shared", {1, 2, 3})
    window.add("new", {4, 5, 6})
    assert window.acquire("completed", {0, 1})
    window.complete("completed")
    assert window.acquire("shared", {1})
    before = dict(window.block_holders)
    assert not admits_in_some_order(window, "new", {4})
    assert not window.acquire("new", {4})
    assert dict(window.block_holders) == before
    assert window.claims["completed"].held == {0, 1}
