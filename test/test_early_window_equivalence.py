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


def blocked_window():
    window = owner.ExecutionWindow(4)
    window.add("holder", {0, 1, 2})
    window.add("waiting", {3, 4, 5})
    assert window.acquire("holder", {0, 1})
    return window


def test_repeated_denial_reuses_proof_without_pinning_or_completing(monkeypatch):
    window = blocked_window()
    proofs = []
    safe = window.safe

    def count_proof(key, blocks):
        proofs.append((key, blocks))
        return safe(key, blocks)

    monkeypatch.setattr(window, "safe", count_proof)
    before = dict(window.block_holders)
    for _ in range(8):
        assert not window.acquire("waiting", {3, 4})
        assert not window.acquire("waiting", {4, 5})
    assert len(proofs) == 2
    assert dict(window.block_holders) == before
    assert not window.claims["waiting"].held
    assert not window.claims["holder"].completed


@pytest.mark.parametrize("change", ["release", "complete", "narrow", "replace", "capacity"])
def test_denied_grant_is_reconsidered_after_proof_inputs_change(change):
    window = blocked_window()
    assert not window.acquire("waiting", {3, 4})
    if change == "release":
        window.release("holder")
    elif change == "complete":
        window.complete("holder")
    elif change == "narrow":
        window.narrow("holder", {0, 1})
    elif change == "replace":
        window.add("holder", {0})
    else:
        window.capacity = 6
    assert admits_in_some_order(window, "waiting", {3, 4})
    assert window.acquire("waiting", {3, 4})


def test_new_pins_invalidate_denials_but_duplicate_grants_do_not(monkeypatch):
    window = blocked_window()
    assert not window.acquire("waiting", {3, 4})
    assert window.acquire("holder", {0, 1})
    assert window._failed_acquisitions
    assert window.acquire("holder", {2})
    assert not window._failed_acquisitions
    proofs = []
    safe = window.safe

    def count_proof(key, blocks):
        proofs.append(key)
        return safe(key, blocks)

    monkeypatch.setattr(window, "safe", count_proof)
    assert not window.acquire("waiting", {3, 4})
    assert proofs == ["waiting"]


def test_denial_cache_is_not_reused_for_a_different_proposal():
    window = blocked_window()
    assert not window.acquire("waiting", {3, 4})
    assert window.acquire("waiting", {3})
    assert window.claims["waiting"].held == {3}
    assert window.used == 3
