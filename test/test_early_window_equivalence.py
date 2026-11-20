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
