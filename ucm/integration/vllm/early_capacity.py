"""Execution eligibility on the broker's verified physical L1 domain.

This is a safe-state test, not a second allocator. Only selected, confirmed
batches enter held; native TryGet still decides actual allocation. A holder's
maximum is a conservative completion bound, never an up-front charge.
"""
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class Claim:
    maximum: set
    held: set = field(default_factory=set)
    completed: bool = False


class ExecutionWindow:
    def __init__(self, blocks):
        self.capacity = blocks
        self.claims = {}
        # Shared prefixes count once even when several live claims protect the same block.
        self.block_holders = Counter()

    def add(self, key, maximum):
        maximum = set(maximum)
        self.release(key)
        self.claims[key] = Claim(maximum)

    def narrow(self, key, maximum):
        claim = self.claims[key]
        maximum = set(maximum)
        if not maximum <= claim.maximum:
            raise ValueError("completion bound must not grow")
        claim.maximum = maximum

    def safe(self, key, blocks):
        blocks = set(blocks)
        # Even counting each proposed block as new, if every live holder could
        # finish first, removing any holder can only make the next one easier.
        spare = self.capacity - sum(len(claim.held) for claim in self.claims.values()) - len(blocks)
        if spare >= 0 and all(
            claim.completed or len(claim.maximum) <= spare
            for name, claim in self.claims.items() if claim.held or name == key
        ):
            return True

        projected = {k: c.held | (blocks if k == key else set())
                     for k, c in self.claims.items()}
        # Only actual holders participate. Non-holders wait without pins.
        pending = {k for k, held in projected.items() if held}
        while pending:
            live = set().union(*(projected[k] for k in pending))
            if len(live) > self.capacity:
                return False
            finishing = next((k for k in pending
                              if self.claims[k].completed or
                              len(live | self.claims[k].maximum) <= self.capacity), None)
            if finishing is None:
                return False
            pending.remove(finishing)
        return True

    def acquire(self, key, blocks):
        blocks = set(blocks)
        claim = self.claims[key]
        if not blocks <= claim.maximum or not self.safe(key, blocks):
            return False
        self.block_holders.update(blocks - claim.held)
        claim.held.update(blocks)
        return True

    def complete(self, key):
        self.claims[key].completed = True

    def release(self, key):
        claim = self.claims.pop(key, None)
        if claim is not None:
            for block in claim.held:
                self.block_holders[block] -= 1
                if self.block_holders[block] == 0:
                    del self.block_holders[block]

    @property
    def used(self):
        return len(self.block_holders)
