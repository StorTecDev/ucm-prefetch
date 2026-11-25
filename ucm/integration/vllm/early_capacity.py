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
        extra = blocks - self.claims[key].held
        available = self.capacity - self.used - sum(
            block not in self.block_holders for block in extra
        )
        if available < 0:
            return False

        # Every holder can finish first under this conservative bound. Charging
        # the unique union also avoids rejecting it just because prefixes overlap.
        if all(
            claim.completed or len(claim.maximum) <= available
            for name, claim in self.claims.items() if claim.held or name == key
        ):
            return True

        pending = [
            (claim, claim.held | blocks if name == key else claim.held)
            for name, claim in self.claims.items()
            if claim.held or (name == key and blocks)
        ]
        if all(
            claim.completed or len(claim.maximum - held) <= available
            for claim, held in pending
        ):
            return True

        # Simulated completion releases a shared block only at its last holder.
        # Rebuilding the union after every completion is quadratic in holders
        # even when no ordering search is needed.
        holders = self.block_holders.copy()
        holders.update(extra)
        live = set(holders)
        while pending:
            remaining = []
            for claim, held in pending:
                if not claim.completed and len(claim.maximum - live) > available:
                    remaining.append((claim, held))
                    continue
                for block in held:
                    holders[block] -= 1
                    if not holders[block]:
                        del holders[block]
                        live.remove(block)
                        available += 1
            if len(remaining) == len(pending):
                return False
            # Removing a finishable holder cannot make another one harder, but
            # earlier blocked holders must be revisited after space is released.
            pending = remaining
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
