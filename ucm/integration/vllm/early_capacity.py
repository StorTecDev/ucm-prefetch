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
        self._failed_acquisitions = {}
        self._failed_capacity = blocks

    def add(self, key, maximum):
        maximum = set(maximum)
        self.release(key)
        self.claims[key] = Claim(maximum)
        self._failed_acquisitions.clear()

    def narrow(self, key, maximum):
        claim = self.claims[key]
        maximum = set(maximum)
        if not maximum <= claim.maximum:
            raise ValueError("completion bound must not grow")
        if maximum != claim.maximum:
            claim.maximum = maximum
            self._failed_acquisitions.clear()

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
        blocks = frozenset(blocks)
        claim = self.claims[key]
        if not blocks <= claim.maximum:
            return False
        if self.capacity != self._failed_capacity:
            self._failed_acquisitions.clear()
            self._failed_capacity = self.capacity
        if blocks in self._failed_acquisitions.get(key, ()):
            return False
        if not self.safe(key, blocks):
            # Ordinary control traffic retries pending grants without necessarily
            # changing any pins. A denial stays valid until the proof inputs change.
            self._failed_acquisitions.setdefault(key, set()).add(blocks)
            return False
        extra = blocks - claim.held
        if extra:
            self.block_holders.update(extra)
            claim.held.update(extra)
            self._failed_acquisitions.clear()
        return True

    def complete(self, key):
        claim = self.claims[key]
        if not claim.completed:
            claim.completed = True
            self._failed_acquisitions.clear()

    def release(self, key):
        claim = self.claims.pop(key, None)
        if claim is not None:
            self._failed_acquisitions.clear()
            for block in claim.held:
                self.block_holders[block] -= 1
                if self.block_holders[block] == 0:
                    del self.block_holders[block]

    @property
    def used(self):
        return len(self.block_holders)
