"""Track actual native task completion; metadata/submission never releases pins."""
from contextvars import ContextVar
import logging
import os
import threading
import time

log = logging.getLogger(__name__)
load_scope = ContextVar("ucm_owner_load_scope", default=())


class LoadConsistency:
    """Keep the original consistency object, including all its task state."""
    def __init__(self, native, observer):
        self.native, self.observer = native, observer

    def __getattr__(self, name):
        return getattr(self.native, name)

    def submit_load(self, store, block_ids_by_request, *args, **kwargs):
        token = load_scope.set(self.observer.scope(block_ids_by_request))
        try:
            return self.native.submit_load(store, block_ids_by_request, *args, **kwargs)
        finally:
            load_scope.reset(token)


class Consumer:
    def __init__(self, peer, send, poison):
        self.peer, self.send, self.poison = peer, send, poison
        self.lock = threading.RLock()
        self.generations, self.requests = {}, {}
        self.tasks = {}
        self.serial = 0
        self.fenced = None
        self.domain_active = False
        self._range_cache = {}

    def bind(self, tickets, layers):
        with self.lock:
            self.requests = {}
            self._range_cache.clear()
            for request_id, ticket in tickets.items():
                identity = (ticket["key"], ticket["attempt"])
                self.requests[request_id] = identity
                state = self.generations.setdefault(identity, dict(ticket=ticket, expected=None,
                    plan=None, remaining=None, layers=set(layers), completed={}, active=0,
                    stopped=False, consumed=False, quiesced=False))
                plan = tuple(ticket["expected"])
                if state["expected"] is None:
                    state["plan"] = plan
                    state["expected"] = frozenset(plan)
                    state["remaining"] = len(state["expected"]) * len(state["layers"])
                elif state["plan"] != plan and state["expected"] != set(plan):
                    self.poison("generation changed its native consumption plan")
                    raise RuntimeError("owner consumption plan changed")
                self._notify(state)

    def scope(self, request_ids):
        with self.lock:
            return tuple(self.requests[r] for r in request_ids if r in self.requests)

    def begin(self, ids, indexes):
        with self.lock:
            self.check_task("load", len(indexes))
            identities = load_scope.get()
            if not identities:
                return None
            for identity in identities:
                state = self.generations.get(identity)
                if state is None or state["stopped"] or state["consumed"]:
                    raise RuntimeError("late native load for a retired owner generation")
            raw = memoryview(ids).tobytes()
            shards = list(indexes)
            if len(raw) != len(shards) * 16:
                raise ValueError("native block identity width mismatch")
            ranges, matched = self._load_ranges(identities, raw, shards)
            self.serial += 1
            token = self.serial
            self.tasks[token] = (identities, ranges, time.monotonic_ns(), matched)
            for identity in identities:
                self.generations[identity]["active"] += 1
            return token

    def _load_ranges(self, identities, raw, shards):
        """Reuse immutable block coverage across layers of the same load plan."""
        if shards and shards.count(shards[0]) == len(shards):
            layer = int(shards[0])
            key = (identities, raw)
            cached = self._range_cache.get(key)
            if cached is None:
                blocks = frozenset(raw[i:i + 16] for i in range(0, len(raw), 16))
                covered = {
                    identity: blocks.intersection(self.generations[identity]["expected"])
                    for identity in identities
                }
                cached = (blocks, covered)
                # Only metadata is retained, never native pins. Bound partial
                # chunk variants and discard them on the next metadata bind.
                if len(self._range_cache) >= max(1, len(self.requests)):
                    self._range_cache.pop(next(iter(self._range_cache)))
                self._range_cache[key] = cached
            blocks, covered = cached
            return {layer: blocks}, {
                identity: {layer: subset} for identity, subset in covered.items()
            }

        ranges = {}
        for i, shard in enumerate(shards):
            ranges.setdefault(int(shard), set()).add(raw[16*i:16*(i+1)])
        ranges = {layer: frozenset(blocks) for layer, blocks in ranges.items()}
        return ranges, {
            identity: {
                layer: blocks.intersection(self.generations[identity]["expected"])
                for layer, blocks in ranges.items()
            }
            for identity in identities
        }

    def check_task(self, operation, shards):
        with self.lock:
            if self.fenced:
                raise RuntimeError("native I/O denied on fenced shared domain: " + self.fenced)

    def unscoped_failure(self):
        with self.lock:
            # A fallback load or native dump shares the same non-robust domain.
            # Its failure matters even when it has no owner-generation receipt.
            if self.domain_active and not self.fenced:
                self.fenced = "unscoped native I/O failed; drain state is not proven"
                self.poison(self.fenced)

    def complete(self, token, task, success):
        if token is None:
            return
        with self.lock:
            receipt = self.tasks.pop(token, None)
            if receipt is None:
                return
            identities, ranges, submitted, matched = receipt
            for identity in identities:
                state = self.generations[identity]
                state["active"] -= 1
                if success:
                    for shard, blocks in matched[identity].items():
                        if shard not in state["layers"]:
                            continue
                        done = state["completed"].get(shard)
                        # Immutable coverage can be shared by successive layers.
                        # Overlapping chunks still contribute only new blocks;
                        # only a successful native Wait reaches this update.
                        new = blocks if done is None else blocks.difference(done)
                        if done is None or new:
                            state["completed"][shard] = blocks if done is None else done.union(new)
                        state["remaining"] -= len(new)
                if log.isEnabledFor(logging.DEBUG):
                    log.debug("UCM_EARLY native_load_completed key=%s attempt=%s consumer=%s task=%s "
                             "layers=%s block_count=%s success=%s submitted_ns=%s completed_ns=%s "
                             "reference_acquired_ns=unknown pid=%s",
                             *identity, self.peer, task, sorted(ranges), sum(map(len, ranges.values())),
                             success, submitted, time.monotonic_ns(), os.getpid())
                if success:
                    self._notify(state)
            if not success:
                # A native failed submission/Wait does not prove all its queue
                # references and accelerator work drained. Quarantine the domain.
                self.fenced = "native consumer load failed; drain state is not proven"
                self.poison(self.fenced)

    def rejected(self, token):
        """The C++ capacity rejection guarantees no I/O was submitted for this receipt."""
        with self.lock:
            receipt = self.tasks.pop(token, None)
            if receipt is None:
                return
            identities, _, _, _ = receipt
            for identity in identities:
                state = self.generations[identity]
                state["active"] -= 1
                # Other tasks of the same generation may still own H2D buffers.
                # Retire through the normal all-consumer/drain protocol.
                self.send(dict(op="worker_retire", key=identity[0], attempt=identity[1],
                               reason="capacity rejection before I/O"))
                self._notify(state)

    def submitted(self, token, task):
        if not log.isEnabledFor(logging.DEBUG):
            return
        with self.lock:
            identities, ranges, started, _ = self.tasks[token]
            for key, attempt in identities:
                log.debug("UCM_EARLY native_load_submitted key=%s attempt=%s consumer=%s task=%s "
                          "layers=%s block_count=%s start_ns=%s ns=%s pid=%s",
                          key, attempt, self.peer, task, sorted(ranges), sum(map(len, ranges.values())),
                          started, time.monotonic_ns(), os.getpid())

    def _notify(self, state):
        if self.fenced:
            return
        # Completion is maintained by actual successful native Wait receipts.
        # Do not rescan the entire prefix across all prior layers on each Wait.
        complete = state["remaining"] == 0
        consumed = not state["stopped"] and not state["consumed"] and complete and not state["active"]
        quiesced = state["stopped"] and not state["active"] and not state["quiesced"]
        if not consumed and not quiesced:
            return
        ticket = state["ticket"]
        envelope = dict(key=ticket["key"], attempt=ticket["attempt"], consumer=self.peer)
        if consumed:
            state["consumed"] = True
            self.send(dict(envelope, op="consumed", ns=time.monotonic_ns()))
            log.info("UCM_EARLY native_consumed key=%s attempt=%s consumer=%s ns=%s",
                     ticket["key"], ticket["attempt"], self.peer, time.monotonic_ns())
        if quiesced:
            state["quiesced"] = True
            self.send(dict(envelope, op="quiesced", ns=time.monotonic_ns()))
            log.info("UCM_EARLY native_quiesced key=%s attempt=%s consumer=%s ns=%s",
                     ticket["key"], ticket["attempt"], self.peer, time.monotonic_ns())

    def retire(self, key, attempt):
        with self.lock:
            state = self.generations.setdefault((key, attempt), dict(
                ticket=dict(key=key, attempt=attempt), expected=None, layers=set(),
                plan=None, remaining=None, completed={}, active=0, stopped=False,
                consumed=False, quiesced=False))
            state["stopped"] = True
            self._notify(state)

    def forget(self, key, attempt):
        with self.lock:
            state = self.generations.get((key, attempt))
            if state and state["quiesced"] and not state["active"]:
                # Worker grant watermark rejects retired metadata before bind.
                del self.generations[(key, attempt)]
                self.requests = {r: identity for r, identity in self.requests.items()
                                 if identity != (key, attempt)}
