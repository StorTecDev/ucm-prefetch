"""Bounded lookahead for owner READY notifications on a local control endpoint."""

from collections import deque
import hashlib
import socket
import sys
import time

import msgspec


class ReadyReserve:
    """Track a single scheduler's accepted attempts and ordered result watermark."""

    def __init__(self, target):
        self.target = target
        self.attempts = set()
        # The scheduler publishes both fields together while holding its job lock.
        self.snapshot = (0, 0)

    def publish(self, processed_sequence=None):
        if processed_sequence is None:
            processed_sequence = self.snapshot[1]
        elif processed_sequence != self.snapshot[1] + 1:
            raise RuntimeError("Noncontiguous scheduler READY result sequence")
        self.snapshot = len(self.attempts), processed_sequence

    def discard(self, job):
        if not self.attempts:
            return
        ticket = (job.key, job.attempt, job.grant)
        if ticket in self.attempts:
            self.attempts.remove(ticket)
            self.publish()

    def promotion_budget(self, last_sent_ready_sequence):
        ready_awaiting_first_schedule, processed_sequence = self.snapshot
        unacknowledged_ready_notifications = last_sent_ready_sequence - processed_sequence
        return max(0, self.target - ready_awaiting_first_schedule - unacknowledged_ready_notifications)


class ReadyHints:
    """Best-effort datagrams independent of the ordered control transport."""

    def __init__(self, endpoint, *, receiver):
        if not sys.platform.startswith("linux") or not endpoint.startswith("ipc://"):
            raise OSError("Local READY hints require a Linux IPC endpoint")
        # An abstract endpoint dies with its socket, including on abnormal exit.
        digest = hashlib.sha256(endpoint.encode()).digest()
        self.address = b"\0ucm-early-ready-" + digest
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        try:
            self.socket.setblocking(False)
            if receiver:
                self.socket.bind(self.address)
        except BaseException:
            self.socket.close()
            raise

    def notify(self, peer, session, tickets):
        if len(tickets) == 1:
            # Existing receivers understand the original single-ticket datagram.
            message = (peer, session, *tickets[0])
        else:
            message = (peer, session, tickets)
        try:
            self.socket.sendto(msgspec.msgpack.encode(message), self.address)
        except OSError:
            pass  # The formal notification is already in the ordered transport.

    def receive(self, broker, tickets):
        for _ in range(8):
            try:
                data = self.socket.recv(8192)
            except BlockingIOError:
                break
            try:
                fields = msgspec.msgpack.decode(data)
            except msgspec.DecodeError:
                continue
            if not isinstance(fields, list) or len(fields) not in (3, 4):
                continue
            peer, session = fields[:2]
            if not isinstance(peer, str) or not isinstance(session, str) or broker.sessions.get(peer) != session:
                continue
            entries = [fields[2:]] if len(fields) == 4 else fields[2]
            if not isinstance(entries, list) or len(entries) > 32:
                continue
            for entry in entries:
                if (not isinstance(entry, list) or len(entry) != 2
                        or not all(isinstance(value, str) for value in entry)):
                    continue
                key, attempt = entry
                job = broker.jobs.get(key)
                if (job is not None and job.get("owner") == peer
                        and job.get("attempt") == attempt and job["status"] == "PENDING"
                        and len(tickets) < 256):
                    tickets.add((peer, key, attempt))


class ReadyInbox:
    """Promote independent READY messages within bounded FIFO service turns."""

    LOOKAHEAD = 24
    TURN_SECONDS = 0.0005
    MAX_PROMOTIONS = 4

    def __init__(self, reserve):
        self.reserve = reserve
        self.tickets = set()
        self.pending = deque()
        self.last_sent_ready_sequence = 0
        self.promoted = 0
        self.crossed = 0

    def forward(self, send, peer, message):
        if peer == "s0" and message.get("state") == "READY" and message["op"] == "result":
            sequence = self.last_sent_ready_sequence + 1
            send(peer, dict(message, ready_sequence=sequence))
            self.last_sent_ready_sequence = sequence
        else:
            send(peer, message)

    @staticmethod
    def can_cross(broker, peer, message):
        op = message["op"]
        if op not in ("resource_wait", "handoff", "consumed", "quiesced",
                      "execution_request", "lookup_complete"):
            return False
        job = broker.jobs.get(message.get("key"))
        if job is None or not job.get("owner"):
            return False
        if op in ("execution_request", "lookup_complete"):
            # Another job's grants may move later without invalidating the pins
            # already held by READY. Malformed input can fence the shared domain,
            # so it remains a barrier even when it belongs to a different job.
            if (peer != job["owner"] or peer not in broker.workers(job["dp"])
                    or message.get("attempt") != job["attempt"]
                    or job["status"] != "PENDING" or job.get("retiring")):
                return False
            if op == "lookup_complete":
                hit = message.get("hit")
                return type(hit) is int and 0 <= hit <= len(job["blocks"])
            values = [message.get(name) for name in ("batch", "begin", "count")]
            return (all(type(value) is int and value >= 0 for value in values)
                    and values[1] + values[2] <= len(job["blocks"]))
        if op in ("resource_wait", "handoff"):
            return True
        if message.get("attempt") != job["attempt"]:
            return False
        if op == "consumed" and job["status"] == "READY":
            received = job["consumers"]
        elif op == "quiesced" and job.get("retiring"):
            received = job["terminal"]
        else:
            return False

        workers = broker.workers(job["dp"])
        if peer not in workers:
            return False
        # Keep one outstanding worker as a barrier for the whole scanned prefix.
        # Checking each ACK in isolation could move READY past a complete cohort.
        barrier_peer = job["owner"]
        if barrier_peer in received:
            barrier_peer = min(workers - received, default=None)
        return barrier_peer is not None and peer != barrier_peer

    @staticmethod
    def valid_ready(broker, peer, message):
        job = broker.jobs.get(message.get("key"))
        hit = message.get("hit")
        return (job is not None and job.get("owner") == peer
                and peer in broker.workers(job["dp"])
                and job.get("attempt") == message.get("attempt")
                and job["status"] == "PENDING" and not job.get("retiring")
                and type(hit) is int and 0 <= hit <= len(job["blocks"])
                and type(message.get("capacity_truncated")) is bool)

    def next_index(self, broker):
        earlier_keys = set()
        for index, (peer, message) in enumerate(self.pending):
            key = message.get("key")
            if (message["op"] == "owner_answer" and message.get("state") == "READY"
                    and key not in earlier_keys
                    and (peer, key, message.get("attempt")) in self.tickets
                    and self.valid_ready(broker, peer, message)):
                return index
            if not self.can_cross(broker, peer, message):
                break
            earlier_keys.add(key)
        return 0

    def dispatch(self, broker, index):
        peer, message = self.pending[index]
        del self.pending[index]
        if message["op"] == "owner_answer":
            self.tickets.discard((peer, message["key"], message.get("attempt")))
        if index:
            self.promoted += 1
            self.crossed += index
        broker.receive(peer, message)

    def receive_hinted(self, broker, item, read_next, *, scheduler_readable):
        """Handle an ordered packet while an owner has advertised readiness."""
        self.pending.append(item)
        if scheduler_readable:
            self.dispatch(broker, 0)
        else:
            self.service(broker, read_next)

    def service(self, broker, read_next):
        if not self.pending:
            item = read_next()
            if item is None:
                return
            self.pending.append(item)
        if not self.tickets or broker.fenced or not self.can_cross(broker, *self.pending[0]):
            self.dispatch(broker, 0)
            return
        quota = min(self.MAX_PROMOTIONS, self.reserve.promotion_budget(self.last_sent_ready_sequence))
        if not quota:
            self.dispatch(broker, 0)
            return

        deadline = time.monotonic() + self.TURN_SECONDS
        # Reading past a barrier cannot reveal an eligible promotion this turn.
        while (len(self.pending) < self.LOOKAHEAD and time.monotonic() < deadline
               and self.can_cross(broker, *self.pending[-1])):
            item = read_next()
            if item is None:
                break
            self.pending.append(item)
        while len(self.pending) > 1 and quota and time.monotonic() < deadline:
            index = self.next_index(broker)
            if not index:
                break
            self.dispatch(broker, index)
            quota = min(quota - 1, self.reserve.promotion_budget(self.last_sent_ready_sequence))
        self.dispatch(broker, 0)
