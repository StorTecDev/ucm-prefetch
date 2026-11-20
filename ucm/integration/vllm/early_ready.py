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

    def notify(self, peer, session, message):
        data = msgspec.msgpack.encode((peer, session, message["key"], message["attempt"]))
        try:
            self.socket.sendto(data, self.address)
        except OSError:
            # A full datagram queue or a closing peer must never delay formal READY.
            pass

    def receive(self, broker, tickets):
        for _ in range(32):
            try:
                data = self.socket.recv(1024)
            except BlockingIOError:
                break
            try:
                fields = msgspec.msgpack.decode(data)
            except msgspec.DecodeError:
                continue
            if not isinstance(fields, list) or len(fields) != 4 or not all(
                isinstance(field, str) for field in fields
            ):
                continue
            peer, session, key, attempt = fields
            job = broker.jobs.get(key)
            if (broker.sessions.get(peer) == session and job is not None
                    and job.get("owner") == peer and job.get("attempt") == attempt
                    and job["status"] == "PENDING" and len(tickets) < 256):
                tickets.add((peer, key, attempt))


class ReadyInbox:
    """Preserve receive order across barriers and yield to the scheduler each turn."""

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
    def can_cross(broker, message):
        job = broker.jobs.get(message.get("key"))
        # Both operations are local bookkeeping for a live owner job. Neither
        # changes admissions, execution grants, claims, nor the domain lifetime.
        return job is not None and job.get("owner") and message["op"] in (
            "resource_wait", "handoff",
        )

    def next_index(self, broker):
        earlier_keys = set()
        for index, (peer, message) in enumerate(self.pending):
            key = message.get("key")
            if (message["op"] == "owner_answer" and message.get("state") == "READY"
                    and key not in earlier_keys
                    and (peer, key, message.get("attempt")) in self.tickets):
                return index
            if not self.can_cross(broker, message):
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

    def service(self, broker, read_next):
        if not self.pending:
            item = read_next()
            if item is None:
                return
            self.pending.append(item)
        quota = min(self.MAX_PROMOTIONS, self.reserve.promotion_budget(self.last_sent_ready_sequence))
        if not self.tickets or quota == 0 or broker.fenced:
            self.dispatch(broker, 0)
            return
        deadline = time.monotonic() + self.TURN_SECONDS
        while len(self.pending) < self.LOOKAHEAD and time.monotonic() < deadline:
            item = read_next()
            if item is None:
                break
            self.pending.append(item)
        while len(self.pending) > 1 and quota and time.monotonic() < deadline:
            if not self.reserve.promotion_budget(self.last_sent_ready_sequence):
                break
            index = self.next_index(broker)
            if not index:
                break
            self.dispatch(broker, index)
            quota -= 1
        self.dispatch(broker, 0)
