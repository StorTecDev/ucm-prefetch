"""UCM control messages for unscheduled jobs and protected READY results.

The endpoint follows the native UCM shared identity. It owns no cache data,
pins, leases, or L1 identities. Each socket is confined to one control thread.
"""

from __future__ import annotations

from collections import deque
import logging
import os
import queue
import socket
import threading
import time

import msgspec
import zmq

from .early_ready import ReadyHints, ReadyInbox

log = logging.getLogger(__name__)


class BrokerState:
    """Single-threaded aggregation; also exercised without transport in tests."""

    def __init__(self, send, tp_size, timeout=300):
        self.send = send
        self.tp_size = tp_size
        self.timeout = timeout
        self.peers = {}
        self.jobs = {}

    def workers(self, dp):
        return {f"w{dp}.{rank}" for rank in range(self.tp_size)}

    def release(self, job, reason="terminal"):
        for worker in self.workers(job["dp"]):
            self.send(worker, {"op": "release", "key": job["key"], "reason": reason})

    def send_result(self, job, state, **fields):
        generation = {}
        if job.get("attempt") is not None:
            generation = {name: job.get(name) for name in ("owner", "attempt", "grant")}
        self.send(
            job["client"],
            dict(op="result", key=job["key"], state=state, **generation, **fields),
        )

    def fail(self, job, reason):
        if job["status"] in ("FAILED", "CANCELLED"):
            return
        job["status"] = "FAILED"
        self.release(job, "failed")
        log.warning("UCM_EARLY failed key=%s reason=%s", job["key"], reason)
        self.send_result(job, "FAILED", reason=reason)

    def dispatch(self, job):
        if job["status"] != "PENDING" or job.get("running"):
            return
        if not self.workers(job["dp"]).issubset(self.peers):
            return
        job["answers"] = {}
        job["running"] = True
        for worker in self.workers(job["dp"]):
            self.send(
                worker,
                {
                    "op": "reserve",
                    "key": job["key"],
                    "blocks": job["blocks"],
                    "request": job.get("request"),
                    "submitted_ns": job.get("submitted_ns"),
                },
            )

    def wake(self):
        for job in self.jobs.values():
            self.dispatch(job)

    def receive(self, peer, message, now=None):
        now = time.monotonic() if now is None else now
        self.peers[peer] = now
        op = message["op"]
        if op in ("hello", "ping"):
            self.send(peer, {"op": "pong"})
            self.wake()
        elif op == "job":
            key = message["key"]
            if key in self.jobs:
                return
            job = dict(message, client=peer, status="PENDING", created=now)
            self.jobs[key] = job
            self.dispatch(job)
        elif op == "answer":
            job = self.jobs.get(message["key"])
            if job is None or job["status"] != "PENDING" or not job.get("running"):
                return
            if peer not in self.workers(job["dp"]):
                return
            if message["state"] != "READY":
                self.fail(job, message.get("reason", "worker failed"))
                return
            job["answers"][peer] = message["hit"]
            if self.workers(job["dp"]) == job["answers"].keys():
                job["status"] = "READY"
                hit = min(job["answers"].values())
                log.info(
                    "UCM_EARLY ready key=%s dp=%s ranks=%s total_hit_blocks=%s ns=%s",
                    job["key"],
                    job["dp"],
                    len(job["answers"]),
                    hit,
                    time.monotonic_ns(),
                )
                self.send(
                    job["client"],
                    {"op": "result", "key": job["key"], "state": "READY", "hit": hit},
                )
        elif op == "handoff":
            job = self.jobs.get(message["key"])
            if job is not None and job["client"] == peer:
                for worker in self.workers(job["dp"]):
                    self.send(worker, message)
        elif op in ("finish", "cancel"):
            job = self.jobs.pop(message["key"], None)
            if job:
                job["status"] = "CANCELLED"
                self.release(job)

    def pending_expired(self, job, now):
        return now - job["created"] > self.timeout

    def expire(self, now=None):
        now = time.monotonic() if now is None else now
        dead = {peer for peer, seen in self.peers.items() if now - seen > 60}
        for peer in dead:
            self.peers.pop(peer, None)
        for job in self.jobs.values():
            if job["status"] == "PENDING" and self.pending_expired(job, now):
                self.fail(job, "bounded pending timeout")
            elif dead.intersection(self.workers(job["dp"]) | {job["client"]}):
                self.fail(job, "control peer lost")


class Control:
    def __init__(self, endpoint, identity, callback, broker_config=None, hello=None, ready_reserve=None):
        self.endpoint, self.identity, self.callback = endpoint, identity, callback
        self.ready_inbox = ReadyInbox(ready_reserve) if ready_reserve is not None else None
        self.hint_sender = None
        self.hint_outbound = deque(maxlen=32)
        self.broker_config = broker_config
        import uuid
        self.hello = dict(hello or {}, session=uuid.uuid4().hex)
        self.outbound = queue.SimpleQueue()
        self.reader, self.writer = socket.socketpair()
        self.writer.setblocking(False)
        self.stopping = threading.Event()
        self.thread = threading.Thread(
            target=self._run, name="ucm-early-control", daemon=True
        )
        self.thread.start()

    def send(self, message):
        if self.stopping.is_set():
            return
        self.outbound.put(message)
        if self.hint_sender is not None and message["op"] == "owner_answer" and message["state"] == "READY":
            self.hint_outbound.append((message["key"], message["attempt"]))
        try:
            self.writer.send(b"x")
        except BlockingIOError:
            pass  # already readable; the queue is drained completely

    def close(self):
        self.stopping.set()
        try:
            self.writer.send(b"x")
        except (OSError, BlockingIOError):
            pass
        self.thread.join(timeout=5)

    def _run(self):
        context = zmq.Context()
        dealer = context.socket(zmq.DEALER)
        dealer.setsockopt(zmq.IDENTITY, self.identity.encode())
        dealer.setsockopt(zmq.LINGER, 0)
        router = hint_receiver = None
        inbox = self.ready_inbox
        encode, decode = msgspec.msgpack.encode, msgspec.msgpack.decode
        try:
            poller = zmq.Poller()
            if self.broker_config is not None:
                router = context.socket(zmq.ROUTER)
                router.setsockopt(zmq.LINGER, 0)
                router.bind(self.endpoint)

                if inbox is not None:
                    try:
                        hint_receiver = ReadyHints(self.endpoint, receiver=True)
                    except OSError:
                        inbox = None
                    else:
                        poller.register(hint_receiver.socket.fileno(), zmq.POLLIN)

                def send_to_peer(peer, message):
                    router.send_multipart([peer.encode(), encode(message)])

                from .early_owner import OwnerBroker
                def route(peer, message):
                    if inbox is None:
                        return send_to_peer(peer, message)
                    if message["op"] == "pong" and peer.startswith("w"):
                        message = dict(message, ready_hints=True)
                    inbox.forward(send_to_peer, peer, message)

                def read_next():
                    try:
                        peer, payload = router.recv_multipart(zmq.NOBLOCK)
                    except zmq.Again:
                        return None
                    return peer.decode(), decode(payload)

                broker = OwnerBroker(route, **self.broker_config)
                poller.register(router, zmq.POLLIN)
            dealer.connect(self.endpoint)
            poller.register(dealer, zmq.POLLIN)
            poller.register(self.reader.fileno(), zmq.POLLIN)
            dealer.send(encode(dict(self.hello, op="hello")))
            heartbeat, last_reply = 0.0, None
            connecting_since = time.monotonic()
            connect_timeout = float(os.getenv("VLLM_ENGINE_READY_TIMEOUT_S", "3600"))
            while not self.stopping.is_set():
                events = dict(
                    poller.poll(0 if inbox is not None and inbox.pending else
                                broker.poll_timeout_ms() if router is not None else 1000)
                )  # control deadlines only; no storage polling
                if self.reader.fileno() in events:
                    self.reader.recv(65536)
                    if self.hint_outbound:
                        # Wake the ordered transport first. If READY is behind normal
                        # messages, its hint can still pass the rest of that backlog.
                        if not self.outbound.empty():
                            dealer.send(encode(self.outbound.get()))
                        tickets = [self.hint_outbound.popleft()
                                   for _ in range(len(self.hint_outbound))]
                        self.hint_sender.notify(self.identity, self.hello["session"], tickets)
                    while not self.outbound.empty():
                        dealer.send(encode(self.outbound.get()))
                if hint_receiver is not None and hint_receiver.socket.fileno() in events:
                    hint_receiver.receive(broker, inbox.tickets)
                if inbox is not None and inbox.pending:
                    inbox.service(broker, read_next)
                elif router is not None and router in events:
                    peer, payload = router.recv_multipart()
                    identity, message = peer.decode(), decode(payload)
                    if inbox is not None and inbox.tickets and inbox.can_cross(broker, message):
                        inbox.pending.append((identity, message))
                        inbox.service(broker, read_next)
                    else:
                        if inbox is not None and message["op"] == "owner_answer":
                            inbox.tickets.discard((identity, message["key"], message.get("attempt")))
                        broker.receive(identity, message)
                if dealer in events:
                    message = decode(dealer.recv())
                    last_reply = time.monotonic()
                    if message.get("ready_hints") and self.hint_sender is None:
                        try:
                            self.hint_sender = ReadyHints(self.endpoint, receiver=False)
                        except OSError:
                            pass
                    if message["op"] != "pong":
                        self.callback(message)
                now = time.monotonic()
                if router is not None:
                    broker.advance_retries(now)
                if now - heartbeat >= 2:
                    dealer.send(encode({"op": "ping", "session": self.hello["session"]}))
                    heartbeat = now
                    if router is not None:
                        broker.expire(now)
                if (
                    last_reply is None and now - connecting_since > connect_timeout
                ) or (last_reply is not None and now - last_reply > 60):
                    raise TimeoutError("Early control broker unavailable")
        except Exception as error:
            log.exception("UCM_EARLY control failed identity=%s", self.identity)
            self.callback({"op": "lost", "reason": str(error)})
        finally:
            self.stopping.set()
            if hint_receiver is not None:
                log.info("UCM_EARLY ready_priority_summary promoted=%s crossed=%s",
                         inbox.promoted, inbox.crossed)
                hint_receiver.socket.close()
            if self.hint_sender is not None:
                self.hint_sender.socket.close()
            dealer.close()
            if router is not None:
                router.close()
            context.term()
            self.reader.close()
            self.writer.close()
