"""Run existing early control handlers without sharing the engine interpreter."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import time
from types import ModuleType

import msgspec
import zmq

# A fresh helper must not execute the top-level UCM/model package initializers.
if __name__ == "__main__" and not __package__:
    package = ModuleType("_ucm_early_broker")
    package.__path__ = [str(Path(__file__).resolve().parent)]
    sys.modules[package.__name__] = package
    __package__ = package.__name__

from .early_ready import ReadyHints, ReadyInbox, ReadyReserve

log = logging.getLogger(__name__)
_HEADER = struct.Struct("!I")


def _send_packet(connection, message):
    payload = msgspec.msgpack.encode(message)
    connection.sendall(_HEADER.pack(len(payload)) + payload)


def _read_exact(connection, count, stopping=None, deadline=None):
    data = bytearray()
    while len(data) < count:
        if stopping is not None and stopping.is_set():
            raise RuntimeError("Early broker startup cancelled")
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Early broker startup timed out")
        try:
            part = connection.recv(count - len(data))
        except socket.timeout:
            if deadline is None:
                raise
            continue
        if not part:
            raise EOFError("Early broker lifecycle connection closed")
        data.extend(part)
    return data


def _receive_packet(connection, stopping=None, deadline=None):
    size, = _HEADER.unpack(_read_exact(connection, _HEADER.size, stopping, deadline))
    if size > 65536:
        raise ValueError("Early broker startup message is too large")
    return msgspec.msgpack.decode(_read_exact(connection, size, stopping, deadline))


class BrokerProcess:
    """Own one broker and terminate it if the engine's control thread exits."""

    def __init__(self, endpoint, config, reserve=None, stopping=None):
        self.endpoint, self.config, self.reserve = endpoint, config, reserve
        self.connection = None
        self.process = None
        self.stopping = stopping

    def start(self):
        if self.process is not None:
            raise RuntimeError("An early broker domain cannot be restarted")
        parent, child = socket.socketpair()
        self.connection = parent
        try:
            shared_descriptor = -1 if self.reserve is None else self.reserve.share()
            descriptors = (child.fileno(),)
            if shared_descriptor >= 0:
                descriptors += (shared_descriptor,)
            # Popen also works inside a daemon engine process. fork would copy
            # initialized accelerator state and the engine's other live threads.
            self.process = subprocess.Popen(
                [sys.executable, "-B", str(Path(__file__).resolve()),
                 str(child.fileno()), str(shared_descriptor)],
                pass_fds=descriptors,
            )
            child.close()
            parent.settimeout(0.1)
            _send_packet(parent, dict(
                endpoint=self.endpoint, config=self.config,
                ready_target=None if self.reserve is None else self.reserve.target,
            ))
            result = _receive_packet(parent, self.stopping, time.monotonic() + 30)
            if result.get("ready") is not True:
                raise RuntimeError(result.get("error", "Early broker failed to start"))
            parent.setblocking(False)
        except BaseException:
            child.close()
            self.close()
            raise

    def close(self):
        connection, self.connection = self.connection, None
        if connection is not None:
            try:
                connection.send(b"q")
            except OSError:
                pass
            connection.close()
        process = self.process
        if process is not None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Only this owned child is targeted; retained worker pins are
                # retired by the existing lost-domain protocol, never fabricated.
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        if self.reserve is not None:
            self.reserve.close_shared()


def run_broker(connection, settings, shared_descriptor=-1):
    """Keep protocol transitions serialized in a CPU-only helper.

    Args:
        connection: Lifecycle socket whose EOF proves the parent has exited.
        settings: Endpoint, original broker options and optional READY target.
        shared_descriptor: Inherited coherent scheduler accounting, or -1.
    """
    from .early_owner import OwnerBroker

    context = zmq.Context()
    router = context.socket(zmq.ROUTER)
    router.setsockopt(zmq.LINGER, 0)
    reserve = inbox = hints = None
    encode, decode = msgspec.msgpack.encode, msgspec.msgpack.decode
    try:
        router.bind(settings["endpoint"])
        poller = zmq.Poller()
        poller.register(router, zmq.POLLIN)
        poller.register(connection.fileno(), zmq.POLLIN)
        if shared_descriptor >= 0:
            reserve = ReadyReserve(settings["ready_target"])
            reserve.attach_shared(shared_descriptor)
            try:
                hints = ReadyHints(settings["endpoint"], receiver=True)
            except OSError:
                pass
            else:
                inbox = ReadyInbox(reserve)
                poller.register(hints.socket.fileno(), zmq.POLLIN)

        def send_to_peer(peer, message):
            router.send_multipart([peer.encode(), encode(message)])

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

        broker = OwnerBroker(route, **settings["config"])
        _send_packet(connection, dict(ready=True))
        connection.setblocking(False)
        last_expiry = time.monotonic()
        while True:
            events = dict(poller.poll(
                0 if inbox is not None and inbox.pending else broker.poll_timeout_ms()
            ))
            if connection.fileno() in events:
                # Parent shutdown and parent death both end this domain's broker.
                connection.recv(1)
                break
            if hints is not None and hints.socket.fileno() in events:
                hints.receive(broker, inbox.tickets)
            if inbox is not None and inbox.pending:
                inbox.service(broker, read_next, transport_readable=router in events)
            elif router in events:
                item = read_next()
                if item is not None:
                    if inbox is not None and inbox.tickets:
                        # Scheduler callbacks now run concurrently in the parent;
                        # the shared watermark accounts for notifications in flight.
                        inbox.receive_hinted(
                            broker, item, read_next, scheduler_readable=False
                        )
                    else:
                        broker.receive(*item)
            now = time.monotonic()
            broker.advance_retries(now)
            if now - last_expiry >= 2:
                broker.expire(now)
                last_expiry = now
    finally:
        if hints is not None:
            log.info(
                "UCM_EARLY ready_priority_summary promoted=%s crossed=%s "
                "scanned=%s priority_turns=%s",
                inbox.promoted, inbox.crossed, inbox.scanned, inbox.priority_turns,
            )
            hints.socket.close()
        if reserve is not None:
            reserve.close_shared()
        if shared_descriptor >= 0:
            os.close(shared_descriptor)
        router.close()
        context.term()


def _main():
    logging.basicConfig(
        level=os.getenv("UCM_LOG_LEVEL", "INFO").upper(),
        format="[%(levelname)s] %(name)s: %(message)s",
    )
    connection = socket.socket(fileno=int(sys.argv[1]))
    try:
        settings = _receive_packet(connection)
        run_broker(connection, settings, int(sys.argv[2]))
    except Exception:
        log.exception("UCM_EARLY independent broker failed")
        raise
    finally:
        connection.close()


if __name__ == "__main__":
    _main()
