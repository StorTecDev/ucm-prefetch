"""Independent broker lifecycle and unchanged real control transport."""

import importlib
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time
from types import ModuleType
from uuid import uuid4

import msgspec
import pytest
import zmq


package = ModuleType("broker_process_source")
package.__path__ = [str(Path(__file__).resolve().parents[1] / "ucm/integration/vllm")]
sys.modules[package.__name__] = package
module = importlib.import_module(package.__name__ + ".early_broker")
control = importlib.import_module(package.__name__ + ".early_control")
ready = importlib.import_module(package.__name__ + ".early_ready")


def endpoint():
    return "ipc://@ucm-independent-" + uuid4().hex


def wait(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("control condition did not complete")
        time.sleep(0.005)


def test_independent_broker_serves_original_hello_without_model_imports():
    broker = module.BrokerProcess(endpoint(), dict(tp_size=1))
    context = zmq.Context()
    dealer = context.socket(zmq.DEALER)
    dealer.setsockopt(zmq.LINGER, 0)
    dealer.setsockopt(zmq.IDENTITY, b"s0")
    try:
        broker.start()
        assert broker.process.pid != os.getpid()
        maps = Path("/proc", str(broker.process.pid), "maps").read_text()
        assert "libtorch" not in maps
        assert "libascendcl" not in maps
        dealer.connect(broker.endpoint)
        dealer.send(msgspec.msgpack.encode(dict(op="hello", session="scheduler")))
        assert dealer.poll(5000)
        assert msgspec.msgpack.decode(dealer.recv())["op"] == "pong"
    finally:
        dealer.close()
        context.term()
        broker.close()
    assert broker.process.returncode == 0


def test_startup_failure_reaps_child_and_closes_shared_accounting():
    address = endpoint()
    context = zmq.Context()
    router = context.socket(zmq.ROUTER)
    router.bind(address)
    reserve = ready.ReadyReserve(12)
    broker = module.BrokerProcess(address, dict(tp_size=1), reserve)
    try:
        with pytest.raises((EOFError, RuntimeError)):
            broker.start()
        assert broker.process.poll() is not None
        assert broker.connection is None
        assert reserve._shared is None
    finally:
        broker.close()
        router.close()
        context.term()


def test_broker_exit_notifies_scheduler_without_restarting_domain():
    messages = []
    scheduler = control.Control(
        endpoint(), "s0", messages.append, dict(tp_size=1),
        ready_reserve=ready.ReadyReserve(12),
    )
    try:
        wait(lambda: scheduler.broker_process is not None
             and scheduler.broker_process.process is not None)
        child = scheduler.broker_process.process
        child.terminate()
        wait(lambda: any(message["op"] == "lost" for message in messages))
        wait(lambda: not scheduler.thread.is_alive())
        assert scheduler.broker_process.process is child
        assert child.poll() is not None
    finally:
        scheduler.close()


def test_parent_death_exits_broker_without_explicit_close():
    script = textwrap.dedent("""
        import importlib, os, sys
        from types import ModuleType
        package = ModuleType("parent_test")
        package.__path__ = [sys.argv[1]]
        sys.modules[package.__name__] = package
        module = importlib.import_module("parent_test.early_broker")
        broker = module.BrokerProcess(sys.argv[2], dict(tp_size=1))
        broker.start()
        print(broker.process.pid, flush=True)
        sys.stdin.readline()
        os._exit(0)
    """)
    parent = subprocess.Popen(
        [sys.executable, "-B", "-c", script, package.__path__[0], endpoint()],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True,
    )
    child = None
    identity = None
    try:
        child = int(parent.stdout.readline())
        stat = Path("/proc", str(child), "stat")
        identity = stat.read_text().split()[21]
        parent.communicate("\n", timeout=10)
        assert parent.returncode == 0

        def exited():
            try:
                fields = stat.read_text().split()
                return fields[21] != identity or fields[2] == "Z"
            except (FileNotFoundError, ProcessLookupError):
                return True

        wait(exited)
    finally:
        if parent.poll() is None:
            parent.terminate()
            parent.wait(timeout=10)
        if child is not None:
            stat = Path("/proc", str(child), "stat")
            try:
                fields = stat.read_text().split()
                if fields[21] == identity and fields[2] != "Z":
                    os.kill(child, 15)
            except (FileNotFoundError, ProcessLookupError):
                pass


def test_ready_watermark_and_release_cross_real_process_boundary():
    from test_early_owner_protocol import capability, blocks

    reserve = ready.ReadyReserve(12)
    responses, events, workers = [], [], []
    address = endpoint()

    def accept(message):
        responses.append(message)
        if "ready_sequence" in message:
            reserve.attempts.add((message["key"], message["attempt"], message["grant"]))
            reserve.publish(message["ready_sequence"])

    scheduler = control.Control(
        address, "s0", accept, dict(tp_size=2), ready_reserve=reserve
    )
    try:
        for rank in range(2):
            def receive(message, rank=rank):
                events.append((rank, message))
                worker = workers[rank]
                op = message["op"]
                if op == "owner_reserve":
                    worker.send(dict(op="owner_answer", key=message["key"],
                                     attempt=message["attempt"], state="READY",
                                     hit=len(message["blocks"]), capacity_truncated=False))
                elif op == "retire":
                    worker.send(dict(op="quiesced", key=message["key"],
                                     attempt=message["attempt"]))
                elif op == "release_owner":
                    worker.send(dict(op="owner_drained", key=message["key"],
                                     attempt=message["attempt"]))
            workers.append(control.Control(
                address, f"w0.{rank}", receive,
                hello=dict(capability=capability(rank=rank)),
            ))
        wait(lambda: all(worker.hint_sender is not None for worker in workers))
        scheduler.send(dict(op="job", key="target", dp=0, request="target",
                            blocks=blocks(20), sequence=1))
        wait(lambda: any(message.get("state") == "READY" for message in responses))
        answer = next(message for message in responses if message.get("state") == "READY")
        assert answer["ready_sequence"] == 1
        assert reserve.snapshot == (1, 1)
        consumed = dict(op="consumed", key="target", attempt=answer["attempt"])
        workers[0].send(consumed)
        time.sleep(0.03)
        assert not any(message["op"] == "release_owner" for _, message in events)
        workers[1].send(consumed)
        wait(lambda: sum(message["op"] == "forget"
                         for _, message in events) == 2)
        assert not any(message["op"] in ("domain_lost", "lost") for message in responses)
    finally:
        for worker in workers:
            worker.close()
        scheduler.close()
    assert not scheduler.thread.is_alive()
    assert scheduler.broker_process.process.returncode == 0

def _daemon_start_broker(address, output):
    broker = module.BrokerProcess(address, dict(tp_size=1))
    try:
        broker.start()
        output.put(broker.process.pid)
    finally:
        broker.close()


def test_broker_can_start_inside_daemon_engine_process():
    import multiprocessing

    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    engine = context.Process(target=_daemon_start_broker, args=(endpoint(), output))
    engine.daemon = True
    try:
        engine.start()
        assert output.get(timeout=10) != engine.pid
        engine.join(timeout=10)
        assert engine.exitcode == 0
    finally:
        if engine.is_alive():
            engine.terminate()
            engine.join(timeout=10)
        output.close()
        output.join_thread()


def test_cancelled_startup_reaps_owned_helper():
    import threading

    stopping = threading.Event()
    stopping.set()
    broker = module.BrokerProcess(endpoint(), dict(tp_size=1), stopping=stopping)
    with pytest.raises(RuntimeError, match="cancelled"):
        broker.start()
    assert broker.connection is None
    assert broker.process.poll() is not None


def test_active_domain_cannot_spawn_a_replacement_broker():
    broker = module.BrokerProcess(endpoint(), dict(tp_size=1))
    try:
        broker.start()
        child = broker.process
        with pytest.raises(RuntimeError, match="cannot be restarted"):
            broker.start()
        assert broker.process is child
        assert child.poll() is None
    finally:
        broker.close()
