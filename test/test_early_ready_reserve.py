"""Broker ordering, scheduler accounting and independent READY notification tests."""

import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from uuid import uuid4

import pytest


package = ModuleType("ready_reserve_source")
package.__path__ = [str(Path(__file__).resolve().parents[1] / "ucm/integration/vllm")]
sys.modules[package.__name__] = package
ready = importlib.import_module(package.__name__ + ".early_ready")


def message(op, key):
    return "worker", dict(op=op, key=key, attempt="generation", state="READY")


class Broker:
    fenced = False

    def __init__(self):
        self.jobs = {
            key: dict(status="PENDING", owner="worker", attempt="generation")
            for key in ("ordinary", "candidate", "third")
        }
        self.sessions = {"worker": "current"}
        self.received = []

    def receive(self, peer, event):
        self.received.append((peer, event))


def test_normal_and_promoted_results_share_one_watermark():
    reserve = ready.ReadyReserve(12)
    inbox = ready.ReadyInbox(reserve)
    sent = []
    for index in range(9):
        inbox.forward(lambda _, event: sent.append(event), "s0",
                      dict(op="result", key=str(index), state="READY"))
    assert [event["ready_sequence"] for event in sent] == list(range(1, 10))
    for event in sent[:7]:
        reserve.attempts.add((event["key"], "generation", 1))
        reserve.publish(event["ready_sequence"])
    assert reserve.promotion_budget(inbox.last_sent_ready_sequence) == 3
    # The eighth notification arrived after cancellation. No candidate is added.
    reserve.publish(8)
    assert reserve.promotion_budget(inbox.last_sent_ready_sequence) == 4
    reserve.discard(SimpleNamespace(key="0", attempt="generation", grant=1))
    assert reserve.promotion_budget(inbox.last_sent_ready_sequence) == 5
    with pytest.raises(RuntimeError, match="Noncontiguous"):
        reserve.publish(10)


@pytest.mark.parametrize("op,key,expected", [
    ("resource_wait", "ordinary", 1), ("handoff", "ordinary", 1),
    ("resource_wait", "candidate", 0), ("lookup_complete", "candidate", 0),
    ("lookup_complete", "ordinary", 0), ("execution_request", "ordinary", 0),
    ("cancel", "candidate", 0), ("domain_fault", "ordinary", 0),
    ("owner_drained", "ordinary", 0), ("future_message", "ordinary", 0),
])
def test_only_independent_bookkeeping_can_be_crossed(op, key, expected):
    inbox = ready.ReadyInbox(ready.ReadyReserve(1))
    inbox.tickets.add(("worker", "candidate", "generation"))
    inbox.pending.extend([message(op, key), message("owner_answer", "candidate")])
    assert inbox.next_index(Broker()) == expected


def test_priority_preserves_ordinary_progress_and_falls_back_to_fifo():
    inbox = ready.ReadyInbox(ready.ReadyReserve(1))
    inbox.TURN_SECONDS = 5
    broker = Broker()
    ordinary = message("resource_wait", "ordinary")
    complete = message("owner_answer", "candidate")
    inbox.tickets.add(("worker", "candidate", "generation"))
    items = iter([ordinary, complete])
    inbox.service(broker, lambda: next(items, None))
    assert broker.received == [complete, ordinary]
    assert not inbox.tickets
    items = iter([ordinary, complete])
    inbox.service(broker, lambda: next(items, None))
    assert broker.received[-1] == ordinary
    inbox.service(broker, lambda: next(items, None))
    assert broker.received[-1] == complete


def test_in_flight_results_disable_promotion_but_do_not_hold_ready():
    inbox = ready.ReadyInbox(ready.ReadyReserve(1))
    inbox.last_sent_ready_sequence = 1
    inbox.tickets.add(("worker", "candidate", "generation"))
    broker = Broker()
    ordinary = message("resource_wait", "ordinary")
    complete = message("owner_answer", "candidate")
    inbox.pending.extend([ordinary, complete])
    inbox.service(broker, lambda: None)
    inbox.service(broker, lambda: None)
    assert broker.received == [ordinary, complete]


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux local IPC")
def test_hints_do_not_publish_ready_and_ignore_stale_sessions():
    endpoint = "ipc://" + uuid4().hex
    receiver = ready.ReadyHints(endpoint, receiver=True)
    sender = ready.ReadyHints(endpoint, receiver=False)
    try:
        broker = Broker()
        tickets = set()
        _, answer = message("owner_answer", "candidate")
        sender.notify("worker", "previous", [(answer["key"], answer["attempt"])])
        receiver.receive(broker, tickets)
        assert not tickets
        for _ in range(2):
            sender.notify("worker", "current", [(answer["key"], answer["attempt"])])
        receiver.receive(broker, tickets)
        assert tickets == {("worker", "candidate", "generation")}
        assert not broker.received
        assert broker.jobs["candidate"]["status"] == "PENDING"
    finally:
        receiver.socket.close()
        sender.socket.close()

def test_barrier_does_not_read_past_the_formal_head():
    broker = Broker()
    inbox = ready.ReadyInbox(ready.ReadyReserve(2))
    inbox.tickets.add(("worker", "candidate", "generation"))
    head = message("execution_request", "ordinary")
    items = iter([head, message("owner_answer", "candidate")])
    inbox.service(broker, lambda: next(items, None))
    assert broker.received == [head]
    assert next(items)[1]["op"] == "owner_answer"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux local IPC")
def test_hint_batch_filters_each_attempt_before_use():
    endpoint = "ipc://" + uuid4().hex
    sender = ready.ReadyHints(endpoint, receiver=False)
    receiver = ready.ReadyHints(endpoint, receiver=True)
    try:
        sender.notify("worker", "current", [("ordinary", "generation"), ("candidate", "old"),
                                            ("third", "generation")])
        tickets = set()
        receiver.receive(Broker(), tickets)
        assert tickets == {("worker", "ordinary", "generation"), ("worker", "third", "generation")}
    finally:
        sender.socket.close()
        receiver.socket.close()

def test_producer_enqueues_formal_ready_and_wakes_without_advisory_io():
    import queue
    import threading
    from collections import deque
    from unittest.mock import Mock

    control_type = importlib.import_module(package.__name__ + ".early_control").Control
    control = control_type.__new__(control_type)
    control.stopping = threading.Event()
    control.outbound = queue.SimpleQueue()
    control.hint_outbound = deque(maxlen=32)
    control.hint_sender = Mock()
    control.writer = Mock()
    _, answer = message("owner_answer", "candidate")
    control.send(answer)
    assert control.outbound.get_nowait() is answer
    assert list(control.hint_outbound) == [("candidate", "generation")]
    control.writer.send.assert_called_once_with(b"x")
    control.hint_sender.notify.assert_not_called()


@pytest.mark.parametrize("op,field", [("consumed", "consumers"), ("quiesced", "terminal")])
def test_ready_only_crosses_incomplete_acknowledgements(op, field):
    broker = Broker()
    workers = {"worker", "second", "third", "fourth"}
    broker.workers = lambda dp: workers
    job = broker.jobs["ordinary"]
    job.update(dp=0, status="READY", retiring=op == "quiesced", consumers=set(), terminal=set())
    inbox = ready.ReadyInbox(ready.ReadyReserve(1))
    inbox.tickets.add(("worker", "candidate", "generation"))
    answer = message("owner_answer", "candidate")

    def selection(*peers):
        ack = message(op, "ordinary")[1]
        inbox.pending.clear()
        inbox.pending.extend((peer, ack) for peer in peers)
        inbox.pending.append(answer)
        return inbox.next_index(broker)

    assert selection("second", "third", "fourth") == 3
    assert selection("second", "third", "fourth", "worker") == 0
    job[field].add("worker")
    assert selection("third", "third", "second") == 3
    assert selection("third", "second", "fourth") == 0
    job[field].update(workers)
    assert selection("third") == 0


@pytest.mark.parametrize("op", ["consumed", "quiesced"])
def test_partial_ack_checks_generation_membership_and_same_key(op):
    broker = Broker()
    broker.workers = lambda dp: {"worker", "second"}
    broker.jobs["ordinary"].update(dp=0, status="READY", retiring=True, consumers=set(), terminal=set())
    event = message(op, "ordinary")[1]
    assert not ready.ReadyInbox.can_cross(broker, "unknown", event)
    assert not ready.ReadyInbox.can_cross(broker, "second", dict(event, attempt="previous"))
    inbox = ready.ReadyInbox(ready.ReadyReserve(1))
    inbox.tickets.add(("worker", "ordinary", "generation"))
    inbox.pending.extend([("second", event), message("owner_answer", "ordinary")])
    assert inbox.next_index(broker) == 0
