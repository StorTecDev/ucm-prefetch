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
    return "worker", dict(op=op, key=key, attempt="generation", state="READY",
                        hit=1, capacity_truncated=False, batch=0, begin=0, count=1)


class Broker:
    fenced = False
    ready_order_epoch = 0

    def __init__(self):
        self.jobs = {
            key: dict(status="PENDING", owner="worker", attempt="generation", dp=0, blocks=[b"block"])
            for key in ("ordinary", "candidate", "third")
        }
        self.sessions = {"worker": "current"}
        self.received = []

    def workers(self, dp):
        return {"worker"}

    def receive(self, peer, event):
        self.received.append((peer, event))


def promoted_crossings(inbox, broker):
    previous = inbox.crossed
    inbox.service(broker, lambda: None)
    return inbox.crossed - previous


class PublishingBroker(Broker):
    def __init__(self, inbox, keys):
        super().__init__()
        self.inbox = inbox
        self.after_ready = lambda: None
        self.jobs.update({
            key: dict(status="PENDING", owner="worker", attempt="generation",
                      dp=0, blocks=[b"block"])
            for key in keys
        })

    def receive(self, peer, event):
        super().receive(peer, event)
        if event["op"] == "owner_answer" and self.inbox.valid_ready(self, peer, event):
            self.jobs[event["key"]]["status"] = "READY"
            self.inbox.forward(
                lambda *_: None, "s0",
                dict(op="result", state="READY", key=event["key"]),
            )
            self.after_ready()


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
    ("lookup_complete", "ordinary", 1), ("execution_request", "ordinary", 1),
    ("cancel", "candidate", 0), ("domain_fault", "ordinary", 0),
    ("owner_drained", "ordinary", 0), ("future_message", "ordinary", 0),
])
def test_only_independent_valid_messages_can_be_crossed(op, key, expected):
    inbox = ready.ReadyInbox(ready.ReadyReserve(1))
    inbox.tickets.add(("worker", "candidate", "generation"))
    inbox.pending.extend([message(op, key), message("owner_answer", "candidate")])
    assert promoted_crossings(inbox, Broker()) == expected


def test_priority_preserves_ordinary_progress_and_falls_back_to_fifo():
    inbox = ready.ReadyInbox(ready.ReadyReserve(1))
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
    head = message("domain_fault", "ordinary")
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
    from unittest.mock import Mock

    control_type = importlib.import_module(package.__name__ + ".early_control").Control
    control = control_type.__new__(control_type)
    control.stopping = threading.Event()
    control.outbound = queue.SimpleQueue()
    control.hint_sender = Mock()
    control.writer = Mock()
    _, answer = message("owner_answer", "candidate")
    control.send(answer)
    assert control.outbound.get_nowait() is answer
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
        inbox.tickets.add(("worker", "candidate", "generation"))
        inbox.pending.clear()
        inbox.pending.extend((peer, ack) for peer in peers)
        inbox.pending.append(answer)
        return promoted_crossings(inbox, broker)

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
    assert promoted_crossings(inbox, broker) == 0


def test_same_request_prerequisite_is_processed_before_ready():
    inbox = ready.ReadyInbox(ready.ReadyReserve(2))
    inbox.tickets.add(("worker", "candidate", "generation"))
    broker = Broker()
    head = message("execution_request", "ordinary")
    lookup = message("lookup_complete", "candidate")
    answer = message("owner_answer", "candidate")
    backlog = iter([lookup, answer])
    inbox.receive_hinted(broker, head, lambda: next(backlog, None), scheduler_readable=False)
    assert broker.received == [head]
    inbox.service(broker, lambda: next(backlog, None))
    assert broker.received == [head, lookup]
    inbox.service(broker, lambda: next(backlog, None))
    assert broker.received == [head, lookup, answer]


@pytest.mark.parametrize("busy,fenced,full", [
    (True, False, False), (False, True, False), (False, False, True),
])
def test_hinted_turn_yields_without_speculative_receive(busy, fenced, full):
    inbox = ready.ReadyInbox(ready.ReadyReserve(1))
    inbox.tickets.add(("worker", "candidate", "generation"))
    inbox.last_sent_ready_sequence = int(full)
    broker = Broker()
    broker.fenced = fenced
    head = message("execution_request", "ordinary")

    def unexpected_read():
        pytest.fail("A yielded turn must leave the next formal packet in order")

    inbox.receive_hinted(broker, head, unexpected_read, scheduler_readable=busy)
    assert broker.received == [head]


@pytest.mark.parametrize("op,change", [
    ("execution_request", {"count": -1}),
    ("execution_request", {"batch": True}),
    ("execution_request", {"begin": 1, "count": 1}),
    ("lookup_complete", {"hit": 2}),
    ("lookup_complete", {"hit": True}),
    ("execution_request", {"attempt": "obsolete"}),
])
def test_invalid_admission_message_keeps_fifo(op, change):
    inbox = ready.ReadyInbox(ready.ReadyReserve(1))
    inbox.tickets.add(("worker", "candidate", "generation"))
    peer, item = message(op, "ordinary")
    inbox.pending.extend([(peer, dict(item, **change)), message("owner_answer", "candidate")])
    assert promoted_crossings(inbox, Broker()) == 0


@pytest.mark.parametrize("change", [
    {"attempt": "obsolete"}, {"hit": -1}, {"hit": True},
    {"capacity_truncated": None}, {"state": "FAILED"},
])
def test_stale_or_invalid_ready_is_not_promoted(change):
    inbox = ready.ReadyInbox(ready.ReadyReserve(1))
    peer, answer = message("owner_answer", "candidate")
    answer.update(change)
    inbox.tickets.add((peer, "candidate", answer["attempt"]))
    inbox.pending.extend([message("execution_request", "ordinary"), (peer, answer)])
    assert promoted_crossings(inbox, Broker()) == 0


def test_ready_passes_unrelated_lookup_and_grant_without_reordering_them():
    inbox = ready.ReadyInbox(ready.ReadyReserve(1))
    broker = Broker()
    inbox.tickets.add(("worker", "candidate", "generation"))
    items = [message("execution_request", "ordinary"), message("lookup_complete", "third")]
    answer = message("owner_answer", "candidate")
    inbox.pending.extend(items + [answer])
    inbox.service(broker, lambda: None)
    assert broker.received == [answer, items[0]]
    inbox.service(broker, lambda: None)
    assert broker.received == [answer, *items]


def test_promoted_ready_keeps_capacity_until_every_consumer_drains():
    from test_early_owner_protocol import Cluster

    cluster = Cluster()
    a = cluster.job("candidate", count=20)
    b = cluster.job("ordinary", count=20, offset=20)
    cluster.message(a, "lookup_complete", hit=20)
    cluster.message(a, "execution_request", batch=0, begin=0, count=20)
    window = a["domain"]["execution"]
    held = window.used
    assert held == 20
    inbox = ready.ReadyInbox(ready.ReadyReserve(1))
    inbox.tickets.add((a["owner"], a["key"], a["attempt"]))
    lookup = (b["owner"], dict(op="lookup_complete", key=b["key"], attempt=b["attempt"], hit=20))
    grant = (b["owner"], dict(op="execution_request", key=b["key"], attempt=b["attempt"], batch=0, begin=0, count=20))
    answer = (a["owner"], dict(op="owner_answer", key=a["key"], attempt=a["attempt"], state="READY", hit=20, capacity_truncated=False))
    inbox.pending.extend([lookup, grant, answer])
    cluster.out.clear()
    inbox.service(cluster.b, lambda: None)
    assert inbox.promoted == 1
    assert a["status"] == "READY" and window.used == held
    assert not b["execution_grants"]
    assert cluster.out[0][1]["op"] == "result"
    inbox.service(cluster.b, lambda: None)
    assert b["execution_grants"] and window.used == 40
    cluster.message(a, "consumed", "w0.0")
    assert window.used == 40 and not a.get("retiring")
    cluster.message(a, "consumed", "w0.1")
    assert a["retiring"] and window.used == 40
    cluster.message(a, "quiesced", "w0.0")
    cluster.message(a, "owner_drained")
    assert window.used == 40
    cluster.message(a, "quiesced", "w0.1")
    cluster.message(a, "owner_drained")
    assert window.used == 20 and a["key"] not in cluster.b.jobs
    cluster.drain(b)
    assert window.used == 0


@pytest.mark.parametrize("op", ["cancel", "domain_fault", "consumer_failed"])
def test_control_barrier_prevents_ready_publication(op):
    from test_early_owner_protocol import Cluster

    cluster = Cluster()
    a, b = cluster.job("candidate"), cluster.job("ordinary")
    inbox = ready.ReadyInbox(ready.ReadyReserve(1))
    inbox.tickets.add((a["owner"], a["key"], a["attempt"]))
    peer = a["client"] if op == "cancel" else b["owner"]
    affected = a if op == "cancel" else b
    barrier = (peer, dict(op=op, key=affected["key"], attempt=affected["attempt"]))
    answer = (a["owner"], dict(op="owner_answer", key=a["key"], attempt=a["attempt"], state="READY", hit=4, capacity_truncated=False))
    inbox.pending.extend([barrier, answer])
    cluster.out.clear()
    inbox.service(cluster.b, lambda: None)
    inbox.service(cluster.b, lambda: None)
    assert not any(msg.get("state") == "READY" for _, msg in cluster.out)
    assert inbox.promoted == 0

def prepare_turn(keys, target=None):
    inbox = ready.ReadyInbox(ready.ReadyReserve(target or len(keys)))
    broker = PublishingBroker(inbox, keys)
    inbox.tickets.update(("worker", key, "generation") for key in keys)
    return inbox, broker


def test_deficit_can_promote_more_than_four_ready_beyond_old_lookahead():
    keys = [f"candidate-{index}" for index in range(7)]
    inbox, broker = prepare_turn(keys)
    ordinary = [message("execution_request", "ordinary") for _ in range(30)]
    answers = [message("owner_answer", key) for key in keys]
    backlog = iter(ordinary + answers)
    inbox.service(broker, lambda: next(backlog, None))
    assert broker.received == answers + ordinary[:1]
    assert list(inbox.pending) == ordinary[1:]
    assert inbox.last_sent_ready_sequence == 7


def test_one_pass_checks_each_deferred_message_once():
    keys = [f"candidate-{index}" for index in range(6)]
    inbox, broker = prepare_turn(keys)
    ordinary = [message("execution_request", "ordinary") for _ in range(40)]
    checked = []
    can_cross = inbox.can_cross

    def check(broker, peer, event):
        checked.append(event)
        return can_cross(broker, peer, event)

    inbox.can_cross = check
    inbox.pending.extend(ordinary + [message("owner_answer", key) for key in keys])
    inbox.service(broker, lambda: None)
    assert checked == [event for _, event in ordinary]
    assert inbox.scanned == len(ordinary) + len(keys)
    assert inbox.promoted == len(keys)


def test_entry_deficit_includes_accepted_and_unacknowledged_results():
    inbox, broker = prepare_turn(["candidate", "third"], target=4)
    inbox.reserve.attempts.add(("accepted", "generation", 1))
    inbox.reserve.publish()
    inbox.last_sent_ready_sequence = 2
    ordinary = message("execution_request", "ordinary")
    first, second = [message("owner_answer", key) for key in ("candidate", "third")]
    inbox.pending.extend([ordinary, first, second])
    inbox.service(broker, lambda: None)
    assert broker.received == [first, ordinary]
    assert list(inbox.pending) == [second]
    assert inbox.last_sent_ready_sequence == 3


def test_consumption_during_turn_cannot_expand_entry_deficit():
    keys = [f"candidate-{index}" for index in range(4)]
    inbox, broker = prepare_turn(keys, target=2)
    broker.after_ready = lambda: inbox.reserve.publish(inbox.last_sent_ready_sequence)
    ordinary = message("execution_request", "ordinary")
    answers = [message("owner_answer", key) for key in keys]
    inbox.pending.extend([ordinary] + answers)
    inbox.service(broker, lambda: None)
    assert broker.received == answers[:2] + [ordinary]
    assert inbox.last_sent_ready_sequence == 2
    assert inbox.reserve.promotion_budget(2) == 2
    assert list(inbox.pending) == answers[2:]


def test_live_reserve_can_shrink_remaining_entry_quota():
    inbox, broker = prepare_turn(["candidate", "third"], target=2)

    def fill_reserve():
        inbox.reserve.attempts.update({("candidate", "generation", 1), ("existing", "old", 1)})
        inbox.reserve.publish(inbox.last_sent_ready_sequence)

    broker.after_ready = fill_reserve
    ordinary = message("execution_request", "ordinary")
    answers = [message("owner_answer", key) for key in ("candidate", "third")]
    inbox.pending.extend([ordinary] + answers)
    inbox.service(broker, lambda: None)
    assert broker.received == answers[:1] + [ordinary]
    assert list(inbox.pending) == answers[1:]


@pytest.mark.parametrize("failure", ["receive", "decode"])
def test_exception_restores_deferred_order_without_replaying_ready(failure):
    inbox, broker = prepare_turn(["candidate", "third"])
    ordinary = [message("execution_request", "ordinary"), message("lookup_complete", "ordinary")]
    first = message("owner_answer", "candidate")
    inbox.pending.extend(ordinary + [first])

    def fail():
        raise RuntimeError("test interruption")

    if failure == "receive":
        broker.after_ready = fail
    with pytest.raises(RuntimeError, match="test interruption"):
        inbox.service(broker, fail)
    assert broker.received == [first]
    assert list(inbox.pending) == ordinary
    assert inbox.last_sent_ready_sequence == 1
    assert ("worker", "candidate", "generation") not in inbox.tickets


@pytest.mark.parametrize("change", ["epoch", "fence"])
def test_state_change_ends_turn_before_cached_predicates_are_reused(change):
    inbox, broker = prepare_turn(["candidate", "third"])

    def invalidate():
        if change == "epoch":
            broker.ready_order_epoch += 1
        else:
            broker.fenced = True

    broker.after_ready = invalidate
    ordinary = [message("execution_request", "ordinary"), message("lookup_complete", "ordinary")]
    answers = [message("owner_answer", key) for key in ("candidate", "third")]
    inbox.pending.extend(ordinary + answers)
    inbox.service(broker, lambda: None)
    assert broker.received == [answers[0], ordinary[0]]
    assert list(inbox.pending) == ordinary[1:] + answers[1:]


def test_missing_formal_ready_still_advances_ordinary_fifo():
    inbox, broker = prepare_turn(["candidate"])
    ordinary = [message("execution_request", "ordinary"), message("lookup_complete", "ordinary")]
    backlog = iter(ordinary)
    inbox.service(broker, lambda: next(backlog, None))
    assert broker.received == ordinary[:1]
    inbox.service(broker, lambda: None)
    assert broker.received == ordinary
    assert not inbox.pending
    assert inbox.tickets


def test_fault_stops_lookahead_and_retains_original_order():
    inbox, broker = prepare_turn(["candidate"])
    ordinary = message("execution_request", "ordinary")
    fault = message("domain_fault", "ordinary")
    inbox.pending.extend([ordinary, fault])
    inbox.service(broker, lambda: pytest.fail("Cannot read beyond a domain fault"))
    assert broker.received == [ordinary]
    assert list(inbox.pending) == [fault]


def test_ordinary_work_progresses_between_successive_ready_deficits():
    keys = [f"candidate-{index}" for index in range(5)]
    inbox, broker = prepare_turn(keys, target=1)
    broker.after_ready = lambda: inbox.reserve.publish(inbox.last_sent_ready_sequence)
    ordinary = [message("execution_request", "ordinary") for _ in keys]
    for event, key in zip(ordinary, keys):
        inbox.pending.extend([event, message("owner_answer", key)])
    for index, key in enumerate(keys):
        inbox.service(broker, lambda: None)
        assert broker.received[-2:] == [message("owner_answer", key), ordinary[index]]
    assert not inbox.pending


def test_same_attempt_prefix_remains_a_barrier_after_another_ready():
    inbox, broker = prepare_turn(["candidate", "third"])
    ordinary = message("execution_request", "ordinary")
    prerequisite = message("lookup_complete", "third")
    first = message("owner_answer", "candidate")
    second = message("owner_answer", "third")
    inbox.pending.extend([ordinary, prerequisite, first, second])
    inbox.service(broker, lambda: None)
    assert broker.received == [first, ordinary]
    assert list(inbox.pending) == [prerequisite, second]
    inbox.service(broker, lambda: None)
    assert broker.received[-1] == prerequisite
    inbox.service(broker, lambda: None)
    assert broker.received[-1] == second


def test_consecutive_real_ready_keeps_shared_pins_and_capacity_until_drain():
    from test_early_owner_protocol import Cluster

    cluster = Cluster()
    first = cluster.job("first", count=20)
    second = cluster.job("second", count=20)
    ordinary = cluster.job("ordinary", count=20, offset=20)
    for job in (first, second):
        cluster.message(job, "lookup_complete", hit=20)
        cluster.message(job, "execution_request", batch=0, begin=0, count=20)
    window = first["domain"]["execution"]
    assert window.used == 20
    inbox = ready.ReadyInbox(ready.ReadyReserve(2))
    send = cluster.b.send
    cluster.b.send = lambda peer, event: inbox.forward(send, peer, event)
    answers = []
    for job in (first, second):
        inbox.tickets.add((job["owner"], job["key"], job["attempt"]))
        answers.append((job["owner"], dict(
            op="owner_answer", key=job["key"], attempt=job["attempt"],
            state="READY", hit=20, capacity_truncated=False,
        )))
    lookup = (ordinary["owner"], dict(
        op="lookup_complete", key=ordinary["key"], attempt=ordinary["attempt"], hit=20,
    ))
    grant = (ordinary["owner"], dict(
        op="execution_request", key=ordinary["key"], attempt=ordinary["attempt"],
        batch=0, begin=0, count=20,
    ))
    inbox.pending.extend([lookup, grant] + answers)
    epoch = cluster.b.ready_order_epoch
    inbox.service(cluster.b, lambda: None)
    assert inbox.promoted == 2
    assert first["status"] == second["status"] == "READY"
    assert window.used == 20
    assert cluster.b.ready_order_epoch == epoch + 1
    assert not ordinary["execution_grants"]
    inbox.service(cluster.b, lambda: None)
    assert window.used == 40
    cluster.drain(first)
    assert window.used == 40
    cluster.drain(second)
    assert window.used == 20
    cluster.drain(ordinary)
    assert window.used == 0


def test_empty_transport_does_not_rescan_known_ordinary_prefix():
    inbox, broker = prepare_turn(["candidate"])
    ordinary = [message("execution_request", "ordinary") for _ in range(40)]
    inbox.pending.extend(ordinary)
    inbox.service(broker, lambda: None, transport_readable=False)
    scanned = inbox.scanned
    while inbox.pending:
        inbox.service(
            broker, lambda: pytest.fail("No new transport readiness"),
            transport_readable=False,
        )
    assert broker.received == ordinary
    assert scanned == inbox.scanned == len(ordinary)
    assert inbox._ordinary_prefix_size == 0


def test_new_transport_readiness_revalidates_prefix_and_promotes_ready():
    inbox, broker = prepare_turn(["candidate"])
    ordinary = [message("execution_request", "ordinary") for _ in range(3)]
    answer = message("owner_answer", "candidate")
    inbox.pending.extend(ordinary)
    inbox.service(broker, lambda: None, transport_readable=False)
    backlog = iter([answer])
    inbox.service(broker, lambda: next(backlog, None), transport_readable=True)
    assert broker.received == [ordinary[0], answer, ordinary[1]]
    assert list(inbox.pending) == ordinary[2:]
    assert inbox.last_sent_ready_sequence == 1


def test_state_change_does_not_require_rescanning_packet_types_without_data():
    inbox, broker = prepare_turn(["candidate"])
    ordinary = [message("execution_request", "ordinary") for _ in range(3)]
    inbox.pending.extend(ordinary)
    inbox.service(broker, lambda: None, transport_readable=False)
    broker.jobs["ordinary"]["status"] = "CANCELLED"
    broker.ready_order_epoch += 1
    inbox.service(
        broker, lambda: pytest.fail("Only FIFO must advance without new input"),
        transport_readable=False,
    )
    assert broker.received == ordinary[:2]
    answer = message("owner_answer", "candidate")
    backlog = iter([answer])
    inbox.service(
        broker, lambda: next(backlog, None), transport_readable=True
    )
    assert broker.received == ordinary
    assert next(backlog) == answer


def test_new_buffered_packet_invalidates_full_prefix_coverage():
    inbox, broker = prepare_turn(["candidate"])
    ordinary = [message("execution_request", "ordinary") for _ in range(3)]
    answer = message("owner_answer", "candidate")
    inbox.pending.extend(ordinary)
    inbox.service(broker, lambda: None, transport_readable=False)
    inbox.pending.append(answer)
    inbox.service(broker, lambda: None, transport_readable=False)
    assert broker.received == [ordinary[0], answer, ordinary[1]]


def test_unknown_transport_state_still_seeks_formal_ready():
    inbox, broker = prepare_turn(["candidate"])
    ordinary = [message("execution_request", "ordinary") for _ in range(3)]
    answer = message("owner_answer", "candidate")
    inbox.pending.extend(ordinary)
    inbox.service(broker, lambda: None, transport_readable=False)
    backlog = iter([answer])
    inbox.service(broker, lambda: next(backlog, None))
    assert broker.received == [ordinary[0], answer, ordinary[1]]


def test_empty_transport_shortcut_preserves_fault_and_cancel_order():
    inbox, broker = prepare_turn(["candidate"])
    ordinary = [message("execution_request", "ordinary") for _ in range(3)]
    inbox.pending.extend(ordinary)
    inbox.service(broker, lambda: None, transport_readable=False)
    broker.fenced = True
    inbox.service(
        broker, lambda: pytest.fail("Fenced broker must retain FIFO"),
        transport_readable=False,
    )
    assert broker.received == ordinary[:2]
    assert inbox.last_sent_ready_sequence == 0

def outbound_control():
    import queue
    from unittest.mock import Mock

    control_type = importlib.import_module(package.__name__ + ".early_control").Control
    control = control_type.__new__(control_type)
    control.outbound = queue.SimpleQueue()
    control.hint_sender = Mock()
    control.identity = "worker"
    control.hello = {"session": "current"}
    return control


def test_ready_hint_follows_its_formal_socket_submission():
    from unittest.mock import Mock

    control = outbound_control()
    events = []
    ordinary = message("execution_request", "ordinary")[1]
    answer = message("owner_answer", "candidate")[1]
    control.outbound.put(ordinary)
    control.outbound.put(answer)
    dealer = Mock()
    dealer.send.side_effect = lambda event: events.append(("formal", event))
    control.hint_sender.notify.side_effect = lambda peer, session, tickets: events.append(
        ("hint", list(tickets))
    )
    control._send_outbound(dealer, lambda event: event)
    assert events == [
        ("formal", ordinary), ("formal", answer),
        ("hint", [("candidate", "generation")]),
    ]


@pytest.mark.parametrize("failed_submission", [0, 1])
def test_failed_formal_submission_does_not_publish_its_hint(failed_submission):
    from unittest.mock import Mock

    control = outbound_control()
    control.outbound.put(message("execution_request", "ordinary")[1])
    control.outbound.put(message("owner_answer", "candidate")[1])
    dealer = Mock()
    dealer.send.side_effect = (
        [RuntimeError("transport failed")]
        if failed_submission == 0 else [None, RuntimeError("transport failed")]
    )
    with pytest.raises(RuntimeError, match="transport failed"):
        control._send_outbound(dealer, lambda event: event)
    control.hint_sender.notify.assert_not_called()


def test_ready_hints_remain_bounded_and_only_cover_submitted_packets():
    from unittest.mock import Mock

    control = outbound_control()
    submitted = set()
    notifications = []
    for index in range(65):
        control.outbound.put(message("owner_answer", str(index))[1])
    dealer = Mock()
    dealer.send.side_effect = lambda event: submitted.add((event["key"], event["attempt"]))

    def notify(peer, session, tickets):
        assert set(tickets) <= submitted
        notifications.append(list(tickets))

    control.hint_sender.notify.side_effect = notify
    control._send_outbound(dealer, lambda event: event)
    assert [len(batch) for batch in notifications] == [32, 32, 1]
    assert sum(notifications, []) == [(str(index), "generation") for index in range(65)]


def test_no_hint_channel_keeps_formal_fifo_unchanged():
    from unittest.mock import Mock

    control = outbound_control()
    control.hint_sender = None
    events = [message("execution_request", "ordinary")[1],
              message("owner_answer", "candidate")[1]]
    for event in events:
        control.outbound.put(event)
    dealer = Mock()
    control._send_outbound(dealer, lambda event: event)
    assert [call.args[0] for call in dealer.send.call_args_list] == events
