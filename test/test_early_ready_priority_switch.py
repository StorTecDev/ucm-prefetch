"""Opt-in READY ordering through the native vLLM scheduler and allocator."""

import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_early_native_scheduler import attach, requests_of_sizes, scheduler_factory
from test_early_ready_reuse import connector
from test_early_request_async import constructor_config
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.core.sched.scheduler import Scheduler

from ucm.integration.vllm import early_scheduling
from ucm.integration.vllm.early_connector import (
    EarlyDirectConnector,
    EarlyLayerWiseConnector,
)
from ucm.integration.vllm.early_prefetch import EarlyScheduler
from ucm.integration.vllm.early_scheduling import (
    ReadyPriority,
    install_scheduler_policy,
)
from ucm.integration.vllm.ucm_connector import KVConnectorRole


@pytest.mark.parametrize("config", [{}, {"early_ready_priority": False}])
def test_native_mode_skips_scheduler_class_adaptation(config, monkeypatch):
    scheduler_config = Mock()
    patch = Mock()
    monkeypatch.setattr(early_scheduling, "patch_scheduler_class", patch)
    install_scheduler_policy(scheduler_config, config)
    assert not ReadyPriority(config).enabled
    scheduler_config.get_scheduler_cls.assert_not_called()
    patch.assert_not_called()


@pytest.mark.parametrize("value", [None, 0, 1, "false", "true"])
def test_invalid_switch_fails_before_scheduler_resources_start(value):
    control = Mock()
    connector = SimpleNamespace(launch_config={"early_ready_priority": value})
    with pytest.raises(ValueError, match="early_ready_priority must be a boolean"):
        EarlyScheduler(connector, control)
    control.assert_not_called()


@pytest.mark.parametrize(
    "name", ["early_ready_max_candidates", "early_ready_max_bypasses"]
)
def test_disabled_policy_still_validates_existing_limits(name):
    with pytest.raises(ValueError, match=name + " must be a positive integer"):
        install_scheduler_policy(Mock(), {"early_ready_priority": False, name: 0})


@pytest.mark.parametrize("enabled", [False, True])
def test_enabled_policy_retains_native_scheduler_requirement(enabled):
    custom = SimpleNamespace(get_scheduler_cls=lambda: object)
    if enabled:
        with pytest.raises(ValueError, match="requires the native vLLM scheduler"):
            install_scheduler_policy(custom, {"early_ready_priority": enabled})
    else:
        install_scheduler_policy(custom, {"early_ready_priority": enabled})


@pytest.mark.parametrize("cls", [EarlyDirectConnector, EarlyLayerWiseConnector])
@pytest.mark.parametrize("enabled", [None, False, True])
def test_connector_constructor_applies_root_switch(
    constructor_config, monkeypatch, cls, enabled
):
    config, launch, _ = constructor_config
    if enabled is not None:
        launch["early_ready_priority"] = enabled
    config.scheduler_config.get_scheduler_cls = lambda: Scheduler
    patch = Mock()
    monkeypatch.setattr(early_scheduling, "patch_scheduler_class", patch)
    monkeypatch.setattr(
        early_scheduling, "install_scheduler_policy", install_scheduler_policy
    )
    cls(config, KVConnectorRole.SCHEDULER)
    assert patch.call_count == int(enabled is True)


def test_invalid_constructor_switch_fails_before_store_creation(
    constructor_config, monkeypatch
):
    config, launch, _ = constructor_config
    launch["early_ready_priority"] = "false"
    store = Mock()
    monkeypatch.setattr(EarlyDirectConnector, "_create_store", store)
    monkeypatch.setattr(
        early_scheduling, "install_scheduler_policy", install_scheduler_policy
    )
    with pytest.raises(ValueError, match="early_ready_priority must be a boolean"):
        EarlyDirectConnector(config, KVConnectorRole.SCHEDULER)
    store.assert_not_called()


@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize(
    "config", [{}, {"early_ready_priority": False}, {"early_ready_priority": True}]
)
def test_ready_order_switches_between_native_queue_and_uncached_work(
    connector, scheduler_factory, async_scheduling, config
):
    scheduler = scheduler_factory(
        max_num_batched_tokens=512, async_scheduling=async_scheduling
    )
    requests = requests_of_sizes(8192, 8192)
    adapter = attach(connector, scheduler, requests, ("READY", "READY"), (48, 60))
    policy = connector._early.scheduling = ReadyPriority(config)
    output = scheduler.schedule()
    selected = "1" if policy.enabled else "0"
    assert output.num_scheduled_tokens == {selected: 512}
    assert adapter.get_num_new_matched_tokens.call_count == 1
    assert adapter.get_num_new_matched_tokens.call_args.args[0].request_id == selected
    assert output.total_num_scheduled_tokens == 512
    if not policy.enabled:
        assert policy.round == 0
        assert not policy.bypasses and not policy.last_rejection
    connector.store.store_.LookupOnPrefix.assert_not_called()


@pytest.mark.parametrize("enabled", [False, True])
def test_native_request_priority_is_only_overridden_when_enabled(
    connector, scheduler_factory, enabled
):
    scheduler = scheduler_factory(max_num_batched_tokens=2048)
    scheduler.policy = SchedulingPolicy.PRIORITY
    scheduler.waiting = create_request_queue(scheduler.policy)
    scheduler.skipped_waiting = create_request_queue(scheduler.policy)
    requests = requests_of_sizes(8192, 8192)
    requests[0].priority, requests[1].priority = -10, 10
    attach(
        connector,
        scheduler,
        requests,
        ("FAILED", "READY"),
        (0, 48),
        ready_priority=enabled,
    )
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"1" if enabled else "0": 2048}


@pytest.mark.parametrize("enabled", [False, True])
def test_native_hbm_refusal_stops_pass_only_when_priority_is_disabled(
    connector, scheduler_factory, enabled
):
    scheduler = scheduler_factory(num_blocks=16)
    requests = requests_of_sizes(8192, 1024)
    adapter = attach(
        connector,
        scheduler,
        requests,
        ("READY", "READY"),
        (60, 4),
        ready_priority=enabled,
    )
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == ({"1": 512} if enabled else {})
    attempted = [
        call.args[0].request_id
        for call in adapter.get_num_new_matched_tokens.call_args_list
    ]
    assert attempted == (["0", "1"] if enabled else ["0"])
    assert scheduler.waiting.peek_request() is requests[0]


@pytest.mark.parametrize("enabled", [False, True])
def test_native_running_limit_and_pending_gate_apply_in_both_modes(
    connector, scheduler_factory, enabled
):
    scheduler = scheduler_factory(max_num_seqs=1)
    requests = requests_of_sizes(8192, 8192, 8192)
    attach(
        connector,
        scheduler,
        requests,
        ("PENDING", "READY", "READY"),
        (0, 48, 48),
        ready_priority=enabled,
    )
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"1": 2048}
    assert len(scheduler.running) == 1
    assert connector._early.jobs["0"].state == "PENDING"
    connector._early.control.send.assert_not_called()


def test_instances_share_patch_but_keep_independent_admission_modes(
    connector, scheduler_factory
):
    native_connector = copy.copy(connector)
    native_connector.requests_meta = {}
    native_connector._early = copy.copy(connector._early)
    native_connector._early.connector = native_connector
    native_connector._early.jobs, native_connector._early.by_key = {}, {}
    priority_scheduler = scheduler_factory(max_num_batched_tokens=2048)
    native_scheduler = scheduler_factory(max_num_batched_tokens=2048)
    attach(
        connector,
        priority_scheduler,
        requests_of_sizes(8192, 8192),
        ("FAILED", "READY"),
        (0, 48),
        ready_priority=True,
    )
    attach(
        native_connector,
        native_scheduler,
        requests_of_sizes(8192, 8192),
        ("FAILED", "READY"),
        (0, 48),
        ready_priority=False,
    )
    assert native_scheduler.schedule().num_scheduled_tokens == {"0": 2048}
    assert priority_scheduler.schedule().num_scheduled_tokens == {"1": 2048}
