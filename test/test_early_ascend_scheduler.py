"""Verify Ascend's disabled wrapper keeps its native admission delegation."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from ucm.integration.vllm.early_scheduling import _native_admission_scheduler


@pytest.fixture
def wrapper(monkeypatch):
    class Scheduler:
        def schedule(self, throttle_prefills=False):
            return ("native", throttle_prefills)

    class BalanceScheduler(Scheduler):
        def schedule(self, throttle_prefills=False):
            if not self._balance_enabled:
                return super().schedule(throttle_prefills)
            return ("balance", throttle_prefills)

    module = "vllm_ascend.patch.platform.patch_balance_schedule"
    Scheduler.__module__ = "vllm.v1.core.sched.scheduler"
    BalanceScheduler.__module__ = module
    platform = ModuleType(module)
    platform._balance_scheduling_enabled = lambda config: config.balance_enabled
    monkeypatch.setitem(sys.modules, module, platform)
    return Scheduler, BalanceScheduler


def test_disabled_wrapper_reaches_adapted_native_method(wrapper, monkeypatch):
    native, platform = wrapper
    original = platform.schedule
    target = _native_admission_scheduler(
        platform, SimpleNamespace(balance_enabled=False)
    )
    assert target is native
    monkeypatch.setattr(target, "schedule", lambda self, value=False: ("adapted", value))
    scheduler = platform()
    scheduler._balance_enabled = False
    assert scheduler.schedule(True) == ("adapted", True)
    assert platform.schedule is original


def test_enabled_balance_is_rejected_without_replacing_methods(wrapper):
    native, platform = wrapper
    original = native.schedule, platform.schedule
    with pytest.raises(ValueError, match="balance scheduling disabled"):
        _native_admission_scheduler(platform, SimpleNamespace(balance_enabled=True))
    assert (native.schedule, platform.schedule) == original


def test_missing_platform_configuration_is_rejected(wrapper):
    with pytest.raises(ValueError, match="balance scheduling disabled"):
        _native_admission_scheduler(wrapper[1], None)


def test_changed_delegation_is_rejected(wrapper, monkeypatch):
    def schedule(self, throttle_prefills=False):
        return ("changed", throttle_prefills)

    monkeypatch.setattr(wrapper[1], "schedule", schedule)
    with pytest.raises(ValueError, match="fast path"):
        _native_admission_scheduler(wrapper[1], SimpleNamespace(balance_enabled=False))


def test_unknown_native_base_is_rejected(wrapper, monkeypatch):
    monkeypatch.setattr(wrapper[0], "__module__", "custom_scheduler")
    with pytest.raises(ValueError, match="scheduler base"):
        _native_admission_scheduler(wrapper[1], SimpleNamespace(balance_enabled=False))


def test_native_scheduler_needs_no_platform_configuration(wrapper):
    assert _native_admission_scheduler(wrapper[0], None) is wrapper[0]
