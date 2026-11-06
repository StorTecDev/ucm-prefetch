"""Baseline Early initialization never patches native or platform schedulers."""

from unittest.mock import Mock

import pytest
from test_early_request_async import constructor_config

from ucm.integration.vllm.early_connector import (
    EarlyDirectConnector,
    EarlyLayerWiseConnector,
)
from ucm.integration.vllm.ucm_connector import KVConnectorRole


@pytest.mark.parametrize("cls", [EarlyDirectConnector, EarlyLayerWiseConnector])
@pytest.mark.parametrize("balance_enabled", [False, True])
def test_baseline_leaves_native_and_platform_methods_unchanged(
    constructor_config, cls, balance_enabled
):
    class Scheduler:
        def schedule(self):
            return "native"

    class BalanceScheduler(Scheduler):
        def schedule(self):
            if not self._balance_enabled:
                return super().schedule()
            return "balance"

    original = Scheduler.schedule, BalanceScheduler.schedule
    config, launch, _ = constructor_config
    launch.update(
        enable_early_prefetch=True, use_layerwise=cls is EarlyLayerWiseConnector
    )
    config.scheduler_config.get_scheduler_cls = Mock(return_value=BalanceScheduler)
    cls(config, KVConnectorRole.SCHEDULER)
    config.scheduler_config.get_scheduler_cls.assert_not_called()
    assert (Scheduler.schedule, BalanceScheduler.schedule) == original
    scheduler = BalanceScheduler()
    scheduler._balance_enabled = balance_enabled
    assert scheduler.schedule() == ("balance" if balance_enabled else "native")
