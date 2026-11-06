"""Removed scheduling modes fail explicitly at the shared Early entry point."""

import pytest
from test_early_request_async import constructor_config

from ucm.integration.vllm.early_connector import (
    EarlyDirectConnector,
    EarlyLayerWiseConnector,
)
from ucm.integration.vllm.ucm_connector import KVConnectorRole, UCMDirectConnector

FLAGS = ["early_prefetch_request_first", "early_ready_priority"]
ROLES = [KVConnectorRole.SCHEDULER, KVConnectorRole.WORKER]


@pytest.mark.parametrize("cls", [EarlyDirectConnector, EarlyLayerWiseConnector])
@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("configured", [None, False])
def test_baseline_modes_default_false_for_every_early_connector(
    constructor_config, cls, role, configured
):
    config, launch, _ = constructor_config
    launch.update(
        enable_early_prefetch=True, use_layerwise=cls is EarlyLayerWiseConnector
    )
    if configured is not None:
        launch.update({name: configured for name in FLAGS})
    connector = cls(config, role)
    for name in FLAGS:
        assert connector.launch_config[name] is False


@pytest.mark.parametrize("cls", [EarlyDirectConnector, EarlyLayerWiseConnector])
@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("flag", FLAGS)
def test_removed_true_modes_fail_before_store_setup(
    constructor_config, cls, role, flag
):
    config, launch, _ = constructor_config
    launch.update(
        enable_early_prefetch=True, use_layerwise=cls is EarlyLayerWiseConnector
    )
    launch[flag] = True
    with pytest.raises(ValueError, match=flag + "=true is no longer supported"):
        cls(config, role)
    UCMDirectConnector._create_store.assert_not_called()
