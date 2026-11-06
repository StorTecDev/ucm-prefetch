"""CPU constructor and store-boundary coverage for Early configuration overlays."""

import copy
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from test_early_request_async import constructor_config

from ucm.integration.vllm import ucm_connector
from ucm.integration.vllm.early_connector import (
    EarlyDirectConnector,
    EarlyLayerWiseConnector,
)
from ucm.integration.vllm.hma_connector import UCMFAWAConnector
from ucm.integration.vllm.ucm_connector import (
    KVConnectorRole,
    UCMCPConnector,
    UCMDirectConnector,
    UCMLayerWiseConnector,
)

NATIVE_CREATE_STORE = UCMDirectConnector._create_store
ROLES = [KVConnectorRole.SCHEDULER, KVConnectorRole.WORKER]
STORE_DEFAULTS = {
    "cache_full_block_load": True,
    "cache_full_block_l1": True,
    "cache_full_block_dump": True,
    "cache_dump_capacity_policy": "wait",
}


@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("cls", [EarlyDirectConnector, EarlyLayerWiseConnector])
@pytest.mark.parametrize("preset", [None, False])
def test_all_early_modes_get_scheduling_defaults_without_storage_preset(
    constructor_config, role, cls, preset
):
    config, launch, _ = constructor_config
    launch.update(
        enable_early_prefetch=True, use_layerwise=cls is EarlyLayerWiseConnector
    )
    if preset is not None:
        launch["early_full_block_defaults"] = preset
    before = copy.deepcopy(launch)
    connector = cls(config, role)
    assert connector.launch_config["early_prefetch_min_submit_reads"] == 16
    assert connector.launch_config["early_owner_max_jobs"] == 16
    assert (
        not STORE_DEFAULTS.keys()
        & connector.connector_configs[0]["ucm_connector_config"].keys()
    )
    assert launch == before
    if cls is EarlyLayerWiseConnector:
        assert not connector._block_dump_enabled
        assert not connector._block_dump_wait_capacity


@pytest.mark.parametrize("role", ROLES)
def test_scheduler_and_worker_pass_the_same_preset_to_native_factory(
    constructor_config, monkeypatch, role
):
    config, launch, _ = constructor_config
    launch.update(
        enable_early_prefetch=True, early_full_block_defaults=True, use_layerwise=True
    )
    config.parallel_config.data_parallel_rank = 0
    before = copy.deepcopy(launch)
    factory = Mock()
    monkeypatch.setattr(UCMDirectConnector, "_create_store", NATIVE_CREATE_STORE)
    monkeypatch.setattr(UCMDirectConnector, "_publish_block_size", Mock())
    monkeypatch.setattr(ucm_connector, "_scheduler_read_block_size", lambda: 8192)
    monkeypatch.setattr(
        ucm_connector.UcmConnectorFactoryV1, "create_connector", factory
    )
    connector = EarlyLayerWiseConnector(config, role)
    if role == KVConnectorRole.WORKER:
        layout = SimpleNamespace(
            pp_size=1,
            shard_size=4096,
            block_size=8192,
            tensor_size_list=[4096],
            base_ptrs=np.array([[1], [2]]),
            buffer_sizes=np.array([[4096], [4096]]),
        )
        connector._create_store(layout)
    name, store, module = factory.call_args.args
    assert name == "EarlyPipelineStore"
    assert module == "ucm.store.pipeline.early_connector"
    assert {key: store[key] for key in STORE_DEFAULTS} == STORE_DEFAULTS
    assert connector._block_dump_enabled
    assert connector._block_dump_wait_capacity
    assert connector.launch_config["early_prefetch_min_submit_reads"] == 16
    assert connector.launch_config["early_owner_max_jobs"] == 16
    assert "early_full_block_defaults" not in store
    assert "early_owner_max_jobs" not in store
    assert launch == before


@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("early", [None, False, True])
@pytest.mark.parametrize(
    "cls", [UCMDirectConnector, UCMLayerWiseConnector, UCMCPConnector, UCMFAWAConnector]
)
def test_native_and_other_connector_loaders_do_not_apply_early_defaults(
    constructor_config, role, cls, early
):
    config, launch, _ = constructor_config
    launch["early_full_block_defaults"] = True
    if early is not None:
        launch["enable_early_prefetch"] = early
    before = copy.deepcopy(launch)
    connector = object.__new__(cls)
    UCMDirectConnector.__init__(connector, config, role)
    assert connector.launch_config == before
    assert connector.connector_configs == before["ucm_connectors"]
    assert launch == before


@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_bad_preset_type_rejected_before_store_creation(
    constructor_config, role, value
):
    config, launch, _ = constructor_config
    launch.update(enable_early_prefetch=True, early_full_block_defaults=value)
    with pytest.raises(ValueError, match="early_full_block_defaults must be a boolean"):
        EarlyLayerWiseConnector(config, role)
    UCMDirectConnector._create_store.assert_not_called()


@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("mode", ["direct", "pp", "chunk"])
def test_preset_retains_layout_safety_checks(constructor_config, role, mode):
    config, launch, _ = constructor_config
    launch.update(
        enable_early_prefetch=True,
        early_full_block_defaults=True,
        use_layerwise=mode != "direct",
    )
    cls = EarlyDirectConnector if mode == "direct" else EarlyLayerWiseConnector
    if mode == "pp":
        config.parallel_config.pipeline_parallel_size = 2
    if role == KVConnectorRole.SCHEDULER and mode != "chunk":
        with pytest.raises(ValueError, match="cache_dump_capacity_policy='spill'"):
            cls(config, role)
        UCMDirectConnector._create_store.assert_not_called()
        return
    connector = cls(config, role)
    UCMDirectConnector._create_store.reset_mock()
    if mode == "chunk":
        connector.blocks_per_chunk = 2
    layout = SimpleNamespace(pp_size=2 if mode == "pp" else 1)
    with pytest.raises(ValueError, match="cache_dump_capacity_policy='spill'"):
        connector._create_store(layout if role == KVConnectorRole.WORKER else None)
    UCMDirectConnector._create_store.assert_not_called()


@pytest.mark.parametrize("role", ROLES)
def test_direct_can_explicitly_disable_dump_and_wait_together(constructor_config, role):
    config, launch, _ = constructor_config
    launch.update(enable_early_prefetch=True, early_full_block_defaults=True)
    store = launch["ucm_connectors"][0]["ucm_connector_config"]
    store.update(
        cache_full_block_dump=False,
        cache_dump_capacity_policy="spill",
        cache_dump_reserved_blocks=0,
    )
    before = copy.deepcopy(launch)
    connector = EarlyDirectConnector(config, role)
    if role == KVConnectorRole.WORKER:
        connector._create_store(SimpleNamespace(pp_size=1))
    assert (
        connector.connector_configs[0]["ucm_connector_config"]["cache_full_block_dump"]
        is False
    )
    assert (
        connector.connector_configs[0]["ucm_connector_config"][
            "cache_dump_reserved_blocks"
        ]
        == 0
    )
    assert launch == before


@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("source", ["inline", "yaml"])
def test_real_config_loader_reaches_early_normalization(
    constructor_config, monkeypatch, tmp_path, role, source
):
    import yaml

    from ucm.utils import Config

    config, launch, _ = constructor_config
    launch.update(
        enable_early_prefetch=True,
        early_full_block_defaults=True,
        use_layerwise=True,
    )
    before = copy.deepcopy(launch)
    if source == "yaml":
        path = tmp_path / "ucm.yaml"
        path.write_text(yaml.safe_dump(launch))
        extra = {"UCM_CONFIG_FILE": str(path)}
    else:
        extra = launch
    config.kv_transfer_config.kv_connector_extra_config = extra
    monkeypatch.setattr(ucm_connector, "Config", Config)
    connector = EarlyLayerWiseConnector(config, role)
    assert connector.launch_config["early_prefetch_min_submit_reads"] == 16
    assert connector.launch_config["early_owner_max_jobs"] == 16
    store = connector.connector_configs[0]["ucm_connector_config"]
    assert {key: store[key] for key in STORE_DEFAULTS} == STORE_DEFAULTS
    assert connector._block_dump_enabled
    assert launch == before
