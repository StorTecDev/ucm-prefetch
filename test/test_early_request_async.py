"""Direct request-async admission through the real connector constructors."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from ucm.integration.vllm import early_connector, early_scheduling, ucm_connector
from ucm.integration.vllm.early_connector import (
    EarlyDirectConnector,
    EarlyLayerWiseConnector,
)
from ucm.integration.vllm.hma_connector import UCMFAWAConnector
from ucm.integration.vllm.ucm_connector import (
    KVConnectorRole,
    UCMConnector,
    UCMCPConnector,
    UCMDirectConnector,
    UCMLayerWiseConnector,
)


class UnverifiedDirectConnector(UCMDirectConnector):
    pass


class UnverifiedEarlyConnector(EarlyDirectConnector):
    pass


@pytest.fixture
def constructor_config(monkeypatch):
    """Replace device/resource setup while retaining constructor admission."""
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            rank=0,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            data_parallel_index=0,
        ),
        cache_config=SimpleNamespace(block_size=16),
        scheduler_config=SimpleNamespace(),
        model_config=SimpleNamespace(
            is_deepseek_mla=True,
            dtype=torch.float16,
            hf_config=SimpleNamespace(),
            get_num_layers=lambda _: 2,
            get_num_kv_heads=lambda _: 1,
            get_head_size=lambda: 8,
        ),
        kv_transfer_config=SimpleNamespace(engine_id="unit-test"),
    )
    launch = {
        "use_layerwise": False,
        "ucm_connectors": [
            {
                "ucm_connector_name": "UcmPipelineStore",
                "ucm_connector_config": {"share_buffer_enable": False},
            }
        ],
    }

    def base_init(self, vllm_config, role, kv_cache_config=None):
        self._vllm_config = vllm_config
        self._role = role
        self._kv_cache_config = kv_cache_config

    monkeypatch.setattr(ucm_connector.KVConnectorBase_V1, "__init__", base_init)
    monkeypatch.setattr(
        ucm_connector, "Config", lambda _: SimpleNamespace(get_config=lambda: launch)
    )
    monkeypatch.setattr(
        ucm_connector, "current_platform", SimpleNamespace(is_cuda_alike=lambda: True)
    )
    monkeypatch.setattr(
        ucm_connector, "get_world_group", lambda: SimpleNamespace(rank=0, local_rank=0)
    )
    monkeypatch.setattr(ucm_connector, "get_current_device_id", lambda: 0)
    monkeypatch.setattr(ucm_connector, "RequestHasher", Mock())
    monkeypatch.setattr(UCMDirectConnector, "_make_other_rank_hashers", lambda *a: [])
    monkeypatch.setattr(
        UCMDirectConnector, "_bind_request_block_hasher", lambda *a: None
    )
    monkeypatch.setattr(UCMDirectConnector, "_create_store", Mock())
    monkeypatch.setattr(UCMFAWAConnector, "_create_store", Mock())
    monkeypatch.setattr(UCMConnector, "_setup_ucm_metrics", lambda *a: None)
    monkeypatch.setattr(early_connector, "EarlyScheduler", Mock())
    monkeypatch.setattr(early_scheduling, "install_scheduler_policy", Mock())
    monkeypatch.setattr(ucm_connector, "logger", Mock())
    monkeypatch.setenv("VLLM_CPU_AFFINITY", "0")
    monkeypatch.setenv("ENABLE_UCM_PATCH", "0")
    return config, launch, ucm_connector.logger


@pytest.mark.parametrize("role", [KVConnectorRole.SCHEDULER, KVConnectorRole.WORKER])
@pytest.mark.parametrize("early", [False, True])
@pytest.mark.parametrize("request_async", [None, False, True])
def test_factory_preserves_flags_and_enables_direct_async(
    constructor_config, role, early, request_async
):
    config, launch, logger = constructor_config
    launch["enable_early_prefetch"] = early
    if request_async is not None:
        launch["use_request_async_load"] = request_async
    wrapper = UCMConnector(config, role)
    expected = EarlyDirectConnector if early else UCMDirectConnector
    assert type(wrapper.connector) is expected
    assert wrapper.connector.use_request_async_load is bool(request_async)
    assert wrapper.connector._pending_load_tasks == {}
    assert wrapper.connector._pending_async_load_dispatches == {}
    logger.warning.assert_not_called()


@pytest.mark.parametrize(
    "cls",
    [
        UCMLayerWiseConnector,
        EarlyLayerWiseConnector,
        UCMCPConnector,
        UCMFAWAConnector,
        UnverifiedDirectConnector,
        UnverifiedEarlyConnector,
    ],
)
@pytest.mark.parametrize("configured", [False, True])
def test_unverified_concrete_types_keep_native_disable_and_warning(
    constructor_config, cls, configured
):
    config, launch, logger = constructor_config
    launch["use_request_async_load"] = configured
    connector = object.__new__(cls)
    # The shared constructor owns admission; specialized layouts need no device setup.
    UCMDirectConnector.__init__(connector, config, KVConnectorRole.SCHEDULER)
    assert not connector.use_request_async_load
    assert logger.warning.call_count == int(configured)
    if configured:
        assert logger.warning.call_args.args[-1] == cls.__name__
