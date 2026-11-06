"""PCP routing, shared-owner membership and native completion boundaries."""

import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_early_owner_protocol import Cluster, capability
from test_early_request_async import constructor_config
from ucm.integration.vllm.early_connector import EarlyCPConnector
from ucm.integration.vllm.early_consumer import Consumer, load_scope
from ucm.integration.vllm.early_topology import worker_count, worker_rank
from ucm.integration.vllm.ucm_connector import (
    KVConnectorRole,
    UCMConnector,
    UCMCPConnector,
    UCMDirectConnector,
    UCMLayerWiseConnector,
)


@pytest.mark.parametrize("role", [KVConnectorRole.SCHEDULER, KVConnectorRole.WORKER])
@pytest.mark.parametrize("early", [False, True])
def test_factory_preserves_native_pcp_and_constructs_one_early_store(
    constructor_config, monkeypatch, role, early
):
    import vllm.distributed as distributed

    config, launch, _ = constructor_config
    config.parallel_config.prefill_context_parallel_size = 16
    launch.update(use_layerwise=True, enable_early_prefetch=early)
    monkeypatch.setattr(
        distributed,
        "get_pcp_group",
        lambda: SimpleNamespace(world_size=16, rank_in_group=7),
    )
    monkeypatch.setattr(
        distributed,
        "get_dcp_group",
        lambda: SimpleNamespace(world_size=1, rank_in_group=0),
    )
    connector = UCMConnector(config, role).connector
    assert type(connector) is (EarlyCPConnector if early else UCMCPConnector)
    assert connector.block_size == 16 * 16
    assert connector.hash_block_size == 16
    assert worker_count(connector) == 16
    assert worker_rank(connector) == 7
    assert connector.use_layerwise
    assert not connector.use_request_async_load
    if role == KVConnectorRole.SCHEDULER:
        UCMDirectConnector._create_store.assert_called_once()
    else:
        UCMDirectConnector._create_store.assert_not_called()
    if early:
        assert (
            connector.connector_configs[0]["ucm_connector_name"] == "EarlyPipelineStore"
        )


@pytest.mark.parametrize("unsupported", ["dcp", "pp", "non_mla", "direct"])
def test_unsupported_pcp_modes_fail_before_store_creation(
    constructor_config, unsupported
):
    config, launch, _ = constructor_config
    config.parallel_config.prefill_context_parallel_size = 16
    launch.update(use_layerwise=True, enable_early_prefetch=True)
    if unsupported == "dcp":
        config.parallel_config.decode_context_parallel_size = 2
    elif unsupported == "pp":
        config.parallel_config.pipeline_parallel_size = 2
    elif unsupported == "non_mla":
        config.model_config.is_deepseek_mla = False
    else:
        launch["use_layerwise"] = False
    with pytest.raises(ValueError, match="Early PCP requires"):
        EarlyCPConnector(config, KVConnectorRole.SCHEDULER)
    UCMDirectConnector._create_store.assert_not_called()


@pytest.mark.parametrize("count", [0, 1, 7, 32])
def test_each_pcp_ticket_matches_native_load_and_preserves_global_metadata(
    monkeypatch, count
):
    blocks = [i.to_bytes(16, "little") for i in range(count)]
    original = SimpleNamespace(
        request_meta={
            "r": SimpleNamespace(
                load_block_ids=(blocks, [3, 4]), dump_block_ids=(blocks, [5, 6])
            )
        },
        early_owners={
            "r": {"key": "job", "attempt": "generation", "expected": tuple(blocks)}
        },
        early_keys={"r": "job"},
    )
    before = copy.deepcopy(original)
    observed = []
    monkeypatch.setattr(
        UCMLayerWiseConnector,
        "bind_connector_metadata",
        lambda self, metadata: observed.append(metadata),
    )
    for rank in range(16):
        connector = object.__new__(EarlyCPConnector)
        connector.current_rank, connector.cp_world_size = rank, 16
        connector._early = Mock()
        connector.bind_connector_metadata(original)
        local = observed[-1]
        assert tuple(local.request_meta["r"].load_block_ids[0]) == tuple(
            blocks[rank::16]
        )
        assert local.early_owners["r"]["expected"] == tuple(blocks[rank::16])
        assert local.request_meta["r"].dump_block_ids[0] == blocks[rank::16]
        connector._early.bind.assert_called_once_with(local)
    assert original == before


def test_pcp_tp_peers_are_unique_without_changing_native_tp_rank():
    peers = set()
    for cp in range(16):
        for tp in range(2):
            c = SimpleNamespace(
                tp_size=2, tp_rank=tp, cp_world_size=16, current_rank=cp
            )
            peers.add(worker_rank(c))
            assert worker_count(c) == 32
            assert c.tp_rank == tp
    assert peers == set(range(32))


def test_one_owner_prefetches_global_prefix_and_waits_for_every_pcp_consumer():
    cluster = Cluster(tp=16)
    job = cluster.job(count=32)
    grants = [
        (peer, message)
        for peer, message in cluster.out
        if message["op"] == "owner_reserve"
    ]
    assert len(grants) == 1
    assert grants[0][1]["blocks"] == job["blocks"]
    cluster.ready(job)
    for rank in range(15):
        cluster.message(job, "consumed", f"w0.{rank}")
    assert not any(message["op"] == "release_owner" for _, message in cluster.out)
    cluster.message(job, "consumed", "w0.15")
    assert job["retiring"]
    assert not any(message["op"] == "release_owner" for _, message in cluster.out)
    for rank in range(15):
        cluster.message(job, "quiesced", f"w0.{rank}")
    assert not any(message["op"] == "release_owner" for _, message in cluster.out)
    cluster.message(job, "quiesced", "w0.15")
    assert sum(message["op"] == "release_owner" for _, message in cluster.out) == 1


def test_pcp_consumer_releases_only_after_its_native_layers_complete():
    sent, poisoned = [], []
    consumer = Consumer("w0.3", sent.append, poisoned.append)
    block = b"x" * 16
    consumer.bind(
        {"r": {"key": "job", "attempt": "generation", "expected": (block,)}}, [0, 1]
    )
    for layer in range(2):
        context = load_scope.set(consumer.scope(["r"]))
        try:
            token = consumer.begin(block, [layer])
        finally:
            load_scope.reset(context)
        assert not any(message["op"] == "consumed" for message in sent)
        consumer.complete(token, layer, True)
    assert len([message for message in sent if message["op"] == "consumed"]) == 1
    assert not poisoned


def test_missing_pcp_peer_cannot_be_replaced_by_duplicate_tp_identity():
    cluster = Cluster(tp=16)
    cluster.b.peers.pop("w0.15")
    cluster.b.receive(
        "w0.0", {"op": "hello", "session": "worker", "capability": capability(0, 0)}
    )
    job = cluster.job(count=16)
    assert not job.get("running")
    assert not any(message["op"] == "owner_reserve" for _, message in cluster.out)


@pytest.mark.parametrize("replicated", [False, True])
def test_verified_pcp_owner_capability_has_unique_rank_and_shared_layout(
    monkeypatch, replicated
):
    import sys

    from test_early_owner_protocol import owner, package

    monkeypatch.setitem(
        sys.modules,
        package.__name__ + ".ucm_connector",
        SimpleNamespace(build_kv_hash_meta=lambda config, rank: str(rank)),
    )

    class Hasher:
        def __init__(self, meta):
            self.rank = int(meta)

        def __call__(self, value):
            return bytes([self.rank]) * 16

    native = dict(
        backing="native-object",
        nodes=120,
        reservable_nodes=100,
        shard_bytes=4,
        block_bytes=8,
    )
    store = SimpleNamespace(
        config=dict(
            share_buffer_enable=True, shard_size=4, block_size=8, tensor_size_list=[4]
        ),
        domain_info=lambda: native,
    )
    c = SimpleNamespace(
        store=store,
        launch_config={},
        is_mla=True,
        cp_world_size=16,
        current_rank=0,
        _early_cp_owner_supported=True,
        layer_ids=[0, 1],
        use_layerwise=True,
        unique_id="cohort",
        hash_block_size=4,
        block_size=64,
        blocks_per_chunk=1,
        request_hasher=Hasher("0"),
        tp_rank=0,
        tp_size=1,
        _vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_index=0),
            model_config=SimpleNamespace(max_model_len=4096),
        ),
    )
    if replicated:
        c.cp_world_size = 1
        c.cp_worker_world_size = 16
        c.block_size = c.hash_block_size
    capabilities = []
    for rank in range(16):
        if replicated:
            c.cp_worker_rank = rank
        else:
            c.current_rank = rank
        capabilities.append(owner.worker_capability(c))
    assert all(cap.get("ready") for cap in capabilities)
    assert {cap["rank"] for cap in capabilities} == set(range(16))
    assert len({cap["layout"] for cap in capabilities}) == 1
    c._early_cp_owner_supported = False
    assert not owner.worker_capability(c).get("ready", False)


@pytest.mark.parametrize("early", [False, True])
def test_replicated_pcp_loads_full_prefix_and_persists_one_copy(
    constructor_config, monkeypatch, early
):
    import vllm.distributed as distributed
    from ucm.integration.vllm import ucm_connector

    config, launch, _ = constructor_config
    config.parallel_config.prefill_context_parallel_size = 16
    config.cache_config.block_size = 128
    launch.update(
        use_layerwise=True,
        enable_early_prefetch=early,
        pcp_kv_cache_layout="replicated",
    )
    monkeypatch.setattr(
        ucm_connector.current_platform, "device_type", "npu", raising=False
    )
    rank = 0
    monkeypatch.setattr(
        distributed,
        "get_pcp_group",
        lambda: SimpleNamespace(world_size=16, rank_in_group=rank),
    )
    monkeypatch.setattr(
        distributed,
        "get_dcp_group",
        lambda: SimpleNamespace(world_size=1, rank_in_group=0),
    )
    # This prefix was previously advertised as 38,912 tokens while each
    # replica restored only 19 physical blocks (2,432 tokens).
    blocks = [i.to_bytes(16, "little") for i in range(304)]
    physical = list(range(1, 305))
    metadata = SimpleNamespace(
        request_meta={
            "r": SimpleNamespace(
                load_block_ids=(blocks, physical),
                dump_block_ids=(blocks, physical),
            )
        },
        early_owners={
            "r": {"key": "job", "attempt": "generation", "expected": tuple(blocks)}
        },
        early_keys={"r": "job"},
    )
    before = copy.deepcopy(metadata)
    observed = []
    monkeypatch.setattr(
        UCMLayerWiseConnector,
        "bind_connector_metadata",
        lambda self, value: observed.append(value),
    )
    peers, writers = set(), []
    for rank in range(16):
        connector = UCMConnector(config, KVConnectorRole.WORKER).connector
        assert connector.block_size == connector.hash_block_size == 128
        assert connector.cp_world_size == 1 and connector.current_rank == 0
        assert worker_count(connector) == 16
        peers.add(worker_rank(connector))
        if early:
            connector._early = Mock()
        connector.bind_connector_metadata(metadata)
        local = observed[-1]
        assert local.request_meta["r"].load_block_ids == (blocks, physical)
        if local.request_meta["r"].dump_block_ids[0]:
            writers.append(rank)
            assert local.request_meta["r"].dump_block_ids == (blocks, physical)
        else:
            assert local.request_meta["r"].dump_block_ids == ([], [])
        if early:
            assert local.early_owners["r"]["expected"] == tuple(blocks)
            connector._early.bind.assert_called_once()
            bound = connector._early.bind.call_args.args[0]
            assert bound.early_owners == local.early_owners
            assert bound.request_meta["r"].load_block_ids == (blocks, physical)
    assert peers == set(range(16))
    assert writers == [0]
    assert metadata == before


@pytest.mark.parametrize("unsupported", ["layout", "dcp", "non_mla", "direct", "cuda"])
def test_replicated_pcp_rejects_unverified_modes_before_store(
    constructor_config, monkeypatch, unsupported
):
    from ucm.integration.vllm import ucm_connector

    config, launch, _ = constructor_config
    config.parallel_config.prefill_context_parallel_size = 16
    launch.update(use_layerwise=True, pcp_kv_cache_layout="replicated")
    monkeypatch.setattr(
        ucm_connector.current_platform, "device_type", "npu", raising=False
    )
    if unsupported == "layout":
        launch["pcp_kv_cache_layout"] = "unknown"
    elif unsupported == "dcp":
        config.parallel_config.decode_context_parallel_size = 2
    elif unsupported == "non_mla":
        config.model_config.is_deepseek_mla = False
    elif unsupported == "direct":
        launch["use_layerwise"] = False
    else:
        monkeypatch.setattr(ucm_connector.current_platform, "device_type", "cuda")
    with pytest.raises(ValueError, match="pcp_kv_cache_layout|Replicated PCP cache"):
        UCMCPConnector(config, KVConnectorRole.SCHEDULER)
    UCMDirectConnector._create_store.assert_not_called()
