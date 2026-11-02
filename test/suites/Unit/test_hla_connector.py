"""CPU-only checks for HLA decisions and transfer metadata.

Like test_kv_cache_layout.py, load production methods via AST to avoid
requiring torch, vLLM or NPU libraries. These tests do not emulate kernels
or establish the safety of every model-selection/state-copy configuration.
"""

import ast
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, List, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
HLA_PATH = ROOT / "ucm/integration/vllm/hla_connector.py"


class KVCacheSpec(SimpleNamespace):
    pass


class FullAttentionSpec(KVCacheSpec):
    pass


class MLAAttentionSpec(FullAttentionSpec):
    pass


class MambaSpec(KVCacheSpec):
    pass


class UniformTypeKVCacheSpecs(KVCacheSpec):
    pass


class FakeTensor:
    def __init__(self, ptr, shape, strides, element_size=2):
        self.ptr = ptr
        self.shape = shape
        self.strides = strides
        self.item_size = element_size

    def numel(self):
        return math.prod(self.shape)

    def dim(self):
        return len(self.shape)

    def stride(self, dim):
        return self.strides[dim]

    def element_size(self):
        return self.item_size

    def data_ptr(self):
        return self.ptr


@pytest.fixture
def hla():
    functions = {
        "_kv_cache_tensor_layers",
        "layer_name_to_kv_cache_spec",
        "block_size_from_kv_cache_spec",
        "is_mamba_align_kv_cache_spec",
        "participates_in_prefix_caching",
        "extend_non_null",
    }
    methods = {
        "KVCacheGroupManager": {
            "__init__",
            "num_groups",
            "compute_block_hashes",
            "compute_all_group_block_ids",
        },
        "HybridLinearAttentionLayout": {
            "_build_layout",
            "_map_layers_to_slots",
            "_layer_components",
            "_build_layer_anchors",
            "_finalize_layer_arrays",
            "extract_layer_addrs",
            "extract_block_addrs",
            "direct_store_config",
        },
        "UCMHybridLinearAttentionConnector": {
            "supports_kv_cache_layout",
            "_scope_blocks",
            "_append_mamba_align_state_block",
            "_generate_hla_dispatch_meta",
            "build_connector_meta",
            "_reset_transfer_state",
            "_prepare_load_plan",
            "_ensure_dump_plan",
            "_mark_load_failed",
        },
        "UCMHybridLinearAttentionLayerWiseConnector": {
            "_MODULE_TIER_ONLY_ARCHITECTURES",
            "layerwise_engine_config_error",
            "register_kv_caches",
            "wait_for_save",
            "start_load_kv",
            "wait_for_layer_load",
            "save_kv_layer",
            "_submit_layer_load",
            "_wait_layer_load_pair",
            "_record_load_bytes",
            "_submit_layer_dump",
            "_submit_commit_dump",
            "_wait_dump_tasks",
        },
    }
    nodes = []
    for node in ast.parse(HLA_PATH.read_text(encoding="utf-8-sig")).body:
        if getattr(node, "name", None) in functions | {
            "GroupInfo",
            "HLARequestMeta",
            "HLARequestDispatchMeta",
        }:
            nodes.append(node)
        elif isinstance(node, ast.ClassDef) and node.name in methods:
            node.bases = (
                [ast.Name(id="UCMHybridLinearAttentionConnector", ctx=ast.Load())]
                if node.name == "UCMHybridLinearAttentionLayerWiseConnector"
                else []
            )
            available = {
                item.name for item in node.body if isinstance(item, ast.FunctionDef)
            } | {
                target.id
                for item in node.body
                if isinstance(item, ast.Assign)
                for target in item.targets
                if isinstance(target, ast.Name)
            }
            assert methods[node.name] <= available, (
                node.name,
                methods[node.name] - available,
            )
            node.body = [
                item
                for item in node.body
                if getattr(item, "name", None) in methods[node.name]
                or (
                    isinstance(item, ast.Assign)
                    and any(
                        isinstance(target, ast.Name) and target.id in methods[node.name]
                        for target in item.targets
                    )
                )
            ]
            nodes.append(node)
    connector_path = ROOT / "ucm/integration/vllm/ucm_connector.py"
    base_nodes = [
        node
        for node in ast.parse(connector_path.read_text(encoding="utf-8-sig")).body
        if isinstance(node, ast.ClassDef)
        and node.name
        in {
            "KVCacheSegment",
            "RequestMeta",
            "RequestDispatchMeta",
            "UCMConnectorMetadata",
            "PendingDumpTask",
        }
    ]
    for node in base_nodes:
        if node.name == "UCMConnectorMetadata":
            node.bases = []
    nodes = base_nodes + nodes
    namespace = {
        "__name__": __name__,
        "math": math,
        "np": np,
        "defaultdict": defaultdict,
        "dataclass": dataclass,
        "field": field,
        "Callable": Callable,
        "List": List,
        "Optional": Optional,
        "KVCacheSpec": KVCacheSpec,
        "FullAttentionSpec": FullAttentionSpec,
        "MLAAttentionSpec": MLAAttentionSpec,
        "MambaSpec": MambaSpec,
        "UniformTypeKVCacheSpecs": UniformTypeKVCacheSpecs,
        "torch": SimpleNamespace(Tensor=FakeTensor),
        "logger": MagicMock(),
        "ucmmetrics": MagicMock(),
        "_short_list": lambda values, limit=12: values[:limit],
        "current_platform": SimpleNamespace(
            device_type="npu", is_cuda_alike=lambda: False
        ),
    }
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future, *nodes], type_ignores=[])
    )
    exec(compile(module, str(HLA_PATH), "exec"), namespace)
    return SimpleNamespace(**namespace)


def group(spec, *names):
    return SimpleNamespace(kv_cache_spec=spec, layer_names=list(names))


@pytest.mark.parametrize(
    "fields,expected",
    [
        ({"layers": ["new"]}, ["new"]),
        ({"shared_by": ["old"]}, ["old"]),
        ({"layers": ["new"], "shared_by": ["old"]}, ["new"]),
        ({"layers": [], "shared_by": ["old"]}, []),
        ({"layers": None, "shared_by": ["old"]}, ["old"]),
        ({"shared_by": None}, []),
        ({"layers": None, "shared_by": None}, []),
        ({"layers": ("a", "b")}, ("a", "b")),
        ({}, []),
    ],
)
def test_tensor_layer_names_preserve_values_and_handle_none(hla, fields, expected):
    result = hla._kv_cache_tensor_layers(SimpleNamespace(**fields))
    assert result == expected
    assert isinstance(result, type(expected))


@pytest.mark.parametrize(
    "fields,expected",
    [
        ({}, True),
        ({"prefix_cacheable": False}, False),
        ({"participates_in_prefix_caching": False}, False),
        ({"prefix_cacheable": True, "participates_in_prefix_caching": False}, False),
        ({"prefix_cacheable": False, "participates_in_prefix_caching": True}, False),
        ({"prefix_cacheable": True, "participates_in_prefix_caching": True}, True),
    ],
)
def test_prefix_eligibility_checks_both_descriptor_versions(hla, fields, expected):
    assert hla.participates_in_prefix_caching(KVCacheSpec(**fields)) is expected


def test_nested_uniform_prefix_eligibility(hla):
    tail = KVCacheSpec(prefix_cacheable=False)
    nested = UniformTypeKVCacheSpecs(kv_cache_specs={"tail": tail})
    outer = UniformTypeKVCacheSpecs(
        kv_cache_specs={"attention": FullAttentionSpec(), "nested": nested}
    )
    assert not hla.participates_in_prefix_caching(outer)
    tail.prefix_cacheable = True
    assert hla.participates_in_prefix_caching(outer)


def test_tail_is_excluded_from_hashes_and_lcm_without_renumbering(hla):
    config = SimpleNamespace(
        kv_cache_groups=[
            group(FullAttentionSpec(block_size=4), "mla"),
            group(KVCacheSpec(block_size=7, prefix_cacheable=False), "tail"),
            group(MambaSpec(block_size=6, mamba_cache_mode="align"), "gdn"),
        ]
    )
    hasher = MagicMock(return_value=b"seed")
    hash_request = MagicMock(return_value=[b"a", b"b", b"c"])
    hasher.make_request_block_hasher.return_value = hash_request
    manager = hla.KVCacheGroupManager(config, hasher, b"base")
    assert manager.lcm_block_size == 12
    assert [g.group_id for g in manager.full_attn_groups] == [0]
    assert [g.group_id for g in manager.non_prefix_groups] == [1]
    assert [g.group_id for g in manager.state_groups] == [2]
    assert manager.compute_all_group_block_ids(
        SimpleNamespace(all_token_ids=list(range(12)))
    ) == [[b"a", b"b", b"c"], [], [b"", b""]]
    hash_request.assert_called_once()


@pytest.mark.parametrize(
    "allocation_blocks,expected_stride", [(1, 64), (2, 32), (4, 16)]
)
def test_allocation_block_count_controls_component_span(
    hla, allocation_blocks, expected_stride
):
    layout = hla.HybridLinearAttentionLayout()
    layout.num_blocks = 2
    tensor = FakeTensor(1000, (4, 2, 4), (8, 4, 1))
    (segment,) = layout._layer_components("mla", tensor, allocation_blocks)
    assert segment.copy_size == segment.block_stride == expected_stride
    assert segment.buffer_size == layout.num_blocks * expected_stride


def hybrid_config():
    return SimpleNamespace(
        kv_cache_tensors=[SimpleNamespace(layers=["mla", "state"])],
        kv_cache_groups=[
            group(MLAAttentionSpec(), "mla"),
            group(MambaSpec(mamba_cache_mode="align"), "state"),
        ],
    )


@pytest.mark.parametrize(
    "device,expected", [("npu", True), ("cuda", True), ("cpu", False)]
)
def test_hybrid_detection_preserves_platform_guard(hla, device, expected):
    hla.current_platform.device_type = device
    hla.current_platform.is_cuda_alike = lambda: device == "cuda"
    assert (
        hla.UCMHybridLinearAttentionConnector.supports_kv_cache_layout(hybrid_config())
        is expected
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "no_config",
        "no_allocations",
        "no_attention",
        "no_state",
        "non_aligned_state",
        "unknown_layers",
    ],
)
def test_hybrid_detection_requires_allocated_attention_and_aligned_state(hla, mutation):
    config = hybrid_config()
    if mutation == "no_config":
        config = None
    elif mutation == "no_allocations":
        config.kv_cache_tensors = []
    elif mutation == "no_attention":
        config.kv_cache_groups = config.kv_cache_groups[1:]
    elif mutation == "no_state":
        config.kv_cache_groups = config.kv_cache_groups[:1]
    elif mutation == "non_aligned_state":
        config.kv_cache_groups[1].kv_cache_spec.mamba_cache_mode = "all"
    else:
        config.kv_cache_tensors[0].layers = ["unallocated"]
    assert not hla.UCMHybridLinearAttentionConnector.supports_kv_cache_layout(config)


def test_hybrid_detection_accepts_separate_allocations_and_uniform_specs(hla):
    config = hybrid_config()
    config.kv_cache_tensors = [
        SimpleNamespace(layers=[name]) for name in ["mla", "state"]
    ]
    config.kv_cache_groups[0].kv_cache_spec = UniformTypeKVCacheSpecs(
        kv_cache_specs={"mla": MLAAttentionSpec()}
    )
    assert hla.UCMHybridLinearAttentionConnector.supports_kv_cache_layout(config)


@pytest.mark.parametrize(
    "architecture",
    [
        "Glm5NextForConditionalGeneration",
        "KimiK3ForConditionalGeneration",
        "OtherHybridForConditionalGeneration",
    ],
)
@pytest.mark.parametrize(
    "mode,graph",
    [
        ("NONE", "NONE"),
        ("NONE", "FULL_DECODE_ONLY"),
        ("NONE", "FULL"),
        ("VLLM_COMPILE", "NONE"),
    ],
)
def test_layerwise_engine_guard_preserves_required_model_hooks(
    hla, architecture, mode, graph
):
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=[architecture])
        ),
        compilation_config=SimpleNamespace(
            mode=SimpleNamespace(name=mode), cudagraph_mode=SimpleNamespace(name=graph)
        ),
    )
    error = (
        hla.UCMHybridLinearAttentionLayerWiseConnector.layerwise_engine_config_error(
            config
        )
    )
    requires_hooks = architecture in {
        "Glm5NextForConditionalGeneration",
        "KimiK3ForConditionalGeneration",
    }
    if requires_hooks and (mode != "NONE" or graph == "FULL"):
        assert architecture in error and "requires mode=NONE" in error
    else:
        assert error is None


def test_empty_rope_is_skipped_before_contiguity_checks(hla):
    layout = hla.HybridLinearAttentionLayout()
    layout.num_blocks = 2
    key = FakeTensor(1000, (4, 2, 4), (8, 4, 1))
    rope = FakeTensor(2000, (4, 2, 0), (8, 4, 1))
    (segment,) = layout._layer_components("mla", (key, rope), 2)
    assert (
        segment.ptr,
        segment.copy_size,
        segment.block_stride,
        segment.buffer_size,
    ) == (1000, 32, 32, 64)


def test_padded_state_uses_whole_page_and_registered_span(hla):
    layout = hla.HybridLinearAttentionLayout()
    layout.num_blocks = 2
    (segment,) = layout._layer_components(
        "state", FakeTensor(1000, (2, 2, 3), (16, 3, 1)), 2
    )
    assert (segment.copy_size, segment.block_stride, segment.buffer_size) == (
        32,
        32,
        64,
    )
    assert segment.block_stride + segment.copy_size <= segment.buffer_size


@pytest.mark.parametrize(
    "shape,strides,allocation_blocks,message",
    [
        ((2, 2, 3), (16, 4, 1), 2, "not a dense page"),
        ((2, 2, 3), (4, 3, 1), 2, "Overlapping KV blocks"),
        ((3, 2, 3), (8, 3, 1), 2, "do not divide"),
    ],
)
def test_invalid_component_geometry_is_rejected(
    hla, shape, strides, allocation_blocks, message
):
    layout = hla.HybridLinearAttentionLayout()
    layout.num_blocks = 2
    with pytest.raises(ValueError, match=message):
        layout._layer_components(
            "mla", FakeTensor(1000, shape, strides), allocation_blocks
        )


@pytest.fixture
def page_layout(hla):
    def build(small_first=False):
        main = SimpleNamespace(layers=["mla0", "state0"], size=32, block_stride=16)
        small = SimpleNamespace(layers=["index0"], size=16, block_stride=8)
        last = SimpleNamespace(
            layers=["mla1", "state1", "tail"], size=32, block_stride=16
        )
        layout = hla.HybridLinearAttentionLayout()
        layout.num_blocks = 2
        layout.kv_cache_config = SimpleNamespace(
            kv_cache_groups=[
                group(FullAttentionSpec(), "mla0", "index0", "mla1"),
                group(KVCacheSpec(prefix_cacheable=False), "tail"),
                group(MambaSpec(mamba_cache_mode="align"), "state0", "state1"),
            ],
            kv_cache_tensors=(
                [small, main, last] if small_first else [main, small, last]
            ),
        )
        tensors = {
            name: FakeTensor(ptr, (2, 2), (stride, 1))
            for name, ptr, stride in [
                ("mla0", 1000, 8),
                ("index0", 2000, 4),
                ("mla1", 3000, 8),
                ("state0", 1004, 8),
                ("state1", 3004, 8),
                ("tail", 9000, 8),
            ]
        }
        layout._build_layout(tensors)
        layout.layer_name_to_id = dict(zip(tensors, range(len(tensors))))
        return layout

    return build


@pytest.mark.parametrize("small_first", [False, True])
def test_page_types_keep_group_ownership_and_null_columns(page_layout, small_first):
    layout = page_layout(small_first)
    assert layout.num_slots == 2 and layout.width == 2
    assert layout.page_types == ([8, 16] if small_first else [16, 8])
    assert "tail" not in layout.layer_anchors
    assert layout.group_pairs == {0: {0, 1}, 2: {0, 1}}
    assert layout.direct_group_bytes == {0: 40, 2: 32}
    expected = (
        [[2008, 1016, 0, 3016], [0, 1016, 0, 3016]]
        if small_first
        else [[1016, 2008, 3016, 0], [1016, 0, 3016, 0]]
    )
    addresses = layout.extract_block_addrs([1, 1], [0, 2])
    assert addresses.tolist() == expected
    assert layout.extract_layer_addrs("mla0", [1]).tolist() == [expected[0][:2]]
    assert layout.extract_layer_addrs("state1", [1]).tolist() == [expected[1][2:]]
    assert layout.direct_store_config() == (layout.page_types * 2, 48, 48)
    assert addresses.flags.c_contiguous
    for row in addresses:
        for address, size in zip(row, layout.page_types * 2):
            if address:
                assert any(
                    base <= address and address + size <= base + span
                    for base, span in zip(layout.base_ptrs, layout.buffer_sizes)
                )


def test_group_address_validation_and_empty_batch(page_layout):
    layout = page_layout()
    with pytest.raises(ValueError, match="lengths differ"):
        layout.extract_block_addrs([1], [])
    assert not layout.extract_block_addrs([1], [5]).any()
    with pytest.raises(ValueError, match="no per-layer anchors"):
        layout.extract_layer_addrs("tail", [1])
    empty = layout.extract_block_addrs([], [])
    assert empty.shape == (0, 4) and empty.flags.c_contiguous


def test_group_packed_descriptors_respect_family_offsets(page_layout):
    layout = page_layout()
    layout.kv_cache_config.kv_cache_groups = [
        group(FullAttentionSpec(), "mla0", "mla1"),
        group(MambaSpec(mamba_cache_mode="align"), "state0"),
    ]
    layout.kv_cache_config.kv_cache_tensors = [
        SimpleNamespace(
            layers=["mla0", "state0"], offset=128, layer_stride=64, block_stride=16
        ),
        SimpleNamespace(layers=["mla1"], offset=256, layer_stride=64, block_stride=16),
    ]
    layout._map_layers_to_slots()
    assert layout.layer_name_to_slot == {"mla0": 0, "state0": 1, "mla1": 2}
    layout.kv_cache_config.kv_cache_tensors[1].offset = 257
    with pytest.raises(ValueError, match="not aligned"):
        layout._map_layers_to_slots()


def transfer_connector(hla):
    connector = hla.UCMHybridLinearAttentionLayerWiseConnector()
    connector.load_tasks = defaultdict(dict)
    connector._load_bytes_by_pair = defaultdict(int)
    connector.request_data = []
    connector._failure_req_ids = set()
    connector._submitted_positions = set()
    connector._waited_layers = set()
    connector._dump_tasks = []
    connector._dumped_layers = set()
    connector._dumped_group_slots = set()
    connector._dump_plan = None
    connector._submitted_group_slots = set()
    connector._waited_group_slots = set()
    connector.is_save = False
    connector._save_bytes_total = 0
    connector._reset_transfer_state()
    connector._invalid_block_ids = set()
    connector._connector_worker_meta = SimpleNamespace(mark_failed=MagicMock())
    connector.tp_rank, connector.tp_size, connector.is_mla = 0, 1, True
    connector.store = object()
    connector.enable_event_sync = False
    connector._get_dump_event_handle = lambda: None
    connector.device = SimpleNamespace(synchronize=MagicMock())
    return connector


@pytest.fixture
def save_connector(hla, page_layout):
    connector = transfer_connector(hla)
    layout = page_layout()
    connector.kv_cache_layout = layout
    connector.layer_order = ["mla0", "index0", "state0", "mla1", "state1"]
    connector.layer_pos = {
        name: index for index, name in enumerate(connector.layer_order)
    }
    connector.layer_to_group = layout.layer_name_to_group
    connector.layer_to_slot = layout.layer_name_to_slot
    connector._register_common = lambda _: (layout, None, None)
    connector._create_store_bound = MagicMock()
    layout.tensor_size_list = layout.page_types
    connector.register_kv_caches({})
    metadata = hla.UCMConnectorMetadata(
        request_meta={
            "req": SimpleNamespace(
                dump_block_ids=([b"attention", b"state"], [1, 1]),
                dump_group_ids=[0, 2],
                dump_full_attn_count=1,
                load_block_ids=([], []),
            )
        }
    )
    connector._connector_metadata = metadata
    connector._get_connector_metadata = lambda: metadata
    connector._rank_consistency = SimpleNamespace(
        submit_dump=MagicMock(side_effect=range(20)),
        wait_dump=MagicMock(),
        finish_dump=MagicMock(),
    )
    return connector


def test_max_pair_is_deferred_and_partner_hooks_are_deduplicated(save_connector):
    connector = save_connector
    assert connector._deferred_dump_layers == {"mla1", "state1"}
    for name in connector.layer_order:
        connector.save_kv_layer(name, None, None)
    calls = connector._rank_consistency.submit_dump.call_args_list
    assert len(calls) == 2
    assert [call.args[2:4] for call in calls] == [
        ([b"attention"], [0]),
        ([b"state"], [0]),
    ]
    assert [call.args[4].tolist() for call in calls] == [[[1016, 2008]], [[1016, 0]]]


@pytest.mark.parametrize("hooked", [[], ["mla0", "index0"]])
def test_post_forward_backfills_pairs_before_commit(save_connector, hooked):
    connector = save_connector
    events = []
    connector._rank_consistency.submit_dump.side_effect = lambda *args: events.append(
        ("submit", args[2], args[3], args[4].tolist())
    ) or len(events)
    connector._rank_consistency.wait_dump.side_effect = lambda task: events.append(
        ("wait", task)
    )
    connector._rank_consistency.finish_dump.side_effect = (
        lambda requests: events.append(("finish", requests))
    )
    for name in hooked:
        connector.save_kv_layer(name, None, None)
    connector.wait_for_save()
    assert events[:3] == [
        ("submit", [b"attention"], [0], [[1016, 2008]]),
        ("submit", [b"state"], [0], [[1016, 0]]),
        ("submit", [b"attention", b"state"], [1, 1], [[3016, 0], [3016, 0]]),
    ]
    assert events[3:] == [("wait", 1), ("wait", 2), ("wait", 3), ("finish", {"req"})]
    assert connector._dump_plan is None and not connector._dump_tasks
    assert not connector.is_save and not connector._dumped_group_slots


def test_deferred_load_start_keeps_in_flight_dumps(save_connector):
    connector = save_connector
    connector.save_kv_layer("mla0", None, None)
    pending = list(connector._dump_tasks)
    plan = connector._dump_plan
    connector.start_load_kv(SimpleNamespace())
    assert connector._dump_tasks == pending and connector._dump_plan is plan
    assert connector._dumped_group_slots == {(0, 0)}
    connector.wait_for_save()
    assert connector._rank_consistency.submit_dump.call_count == 3
    connector._rank_consistency.wait_dump.assert_any_call(pending[0].task)


@pytest.mark.parametrize("is_mla", [False, True])
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("is_dump", [False, True])
def test_rank_scoping_keeps_keys_blocks_and_groups_paired(hla, is_mla, rank, is_dump):
    connector = hla.UCMHybridLinearAttentionConnector()
    connector.tp_rank = rank
    connector.tp_size = 2
    connector.is_mla = is_mla
    connector.request_hasher = lambda key: b"rank1:" + key
    result = connector._scope_blocks([b"mla", b"gdn"], [11, 22], [0, 2], 1, is_dump)
    selected = [1] if is_mla and rank == 0 and is_dump else [0, 1]
    expected_keys = [b"mla", b"gdn"]
    expected_scoped = [
        b"mla" if is_mla or rank == 0 else b"rank1:mla",
        b"gdn" if rank == 0 else b"rank1:gdn",
    ]
    assert result == tuple(
        [values[index] for index in selected]
        for values in [expected_keys, expected_scoped, [11, 22], [0, 2]]
    )


def test_null_filter_keeps_group_pairing(hla):
    keys, blocks, groups = [], [], []
    hla.extend_non_null(keys, blocks, groups, [b"a", b"b", b"c"], [3, 0, 5], 7)
    assert (keys, blocks, groups) == ([b"a", b"c"], [3, 5], [7, 7])


@pytest.fixture
def state_connector(hla):
    connector = hla.UCMHybridLinearAttentionConnector()
    connector.is_mla = True
    connector.group_manager = SimpleNamespace(
        num_groups=2,
        lcm_block_size=4,
        groups_by_id=[
            SimpleNamespace(block_size=4, is_full_attention=True, is_mamba_align=False),
            SimpleNamespace(block_size=4, is_full_attention=False, is_mamba_align=True),
        ],
        compute_mamba_align_state_hash=MagicMock(return_value=b"checkpoint"),
    )
    return connector


@pytest.mark.parametrize(
    "seq_len,scheduled,table,reason,expected",
    [
        (8, 1, [21, 22, 23, 99, 100], "load", [22, 23]),
        (8, 4, [21, 22, 23, 99, 100], "load", [22, 23]),
        (8, 0, [21, 22, 99, 100], "load", [22]),
        (7, 1, [21, 22, 99, 100], "load", [22]),
        (8, 1, [21, 22, 22, 99, 100], "load", [22]),
        (8, 1, [21, 0, 23, 99, 100], "load", [23]),
        (8, 1, [21, 22, 0, 99, 100], "load", [22]),
        (8, 1, [21, 0, 0, 99, 100], "load", []),
        (8, 4, [21, 22, 23, 99, 100], "dump", [22]),
    ],
)
def test_state_restore_selects_checkpoint_and_running_not_scratch(
    hla, state_connector, seq_len, scheduled, table, reason, expected
):
    req_meta = hla.HLARequestMeta(
        group_vllm_block_ids=[[11, 12, 13], table],
        group_ucm_block_ids=[[b"a", b"b", b"c"], []],
    )
    keys, blocks, groups = [], [], []
    state_connector._append_mamba_align_state_block(
        keys, blocks, groups, req_meta, "req", 1, seq_len, reason, scheduled
    )
    assert blocks == expected
    assert keys == [b"checkpoint"] * len(expected)
    assert groups == [1] * len(expected)
    assert 99 not in blocks and 100 not in blocks
    if expected:
        state_connector.group_manager.compute_mamba_align_state_hash.assert_called_once_with(
            state_connector.group_manager.groups_by_id[1],
            seq_len,
            req_meta.group_ucm_block_ids,
        )
    else:
        state_connector.group_manager.compute_mamba_align_state_hash.assert_not_called()


@pytest.mark.parametrize(
    "failure", ["missing_checkpoint", "missing_running", "missing_hash"]
)
def test_incomplete_state_restore_does_not_append_partial_metadata(
    hla, state_connector, failure
):
    tables = {
        "missing_checkpoint": [21],
        "missing_running": [21, 22],
        "missing_hash": [21, 22, 23],
    }
    if failure == "missing_hash":
        state_connector.group_manager.compute_mamba_align_state_hash.return_value = None
    req_meta = hla.HLARequestMeta(
        group_vllm_block_ids=[[], tables[failure]],
        group_ucm_block_ids=[[], []],
    )
    keys, blocks, groups = [b"existing"], [7], [0]
    state_connector._append_mamba_align_state_block(
        keys, blocks, groups, req_meta, "req", 1, 8, "load", 1
    )
    assert (keys, blocks, groups) == ([b"existing"], [7], [0])
    hla.logger.error.assert_called_once()


def _restore_request_meta(hla, *, hbm_hits=0):
    return hla.HLARequestMeta(
        hbm_hit_block_num=hbm_hits,
        total_hit_block_num=2,
        num_token_ids=40,
        token_processed=8,
        group_ucm_block_ids=[
            [b"fa0", b"fa1", b"fa2", b"fa3"],
            [b"", b"", b"", b""],
        ],
        group_vllm_block_ids=[
            [11, 12, 13, 14],
            [21, 22, 23, 99, 100],
        ],
    )


@pytest.mark.parametrize("format_name", ["legacy_list", "resumed_flags", "resumed_set"])
@pytest.mark.parametrize("resumed", [False, True])
def test_cached_request_only_reloads_on_resume(
    hla, state_connector, format_name, resumed
):
    req_meta = _restore_request_meta(hla)
    state_connector.requests_meta = {"req": req_meta}
    if format_name == "legacy_list":
        cached = [
            SimpleNamespace(
                req_id="req",
                new_block_ids=(
                    tuple(list(ids) for ids in req_meta.group_vllm_block_ids)
                    if resumed
                    else ([], [])
                ),
                resumed_from_preemption=resumed,
            )
        ]
    else:
        cached = SimpleNamespace(
            req_ids=["req"],
            new_block_ids=[
                (
                    tuple(list(ids) for ids in req_meta.group_vllm_block_ids)
                    if resumed
                    else None
                )
            ],
        )
        if format_name == "resumed_flags":
            cached.resumed_from_preemption = [resumed]
        else:
            cached.resumed_req_ids = {"req"} if resumed else set()
    output = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached,
        num_scheduled_tokens={"req": 1},
        finished_req_ids=set(),
        preempted_req_ids=set(),
    )
    dispatch = state_connector.build_connector_meta(output).request_meta["req"]
    if resumed:
        assert dispatch.load_block_ids == (
            [b"fa0", b"fa1", b"checkpoint", b"checkpoint"],
            [11, 12, 22, 23],
        )
        assert dispatch.load_group_ids == [0, 0, 1, 1]
        assert not dispatch.load_async
    else:
        # Keep a positive external-hit count to ensure the continuation
        # gate, rather than the absence of a hit, prevents a reload.
        assert dispatch.load_block_ids == ([], [])
        assert dispatch.load_group_ids == []
        state_connector.group_manager.compute_mamba_align_state_hash.assert_not_called()


@pytest.mark.parametrize("hbm_hits", [0, 1, 2])
def test_new_request_restores_only_external_hit_increment(
    hla, state_connector, hbm_hits
):
    req_meta = _restore_request_meta(hla, hbm_hits=hbm_hits)
    state_connector.requests_meta = {"req": req_meta}
    output = SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(req_id="req", block_ids=([11, 12, 13], [21, 22, 23, 99]))
        ],
        scheduled_cached_reqs=[],
        num_scheduled_tokens={"req": 1},
        finished_req_ids=set(),
        preempted_req_ids=set(),
    )
    dispatch = state_connector.build_connector_meta(output).request_meta["req"]
    if hbm_hits < 2:
        expected_attention_keys = [b"fa0", b"fa1"][hbm_hits:]
        expected_attention_blocks = [11, 12][hbm_hits:]
        assert dispatch.load_block_ids == (
            expected_attention_keys + [b"checkpoint", b"checkpoint"],
            expected_attention_blocks + [22, 23],
        )
        assert dispatch.load_group_ids == [0] * (2 - hbm_hits) + [1, 1]
    else:
        assert dispatch.load_block_ids == ([], [])
        state_connector.group_manager.compute_mamba_align_state_hash.assert_not_called()


def restore_connector(hla, state_connector, state_count):
    dispatch = state_connector._generate_hla_dispatch_meta(
        _restore_request_meta(hla), 1, ([], []), request_id="req"
    )
    metadata = hla.UCMConnectorMetadata(request_meta={"req": dispatch})
    connector = transfer_connector(hla)
    connector._connector_metadata = metadata
    connector._get_connector_metadata = lambda: metadata
    names = ["attention", *[f"state{i}" for i in range(state_count)]]
    connector.layer_order = names
    connector.layer_pos = {name: i for i, name in enumerate(names)}
    connector.layer_to_group = {name: int(name != "attention") for name in names}
    connector.layer_to_slot = {
        "attention": 0,
        **{name: i for i, name in enumerate(names[1:])},
    }
    layout = hla.HybridLinearAttentionLayout()
    layout.layer_column_bytes = dict.fromkeys(names, 16)
    layout.extract_layer_addrs = lambda name, blocks: np.asarray(
        [[block] for block in blocks], dtype=np.uint64
    )
    connector.kv_cache_layout = layout
    events, pending = [], []
    memory = {
        name: {22: b"old-checkpoint", 23: b"old-running", 99: b"scratch"}
        for name in names[1:]
    }
    connector.device.synchronize.side_effect = lambda: events.append(("sync",))

    def submit(store, requests, keys, shards, pointers):
        assert store is connector.store and len(set(shards)) == 1
        assert requests == {"req": keys}
        name = "attention" if keys[0] == b"fa0" else names[shards[0] + 1]
        assert pointers[:, 0].tolist() == (
            [11, 12] if name == "attention" else [22, 23]
        )
        task = SimpleNamespace(layer=name, keys=keys, blocks=pointers[:, 0].tolist())
        pending.append(task)
        assert len(pending) <= 2
        events.append(("submit", name))
        return task

    def wait(task):
        pending.remove(task)
        events.append(("wait", task.layer))
        if task.layer in memory:
            for key, block in zip(task.keys, task.blocks):
                memory[task.layer][block] = key

    connector._rank_consistency = SimpleNamespace(
        submit_load=MagicMock(side_effect=submit), wait_load=MagicMock(side_effect=wait)
    )
    return connector, events, pending, memory


@pytest.mark.parametrize("state_count", [2, 3])
def test_layerwise_restore_waits_for_each_consumer_with_depth_two(
    hla, state_connector, state_count
):
    connector, events, pending, memory = restore_connector(
        hla, state_connector, state_count
    )
    connector.start_load_kv(SimpleNamespace())
    assert events == [("sync",), ("submit", "attention"), ("submit", "state0")]
    for values in memory.values():
        assert values[22] == b"old-checkpoint" and values[23] == b"old-running"
    for name in connector.layer_order:
        connector.wait_for_layer_load(name)
        if name in memory:
            assert memory[name] == {
                22: b"checkpoint",
                23: b"checkpoint",
                99: b"scratch",
            }
        seen = list(events)
        connector.wait_for_layer_load(name)
        assert events == seen
    assert [
        name for op, *rest in events if op == "wait" for name in rest
    ] == connector.layer_order
    assert not pending and not connector.load_tasks
    assert connector._load_bytes_total == (state_count + 1) * 2 * 16
    assert connector._load_bytes_recorded
    hla.ucmmetrics.update_stats.assert_called_once_with(
        {"load_bytes_total": connector._load_bytes_total}
    )


@pytest.mark.parametrize("failure", ["submit", "wait"])
def test_layerwise_load_failure_invalidates_request_and_stops_later_submissions(
    hla, state_connector, failure
):
    connector, _, _, _ = restore_connector(hla, state_connector, 3)
    rank = connector._rank_consistency
    if failure == "submit":
        rank.submit_load.side_effect = RuntimeError("storage unavailable")
    else:
        rank.wait_load.side_effect = RuntimeError("read failed")
    connector.start_load_kv(SimpleNamespace())
    connector.wait_for_layer_load("attention")
    count = rank.submit_load.call_count
    for name in connector.layer_order[1:]:
        connector.wait_for_layer_load(name)
    assert rank.submit_load.call_count == count
    assert connector._failure_req_ids == {"req"}
    assert connector._invalid_block_ids == {11, 12, 22, 23}
    connector._connector_worker_meta.mark_failed.assert_called_with("req")


@pytest.mark.parametrize("count", [0, 1, 2, 5, 7])
@pytest.mark.parametrize("ranks", [2, 3])
def test_mla_dump_partition_is_disjoint_and_complete(hla, count, ranks):
    connector = hla.UCMHybridLinearAttentionConnector()
    connector.is_mla = True
    connector.tp_size = ranks
    keys = [f"mla{i}".encode() for i in range(count)]
    blocks = list(range(10, 10 + count))
    assignments = []
    for rank in range(ranks):
        connector.tp_rank = rank
        connector.request_hasher = lambda key: b"private:" + key
        context, stored, selected, groups = connector._scope_blocks(
            keys + [b"state"], blocks + [99], [0] * count + [2], count, True
        )
        assignments.append(context[:-1])
        assert stored[:-1] == context[:-1]
        assert selected[:-1] == [blocks[keys.index(key)] for key in context[:-1]]
        assert groups == [0] * len(context[:-1]) + [2]
        assert context[-1] == b"state" and selected[-1] == 99
        assert stored[-1] == (b"state" if rank == 0 else b"private:state")
    assert [key for assigned in assignments for key in assigned] == keys
    assert max(map(len, assignments)) - min(map(len, assignments)) <= 1
