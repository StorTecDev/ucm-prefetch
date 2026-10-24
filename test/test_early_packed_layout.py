"""Compact Early files must match the actual device tensors."""

import importlib.util
from pathlib import Path

import pytest

SOURCE = Path(__file__).parents[1] / "ucm/integration/vllm/early_storage_layout.py"
SPEC = importlib.util.spec_from_file_location("early_storage_layout", SOURCE)
layout = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(layout)


def configuration(path, alignment=4096):
    return {
        "store_pipeline": "Cache|Posix",
        "storage_backends": [str(path)],
        "store_io_alignment": alignment,
        "cache_buffer_capacity_gb": 18,
        "posix_capacity_gb": 128,
    }


def packed_fixture():
    from types import SimpleNamespace
    import numpy as np

    indexed = {0, 1, 2, 6, 10}
    lengths = [86016 if i in indexed else 73728 for i in range(14)]
    config = dict(
        configuration("unused", 4096),
        cache_full_block_load=True,
        posix_packed_layer_bytes=lengths,
        shard_size=86016,
        tensor_size_list=[73728, 8448],
    )
    native = SimpleNamespace(
        base_ptrs=np.array([[1, 2 if i in indexed else 0] for i in range(14)]),
        tensor_size_lists=np.array([[73728, 8448]] * 14),
    )
    return config, native


def test_packed_layout_binds_actual_tensors_and_keeps_l1_capacity():
    config, actual = packed_fixture()
    assert (
        layout.validate_packed_layers(
            config, actual, layerwise=True, blocks_per_chunk=1
        )
        == 1093632
    )
    assert config["shard_size"] == 86016
    assert config["cache_buffer_capacity_gb"] == 18
    assert config["posix_capacity_gb"] == 128


@pytest.mark.parametrize(
    "lengths", [[], [True], [4096.0], [0], [-4096], [73729], [2**32], "73728"]
)
def test_packed_invalid_lengths_reject_without_files(tmp_path, lengths):
    config, _ = packed_fixture()
    config.update(storage_backends=[str(tmp_path)], posix_packed_layer_bytes=lengths)
    with pytest.raises(ValueError):
        layout.validate_packed_config(config)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "mutation", ["real_tail", "extra_padding", "count", "offsets", "chunk", "direct"]
)
def test_packed_actual_layout_mismatch_rejected(mutation):
    config, actual = packed_fixture()
    chunk, layerwise = 1, True
    if mutation == "real_tail":
        actual.base_ptrs[3, 1] = 2
    elif mutation == "extra_padding":
        config["posix_packed_layer_bytes"][3] = 86016
    elif mutation == "count":
        config["posix_packed_layer_bytes"].pop()
    elif mutation == "offsets":
        actual.tensor_size_lists[0, 0] += 4096
    elif mutation == "chunk":
        chunk = 2
    else:
        layerwise = False
    with pytest.raises(ValueError):
        layout.validate_packed_layers(
            config, actual, layerwise=layerwise, blocks_per_chunk=chunk
        )
