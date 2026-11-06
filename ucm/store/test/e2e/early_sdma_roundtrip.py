"""Verify Early whole-block writeback through A3 mapped SDMA and persisted L2.

Run only on an idle A3 device with an ascend-a3 build. The caller supplies a new
directory; artifacts remain there for inspection. CPU mode requires a simu build.
"""

import argparse
import gc
import hashlib
import json
import uuid
from pathlib import Path

import numpy as np
import torch
from ucm.store.pipeline.early_connector import EarlyPipelineStore

TENSOR_BYTES = (4096, 2048, 2048)
LAYERS = 3
BLOCKS = 3


def configuration(directory, device, policy, direct):
    return {
        "store_pipeline": "Cache|Posix",
        "storage_backends": [str(directory)],
        "unique_id": uuid.uuid4().hex,
        "device_id": device,
        "local_rank_size": 1,
        "tensor_size_list": list(TENSOR_BYTES),
        "shard_size": sum(TENSOR_BYTES),
        "block_size": sum(TENSOR_BYTES) * LAYERS,
        "cache_buffer_capacity_gb": 1,
        "cache_load_exclusive_buffer_number": 0,
        "cache_stream_number": 1,
        "share_buffer_enable": False,
        "io_direct": True,
        "posix_io_engine": "aio",
        "posix_data_trans_concurrency": 1,
        "posix_lookup_concurrency": 1,
        "cache_full_block_load": True,
        "cache_full_block_l1": True,
        "cache_full_block_dump": True,
        "cache_dump_capacity_policy": policy,
        "cache_sdma_direct": direct,
        "cache_host_memory_mode": "default",
        "cache_pin_allow_pageable_fallback": False,
    }


def check_policy(directory, device, policy):
    direct = device != "cpu"
    target = f"npu:{device}" if direct else "cpu"
    index = int(device) if direct else 0
    config = configuration(directory, index, policy, direct)
    ids = [hashlib.sha256(f"block-{i}".encode()).digest()[:16] for i in range(BLOCKS)]
    expected, source = {}, {}
    for block in range(BLOCKS):
        for layer in range(LAYERS):
            values = [
                torch.from_numpy(
                    (
                        (np.arange(size) * 17 + block * 31 + layer * 7 + part) % 251
                    ).astype(np.uint8)
                )
                for part, size in enumerate(TENSOR_BYTES)
            ]
            # A null source must leave zeros, not a previous slot's stale bytes.
            if layer % 2:
                values[-1].zero_()
            expected[block, layer] = values
            source[block, layer] = [value.to(target) for value in values]
    if direct:
        torch.npu.synchronize()
    writer = EarlyPipelineStore(config)
    for layer in range(LAYERS):
        addresses = [
            [tensor.data_ptr() for tensor in source[block, layer]]
            for block in range(BLOCKS)
        ]
        if layer % 2:
            for row in addresses:
                row[-1] = 0
        task = writer.dump_data(ids, [layer] * BLOCKS, addresses)
        writer.wait_dump_sources(task)
        writer.wait(task)
    assert all(writer.lookup(ids))
    del writer
    gc.collect()

    # A different private L1 pool forces reconstruction from committed L2 bytes.
    reader = EarlyPipelineStore(dict(config, unique_id=uuid.uuid4().hex))
    assert all(reader.lookup(ids))
    checked = 0
    for layer in range(LAYERS):
        outputs = [
            [
                torch.full((size,), 0xA5, dtype=torch.uint8, device=target)
                for size in TENSOR_BYTES
            ]
            for _ in range(BLOCKS)
        ]
        if direct:
            # UCM owns a separate stream; initialization must finish before it
            # overwrites these tensors, or the fill can race with a correct H2D.
            torch.npu.synchronize()
        reader.wait(reader.load(ids, [layer] * BLOCKS, outputs))
        if direct:
            torch.npu.synchronize()
        for block, tensors in enumerate(outputs):
            for actual, wanted in zip(tensors, expected[block, layer]):
                torch.testing.assert_close(actual.cpu(), wanted, rtol=0, atol=0)
                checked += wanted.numel()
    del reader
    gc.collect()
    return {"policy": policy, "bytes_checked": checked, "passed": True}


def check_unsupported_sdma(directory):
    for device in (-1, 0):
        config = configuration(directory, device, "wait", True)
        try:
            EarlyPipelineStore(config)
        except RuntimeError as error:
            assert "SDMA Direct" in str(error), str(error)
        else:
            raise AssertionError("simu accepted unsupported SDMA Direct")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="0", help="Idle A3 device index, or cpu with a simu build"
    )
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=False)
    if args.device != "cpu":
        import torch_npu  # noqa: F401

        torch.npu.set_device(int(args.device))
    if args.device == "cpu":
        check_unsupported_sdma(args.directory)
    results = []
    for policy in ("spill", "wait"):
        directory = args.directory / policy
        directory.mkdir()
        results.append(check_policy(directory, args.device, policy))
    result = {
        "passed": True,
        "device": args.device,
        "sdma_direct": args.device != "cpu",
        "fresh_reader_l1": True,
        "results": results,
    }
    (args.directory / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
