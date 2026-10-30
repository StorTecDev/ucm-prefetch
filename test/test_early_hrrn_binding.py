"""Request rounds through real native Posix reads and simulated device transfer."""

import array
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_early_batch_pipeline import Control, wait

from ucm.integration.vllm.early_prefetch import EarlyWorker, WorkerJob
from ucm.store.pipeline import ucmpipelinestore as native
from ucm.store.pipeline.early_connector import EarlyPipelineStore, _ReservedLoad


@pytest.mark.parametrize("block_l1", [False, True])
def test_request_round_reads_all_batches_and_hands_off_exact_data(tmp_path, block_l1):
    store_dir = Path(native.__file__).resolve().parent.parent
    shard_bytes, layers, long_count, short_count = 4096, 2, 192, 4
    count = long_count + short_count
    config = dict(
        storage_backends=[str(tmp_path)],
        device_id=0,
        tensor_size=shard_bytes,
        shard_size=shard_bytes,
        block_size=layers * shard_bytes,
        data_dir_shard_bytes=1,
        posix_data_trans_concurrency=1,
        posix_lookup_concurrency=1,
        posix_io_engine="psync",
        io_direct=False,
        store_health={"enabled": False},
    )
    blocks = [uuid.uuid4().bytes for _ in range(count)]
    ids = np.frombuffer(b"".join(block * layers for block in blocks), dtype=np.uint8)
    indexes = array.array("Q", list(range(layers)) * count)
    source = np.arange(count * layers * shard_bytes, dtype=np.uint8).reshape(
        count * layers, shard_bytes
    )
    source ^= np.arange(count * layers, dtype=np.uint16).astype(np.uint8)[:, None]
    addresses = np.array([[row.ctypes.data] for row in source], dtype=np.uint64)
    seed = native.PipelineStore()
    seed.Stack("Posix", str(store_dir / "posix/libposixstore.so"), config)
    seed.Wait(seed.Dump(ids, indexes, addresses))
    del seed

    pipeline = native.PipelineStore()
    if block_l1:
        config.update(
            posix_io_engine="aio",
            io_direct=True,
            cache_full_block_load=True,
            cache_full_block_l1=True,
        )
    pipeline.Stack("EarlyPosix", str(store_dir / "posix/libposixstore.so"), config)
    pipeline.Stack(
        "EarlyCache",
        str(store_dir / "cache/libcachestore.so"),
        dict(
            config,
            unique_id=uuid.uuid4().hex,
            cache_buffer_capacity_gb=1,
            share_buffer_enable=False,
            cache_load_exclusive_buffer_number=128,
            waiting_queue_depth=8,
            running_queue_depth=16,
            cache_stream_number=1,
        ),
    )
    store = EarlyPipelineStore.__new__(EarlyPipelineStore)
    store._native_store, store.store_, store.config = (
        pipeline,
        _ReservedLoad(pipeline),
        config,
    )
    connector = SimpleNamespace(
        store=store,
        is_mla=True,
        tp_rank=0,
        tp_size=1,
        unique_id="hrrn-native-test",
        use_request_async_load=False,
        launch_config={"enable_early_owner": False},
        _vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_index=0)
        ),
    )
    worker = EarlyWorker(connector, Control)
    submitted = []
    step = worker._step

    def record_step(job, batch, available):
        before = batch.stats.get("l2_read_bytes", 0)
        advanced = step(job, batch, available)
        if batch.stats.get("l2_read_bytes", 0) > before:
            submitted.append(job.key)
        return advanced

    worker._step = record_step
    jobs = [
        WorkerJob("long", blocks[:long_count], entered=time.monotonic() - 1000),
        WorkerJob("short", blocks[long_count:], entered=time.monotonic() - 1),
    ]
    try:
        with worker.condition:
            for job in jobs:
                worker.jobs[job.key] = job
                worker._reserve(job)
            worker.condition.notify_all()
        wait(lambda: all(job.ready for job in jobs))
        assert len(jobs[0].batches) == 3
        first_short = submitted.index("short")
        assert first_short > 0
        assert submitted[:first_short] == ["long"] * first_short
        assert submitted[first_short:] == ["short"] * (len(submitted) - first_short)
        with worker.condition:
            assert {m["key"]: m["hit"] for m in worker.control.messages} == {
                "long": long_count,
                "short": short_count,
            }
        batches = [batch for job in jobs for batch in job.batches]
        assert sum(batch.stats["l2_read_bytes"] for batch in batches) == source.nbytes
        target = np.zeros_like(source)
        destinations = np.array([[row.ctypes.data] for row in target], dtype=np.uint64)
        handles = [batch.reservation["handle"] for batch in batches]
        task = native.EarlyLoad(pipeline, ids, indexes, destinations, handles)
        pipeline.Wait(task)
        np.testing.assert_array_equal(source, target)
    finally:
        worker.close()
    assert not worker.data_thread.is_alive()
    assert all(batch.reservation["batch"].protected_bytes == 0 for batch in batches)
