# Whole-block L1 → L2 writeback

`cache_full_block_dump` is an opt-in write path for Early layerwise connectors.
Early prefix reservation, L2 reads, ordinary loads, and owner scheduling keep their
existing implementation. This option does not change HBM capacity or write-window
settings, and it does not evict HBM: D2H copies selected cache blocks.

## Configuration

In the existing `ucm_connector_config`, use:

```yaml
cache_full_block_load: true
cache_full_block_l1: true
cache_full_block_dump: true
cache_dump_reserved_blocks: 16
posix_io_engine: aio
```

Keep `use_layerwise: true` and one block per chunk. This implementation requires
EarlyCache/EarlyPosix, whole-block L1, and ordinary DMA. Pipeline parallelism,
GDR, SDMA direct and IO aggregation are rejected. Uniform and packed layouts are
supported. Do not mix legacy layerwise writers and the new transactional writers
in one live storage namespace: legacy writers do not take the new writer lease.
Existing archived blocks can be reused. Every participating writer must use the
same mode. Unsupported configurations fail explicitly rather than selecting a
fallback pipeline.

`cache_dump_reserved_blocks` bounds pinned incomplete/active L1 blocks per store,
including a block whose backend write is being waited. The default is 16, separate
from `early_owner_max_jobs` and `posix_write_inflight_limit`. Pending task descriptors
and unfinished block bookkeeping are each bounded by `waiting_queue_depth`.
The write worker never waits indefinitely for an L1 slot held by a reader.

## Data and control flow

1. Before the first layer's D2H, acquire a full physical L1 slot and keep a whole-block
   reference. The block is not a complete cache hit while any layer is missing.
2. Wait for the submitted compute event, copy that layer, synchronize DMA, then
   mark only that layer ready. Null tensor slots and physical padding are zeroed.
3. Once all layers are ready, issue one offset-zero POSIX `PWRITE` whose length is
   the physical block size. Packed blocks use the sum of their physical layer sizes.
4. Wait for physical I/O completion, then synchronously rename the temporary file
   into the archived namespace. A failed write or short write never publishes it.
5. At `wait_for_save`, first drain stage tasks. Resubmit addresses of layers whose
   save callbacks were actually visited, with a new compute event. Native finalization
   copies missing layers and waits before HBM can be recycled. Never infer that an
   unvisited layer is complete. The finalizer runs once per connector batch.

An OFD write lease serializes participating processes for each content identity.
It uses a separate persistent lock file, so it does not block the read lease.
Lease contention defers a normal stage; finalization reports unresolved contention
instead of spinning or accessing another writer's transaction. Content identities
must continue to refer to immutable KV data, as in the existing cache protocol.

## Capacity pressure and incomplete blocks

If the reservation budget is exhausted, spill the READY layers of another incomplete
block using the original layer offsets. Wait for those writes before releasing its
L1 reference. Track successfully persisted layers independently of L1 residency.
This permits progress even when a batch contains more blocks than L1 can hold.

These layer writes suppress the legacy "last layer means publish" rule. Publication
requires every layer to have completed successfully, even when L1 was evicted and
refilled between layers. Complete L1 data still takes the whole-block path.

If finalization still lacks a layer, save the available layers to the temporary file,
release successfully spilled L1 reservations, and return an incomplete-block error.
The partial file is not visible to lookup or Early reads. It is not recoverable through
the ordinary L2 read API until complete; a later complete source can overwrite it.
Do not write the uninitialized remainder as a purported complete block.

AIO errors keep the resident L1 copy for a subsequent valid-source retry. A failed
explicit rename retains the temporary file. Shutdown drains accepted tasks and
attempts to spill incomplete blocks. A permanent storage failure is reported; this
cache is not a durable storage system and cannot guarantee preservation after process
termination or hardware failure. HBM addresses are never retained for use after their
connector batch has ended.

## Performance and observability

This first implementation uses one background write coordinator and waits for each
backend task before proceeding. The caller can still enqueue layer saves asynchronously,
but completion waiting provides backpressure and may reduce write parallelism or increase
`wait_for_save` time. Fewer syscalls alone do not prove better TTFT or throughput. No
GPU performance result is implied by the CPU regressions.

Legacy DumpQueue-specific cache timing counters do not cover this separate coordinator;
use client latency and the backend I/O counters for an initial comparison.

The successful AIO counters are `posix_dump_write_ops_total`,
`posix_dump_write_bytes_total`, and `posix_dump_full_block_ops_total`. A fallback can
write some layers and later rewrite a complete L1 block; include those additional
bytes when comparing implementations. The existing POSIX write window counts I/O
operations, so its byte budget differs between layer writes and whole-block writes.

## Tests

The default native test target includes `UCCacheBlockDumpTest.*` in
`ucm/store/test/case/cache/cache_block_dump_test.cc`. It uses CPU simulated DMA plus
real Linux POSIX AIO, with controlled submit hooks. Coverage includes 14-layer uniform
and packed blocks, exactly one complete write, missing-layer backfill, unpublished
partial data, more blocks than L1 slots, failed writes and retries, write leases,
shared L1, shape validation, factory routing and shutdown draining. The connector
lifecycle cases live in `test/test_ucm_connector_metrics.py`.

Build and run in an independent source/build copy with the existing offline dependencies:

```sh
cmake -S source -B build -DRUNTIME_ENVIRONMENT=simu -DBUILD_UNIT_TESTS=ON \
  -DDOWNLOAD_DEPENDENCE=OFF -DBUILD_UCM_SPARSE=OFF
cmake --build build --target ucmstore.test ucmpipelinestore --parallel 2
ctest --test-dir build --output-on-failure --timeout 30 -R UCCacheBlockDumpTest
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  source/test/test_ucm_connector_metrics.py
```

Use an interpreter with pytest for Python tests and a compatible Python development
installation for the native binding. No GPU, vLLM service, model or downloaded weights
are required for these tests.
