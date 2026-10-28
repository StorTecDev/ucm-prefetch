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
from `early_owner_max_jobs` and `posix_write_inflight_limit`. Queued plus unfinished task descriptors, and unfinished block bookkeeping,
are each bounded by `waiting_queue_depth`. The existing limit also covers background
completions so separating I/O cannot create an unbounded queue.
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
5. At `wait_for_save`, first wait for stage **source** fences. Resubmit addresses of
   layers whose save callbacks were actually visited, with a new compute event.
   Native finalization copies missing layers. `wait_dump_sources()` releases the
   source/event dependency without consuming the task; `check()`/`wait()` still
   report final I/O and publication. Never infer that an unvisited layer is complete.
   The finalizer runs once per connector batch. A failed source fence propagates:
   logging it and allowing HBM reuse is unsafe.
6. A drained D2H failure marks only the failed layer. Other READY layers remain
   valid; a later valid-source submission retries missing/failed layers. Readers
   waiting on that failed layer can observe FAILED, while the whole block remains
   unavailable until every layer is READY. Unknown DMA completion retains both
   endpoints until synchronization proves completion, potentially indefinitely.
7. A timeout does not remove accepted tasks from the source queue. Queued tasks
   still execute, and their HBM addresses/events remain protected until source
   copying drains. A source-fence timeout is retained in the final task result;
   `wait_dump_sources()` returns OK only after HBM is safe. A final `wait()` timeout
   also waits for task completion before returning Timeout, even if storage later
   succeeds. The cancellation callback is a no-op, as in the legacy cache queue.
   Unlike legacy DumpQueue's dispatch-time failure-set check, this coordinator
   still executes a queued task after its deadline; timeout is an outcome, not a
   reason to skip accepted work. The existing POSIX AIO backend timeout/cancellation
   policy is unchanged.
   Accepted DMA/I/O must drain before its buffers are released, so there is no hard
   return deadline on faulty hardware. L2 failures are observed by ordinary `wait()`,
   not by the source-only fence. Finished sending to the scheduler refers to HBM
   safety; successful-dump metadata is delayed until final storage completion.

Worker success reporting treats a successful explicit finalization as repair of
older dump failures for the same request, logical block and Store. Merely accepting
finalization, or successfully writing another layer, does not clear a failure.
Submission order determines which failures are covered even if task waits occur
out of order. Newer failures and failures in another Store remain visible; accounting
is released when the request finishes. This reporting rule assumes finalization
success means all layers were persisted and the block was published.

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

One source worker owns layer state and one independent I/O worker drains L1-to-L2
writes. With available reservations, another block's D2H can proceed while a disk
write is pending. Source fences remove the unconditional L2 wait from forward exit
and request completion. Reservation pressure or a dependency on the same block can
still require a physical write to drain before source copying continues. The I/O
worker remains serial; this change does not claim greater storage bandwidth. Fewer syscalls alone do not prove better TTFT or throughput. No
GPU performance result is implied by the CPU regressions.

Legacy DumpQueue-specific cache timing counters do not cover this separate coordinator;
use client latency and the backend I/O counters for an initial comparison.

The successful AIO counters are `posix_dump_write_ops_total`,
`posix_dump_write_bytes_total`, and `posix_dump_full_block_ops_total`. A fallback can
write some layers and later rewrite a complete L1 block; include those additional
bytes when comparing implementations. The existing POSIX write window counts I/O
operations, so its byte budget differs between layer writes and whole-block writes.

## Compatibility and unresolved policies

The source-fence binding requires `EARLY_BLOCK_DUMP_PROTOCOL_VERSION = 2`.
Rebuild the complete native pipeline (StoreV1 gained an optional method), and
restart all participants. The shared whole-block layout fingerprint also changes
for the layer-local failure state; do not attach old binaries to a new live domain.
Early lookup, reservation, and loading algorithms remain unchanged.

Cross-batch HBM retention, durable partial-layer metadata, automatic orphan `.tmp`
cleanup and cross-process duplicate-writer coalescing are not enabled by this change.
Incomplete transactions still follow the existing finalization policy: available
layers can be spilled as unpublished data, and a later valid complete source can
retry. A failed spill can retain a bounded transaction/pin until retry or shutdown.
There is no new promise of persistent recovery after a process or machine failure.

## Tests

The default native test target includes `UCCacheBlockDumpTest.*` in
`ucm/store/test/case/cache/cache_block_dump_test.cc`. It uses CPU simulated DMA plus
real Linux POSIX AIO, with controlled submit hooks. Coverage includes 14-layer uniform
and packed blocks, exactly one complete write, missing-layer backfill, unpublished
partial data, more blocks than L1 slots, failed writes and retries, write leases,
shared L1, shape validation, factory routing and shutdown draining. The connector
lifecycle cases live in `test/test_ucm_connector_metrics.py`. Additional regressions
inject a rejected layer copy, verify retry does not copy READY layers, exercise the
shared failed-layer state, hold an actual POSIX AIO completion while another source
copy advances, preserve queued writes through both source-fence and final-wait
timeouts, and retain an active write lease through timeout.
Python cases distinguish HBM source release from successful storage publication.

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
