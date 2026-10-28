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
cache_dump_reserved_blocks: 0
cache_dump_capacity_policy: spill
posix_io_engine: aio
```

Keep `use_layerwise: true` and one block per chunk. This implementation requires
EarlyCache/EarlyPosix, whole-block L1, and ordinary DMA. Pipeline parallelism,
GDR, SDMA direct and IO aggregation are rejected. Uniform and packed layouts are
supported. Do not mix legacy layerwise writers and the new transactional writers
in one live storage namespace: legacy writers do not take the new writer lease.
Existing archived blocks can be reused. Every participating writer must use the
same mode. Unsupported configurations fail explicitly rather than selecting a
fallback pipeline. In particular, an explicit `cache_io_aggregation: true`
is rejected on builds without IO aggregation support, including scheduler-only
stores and configurations with block dumping disabled. The requested value is
not silently changed to false; omit the option or explicitly set it to false to
use ordinary DMA on those builds.

Both scheduler and worker stores validate configuration dependencies, reservation
limits and capacity policies. Runtime full-block write capability is checked only
on workers (`device_id >= 0`): scheduler stores may not have KV dimensions and do
not create a dump queue. A scheduler can query L2 or attach to existing shared L1
metadata without supplying worker tensor sizes. This does not relax worker backend
capability checks or shared-layout validation.

`cache_dump_reserved_blocks` defaults to **0**, meaning no additional limit on
pinned incomplete/active writeback blocks per store. A positive value sets that
holding limit, including blocks awaiting backend write completion. This setting
never partitions or preallocates a physical write-only region; writers acquire
ordinary L1 slots, shared with Early prefetch. Zero does not remove actual L1
capacity, reference protection or queue backpressure. It is separate from
`early_owner_max_jobs` and `posix_write_inflight_limit`, which are unchanged.
Queued plus unfinished task descriptors, and unfinished block bookkeeping, are
each bounded by `waiting_queue_depth`. The existing limit also covers background
completions so separating I/O cannot create an unbounded queue. Configurations that
omit this option now use zero; set an explicit positive value to retain a previous
holding limit.

`cache_dump_capacity_policy` accepts `spill` (the default) or `wait`. Unknown
values fail configuration; `wait` requires `cache_full_block_dump`. Both policies
use the same reservation budget, which is a per-store holding limit rather than a
physically isolated region. Other L1 users can reduce the slots actually available.
`wait` relies on those users eventually releasing capacity; it does not forcibly
reclaim reader pins or reserve physical slots away from Early. Persistent external
L1 occupation can therefore keep a writer and its HBM source waiting.
In `wait` mode, one task admission slot is reserved for batch finalization; ordinary
queued plus unfinished tasks use at most `waiting_queue_depth - 1` slots. A rejected
submission is reported as NoSpace and never reported as saved data.

## Data and control flow

1. Before the first layer's D2H, acquire a full physical L1 slot and keep a whole-block
   reference. The block is not a complete cache hit while any layer is missing.
2. Wait for the submitted compute event, copy that layer, synchronize DMA, then
   mark only that layer ready. Null tensor slots and physical padding are zeroed.
3. Once all layers are ready, issue one offset-zero POSIX `PWRITE` whose length is
   the physical block size. Packed blocks use the sum of their physical layer sizes.
4. Wait for physical I/O completion, then synchronously rename the temporary file
   into the archived namespace. A failed write or short write never publishes it.
5. At `wait_for_save`, resubmit addresses of layers whose save callbacks were
   actually visited, with a new compute event. `spill` waits for previous source
   fences first. `wait` submits finalization before those fences: a waiting block
   may depend on this repair to release another block's slot.
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

With `cache_dump_capacity_policy: spill`, an exhausted positive reservation budget
or actual L1 allocation failure causes the READY layers of another incomplete block
to be written using their original layer offsets. At budget zero, only actual L1
allocation pressure triggers this capacity fallback. Wait for those writes before
releasing its L1 reference. Track successfully persisted layers independently of
L1 residency. This permits progress
even when a batch contains more blocks than L1 can hold.

With `cache_dump_capacity_policy: wait`, ordinary capacity pressure never causes a
partial write. Defer the block needing a new slot while continuing later layer
submissions for blocks that already hold one. Budget zero removes quota-induced
waiting; a block still waits when no physical L1 slot can be acquired. For the
order A0, B0, A1, B1, A2, B2 and one reservation, B waits while A1/A2 complete A; A is written as one full block,
and its slot becomes available to B only after physical I/O completion. Preserve
submission order within each block, including finalization. The unit is a block,
not a request, which may contain many blocks.

Waiting tasks retain their HBM addresses/events, including across an expired wait;
the source fence cannot return before their copies drain. New work and local I/O
completion wake the coordinator immediately. Releases by other shared-L1 users are
detected through the shared release epoch, checked on a 2 ms wait interval while idle.
Capacity-only waiters are rescanned on release or task/I/O progress. A follower of
another loader also rechecks READY on that interval: the loader can finish while
keeping a consumer pin, without changing the release epoch.
Wait-policy quota checks and physical-capacity failure checks scan only resident
writers. Their number is bounded by actual L1 slots and any positive reservation
budget, rather than all blocks described by queued tasks.
A known failed writer is retained for a valid-source retry; blocked submissions
report that storage failure instead of waiting forever for its pinned slot.

`wait` changes normal capacity handling only. Explicit finalization with genuinely
missing layers and shutdown still preserve available layers through the existing
terminal spill path below. No incomplete block is advertised as a full cache hit.
Callbacks must be submitted asynchronously before waiting for a whole mixed-block
task; synchronously waiting on B0 before supplying A1/A2 cannot make progress with
one slot. The connector enforces the required finalization/source-fence ordering.

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

Native retry success does not yet establish recovery through the Early owner
protocol. The Python wrapper currently treats a failed dump `Wait` as unproven I/O
completion even when the native task has drained. An installed I/O failure handler,
or an active consumer observing an unscoped failure, can fence the entire domain.
This also occurs for missing-layer finalization and failed archive renames after
successful writes. The native task can be retried with valid sources, but that does
not clear the Python fence. Do not bypass the fence in a live service; distinguishing
completed write errors from unknown completion remains unresolved.

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
shared L1, shape validation, factory routing and shutdown draining. Wait-policy
regressions cover interleaved blocks sharing one slot, configurable slot counts,
physical write completion, failed-layer repair, terminal incomplete blocks,
external L1 pressure, READY publication without foreign pin release, queued
timeouts, reserved finalizer admission, and storage failure propagation. Scheduler
regressions cover initialization without worker KV dimensions, shared-L1 lookup,
and rejection of invalid static configuration; workers still reject backends
without full-block write capability. Both capacity policies are covered. Invalid
policies and wait-without-block-dump are rejected. Zero-budget regressions hold
20 incomplete blocks without premature spill in both policies, then verify one
full write per block. They also exhaust physical L1, retain pins through slow
writes, preserve bounded queue/finalizer admission, and report a failed writer
instead of waiting forever for its occupied slot. Existing positive-budget cases
continue to exercise explicit holding limits.
The connector
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
