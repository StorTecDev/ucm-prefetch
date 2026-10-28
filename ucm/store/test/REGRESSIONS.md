# Cache and POSIX lifecycle regressions

These tests belong to the existing `ucmstore.test` GoogleTest executable and are
registered with CTest when `BUILD_UNIT_TESTS=ON`. Use `RUNTIME_ENVIRONMENT=simu`
for CPU-only execution. Model weights and an inference server are not required.
Prepare the dependency versions declared in `ucm/shared/vendor/dep-*.cmake`
before an offline build.

```bash
cmake -S . -B build/regression -DRUNTIME_ENVIRONMENT=simu \
  -DBUILD_UNIT_TESTS=ON -DDOWNLOAD_DEPENDENCE=OFF -DBUILD_UCM_SPARSE=OFF
cmake --build build/regression --target ucmstore.test --parallel 2
ctest --test-dir build/regression --output-on-failure --timeout 30 \
  -R 'UCPosixWriteDeadlineTest|UCCacheReservedRegionTest|UCCacheReservedRecoveryTest'
```

## Slow writes and queued deadlines

`case/posix/posix_write_deadline_test.cc` uses three small writes and an admission
window of one. The first write consumes most of the later tasks' deadline. The
second is then admitted but cannot finish before its original deadline; the third
remains queued. This is a lifecycle regression, not a window-size benchmark.

The existing AIO hooks hold accepted control blocks and reject cancellation until
the test releases them to the real Linux `io_submit` syscall. The assertions check
that admission does not restart a task's timeout, a queued timeout does not submit
I/O, and an accepted write retains ownership until its physical callback. Both
expired callers must report `Timeout`; the first write and a subsequent recovery
write must succeed. The recovery block is read back and compared byte for byte.

On base `acafcda`, this case passes with the expected timeout results. It covers
queue deadline and cleanup behavior; it does not prevent writeback overload.

The fixture releases held I/O before joining waiters, including after a failed
assertion. The CTest timeout also bounds failures that prevent native teardown.
The filesystem used for test output must support Linux AIO and direct I/O.

## READY block in the load-exclusive region

`case/cache/cache_reserved_region_test.cc` keeps the real `LoadQueue`,
`TransBuffer`, and `ReservedPrefetch`. Its controlled GoogleMock backend supplies
one known block, successful reads, and a read lease. With a three-slot L1
(two ordinary slots and one load-exclusive slot), pins force an ordinary load of
X into the exclusive slot. The native workers then exit and release their pins.
The test verifies X is READY, previously unreferenced, and ordinary slots remain
reclaimable before asking deferred Early to fetch the same X.

The expected behavior is completion with intact data, no capacity truncation,
and no leaked leases or slots. A bounded retry loop performs genuine ordinary
slot releases so persistent waiting cannot be explained solely by a missing
notification. Failure diagnostics include capacity retries, pending reads,
in-flight bytes, and backend load count. Ordinary placement checks direct L1
reuse without another backend read; exclusive placement requires one additional
L2 read into an ordinary slot.

On base `acafcda`, exclusive placement stays in `waitingCapacity` with no pending
I/O despite reclaimable ordinary capacity. The recovery now requires a positive
L2 lookup protected by a read lease, a READY source with no other references,
and an ordinary destination. It obtains that destination before removing X's
exclusive mapping. The destination stays LOADING until the L2 read completes;
this path does not copy the old L1 bytes or expose the exclusive region to Early.

Both placement cases run with local layer storage, local whole-block storage,
and shared whole-block storage. `UCCacheReservedRecoveryTest` additionally checks:

- A live source reference prevents eviction; releasing it lets recovery proceed.
  The shared case takes this reference through a second mapping of the same domain.
- Exhausted ordinary capacity preserves the source until a slot is released.
- Missing L2 data, a missing lease, or an incomplete source cannot authorize eviction.
- Activation retains newly verified L2 availability if deferred X changes regions.
- Cancellation drains the reread before releasing its destination and lease.
- A failed reread never publishes READY and can be retried after references drain.
- Concurrent Early consumers and ordinary loading share one reread destination.

Shared cases use fresh whole-block domains and release only their own backing.
The legacy shared layer-storage cases are not added here because their setup can
clean unrelated domains. These tests control storage completion; they do not
model disk latency or replace GPU/end-to-end validation.
