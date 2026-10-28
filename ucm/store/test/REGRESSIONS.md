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
  -R 'UCPosixWriteDeadlineTest|UCCacheReservedRegionTest'
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
one known block, successful reads, and a read lease. With a three-slot local L1
(two ordinary slots and one load-exclusive slot), pins force an ordinary load of
X into the exclusive slot. The native workers then exit and release their pins.
The test verifies X is READY, previously unreferenced, and ordinary slots remain
reclaimable before asking deferred Early to fetch the same X.

The expected behavior is completion with intact data, no capacity truncation,
and no leaked leases or slots. A bounded retry loop performs genuine ordinary
slot releases so persistent waiting cannot be explained solely by a missing
notification. Failure diagnostics include capacity retries, pending reads,
in-flight bytes, and backend load count. A second placement case leaves X in an
ordinary slot and checks direct L1 reuse without another backend read.

The exclusive-placement case is a known failing regression on base
`acafcda`: it stays in `waitingCapacity` with no pending I/O despite free
ordinary capacity. It is intentionally in the default suite, not disabled or
marked as an expected failure. This change adds tests only. The assertions do
not prescribe whether a future fix migrates, reuses, or safely rereads X.

This test controls storage completion; it does not model disk latency, exercise
shared-memory cross-process behavior, or replace GPU/end-to-end validation.
