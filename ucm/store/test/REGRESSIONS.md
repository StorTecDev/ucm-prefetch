# Cache and POSIX lifecycle regressions

These tests belong to the existing `ucmstore.test` GoogleTest executable and are
registered with CTest when `BUILD_UNIT_TESTS=ON`. Use `RUNTIME_ENVIRONMENT=simu`
for CPU-only execution. Model weights and an inference server are not required.
The default configuration downloads the pinned dependency versions declared in
`ucm/shared/vendor/dep-*.cmake`. For an offline build, prepare those dependencies
first and add `-DDOWNLOAD_DEPENDENCE=OFF`.

```bash
cmake -S . -B build/regression -DCMAKE_BUILD_TYPE=Debug \
  -DRUNTIME_ENVIRONMENT=simu -DBUILD_UNIT_TESTS=ON -DBUILD_UCM_SPARSE=OFF
cmake --build build/regression --parallel 4
ctest --test-dir build/regression --output-on-failure --timeout 90 \
  -R 'UCCache|EarlyPosix|ReadLease|UCPosixWriteDeadlineTest|ucm_.*_probe'
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

This covers queue deadline and cleanup behavior; it does not prevent writeback
overload.

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

Without the reread path, exclusive placement can stay in `waitingCapacity` with
no pending I/O despite reclaimable ordinary capacity. Recovery requires a positive
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

## Whole-block writeback

`UCCacheBlockDumpTest.*` exercises the opt-in write coordinator with CPU DMA and real
POSIX AIO, including full-block byte identity, missing layers, pressure-driven layer
spills, failure recovery and shutdown. Configuration, publication semantics and Python
lifecycle coverage are described in [full_block_writeback.md](../../../docs/full_block_writeback.md).

## Whole-block reads and packed layouts

The standalone correctness probes in `test/` are registered with CTest for the
simulation runtime. They use generated byte patterns and temporary directories;
they do not need model weights, workload traces, or accelerator identifiers.

- `ucm_full_block_l2_probe` covers vector reads, partial hits, cancellation,
  capacity rollback, fallback loading, and short reads. A second invocation uses
  the larger supported shard size.
- `ucm_packed_l2_probe` covers packed offsets, AIO/PSync agreement, retained
  buffers, and invalid tensor layouts.
- `ucm_early_scope_probe` checks the Early-only boundary and read-lease deletion
  protection, then runs the packed-layout cases.
- `ucm_block_l1_probe` covers whole-block L1 publication, layer views, native
  loads/dumps, sharing, and failure propagation.
- `ucm_write_window_probe` covers real AIO read bypass, write admission,
  cancellation, fairness, and permit release after errors.

## Python control and connector regressions

Run the isolated modules with `--noconftest` to avoid loading the upstream
inference-server fixtures. The control tests need `pytest`, `numpy`, `pyzmq`,
and `msgspec`. Run the connector fixture in a separate process because it
installs module stubs, including a NumPy stub.

```bash
python -m pytest --noconftest -q -o addopts= \
  test/test_ucm_connector_metrics.py
test_tmp=$(mktemp -d -t ucm-py-XXXXXX)
python -m pytest --noconftest -q -o addopts= --basetemp="$test_tmp" \
  test/test_early_batch_pipeline.py test/test_early_owner_protocol.py \
  test/test_early_capacity_wait.py test/test_early_owner_budget.py \
  test/test_early_packed_layout.py test/suites/Unit/test_early_ready_prefix.py
```

These cases cover generation ownership, safe capacity admission, release-event
notification, cancellation/draining, packed-layout validation, metrics, and dump
source-completion/finalization. Stubbed connector tests do not exercise a running
vLLM engine or device transfers. The short temporary directory avoids the Unix
socket pathname limit in the multiprocess ZeroMQ test; use a short temporary
directory root if the system's default is deeply nested.

For the real CPython/C++ binding cases, use the simulation build without
installing its libraries into the source tree:

```bash
python test/python_with_build.py --build "$(pwd)/build/regression" \
  --noconftest -q -o addopts= test/test_early_batch_binding.py
```

This requires the normal UCM Python dependencies, including PyTorch and NumPy.
The helper selects extensions from the specified build. Binding tests exercise
real POSIX storage and simulated device copies; they do not establish GPU/NPU
correctness, model-output equivalence, or end-to-end performance.

The remaining integration modules (`test_early_ready_reuse.py`,
`test_early_request_async.py`, `test_early_request_async_lifecycle.py`,
`test_early_v080_compat.py`, and `test_early_native_scheduler.py`) additionally
require a compatible vLLM environment. The scheduler module imports vLLM's
`tests.v1.core.utils` from its source checkout. Run these with the same build
helper, in a separate process from the stubbed connector fixture. A CPU control
or binding pass does not establish that these integration cases passed.

Keep generated build files, logs, environment inventories, and benchmark outputs
outside tracked source. Documentation examples should use relative paths or
explicit placeholders rather than personal filesystem locations.
