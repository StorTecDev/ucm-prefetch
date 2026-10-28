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
  -R 'UCPosixWriteDeadlineTest'
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

The fixture releases held I/O before joining waiters, including after a failed
assertion. The CTest timeout also bounds failures that prevent native teardown.
The filesystem used for test output must support Linux AIO and direct I/O.
