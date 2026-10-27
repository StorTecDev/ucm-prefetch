# Bounded background write admission

`posix_write_inflight_limit` is an optional native Posix AIO setting. The default
`0` retains the existing shared FIFO. A value from 1 through 4096 limits write
operations from dequeue-before-open until physical completion, or terminal
open/submission failure. It counts operations, not bytes or backend tasks.
Nonzero values require `posix_io_engine=aio`.

With the limit enabled, reads use a separate queue and can bypass capped writes.
An eligible queued write is selected after at most 16 read dequeues across all
open workers. This is service fairness, not a wall-clock deadline: in-flight
writes must complete to make a permit available. The cap also applies when no
reads are queued, so overly small values can reduce write-only throughput.

Write permits are retained by the AIO callback, including uncancellable I/O.
Queued cancellation consumes no permit; an open error, submission failure or
short write releases its permit without claiming successful persistence. Buffer
pins, write commit, read READY, lease ownership and timeout semantics are unchanged.
Waiting for admission does not occupy an open worker and cannot block readers
behind a write-limit condition variable.

Set this option inside `ucm_connector_config`. It can be combined with whole-block
Early reads, request-first scheduling and refill coalescing; none of these options
is required to enable the write limit. Layerwise H2D behavior is unchanged.
Choose the limit using end-to-end latency, write backlog and drain time as well
as read throughput. A smaller write window does not guarantee better performance.

`test/write_window_probe.cc` covers real AIO read bypass, the 16-write cap during
refill/drain, queued cancellation, open/submit/short-write failures, bounded
write service, legacy FIFO and invalid configurations. CPU probes are correctness
evidence and do not establish GPU-workload bandwidth.
