# Early writeback progress and fallback read leases

The shared owner execution window leaves a bound on the current physical HBM
generation outside the budget of future READY prefetch references. In whole-block
wait-mode writeback, those references cannot be treated as independently
releasable: the model step waiting for a new dump slot must finish before their
requests can execute.

For synchronous layerwise execution with unsharded hash pages (including
replicated PCP), the source-copy fence completes before the next model step.
Dispatch metadata dumps only newly completed pages. If a step schedules at most
T tokens across S sequences and a page has B tokens, the count is at most
floor((T + min(S, T) * (B - 1)) / B), also capped by physical HBM. This includes
unaligned sequence boundaries. With T=8192, S=12 and B=128 the bound is 75 blocks,
not the entire HBM pool. This is calculated once at capability handshake; no
per-layer lock, extra transfer or new scheduling round trip is added.

For other execution/layout contracts the materialized HBM generation remains a
conservative bound. Replicated workers in one DP group contribute one maximum;
distinct DP generations are summed. The buffer remains the native allocator.
This is not additional allocated memory, token-capacity calibration, or a
per-rank division of L1. Older source-drained write jobs may temporarily occupy
this margin but complete through independent L2 I/O. Native load-exclusive slots
are not counted as writable. If ordinary L1 cannot cover the bound, the domain
fails closed. Capability protocol version 13 prevents mixing old handshake rules.

Normal prefetch selection, ordering, I/O submission and READY consumption are
unchanged below the execution-window capacity boundary. Only admission at that
boundary is reduced by the progress bound; its performance effect must still be
measured rather than assumed zero.

The ordinary fallback after an unavailable prefetch also needs a protected
lookup. EarlyLookupWithLease obtains the POSIX read lease before querying the
prefix and keeps it through normal request completion. A deletion that already
holds the exclusive lock makes the query a miss. Taking a lease after querying
would leave a GC race; acquiring a lease without querying does not prove that
the block still exists. Aborted generations retain leases until Store teardown
because cancellation alone does not prove that remote I/O drained.

The READY reservation path continues using its existing native pin and lease
protocol. Source guards, failed-stream quarantine and shared-domain fencing
remain intact. These changes address reproducible safety/progress boundaries;
they do not reconstruct an unrecorded deletion history for a particular failure.

CPU validation covers replicated and independent DP accounting, oversized bounds,
real OFD-lock exclusion and request lifetime, plus a simulator writeback probe
where all future READY references stay held throughout a multi-layer current
batch. No accelerator or trace collector is required by these tests.
