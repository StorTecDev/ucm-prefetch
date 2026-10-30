# Early READY admission in vLLM 0.28.0

`early_ready_priority` is a boolean at the UCM YAML root and defaults to `false`.
With the default, Early uses the original native waiting-queue admission order,
including native FCFS/request priorities and stopping conditions after an HBM
allocation refusal. Set it to `true` to enable the READY admission policy below:

```yaml
enable_early_prefetch: true
early_ready_priority: true
```

Omitting the option or setting `early_ready_priority: false` selects native order.
Restart the engine after changing it. Both EarlyDirectConnector and
EarlyLayerWiseConnector support the switch on vLLM 0.28.0, including its native
Scheduler and AsyncScheduler. The switch changes waiting admission only; Early
prefetch, PENDING/READY/FAILED handling, request-level HRRN loading and owner
resource lifetimes remain active in either mode.

When explicitly enabled, Early requests with a protected, nonzero cache prefix receive an opportunity to
enter HBM before failed requests consume the native fallback budget. This policy
applies to the native vLLM 0.28.0 Scheduler and AsyncScheduler when an Early
connector enables it. Other connectors retain the native schedule method.

READY preference can cross native request priorities. Within the preferred
candidates, a smaller estimated uncached suffix is tried first. A zero-hit READY
result receives no cache preference. The native prefix-cache and external-match
checks remain authoritative; the estimate does not reserve HBM or promise a
particular TTFT.

## Admission and fairness

- `early_ready_max_candidates` defaults to 64 and limits native matching and HBM
  allocation attempts per scheduler pass. Each candidate is tried at most once.
- `early_ready_max_bypasses` defaults to 8. An older request passed by that many
  successful younger admissions receives a priority attempt. Failed allocations,
  resource checks and wakeups do not increment this count.
- A priority attempt that still cannot fit yields to other candidates. The policy
  does not deliberately stop READY admission to reserve space for a large request;
  under continuous load, eventual admission of such a request is not guaranteed.
- Rejected candidates rotate within each readiness class across passes even when
  nothing is admitted. Cross-class rotation is not implemented: if unfit READY
  candidates exhaust every bounded scan, a smaller fallback candidate outside
  that class can remain unvisited. This boundary is deferred from the current scope.

Both limits take effect only with `early_ready_priority: true`; their types and
ranges are still validated when it is false. Non-boolean switch values, including
strings such as `"false"`, are rejected.

Both settings are positive integers at the YAML root. Original request arrival
times remain unchanged. The counters use request objects rather than reusable
request IDs and do not retain completed requests.

Global token/input budgets, running slots, DP prefill deferral, LoRA limits,
structured-output readiness, native prefix-cache semantics and asynchronous KV
loading remain native. A per-request HBM refusal or insufficient per-request
prefill budget permits another candidate. Encoder references touched by a refused
request are released before backfill. A rejected request stays in its original
queue until a later pass; it does not lose its native priority or queue membership.

## Implementation boundary

Disabled connectors do not install the adapter. Because installation patches a
shared scheduler class, an already patched method also checks each connector's
switch and calls the original native method for disabled instances. Enabled and
disabled instances can therefore coexist without sharing admission policy.

The adapter transforms the verified waiting-loop selection and per-request exit
branches of the installed native method. It preserves the rest of that method,
including running-request scheduling and native allocation rollback. An unknown
waiting-loop stop condition is rejected at installation rather than guessed to be
safe to skip. This avoids maintaining a separate copy of the scheduler.

Candidate ranking costs O(N log B), after native queue enumeration; native
priority-queue iteration costs O(N log N), and removing a selected non-head request
can cost O(N). B bounds matching/allocation work, not the complete queue-enumeration
cost. Scheduler overhead therefore needs measurement alongside TTFT and throughput.

## Verification

`test/test_early_native_scheduler.py` uses the real vLLM 0.28.0 scheduler and KV
block allocator with the real UCM Python matching path. Storage results and H2D
transfers are mocked. It covers fallback blocking, a large READY request refusing
HBM, priority crossing, fairness, bounded scans, native budgets, async scheduling,
and encoder rollback. These tests do not establish GPU transfer correctness or a
TTFT improvement.

`test/test_early_ready_priority_switch.py` verifies default and explicit native
order, opt-in uncached-work ordering, both connector constructors, synchronous
and asynchronous scheduling, independent instances in a patched process, native
priority/HBM stopping behavior, PENDING gating, global budgets and configuration
validation. Existing READY-preference regressions explicitly enable the option.

Owner admission waiting is described in `EARLY_PREFETCH_DESIGN.md`; it is separate
from HBM admission. READY or scheduling success does not return owner or L1 credits
before the existing consumer and owner-drain protocol completes.

## Safe execution retry

Only the owner/deferred path can retry `lookup_unavailable`: a typed native
`StoreUnhealthyError` from its first metadata reservation, before any batch has
been published. The query submits no data I/O; temporary native lookup references
are scoped. String-matched errors, later queries, activation, accepted I/O,
unknown failures and shared-domain faults are not eligible.

After that error the scheduler stays PENDING. Every old consumer must quiesce and
the old owner must report drained before another attempt is admitted. New attempts
retain the logical key, candidate hashes and original age, but get a fresh attempt
and monotonically increasing grant. Stale results cannot authorize HBM use or
change the new attempt. Cancel/finish prevents reexecution at every waiting stage.
Failure to establish drain within the independent execution timeout fences the
domain and does not return uncertain credits.

`early_owner_max_retries` defaults to 1; 0 disables new execution attempts.
`early_owner_retry_backoff_ms` defaults to 100. Both are nonnegative integers at the
YAML root. Backoff starts at the classified error; dispatch additionally requires
drain and owner capacity. The budget increments only when a new execution is
actually dispatched. Resource checks, owner/L1 waiting and resuming the original
batch never consume it or become FAILED merely because they wait a long time.
Retry waiting shares the bounded owner admission queue; queue overflow falls back.
Budget exhaustion falls back only after the failed generation drains.

The default permits one cheap recovery attempt and bounds repeated failure work.
It does not wait out the backend's independent health-breaker cooldown (which may
be much longer than 100 ms), so a prolonged outage can exhaust it. Configure
backoff for the deployment; normal capacity contention must not be classified as
this execution error. Release events continue against the live L1 mapping during
backend health failures, without submitting I/O or bypassing domain fencing.

`test/test_early_retry.py` covers retry/drain/backoff ordering, budgets, resource
waiting, cancellation, stale generations and real worker classification.
`test/test_early_batch_binding.py` injects a real backend health failure, checks
typed rejection and release events, restores health, and validates native batch
lifetime and simulated transfers. These checks do not certify accelerator timing.
