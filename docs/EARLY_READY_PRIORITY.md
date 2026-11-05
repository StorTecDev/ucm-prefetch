# Early READY/PENDING admission in vLLM

The two boolean options below live at the UCM YAML root, alongside
`enable_early_prefetch`, not inside `ucm_connector_config`.

- `early_ready_priority` defaults to `false` and controls READY preference.
- `early_pending_hrrn` independently controls PENDING admission and HRRN selection.
  If omitted, it inherits `early_ready_priority` for compatibility with existing
  configurations. An explicit `false` disables PENDING admission even when READY
  preference is on.

To allow mixed READY/PENDING compute batches, use worker HRRN for prefetch
loading, and disable READY ordering by estimated uncached computation:

```yaml
enable_early_prefetch: true
early_prefetch_request_first: false
early_ready_priority: false
early_pending_hrrn: true
use_layerwise: true
```

These switches control different stages:

- `early_prefetch_request_first: false` selects whole-request HRRN in the
  prefetch worker for L2-to-L1 reads. `true` selects the oldest eligible request.
- `early_pending_hrrn: true` permits eligible PENDING requests to enter compute
  batches and selects them by HRRN. Despite its name, this is not the worker
  HRRN switch. It installs the admission adapter even with READY preference off.
- `early_ready_priority: false` disables READY promotion, uncached-suffix
  ordering and READY bypass fairness. Non-PENDING positions retain native order;
  READY requests are not gathered ahead of every PENDING request.

To disable mixed compute batches while retaining worker HRRN and native queue
order, change only `early_pending_hrrn` to `false` in this configuration. Both
admission options are then off; an already adapted process delegates to the
original native scheduler for this connector. This does not disable unrelated
UCM runtime hooks. Apply the YAML at the UCM root and restart all engine members.

Worker HRRN and PENDING admission HRRN both use queue age and a service-work
estimate. Turning off READY computation ordering does not remove the work term
from HRRN: `R = 1 + W / max(S, minimum_work)` is not shortest-work-first.
`test/test_early_hrrn_modes.py` verifies the combined configuration, worker HRRN
with mixed admission on or off, native READY positions, and stable PENDING gates.

| READY option | PENDING option | Admission policy |
|---|---|---|
| false | omitted / false | Original native queue and PENDING gate |
| false | true | Native non-PENDING order; HRRN in eligible PENDING slots |
| true | omitted / true | READY preference followed by HRRN PENDING backfill |
| true | false | READY preference with the original PENDING gate |

Both EarlyDirectConnector and EarlyLayerWiseConnector support these options on
native vLLM 0.28.0 and 0.30.0 Scheduler/AsyncScheduler. Restart all engine members
after changing them. Early prefetch, worker HRRN and owner resource lifetimes
remain active independently of these admission switches. Non-boolean values,
including strings such as `"true"`, are rejected.

Ascend may export a `BalanceScheduler` wrapper even with DP balancing disabled.
For its verified disabled fast path, admission adapts the original native base
method and preserves the wrapper. Active balance scheduling, a changed delegation
path, or an unknown base class is rejected; no balance option is silently disabled.
The CPU regressions in `test/test_early_ascend_scheduler.py` cover delegation and
fail-closed compatibility checks.

## PENDING-only admission

With READY preference off, FCFS retains skipped-waiting then waiting order.
PRIORITY uses the native comparison of the two queue heads, including
skipped-first ties. The pass preserves every non-PENDING candidate position and
replaces only eligible PENDING positions with the highest HRRN candidates from
the complete waiting pool. It never promotes READY or applies READY bypass
fairness. For example, native `[PENDING_A, READY_B, PENDING_C]` can become
`[PENDING_C, READY_B, PENDING_A]` when C has the higher HRRN.

`early_ready_max_candidates` remains the shared attempt cap (default 64), even
in PENDING-only mode; it does not enable READY preference. The first bounded
native candidate window determines how many PENDING slots are available. The
whole eligible PENDING pool competes for those slots. Non-PENDING requests keep
their native per-request refusal/stop behavior; an optimized PENDING refusal may
yield to the next selected candidate with native rollback preserved. A pass does
not extend beyond the cap, so it can leave unused budget when that window cannot
fill a batch. Global token, running-slot and prefill-deferral exits stay native.

`test/test_early_pending_switch.py` covers independent options, inherited defaults,
explicit disable, native FCFS/PRIORITY ordering, cross-queue PENDING selection,
bounded scans, non-PENDING stopping, PENDING refusal backfill and both scheduler
modes. Worker publication uses the same PENDING option as the scheduler.

## READY preference

When explicitly enabled, Early requests with a protected, nonzero cache prefix receive an opportunity to
enter HBM before failed requests consume the native fallback budget. This policy
applies to the native vLLM 0.28.0/0.30.0 Scheduler and AsyncScheduler when an Early
connector enables it. Other connectors retain the native schedule method.

READY preference can cross native request priorities. Within the preferred
candidates, a smaller estimated uncached suffix is tried first. A zero-hit READY
result follows cache-saving READY candidates and precedes PENDING and fallback. The native prefix-cache and external-match
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

The candidate limit applies when either option is active; the bypass limit
applies only with `early_ready_priority: true`. Their types and ranges are
validated in either mode. Non-boolean switch values, including
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

Connectors with both options disabled do not install the adapter. Because installation patches a
shared scheduler class, an already patched method also checks each connector's
active options and calls the original native method for disabled instances. Enabled and
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

`test/test_early_native_scheduler.py` uses the real supported vLLM scheduler and KV
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

## PENDING backfill

With `early_pending_hrrn` enabled, stable, still-loading PENDING requests can
enter a batch, including a pass with no READY candidates. READY preference, when
separately enabled, tries READY first; PENDING-only mode follows the slot rules
above. Disabling both options retains native order and the original gate.

A PENDING request is eligible only after lookup has finished, the hit prefix is
stable, and every shard of that prefix has acquired its native cache reference.
Reads may still be in flight. The worker publishes this bound separately from
READY; hashing, lookup, activation and capacity waiting alone never authorize
external hits. Each unfinished native batch must have exhausted submission and
must not be waiting for capacity or truncating an incomplete tail. Completed
batches can contribute their final protected prefix.

Among eligible PENDING requests, the highest HRRN response ratio ranks first:
`R = 1 + W / max(S, 1)`. `W` is elapsed monotonic time since the logical Early
request entered PENDING; `S` is its current service-work estimate in L2 shards.
This shares the worker's HRRN formula, which uses byte work units. A common
byte/shard scale does not change ordering within one connector. Service work is
an estimate inside HRRN, never a separate ascending sort key. For example,
`W=100, S=8` gives 13.5 and precedes `W=1, S=1`, whose ratio is 2.

Queue age survives progress notifications and owner retries. Each admission pass
freezes one time snapshot for all candidates. Equal HRRN scores retain original
queue order, without a hidden work or rejection-order tie-breaker.
Cache-saving READY ordering is
unchanged, including existing fairness attempts for other requests. Zero-hit
READY requests are tried before PENDING and fallback requests. An aged
PENDING request still follows READY. Progress is sent only when its hit/work
pair changes; the legacy broadcast path waits for every required rank and uses
the minimum protected hit and the sum of remaining work.

Matching reuses the published prefix and native HBM/missing-block filtering;
it performs no scheduler-side storage lookup, data wait or HBM reservation.
Native direct or layerwise loads acquire their own references to the existing
LOADING buffers and retain their normal readiness waits. The prefix references
keep those buffers alive while background reads continue. PENDING remains a
PENDING state until the worker reports completion; it is never relabeled READY
to gain admission. A batch can therefore include still-loading requests, but
computation must wait wherever its required KV is not yet available. Completion
of the next batch's prefetch is workload-dependent, not guaranteed.

Owner metadata carries the same generation and native consumption plan for
eligible PENDING and READY. Early consumption receipts are retained until READY;
retirement still requires every consumer to quiesce before owner pins return.
After publishing a PENDING prefix, owner failure stops new work without releasing
pins ahead of consumer retirement. Unknown completion still fences the domain.
Restart all engine members together when updating this protocol.

`test/test_early_pending_protocol.py` covers stable-prefix eligibility, in-flight
worker progress, rank aggregation, generation validation, ranking, early
consumption and retirement. The native scheduler and prepared-result tests cover
mixed READY/PENDING batches, all-PENDING passes, both scheduler modes, both Early
connectors, the disabled switch, native budgets and filtered owner metadata.
Additional native regressions cover an unfit READY yielding to a smaller PENDING
and Mamba alignment refusing one candidate. These are correctness checks; they do not establish an accelerator performance
gain. The adapter supports the verified native vLLM 0.28.0 and 0.30.0 waiting-loop
structures. Version 0.30.0 has a separate speculative-padding budget rejection
and two zero-token rejection sites; their per-request exits also permit backfill.
Unrecognized waiting-loop stop conditions still reject installation. Custom
scheduler classes remain unsupported.
