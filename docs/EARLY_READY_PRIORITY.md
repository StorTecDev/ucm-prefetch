# Early READY admission in vLLM 0.28.0

Early requests with a protected, nonzero cache prefix receive an opportunity to
enter HBM before failed requests consume the native fallback budget. This policy
applies to the native vLLM 0.28.0 Scheduler and AsyncScheduler when an Early
connector is active. Other connectors retain the native schedule method.

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
- Rejected candidates rotate across passes even when nothing is admitted, so a
  bounded scan cannot permanently revisit only the same blocked requests.

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

Owner admission waiting is described in `EARLY_PREFETCH_DESIGN.md`; it is separate
from HBM admission. READY or scheduling success does not return owner or L1 credits
before the existing consumer and owner-drain protocol completes.
