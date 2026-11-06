# Early baseline scheduling

Early uses whole-request HRRN for background L2-to-L1 loading and the native
vLLM queue for compute admission. The two historical mode switches are retained
at the YAML root with `false` defaults:

```yaml
enable_early_prefetch: true
early_prefetch_request_first: false
early_ready_priority: false
```

When Early is enabled, omitted switches are filled with `false`. Explicit `true`
is rejected with a configuration error because the request-first and READY
priority implementations have been removed. Non-boolean values are also rejected.
The switches are inactive when Early is disabled or absent. They do not control
the independent `early_full_block_defaults` storage preset or the numeric Early
scheduling defaults; see [conditional defaults](full_block_writeback.md#conditional-early-defaults).

## Background loading

Each HRRN round snapshots eligible requests and scores each once using its whole
remaining native load, including a conservative estimate for unqueried tails.
The round's order persists across data batches. Later arrivals join later rounds;
lookup, activation or capacity waits allow useful backfill, and earlier rounds
resume first when eligible again. Native reads remain concurrent within the
shared I/O window. A request becomes READY only after all required batches finish.
The owner protocol, capacity protection, retries and completion fences remain in
place. Loading progress never authorizes compute against an unfinished prefix.

## Compute admission

Early no longer rewrites or patches scheduler methods. Native waiting order,
request priorities, HBM allocation refusal and running/batch limits govern
admission for synchronous and asynchronous schedulers. READY requests do not jump
older requests or higher native priorities. PENDING external prefixes continue to
wait for READY, while an already sufficient HBM prefix can cancel redundant
background prefetch and use the existing native matching path.

The former READY scan and bypass settings (`early_ready_max_candidates` and
`early_ready_max_bypasses`) are no longer consumed. They cannot enable priority
admission. No native or platform scheduler inspection is needed for baseline mode.

CPU coverage is in `test/test_early_config.py`,
`test/test_early_ready_priority_switch.py`, `test/test_early_native_scheduler.py`,
`test/test_early_hrrn_modes.py`, and `test/test_early_batch_pipeline.py`.
