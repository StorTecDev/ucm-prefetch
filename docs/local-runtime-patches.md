# Packaged runtime patches on UCM v0.8.0

`v0.8.0-patch` starts from official UCM v0.8.0 commit
`3e781c1c90f3a8ce4e6453f63c03360dfbcdeb5d`. It contains the common runtime
synchronization fix without adding Early prefetch. The same fix is included in
`codex/ucm-deferred-pin-wait-v0.8.0`, alongside the Early-only query-range fix.

## Sparse MLA layer-load synchronization

Enable the packaged patch before Python imports vLLM:

```bash
export ENABLE_UCM_PATCH=1
export GLM51_LAYERWISE_SYNC_FIX=1
```

Install UCM from the chosen branch using the repository's normal installation
instructions. The existing `ucm_patch.pth` entry loads the patch through
`apply_all_patches()`. When the flag is absent or not exactly `1`, this fix is
not installed. Installation markers prevent wrapping methods twice when a
compatible hook is already installed.

For sparse `MultiHeadLatentAttentionWrapper` instances with an indexer and an
active UCM layerwise connector, the patch waits before the original forward
reads indexer cache data. Each batch resets the set of completed layer waits;
the original main-attention entry's repeated wait is skipped by layer ID. This
also avoids submitting the next layer's load twice. Original computation,
post-attention cache saving, and connector load-failure handling are retained.

The implementation is in
`ucm/integration/vllm/patch/layerwise_sync_patch.py` and is installed through
UCM's normal patch entry point.

The patch targets the NVIDIA sparse MLA path. Applicability to other models,
vLLM versions, and Ascend paths requires separate validation. The offline tests
below do not establish a performance benefit or hardware inference correctness.

## Early-only READY prefix query range

On `codex/ucm-deferred-pin-wait-v0.8.0`, `EarlyPipelineStore.lookup_on_prefix`
counts protected hits from the actual query slice. A waiting-request retry can
resume near the previous hit boundary instead of the HBM boundary; counting
from HBM again overstates the remaining protected hits. The implementation locates
the query in the prepared block list, bounds the result by the protected hit
range and query length, and stops at a mismatched block.

The implementation is in `ucm/store/pipeline/early_connector.py`, with integration
regressions in `test/test_early_ready_reuse.py`. Official v0.8.0 has no
`EarlyPipelineStore`, so this fix is only present on the Early branch.

## Focused offline validation

The synchronization unit tests exercise ordering before the indexer, original
forward/save behavior, repeated waits, batch resets, failed waits, inactive
paths, compatible hook coexistence, and opt-in routing. They use stand-in vLLM
interfaces and do not require model weights or a GPU:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest --confcutdir=test/suites/Unit \
  -q test/suites/Unit/test_layerwise_sync_patch.py \
  test/suites/Unit/test_apply_patch_version.py
```

The Early branch additionally includes offline prefix-range unit tests and
the original integration regressions in `test/test_early_ready_reuse.py`.
The latter require a compatible vLLM installation and built UCM extensions.

For native storage and Python control regression entry points, see
[Cache and POSIX lifecycle regressions](../ucm/store/test/REGRESSIONS.md).
Offline unit coverage does not establish complete end-to-end validation.
