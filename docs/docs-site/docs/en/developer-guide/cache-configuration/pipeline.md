# Pipeline Store

Use `Cache|Posix` when KV blocks should survive a serving-process restart on a
local disk or mounted filesystem, while recently accessed blocks remain in host
memory. `Cache` owns device-to-host transfers and the host buffer; `Posix` owns
files, lookup, and filesystem I/O. The engine still owns its device KV cache.

Select a registered pipeline through `UcmPipelineStore`; see the [configuration reference](../../reference/config-parameters.md) for available names.

## Configure Cache and Posix

Start from [Installation](../../user-guide/quick_start/index.md), or
[build from source](../build_from_source.md) in the target
engine environment. This pipeline requires the `ucmpipelinestore` extension,
`libcachestore.so`, and `libposixstore.so`. A successful package import alone does
not show that the selected pipeline can initialize.

Create a dedicated writable directory, mount it into every process that needs
access, and save this as a UCM YAML file:

```yaml
ucm_connectors:
  - ucm_connector_name: "UcmPipelineStore"
    ucm_connector_config:
      store_pipeline: "Cache|Posix"
      store_health:
        enabled: true
      storage_backends: "./ucm-cache"
      io_direct: false
      # cache_buffer_capacity_gb: 256
      # posix_capacity_gb: 1024
      use_gdr: false
enable_event_sync: true
use_layerwise: true
enable_record_traces: false
use_lite: false
persist_token_threshold: 0
```

Connect the file through `kv_connector_extra_config.UCM_CONFIG_FILE` using the
[vLLM quickstart](../../user-guide/quick_start/index.md#vllm), or its
[Ascend counterpart](../../user-guide/quick_start/index.md#vllm-ascend). The connector
supplies device IDs, block sizes, and tensor layout; do not copy these values
from a different model's Store test.

The sample retains the original `io_direct: false` setting without overriding the I/O engine.
`io_direct` defaults to `true` in the native Cache and Posix stages. Switch to
`posix_io_engine: aio` only with `io_direct: true` and a filesystem that supports
the resulting aligned I/O. Put `timeout_ms` inside `ucm_connector_config`; it is
a Store task timeout, not an HTTP request timeout.

## Budget host memory

The capacity fields remain commented out, so runtime defaults apply. The commented 256 is not the effective capacity for every model.
Cache Store requires enough space for at least
`max(1024, 2 * cache_load_exclusive_buffer_number)` shards; the default exclusive
buffer count is 1024. Initialization reports the minimum required GiB when the
chosen capacity is too small.

| Allocation path without an explicit capacity | Effective default |
| --- | --- |
| Native Cache with shared buffers | 256 GiB |
| Native Cache with `share_buffer_enable: false` | 32 GiB per worker |
| Current vLLM connector with shared buffers | 128 GiB |

The vLLM connector defaults `share_buffer_enable` to whether the model uses MLA.
A positive `cache_buffer_capacity_gb` overrides the native defaults. Shared
buffers need sufficient `/dev/shm`; unshared workers allocate separately, so
multiply their budget by the number of workers on the host. Model weights,
device KV cache, and other host allocations remain additional costs.

## Storage capacity and health

`posix_capacity_gb: 0` disables capacity-driven garbage collection. A positive
value supplies a capacity budget; it does not reserve filesystem space. The
vLLM connector assigns Posix GC to its DP0 scheduler. When multiple serving
instances share a directory, account for their combined data and retain the
Posix coordination settings unless you have validated a different ownership
arrangement.

Pipeline health checking is enabled by default, with a 10-second interval,
3-second timeout, 8-sample window, and failure threshold of 2. Posix implements
a write/read/compare/remove probe. A successful probe establishes filesystem
health; only request-level hits and completed loads establish KV reuse. See
[Health metrics](../../user-guide/observability/health-metrics.md).

## Other registered pipelines

| Requirement | Selection |
| --- | --- |
| Existing vLLM NFS connector configuration | [NFS Store](nfs.md) |
| 3FS client I/O | [`Cache|Ds3fs`](ds3fs.md) |
| Smaller stored BF16 payload, with a quality tradeoff | [`Cache|Compress|Posix`](compress.md) |
| Shared Mooncake memory, optionally backed by files | [`Mooncake` or `Mooncake|Posix`](mooncake.md) |

SGLang already manages a host cache, so its adapter selects the `Posix` stage
directly. Use its [quickstart](../../user-guide/quick_start/index.md#sglang) rather than
passing the vLLM YAML unchanged.

## Original serving and request commands

Save the YAML above as `/etc/ucm/ucm.yaml` in the matching engine environment, then use the original Qwen2.5-14B-Instruct serving parameters below. `--no-enable-prefix-caching` is retained for the original storage test; remove it for normal deployments that also use native prefix caching.

```bash
export ENABLE_UCM_PATCH=1
vllm serve Qwen/Qwen2.5-14B-Instruct \
--max-model-len 20000 \
--tensor-parallel-size 2 \
--gpu_memory_utilization 0.87 \
--block_size 128 \
--trust-remote-code \
--port 7800 \
--enforce-eager \
--no-enable-prefix-caching \
--kv-transfer-config \
'{
    "kv_connector": "UCMConnector",
    "kv_role": "kv_both",
    "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
    "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/etc/ucm/ucm.yaml"}
}'
```

After readiness, run the original request workload from another terminal. These are workload inputs, not reproduced performance results:

```bash
vllm bench serve \
--backend vllm \
--model Qwen/Qwen2.5-14B-Instruct \
--host 127.0.0.1 \
--port 7800 \
--dataset-name random \
--num-prompts 12 \
--random-input-len 16000 \
--random-output-len 2 \
--request-rate inf \
--seed 123456 \
--percentile-metrics "ttft,tpot,itl,e2el" \
--metric-percentiles "90,99" \
--ignore-eos
```

## Verify writes and reads

Use the [shared verification procedure](../../user-guide/observability/verify-cache.md) to save, restart and replay. Check Posix reads and errors for filesystem access. If host cache satisfies the load, a dedicated diagnostic run can use `cache_load_backend_only: true` to inspect the backend path.


## Measure the storage benefit { #historical-performance-report }

Compare uncached prefill, storage replay, and warm-memory replay separately.
Record external-hit tokens, Posix bytes read, host memory, TTFT, and throughput
with the same model and input set. These measurements answer whether storage
I/O is cheaper than recomputation for your workload.

[Implementation and extension](../extending-store.md#backend-entrypoints).

## Full-block L2 reads with layerwise device transfers

Full-block and packed L2 loading are opt-in features of the Early pipeline.
Enable `enable_early_prefetch` and set `cache_full_block_load: true` inside
`ucm_connector_config`. Ordinary Cache/Posix pipelines reject these options;
their uniform file layout and public defaults remain unchanged. Early's cold-load
fallback also uses full-block reads when this option is enabled.

Apply these settings to an existing configuration, retaining its storage,
capacity and device settings:

```yaml
enable_early_prefetch: true
use_layerwise: true
use_request_async_load: false
ucm_connectors:
  - ucm_connector_name: UcmPipelineStore
    ucm_connector_config:
      store_pipeline: "Cache|Posix"
      posix_io_engine: aio
      io_direct: true
      cache_full_block_load: true
```

`cache_full_block_load` defaults to `false`. Without a packed layout, disabling
it restores layer-shard reads. The option does not enable request-level async
loading for layerwise connectors. Full-block reads require the Early Posix AIO
backend, at most `IOV_MAX` layers per block, and a uniform host block no larger
than `UINT32_MAX` bytes.

With the default layer-based L1, one `IOCB_CMD_PREADV` fills missing layer buffers directly. Cached or peer-owned
destinations use private scratch so active data is not overwritten. H2D still
transfers only the requested layer. L1 nodes, tensor offsets and eviction units
remain layer-based. Without packing, disk padding and the dump format also remain
unchanged.

### Whole-block L1 with layerwise H2D

Set `cache_full_block_l1: true` alongside `cache_full_block_load: true` to use
one contiguous L1 allocation and one `IOCB_CMD_PREAD` per block. Keep
`use_layerwise: true`: each H2D task receives a view of its layer within that
allocation, using the existing layerwise queues, events and tensor offsets.
There is no intermediate full-block staging allocation or CPU scatter copy.

The option defaults to `false` and requires the Early full-block path. With a
packed L2 layout, L1 uses the same prefix-sum layer offsets and physical block
length; tensor offsets *within* each layer remain unchanged. Every layer view
retains the entire native node until its H2D completes. Eviction and prefetch
pins operate on complete blocks. The existing layerwise D2H/dump path fills
those views independently and publishes a complete L1 hit only after all layers
are ready. A concurrent whole-block read cannot overwrite an active layer writer.

`cache_buffer_capacity_gb` still limits physical L1 bytes. In this mode
`cache_load_exclusive_buffer_number` counts whole-block nodes; reserved capacity
and owner eligibility use the same physical units. Native domain information
reports physical `shard_bytes`/`block_bytes` separately from
`consumer_shard_bytes`/`consumer_block_bytes`. Early L2 shard statistics count
one physical read unit per block; H2D statistics remain per-layer.

Rebuild all native libraries and restart the shared domain when switching this
option. Whole-block L1 uses shared-layout version 4 and owner protocol version 10;
legacy L1 retains shared-layout version 3. `test/block_l1_probe.cc` checks direct
kernel PREAD, packed layer views, H2D order, per-layer dumps, cancellation,
short reads and shared-memory compatibility.

### Packed L2 layout

`posix_packed_layer_bytes` is an optional integer list inside
`ucm_connector_config`. It specifies the physical disk length of each layer,
in layer order. The default L1 retains uniform layer slots; whole-block L1 uses
the packed offsets described above. Omit it to keep
the uniform disk layout. Packing is not inferred automatically from the model.

For the Early `Cache|Posix` pipeline, packing requires all of the following:

- `cache_full_block_load: true`, `use_layerwise: true`, and one KV block per
  chunk (`blocks_per_chunk == 1`).
- `store_io_alignment: 4096` (the default), positive lengths divisible by 4096,
  at most 1024 entries and no more than `IOV_MAX`, and a sum no larger than
  `UINT32_MAX` bytes.
- One entry per actual layer, with each entry no larger than its L1 shard.
  The length must be the end offset of the last non-null tensor in that layer,
  rounded up to 4096 bytes. Preserve preceding tensor slots and gaps; only unused
  trailing padding may be removed.
- The existing uniform L1 tensor-size list and offsets. Runtime validation rejects
  layouts or load/dump addresses that would truncate a real tensor.
- `cache_load_backend_only`, `cache_io_aggregation`, `cache_sdma_direct`, and
  `use_gdr` disabled.

Files use prefix-sum layer offsets and a separate `.ucm-packed-v1-<count>`
namespace whose nested names encode all layer lengths. Different layouts do not
reuse one another's files. Changing the list selects another namespace; populate
cache data for the new layout instead of treating uniform files as packed files.
Existing Early read leases protect packed files from deletion until the last
reader releases its lease.

### Capacity, completion and upgrades

Early submits up to 64 read units per turn and keeps at most 128 in flight.
Full-block I/O credit, successful-read byte accounting, and Posix capacity/GC use
the actual disk block size: the packed-length sum when packing is enabled.
With `cache_full_block_l1: false`, L1 reservations and capacity use uniform
layer-shard slots. Whole-block L1 instead charges the actual physical block
length while retaining the configured total L1 byte capacity.

Read accounting includes ranges discarded into scratch. Partial hits can increase
bytes read and temporary memory, and capacity pressure or concurrent ownership can
cause repeated reads. One AIO request does not guarantee one physical device command.
Accepted reads retain buffers and vector metadata until physical completion,
including after cancellation or timeout. Short reads fail without publishing
partially filled nodes as ready.

Use `posix_load_read_ops_total`, `posix_load_read_bytes_total`, and
`posix_load_full_block_ops_total` to inspect successful reads. The ratio of byte
and operation deltas identifies the effective read size.

Rebuild all native libraries together and restart workers in a shared Early
domain when upgrading. Store capabilities, domain metadata and the owner protocol
changed; do not mix old and new libraries or workers. Generic coverage is in
`test/test_early_packed_layout.py`, `test/full_block_l2_probe.cc`,
`test/packed_l2_probe.cc`, and `test/early_scope_probe.cc`.

The connector-level `early_prefetch_request_first: true` option prioritizes the
oldest eligible request for L2 reads (default: `false`, existing HRRN rounds).
Reads within that request remain concurrent under the shared byte window. Once
its remaining reads are submitted, another request can fill spare credit while
the tail completes. Lookup, activation and capacity waits also allow backfill;
priority is reconsidered when they clear. READY still requires all request
batches to complete. L1-to-device layerwise copies and native read batching are
unchanged. Large requests can delay younger requests; this option does not
promise greater device bandwidth or physically contiguous files.

Completed AIO reads now return their requested-byte I/O credit before the entire
task is reaped. This counter is only admission accounting: buffer pins, leases,
error status and READY still use the original Check/Wait completion boundary,
and task reaping remains FIFO. A short read or terminal pre-submission error
retires its requested credit without claiming successful data.

The connector can coalesce refills with `early_prefetch_min_submit_reads: 16`
and `early_prefetch_submit_max_wait_ms: 2`. Below the read-unit threshold, it
continues polling completion and flushes at the next executor turn after the
maximum wait. An empty I/O window is filled immediately. The threshold is
clamped to the global window, so a small owner quota cannot deadlock. The
minimum defaults to 1 (no coalescing); supported ranges are 1–64 reads and
(0, 10] milliseconds. The wait is a scheduling deadline, not a real-time bound
under OS descheduling. Completed credit reaching zero does not permit sleeping
through pending native task completion or releasing their references.

### Bounded background AIO writes

Set `posix_write_inflight_limit` inside `ucm_connector_config` to limit concurrent
write operations from dequeue-before-open until physical I/O completion or a
terminal open/submission failure. The default `0` preserves the shared FIFO;
values 1–4096 require `posix_io_engine: aio`. The limit counts I/O operations,
not bytes or whole dump tasks, and applies to the local Posix AIO engine.

When enabled, reads use a separate queue and bypass writes waiting for a permit.
An eligible write is selected after at most 16 read dequeues across the open
workers. This bounds dequeue fairness, not elapsed time. Accepted writes retain
their permits through physical completion, including uncancellable I/O. Queued
cancellation consumes no permit; open, submission and short-write failures
release it without reporting successful persistence. Buffer pins, read READY,
leases and write commit semantics are unchanged.

The option can be used independently of whole-block L1, request-first scheduling
and refill coalescing. It does not change layerwise H2D. Small limits can reduce
write-only throughput; evaluate end-to-end latency and write backlog/drain as
well as read throughput. `test/write_window_probe.cc` covers the cap, read bypass,
refill/drain, failure and cancellation paths, fairness and the disabled FIFO.
