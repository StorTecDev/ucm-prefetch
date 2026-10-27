# 🌟 PipelineStore

**PipelineStore** is a composite store built by **chaining multiple Store implementations** together to form a data transfer pipeline.  

Currently, the pipeline supports a chain composed of **Cache Store** and **Posix Store**.

In this chained pipeline:
- **Cache Store** handles data transfer between the **Device and Host**.
- Once the data flows from the Device to the Host, **Posix Store** is responsible for transferring the data between the **Host and POSIX-compliant persistent storage**, such as local disks, SSDs, or remote NFS (including NFS over RDMA) mount points.

At present, only this Store chain is supported.  
Additional Store implementations will be developed in the future and **chained** into the pipeline to enable more flexible and extensible transfer paths.


## Performance

### Overview
The following are the multi-concurrency performance test results of UCM in the Prefix Cache scenario under a CUDA environment, showing the performance improvements of UCM.
During the tests, HBM cache was disabled, and KV Cache was retrieved and matched only from SSD. 

Here, Full Compute refers to pure VLLM inference, while SSD80% indicates that after UCM pooling, the SSD hit rate of the KV cache is 80%.

The following table shows the results on the QwQ-32B model(**4 x H100 GPUs**):
|      **QwQ-32B** |                |                      |                |               |
| ---------------: | -------------: | -------------------: | -------------: | :------------ |
| **Input length** | **Concurrent** | **Full Compute (ms)** | **SSD80% (ms)** | **Speedup (%)** |
|            4 000 |              1 |              223.05 |         156.54 | **+42.5%**   |
|            8 000 |              1 |              350.47 |         228.27 | **+53.5%**   |
|           16 000 |              1 |              708.94 |         349.17 | **+103.0%**  |
|           32 000 |              1 |             1512.04 |         635.18 | **+138.0%**  |
|            4 000 |              8 |              908.52 |         625.92 | **+45.1%**   |
|            8 000 |              8 |             1578.72 |         955.25 | **+65.3%**   |
|           16 000 |              8 |             3139.03 |        1647.72 | **+90.5%**   |
|           32 000 |              8 |             6735.25 |        3025.23 | **+122.6%**  |
|            4 000 |             16 |             1509.79 |         919.53 | **+64.2%**   |
|            8 000 |             16 |             2602.34 |        1480.30 | **+75.8%**   |
|           16 000 |             16 |             5732.49 |        2393.54 | **+139.5%**  |
|           32 000 |             16 |            11891.61 |        4790.00 | **+148.3%**  |


The following table shows the results on the DeepSeek-R1-awq model (**8 × H100 GPUs**):
|**DeepSeek-R1-awq**|                |                      |                |               |
| -----------------:| -------------: | -------------------: | -------------: | :------------ |
| **Input length**  | **Concurrent** | **Full Compute (ms)** | **SSD80% (ms)** | **Speedup (%)** |
|             4 000 |              1 |               429.30 |        261.34 | **+64.3%**   |
|             8 000 |              1 |               762.23 |        363.37 | **+109.8%**  |
|            16 000 |              1 |              1426.06 |        586.17 | **+143.3%**  |
|            32 000 |              1 |              3086.85 |       1073.25 | **+187.6%**  |
|             4 000 |              8 |              1823.55 |       1017.72 | **+79.2%**   |
|             8 000 |              8 |              3214.76 |       1511.16 | **+112.7%**  |
|            16 000 |              8 |              6417.81 |       2596.70 | **+147.2%**  |
|            32 000 |              8 |             14278.00 |       5111.67 | **+179.3%**  |
|             4 000 |             16 |              3205.22 |       1534.00 | **+108.9%**  |
|             8 000 |             16 |              5813.09 |       2208.60 | **+163.2%**  |
|            16 000 |             16 |             11752.48 |       4000.46 | **+193.8%**  |
|            32 000 |             16 |             38643.73 |      19910.41 | **+94.1%**   |



## Configuration for Prefix Caching

Modify the UCM configuration file to specify which UCM connector to use and where KV blocks should be stored.  
You may directly edit the example file at:

`unified-cache-management/examples/ucm_config_example.yaml`

A minimal configuration looks like this:

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

### Required Parameters

* **ucm_connector_name**:  
  Specifies `UcmPipelineStore` as the UCM connector.

* **store_pipeline: "Cache|Posix"**  
  Specifies a pipeline built by **chaining the Cache Store and the Posix Store**.  
  In this chained pipeline, the Cache Store handles data transfer between the **Device and Host**,  
  and once the data reaches the Host, the Posix Store transfers it between the **Host and POSIX-compliant persistent storage**.

  The pipeline must be registered in advance in  
  `unified-cache-management/ucm/store/pipeline/connector.py` under `PIPELINE_REGISTRY`.

  Currently, **only this Store chain is supported**.


* **storage_backends**:  
  Directory used for storing KV blocks. Can be a local path or an NFS-mounted path.  
  **⚠️ Replace `"./ucm-cache"` with your actual storage directory.**

### Optional Parameters

Parameters inside `ucm_connector_config`:

* **store_health.enabled** *(optional, default: true)*
  Whether to wrap each Store in the Pipeline with `HealthBreakerStore`.
  Set it to `false` to disable automatic health probing, circuit breaking, and recovery.
  This setting does not control a Breaker explicitly added as a Store by a custom Pipeline implementation.

* **io_direct** (optional, default: `false`):  
  Whether to enable direct I/O. For Posix Store, this bypasses the OS page cache when reading/writing files to disk.

* **cache_buffer_capacity_gb** *(optional, default: 256)*  
  The capacity of the Cache Store host-side buffer in GB.  
  The default value of 256 GB is sufficient in most cases.  
  When using models like MLA, make sure `/dev/shm` has enough space,  
  or set this parameter to a smaller value.  
  Note: when multiple data-parallel instances run on the same node,  
  each creates its own Cache Store buffer.

* **posix_capacity_gb** *(optional, default: 0)*  
  The maximum storage capacity in GB for the Posix Store.  
  When set to a value greater than 0, garbage collection (GC) is enabled  
  and will recycle disk space when the threshold is reached.  
  When set to 0 (default), no capacity limit or GC is applied.

* **posix_io_engine** *(optional, default: "psync")*  
  I/O engine type for the Posix Store. Supported values: `"psync"` (pread/pwrite), `"aio"` (libaio).

* **use_gdr** *(optional, default: false)*  
  Enable GPUDirect RDMA transfers for Cache Store/Pc Store H2D and D2H copies.  
  This requires an RDMA-capable NIC, a working GPUDirect RDMA environment,  
  and building UCM with `ENABLE_GDR=1`.

* **waiting_queue_depth** *(optional, default: 8192)*  
  Depth of the waiting queue for transfer tasks in the Cache Store.  

* **running_queue_depth** *(optional, default: 524288)*  
  Depth of the running queue for transfer tasks in the Cache Store.  

* **timeout_ms** *(optional, default: 30000)*  
  Timeout in milliseconds for external interfaces (applies to both Cache Store and Posix Store).

Top-level parameters (outside `ucm_connector_config`):

* **enable_event_sync** *(optional, default: true)*  
  Whether to enable event synchronization.  
  When using `UcmNfsStore`, this should be set to `false`.

* **use_layerwise** *(optional, default: true)*  
  Whether to use layerwise loading/saving of KV cache blocks.  
  Enabled by default for `UCMConnector`.

* **hit_ratio** *(optional)*  
  When set, limits the number of hit tokens to `hit_ratio × num_prompt_tokens`.  
  Useful for controlling the proportion of cached tokens that are actually loaded.

* **enable_record_traces** *(optional, default: false)*  
  Whether to record request traces.  
  When enabled, logs per-request traces (timestamp, input_length, output_length, hash_ids).  
  Each hash_id takes 32 bytes. Enable only when needed,  
  and configure `UCM_LOG_MAX_FILES` and `UCM_LOG_MAX_SIZE` accordingly.

* **use_lite** *(optional, default: false)*  
  Whether to use the UCM Lite Connector.  
  UCM Lite Connector works with Fake Store to skip actual KV dump/load operations,  
  only collecting hit ratio statistics.  
  It is suggested to first collect hit ratio statistics and confirm with relevant  
  project members whether to adopt UCM.

* **persist_token_threshold** *(optional, default: 0)*  
  Minimum token threshold for KV persistence.  
  Requests with fewer tokens than this threshold will skip KV load and dump operations.

* **metrics_config_path** *(optional)*  
  Path to a UCM metrics configuration YAML file.  
  When set, enables UCM metrics that can be monitored online via Grafana and Prometheus.


## Launching Inference

In this guide, we describe **online inference** using vLLM with the UCM connector, deployed as an OpenAI-compatible server. For best performance with UCM, it is recommended to set `block_size` to 128.

To start the vLLM server with the Qwen/Qwen2.5-14B-Instruct model, run:

```bash
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
    "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/vllm-workspace/unified-cache-management/examples/ucm_config_example.yaml"}
}'
```
You can also use the Layerwise Connector by adding `"use_layerwise": true` in the `UCM_CONFIG_FILE`.
for example:

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

**⚠️ Make sure to replace `"/vllm-workspace/unified-cache-management/examples/ucm_config_example.yaml"` with your actual config file path.**

If you see log as below:

```bash
INFO:     Started server process [1049932]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

Congratulations, you have successfully started the vLLM server with UCM connector!

## Evaluating UCM Prefix Caching Performance
After launching the vLLM server with `UCMConnector` enabled, the easiest way to observe the prefix caching effect is to run the built-in `vllm bench` CLI. Executing the following command **twice** in a separate terminal shows the improvement clearly.

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

### After the first execution
The `vllm bench` terminal prints the benchmark result:

```
---------------Time to First Token----------------
Mean TTFT (ms):                           15001.64
```

Inspecting the vLLM server logs reveals entries like:

```
INFO ucm_connector.py:317: request_id: xxx, total_blocks_num: 125, hit hbm: 0, hit external: 0
```

This indicates that for the first inference request, UCM did not hit any cached KV blocks. As a result, the full 16K-token prefill must be computed, leading to a relatively large TTFT.

### After the second execution
Running the same benchmark again produces:

```
---------------Time to First Token----------------
Mean TTFT (ms):                            2874.21
```

The vLLM server logs now contain similar entries:

```
INFO ucm_connector.py:317: request_id: xxx, total_blocks_num: 125, hit hbm: 0, hit external: 125
```

This indicates that during the second request, UCM successfully retrieved all 125 cached KV blocks from the storage backend. Leveraging the fully cached prefix significantly reduces the initial latency observed by the model, yielding an approximate **5× improvement in TTFT** compared to the initial run.

### Log Message Structure
> If you want to view detailed transfer information, set the environment variable `UC_LOGGER_LEVEL` to `debug`.

You may see the following typical log messages in the logs.

```text
[UC][D] Cache task({task_id},{operation},{subtask_number},{size}) dispatching. [PID,TID]
```
This log indicates that the **Cache Store** has received a **load or dump task**
| Component    | Description                                                                 |
|--------------|-----------------------------------------------------------------------------|
| `task_id`    | Unique identifier for the Cache Ctore task                                              |
| `operation`  | `DUMP`: Dump to Host(Device → Host) <br>`LOAD`: Load from Host (Host → Device) |
| `subtask_number` | Number of subtasks executed in this operation                                  |
| `size`       | Total size of data transferred in bytes (across all tasks)                  |

```text
[UC][D] Cache task({task_id},{operation},{subtask_number},{size}) finished, cost {time}ms. [PID,TID]
```
This log indicates that a load or dump task in the **Cache Store** has completed, along with its execution time **in ms**.

```text
[UC][D] Posix task({task_id},{operation},{subtask_number},{size}) dispatching. [PID,TID]
```
This log indicates that the **Posix Store** has received a **load or dump task**
| Component    | Description                                                                 |
|--------------|-----------------------------------------------------------------------------|
| `task_id`    | Unique identifier for the Posix Store task                                              |
| `operation` | `Cache2Backend`: Dump data from Cache Store to Posix Store.<br>`Backend2Cache`: Load data from Posix Store back to Cache Store. |
| `subtask_number` | Number of subtasks executed in this operation                                  |
| `size`       | Total size of data transferred in bytes (across all tasks)                  |

```text
[UC][D] Posix task({task_id},{operation},{subtask_number},{size}) finished, cost {time}ms. [PID,TID]
```
This log indicates that a load or dump task in the **Posix Store** has completed, along with its execution time in **in ms**.

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
