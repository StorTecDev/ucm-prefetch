# Pipeline Store

需要让 KV 块在推理进程重启后继续从本地磁盘或挂载文件系统复用，同时将近期访问的块保存在主机内存中时，可以选择 `Cache|Posix`。`Cache` 负责设备与主机之间的数据搬运及主机缓冲区，`Posix` 负责文件、查询和文件系统 I/O；设备上的 KV Cache 仍由推理引擎管理。

通过 `UcmPipelineStore` 选择已注册的管线名称，完整可选值见[配置参考](../../reference/config-parameters.md)。

## 配置 Cache 与 Posix

先从[安装页面](../../user-guide/quick_start/index.md)选择制品，或在目标引擎环境中[从源码构建](../build_from_source.md)。这条管线需要 `ucmpipelinestore` 扩展、`libcachestore.so` 和 `libposixstore.so`。仅能导入 Python 包，还不能说明所选管线能够初始化。

准备独立且可写的目录，并将其挂载到需要访问它的进程中。将下面的配置保存为 UCM YAML 文件：

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

按 [vLLM 快速开始](../../user-guide/quick_start/index.md#vllm)或 [Ascend 快速开始](../../user-guide/quick_start/index.md#vllm-ascend)，通过 `kv_connector_extra_config.UCM_CONFIG_FILE` 指向该文件。Connector 会提供设备 ID、块大小和张量布局，不要复制其他模型 Store 测试中的这些数值。

示例沿用原配置的 `io_direct: false`，不额外覆盖 I/O 引擎。原生 Cache 和 Posix 阶段的 `io_direct` 默认值为 `true`。只有在设置 `io_direct: true`，且文件系统支持对应的对齐 I/O 时，才切换到 `posix_io_engine: aio`。`timeout_ms` 应放在 `ucm_connector_config` 内，它是 Store 任务超时，不是 HTTP 请求超时。

## 规划主机内存 { #budget-host-memory }

示例中的容量字段保持注释状态，因此使用运行时默认值；注释中的 256 不是对所有模型生效的固定容量。Cache Store 至少需要容纳 `max(1024, 2 * cache_load_exclusive_buffer_number)` 个 shard，独占缓冲数量默认是 1024。如果容量太小，初始化错误会给出至少需要多少 GiB。

| 未显式设置容量时的分配路径 | 实际默认值 |
| --- | --- |
| 原生 Cache，启用共享缓冲 | 256 GiB |
| 原生 Cache，`share_buffer_enable: false` | 每个 worker 32 GiB |
| 当前 vLLM Connector，启用共享缓冲 | 128 GiB |

vLLM Connector 根据模型是否使用 MLA 决定 `share_buffer_enable` 的默认值。正数 `cache_buffer_capacity_gb` 会覆盖原生默认值。共享缓冲需要足够的 `/dev/shm`；非共享 worker 分别分配内存，因此要按同一主机上的 worker 数量汇总预算。模型权重、设备 KV Cache 和其他主机内存开销需要另外计入。

## 存储容量与健康检查

`posix_capacity_gb: 0` 表示不启用基于容量的垃圾回收。正数值提供容量预算，但不会预留文件系统空间。vLLM Connector 将 Posix GC 交给 DP0 scheduler。多个推理实例共用目录时，需要统计它们共同写入的数据；没有验证其他回收归属方案之前，应保留 Posix 的协调设置。

管线健康检查默认开启，检查间隔为 10 秒、超时为 3 秒、窗口为 8 个样本、失败阈值为 2。Posix 探针实际执行写入、读取、比较和删除。探针成功说明文件系统健康，只有请求级命中和完成的加载才能说明 KV 被复用。详见[健康指标](../../user-guide/observability/health-metrics.md)。

## 其他已注册管线

| 需求 | 选择 |
| --- | --- |
| 保留已有 vLLM NFS Connector 配置 | [NFS Store](nfs.md) |
| 使用 3FS 客户端 I/O | [`Cache|Ds3fs`](ds3fs.md) |
| 缩小 BF16 存储载荷，并接受精度权衡 | [`Cache|Compress|Posix`](compress.md) |
| 使用共享 Mooncake 内存，可选文件后端 | [`Mooncake` 或 `Mooncake|Posix`](mooncake.md) |

SGLang 已经管理主机缓存，因此其适配器直接选择 `Posix` 阶段。请使用 [SGLang 快速开始](../../user-guide/quick_start/index.md#sglang)，不要直接套用 vLLM YAML。

## 原服务启动与请求命令

在匹配的引擎环境中将上面的 YAML 保存为 `/etc/ucm/ucm.yaml`，使用下面原 Qwen2.5-14B-Instruct 启动参数。`--no-enable-prefix-caching` 为原存储测试保留；正常部署需要同时使用原生前缀缓存时移除此参数。

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

服务就绪后，在另一终端运行原请求负载。这里只恢复测试输入，不表示复现了原性能结果：

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

## 验证写入与读取

按[公共验证步骤](../../user-guide/observability/verify-cache.md)完成保存、重启和重放。文件系统路径还需确认 Posix 读取及相关错误；如果加载由主机缓存满足，可以在专用诊断运行中设置 `cache_load_backend_only: true`，检查后端加载路径。


## 测量存储收益 { #historical-performance-report }

分别比较无缓存 prefill、存储重放和内存热缓存重放。保持模型与输入集相同，记录外部命中 token、Posix 读取字节数、主机内存、TTFT 和吞吐，判断当前负载下存储 I/O 是否比重新计算更划算。

[实现与扩展说明](../extending-store.md#backend-entrypoints)。

## L2 整块读取与逐层设备搬运 { #full-block-l2-reads-with-layerwise-device-transfers }

整块读取和紧凑 L2 布局是 Early 管线的可选功能。启用 `enable_early_prefetch`，并在
`ucm_connector_config` 中设置 `cache_full_block_load: true`。普通 Cache/Posix 管线
会拒绝这些选项，其均匀文件布局和公开默认值保持不变。启用整块读取时，Early 的冷加载回退
也通过整块读取完成。

将以下设置合入已有配置，并保留其存储、容量和设备设置：

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

`cache_full_block_load` 默认为 `false`。没有配置紧凑布局时，关闭它可恢复逐层 shard
读取。它不会为 layerwise connector 启用请求级异步加载。整块读取要求使用 Early Posix
AIO 后端，每个 block 的层数不超过 `IOV_MAX`，均匀主机 block 不超过 `UINT32_MAX` 字节。

默认逐层 L1 模式通过一次 `IOCB_CMD_PREADV` 直接填充缺失层的缓冲区。已缓存或由其他任务持有写入所有权的层，
使用私有 scratch 接收对应数据，避免覆盖活跃内容。H2D 仍只搬运请求的当前层；L1 节点、
张量偏移和淘汰单位保持逐层组织。没有启用紧凑布局时，磁盘 padding 和 dump 格式也保持不变。

### 整块 L1 与逐层 H2D { #whole-block-l1-with-layerwise-h2d }

同时设置 `cache_full_block_l1: true` 与 `cache_full_block_load: true`，可让每个 block
使用一个连续 L1 缓冲区，并发起一次 `IOCB_CMD_PREAD`。继续保留 `use_layerwise: true`：
H2D 按原有逐层队列、事件和张量偏移消费该整块内存的各层视图，不增加中间整块缓冲区或 CPU 拷贝。

该选项默认关闭，仅用于 Early 整块读取路径。启用 packed L2 时，L1 使用相同的层长度前缀和
与物理 block 大小；每层内部张量偏移不变。任一层的 H2D 引用都会保留整个 native 节点，
预取 pin 和淘汰以整块为单位。原有逐层 D2H/dump 独立填充各层视图，全部层完成后才发布整块
L1 命中。整块读取不会覆盖仍在工作的逐层写入者。

`cache_buffer_capacity_gb` 仍限制物理 L1 字节数。该模式下
`cache_load_exclusive_buffer_number` 按整块节点计数，预留容量与 owner 资格也按物理整块计算。
native domain 分别报告物理 `shard_bytes`/`block_bytes` 和逻辑消费布局
`consumer_shard_bytes`/`consumer_block_bytes`。Early 的 L2 shard 统计每 block 计一个物理
读取单元，H2D 统计仍按层计数。

切换此选项需要重编所有 native 库并重启共享域。整块 L1 使用共享布局版本 4 和 owner 协议
版本 10，原有逐层 L1 保持共享布局版本 3。`test/block_l1_probe.cc` 覆盖真实 PREAD、packed
层视图、逐层 H2D 顺序、逐层回存、取消、短读和共享布局兼容性。

### 紧凑 L2 布局 { #packed-l2-layout }

`posix_packed_layer_bytes` 是 `ucm_connector_config` 内的可选整数列表，按层顺序指定
每层的物理磁盘长度。默认 L1 保留均匀逐层槽位，整块 L1 使用上述 packed 层偏移。
不设置时使用均匀磁盘布局；系统不会自动
根据模型生成这个列表。

在 Early `Cache|Posix` 管线中，紧凑布局必须满足以下条件：

- `cache_full_block_load: true`、`use_layerwise: true`，且每个 chunk 只有一个 KV
  block，即 `blocks_per_chunk == 1`。
- `store_io_alignment: 4096`（默认值），每项为正数且是 4096 的倍数；列表最多 1024
  项且不超过 `IOV_MAX`，总和不超过 `UINT32_MAX` 字节。
- 每个实际层对应一项，且不超过该层的 L1 shard 大小。长度必须等于该层最后一个非空张量
  的结束偏移向上对齐到 4096 字节；保留其前面的张量槽位和间隙，只移除未使用的尾部 padding。
- 保持既有均匀 L1 张量大小列表和偏移。运行时会拒绝可能截断真实张量的布局及 load/dump 地址。
- 关闭 `cache_load_backend_only`、`cache_io_aggregation`、`cache_sdma_direct` 和 `use_gdr`。

磁盘层偏移按长度前缀和计算。文件放入独立的 `.ucm-packed-v1-<count>` 命名空间，其下目录名
编码完整的逐层长度列表，不同布局不会相互复用文件。改变列表会选择另一个命名空间，需要
为新布局重新填充缓存数据，不能将均匀文件直接解释成紧凑文件。原有 Early 读租约会保护紧凑
文件，直到最后一个读取者释放租约后才允许删除。

### 容量、完成与升级 { #capacity-completion-and-upgrades }

Early 每次最多提交 64 个读取单元，在途窗口为 128 个读取单元。整块 I/O 预算、成功读取
字节计账和 Posix 容量/GC 使用实际磁盘 block 大小；启用紧凑布局时即长度列表之和。L1
在默认模式下按均匀逐层 shard 槽位计费；整块 L1 按实际物理 block 大小计费，
配置的 L1 总字节容量保持不变。

读取计账包括读入 scratch 后丢弃的范围。部分命中可能增加读取字节和临时内存，容量压力
或并发所有权可能导致重复读取。一次 AIO 请求不保证一次物理设备命令。已提交读取会保留
缓冲区和向量元数据直到物理 I/O 完成，取消或超时也遵循这一规则。短读会失败，不会把未完整
填充的节点发布为就绪状态。

使用 `posix_load_read_ops_total`、`posix_load_read_bytes_total` 和
`posix_load_full_block_ops_total` 观察成功读取，通过字节与操作数增量之比判断实际读取粒度。

升级时统一重编全部原生库，并重启共享 Early 域内的 worker。Store 能力接口、域元数据和
owner 协议均有变化，不得混用新旧库或 worker。通用测试位于
`test/test_early_packed_layout.py`、`test/full_block_l2_probe.cc`、
`test/packed_l2_probe.cc` 和 `test/early_scope_probe.cc`。

连接器顶层配置 `early_prefetch_request_first: true` 可优先读取最早且可提交的请求；
默认为 `false`，保留原 HRRN 轮转。单个请求内部仍在全局字节窗口内并发读取。
其剩余读取全部提交后，下一个请求可在尾部尚未完成时使用空余额度；查询、激活或
容量等待也允许其他请求补位，等待解除后重新判断优先级。READY 仍要求该请求所有
批次完成。原生读取批量和 L1 到设备的 layerwise 复制保持不变。大请求可能推迟
后来的请求；此选项本身不保证提升设备带宽，也不保证文件在磁盘上物理连续。

已完成 AIO 读取现在会在整个 task 回收前归还其请求字节数对应的 I/O 额度。
该计数只用于读取准入；pin、lease、错误状态与 READY 仍按原有 Check/Wait
完成边界处理，task 仍按 FIFO 回收。短读或提交前的终止失败会退额度，但不表示数据成功。

连接器可设置 `early_prefetch_min_submit_reads: 16` 与
`early_prefetch_submit_max_wait_ms: 2` 对补充提交合批。空余额度不足阈值时继续
轮询完成，到期后的下一次执行器调度即提交；没有在途 I/O 时立即填充。
阈值自动限制在全局窗口以内，小 owner 配额不会因此死锁。默认最小提交数为 1
（不合批）；允许 1–64 块及 (0, 10] ms。期限是调度期限，并非操作系统抢占下的
实时保证。即使 I/O 额度已归零，仍需及时回收尚未结束的原生任务并保留其引用。

### 后台 AIO 写入窗口 { #bounded-background-aio-writes }

在 `ucm_connector_config` 内设置 `posix_write_inflight_limit`，可限制从出队、打开
文件之前到物理 I/O 完成或打开/提交终止失败期间的并发写操作数。默认 `0` 保留共享
FIFO；1–4096 要求 `posix_io_engine: aio`。额度按 I/O 操作数计算，不按字节或整个
回存任务计算，作用范围是当前 Posix AIO 引擎。

启用后，读取使用独立队列，可绕过等待写入额度的任务。所有 open worker 合计每取出
最多 16 个读取任务，就会选择一个具有可用额度的写任务。这是出队公平性，不是耗时
保证。已接受写入的额度保留到物理完成，包括无法取消的 I/O。取消排队任务不消耗
额度；打开、提交或短写失败会归还额度，但不会报告成功持久化。pin、读取 READY、
lease 和写入提交语义保持不变。

该选项可独立于整块 L1、请求优先调度及补充提交合批使用，不改变逐层 H2D。
过小的上限可能降低纯写吞吐，应同时评估端到端延迟、写入积压和排空耗时及读取
吞吐。`test/write_window_probe.cc` 覆盖并发上限、读取绕过、补充及排空、失败和
取消、公平性以及关闭时的 FIFO 行为。
