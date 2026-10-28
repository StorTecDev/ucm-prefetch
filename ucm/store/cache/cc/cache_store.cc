/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * */
#include <algorithm>
#include <memory>
#include <numeric>
#include "buffer_manager.h"
#include "logger/logger.h"
#include "trans/cuda/gdr/gdr_config.h"
#include "trans_manager.h"
#include "reserved_prefetch.h"
#include "early_native_transfer.h"
#include "block_dump_queue.h"

#ifndef UCM_RUNTIME_ASCEND_IO_AGGREGATION
#define UCM_RUNTIME_ASCEND_IO_AGGREGATION 0
#endif

#ifndef UCM_RUNTIME_ASCEND_SDMA_DIRECT
#define UCM_RUNTIME_ASCEND_SDMA_DIRECT 0
#endif

namespace UC::CacheStore {

class CacheStore : public StoreV1 {
    friend class EarlyCacheStore;

    BufferManager bufferMgr_;
    bool transEnable_{false};
    std::vector<size_t> packedLayerBytes_;
    std::vector<size_t> tensorSizes_;
    TransManager transMgr_;
    std::unique_ptr<BlockDumpQueue> blockDumps_;
    std::unique_ptr<Trans::GdrKVBufferConfig> gpuKvBufferRegistrations_{nullptr};

public:
    Status Setup(const Detail::Dictionary& inConfig) override
    {
        auto config = ParseConfig(inConfig);
        auto s = CheckConfig(config);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed to check config params: {}.", s);
            return s;
        }
        if (config.deviceId >= 0 && !config.gpuKvBufferAddrs.empty()) {
            gpuKvBufferRegistrations_ = std::make_unique<Trans::GdrKVBufferConfig>();
            s = gpuKvBufferRegistrations_->Register(config.gpuKvBufferAddrs,
                                                    config.gpuKvBufferSizes);
            if (s.Failure()) [[unlikely]] {
                UC_ERROR("Failed({}) to register GPU KV buffers.", s);
                return s;
            }
        }
        s = bufferMgr_.Setup(config);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to setup buffer manager.", s);
            return s;
        }
        transEnable_ = config.deviceId >= 0;
        if (transEnable_ && config.storeBackend->PackedLayerBytes(0)) {
            tensorSizes_ = config.tensorSizes;
            for (size_t i = 0; i < config.blockSize / config.shardSize; ++i) {
                packedLayerBytes_.push_back(config.storeBackend->PackedLayerBytes(i));
            }
        }
        if (transEnable_) {
            s = transMgr_.Setup(config, bufferMgr_.GetTransBuffer());
            if (s.Failure()) [[unlikely]] { return s; }
        }
        if (transEnable_ && config.cacheFullBlockDump) {
            blockDumps_ = std::make_unique<BlockDumpQueue>();
            s = blockDumps_->Setup(config, bufferMgr_.GetTransBuffer());
            if (s.Failure()) { return s; }
        }
        ShowConfig(config);
        return Status::OK();
    }
    std::string Readme() const override { return "CacheStore"; }
    Expected<std::vector<uint8_t>> Lookup(const Detail::BlockId* blocks, size_t num) override
    {
        auto res = bufferMgr_.Lookup(blocks, num);
        if (!res) [[unlikely]] { UC_ERROR("Failed({}) to lookup blocks({}).", res.Error(), num); }
        return res;
    }
    Expected<ssize_t> LookupOnPrefix(const Detail::BlockId* blocks, size_t num) override
    {
        auto res = bufferMgr_.LookupOnPrefix(blocks, num);
        if (!res) [[unlikely]] { UC_ERROR("Failed({}) to lookup blocks({}).", res.Error(), num); }
        return res;
    }
    Expected<ssize_t> LookupOnReverse(const Detail::BlockId* blocks, size_t num) override
    {
        auto res = bufferMgr_.LookupOnReverse(blocks, num);
        if (!res) [[unlikely]] { UC_ERROR("Failed({}) to lookup blocks({}).", res.Error(), num); }
        return res;
    }
    void Prefetch(const Detail::BlockId* blocks, size_t num) override
    {
        bufferMgr_.Prefetch(blocks, num);
    }
    Expected<Detail::TaskHandle> Load(Detail::TaskDesc task) override
    {
        if (!transEnable_) { return Status::Error("transfer is not enable"); }
        auto res = transMgr_.Submit({TransTask::Type::LOAD, std::move(task)});
        if (!res) [[unlikely]] {
            UC_ERROR("Failed({}) to submit load task({}).", res.Error(), task.brief);
        }
        return res;
    }
    Expected<Detail::TaskHandle> Dump(Detail::TaskDesc task) override
    {
        if (!transEnable_) { return Status::Error("transfer is not enable"); }
        auto res = transMgr_.Submit({TransTask::Type::DUMP, std::move(task)});
        if (!res) [[unlikely]] {
            UC_ERROR("Failed({}) to submit dump task({}).", res.Error(), task.brief);
        }
        return res;
    }
    Expected<bool> Check(Detail::TaskHandle taskId) override
    {
        auto res = transMgr_.Check(taskId);
        if (!res) [[unlikely]] { UC_ERROR("Failed({}) to check task({}).", res.Error(), taskId); }
        return res;
    }
    Status Wait(Detail::TaskHandle taskId) override
    {
        auto s = transMgr_.Wait(taskId);
        if (s.Failure() && s != Status::StoreUnhealthy()) [[unlikely]] {
            UC_ERROR("Failed({}) to wait task({}).", s, taskId);
        }
        return s;
    }

protected:
    virtual bool EarlyLayoutEnabled() const { return false; }

    Status CheckPackedTask(const Detail::TaskDesc& task) const
    {
        if (packedLayerBytes_.empty()) { return Status::OK(); }
        for (const auto& shard : task) {
            if (shard.index >= packedLayerBytes_.size() || shard.addrs.size() != tensorSizes_.size()) {
                return Status::InvalidParam("invalid packed device layer");
            }
            size_t offset = 0;
            for (size_t i = 0; i < tensorSizes_.size(); ++i) {
                offset += tensorSizes_[i];
                // A real tensor in a truncated tail would silently corrupt KV.
                // Validate both load and dump before acquiring pins or starting I/O.
                if (shard.addrs[i] && offset > packedLayerBytes_[shard.index]) {
                    return Status::InvalidParam("packed layer would truncate a device tensor");
                }
            }
        }
        return Status::OK();
    }

private:
    Config ParseConfig(const Detail::Dictionary& config)
    {
        Config param;
        config.Get("store_backend", param.storeBackend);
        config.Get("unique_id", param.uniqueId);
        config.Get("cache_load_backend_only", param.cacheLoadBackendOnly);
        config.Get("cache_full_block_load", param.cacheFullBlockLoad);
        config.Get("cache_full_block_l1", param.cacheFullBlockL1);
        config.Get("cache_full_block_dump", param.cacheFullBlockDump);
        config.GetNumber("cache_dump_reserved_blocks", param.cacheDumpReservedBlocks);
        config.GetNumber("device_id", param.deviceId);
        size_t tensorSize = 0;
        config.GetNumber("tensor_size", tensorSize);
        config.GetNumber("shard_size", param.shardSize);
        if (tensorSize != 0) {
            param.tensorSizes.assign(param.shardSize / tensorSize, tensorSize);
        } else {
            config.GetNumbers("tensor_size_list", param.tensorSizes);
        }
        config.GetNumber("block_size", param.blockSize);
        config.Get("cpu_affinity_cores", param.cpuAffinityCores);
        if (param.shardSize > 0) { param.waitingQueueDepth *= (param.blockSize / param.shardSize); }
        config.Get("share_buffer_enable", param.shareBufferEnable);
        if (!param.shareBufferEnable) { param.bufferCapacity /= 8; }
        config.Get("io_direct", param.ioDirect);
        size_t bufferCapacityGb = 0;
        config.GetNumber("cache_buffer_capacity_gb", bufferCapacityGb);
        if (bufferCapacityGb != 0) { param.bufferCapacity = bufferCapacityGb << 30; }
        config.GetNumber("waiting_queue_depth", param.waitingQueueDepth);
        config.GetNumber("running_queue_depth", param.runningQueueDepth);
        config.GetNumber("timeout_ms", param.timeoutMs);
        config.GetNumber("cache_stream_number", param.streamNumber);
        config.GetNumber("cache_load_exclusive_buffer_number", param.loadExclusiveBufferNumber);
        config.GetNumbers("gpu_kv_buffer_addrs", param.gpuKvBufferAddrs);
        config.GetNumbers("gpu_kv_buffer_sizes", param.gpuKvBufferSizes);
        config.Get("use_gdr", param.useGdr);
        config.Get("cache_io_aggregation", param.cacheIOAggregation);
        param.cacheIOAggregation = param.cacheIOAggregation && UCM_RUNTIME_ASCEND_IO_AGGREGATION;
        config.Get("cache_sdma_direct", param.cacheSdmaDirect);
        config.GetNumber("local_rank_size", param.localRankSize);
        return param;
    }
    Status CheckSizeConfig(const Config& config)
    {
        if (config.tensorSizes.empty()) { return Status::InvalidParam("invalid tensor size"); }
        if (config.shardSize == 0) { return Status::InvalidParam("invalid shard size"); }
        if (config.blockSize == 0) { return Status::InvalidParam("invalid block size"); }
        if (std::accumulate(config.tensorSizes.begin(), config.tensorSizes.end(), size_t(0)) >
            config.shardSize) {
            return Status::InvalidParam("invalid shard size({})", config.shardSize);
        }
        if (config.blockSize % config.shardSize != 0) {
            return Status::InvalidParam("invalid block size({})", config.blockSize);
        }
        return Status::OK();
    }
    Status CheckConfig(const Config& config)
    {
        if (!config.storeBackend) { return Status::InvalidParam("invalid store backend"); }
        if (!EarlyLayoutEnabled() && (config.cacheFullBlockLoad || config.cacheFullBlockL1 ||
                                      config.storeBackend->PackedLayerBytes(0))) {
            return Status::InvalidParam("full-block and packed layouts require EarlyCache");
        }
        if (config.deviceId >= 0 && config.cacheFullBlockLoad &&
            !config.storeBackend->SupportsFullBlockRead()) {
            return Status::InvalidParam("cache_full_block_load requires a full-block read backend");
        }
        if (config.cacheFullBlockL1 && (!config.cacheFullBlockLoad || config.cacheLoadBackendOnly)) {
            return Status::InvalidParam("whole-block L1 requires the Early full-block read path");
        }
        if (config.cacheFullBlockDump && (!EarlyLayoutEnabled() || !config.cacheFullBlockL1 ||
            !config.storeBackend->SupportsFullBlockWrite() || !config.cacheDumpReservedBlocks ||
            config.useGdr || config.cacheIOAggregation || config.cacheSdmaDirect)) {
            return Status::InvalidParam("block dump requires Early whole L1 / POSIX AIO and normal DMA");
        }
        if (config.deviceId < -1) {
            return Status::InvalidParam("invalid device({})", config.deviceId);
        }
        if (config.uniqueId.empty()) { return Status::InvalidParam("invalid unique id"); }
        auto s =
            Trans::GdrKVBufferConfig::Validate(config.gpuKvBufferAddrs, config.gpuKvBufferSizes);
        if (s.Failure()) { return s; }
        for (const auto core : config.cpuAffinityCores) {
            if (core < 0 || core >= CPU_SETSIZE) {
                return Status::InvalidParam("invalid cpu core({})", core);
            }
        }
        if (config.deviceId == -1) { return Status::OK(); }
        s = CheckSizeConfig(config);
        if (s.Failure()) { return s; }
        if (config.storeBackend->PackedLayerBytes(0)) {
            if (!config.cacheFullBlockLoad || config.cacheLoadBackendOnly ||
                config.cacheIOAggregation || config.cacheSdmaDirect || config.useGdr) {
                return Status::InvalidParam("packed L2 requires the Early full-block cache path");
            }
            size_t total = 0;
            for (size_t i = 0; i < config.blockSize / config.shardSize; ++i) {
                auto bytes = config.storeBackend->PackedLayerBytes(i);
                if (!bytes || bytes > config.shardSize) {
                    return Status::InvalidParam("packed L2 and L1 layer counts disagree");
                }
                total += bytes;
            }
            if (config.storeBackend->PackedLayerBytes(config.blockSize / config.shardSize) ||
                total != config.storeBackend->FullBlockReadBytes()) {
                return Status::InvalidParam("packed L2 block size disagrees with L1 layers");
            }
        }
#if !UCM_RUNTIME_ASCEND_SDMA_DIRECT
        if (config.cacheSdmaDirect) {
            return Status::InvalidParam("Cache SDMA Direct requires RUNTIME_ENVIRONMENT=ascend-a3");
        }
#endif
        const auto nodeBytes = config.cacheFullBlockL1 ? config.storeBackend->FullBlockReadBytes() : config.shardSize;
        if (!nodeBytes) { return Status::InvalidParam("invalid physical L1 node size"); }
        auto bufferNumber = config.bufferCapacity / nodeBytes;
        const size_t minBufferNumber = std::max(size_t(1024), config.loadExclusiveBufferNumber * 2);
        if (bufferNumber < minBufferNumber) {
            const size_t minBufferCapacityGb =
                (minBufferNumber * nodeBytes + (size_t(1) << 30) - 1) >> 30;
            return Status::InvalidParam(
                "too small buffer({}) on shard({}), please set cache_buffer_capacity_gb >= {}GB",
                config.bufferCapacity, nodeBytes, minBufferCapacityGb);
        }
        if (config.waitingQueueDepth <= 1 || config.runningQueueDepth <= 1) {
            return Status::InvalidParam("invalid queue depth({},{})", config.waitingQueueDepth,
                                        config.runningQueueDepth);
        }
        if (config.cacheIOAggregation && config.cacheSdmaDirect) {
            return Status::InvalidParam(
                "Cache IO aggregation is incompatible with Cache SDMA Direct");
        }
        if (config.streamNumber < 1 || config.streamNumber > 32) {
            return Status::InvalidParam("invalid stream number({})", config.streamNumber);
        }
        if (config.localRankSize == 0) {
            return Status::InvalidParam("invalid local rank size({})", config.localRankSize);
        }
        return Status::OK();
    }
    void ShowConfig(const Config& config)
    {
        constexpr const char* ns = "CacheStore";
        std::string buildType = UCM_BUILD_TYPE;
        if (buildType.empty()) { buildType = "Release"; }
        UC_INFO("{}-{}({}).", ns, UCM_COMMIT_ID, buildType);
        UC_INFO("Set {}::StoreBackend to {}.", ns, config.storeBackend->Readme());
        UC_INFO("Set {}::UniqueId to {}.", ns, config.uniqueId);
        UC_INFO("Set {}::CacheLoadBackendOnly to {}.", ns, config.cacheLoadBackendOnly);
        UC_INFO("Set {}::DeviceId to {}.", ns, config.deviceId);
        const auto& v = config.tensorSizes;
        if (v.empty()) {
            UC_INFO("Set {}::TensorSizes to [].", ns);
        } else if (std::all_of(v.begin(), v.end(), [&](auto d) { return d == v[0]; })) {
            UC_INFO("Set {}::TensorSizes to {}(*{}).", ns, v[0], v.size());
        } else {
            UC_INFO("Set {}::TensorSizes to {}.", ns, v);
        }
        UC_INFO("Set {}::ShardSize to {}.", ns, config.shardSize);
        UC_INFO("Set {}::BlockSize to {}.", ns, config.blockSize);
        UC_INFO("Set {}::CacheFullBlockL1 to {}.", ns, config.cacheFullBlockL1);
        UC_INFO("Set {}::IoDirect to {}.", ns, config.ioDirect);
        UC_INFO("Set {}::CpuAffinityCores to {}.", ns, config.cpuAffinityCores);
        UC_INFO("Set {}::BufferCapacity to {}GB.", ns, config.bufferCapacity >> 30);
        UC_INFO("Set {}::ShareBufferEnable to {}.", ns, config.shareBufferEnable);
        UC_INFO("Set {}::CacheIOAggregation to {}.", ns, config.cacheIOAggregation);
        UC_INFO("Set {}::CacheFullBlockDump to {}.", ns, config.cacheFullBlockDump);
        UC_INFO("Set {}::CacheDumpReservedBlocks to {}.", ns, config.cacheDumpReservedBlocks);
        if (config.cacheIOAggregation) {
            UC_INFO("Set {}::AggregationObject to CacheStoreShard.", ns);
        }
        UC_INFO("Set {}::WaitingQueueDepth to {}.", ns, config.waitingQueueDepth);
        UC_INFO("Set {}::RunningQueueDepth to {}.", ns, config.runningQueueDepth);
        UC_INFO("Set {}::TimeoutMs to {}.", ns, config.timeoutMs);
        if (config.cacheSdmaDirect) {
            UC_INFO(
                "Set {}::StreamNumber to {} (configured={}, Cache SDMA Direct uses one stream).",
                ns, config.EffectiveStreamNumber(), config.streamNumber);
        } else {
            UC_INFO("Set {}::StreamNumber to {}.", ns, config.EffectiveStreamNumber());
        }
        UC_INFO("Set {}::CacheSdmaDirect to {}.", ns, config.cacheSdmaDirect);
        UC_INFO("Set {}::LoadExclusiveBufferNumber to {}.", ns, config.loadExclusiveBufferNumber);
        UC_INFO("Set {}::GpuKvBufferNumber to {}.", ns, config.gpuKvBufferAddrs.size());
        UC_INFO("Set {}::UseGdr to {}.", ns, config.useGdr);
        UC_INFO("Set {}::LocalRankSize to {}.", ns, config.localRankSize);
        UC_INFO("Set {}::CacheFullBlockLoad to {}.", ns, config.cacheFullBlockLoad);
    }
};

// Keep the native store's initialization, transfer queues and buffer ownership.
// Reservations are destroyed before the base store releases those resources.
class EarlyCacheStore final : public CacheStore, public Early::DomainProvider {
    std::unique_ptr<ReservedPrefetch> early_;
    std::unique_ptr<EarlyNativeTransfer> transfers_;
    size_t earlyBlockBytes_{0}, earlyShardBytes_{0}, diskBlockBytes_{0};
    size_t consumerBlockBytes_{0}, consumerShardBytes_{0};
    bool fullBlockL1_{false};

protected:
    bool EarlyLayoutEnabled() const override { return true; }

public:
    Status Setup(const Detail::Dictionary& config) override
    {
        auto status = CacheStore::Setup(config);
        if (status.Failure()) { return status; }

        const auto param = ParseConfig(config);
        earlyBlockBytes_ = consumerBlockBytes_ = param.blockSize;
        earlyShardBytes_ = consumerShardBytes_ = param.shardSize;
        fullBlockL1_ = param.cacheFullBlockL1;
        diskBlockBytes_ = param.storeBackend->FullBlockReadBytes();
        if (!diskBlockBytes_) { diskBlockBytes_ = earlyBlockBytes_; }
        if (param.cacheFullBlockL1) { earlyBlockBytes_ = earlyShardBytes_ = diskBlockBytes_; }
        if (param.deviceId >= 0 && !param.cacheLoadBackendOnly) {
            early_ = std::make_unique<ReservedPrefetch>(
                &bufferMgr_, param.storeBackend, earlyBlockBytes_, earlyShardBytes_,
                param.timeoutMs, param.cacheFullBlockLoad);
            transfers_ = std::make_unique<EarlyNativeTransfer>(*param.storeBackend, transMgr_, param);
        }
        return Status::OK();
    }

    Early::Domain DescribeDomain() override
    {
        if (!early_) { return {}; }
        auto* buffer = bufferMgr_.GetTransBuffer();
        if (!buffer) { return {}; }
        auto result = EarlyBufferPins::DescribeDomain(*buffer);
        result.blockBytes = earlyBlockBytes_;
        result.shardBytes = earlyShardBytes_;
        result.diskBlockBytes = diskBlockBytes_;
        result.consumerBlockBytes = consumerBlockBytes_;
        result.consumerShardBytes = consumerShardBytes_;
        result.fullBlockL1 = fullBlockL1_;
        return result;
    }

    Expected<PrefixReservation> ReserveDeferred(const Detail::BlockId* blocks, size_t num) override
    {
        if (!early_) { return Status::Error("Early reservation disabled"); }
        return early_->Reserve(blocks, num, true);
    }
    uint32_t ReleaseEpoch() override
    { return EarlyBufferPins::ReleaseEpoch(*bufferMgr_.GetTransBuffer()); }
    uint32_t WaitRelease(uint32_t epoch, unsigned milliseconds) override
    { return EarlyBufferPins::WaitRelease(*bufferMgr_.GetTransBuffer(), epoch, milliseconds); }

    Expected<PrefixReservation> ReservePrefix(const Detail::BlockId* blocks, size_t num) override
    {
        if (!early_) { return Status::Error("Early reservation disabled"); }
        // EarlyWorker already publishes bounded candidate batches. Avoid
        // splitting a confirmed batch into another series of L2 round trips.
        return early_->Reserve(blocks, num);
    }
    Expected<PrefetchResult> PrefetchReserved(uint64_t handle) override
    {
        if (!early_) { return Status::Error("Early reservation disabled"); }
        return early_->Prefetch(handle);
    }
    void ReleaseReservation(uint64_t handle) override
    { if (early_) { early_->Release(handle); } }

    Expected<Detail::TaskHandle> Load(Detail::TaskDesc task) override
    {
        if (!early_) { return CacheStore::Load(std::move(task)); }
        return SubmitBounded(TransTask::Type::LOAD, std::move(task));
    }
    Expected<Detail::TaskHandle> Dump(Detail::TaskDesc task) override
    {
        if (blockDumps_) {
            auto status = CheckPackedTask(task);
            if (status.Failure()) { return status; }
            for (const auto& shard : task) {
                if (shard.index >= bufferMgr_.GetTransBuffer()->LayerCount()) {
                    return Status::InvalidParam("invalid block dump layer");
                }
            }
            return blockDumps_->Submit({TransTask::Type::DUMP, std::move(task)});
        }
        if (task.finalizeBlockDump) { return Status::InvalidParam("block dump is disabled"); }
        if (!early_) { return CacheStore::Dump(std::move(task)); }
        return SubmitBounded(TransTask::Type::DUMP, std::move(task));
    }

    Expected<bool> Check(Detail::TaskHandle task) override
    {
        if (blockDumps_ && blockDumps_->Owns(task)) { return blockDumps_->Check(task); }
        return transfers_ ? transfers_->Check(task) : CacheStore::Check(task);
    }
    Status Wait(Detail::TaskHandle task) override
    {
        if (blockDumps_ && blockDumps_->Owns(task)) { return blockDumps_->Wait(task); }
        return transfers_ ? transfers_->Wait(task) : CacheStore::Wait(task);
    }

    Status WaitDumpSources(Detail::TaskHandle task) override
    {
        return blockDumps_ ? blockDumps_->WaitSources(task) :
            Status::Unsupported();
    }

private:
    Expected<Detail::TaskHandle> SubmitBounded(TransTask::Type type, Detail::TaskDesc task)
    {
        auto status = CheckPackedTask(task);
        if (status.Failure()) { return status; }
        TransTask prepared{type, std::move(task)};
        if (type == TransTask::Type::LOAD && !prepared.desc.loadReservations.empty()) {
            prepared.loadHandles = early_->LoadHandles(prepared.desc);
        }
        prepared.loadHandles.resize(prepared.desc.size());
        auto& buffer = *bufferMgr_.GetTransBuffer();
        for (size_t i = 0; i < prepared.desc.size(); ++i) {
            if (prepared.loadHandles[i]) { continue; }
            const auto& shard = prepared.desc[i];
            prepared.loadHandles[i] = EarlyBufferPins::Acquire(
                buffer, shard.owner, shard.index, type == TransTask::Type::LOAD);
            if (!prepared.loadHandles[i]) {
                // Nothing has entered a native queue or touched an I/O stream.
                // EarlyTaskPins rolls back only allocation owners; shared readers
                // retain their references. Never report this dump as persisted.
                return Status::NoSpace();
            }
        }
        return transfers_->Submit(EarlyNativeTask{std::move(prepared)});
    }

};

}  // namespace UC::CacheStore

extern "C" UC::StoreV1* MakeCacheStore() { return new UC::CacheStore::CacheStore(); }
extern "C" UC::StoreV1* MakeEarlyCacheStore() { return new UC::CacheStore::EarlyCacheStore(); }
