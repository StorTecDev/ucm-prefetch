#ifndef UCM_CACHE_BLOCK_DUMP_QUEUE_H
#define UCM_CACHE_BLOCK_DUMP_QUEUE_H

#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <future>
#include <thread>
#include <unordered_map>
#include "copy_stream.h"
#include "early_buffer_pins.h"
#include "template/task_wrapper.h"
#include "trans_task.h"

namespace UC::CacheStore {

// A write-only coordinator. The read queues and reservation protocol are unchanged.
// The worker serializes D2H and L2 completion; submitters retain HBM through Wait.
class BlockDumpQueue final : public Detail::TaskWrapper<TransTask, Detail::TaskHandle> {
    struct Block {
        TransBuffer::Handle pin;
        std::shared_ptr<void> lease;
        std::vector<bool> persisted;
        bool complete{false};
    };
    using Blocks = std::unordered_map<Detail::BlockId, Block, Detail::BlockIdHasher>;
    Config config_;
    TransBuffer* buffer_{};
    Blocks blocks_;
    std::mutex queueMutex_;
    std::condition_variable wake_;
    std::deque<TaskPair> pending_;
    bool stopping_{false};
    std::thread worker_;

public:
    ~BlockDumpQueue()
    {
        {
            std::lock_guard lock(queueMutex_);
            stopping_ = true;
        }
        wake_.notify_one();
        if (worker_.joinable()) { worker_.join(); }
    }

    Status Setup(const Config& config, TransBuffer* buffer)
    {
        if (!buffer->WholeBlock() || !config.storeBackend->SupportsFullBlockWrite() ||
            config.cacheDumpReservedBlocks == 0) {
            return Status::InvalidParam("block dump requires whole L1, a write backend and a budget");
        }
        config_ = config;
        buffer_ = buffer;
        timeoutMs_ = config.timeoutMs;
        std::promise<Status> started;
        auto ready = started.get_future();
        worker_ = std::thread([this, &started] { Run(started); });
        return ready.get();
    }

    Expected<Detail::TaskHandle> Submit(TransTask task)
    {
        if (task.type != TransTask::Type::DUMP || task.desc.fullBlockRead ||
            task.desc.fullBlockWrite || task.desc.deferWriteCommit) {
            return Status::InvalidParam("invalid cache block dump options");
        }
        for (const auto& shard : task.desc) {
            if (shard.index >= buffer_->LayerCount() ||
                shard.addrs.size() != config_.tensorSizes.size()) {
                return Status::InvalidParam("invalid cache block dump sources");
            }
            auto bytes = config_.storeBackend->PackedLayerBytes(shard.index);
            if (!bytes) { bytes = config_.shardSize; }
            size_t end = 0;
            for (size_t i = 0; i < shard.addrs.size(); ++i) {
                end += config_.tensorSizes[i];
                if (shard.addrs[i] && end > bytes) {
                    return Status::InvalidParam("block dump would truncate a tensor");
                }
            }
        }
        return Detail::TaskWrapper<TransTask, Detail::TaskHandle>::Submit(std::move(task));
    }

    bool Owns(Detail::TaskHandle id)
    {
        std::shared_lock lock(mutex_);
        return tasks_.count(id) != 0;
    }

private:
    Status FailureStatus(const TaskPtr& task) const override { return task->FailureStatus(); }

    void Dispatch(TaskPtr task, WaiterPtr waiter) override
    {
        waiter->Set(1);
        {
            std::lock_guard lock(queueMutex_);
            if (!stopping_ && pending_.size() < config_.waitingQueueDepth) {
                pending_.emplace_back(task, waiter);
                wake_.notify_one();
                return;
            }
        }
        task->Fail(Status::NoSpace());
        failureSet_.Insert(task->id);
        waiter->Done();
    }

    void Run(std::promise<Status>& started)
    {
        CopyStream stream;
        auto status = stream.Setup(config_.deviceId, config_.EffectiveStreamNumber(), false);
        started.set_value(status);
        if (status.Failure()) { return; }
        for (;;) {
            TaskPair pair;
            {
                std::unique_lock lock(queueMutex_);
                wake_.wait(lock, [&] { return stopping_ || !pending_.empty(); });
                if (pending_.empty()) { break; }
                pair = std::move(pending_.front());
                pending_.pop_front();
            }
            auto& task = pair.first;
            // Even a timed-out caller must retain its HBM until this worker drains.
            status = Process(stream, task->desc);
            if (status.Failure()) {
                task->Fail(status);
                failureSet_.Insert(task->id);
            }
            pair.second->Done();
        }
        // On shutdown, preserve available layers as unpublished L2 data. Every
        // accepted backend task is waited before releasing its L1 source/lease.
        for (auto& [id, block] : blocks_) {
            if (!block.complete) {
                auto saved = Spill(id, block);
                if (saved.Failure()) { UC_ERROR("Block dump shutdown spill failed: {}", saved); }
            }
        }
    }

    Status Write(Detail::TaskDesc desc)
    {
        desc.deferWriteCommit = true;
        auto task = config_.storeBackend->Dump(std::move(desc));
        if (!task) { return task.Error(); }
        // POSIX Wait drains cancellations as well as successful I/O. Thus a
        // released pin can never still be referenced by the kernel.
        return config_.storeBackend->Wait(task.Value());
    }

    Status Spill(const Detail::BlockId& id, Block& block)
    {
        if (!block.pin) { return Status::OK(); }
        Detail::TaskDesc desc;
        std::vector<size_t> layers;
        for (size_t layer = 0; layer < buffer_->LayerCount(); ++layer) {
            auto view = block.pin.LayerView(layer);
            if (view.Ready() && !block.persisted[layer]) {
                desc.push_back({id, layer, {view.Data()}});
                layers.push_back(layer);
            }
        }
        if (!desc.empty()) {
            auto status = Write(std::move(desc));
            if (status.Failure()) { return status; }
            for (auto layer : layers) { block.persisted[layer] = true; }
        }
        block.pin = {};
        return Status::OK();
    }

    Status MakeRoom(const Detail::BlockId& id)
    {
        size_t pins = 0;
        for (const auto& [key, block] : blocks_) { pins += bool(block.pin); }
        if (pins < config_.cacheDumpReservedBlocks) { return Status::OK(); }
        for (auto& [key, block] : blocks_) {
            if (key != id && block.pin) { return Spill(key, block); }
        }
        return Status::NoSpace();
    }

    Status Fill(CopyStream& stream, const Detail::Shard& shard, Block& block)
    {
        if (block.complete || block.persisted[shard.index]) { return Status::OK(); }
        if (!block.pin) {
            auto status = MakeRoom(shard.owner);
            if (status.Failure()) { return status; }
        }
        auto handle = EarlyBufferPins::Acquire(*buffer_, shard.owner, shard.index, false);
        if (!handle) {
            // A reservation of ours may be the only reclaimable ordinary slot.
            for (auto& [key, other] : blocks_) {
                if (key != shard.owner && other.pin) {
                    auto status = Spill(key, other);
                    if (status.Failure()) { return status; }
                    handle = EarlyBufferPins::Acquire(*buffer_, shard.owner, shard.index, false);
                    if (handle) { break; }
                }
            }
        }
        if (!handle || (!handle.Owner() && !handle.Ready())) { return Status::NoSpace(); }
        // Acquiring a layer reserves a complete physical L1 block before any DMA.
        if (!block.pin) { block.pin = handle.WholeView(); }
        if (handle.Ready()) { return Status::OK(); }
        const auto packed = config_.storeBackend->PackedLayerBytes(shard.index);
        const auto bytes = packed ? packed : config_.shardSize;
        // Null tensor slots and alignment padding must not publish stale bytes.
        std::memset(handle.Data(), 0, bytes);
        auto addresses = shard.addrs;
        auto status = stream.DeviceToHostAsync(addresses.data(), handle.Data(), config_.tensorSizes);
        auto drained = stream.Synchronize();
        if (status.Failure() || drained.Failure()) {
            auto error = status.Failure() ? status : drained;
            handle.MarkFailed(error);
            return error;
        }
        handle.MarkReady();
        return Status::OK();
    }

    Status Publish(const Detail::BlockId& id, Block& block)
    {
        if (block.complete) { return Status::OK(); }
        if (block.pin && block.pin.Ready()) {
            Detail::TaskDesc desc;
            desc.fullBlockWrite = true;
            desc.push_back({id, 0, {block.pin.Data()}});
            auto status = Write(std::move(desc));
            if (status.Failure()) { return status; }
            std::fill(block.persisted.begin(), block.persisted.end(), true);
        }
        if (!std::all_of(block.persisted.begin(), block.persisted.end(), [](bool x) { return x; })) {
            return Status::NotFound();
        }
        auto status = config_.storeBackend->CommitDump(id);
        if (status.Failure()) { return status; }
        block.complete = true;
        block.pin = {};
        return Status::OK();
    }

    Status Process(CopyStream& stream, const Detail::TaskDesc& desc)
    {
        if (desc.prerequisiteHandle) {
            auto status = stream.WaitEvent(Trans::Event{desc.prerequisiteHandle});
            if (status.Failure()) { stream.Synchronize(); return status; }
        }
        std::vector<Detail::BlockId> order;
        std::unordered_map<Detail::BlockId, std::vector<const Detail::Shard*>,
                           Detail::BlockIdHasher> grouped;
        for (const auto& shard : desc) {
            if (!grouped.count(shard.owner)) { order.push_back(shard.owner); }
            grouped[shard.owner].push_back(&shard);
        }
        auto result = Status::OK();
        for (const auto& id : order) {
            auto it = blocks_.find(id);
            if (it == blocks_.end()) {
                if (blocks_.size() >= config_.waitingQueueDepth) {
                    if (desc.finalizeBlockDump) { result = Status::NoSpace(); }
                    continue;
                }
                auto lease = config_.storeBackend->AcquireWriteLease(id);
                if (!lease) {
                    if (desc.finalizeBlockDump || lease.Error() != Status::Retry()) {
                        result = lease.Error();
                    }
                    continue;
                }
                it = blocks_.emplace(id, Block{{}, std::move(lease.Value()),
                                               std::vector<bool>(buffer_->LayerCount()), false}).first;
            }
            auto& block = it->second;
            auto status = Status::OK();
            for (const auto* shard : grouped[id]) {
                status = Fill(stream, *shard, block);
                if (status.Failure()) { break; }
            }
            if (status.Success()) {
                status = Publish(id, block);
                if (status == Status::NotFound()) { status = Status::OK(); }
            }
            if (desc.finalizeBlockDump) {
                // Retrying supplied layers is safe only because the connector
                // keeps this batch's HBM valid and supplied a compute event.
                auto spilled = block.complete ? Status::OK() : Spill(id, block);
                if (status.Success()) { status = spilled; }
                if (status.Success()) { status = Publish(id, block); }
                if (spilled.Success()) { blocks_.erase(it); }
            } else if (status == Status::NoSpace()) {
                // No DMA occurred for the missing layer. Finish supplies it again
                // before the connector allows HBM reuse; never spin on readers.
                status = Status::OK();
            }
            if (status.Failure() && result.Success()) { result = status; }
        }
        auto drained = stream.Synchronize();
        return result.Failure() ? result : drained;
    }
};
}  // namespace UC::CacheStore
#endif
