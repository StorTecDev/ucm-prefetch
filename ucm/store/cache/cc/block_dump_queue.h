#ifndef UCM_CACHE_BLOCK_DUMP_QUEUE_H
#define UCM_CACHE_BLOCK_DUMP_QUEUE_H

#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <future>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include "copy_stream.h"
#include "early_buffer_pins.h"
#include "template/task_wrapper.h"
#include "trans_task.h"

namespace UC::CacheStore {

#ifdef UCM_ENABLE_TEST_HOOKS
namespace TestHooks {
// Set/clear only while no test task is running. A failure models rejected D2H.
inline std::function<Status(const Detail::Shard&)> blockDumpCopyHook;
}
#endif

// One source worker owns block bookkeeping. The independent I/O worker only
// touches pinned, immutable L1 views; it never receives HBM addresses.
class BlockDumpQueue final : public Detail::TaskWrapper<TransTask, Detail::TaskHandle> {
    struct WriteJob {
        Detail::BlockId id;
        Detail::TaskDesc desc;
        TransBuffer::Handle pin;
        std::shared_ptr<void> lease;
        std::vector<size_t> layers;
        bool commit{false};
        bool releasePin{false};
        std::mutex mutex;
        std::condition_variable cv;
        std::atomic_bool done{false};
        Status status{Status::OK()};
    };
    struct Block {
        TransBuffer::Handle pin;
        std::shared_ptr<void> lease;
        std::vector<bool> persisted;
        bool complete{false};
        bool closing{false};
        std::shared_ptr<WriteJob> write;
        Status writeFailure{Status::OK()};
    };
    struct SourceGroup {
        Detail::BlockId id;
        std::vector<const Detail::Shard*> shards;
        bool done{false};
    };
    struct SourceTask {
        TaskPair pair;
        Latch sources;
        Status sourceStatus{Status::OK()};
        std::vector<std::shared_ptr<WriteJob>> writes;
        std::vector<SourceGroup> groups;
        bool prepared{false};
        bool eventQueued{false};
        explicit SourceTask(TaskPair value) : pair(std::move(value)) { sources.Set(1); }
    };
    using Work = std::shared_ptr<SourceTask>;
    using Blocks = std::unordered_map<Detail::BlockId, Block, Detail::BlockIdHasher>;
    Config config_;
    TransBuffer* buffer_{};
    Blocks blocks_;  // source worker only
    // Admission scans only pinned blocks, not all blocks in waiting tasks.
    // unordered_map keeps these node addresses stable across rehashes.
    std::unordered_set<Block*> resident_;
    std::vector<Work> finishing_;  // source worker only
    std::mutex queueMutex_;
    std::condition_variable wake_;
    std::deque<Work> pending_;
    std::unordered_map<Detail::TaskHandle, Work> sources_;
    std::atomic<size_t> admitted_{0};
    std::atomic<size_t> ioEpoch_{0};
    bool stopping_{false};
    std::thread worker_;
    std::mutex ioMutex_;
    std::condition_variable ioWake_;
    std::deque<std::shared_ptr<WriteJob>> ioPending_;
    bool ioStopping_{false};
    std::thread ioWorker_;

public:
    ~BlockDumpQueue()
    {
        { std::lock_guard lock(queueMutex_); stopping_ = true; }
        wake_.notify_one();
        if (worker_.joinable()) { worker_.join(); }
        { std::lock_guard lock(ioMutex_); ioStopping_ = true; }
        ioWake_.notify_one();
        if (ioWorker_.joinable()) { ioWorker_.join(); }
    }

    Status Setup(const Config& config, TransBuffer* buffer)
    {
        if (!buffer->WholeBlock() || !config.storeBackend->SupportsFullBlockWrite() ||
            config.cacheDumpReservedBlocks == 0) {
            return Status::InvalidParam("block dump requires whole L1, a write backend and a budget");
        }
        if (config.cacheDumpCapacityPolicy != "spill" && config.cacheDumpCapacityPolicy != "wait") {
            return Status::InvalidParam("cache_dump_capacity_policy must be spill or wait");
        }
        if (config.cacheDumpCapacityPolicy == "wait" && config.waitingQueueDepth < 2) {
            return Status::InvalidParam("wait policy needs queue space for finalization");
        }
        config_ = config;
        buffer_ = buffer;
        timeoutMs_ = config.timeoutMs;
        ioWorker_ = std::thread([this] { RunIo(); });
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
    { std::shared_lock lock(mutex_); return tasks_.count(id) != 0; }

    // Non-consuming: caller must still Check/Wait for successful cache publication.
    // OK proves HBM is no longer accessed; copy/I/O errors remain on the final task.
    Status WaitSources(Detail::TaskHandle id)
    {
        Work work;
        {
            std::lock_guard lock(queueMutex_);
            auto it = sources_.find(id);
            if (it == sources_.end()) { return Status::NotFound(); }
            work = it->second;
        }
        if (!work->sources.WaitFor(timeoutMs_)) {
            // A deadline reports failure; it does not discard accepted work.
            // Keep queued source addresses/events valid until their copies drain.
            work->pair.first->Fail(Status::Timeout());
            failureSet_.Insert(id);
            UC_WARN("Block dump sources task={} timed out; retaining queued work and source protection", id);
            while (!work->sources.WaitForDuration(2000)) {
                UC_WARN("Block dump sources task={} still draining", id);
            }
        }
        return Status::OK();
    }

    Status Wait(Detail::TaskHandle id)
    {
        auto status = Detail::TaskWrapper<TransTask, Detail::TaskHandle>::Wait(id);
        std::lock_guard lock(queueMutex_);
        sources_.erase(id);
        return status;
    }

private:
    Status FailureStatus(const TaskPtr& task) const override { return task->FailureStatus(); }
    bool WaitForCapacity() const { return config_.cacheDumpCapacityPolicy == "wait"; }

    void Dispatch(TaskPtr task, WaiterPtr waiter) override
    {
        waiter->Set(1);
        auto work = std::make_shared<SourceTask>(TaskPair{task, waiter});
        std::lock_guard lock(queueMutex_);
        sources_.emplace(task->id, work);
        // The batch finalizer may be the only source of a resident block's
        // missing layer. Capacity waiters must not consume its admission slot.
        const auto limit = config_.waitingQueueDepth -
            size_t(WaitForCapacity() && !task->desc.finalizeBlockDump);
        if (!stopping_ && admitted_.load() < limit) {
            ++admitted_;
            pending_.push_back(work);
            wake_.notify_one();
            return;
        }
        work->sourceStatus = Status::NoSpace();
        task->Fail(work->sourceStatus);
        failureSet_.Insert(task->id);
        work->sources.Done();
        waiter->Done();
    }

    void Cancel(TaskPtr) override
    {
        // Match the legacy cache task cancellation policy: an expired wait does
        // not remove accepted tasks. TaskWrapper::Wait keeps waiting for Done;
        // pending source work and already accepted DMA/I/O must all drain.
    }

    Status Drain(CopyStream& stream)
    {
        auto first = stream.Synchronize();
        if (first.Success()) { return first; }
        // Do not release either DMA endpoint on an unproven stream error.
        while (stream.Synchronize().Failure()) {
            UC_ERROR("Block dump DMA completion unknown; retaining source and L1 pins");
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        return first;
    }

    void RunIo()
    {
        for (;;) {
            std::shared_ptr<WriteJob> job;
            {
                std::unique_lock lock(ioMutex_);
                ioWake_.wait(lock, [&] { return ioStopping_ || !ioPending_.empty(); });
                if (ioPending_.empty()) { return; }
                job = std::move(ioPending_.front()); ioPending_.pop_front();
            }
            auto status = Status::OK();
            if (!job->desc.empty()) {
                job->desc.deferWriteCommit = true;
                auto task = config_.storeBackend->Dump(std::move(job->desc));
                status = task ? config_.storeBackend->Wait(task.Value()) : task.Error();
            }
            if (status.Success() && job->commit) {
                status = config_.storeBackend->CommitDump(job->id);
            }
            job->pin = {}; job->lease.reset();
            {
                std::lock_guard lock(job->mutex);
                job->status = status;
                job->done.store(true, std::memory_order_release);
            }
            job->cv.notify_all();
            { std::lock_guard lock(queueMutex_); ++ioEpoch_; }
            wake_.notify_one();
        }
    }

    void ApplyWrite(Block& block)
    {
        auto job = block.write;
        if (!job || !job->done.load(std::memory_order_acquire)) { return; }
        block.writeFailure = job->status;
        if (job->status.Success()) {
            for (auto layer : job->layers) { block.persisted[layer] = true; }
            if (job->commit) { block.complete = true; }
            if (job->releasePin) { block.pin = {}; resident_.erase(&block); }
        } else {
            // Retain valid L1 data and the transaction for an explicit retry.
            block.closing = false;
        }
        block.write.reset();
    }

    Status WaitWrite(Block& block)
    {
        auto job = block.write;
        if (!job) { return Status::OK(); }
        {
            std::unique_lock lock(job->mutex);
            job->cv.wait(lock, [&] { return job->done.load(std::memory_order_acquire); });
        }
        auto status = job->status;
        ApplyWrite(block);
        return status;
    }

    void Reap()
    {
        for (auto it = blocks_.begin(); it != blocks_.end();) {
            ApplyWrite(it->second);
            if (it->second.closing && !it->second.write) {
                resident_.erase(&it->second);
                it = blocks_.erase(it);
            } else { ++it; }
        }
        for (auto it = finishing_.begin(); it != finishing_.end();) {
            const auto& work = *it;
            if (!std::all_of(work->writes.begin(), work->writes.end(),
                [](const auto& job) { return job->done.load(std::memory_order_acquire); })) {
                ++it; continue;
            }
            auto& task = work->pair.first;
            for (const auto& job : work->writes) {
                if (job->status.Failure()) { task->Fail(job->status); }
            }
            if (task->failureStatus.load() != Status::OK().Underlying()) {
                failureSet_.Insert(task->id);
            }
            --admitted_;
            work->pair.second->Done();
            it = finishing_.erase(it);
        }
    }

    void Run(std::promise<Status>& started)
    {
        CopyStream stream;
        auto status = stream.Setup(config_.deviceId, config_.EffectiveStreamNumber(), false);
        started.set_value(status);
        if (status.Failure()) { return; }
        if (WaitForCapacity()) { RunWaiting(stream); }
        else { RunSpilling(stream); }
        for (auto& [id, block] : blocks_) {
            auto saved = Spill(id, block);
            if (saved.Failure()) { UC_ERROR("Block dump shutdown spill failed: {}", saved); }
        }
    }

    void RunSpilling(CopyStream& stream)
    {
        for (;;) {
            const auto epoch = ioEpoch_.load();
            Reap();
            Work work;
            {
                std::unique_lock lock(queueMutex_);
                if (pending_.empty() && stopping_ && finishing_.empty()) { break; }
                if (pending_.empty()) {
                    wake_.wait(lock, [&] {
                        return !pending_.empty() || ioEpoch_.load() != epoch ||
                            (stopping_ && finishing_.empty());
                    });
                    continue;
                }
                work = std::move(pending_.front()); pending_.pop_front();
            }
            FinishSources(work, Process(stream, *work));
        }
    }

    void FinishSources(const Work& work, Status status)
    {
        if (status.Failure()) { work->pair.first->Fail(status); }
        work->sourceStatus = status;
        // All source copies have drained. Only pinned L1 is used from here.
        work->sources.Done();
        finishing_.push_back(work);
    }

    void RunWaiting(CopyStream& stream)
    {
        std::deque<Work> active;
        for (;;) {
            const auto epoch = ioEpoch_.load();
            const auto released = EarlyBufferPins::ReleaseEpoch(*buffer_);
            Reap();
            bool stopping;
            {
                std::lock_guard lock(queueMutex_);
                stopping = stopping_;
                while (!pending_.empty()) {
                    active.push_back(std::move(pending_.front())); pending_.pop_front();
                }
            }
            if (stopping && active.empty() && finishing_.empty()) { break; }
            bool advanced = false;
            bool pollReadiness = false;
            std::unordered_set<Detail::BlockId, Detail::BlockIdHasher> blocked;
            for (auto it = active.begin(); it != active.end();) {
                auto& work = *it;
                PrepareGroups(*work);
                for (auto& group : work->groups) {
                    if (group.done || blocked.count(group.id)) { continue; }
                    bool waiting = false;
                    auto status = ProcessGroup(stream, *work, group, waiting, pollReadiness);
                    if (waiting) { blocked.insert(group.id); continue; }
                    if (status.Failure()) { work->pair.first->Fail(status); }
                    group.done = true;
                    advanced = true;
                }
                if (std::all_of(work->groups.begin(), work->groups.end(),
                                [](const auto& group) { return group.done; })) {
                    FinishSources(work, Drain(stream));
                    it = active.erase(it);
                    advanced = true;
                } else { ++it; }
            }
            if (advanced) { continue; }
            if (stopping && !active.empty()) {
                // No future layers arrive after shutdown. Preserve partial data
                // using the same terminal spill as ordinary finalization.
                for (auto& [id, block] : blocks_) {
                    if (!block.pin || block.write) { continue; }
                    auto status = Spill(id, block);
                    if (status.Failure()) { block.writeFailure = status; }
                    else { advanced = true; }
                }
                if (advanced) { continue; }
            }
            std::unique_lock lock(queueMutex_);
            // Other processes notify the shared L1 epoch, not this local CV.
            // A foreign loader can also mark READY while retaining its pin;
            // those followers must recheck readiness without waiting for release.
            while (!wake_.wait_for(lock, std::chrono::milliseconds(2), [&] {
                return !pending_.empty() || ioEpoch_.load() != epoch ||
                    EarlyBufferPins::ReleaseEpoch(*buffer_) != released || stopping_ != stopping;
            }) && !pollReadiness) {}
        }
    }

    std::shared_ptr<WriteJob> Schedule(const Detail::BlockId& id, Block& block,
                                      bool full, bool releasePin, bool commit)
    {
        auto job = std::make_shared<WriteJob>();
        job->id = id; job->pin = block.pin; job->lease = block.lease;
        job->releasePin = releasePin; job->commit = commit;
        if (full) {
            job->desc.fullBlockWrite = true;
            job->desc.push_back({id, 0, {block.pin.Data()}});
            for (size_t layer = 0; layer < buffer_->LayerCount(); ++layer) {
                job->layers.push_back(layer);
            }
        } else if (block.pin) {
            for (size_t layer = 0; layer < buffer_->LayerCount(); ++layer) {
                auto view = block.pin.LayerView(layer);
                if (view.Ready() && !block.persisted[layer]) {
                    job->desc.push_back({id, layer, {view.Data()}});
                    job->layers.push_back(layer);
                }
            }
        }
        block.write = job;
        block.writeFailure = Status::OK();
        { std::lock_guard lock(ioMutex_); ioPending_.push_back(job); }
        ioWake_.notify_one();
        return job;
    }

    Status Spill(const Detail::BlockId& id, Block& block)
    {
        auto status = WaitWrite(block);
        if (status.Failure() || !block.pin) { return status; }
        Schedule(id, block, false, true, false);
        return WaitWrite(block);
    }

    Status MakeRoom(const Detail::BlockId& id)
    {
        if (WaitForCapacity()) {
            if (resident_.size() < config_.cacheDumpReservedBlocks) { return Status::OK(); }
            for (const auto* block : resident_) {
                // A known failed writer cannot release its slot by waiting.
                // Preserve its data for retry and surface the failure to waiters.
                if (block->writeFailure.Failure()) { return block->writeFailure; }
            }
            return Status::NoSpace();
        }
        size_t pins = 0;
        for (auto& [key, block] : blocks_) { ApplyWrite(block); pins += bool(block.pin); }
        if (pins < config_.cacheDumpReservedBlocks) { return Status::OK(); }
        for (auto& [key, block] : blocks_) {
            if (key != id && block.pin) { return Spill(key, block); }
        }
        return Status::NoSpace();
    }

    Status Fill(CopyStream& stream, const Detail::Shard& shard, Block& block,
                bool& waiting, bool& pollReadiness)
    {
        ApplyWrite(block);
        if (block.complete || block.persisted[shard.index]) { return Status::OK(); }
        if (block.pin && block.pin.LayerView(shard.index).Ready()) { return Status::OK(); }
        if (block.write) {
            if (WaitForCapacity() && !block.write->done.load(std::memory_order_acquire)) {
                waiting = true;
                return Status::OK();
            }
            auto status = WaitWrite(block);
            if (status.Failure()) { return status; }
        }
        if (!block.pin) {
            auto status = MakeRoom(shard.owner);
            if (WaitForCapacity() && status == Status::NoSpace()) {
                waiting = true;
                return Status::OK();
            }
            if (status.Failure()) { return status; }
        }
        auto handle = EarlyBufferPins::Acquire(*buffer_, shard.owner, shard.index, false);
        if (!handle && !WaitForCapacity()) {
            for (auto& [key, other] : blocks_) {
                if (key != shard.owner && other.pin) {
                    auto status = Spill(key, other);
                    if (status.Failure()) { return status; }
                    handle = EarlyBufferPins::Acquire(*buffer_, shard.owner, shard.index, false);
                    if (handle) { break; }
                }
            }
        }
        if (!handle || (!handle.Owner() && !handle.Ready())) {
            if (WaitForCapacity()) {
                if (handle && handle.GetState() == TransBuffer::State::FAILED) {
                    return handle.FailureStatus();
                }
                waiting = true;
                pollReadiness = pollReadiness || bool(handle);
                return Status::OK();
            }
            return Status::NoSpace();
        }
        if (!block.pin) { block.pin = handle.WholeView(); resident_.insert(&block); }
        if (handle.Ready()) { return Status::OK(); }
        const auto packed = config_.storeBackend->PackedLayerBytes(shard.index);
        std::memset(handle.Data(), 0, packed ? packed : config_.shardSize);
        auto addresses = shard.addrs;
        auto status = Status::OK();
#ifdef UCM_ENABLE_TEST_HOOKS
        if (TestHooks::blockDumpCopyHook) { status = TestHooks::blockDumpCopyHook(shard); }
#endif
        if (status.Success()) {
            status = stream.DeviceToHostAsync(addresses.data(), handle.Data(), config_.tensorSizes);
        }
        auto drained = Drain(stream);
        if (status.Failure() || drained.Failure()) {
            auto error = status.Failure() ? status : drained;
            handle.MarkLayerFailed(error);
            return error;
        }
        handle.MarkReady();
        return Status::OK();
    }

    Status Publish(const Detail::BlockId& id, Block& block, SourceTask& task)
    {
        if (block.complete) { return Status::OK(); }
        if (block.write) {
            if (block.write->commit) { task.writes.push_back(block.write); return Status::OK(); }
            auto status = WaitWrite(block);
            if (status.Failure()) { return status; }
        }
        const bool full = block.pin && block.pin.Ready();
        const bool persisted = std::all_of(block.persisted.begin(), block.persisted.end(),
                                         [](bool ready) { return ready; });
        for (size_t layer = 0; layer < buffer_->LayerCount(); ++layer) {
            if (!block.persisted[layer] && (!block.pin || !block.pin.LayerView(layer).Ready())) {
                return Status::NotFound();
            }
        }
        // Earlier spills may already cover some layers; persist the remaining
        // READY views and publish only after their physical writes succeed.
        task.writes.push_back(Schedule(id, block, full && !persisted, true, true));
        return Status::OK();
    }

    void PrepareGroups(SourceTask& work)
    {
        if (work.prepared) { return; }
        std::unordered_map<Detail::BlockId, size_t, Detail::BlockIdHasher> positions;
        for (const auto& shard : work.pair.first->desc) {
            auto [it, inserted] = positions.emplace(shard.owner, work.groups.size());
            if (inserted) { work.groups.push_back({shard.owner, {}, false}); }
            work.groups[it->second].shards.push_back(&shard);
        }
        work.prepared = true;
    }

    Status ProcessGroup(CopyStream& stream, SourceTask& work, const SourceGroup& group,
                        bool& waiting, bool& pollReadiness)
    {
        const auto& desc = work.pair.first->desc;
        const auto& id = group.id;
        auto it = blocks_.find(id);
        if (it == blocks_.end()) {
            if (blocks_.size() >= config_.waitingQueueDepth) {
                if (WaitForCapacity()) { waiting = true; return Status::OK(); }
                return desc.finalizeBlockDump ? Status::NoSpace() : Status::OK();
            }
            auto lease = config_.storeBackend->AcquireWriteLease(id);
            if (!lease) {
                return desc.finalizeBlockDump || lease.Error() != Status::Retry()
                    ? lease.Error() : Status::OK();
            }
            it = blocks_.emplace(id, Block{{}, std::move(lease.Value()),
                std::vector<bool>(buffer_->LayerCount()), false, false, {}}).first;
        }
        auto& block = it->second;
        // A later submission can join an immutable block's pending final write.
        // Do not reap its state before this submission has finished with it.
        block.closing = false;
        if (desc.prerequisiteHandle && !work.eventQueued) {
            auto status = stream.WaitEvent(Trans::Event{desc.prerequisiteHandle});
            if (status.Failure()) { Drain(stream); return status; }
            work.eventQueued = true;
        }
        auto status = Status::OK();
        for (const auto* shard : group.shards) {
            auto copied = Fill(stream, *shard, block, waiting, pollReadiness);
            if (waiting) { return status; }
            // A failed layer must not suppress other independent valid layers.
            if (copied.Failure() && status.Success()) { status = copied; }
        }
        if (status.Success()) {
            status = Publish(id, block, work);
            if (status == Status::NotFound() && !desc.finalizeBlockDump) { status = Status::OK(); }
        }
        if (desc.finalizeBlockDump) {
            if (!block.complete && !block.write) {
                work.writes.push_back(Schedule(id, block, false, true, false));
                if (status.Success()) { status = Status::NotFound(); }
            } else if (block.write) {
                work.writes.push_back(block.write);
            }
            block.closing = true;
        } else if (status == Status::NoSpace() && !WaitForCapacity()) {
            status = Status::OK();  // finalization supplies still-valid missing layers
        }
        return status;
    }

    Status Process(CopyStream& stream, SourceTask& work)
    {
        PrepareGroups(work);
        auto result = Status::OK();
        for (const auto& group : work.groups) {
            bool waiting = false, pollReadiness = false;
            auto status = ProcessGroup(stream, work, group, waiting, pollReadiness);
            if (status.Failure() && result.Success()) { result = status; }
        }
        auto drained = Drain(stream);
        return result.Failure() ? result : drained;
    }

};
}  // namespace UC::CacheStore
#endif
