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
// Install failure-injection hooks only while no test task is running.
inline std::function<Status(const Detail::Shard&)> blockDumpCopyHook;
inline std::function<Status()> blockDumpSyncHook;
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
    struct GroupProgress {
        SourceGroup* group;
        Status status{Status::OK()};
        bool waiting{false};
        bool publish{false};
    };
    struct PendingCopy {
        TransBuffer::Handle handle;
        Status submitted;
        Status* result;
    };
    struct CopyBatch {
        std::vector<PendingCopy> copies;
        std::unordered_set<Detail::BlockId, Detail::BlockIdHasher> blocks;
        size_t layer{0};
        bool dirty{false};
        Status failure{Status::OK()};
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
        if (!buffer->WholeBlock() || !config.storeBackend->SupportsFullBlockWrite()) {
            return Status::InvalidParam("block dump requires whole L1 and a write backend");
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

    Status Synchronize(CopyStream& stream)
    {
#ifdef UCM_ENABLE_TEST_HOOKS
        if (TestHooks::blockDumpSyncHook) {
            auto status = TestHooks::blockDumpSyncHook();
            if (status.Failure()) { return status; }
        }
#endif
        return stream.Synchronize();
    }

    Status Drain(CopyStream& stream)
    {
        auto first = Synchronize(stream);
        if (first.Success()) { return first; }
        // Do not release either DMA endpoint on an unproven stream error.
        while (Synchronize(stream).Failure()) {
            UC_ERROR("Block dump DMA completion unknown; retaining source and L1 pins");
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        return first;
    }

    void FlushCopies(CopyStream& stream, CopyBatch& batch)
    {
        if (!batch.dirty) { return; }
        const auto drained = Drain(stream);
        if (drained.Failure() && batch.failure.Success()) { batch.failure = drained; }
        // Keep every destination pinned until the entire batch is proven drained.
        // READY also permits the I/O worker to read it, so publication follows DMA.
        for (auto& copy : batch.copies) {
            const auto status = copy.submitted.Failure() ? copy.submitted : drained;
            if (status.Failure()) {
                copy.handle.MarkLayerFailed(status);
                if (copy.result->Success()) { *copy.result = status; }
            } else {
                copy.handle.MarkReady();
            }
        }
        batch.copies.clear();
        batch.blocks.clear();
        batch.dirty = false;
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
        auto status = config_.cacheSdmaDirect
            ? stream.SetupSdmaDirect(config_.deviceId, false)
            : stream.Setup(config_.deviceId, config_.EffectiveStreamNumber(), false,
                           config_.EffectiveStagingBytes());
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
                auto status = ProcessGroups(stream, *work, &blocked, advanced, pollReadiness);
                if (std::all_of(work->groups.begin(), work->groups.end(),
                                [](const auto& group) { return group.done; })) {
                    FinishSources(work, status);
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

    Status ResidentWriteFailure() const
    {
        for (const auto* block : resident_) {
            // A known failed writer needs a retry, not more capacity waiting.
            if (block->writeFailure.Failure()) { return block->writeFailure; }
        }
        return Status::OK();
    }

    Status MakeRoom(const Detail::BlockId& id)
    {
        // Zero removes only the holding quota; Fill still acquires a real L1 slot.
        if (config_.cacheDumpReservedBlocks == 0) { return Status::OK(); }
        if (WaitForCapacity()) {
            if (resident_.size() < config_.cacheDumpReservedBlocks) { return Status::OK(); }
            auto failure = ResidentWriteFailure();
            return failure.Failure() ? failure : Status::NoSpace();
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
                bool& waiting, bool& pollReadiness, CopyBatch& batch, Status& result)
    {
        if (!batch.copies.empty() && batch.layer != shard.index) { FlushCopies(stream, batch); }
        if (batch.blocks.count(shard.owner)) { return Status::OK(); }
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
                if (!handle) {
                    auto failure = ResidentWriteFailure();
                    if (failure.Failure()) { return failure; }
                }
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
            // SDMA addresses the device alias; CPU padding and POSIX I/O still
            // use the host view of the same pinned layer.
            auto* destination = config_.cacheSdmaDirect ? handle.DeviceData() : handle.Data();
            status = destination
                ? stream.DeviceToHostAsync(addresses.data(), destination, config_.tensorSizes)
                : Status::InvalidParam("block dump has no mapped destination");
        }
        batch.layer = shard.index;
        batch.dirty = true;
        batch.blocks.insert(shard.owner);
        batch.copies.push_back({std::move(handle), status, &result});
        if (status.Failure()) {
            if (batch.failure.Success()) { batch.failure = status; }
            // A rejected tensor may follow accepted copies in the same submission.
            FlushCopies(stream, batch);
        }
        return status;
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

    Status PrepareGroup(CopyStream& stream, SourceTask& work, GroupProgress& progress,
                        bool& pollReadiness, CopyBatch& batch)
    {
        const auto& group = *progress.group;
        auto& waiting = progress.waiting;
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
            batch.dirty = true;
            if (status.Failure()) { FlushCopies(stream, batch); return status; }
            work.eventQueued = true;
        }
        progress.publish = true;
        auto& status = progress.status;
        for (const auto* shard : group.shards) {
            auto copied = Fill(stream, *shard, block, waiting, pollReadiness, batch, status);
            if (waiting) { return status; }
            // A failed layer must not suppress other independent valid layers.
            if (copied.Failure() && status.Success()) { status = copied; }
        }
        return status;
    }

    Status CompleteGroup(SourceTask& work, GroupProgress& progress)
    {
        auto status = progress.status;
        if (!progress.publish) { return status; }
        const auto& desc = work.pair.first->desc;
        const auto& id = progress.group->id;
        auto& block = blocks_.at(id);
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

    Status ProcessGroups(
        CopyStream& stream, SourceTask& work,
        std::unordered_set<Detail::BlockId, Detail::BlockIdHasher>* blocked,
        bool& advanced, bool& pollReadiness)
    {
        PrepareGroups(work);
        std::vector<GroupProgress> progress;
        // Pending copies refer to these results until the task-local batch drains.
        progress.reserve(work.groups.size());
        CopyBatch batch;
        auto result = Status::OK();
        for (auto& group : work.groups) {
            if (group.done || (blocked && blocked->count(group.id))) { continue; }
            progress.push_back({&group});
            auto& current = progress.back();
            auto status = PrepareGroup(stream, work, current, pollReadiness, batch);
            if (status.Failure() && current.status.Success()) { current.status = status; }
            if (current.waiting && blocked) { blocked->insert(group.id); }
            if (!WaitForCapacity()) {
                // Spill may reclaim a preceding block before acquiring the next one.
                FlushCopies(stream, batch);
                current.status = CompleteGroup(work, current);
                current.publish = false;
            }
        }
        // Do not wait for unavailable capacity while holding an uncompleted batch.
        FlushCopies(stream, batch);
        for (auto& current : progress) {
            if (!current.waiting) {
                current.status = CompleteGroup(work, current);
                current.group->done = true;
                advanced = true;
            }
            if (current.status.Failure()) {
                work.pair.first->Fail(current.status);
                if (result.Success()) { result = current.status; }
            }
        }
        if (batch.failure.Failure()) {
            work.pair.first->Fail(batch.failure);
            if (result.Success()) { result = batch.failure; }
        }
        return result;
    }

    Status Process(CopyStream& stream, SourceTask& work)
    {
        bool advanced = false, pollReadiness = false;
        return ProcessGroups(stream, work, nullptr, advanced, pollReadiness);
    }

};
}  // namespace UC::CacheStore
#endif
