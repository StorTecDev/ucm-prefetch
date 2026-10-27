#include <atomic>
#include <cstring>
#include <filesystem>
#include <future>
#include <iostream>
#include <stdexcept>
#include <sys/syscall.h>
#include "posix/cc/posix_store.cc"
#include "cache/cc/reserved_prefetch.h"
#include "cache/cc/early_native_transfer.h"
#include "trans/device.h"

using namespace UC;
using namespace UC::PosixStore;

static void Require(bool ok, const char* why)
{
    if (!ok) { throw std::runtime_error(why); }
}

struct Fixture {
    inline static size_t layerBytes = 4096;
    static constexpr size_t layers = 14;
    std::string path;
    Detail::BlockId block{};
    std::vector<std::shared_ptr<void>> buffers;
    UC::PosixStore::EarlyPosixStore store;
    Fixture(size_t timeout = 5000)
    {
        auto pattern = (std::filesystem::temp_directory_path() / "ucm-full-block-probe-XXXXXX").string();
        auto* directory = mkdtemp(pattern.data());
        Require(directory != nullptr, "temporary directory");
        path = directory;
        block[0] = std::byte{42};
        Detail::Dictionary config;
        config.SetNumber("device_id", 0);
        config.Set("storage_backends", std::vector<std::string>{path});
        config.SetNumber("tensor_size", layerBytes);
        config.SetNumber("shard_size", layerBytes);
        config.SetNumber("block_size", layers * layerBytes);
        config.Set("posix_io_engine", std::string("aio"));
        config.SetNumber("timeout_ms", timeout);
        config.SetNumber("posix_open_concurrency", size_t{1});
        config.SetNumber("posix_commit_concurrency", size_t{1});
        config.SetNumber("data_dir_shard_bytes", size_t{0});
        Require(store.Setup(config).Success(), "Posix setup");
        Require(store.SupportsFullBlockRead(), "AIO capability");
        for (size_t i = 0; i < layers; ++i) {
            void* buffer = nullptr;
            Require(posix_memalign(&buffer, 4096, layerBytes) == 0, "aligned allocation");
            buffers.emplace_back(buffer, std::free);
            memset(buffer, i + 1, layerBytes);
            Detail::TaskDesc dump{{block, i, {buffer}}};
            auto task = store.Dump(std::move(dump));
            Require(bool(task) && store.Wait(task.Value()).Success(), "layer dump");
        }
        const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        for (;;) {
            auto hit = store.Lookup(&block, 1);
            if (hit && hit.Value()[0]) { break; }
            Require(std::chrono::steady_clock::now() < end, "block commit");
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    ~Fixture()
    {
        TestHooks::ClearAioHooks();
        std::filesystem::remove_all(path);
    }
    Detail::TaskDesc Read(bool partial = false)
    {
        Detail::TaskDesc read;
        read.fullBlockRead = true;
        read.push_back({block, 0, {}});
        for (size_t i = 0; i < layers; ++i) {
            memset(buffers[i].get(), 0xff, layerBytes);
            read[0].addrs.push_back(partial && i % 2 ? nullptr : buffers[i].get());
        }
        return read;
    }
    void Verify(bool partial = false)
    {
        for (size_t i = 0; i < layers; ++i) {
            auto* data = static_cast<unsigned char*>(buffers[i].get());
            for (size_t j = 0; j < layerBytes; ++j) {
                Require(data[j] == (partial && i % 2 ? 0xff : i + 1), "layer bytes");
            }
        }
    }
};

static void FullRead(bool partial)
{
    Fixture f;
    std::atomic<size_t> reads{0};
    TestHooks::SetAioSubmitHook([&](aio_context_t ctx, int64_t nr, iocb** cb) {
        Require(nr == 1 && cb[0]->aio_lio_opcode == IOCB_CMD_PREADV, "single vector read");
        Require(cb[0]->aio_offset == 0 && cb[0]->aio_nbytes == Fixture::layers,
                "whole file vectors");
        auto* vectors = reinterpret_cast<iovec*>(cb[0]->aio_buf);
        size_t bytes = 0;
        for (size_t i = 0; i < cb[0]->aio_nbytes; ++i) { bytes += vectors[i].iov_len; }
        Require(bytes == Fixture::layerBytes * Fixture::layers, "whole block bytes");
        ++reads;
        return static_cast<int32_t>(syscall(SYS_io_submit, ctx, nr, cb));
    });
    auto desc = f.Read(partial);
    auto progress = desc.readProgress = std::make_shared<Detail::ReadProgress>();
    auto read = f.store.Load(std::move(desc));
    Require(bool(read) && f.store.Wait(read.Value()).Success(), "full read completion");
    Require(progress->retiredBytes == Fixture::layers * Fixture::layerBytes,
            "success retires exact bytes once");
    Require(reads == 1, "one IO per block");
    f.Verify(partial);
    auto bad = f.Read();
    bad[0].addrs.pop_back();
    Require(!f.store.Load(std::move(bad)), "invalid vector shape rejected");
    Require(!f.store.Dump(f.Read()), "read descriptor rejected by dump");
}

static void TimeoutRetainsVectors()
{
    Fixture f(50);
    std::promise<std::pair<aio_context_t, iocb*>> accepted;
    TestHooks::SetAioSubmitHook([&](aio_context_t ctx, int64_t nr, iocb** cb) {
        accepted.set_value({ctx, *cb});
        return static_cast<int32_t>(nr);
    });
    TestHooks::SetAioCancelHook([](aio_context_t, iocb*, io_event*) {
        errno = EAGAIN;
        return -1;
    });
    auto read = f.store.Load(f.Read(true));
    Require(bool(read), "delayed read accepted");
    auto [ctx, cb] = accepted.get_future().get();
    auto wait = std::async(std::launch::async, [&] { return f.store.Wait(read.Value()); });
    Require(wait.wait_for(std::chrono::milliseconds(150)) == std::future_status::timeout,
            "timeout must retain live read vectors");
    Require(syscall(SYS_io_submit, ctx, 1, &cb) == 1, "delayed real kernel read");
    Require(wait.get() == Status::Timeout(), "deadline propagated after physical drain");
    f.Verify(true);
}

static UC::CacheStore::Config CacheConfig(StoreV1& backend, bool full, size_t slots = 128)
{
    UC::CacheStore::Config config;
    config.storeBackend = &backend;
    config.uniqueId = "full-block-probe";
    config.deviceId = 0;
    config.tensorSizes = {Fixture::layerBytes};
    config.shardSize = Fixture::layerBytes;
    config.blockSize = Fixture::layers * Fixture::layerBytes;
    config.bufferCapacity = slots * Fixture::layerBytes;
    config.loadExclusiveBufferNumber = 0;
    config.shareBufferEnable = false;
    config.localRankSize = 1;
    config.cacheFullBlockLoad = full;
    config.timeoutMs = 5000;
    return config;
}

static void CacheRead(bool full, bool early, bool partial)
{
    Fixture f;
    auto config = CacheConfig(f.store, full);
    UC::CacheStore::BufferManager manager;
    Require(manager.Setup(config).Success(), "cache setup");
    auto& buffer = *manager.GetTransBuffer();
    if (partial) {
        auto pin = buffer.Get(f.block, 3);
        memset(pin.Data(), 4, Fixture::layerBytes);
        pin.MarkReady();
    }
    std::atomic<size_t> reads{0};
    TestHooks::SetAioSubmitHook([&](aio_context_t ctx, int64_t nr, iocb** cb) {
        Require(cb[0]->aio_lio_opcode == (full ? IOCB_CMD_PREADV : IOCB_CMD_PREAD),
                "cache IO granularity");
        ++reads;
        return static_cast<int32_t>(syscall(SYS_io_submit, ctx, nr, cb));
    });
    UC::CacheStore::ReservedPrefetch prefetch(&manager, &f.store, config.blockSize,
                                             config.shardSize, config.timeoutMs, full);
    uint64_t reservation = 0;
    if (early) {
        auto first = prefetch.Reserve(&f.block, 1);
        auto second = prefetch.Reserve(&f.block, 1);
        Require(bool(first) && bool(second), "duplicate reservations");
        auto one = first.Value().batch->Step(128 * config.blockSize);
        Require(bool(one), "first prefetch step");
        auto two = second.Value().batch->Step(128 * config.blockSize);
        Require(bool(two), "coalesced prefetch step");
        reservation = first.Value().handle;
        auto result = first.Value().batch->Snapshot();
        while (!result.done) {
            auto step = first.Value().batch->Step(128 * config.blockSize);
            Require(bool(step), "prefetch completion");
            result = step.Value();
        }
        Require(result.result.l2ReadBytes == (full ? config.blockSize :
                (Fixture::layers - (partial ? 1 : 0)) * config.shardSize),
                        "physical byte accounting");
        while (!two.Value().done) {
            two = second.Value().batch->Step(128 * config.blockSize);
            Require(bool(two), "coalesced completion");
        }
        Require(two.Value().result.l2ReadBytes == 0, "duplicate read coalescing");
        prefetch.Release(second.Value().handle);
    }
    UC::CacheStore::TransManager native;
    Require(native.Setup(config, &buffer).Success(), "native layer transfer setup");
    std::vector<std::vector<unsigned char>> target(Fixture::layers,
                                                  std::vector<unsigned char>(Fixture::layerBytes, 0xff));
    Trans::Device device;
    Require(device.Setup(0).Success(), "device setup");
    auto allocator = device.MakeBuffer();
    auto stream = device.MakeStream();
    auto destination = allocator->MakeDeviceBuffer(config.blockSize);
    Require(bool(destination), "device allocation");
    for (size_t layer = 0; layer < Fixture::layers; ++layer) {
        auto* address = static_cast<std::byte*>(destination.get()) + layer * config.shardSize;
        Require(stream->HostToDevice(target[layer].data(), address, config.shardSize).Success(),
                "initialize device");
    }
    for (size_t layer = 0; layer < Fixture::layers; ++layer) {
        auto* address = static_cast<std::byte*>(destination.get()) + layer * config.shardSize;
        Detail::TaskDesc desc{{f.block, layer, {address}}};
        auto task = native.Submit({UC::CacheStore::TransTask::Type::LOAD, std::move(desc)});
        Require(bool(task) && native.Wait(task.Value()).Success(), "layer transfer");
        for (size_t i = 0; i < Fixture::layers; ++i) {
            auto* src = static_cast<std::byte*>(destination.get()) + i * config.shardSize;
            Require(stream->DeviceToHost(src, target[i].data(), config.shardSize).Success(),
                    "read back device");
        }
        for (size_t later = layer + 1; later < Fixture::layers; ++later) {
            Require(target[later][0] == 0xff, "future layers must not transfer early");
        }
        for (auto byte : target[layer]) { Require(byte == layer + 1, "layer content"); }
    }
    Require(reads == (full ? 1 : Fixture::layers - (partial ? 1 : 0)), "cache read count");
    if (reservation) { prefetch.Release(reservation); }
}

static void CapacityRollback()
{
    Fixture f;
    auto config = CacheConfig(f.store, true, Fixture::layers - 1);
    UC::CacheStore::BufferManager manager;
    Require(manager.Setup(config).Success(), "small cache setup");
    UC::CacheStore::ReservedPrefetch prefetch(&manager, &f.store, config.blockSize,
                                             config.shardSize, config.timeoutMs, true);
    std::atomic<size_t> reads{0};
    TestHooks::SetAioSubmitHook([&](aio_context_t ctx, int64_t nr, iocb** cb) {
        ++reads;
        return static_cast<int32_t>(syscall(SYS_io_submit, ctx, nr, cb));
    });
    auto reserved = prefetch.Reserve(&f.block, 1);
    Require(bool(reserved), "capacity reservation");
    auto result = reserved.Value().batch->Run();
    Require(bool(result) && result.Value().capacityTruncated && result.Value().totalHitBlocks == 0,
            "whole-block capacity truncation");
    Require(reads == 0 && reserved.Value().batch->ProtectedBytes() == 0,
            "no partial IO or leaked pins");
    prefetch.Release(reserved.Value().handle);
}

class ImmediateBackend : public StoreV1 {
public:
    size_t reads{0}, maxBatch{0};
    Status Setup(const Detail::Dictionary&) override { return Status::OK(); }
    std::string Readme() const override { return "test backend"; }
    bool SupportsFullBlockRead() const override { return true; }
    Expected<std::vector<uint8_t>> Lookup(const Detail::BlockId*, size_t n) override
    { return std::vector<uint8_t>(n, 1); }
    Expected<ssize_t> LookupOnPrefix(const Detail::BlockId*, size_t n) override
    { return static_cast<ssize_t>(n) - 1; }
    Expected<ssize_t> LookupOnReverse(const Detail::BlockId*, size_t n) override
    { return static_cast<ssize_t>(n) - 1; }
    void Prefetch(const Detail::BlockId*, size_t) override {}
    Expected<std::shared_ptr<ReadLease>> AcquireReadLease(const Detail::BlockId*, size_t) override
    { return std::make_shared<ReadLease>(); }
    Expected<Detail::TaskHandle> Load(Detail::TaskDesc task) override
    {
        Require(task.fullBlockRead, "full-block batch descriptor");
        maxBatch = std::max(maxBatch, task.size());
        for (const auto& block : task) {
            Require(block.addrs.size() == Fixture::layers && block.index == 0,
                    "complete block shape");
            for (size_t i = 0; i < block.addrs.size(); ++i) {
                if (block.addrs[i]) { memset(block.addrs[i], i + 1, Fixture::layerBytes); }
            }
            ++reads;
        }
        return size_t{reads};
    }
    Expected<Detail::TaskHandle> Dump(Detail::TaskDesc) override { return Status::Error(); }
    Expected<bool> Check(Detail::TaskHandle) override { return true; }
    Status Wait(Detail::TaskHandle) override { return Status::OK(); }
};

static void MultiBlockWindow()
{
    ImmediateBackend backend;
    auto config = CacheConfig(backend, true, 72 * Fixture::layers);
    UC::CacheStore::BufferManager manager;
    Require(manager.Setup(config).Success(), "multi-block cache");
    UC::CacheStore::ReservedPrefetch prefetch(&manager, &backend, config.blockSize,
                                             config.shardSize, config.timeoutMs, true);
    std::vector<Detail::BlockId> blocks(70);
    for (size_t i = 0; i < blocks.size(); ++i) { blocks[i][0] = std::byte(i); }
    auto reservation = prefetch.Reserve(blocks.data(), blocks.size());
    Require(bool(reservation), "multi-block reservation");
    auto batch = reservation.Value().batch;
    auto none = batch->Step(config.blockSize - 1);
    Require(bool(none) && backend.reads == 0, "block-sized minimum credit");
    auto first = batch->Step(batch->windowBytes);
    Require(bool(first) && backend.reads == 64 && backend.maxBatch == 64,
            "64 complete blocks per batch");
    auto state = first.Value();
    while (!state.done) {
        auto next = batch->Step(batch->windowBytes);
        Require(bool(next), "multi-block step");
        state = next.Value();
    }
    Require(backend.reads == blocks.size() && state.result.totalHitBlocks == blocks.size(),
            "multi-block completion");
    Require(state.result.l2ReadBytes == blocks.size() * config.blockSize && state.inflightBytes == 0,
            "actual bytes and drained window");
    Require(state.completedReadBytes == state.result.l2ReadBytes, "completed byte accounting");
    prefetch.Release(reservation.Value().handle);
    Require(batch->ProtectedBytes() == 0, "released reservation pins");
}

static void NativeEarlyFallback()
{
    Fixture f;
    auto config = CacheConfig(f.store, true);
    UC::CacheStore::BufferManager manager;
    Require(manager.Setup(config).Success(), "fallback cache");
    auto& buffer = *manager.GetTransBuffer();
    UC::CacheStore::TransManager native;
    Require(native.Setup(config, &buffer).Success(), "fallback native setup");
    UC::CacheStore::EarlyNativeTransfer transfer(f.store, native, config);
    Trans::Device device;
    Require(device.Setup(0).Success(), "fallback device setup");
    auto allocator = device.MakeBuffer();
    auto stream = device.MakeStream();
    auto destination = allocator->MakeDeviceBuffer(config.shardSize);
    std::vector<unsigned char> data(config.shardSize);
    std::atomic<size_t> reads{0};
    TestHooks::SetAioSubmitHook([&](aio_context_t ctx, int64_t nr, iocb** cb) {
        Require(cb[0]->aio_lio_opcode == IOCB_CMD_PREADV, "fallback whole-block IO");
        ++reads;
        return static_cast<int32_t>(syscall(SYS_io_submit, ctx, nr, cb));
    });
    for (size_t layer = 0; layer < Fixture::layers; ++layer) {
        Detail::TaskDesc desc{{f.block, layer, {destination.get()}}};
        UC::CacheStore::TransTask task{UC::CacheStore::TransTask::Type::LOAD, std::move(desc)};
        task.loadHandles.push_back(
            UC::CacheStore::EarlyBufferPins::Acquire(buffer, f.block, layer, true));
        auto accepted = transfer.Submit(UC::CacheStore::EarlyNativeTask{std::move(task)});
        Require(bool(accepted) && transfer.Wait(accepted.Value()).Success(), "fallback transfer");
        Require(stream->DeviceToHost(destination.get(), data.data(), config.shardSize).Success(),
                "fallback readback");
        for (auto byte : data) { Require(byte == layer + 1, "fallback layer bytes"); }
    }
    Require(reads == 1, "fallback one read per block");
}

static void CancelledPrefetchRetainsDestinations()
{
    Fixture f;
    auto config = CacheConfig(f.store, true);
    UC::CacheStore::BufferManager manager;
    Require(manager.Setup(config).Success(), "cancel cache setup");
    UC::CacheStore::ReservedPrefetch prefetch(&manager, &f.store, config.blockSize,
                                             config.shardSize, config.timeoutMs, true);
    auto reservation = prefetch.Reserve(&f.block, 1);
    Require(bool(reservation), "cancel reservation");
    auto batch = reservation.Value().batch;
    std::promise<std::pair<aio_context_t, iocb*>> accepted;
    TestHooks::SetAioSubmitHook([&](aio_context_t ctx, int64_t nr, iocb** cb) {
        accepted.set_value({ctx, *cb});
        return static_cast<int32_t>(nr);
    });
    auto started = batch->Step(config.blockSize);
    Require(bool(started) && !started.Value().done, "cancel starts actual read");
    auto [ctx, cb] = accepted.get_future().get();
    prefetch.Release(reservation.Value().handle);
    auto pending = batch->Step(0);
    Require(bool(pending) && !pending.Value().done, "cancel cannot complete physical IO");
    Require(batch->ProtectedBytes() == config.blockSize, "cancel retains all destinations");
    Require(syscall(SYS_io_submit, ctx, 1, &cb) == 1, "complete cancelled vector read");
    for (;;) {
        auto state = batch->Step(0);
        if (!state) { break; }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    Require(batch->ProtectedBytes() == 0, "cancel releases after physical completion");
}

static void ShortReadFailsWholeBlock()
{
    Fixture f;
    bool truncated = false;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(f.path)) {
        if (entry.is_regular_file() && entry.file_size() == Fixture::layers * Fixture::layerBytes) {
            std::filesystem::resize_file(entry.path(), Fixture::layerBytes);
            truncated = true;
        }
    }
    Require(truncated, "truncate owned block file");
    auto desc = f.Read();
    auto progress = desc.readProgress = std::make_shared<Detail::ReadProgress>();
    auto task = f.store.Load(std::move(desc));
    Require(bool(task) && f.store.Wait(task.Value()) == Status::NotFound(), "short block rejected");
    Require(progress->retiredBytes == Fixture::layers * Fixture::layerBytes,
            "short read returns requested credit without claiming success");
}


// Controlled backend isolates the budget from task completion and pin lifetime.
class ProgressBackend : public ImmediateBackend {
public:
    std::vector<Detail::TaskDesc> tasks;
    std::vector<bool> done;
    std::vector<size_t> waited;
    std::weak_ptr<ReadLease> lease;
    bool unknown{false}, fail{false};
    Expected<std::shared_ptr<ReadLease>> AcquireReadLease(const Detail::BlockId*, size_t) override
    {
        auto owned = std::make_shared<ReadLease>();
        lease = owned;
        return owned;
    }
    Expected<Detail::TaskHandle> Load(Detail::TaskDesc task) override
    {
        Require(bool(task.readProgress), "progress installed before submission");
        tasks.push_back(std::move(task));
        done.push_back(false);
        return tasks.size();
    }
    Expected<bool> Check(Detail::TaskHandle id) override
    {
        if (id == 1 && unknown) { return Status::Error("unknown completion"); }
        return bool(done.at(id - 1));
    }
    Status Wait(Detail::TaskHandle id) override
    {
        Require(done.at(id - 1), "never Wait for an unfinished task");
        Require(!lease.expired(), "lease retained through Wait");
        waited.push_back(id);
        return fail && id != 2 ? Status::NotFound() : Status::OK();
    }
};

static void PartialCreditWithFifoReaping(bool full, int ending)
{
    ProgressBackend backend;
    const auto layers = full ? Fixture::layers : 1;
    auto config = CacheConfig(backend, full, 192 * layers);
    config.blockSize = layers * config.shardSize;
    UC::CacheStore::BufferManager manager;
    Require(manager.Setup(config).Success(), "progress cache setup");
    UC::CacheStore::ReservedPrefetch early(&manager, &backend, config.blockSize,
                                           config.shardSize, 5000, full);
    std::vector<Detail::BlockId> blocks(192);
    for (size_t i = 0; i < blocks.size(); ++i) { memcpy(blocks[i].data(), &i, sizeof(i)); }
    auto reserved = early.Reserve(blocks.data(), blocks.size());
    Require(bool(reserved), "progress reservation");
    auto batch = reserved.Value().batch;
    auto unit = config.blockSize;
    Require(batch->Step(128 * unit).Value().inflightBytes == 64 * unit, "first 64");
    Require(batch->Step(128 * unit).Value().inflightBytes == 128 * unit, "window full");
    backend.tasks[0].readProgress->retiredBytes = 63 * unit;
    auto p = batch->Step(0).Value();
    Require(p.inflightBytes == 65 * unit && !p.done && p.completedReadBytes == 0,
            "63 callbacks reclaim credit without task completion");
    Require(batch->ProtectedBytes() == 128 * unit, "credit does not release pins");
    Require(p.pendingReadTasks == 2, "partial credit retains both pending tasks");
    auto first = manager.GetTransBuffer()->TryGet(blocks[0], 0);
    Require(first && !first.Ready(), "credit does not publish READY");
    Require(batch->Step(0).Value().inflightBytes == 65 * unit, "no double credit");
    Require(backend.tasks.size() == 2, "Step(0) never submits");
    p = batch->Step(63 * unit).Value();
    Require(backend.tasks.size() == 3 && backend.tasks[2].size() == 63 &&
            p.inflightBytes == 128 * unit, "refill reclaimed window while tail is pending");
    // A later task returns credit without bypassing FIFO Wait or publishing READY.
    backend.done[1] = true;
    backend.tasks[1].readProgress->retiredBytes = 64 * unit;
    backend.unknown = ending == 2;
    p = batch->Step(0).Value();
    Require(p.inflightBytes == 64 * unit && p.completedReadBytes == 0,
            "completed second task returns credit behind slow or unknown first");
    Require(backend.waited.empty(), "FIFO Wait retained");
    Require(p.pendingReadTasks == 3, "credit does not reap any task");
    auto second = manager.GetTransBuffer()->TryGet(blocks[64], 0);
    Require(second && !second.Ready() && !first.Ready(), "unreaped tasks are not READY");
    if (ending == 0) { batch->Discard(); }
    else { early.Release(reserved.Value().handle); }
    Require(!backend.lease.expired(), "discard/cancel keeps live read lease");
    p = batch->Step(0).Value();
    Require(!p.done && batch->ProtectedBytes() == 191 * unit, "destinations retained");
    backend.tasks[0].readProgress->retiredBytes = 64 * unit;
    backend.tasks[2].readProgress->retiredBytes = 63 * unit;
    p = batch->Step(0).Value();
    Require(p.inflightBytes == 0 && p.pendingReadTasks == 3 && !p.done,
            "zero credit is independent of pending native task completion");
    backend.unknown = false;
    backend.fail = ending == 3;
    backend.done[0] = backend.done[2] = true;
    backend.tasks[0].readProgress->retiredBytes = 64 * unit;
    backend.tasks[2].readProgress->retiredBytes = 63 * unit;
    auto final = batch->Step(0);
    Require(ending == 0 ? bool(final) : !final, "terminal status after drain");
    Require(batch->Snapshot().inflightBytes == 0 && batch->Snapshot().pendingReadTasks == 0 && batch->Snapshot().done,
            "all credit returned exactly once");
    Require(backend.tasks.size() == 3, "no suffix submission after cancel or discard");
    if (ending == 3) {
        Require(!first.Ready() && batch->Snapshot().completedReadBytes == 64 * unit,
                "failed read is not successful bytes or READY");
    }
    early.Release(reserved.Value().handle);
    Require(backend.lease.expired(), "lease released only after physical drain");
}

static void PartialAioCredit()
{
    Fixture f;
    auto desc = f.Read();
    auto progress = desc.readProgress = std::make_shared<Detail::ReadProgress>();
    std::vector<std::shared_ptr<void>> secondBuffers;
    Detail::Shard second{f.block, 0, {}};
    for (size_t i = 0; i < Fixture::layers; ++i) {
        void* p = nullptr;
        Require(posix_memalign(&p, 4096, Fixture::layerBytes) == 0, "second destination");
        secondBuffers.emplace_back(p, std::free);
        second.addrs.push_back(p);
    }
    desc.push_back(std::move(second));
    std::promise<std::pair<aio_context_t, iocb*>> held;
    std::atomic<size_t> submitted{0};
    TestHooks::SetAioSubmitHook([&](aio_context_t ctx, int64_t nr, iocb** cb) {
        if (submitted.fetch_add(1) == 0) {
            held.set_value({ctx, *cb});
            return int32_t{1};
        }
        return static_cast<int32_t>(syscall(SYS_io_submit, ctx, nr, cb));
    });
    auto task = f.store.Load(std::move(desc));
    Require(bool(task), "two-read task accepted");
    auto [ctx, cb] = held.get_future().get();
    auto end = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    auto unit = Fixture::layers * Fixture::layerBytes;
    while (progress->retiredBytes != unit && std::chrono::steady_clock::now() < end) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    const auto partial = progress->retiredBytes.load();
    auto check = f.store.Check(task.Value());
    // Drain even when assertions would fail, so a failing probe cannot strand IO.
    Require(syscall(SYS_io_submit, ctx, 1, &cb) == 1, "release held physical read");
    Require(f.store.Wait(task.Value()).Success(), "partial task drains");
    Require(partial == unit && check && !check.Value(), "real callback credit before whole task");
    Require(progress->retiredBytes == 2 * unit, "each callback contributes exactly once");
}

static void PreSubmitFailureCredit(bool missing)
{
    Fixture f;
    auto desc = f.Read();
    auto progress = desc.readProgress = std::make_shared<Detail::ReadProgress>();
    if (missing) { desc[0].owner[0] = std::byte{99}; }
    else {
        TestHooks::SetAioSubmitHook([](aio_context_t, int64_t, iocb**) {
            errno = EIO;
            return int32_t{-1};
        });
    }
    auto task = f.store.Load(std::move(desc));
    Require(task && f.store.Wait(task.Value()).Failure(), "pre-submit error status");
    Require(progress->retiredBytes == Fixture::layers * Fixture::layerBytes,
            "open/submit error returns exact credit once");
}

int main(int argc, char** argv)
{
    try {
        if (argc > 1) { Fixture::layerBytes = std::stoull(argv[1]); }
        Require(Fixture::layerBytes == 4096 || Fixture::layerBytes == 86016, "validated probe stride");
        for (bool full : {false, true}) {
            for (int ending = 0; ending < 4; ++ending) { PartialCreditWithFifoReaping(full, ending); }
        }
        PartialAioCredit();
        PreSubmitFailureCredit(true);
        PreSubmitFailureCredit(false);
        FullRead(false);
        FullRead(true);
        TimeoutRetainsVectors();
        for (bool full : {false, true}) {
            for (bool early : {false, true}) {
                // Ordinary full-block opt-in is rejected by the scope probe.
                if (full && !early) { continue; }
                for (bool partial : {false, true}) { CacheRead(full, early, partial); }
            }
        }
        CapacityRollback();
        MultiBlockWindow();
        NativeEarlyFallback();
        CancelledPrefetchRetainsDestinations();
        ShortReadFailsWholeBlock();
        std::cout << "PASS: full block read, partial hits, shape validation, delayed completion, layer transfers, Early coalescing, capacity rollback\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
