#include <atomic>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <future>
#include <iostream>
#include <stdexcept>
#include <sys/syscall.h>
#include "posix/cc/posix_store.cc"
#include "cache/cc/reserved_prefetch.h"
#include "cache/cc/trans_manager.h"
#include "trans/device.h"
#include <fstream>
#include <numeric>

using namespace UC;
using namespace UC::PosixStore;

static void Require(bool ok, const char* why)
{
    if (!ok) { throw std::runtime_error(why); }
}



// Real packed L2 files; direct kernel AIO and the simu transfer backend.
struct PackedFixture {
    static constexpr size_t stride = 86016;
    std::string path;
    Detail::BlockId block{};
    Detail::Dictionary config;
    std::vector<ssize_t> lengths;
    std::vector<std::shared_ptr<void>> buffers;
    std::unique_ptr<EarlyPosixStore> store;

    explicit PackedFixture(const std::string& engine = "aio", size_t timeout = 5000)
    {
        auto pattern = (std::filesystem::temp_directory_path() /
                        ("ucm-block-l1-probe-" + std::to_string(getpid()) + "-XXXXXX")).string();
        auto* directory = mkdtemp(pattern.data());
        Require(directory != nullptr, "temporary directory");
        path = directory;
        block[0] = std::byte{71};
        for (size_t i = 0; i < 14; ++i) {
            const bool indexed = i < 3 || i == 6 || i == 10;
            lengths.push_back(indexed ? 86016 : 73728);
            void* data = nullptr;
            Require(posix_memalign(&data, 4096, stride) == 0, "packed aligned buffer");
            buffers.emplace_back(data, std::free);
            memset(data, i + 1, stride);
        }
        config.SetNumber("device_id", 0);
        config.Set("storage_backends", std::vector<std::string>{path});
        config.SetNumber("tensor_size", stride);
        config.SetNumber("shard_size", stride);
        config.SetNumber("block_size", stride * 14);
        config.Set("posix_packed_layer_bytes", lengths);
        config.Set("posix_io_engine", engine);
        config.Set("io_direct", engine == "aio");
        config.Set("posix_gc_enable", false);
        config.SetNumber("posix_open_concurrency", 1);
        config.SetNumber("posix_commit_concurrency", 1);
        config.SetNumber("posix_data_trans_concurrency", 1);
        config.SetNumber("data_dir_shard_bytes", 0);
        config.SetNumber("timeout_ms", timeout);
        store = std::make_unique<EarlyPosixStore>();
        Require(store->Setup(config).Success(), "packed Posix setup");
        Require(store->FullBlockReadBytes() == 1093632, "actual packed block size");
        for (size_t i = 0; i < 14; ++i) {
            auto handle = store->Dump({{block, i, {buffers[i].get()}}});
            Require(handle && store->Wait(handle.Value()).Success(), "packed layer dump");
            if (i < 13) {
                auto found = store->Lookup(&block, 1);
                Require(found && !found.Value()[0], "partial packed prefix hidden");
            }
        }
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!store->Lookup(&block, 1).Value()[0]) {
            Require(std::chrono::steady_clock::now() < deadline, "packed publication");
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    ~PackedFixture()
    {
        TestHooks::ClearAioHooks();
        store.reset();
        std::filesystem::remove_all(path);
    }
    Detail::TaskDesc Read(bool partial = false)
    {
        Detail::TaskDesc task;
        task.fullBlockRead = true;
        task.push_back({block, 0, {}});
        for (size_t i = 0; i < 14; ++i) {
            memset(buffers[i].get(), 0xff, stride);
            task[0].addrs.push_back(partial && i % 2 ? nullptr : buffers[i].get());
        }
        return task;
    }
    void Verify(bool partial = false)
    {
        for (size_t i = 0; i < 14; ++i) {
            const auto* bytes = static_cast<const unsigned char*>(buffers[i].get());
            for (size_t j = 0; j < stride; ++j) {
                const auto expected = (partial && i % 2) || j >= static_cast<size_t>(lengths[i])
                    ? 0xff : i + 1;
                Require(bytes[j] == expected, "packed payload or untouched L1 tail");
            }
        }
    }
};


using namespace UC::CacheStore;

static UC::CacheStore::Config BlockConfig(PackedFixture& fixture, size_t slots, bool shared = false)
{
    UC::CacheStore::Config config;
    config.storeBackend = fixture.store.get();
    config.uniqueId = "block-l1-probe-" + std::to_string(getpid()) + (shared ? "-shared" : "-local");
    config.deviceId = 0;
    config.tensorSizes = {73728, 12288};
    config.shardSize = PackedFixture::stride;
    config.blockSize = 14 * PackedFixture::stride;
    config.bufferCapacity = slots * fixture.store->FullBlockReadBytes();
    config.loadExclusiveBufferNumber = 0;
    config.shareBufferEnable = shared;
    config.localRankSize = 1;
    config.cacheFullBlockLoad = true;
    config.cacheFullBlockL1 = true;
    config.timeoutMs = 5000;
    return config;
}

static Detail::BlockId Other(Detail::BlockId block, unsigned value)
{
    block[1] = std::byte{static_cast<unsigned char>(value)};
    return block;
}

static void PrefetchThenLayerwiseH2D(bool shared)
{
    PackedFixture f;
    auto config = BlockConfig(f, 1, shared);
    BufferManager manager;
    Require(manager.Setup(config).Success(), "one-block cache setup");
    auto& buffer = *manager.GetTransBuffer();
    Require(buffer.WholeBlock() && buffer.LayerCount() == 14, "whole-block view layout");
    auto physical = f.store->FullBlockReadBytes();
    ReservedPrefetch prefetch(&manager, f.store.get(), physical, physical, 5000, true);
    std::atomic<size_t> reads{0};
    TestHooks::SetAioSubmitHook([&](aio_context_t ctx, int64_t nr, iocb** cb) {
        Require(nr == 1 && cb[0]->aio_lio_opcode == IOCB_CMD_PREAD, "B-style contiguous PREAD");
        Require(cb[0]->aio_offset == 0 && cb[0]->aio_nbytes == physical, "whole packed block read");
        ++reads;
        return static_cast<int32_t>(syscall(SYS_io_submit, ctx, nr, cb));
    });
    auto reserved = prefetch.Reserve(&f.block, 1);
    Require(bool(reserved), "block reservation");
    auto result = reserved.Value().batch->Run();
    Require(result && result.Value().l2ReadBytes == physical && result.Value().l2ReadShards == 1,
            "one physical read unit");
    Require(reserved.Value().batch->ProtectedBytes() == physical, "one physical pin budget");
    Require(reads == 1, "single read for all layers");

    Trans::Device device;
    Require(device.Setup(0).Success(), "simu device");
    auto allocator = device.MakeBuffer();
    auto stream = device.MakeStream();
    auto destination = allocator->MakeDeviceBuffer(config.blockSize);
    std::vector<unsigned char> initial(config.blockSize, 0xff), observed(config.blockSize);
    Require(stream->HostToDevice(initial.data(), destination.get(), initial.size()).Success(), "initial HBM");
    Detail::TaskDesc descriptions;
    descriptions.loadReservations = {reserved.Value().handle};
    for (size_t layer = 0; layer < 14; ++layer) {
        auto* target = static_cast<std::byte*>(destination.get()) + layer * config.shardSize;
        descriptions.push_back({f.block, layer, {target, f.lengths[layer] > 73728 ? target + 73728 : nullptr}});
    }
    auto handles = prefetch.LoadHandles(descriptions);
    Require(handles.size() == 14, "all layer receipts prepared");
    const auto positions = EarlyBufferPins::Locations(handles);
    size_t offset = 0;
    auto* begin = static_cast<std::byte*>(handles.front().Data());
    for (size_t layer = 0; layer < 14; ++layer) {
        Require(positions[layer] == positions.front(), "one native cache node");
        Require(handles[layer].Data() == begin + offset && handles[layer].Ready(), "packed layer view");
        offset += f.lengths[layer];
    }
    Require(offset == physical, "packed views cover the physical block");
    prefetch.Release(reserved.Value().handle);
    reserved.Value().batch.reset();
    auto contender = Other(f.block, 1);
    Require(!buffer.TryGet(contender, 0), "prepared layer views retain whole block after reservation release");
    UC::CacheStore::TransManager native;
    Require(native.Setup(config, &buffer).Success(), "unchanged layerwise native queue");
    for (size_t layer = 0; layer < 14; ++layer) {
        UC::CacheStore::TransTask task{UC::CacheStore::TransTask::Type::LOAD, {descriptions[layer]}};
        task.loadHandles.resize(1);
        task.loadHandles[0] = std::move(handles[layer]);
        auto submitted = native.Submit(std::move(task));
        Require(submitted && native.Wait(submitted.Value()).Success(), "individual layer H2D");
        Require(stream->DeviceToHost(destination.get(), observed.data(), observed.size()).Success(), "HBM readback");
        for (size_t index = 0; index < 14; ++index) {
            for (size_t byte = 0; byte < config.shardSize; ++byte) {
                auto expected = index <= layer && byte < static_cast<size_t>(f.lengths[index]) ? index + 1 : 0xff;
                Require(observed[index * config.shardSize + byte] == expected, "future layers stay untouched; selected layers match");
            }
        }
        if (layer != 13) { Require(!buffer.TryGet(contender, 0), "later layer references prevent whole-block eviction"); }
    }
    Require(reads == 1, "H2D views do not reload L2");
    Require(bool(buffer.TryGet(contender, 0)), "node reusable after final layer receipt drains");
}

// Exercise the real native queue when early prefetch did not provide receipts,
// including retry around an incomplete dump held by another consumer.
static void NativeColdLoad(bool partialDump)
{
    PackedFixture f;
    auto config = BlockConfig(f, 1);
    BufferManager manager;
    Require(manager.Setup(config).Success(), "native cold cache setup");
    auto& buffer = *manager.GetTransBuffer();
    TransBuffer::Handle writer;
    if (partialDump) {
        writer = EarlyBufferPins::Acquire(buffer, f.block, 0, false);
        memcpy(writer.Data(), f.buffers[0].get(), f.lengths[0]);
        writer.MarkReady();
    }
    std::atomic<size_t> reads{0};
    TestHooks::SetAioSubmitHook([&](aio_context_t ctx, int64_t nr, iocb** cb) {
        Require(nr == 1 && cb[0]->aio_lio_opcode == IOCB_CMD_PREAD, "native cold load uses one PREAD");
        Require(cb[0]->aio_nbytes == f.store->FullBlockReadBytes(), "native cold load reads packed block");
        ++reads;
        return static_cast<int32_t>(syscall(SYS_io_submit, ctx, nr, cb));
    });
    Trans::Device device;
    Require(device.Setup(0).Success(), "cold load device");
    auto allocator = device.MakeBuffer();
    auto stream = device.MakeStream();
    auto destination = allocator->MakeDeviceBuffer(config.shardSize);
    UC::CacheStore::TransManager native;
    Require(native.Setup(config, &buffer).Success(), "native cold queue");
    // Start with a missing, nonzero layer to exercise its packed byte offset.
    constexpr size_t first = 6;
    auto* target = static_cast<std::byte*>(destination.get());
    auto task = native.Submit({UC::CacheStore::TransTask::Type::LOAD,
        {{f.block, first, {target, target + 73728}}}});
    Require(bool(task), "cold task accepted");
    if (partialDump) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        Require(reads == 0, "live partial dump must not be overwritten");
        writer = {};
    }
    Require(native.Wait(task.Value()).Success(), "cold load completes after existing views drain");
    std::vector<unsigned char> bytes(config.shardSize);
    Require(stream->DeviceToHost(target, bytes.data(), bytes.size()).Success(), "cold layer readback");
    for (auto byte : bytes) { Require(byte == first + 1, "cold load uses selected layer offset"); }
    Require(buffer.Exist(f.block, 0), "native full read publishes all layers");
    for (size_t layer = 0; layer < 14; ++layer) {
        auto next = native.Submit({UC::CacheStore::TransTask::Type::LOAD,
            {{f.block, layer, {target, f.lengths[layer] > 73728 ? target + 73728 : nullptr}}}});
        Require(next && native.Wait(next.Value()).Success(), "later layer consumes resident block");
        Require(stream->DeviceToHost(target, bytes.data(), bytes.size()).Success(), "later layer readback");
        for (size_t i = 0; i < static_cast<size_t>(f.lengths[layer]); ++i) {
            Require(bytes[i] == layer + 1, "resident layer data matches");
        }
    }
    Require(reads == 1, "all cold fallback layers share the same L2 read");
}

static void LayerClaimsAndPublication()
{
    PackedFixture f;
    auto config = BlockConfig(f, 1);
    BufferManager manager;
    Require(manager.Setup(config).Success(), "layer fill setup");
    auto& buffer = *manager.GetTransBuffer();
    auto block = Other(f.block, 2);
    TransBuffer::Handle retained;
    for (size_t layer = 0; layer < 14; ++layer) {
        auto writer = EarlyBufferPins::Acquire(buffer, block, layer, false);
        Require(writer && writer.Owner() && !writer.Ready(), "each new dump layer owns its slice");
        auto duplicate = EarlyBufferPins::Acquire(buffer, block, layer, false);
        Require(duplicate && !duplicate.Owner(), "same-layer writer coalesced");
        memset(writer.Data(), layer + 1, f.lengths[layer]);
        writer.MarkReady();
        Require(writer.Ready() && duplicate.Ready(), "layer READY publication");
        writer.MarkReady();  // must not count one layer twice
        if (!layer) { retained = writer; }
        auto whole = buffer.TryGet(block, 0);
        Require(layer == 13 ? bool(whole) && whole.Ready() : !whole,
                "whole-block reader retries while a partial dump remains referenced");
        Require(buffer.Exist(block, 0) == (layer == 13), "scheduler hides partial L1 blocks");
        buffer.Prealloc(block, (layer + 1) % 14);
        auto locations = EarlyBufferPins::Locations({retained, writer});
        Require(locations[0] == locations[1], "prealloc keeps one native node");
    }
    auto next = Other(f.block, 3);
    Require(!buffer.TryGet(next, 0), "retained layer still pins complete block");
    retained = {};
    auto root = buffer.TryGet(next, 0);
    Require(root && root.Owner(), "released block evicted atomically");
    root.MarkFailed(Status::Error("test read failure"));
    auto observer = buffer.TryGet(next, 0);
    Require(observer && observer.GetState() == TransBuffer::State::FAILED, "failure not reset with live views");
    root = {}; observer = {};
    auto retry = buffer.TryGet(next, 0);
    Require(retry && retry.Owner() && !retry.Ready(), "retry after complete reference drain");
}

static void PartialDumpDoesNotTrapWholeBlockPrefetch()
{
    PackedFixture f;
    auto config = BlockConfig(f, 1);
    BufferManager manager;
    Require(manager.Setup(config).Success(), "partial dump prefetch setup");
    auto& buffer = *manager.GetTransBuffer();
    auto writer = EarlyBufferPins::Acquire(buffer, f.block, 0, false);
    writer.MarkReady();
    Require(!buffer.TryGet(f.block, 0), "live partial dump cannot be overwritten");
    auto probe = EarlyBufferPins::PinBlocks(buffer, &f.block, 1, 1);
    Require(!probe.pins[0], "localization must not retain a partial dump indefinitely");
    Require(!EarlyBufferPins::Acquire(buffer, f.block, 1, true), "missing layer retries around live views");
    auto readyLayer = EarlyBufferPins::Acquire(buffer, f.block, 0, true);
    Require(readyLayer && readyLayer.Ready(), "ready partial layer remains consumable");
    readyLayer = {}; writer = {};
    const auto bytes = f.store->FullBlockReadBytes();
    ReservedPrefetch prefetch(&manager, f.store.get(), bytes, bytes, 5000, true);
    auto reserved = prefetch.Reserve(&f.block, 1);
    Require(bool(reserved), "retry whole prefetch after partial writer drained");
    auto completed = reserved.Value().batch->Run();
    Require(completed && completed.Value().l2ReadBytes == bytes, "partial cached block reloads and completes");
    Require(buffer.Exist(f.block, 0), "reloaded block published as complete");
    prefetch.Release(reserved.Value().handle);
}

static void LayerFailureCannotPublishWhole()
{
    PackedFixture f;
    auto config = BlockConfig(f, 1);
    BufferManager manager;
    Require(manager.Setup(config).Success(), "failure setup");
    auto& buffer = *manager.GetTransBuffer();
    auto a = EarlyBufferPins::Acquire(buffer, f.block, 0, false);
    auto b = EarlyBufferPins::Acquire(buffer, f.block, 1, false);
    Require(a.Owner() && b.Owner(), "independent layer producers");
    a.MarkFailed(Status::Error("D2H failed"));
    b.MarkReady();
    auto whole = buffer.TryGet(f.block, 0);
    Require(whole.GetState() == TransBuffer::State::FAILED && !whole.Ready(), "later completion cannot erase failure");
}

static void NativeLayerwiseDump()
{
    PackedFixture f;
    auto config = BlockConfig(f, 2);
    BufferManager manager;
    Require(manager.Setup(config).Success(), "native dump setup");
    auto& buffer = *manager.GetTransBuffer();
    UC::CacheStore::TransManager native;
    Require(native.Setup(config, &buffer).Success(), "native dump queue");
    Trans::Device device;
    Require(device.Setup(0).Success(), "dump device");
    auto allocator = device.MakeBuffer();
    auto stream = device.MakeStream();
    auto source = allocator->MakeDeviceBuffer(config.blockSize);
    auto block = Other(f.block, 4);
    for (size_t layer = 0; layer < 14; ++layer) {
        auto* address = static_cast<std::byte*>(source.get()) + layer * config.shardSize;
        Require(stream->HostToDevice(f.buffers[layer].get(), address, config.shardSize).Success(), "dump source");
        Detail::TaskDesc desc{{block, layer, {address, f.lengths[layer] > 73728 ? address + 73728 : nullptr}}};
        auto task = native.Submit({UC::CacheStore::TransTask::Type::DUMP, std::move(desc)});
        Require(task && native.Wait(task.Value()).Success(), "original per-layer D2H and L2 dump");
        auto view = buffer.Get(block, layer, false, true);
        Require(view.Ready(), "completed layer readable while later layers are absent");
        auto* data = static_cast<unsigned char*>(view.Data());
        for (size_t i = 0; i < static_cast<size_t>(f.lengths[layer]); ++i) {
            Require(data[i] == layer + 1, "D2H packed offset/content");
        }
        Require(buffer.Exist(block, 0) == (layer == 13), "native dump whole READY boundary");
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!f.store->Lookup(&block, 1).Value()[0]) {
        Require(std::chrono::steady_clock::now() < deadline, "native dump file committed");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    auto read = f.Read();
    read[0].owner = block;
    auto task = f.store->Load(std::move(read));
    Require(task && f.store->Wait(task.Value()).Success(), "read back persisted layerwise dump");
    f.Verify();
}

static void SharedLayoutAndWatcher()
{
    PackedFixture f;
    auto config = BlockConfig(f, 3, true);
    BufferManager manager;
    Require(manager.Setup(config).Success(), "shared block setup");
    auto& buffer = *manager.GetTransBuffer();
    auto root = buffer.TryGet(f.block, 0);
    Require(root && root.Owner(), "shared whole-block owner");
    root.MarkReady();
    {
        BufferManager peer;
        Require(peer.Setup(config).Success(), "same-layout shared attach");
        auto view = peer.GetTransBuffer()->Get(f.block, 9, true, true);
        Require(view.Ready(), "shared peer observes whole readiness");
    }
    {
        BufferManager bad;
        auto mismatch = config;
        mismatch.loadExclusiveBufferNumber = 1;
        Require(bad.Setup(mismatch).Failure(), "shared layout mismatch rejected");
    }
    {
        BufferManager watcher;
        auto observing = config;
        observing.deviceId = -1;
        observing.shardSize = observing.blockSize = 0;
        Require(watcher.Setup(observing).Success(), "metadata-only watcher reads block layout");
        Require(watcher.GetTransBuffer()->Exist(f.block, 0), "watcher sees complete block");
    }
    BufferManager reattach;
    Require(reattach.Setup(config).Success(), "peer/watcher/error cleanup preserves creator backing");
}

static void CancelledBlockReadRetainsNode()
{
    PackedFixture f;
    auto config = BlockConfig(f, 1);
    BufferManager manager;
    Require(manager.Setup(config).Success(), "cancel block setup");
    auto& buffer = *manager.GetTransBuffer();
    const auto bytes = f.store->FullBlockReadBytes();
    ReservedPrefetch prefetch(&manager, f.store.get(), bytes, bytes, 5000, true);
    auto reserved = prefetch.Reserve(&f.block, 1);
    Require(bool(reserved), "cancel block reservation");
    auto batch = reserved.Value().batch;
    std::promise<std::pair<aio_context_t, iocb*>> accepted;
    TestHooks::SetAioSubmitHook([&](aio_context_t ctx, int64_t nr, iocb** cb) {
        Require(nr == 1 && cb[0]->aio_lio_opcode == IOCB_CMD_PREAD, "cancelled contiguous read");
        accepted.set_value({ctx, *cb});
        return static_cast<int32_t>(nr);
    });
    auto started = batch->Step(bytes);
    Require(started && !started.Value().done, "start deferred physical read");
    auto [ctx, cb] = accepted.get_future().get();
    prefetch.Release(reserved.Value().handle);
    auto pending = batch->Step(0);
    const bool retained = pending && !pending.Value().done && batch->ProtectedBytes() == bytes;
    const bool unreusable = !buffer.TryGet(Other(f.block, 7), 0);
    Require(syscall(SYS_io_submit, ctx, 1, &cb) == 1, "physically finish accepted PREAD");
    Require(retained && unreusable, "cancel must retain the one physical node");
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    for (;;) {
        auto state = batch->Step(0);
        if (!state) { break; }
        Require(std::chrono::steady_clock::now() < deadline, "cancelled batch drains");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    Require(batch->ProtectedBytes() == 0 && bool(buffer.TryGet(Other(f.block, 7), 0)),
            "whole node reusable only after accepted I/O drains");
}

static void ShortContiguousReadFailsEveryView()
{
    PackedFixture f;
    const auto bytes = f.store->FullBlockReadBytes();
    bool truncated = false;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(f.path)) {
        if (entry.is_regular_file() && entry.file_size() == bytes) {
            std::filesystem::resize_file(entry.path(), 4096);
            truncated = true;
        }
    }
    Require(truncated, "owned packed file truncated");
    auto config = BlockConfig(f, 1);
    BufferManager manager;
    Require(manager.Setup(config).Success(), "short read cache setup");
    auto& buffer = *manager.GetTransBuffer();
    ReservedPrefetch prefetch(&manager, f.store.get(), bytes, bytes, 5000, true);
    auto reserved = prefetch.Reserve(&f.block, 1);
    Require(bool(reserved), "short read reservation");
    auto result = reserved.Value().batch->Run();
    Require(!result && result.Error() == Status::NotFound(), "short contiguous read rejected");
    Require(!buffer.Exist(f.block, 0), "short read cannot publish a whole L1 hit");
    auto retained = buffer.TryGet(f.block, 0);
    Require(retained.GetState() == TransBuffer::State::FAILED, "failed block retained by reservation");
    for (size_t layer = 0; layer < 14; ++layer) {
        Require(retained.LayerView(layer).GetState() == TransBuffer::State::FAILED,
                "no layer may expose a short whole-block read");
    }
    prefetch.Release(reserved.Value().handle);
}

int main()
{
    try {
        PrefetchThenLayerwiseH2D(false);
        PrefetchThenLayerwiseH2D(true);
        NativeColdLoad(false);
        NativeColdLoad(true);
        LayerClaimsAndPublication();
        PartialDumpDoesNotTrapWholeBlockPrefetch();
        LayerFailureCannotPublishWhole();
        NativeLayerwiseDump();
        SharedLayoutAndWatcher();
        CancelledBlockReadRetainsNode();
        ShortContiguousReadFailsEveryView();
        std::cout << "block L1 probe passed: PREAD, packed layer views, H2D ordering, pins, layer dumps, failure and sharing\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "block L1 probe failed: " << error.what() << '\n';
        return 1;
    }
}
