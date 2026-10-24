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
#include "trans/device.h"
#include <fstream>
#include <numeric>

using namespace UC;
using namespace UC::PosixStore;

static void Require(bool ok, const char* why)
{
    if (!ok) { throw std::runtime_error(why); }
}

extern "C" StoreV1* MakeEarlyCacheStore();

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
        auto pattern = std::string("/tmp/ucm-packed-probe-") + std::to_string(getpid()) + "-XXXXXX";
        path = mkdtemp(pattern.data());
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

static void PackedRead(bool partial, const std::string& writer)
{
    PackedFixture f(writer);
    size_t files = 0;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(f.path)) {
        if (!entry.is_regular_file() || entry.file_size() != 1093632) { continue; }
        ++files;
        std::ifstream file(entry.path(), std::ios::binary);
        for (size_t i = 0; i < 14; ++i) {
            std::vector<unsigned char> bytes(f.lengths[i]);
            Require(bool(file.read(reinterpret_cast<char*>(bytes.data()), bytes.size())), "raw packed file read");
            Require(std::all_of(bytes.begin(), bytes.end(), [i](auto value) { return value == i + 1; }),
                    "raw file prefix offsets");
        }
    }
    Require(files == 1, "exact physical file size");
    EarlyPosixStore reader;
    auto config = f.config;
    config.Set("posix_io_engine", std::string("aio"));
    Require(reader.Setup(config).Success(), "cross-engine reader");
    std::atomic<size_t> ops{0};
    TestHooks::SetAioSubmitHook([&](aio_context_t ctx, int64_t nr, iocb** cb) {
        Require(nr == 1 && cb[0]->aio_lio_opcode == IOCB_CMD_PREADV, "one packed preadv");
        Require(cb[0]->aio_nbytes == 14 && cb[0]->aio_offset == 0, "packed vector shape");
        auto* vectors = reinterpret_cast<iovec*>(cb[0]->aio_buf);
        for (size_t i = 0; i < 14; ++i) {
            Require(vectors[i].iov_len == static_cast<size_t>(f.lengths[i]), "actual vector length");
        }
        ++ops;
        return static_cast<int32_t>(syscall(SYS_io_submit, ctx, nr, cb));
    });
    auto read = reader.Load(f.Read(partial));
    Require(read && reader.Wait(read.Value()).Success(), "packed full read");
    Require(ops == 1, "single physical read");
    f.Verify(partial);
    TestHooks::ClearAioHooks();
    auto watcherConfig = f.config;
    watcherConfig.SetNumber("device_id", -1);
    watcherConfig.SetNumber("shard_size", 0);
    EarlyPosixStore watcher;
    Require(watcher.Setup(watcherConfig).Success(), "scheduler zero layout");
    Require(watcher.Lookup(&f.block, 1).Value()[0], "scheduler packed namespace");
    auto altered = f.lengths;
    std::swap(altered[0], altered[3]);
    watcherConfig.Set("posix_packed_layer_bytes", altered);
    EarlyPosixStore other;
    Require(other.Setup(watcherConfig).Success(), "different equal-sized packed layout");
    Require(!other.Lookup(&f.block, 1).Value()[0], "different offsets cannot alias old file");
    Detail::Dictionary uniform;
    uniform.SetNumber("device_id", -1);
    uniform.Set("storage_backends", std::vector<std::string>{f.path});
    uniform.Set("posix_gc_enable", false);
    EarlyPosixStore legacy;
    Require(legacy.Setup(uniform).Success(), "uniform namespace setup");
    Require(!legacy.Lookup(&f.block, 1).Value()[0], "uniform cannot see packed file");
}

static void PackedBudget(bool partial)
{
    PackedFixture f;
    UC::CacheStore::Config config;
    config.storeBackend = f.store.get();
    config.uniqueId = "packed-budget-probe";
    config.deviceId = 0;
    config.shardSize = PackedFixture::stride;
    config.blockSize = 14 * config.shardSize;
    config.bufferCapacity = 128 * config.shardSize;
    config.loadExclusiveBufferNumber = 0;
    config.shareBufferEnable = false;
    config.localRankSize = 1;
    config.cacheFullBlockLoad = true;
    config.timeoutMs = 5000;
    config.tensorSizes = {73728, 8448};
    UC::CacheStore::BufferManager manager;
    Require(manager.Setup(config).Success(), "packed buffer setup");
    auto& buffer = *manager.GetTransBuffer();
    if (partial) {
        auto pin = buffer.Get(f.block, 3);
        memset(pin.Data(), 0x99, PackedFixture::stride);
        pin.MarkReady();
    }
    UC::CacheStore::ReservedPrefetch prefetch(&manager, f.store.get(), config.blockSize,
                                              config.shardSize, config.timeoutMs, true);
    auto result = prefetch.Reserve(&f.block, 1);
    Require(bool(result), "packed reservation");
    auto batch = result.Value().batch;
    Require(batch->shardBytes == 1093632 && batch->windowBytes == 128 * 1093632,
            "physical disk budget unit");
    auto shortBudget = batch->Step(1093631);
    Require(shortBudget && shortBudget.Value().inflightBytes == 0 &&
            shortBudget.Value().result.l2ReadBytes == 0, "no fractional packed submission");
    auto state = batch->Step(1093632);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (state && !state.Value().done) {
        Require(std::chrono::steady_clock::now() < deadline, "packed reservation completion");
        state = batch->Step(0);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    Require(state && state.Value().result.l2ReadBytes == 1093632 &&
            state.Value().result.l2ReadShards == 14 && state.Value().completedReadBytes == 1093632,
            "packed byte and shard accounting");
    Require(batch->ProtectedBytes() == 14 * PackedFixture::stride, "L1 pins keep uniform capacity");
    for (size_t layer = 0; layer < 14; ++layer) {
        auto pin = buffer.Get(f.block, layer);
        Require(pin.Ready(), "packed L1 publication");
        auto* bytes = static_cast<const unsigned char*>(pin.Data());
        for (size_t j = 0; j < static_cast<size_t>(f.lengths[layer]); ++j) {
            Require(bytes[j] == (partial && layer == 3 ? 0x99 : layer + 1), "shared L1 preserved");
        }
    }
    prefetch.Release(result.Value().handle);
}

static void PackedTimeout()
{
    PackedFixture f("aio", 50);
    std::promise<std::pair<aio_context_t, iocb*>> accepted;
    TestHooks::SetAioSubmitHook([&](aio_context_t ctx, int64_t nr, iocb** cb) {
        accepted.set_value({ctx, *cb});
        return static_cast<int32_t>(nr);
    });
    TestHooks::SetAioCancelHook([](aio_context_t, iocb*, io_event*) { errno = EAGAIN; return -1; });
    auto read = f.store->Load(f.Read(true));
    Require(bool(read), "packed delayed read accepted");
    auto [ctx, cb] = accepted.get_future().get();
    auto wait = std::async(std::launch::async, [&] { return f.store->Wait(read.Value()); });
    Require(wait.wait_for(std::chrono::milliseconds(150)) == std::future_status::timeout,
            "packed scratch retained after timeout");
    Require(syscall(SYS_io_submit, ctx, 1, &cb) == 1, "delayed packed kernel read");
    Require(wait.get() == Status::Timeout(), "packed timeout after physical drain");
    f.Verify(true);
}

static void RejectRealTail()
{
    PackedFixture f;
    auto config = f.config;
    config.Set<StoreV1*>("store_backend", f.store.get());
    config.SetNumber("tensor_size", 0);
    config.Set("tensor_size_list", std::vector<ssize_t>{73728, 8448});
    config.Set("unique_id", std::string("packed-tail-probe"));
    config.Set("share_buffer_enable", false);
    config.Set("cache_full_block_load", true);
    config.SetNumber("cache_buffer_capacity_gb", 1);
    config.SetNumber("local_rank_size", 1);
    std::unique_ptr<StoreV1> cache{MakeEarlyCacheStore()};
    Require(cache->Setup(config).Success(), "packed native cache setup");
    Detail::TaskDesc invalid{{f.block, 3, {f.buffers[3].get(), f.buffers[0].get()}}};
    Require(!cache->Dump(invalid) && !cache->Load(invalid), "real tensor tail rejected before I/O");
    invalid[0].index = 14;
    Require(!cache->Dump(invalid) && !cache->Load(invalid), "out-of-range layer rejected");
    invalid[0].index = 0;
    invalid[0].addrs.pop_back();
    Require(!cache->Dump(invalid) && !cache->Load(invalid), "wrong tensor count rejected");
}

int main()
{
    try {
        PackedRead(false, "aio");
        PackedRead(true, "aio");
        PackedRead(false, "psync");
        PackedBudget(false);
        PackedBudget(true);
        PackedTimeout();
        RejectRealTail();
        std::cout << "PASS packed L2: raw offsets, cross-engine IO, isolation, scheduler, budgets, pin/scratch lifetime, tensor rejection\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL packed L2: " << error.what() << '\n';
        return 1;
    }
}
