#include <array>
#include <filesystem>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <unistd.h>
#include "cache/cc/block_dump_queue.h"
#include "cache/cc/buffer_manager.h"

extern "C" UC::StoreV1* MakeEarlyPosixStore();

namespace {
using namespace UC;
using CacheStore::TransTask;
constexpr size_t kLayerBytes = 4096;

void Require(bool ok, const char* why)
{
    if (!ok) { throw std::runtime_error(why); }
}

Detail::BlockId Id(unsigned value)
{
    Detail::BlockId id{};
    id[0] = static_cast<std::byte>(value);
    return id;
}
}

int main(int argc, char** argv)
try {
    Require(argc == 2, "temporary data directory required");
    std::filesystem::create_directories(argv[1]);
    Detail::Dictionary config;
    config.Set("storage_backends", std::vector<std::string>{argv[1]});
    config.SetNumber("device_id", 0);
    config.SetNumber("tensor_size", kLayerBytes);
    config.SetNumber("shard_size", kLayerBytes);
    config.SetNumber("block_size", 2 * kLayerBytes);
    config.SetNumber("data_dir_shard_bytes", 1);
    config.SetNumber("posix_data_trans_concurrency", 1);
    config.SetNumber("posix_lookup_concurrency", 1);
    config.Set("posix_io_engine", std::string{"aio"});
    config.Set("io_direct", true);
    std::unique_ptr<StoreV1> backend(MakeEarlyPosixStore());
    Require(backend->Setup(config).Success(), "backend setup");
    CacheStore::Config cache;
    cache.uniqueId = "writeback-headroom-" + std::to_string(getpid());
    cache.deviceId = 0;
    cache.tensorSizes = {kLayerBytes};
    cache.shardSize = kLayerBytes;
    cache.blockSize = 2 * kLayerBytes;
    cache.bufferCapacity = cache.blockSize * 8;
    cache.loadExclusiveBufferNumber = 0;
    cache.shareBufferEnable = false;
    cache.ioDirect = true;
    cache.streamNumber = 1;
    cache.cacheFullBlockL1 = cache.cacheFullBlockLoad = cache.cacheFullBlockDump = true;
    cache.cacheDumpReservedBlocks = 0;
    cache.cacheDumpCapacityPolicy = "wait";
    cache.storeBackend = backend.get();
    cache.timeoutMs = 3000;
    CacheStore::BufferManager manager;
    Require(manager.Setup(cache).Success(), "L1 setup");
    CacheStore::BlockDumpQueue queue;
    Require(queue.Setup(cache, manager.GetTransBuffer()).Success(), "writeback setup");
    std::vector<CacheStore::TransBuffer::Handle> futureReady;
    for (unsigned block = 1; block <= 5; ++block) {
        for (size_t layer = 0; layer < 2; ++layer) {
            auto pin = CacheStore::EarlyBufferPins::Acquire(*manager.GetTransBuffer(), Id(block), layer, false);
            Require(bool(pin), "prefetch protection");
            pin.MarkReady();
            if (layer == 0) { futureReady.push_back(pin.WholeView()); }
        }
    }
    std::array<std::array<char, kLayerBytes>, 3> source;
    for (size_t i = 0; i < source.size(); ++i) { source[i].fill('a' + i); }
    for (size_t layer = 0; layer < 2; ++layer) {
        Detail::TaskDesc desc;
        desc.finalizeBlockDump = layer == 1;
        for (size_t i = 0; i < source.size(); ++i) {
            desc.push_back({Id(10 + i), layer, {source[i].data()}});
        }
        auto submitted = queue.Submit({TransTask::Type::DUMP, std::move(desc)});
        Require(bool(submitted), "writeback admission");
        auto wait = std::async(std::launch::async, [&] {
            return queue.WaitSources(submitted.Value());
        });
        Require(wait.wait_for(std::chrono::seconds(2)) == std::future_status::ready,
                "current layer needs progress while future READY pins remain held");
        Require(wait.get().Success(), "source copies drained");
        Require(queue.Wait(submitted.Value()).Success(), "writeback completed");
    }
    // The test never releases or mutates any future request's pins to unblock
    // the current compute batch. All three independently persisted blocks exist.
    Require(futureReady.size() == 5, "future prefetch pins retained");
    for (unsigned block = 10; block < 13; ++block) {
        auto id = Id(block);
        auto hit = backend->LookupOnPrefix(&id, 1);
        Require(hit && hit.Value() == 0, "all layers persisted");
    }
    std::cout << "writeback progressed with five future blocks held and three HBM slots\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
