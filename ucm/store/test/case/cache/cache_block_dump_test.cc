#include <sys/syscall.h>
#include <unistd.h>
#include <array>
#include <atomic>
#include <cstring>
#include <filesystem>
#include <mutex>
#include "cache/cc/block_dump_queue.h"
#include "cache/cc/buffer_manager.h"
#include "detail/mock_store.h"
#include "detail/path_base.h"
#include "posix/cc/aio_impl.h"
#include "posix/cc/space_layout.h"

extern "C" UC::StoreV1* MakeEarlyPosixStore();
extern "C" UC::StoreV1* MakeEarlyCacheStore();

namespace {
using namespace UC;
using CacheStore::TransTask;
constexpr size_t kBytes = 8192;

Detail::BlockId Id(unsigned char value)
{
    Detail::BlockId id{};
    id[0] = std::byte{value};
    return id;
}

Detail::Dictionary SchedulerBackendConfig(const std::string& path)
{
    Detail::Dictionary config;
    config.Set("storage_backends", std::vector<std::string>{path});
    config.SetNumber("device_id", -1);
    config.SetNumber("data_dir_shard_bytes", 1);
    config.SetNumber("posix_lookup_concurrency", 1);
    config.Set("posix_io_engine", std::string{"aio"});
    return config;
}

Detail::Dictionary BlockDumpConfig(StoreV1* backend)
{
    Detail::Dictionary config;
    config.Set("store_backend", backend);
    config.Set("unique_id", UC::Test::Detail::Random().RandomString(16));
    config.SetNumber("device_id", -1);
    config.Set("share_buffer_enable", false);
    config.Set("cache_full_block_load", true);
    config.Set("cache_full_block_l1", true);
    config.Set("cache_full_block_dump", true);
    return config;
}

class WriteTrace {
    std::mutex mutex_;
public:
    std::vector<std::pair<size_t, size_t>> writes;
    std::atomic_bool fail{false};
    std::atomic_bool shortWrite{false};
    std::function<void()> beforeCompletion;
    WriteTrace()
    {
        PosixStore::TestHooks::SetAioSubmitHook([this](auto context, auto count, auto requests) {
            for (int64_t i = 0; i < count; ++i) {
                if (requests[i]->aio_lio_opcode == IOCB_CMD_PWRITE) {
                    if (fail.load()) { errno = EIO; return int32_t{-1}; }
                    if (shortWrite.load() || beforeCompletion) {
                        auto* callback = reinterpret_cast<PosixStore::AioImpl::Callback*>(
                            requests[i]->aio_data);
                        *callback = [this, original = std::move(*callback)](auto result) {
                            if (beforeCompletion) { beforeCompletion(); }
                            if (shortWrite.load() && result.error == 0 && result.nBytes > 0) {
                                --result.nBytes;
                            }
                            original(result);
                        };
                    }
                    std::lock_guard lock(mutex_);
                    writes.emplace_back(requests[i]->aio_offset, requests[i]->aio_nbytes);
                }
            }
            return static_cast<int32_t>(syscall(SYS_io_submit, context, count, requests));
        });
    }
    ~WriteTrace() { PosixStore::TestHooks::ClearAioHooks(); }
};

class UCCacheBlockDumpTest : public UC::Test::Detail::PathBase {
protected:
    std::unique_ptr<StoreV1> backend;
    std::unique_ptr<CacheStore::BufferManager> manager;
    std::unique_ptr<CacheStore::BlockDumpQueue> queue;
    CacheStore::Config cache;
    std::vector<std::array<char, kBytes>> source;
    std::vector<size_t> sizes;

    void Start(size_t layers = 3, size_t budget = 0, bool packed = false, bool shared = false,
               std::string policy = "spill", size_t slots = 4)
    {
        Detail::Dictionary config;
        config.Set("storage_backends", std::vector<std::string>{Path()});
        config.SetNumber("device_id", 0);
        config.SetNumber("tensor_size", kBytes);
        config.SetNumber("shard_size", kBytes);
        config.SetNumber("block_size", kBytes * layers);
        config.SetNumber("data_dir_shard_bytes", 1);
        config.SetNumber("posix_data_trans_concurrency", 1);
        config.SetNumber("posix_lookup_concurrency", 1);
        config.SetNumber("posix_write_inflight_limit", 1);
        config.Set("posix_io_engine", std::string{"aio"});
        config.Set("io_direct", false);
        sizes.assign(layers, kBytes);
        if (packed) {
            std::vector<ssize_t> lengths(layers);
            for (size_t i = 0; i < layers; ++i) { lengths[i] = sizes[i] = i % 2 ? 4096 : kBytes; }
            config.Set("posix_packed_layer_bytes", lengths);
        }
        backend.reset(MakeEarlyPosixStore());
        ASSERT_EQ(backend->Setup(config), Status::OK());
        cache.uniqueId = UC::Test::Detail::Random().RandomString(16);
        cache.deviceId = 0;
        cache.tensorSizes = {4096, 4096};
        cache.shardSize = kBytes;
        cache.blockSize = kBytes * layers;
        cache.bufferCapacity = cache.blockSize * slots;
        cache.loadExclusiveBufferNumber = 0;
        cache.shareBufferEnable = shared;
        cache.localRankSize = 1;
        cache.ioDirect = false;
        cache.streamNumber = 1;
        cache.cacheFullBlockL1 = cache.cacheFullBlockLoad = cache.cacheFullBlockDump = true;
        cache.cacheDumpReservedBlocks = budget;
        cache.cacheDumpCapacityPolicy = std::move(policy);
        cache.storeBackend = backend.get();
        cache.timeoutMs = 3000;
        manager = std::make_unique<CacheStore::BufferManager>();
        ASSERT_EQ(manager->Setup(cache), Status::OK());
        queue = std::make_unique<CacheStore::BlockDumpQueue>();
        ASSERT_EQ(queue->Setup(cache, manager->GetTransBuffer()), Status::OK());
        source.resize(layers);
        for (size_t i = 0; i < layers; ++i) { source[i].fill(static_cast<char>('a' + i)); }
    }

    Detail::TaskDesc Describe(const Detail::BlockId& id, std::vector<size_t> layers, bool finish)
    {
        Detail::TaskDesc desc;
        desc.finalizeBlockDump = finish;
        for (auto layer : layers) {
            desc.push_back({id, layer, {source[layer].data(),
                sizes[layer] == kBytes ? source[layer].data() + 4096 : nullptr}});
        }
        return desc;
    }

    Status Send(Detail::TaskDesc desc)
    {
        auto task = queue->Submit({TransTask::Type::DUMP, std::move(desc)});
        if (!task) { return task.Error(); }
        return queue->Wait(task.Value());
    }

    void Verify(const Detail::BlockId& id)
    {
        auto hit = backend->LookupOnPrefix(&id, 1);
        ASSERT_TRUE(hit);
        ASSERT_EQ(hit.Value(), 0);
        size_t bytes = 0;
        for (auto length : sizes) { bytes += length; }
        std::vector<char> result(bytes);
        Detail::TaskDesc load;
        load.fullBlockRead = true;
        load.push_back({id, 0, {result.data()}});
        auto task = backend->Load(std::move(load));
        ASSERT_TRUE(task);
        ASSERT_EQ(backend->Wait(task.Value()), Status::OK());
        size_t offset = 0;
        for (size_t i = 0; i < sizes.size(); ++i) {
            EXPECT_EQ(std::memcmp(result.data() + offset, source[i].data(), sizes[i]), 0);
            offset += sizes[i];
        }
    }

    void CheckZeroBudgetResidency(const std::string& policy)
    {
        EXPECT_EQ(CacheStore::Config{}.cacheDumpReservedBlocks, size_t{0});
        ASSERT_NO_FATAL_FAILURE(Start(3, 0, false, false, policy, 20));
        WriteTrace trace;
        for (unsigned char block = 100; block < 120; ++block) {
            ASSERT_EQ(Send(Describe(Id(block), {0}, false)), Status::OK());
        }
        EXPECT_TRUE(trace.writes.empty());
        for (unsigned char block = 100; block < 120; ++block) {
            auto pin = CacheStore::EarlyBufferPins::Acquire(
                *manager->GetTransBuffer(), Id(block), 0, false);
            ASSERT_TRUE(pin && pin.Ready());
            EXPECT_FALSE(pin.WholeView().Ready());
        }
        for (unsigned char block = 100; block < 120; ++block) {
            ASSERT_EQ(Send(Describe(Id(block), {0, 1, 2}, true)), Status::OK());
            Verify(Id(block));
        }
        ASSERT_EQ(trace.writes.size(), 20);
        for (const auto& write : trace.writes) {
            EXPECT_EQ(write, std::make_pair(size_t{0}, 3 * kBytes));
        }
    }

    void TearDown() override
    {
        queue.reset();
        manager.reset();
        backend.reset();
        PathBase::TearDown();
    }
};

TEST_F(UCCacheBlockDumpTest, FourteenLayersBecomeOneCompleteWrite)
{
    Start(14);
    WriteTrace trace;
    auto id = Id(1);
    for (size_t layer = 0; layer < 14; ++layer) {
        ASSERT_EQ(Send(Describe(id, {layer}, false)), Status::OK());
        if (layer < 13) { EXPECT_EQ(backend->LookupOnPrefix(&id, 1).Value(), -1); }
    }
    ASSERT_EQ(Send(Describe(id, {0}, true)), Status::OK());
    ASSERT_EQ(trace.writes.size(), 1);
    EXPECT_EQ(trace.writes[0], std::make_pair(size_t{0}, 14 * kBytes));
    Verify(id);
}

TEST_F(UCCacheBlockDumpTest, PackedFourteenLayersUsePhysicalBlockLength)
{
    Start(14, 16, true);
    WriteTrace trace;
    auto id = Id(2);
    for (size_t layer = 0; layer < 14; ++layer) {
        ASSERT_EQ(Send(Describe(id, {layer}, layer == 13)), Status::OK());
    }
    ASSERT_EQ(trace.writes.size(), 1);
    EXPECT_EQ(trace.writes[0].second, 7 * (kBytes + 4096));
    Verify(id);
}

TEST_F(UCCacheBlockDumpTest, FinalizationBackfillsMissingComputeReadyLayers)
{
    Start();
    WriteTrace trace;
    auto id = Id(3);
    ASSERT_EQ(Send(Describe(id, {0}, false)), Status::OK());
    ASSERT_EQ(Send(Describe(id, {0, 1, 2}, true)), Status::OK());
    EXPECT_EQ(trace.writes.size(), 1);
    Verify(id);
}

TEST_F(UCCacheBlockDumpTest, MissingUnvisitedLayerIsSpilledButNeverPublished)
{
    Start();
    WriteTrace trace;
    auto id = Id(4);
    ASSERT_EQ(Send(Describe(id, {0}, false)), Status::OK());
    EXPECT_EQ(Send(Describe(id, {0, 2}, true)), Status::NotFound());
    EXPECT_EQ(backend->LookupOnPrefix(&id, 1).Value(), -1);
    ASSERT_EQ(trace.writes.size(), 2);
    EXPECT_EQ(trace.writes[0], std::make_pair(size_t{0}, kBytes));
    EXPECT_EQ(trace.writes[1], std::make_pair(2 * kBytes, kBytes));
    // A later complete source may repair the unpublished partial file.
    ASSERT_EQ(Send(Describe(id, {0, 1, 2}, true)), Status::OK());
    Verify(id);
}

TEST_F(UCCacheBlockDumpTest, OneReservationMakesProgressWithMoreBlocksThanL1Slots)
{
    Start(3, 1);
    WriteTrace trace;
    for (size_t layer = 0; layer < 3; ++layer) {
        for (unsigned char block = 10; block < 18; ++block) {
            ASSERT_EQ(Send(Describe(Id(block), {layer}, false)), Status::OK());
        }
    }
    for (unsigned char block = 10; block < 18; ++block) {
        auto status = Send(Describe(Id(block), {0, 1, 2}, true));
        ASSERT_EQ(status, Status::OK()) << "block=" << unsigned(block) << " " << status.ToString();
        Verify(Id(block));
    }
    EXPECT_GT(trace.writes.size(), 8);
}

TEST_F(UCCacheBlockDumpTest, FailedIoDoesNotPublishAndCanRetryWhileSourceIsValid)
{
    Start();
    WriteTrace trace;
    auto id = Id(5);
    trace.fail = true;
    EXPECT_TRUE(Send(Describe(id, {0, 1, 2}, true)).Failure());
    EXPECT_EQ(backend->LookupOnPrefix(&id, 1).Value(), -1);
    trace.fail = false;
    ASSERT_EQ(Send(Describe(id, {0, 1, 2}, true)), Status::OK());
    Verify(id);
}

TEST_F(UCCacheBlockDumpTest, WriterLeaseIsExclusiveWithoutBlockingReaders)
{
    Start();
    auto id = Id(6);
    auto first = backend->AcquireWriteLease(id);
    ASSERT_TRUE(first);
    EXPECT_FALSE(backend->AcquireWriteLease(id));
    EXPECT_TRUE(backend->AcquireReadLease(&id, 1));
    first.Value().reset();
    EXPECT_TRUE(backend->AcquireWriteLease(id));
}

TEST_F(UCCacheBlockDumpTest, BusyOrdinarySlotsDeferThenRetryWithoutSpinning)
{
    Start();
    std::vector<CacheStore::TransBuffer::Handle> occupied;
    for (unsigned char i = 20; i < 24; ++i) {
        auto pin = CacheStore::EarlyBufferPins::Acquire(*manager->GetTransBuffer(), Id(i), 0, false);
        ASSERT_TRUE(pin);
        occupied.push_back(std::move(pin));
    }
    auto id = Id(7);
    ASSERT_EQ(Send(Describe(id, {0}, false)), Status::OK());
    EXPECT_EQ(backend->LookupOnPrefix(&id, 1).Value(), -1);
    occupied.clear();
    ASSERT_EQ(Send(Describe(id, {0, 1, 2}, true)), Status::OK());
    Verify(id);
}

TEST_F(UCCacheBlockDumpTest, ShutdownDrainsAcceptedTasksAndSpillsPartialData)
{
    Start();
    WriteTrace trace;
    auto id = Id(8);
    auto task = queue->Submit({TransTask::Type::DUMP, Describe(id, {0}, false)});
    ASSERT_TRUE(task);
    queue.reset();
    ASSERT_EQ(trace.writes.size(), 1);
    EXPECT_EQ(backend->LookupOnPrefix(&id, 1).Value(), -1);
    EXPECT_TRUE(backend->AcquireWriteLease(id));
}
TEST_F(UCCacheBlockDumpTest, SharedWholeBlockRetainsLayersUntilComplete)
{
    Start(3, 2, false, true);
    auto id = Id(30);
    ASSERT_EQ(Send(Describe(id, {2}, false)), Status::OK());
    ASSERT_EQ(Send(Describe(id, {0}, false)), Status::OK());
    ASSERT_EQ(Send(Describe(id, {1}, true)), Status::OK());
    Verify(id);
}

TEST_F(UCCacheBlockDumpTest, InvalidLayerAndTensorShapesFailBeforeDma)
{
    Start();
    auto desc = Describe(Id(31), {0}, false);
    desc[0].index = 3;
    EXPECT_FALSE(queue->Submit({TransTask::Type::DUMP, desc}));
    desc[0].index = 0;
    desc[0].addrs.clear();
    EXPECT_FALSE(queue->Submit({TransTask::Type::DUMP, desc}));
}

TEST_F(UCCacheBlockDumpTest, FactoryRoutesWriteCompletionWithoutChangingReadEntry)
{
    Start();
    Detail::Dictionary config;
    config.Set("store_backend", backend.get());
    config.Set("unique_id", UC::Test::Detail::Random().RandomString(16));
    config.SetNumber("device_id", 0);
    config.Set("tensor_size_list", std::vector<ssize_t>{4096, 4096});
    config.SetNumber("shard_size", kBytes);
    config.SetNumber("block_size", 3 * kBytes);
    config.SetNumber("cache_buffer_capacity_gb", 1);
    config.SetNumber("cache_load_exclusive_buffer_number", 0);
    config.SetNumber("cache_stream_number", 1);
    config.Set("share_buffer_enable", false);
    config.Set("io_direct", false);
    config.Set("cache_full_block_load", true);
    config.Set("cache_full_block_l1", true);
    config.Set("cache_full_block_dump", true);
    config.Set("cache_dump_capacity_policy", std::string{"wait"});
    config.SetNumber("cache_dump_reserved_blocks", 1);
    std::unique_ptr<StoreV1> store{MakeEarlyCacheStore()};
    ASSERT_EQ(store->Setup(config), Status::OK());
    auto id = Id(32);
    auto task = store->Dump(Describe(id, {0, 1, 2}, true));
    ASSERT_TRUE(task);
    ASSERT_EQ(store->Wait(task.Value()), Status::OK());
    Verify(id);
    std::array<char, 4096> first{}, second{};
    Detail::TaskDesc load;
    load.push_back({id, 1, {first.data(), second.data()}});
    auto read = store->Load(std::move(load));
    ASSERT_TRUE(read);
    ASSERT_EQ(store->Wait(read.Value()), Status::OK());
    EXPECT_EQ(std::memcmp(first.data(), source[1].data(), 4096), 0);
    EXPECT_EQ(std::memcmp(second.data(), source[1].data() + 4096, 4096), 0);
}

TEST_F(UCCacheBlockDumpTest, SchedulerStartsWithoutWorkerKvLayout)
{
    backend.reset(MakeEarlyPosixStore());
    ASSERT_EQ(backend->Setup(SchedulerBackendConfig(Path())), Status::OK());
    ASSERT_FALSE(backend->SupportsFullBlockRead());
    ASSERT_FALSE(backend->SupportsFullBlockWrite());
    for (const std::string policy : {"spill", "wait"}) {
        SCOPED_TRACE(policy);
        auto config = BlockDumpConfig(backend.get());
        config.Set("cache_dump_capacity_policy", policy);
        std::unique_ptr<StoreV1> scheduler{MakeEarlyCacheStore()};
        ASSERT_EQ(scheduler->Setup(config), Status::OK());
        const auto id = Id(90);
        auto hit = scheduler->LookupOnPrefix(&id, 1);
        ASSERT_TRUE(hit);
        EXPECT_EQ(hit.Value(), -1);
    }
}

TEST_F(UCCacheBlockDumpTest, SchedulerAttachesSharedBlockL1WithoutWorkerKvLayout)
{
    Start(3, 1, false, true);
    const auto id = Id(91);
    auto pin = CacheStore::EarlyBufferPins::Acquire(*manager->GetTransBuffer(), id, 0, true);
    ASSERT_TRUE(pin && pin.Owner());
    auto full = pin.WholeView();
    std::memset(full.Data(), 'x', 3 * kBytes);
    full.MarkReady();
    ASSERT_EQ(backend->LookupOnPrefix(&id, 1).Value(), -1);
    std::unique_ptr<StoreV1> schedulerBackend{MakeEarlyPosixStore()};
    auto backendConfig = SchedulerBackendConfig(Path());
    // GC shares the disk block size even without the worker's tensor layout.
    backendConfig.SetNumber("block_size", 3 * kBytes);
    auto setup = schedulerBackend->Setup(backendConfig);
    ASSERT_EQ(setup, Status::OK()) << setup.ToString();
    ASSERT_FALSE(schedulerBackend->SupportsFullBlockWrite());
    for (const std::string policy : {"spill", "wait"}) {
        SCOPED_TRACE(policy);
        auto config = BlockDumpConfig(schedulerBackend.get());
        config.Set("unique_id", cache.uniqueId);
        config.Set("share_buffer_enable", true);
        config.Set("cache_dump_capacity_policy", policy);
        std::unique_ptr<StoreV1> scheduler{MakeEarlyCacheStore()};
        ASSERT_EQ(scheduler->Setup(config), Status::OK());
        auto hit = scheduler->LookupOnPrefix(&id, 1);
        ASSERT_TRUE(hit);
        EXPECT_EQ(hit.Value(), 0);
    }
    EXPECT_TRUE(full.Ready());
}

TEST_F(UCCacheBlockDumpTest, SchedulerStillRejectsInvalidBlockDumpConfiguration)
{
    backend.reset(MakeEarlyPosixStore());
    ASSERT_EQ(backend->Setup(SchedulerBackendConfig(Path())), Status::OK());
    const std::vector<std::pair<std::string, std::function<void(Detail::Dictionary&)>>> invalid = {
        {"missing full L1", [](auto& c) { c.Set("cache_full_block_l1", false); }},
        {"missing full load", [](auto& c) { c.Set("cache_full_block_load", false); }},
        {"backend only", [](auto& c) { c.Set("cache_load_backend_only", true); }},
        {"unknown policy", [](auto& c) {
             c.Set("cache_dump_capacity_policy", std::string{"unknown"});
         }},
        {"wait without dump", [](auto& c) {
             c.Set("cache_dump_capacity_policy", std::string{"wait"});
             c.Set("cache_full_block_dump", false);
         }},
        {"GDR", [](auto& c) { c.Set("use_gdr", true); }},
        {"SDMA direct", [](auto& c) { c.Set("cache_sdma_direct", true); }},
    };
    for (const auto& [name, mutate] : invalid) {
        SCOPED_TRACE(name);
        auto config = BlockDumpConfig(backend.get());
        mutate(config);
        std::unique_ptr<StoreV1> scheduler{MakeEarlyCacheStore()};
        EXPECT_EQ(scheduler->Setup(config), Status::InvalidParam());
    }
}

// Build valid worker dimensions too, so a rejection cannot be explained by
// scheduler-only setup or missing KV layout rather than the requested option.
Detail::Dictionary IoAggregationConfig(StoreV1* backend, int32_t device)
{
    auto config = BlockDumpConfig(backend);
    config.SetNumber("device_id", device);
    config.SetNumber("tensor_size", kBytes);
    config.SetNumber("shard_size", kBytes);
    config.SetNumber("block_size", 3 * kBytes);
    config.SetNumber("cache_buffer_capacity_gb", 1);
    config.SetNumber("cache_load_exclusive_buffer_number", 0);
    config.SetNumber("cache_stream_number", 1);
    config.Set("io_direct", false);
    config.Set("cache_sdma_direct", false);
    return config;
}

TEST_F(UCCacheBlockDumpTest, FullBlockDumpRejectsRequestedIoAggregationForBothRoles)
{
    ASSERT_NO_FATAL_FAILURE(Start());
    for (int32_t device : {-1, 0}) {
        SCOPED_TRACE(device);
        auto config = IoAggregationConfig(backend.get(), device);
        config.Set("cache_io_aggregation", true);
        std::unique_ptr<StoreV1> store{MakeEarlyCacheStore()};
        EXPECT_EQ(store->Setup(config), Status::InvalidParam());
    }
}

#if !UCM_RUNTIME_ASCEND_IO_AGGREGATION
TEST_F(UCCacheBlockDumpTest, UnsupportedIoAggregationIsNotSilentlyDisabledWithoutBlockDump)
{
    ASSERT_NO_FATAL_FAILURE(Start());
    for (int32_t device : {-1, 0}) {
        SCOPED_TRACE(device);
        auto config = IoAggregationConfig(backend.get(), device);
        config.Set("cache_full_block_dump", false);
        config.Set("cache_io_aggregation", true);
        std::unique_ptr<StoreV1> store{MakeEarlyCacheStore()};
        auto status = store->Setup(config);
        EXPECT_EQ(status, Status::InvalidParam());
        EXPECT_NE(status.ToString().find("cache_io_aggregation"), std::string::npos);
    }
}
#endif

TEST_F(UCCacheBlockDumpTest, ExplicitlyDisabledIoAggregationSupportsBothRoles)
{
    ASSERT_NO_FATAL_FAILURE(Start());
    for (int32_t device : {-1, 0}) {
        SCOPED_TRACE(device);
        auto config = IoAggregationConfig(backend.get(), device);
        config.Set("cache_io_aggregation", false);
        std::unique_ptr<StoreV1> store{MakeEarlyCacheStore()};
        EXPECT_EQ(store->Setup(config), Status::OK());
    }
}

TEST_F(UCCacheBlockDumpTest, WorkerStillRequiresFullBlockWriteCapability)
{
    class ReadOnlyBackend : public testing::NiceMock<UC::Test::Detail::MockStore> {
    public:
        bool SupportsFullBlockRead() const override { return true; }
    } readOnly;
    ASSERT_FALSE(readOnly.SupportsFullBlockWrite());
    for (const std::string policy : {"spill", "wait"}) {
        SCOPED_TRACE(policy);
        auto config = BlockDumpConfig(&readOnly);
        config.SetNumber("device_id", 0);
        config.SetNumber("tensor_size", kBytes);
        config.SetNumber("shard_size", kBytes);
        config.SetNumber("block_size", 3 * kBytes);
        config.Set("cache_dump_capacity_policy", policy);
        std::unique_ptr<StoreV1> worker{MakeEarlyCacheStore()};
        auto status = worker->Setup(config);
        EXPECT_EQ(status, Status::InvalidParam());
        EXPECT_NE(status.ToString().find("full-block write backend"), std::string::npos);
    }
}

TEST_F(UCCacheBlockDumpTest, ShortWriteIsNotPublishedAsACompleteBlock)
{
    Start();
    WriteTrace trace;
    auto id = Id(33);
    trace.shortWrite = true;
    EXPECT_TRUE(Send(Describe(id, {0, 1, 2}, true)).Failure());
    EXPECT_EQ(backend->LookupOnPrefix(&id, 1).Value(), -1);
    trace.shortWrite = false;
    ASSERT_EQ(Send(Describe(id, {0, 1, 2}, true)), Status::OK());
    Verify(id);
}

TEST_F(UCCacheBlockDumpTest, FailedRenameRetainsUnpublishedDataForRetry)
{
    Start();
    auto id = Id(34);
    PosixStore::Config config;
    config.storageBackends = {Path()};
    config.dataDirShardBytes = 1;
    PosixStore::SpaceLayout layout;
    ASSERT_EQ(layout.Setup(config), Status::OK());
    const auto archived = layout.DataFilePath(id, false);
    WriteTrace trace;
    // Create the obstruction after I/O, so this exercises rename rather than open.
    trace.beforeCompletion = [archived] { std::filesystem::create_directory(archived); };
    const auto failed = Send(Describe(id, {0, 1, 2}, true));
    EXPECT_TRUE(failed.Failure());
    ASSERT_TRUE(std::filesystem::exists(layout.DataFilePath(id, true)))
        << failed.ToString() << "; writes=" << trace.writes.size();
    EXPECT_EQ(std::filesystem::file_size(layout.DataFilePath(id, true)), 3 * kBytes);
    ASSERT_TRUE(std::filesystem::remove(archived));
    trace.beforeCompletion = {};
    ASSERT_EQ(Send(Describe(id, {0, 1, 2}, true)), Status::OK());
    Verify(id);
}
}  // namespace

TEST_F(UCCacheBlockDumpTest, FailedLayerKeepsReadyLayersAndOnlyMissingLayerIsRetried)
{
    Start();
    WriteTrace trace;
    auto id = Id(40);
    std::array<size_t, 3> copies{};
    CacheStore::TestHooks::blockDumpCopyHook = [&](const auto& shard) {
        ++copies[shard.index];
        return shard.index == 1 && copies[1] == 1 ? Status::Error("controlled D2H failure") : Status::OK();
    };
    EXPECT_EQ(Send(Describe(id, {0}, false)), Status::OK());
    EXPECT_TRUE(Send(Describe(id, {1}, false)).Failure());
    auto ready = CacheStore::EarlyBufferPins::Acquire(*manager->GetTransBuffer(), id, 0, false);
    auto failed = CacheStore::EarlyBufferPins::Acquire(*manager->GetTransBuffer(), id, 1, true);
    EXPECT_TRUE(ready && ready.Ready());
    if (ready) { EXPECT_FALSE(ready.WholeView().Ready()); }
    EXPECT_FALSE(failed);  // a new reader must not pin an unfillable failed layer
    ready = {};
    EXPECT_EQ(Send(Describe(id, {2}, false)), Status::OK());
    EXPECT_EQ(Send(Describe(id, {0, 1, 2}, true)), Status::OK());
    CacheStore::TestHooks::blockDumpCopyHook = {};
    EXPECT_EQ(copies, (std::array<size_t, 3>{1, 2, 1}));
    EXPECT_EQ(trace.writes.size(), 1);
    Verify(id);
}

TEST_F(UCCacheBlockDumpTest, CopyFailureDoesNotSkipLaterIndependentLayers)
{
    Start();
    WriteTrace trace;
    std::array<size_t, 3> copies{};
    CacheStore::TestHooks::blockDumpCopyHook = [&](const auto& shard) {
        ++copies[shard.index];
        return shard.index == 1 ? Status::Error("controlled D2H failure") : Status::OK();
    };
    auto id = Id(41);
    EXPECT_TRUE(Send(Describe(id, {0, 1, 2}, true)).Failure());
    CacheStore::TestHooks::blockDumpCopyHook = {};
    EXPECT_EQ(copies, (std::array<size_t, 3>{1, 1, 1}));
    EXPECT_EQ(backend->LookupOnPrefix(&id, 1).Value(), -1);
    EXPECT_EQ(trace.writes.size(), 2);
    EXPECT_EQ(Send(Describe(id, {0, 1, 2}, true)), Status::OK());
    Verify(id);
}

TEST_F(UCCacheBlockDumpTest, SourceFenceAndAnotherCopyAdvanceBeforeSlowWriteCompletes)
{
    Start();
    WriteTrace trace;
    std::promise<void> entered, release;
    auto enteredFuture = entered.get_future();
    auto releaseFuture = release.get_future().share();
    std::atomic_bool once{false};
    trace.beforeCompletion = [&] {
        if (!once.exchange(true)) { entered.set_value(); releaseFuture.wait(); }
    };
    auto first = queue->Submit({TransTask::Type::DUMP, Describe(Id(42), {0, 1, 2}, true)});
    ASSERT_TRUE(first);
    EXPECT_EQ(queue->WaitSources(first.Value()), Status::OK());
    EXPECT_EQ(enteredFuture.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    EXPECT_FALSE(queue->Check(first.Value()).Value());
    EXPECT_FALSE(backend->AcquireWriteLease(Id(42)));
    auto second = queue->Submit({TransTask::Type::DUMP, Describe(Id(43), {0, 1, 2}, true)});
    EXPECT_TRUE(second);
    auto fence = std::async(std::launch::async, [&] { return queue->WaitSources(second.Value()); });
    auto advanced = fence.wait_for(std::chrono::milliseconds(300)) == std::future_status::ready;
    EXPECT_TRUE(advanced);
    auto original = source;
    if (advanced) { for (auto& layer : source) { layer.fill('z'); } }
    release.set_value();
    EXPECT_EQ(fence.get(), Status::OK());
    EXPECT_EQ(queue->Wait(first.Value()), Status::OK());
    EXPECT_EQ(queue->Wait(second.Value()), Status::OK());
    source = original;
    Verify(Id(42)); Verify(Id(43));
    EXPECT_TRUE(backend->AcquireWriteLease(Id(42)));
}

TEST_F(UCCacheBlockDumpTest, QueuedWaitTimeoutStillExecutesAndPublishes)
{
    Start();
    queue.reset();
    cache.timeoutMs = 80;
    queue = std::make_unique<CacheStore::BlockDumpQueue>();
    ASSERT_EQ(queue->Setup(cache, manager->GetTransBuffer()), Status::OK());
    std::promise<void> entered, release;
    auto enteredFuture = entered.get_future();
    auto releaseFuture = release.get_future().share();
    std::atomic<size_t> laterCopies{0};
    CacheStore::TestHooks::blockDumpCopyHook = [&](const auto& shard) {
        if (shard.owner == Id(44)) { entered.set_value(); releaseFuture.wait(); }
        else { ++laterCopies; }
        return Status::OK();
    };
    auto running = queue->Submit({TransTask::Type::DUMP, Describe(Id(44), {0}, false)});
    ASSERT_TRUE(running);
    EXPECT_EQ(enteredFuture.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    auto queued = queue->Submit({TransTask::Type::DUMP, Describe(Id(45), {0, 1, 2}, true)});
    ASSERT_TRUE(queued);
    auto wait = std::async(std::launch::async, [&] { return queue->Wait(queued.Value()); });
    EXPECT_EQ(wait.wait_for(std::chrono::milliseconds(400)), std::future_status::timeout);
    EXPECT_EQ(laterCopies.load(), 0);
    release.set_value();
    EXPECT_EQ(wait.get(), Status::Timeout());
    EXPECT_EQ(queue->WaitSources(running.Value()), Status::OK());
    (void)queue->Wait(running.Value());
    CacheStore::TestHooks::blockDumpCopyHook = {};
    EXPECT_EQ(laterCopies.load(), 3);
    Verify(Id(45));
}

TEST_F(UCCacheBlockDumpTest, QueuedSourceFenceTimeoutStillExecutesAndReportsTimeout)
{
    Start();
    queue.reset();
    cache.timeoutMs = 80;
    queue = std::make_unique<CacheStore::BlockDumpQueue>();
    ASSERT_EQ(queue->Setup(cache, manager->GetTransBuffer()), Status::OK());
    std::promise<void> entered, release;
    auto enteredFuture = entered.get_future();
    auto releaseFuture = release.get_future().share();
    std::atomic<size_t> laterCopies{0};
    CacheStore::TestHooks::blockDumpCopyHook = [&](const auto& shard) {
        if (shard.owner == Id(49)) { entered.set_value(); releaseFuture.wait(); }
        else { ++laterCopies; }
        return Status::OK();
    };
    auto running = queue->Submit({TransTask::Type::DUMP, Describe(Id(49), {0}, false)});
    ASSERT_TRUE(running);
    EXPECT_EQ(enteredFuture.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    auto queued = queue->Submit({TransTask::Type::DUMP, Describe(Id(50), {0, 1, 2}, true)});
    ASSERT_TRUE(queued);
    auto fence = std::async(std::launch::async, [&] { return queue->WaitSources(queued.Value()); });
    EXPECT_EQ(fence.wait_for(std::chrono::milliseconds(400)), std::future_status::timeout);
    EXPECT_EQ(laterCopies.load(), 0);
    EXPECT_FALSE(queue->Check(queued.Value()).Value());
    release.set_value();
    // The source fence proves only that HBM is safe, while final Wait must
    // preserve the expired deadline even though the write subsequently succeeds.
    EXPECT_EQ(fence.get(), Status::OK());
    EXPECT_EQ(queue->Wait(queued.Value()), Status::Timeout());
    EXPECT_EQ(queue->Wait(running.Value()), Status::OK());
    CacheStore::TestHooks::blockDumpCopyHook = {};
    EXPECT_EQ(laterCopies.load(), 3);
    Verify(Id(50));
}

TEST_F(UCCacheBlockDumpTest, SharedBufferFailedLayerCanRetryWithoutResettingReadyLayer)
{
    Start(3, 16, false, true);
    auto& buffer = *manager->GetTransBuffer();
    auto first = CacheStore::EarlyBufferPins::Acquire(buffer, Id(46), 0, false);
    ASSERT_TRUE(first && first.Owner());
    first.MarkReady();
    auto bad = CacheStore::EarlyBufferPins::Acquire(buffer, Id(46), 1, false);
    ASSERT_TRUE(bad && bad.Owner());
    auto follower = bad;
    bad.MarkLayerFailed(Status::Error("drained simulated failure"));
    EXPECT_TRUE(first.Ready());
    EXPECT_EQ(follower.GetState(), CacheStore::TransBuffer::State::FAILED);
    auto retry = CacheStore::EarlyBufferPins::Acquire(buffer, Id(46), 1, false);
    ASSERT_TRUE(retry && retry.Owner());
    retry.MarkReady();
    EXPECT_TRUE(first.Ready());
    EXPECT_FALSE(first.WholeView().Ready());
    auto last = CacheStore::EarlyBufferPins::Acquire(buffer, Id(46), 2, false);
    ASSERT_TRUE(last && last.Owner());
    last.MarkReady();
    EXPECT_TRUE(first.WholeView().Ready());
}

TEST_F(UCCacheBlockDumpTest, StartedWriteTimeoutRetainsLeaseUntilPhysicalCompletion)
{
    Start();
    queue.reset();
    cache.timeoutMs = 80;
    queue = std::make_unique<CacheStore::BlockDumpQueue>();
    ASSERT_EQ(queue->Setup(cache, manager->GetTransBuffer()), Status::OK());
    WriteTrace trace;
    std::promise<void> entered, release;
    auto enteredFuture = entered.get_future();
    auto releaseFuture = release.get_future().share();
    trace.beforeCompletion = [&] { entered.set_value(); releaseFuture.wait(); };
    auto id = Id(47);
    auto task = queue->Submit({TransTask::Type::DUMP, Describe(id, {0, 1, 2}, true)});
    ASSERT_TRUE(task);
    EXPECT_EQ(queue->WaitSources(task.Value()), Status::OK());
    EXPECT_EQ(enteredFuture.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    auto wait = std::async(std::launch::async, [&] { return queue->Wait(task.Value()); });
    EXPECT_EQ(wait.wait_for(std::chrono::milliseconds(200)), std::future_status::timeout);
    EXPECT_FALSE(backend->AcquireWriteLease(id));
    release.set_value();
    EXPECT_EQ(wait.get(), Status::Timeout());
    EXPECT_TRUE(backend->AcquireWriteLease(id));
    Verify(id);
}

TEST_F(UCCacheBlockDumpTest, LaterFinalizationCompletesAnInFlightPartialSpill)
{
    Start();
    WriteTrace trace;
    std::promise<void> entered, release;
    auto enteredFuture = entered.get_future();
    auto releaseFuture = release.get_future().share();
    std::atomic_bool once{false};
    trace.beforeCompletion = [&] {
        if (!once.exchange(true)) { entered.set_value(); releaseFuture.wait(); }
    };
    auto id = Id(48);
    auto partial = queue->Submit({TransTask::Type::DUMP, Describe(id, {0}, true)});
    ASSERT_TRUE(partial);
    EXPECT_EQ(queue->WaitSources(partial.Value()), Status::OK());
    EXPECT_EQ(enteredFuture.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    auto repaired = queue->Submit({TransTask::Type::DUMP, Describe(id, {0, 1, 2}, true)});
    EXPECT_TRUE(repaired);
    release.set_value();
    EXPECT_EQ(queue->Wait(partial.Value()), Status::NotFound());
    EXPECT_EQ(queue->Wait(repaired.Value()), Status::OK());
    Verify(id);
}

TEST_F(UCCacheBlockDumpTest, WaitPolicyCompletesInterleavedBlocksBeyondL1Capacity)
{
    Start(3, 1, false, false, "wait");
    WriteTrace trace;
    std::vector<Detail::TaskHandle> tasks;
    for (size_t layer = 0; layer < 3; ++layer) {
        Detail::TaskDesc desc;
        for (unsigned char block = 60; block < 68; ++block) {
            auto part = Describe(Id(block), {layer}, false);
            desc.insert(desc.end(), part.begin(), part.end());
        }
        auto task = queue->Submit({TransTask::Type::DUMP, std::move(desc)});
        ASSERT_TRUE(task);
        tasks.push_back(task.Value());
    }
    Detail::TaskDesc final;
    final.finalizeBlockDump = true;
    for (unsigned char block = 60; block < 68; ++block) {
        auto part = Describe(Id(block), {0, 1, 2}, true);
        final.insert(final.end(), part.begin(), part.end());
    }
    auto task = queue->Submit({TransTask::Type::DUMP, std::move(final)});
    ASSERT_TRUE(task);
    tasks.push_back(task.Value());
    for (auto id : tasks) { EXPECT_EQ(queue->Wait(id), Status::OK()); }
    ASSERT_EQ(trace.writes.size(), 8);
    for (const auto& write : trace.writes) {
        EXPECT_EQ(write, std::make_pair(size_t{0}, 3 * kBytes));
    }
    for (unsigned char block = 60; block < 68; ++block) { Verify(Id(block)); }
}

TEST_F(UCCacheBlockDumpTest, WaitPolicyDoesNotReuseSlotBeforePhysicalWriteCompletes)
{
    Start(3, 1, false, false, "wait");
    WriteTrace trace;
    std::promise<void> entered, release;
    auto enteredFuture = entered.get_future();
    auto releaseFuture = release.get_future().share();
    std::atomic_bool once{false};
    trace.beforeCompletion = [&] {
        if (!once.exchange(true)) { entered.set_value(); releaseFuture.wait(); }
    };
    ASSERT_EQ(Send(Describe(Id(62), {0}, false)), Status::OK());
    auto waiting = queue->Submit({TransTask::Type::DUMP, Describe(Id(63), {0}, false)});
    ASSERT_TRUE(waiting);
    auto fence = std::async(std::launch::async, [&] { return queue->WaitSources(waiting.Value()); });
    EXPECT_EQ(fence.wait_for(std::chrono::milliseconds(100)), std::future_status::timeout);
    auto owner = queue->Submit({TransTask::Type::DUMP, Describe(Id(62), {1, 2}, true)});
    ASSERT_TRUE(owner);
    EXPECT_EQ(queue->WaitSources(owner.Value()), Status::OK());
    EXPECT_EQ(enteredFuture.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    EXPECT_EQ(fence.wait_for(std::chrono::milliseconds(100)), std::future_status::timeout);
    EXPECT_FALSE(backend->AcquireWriteLease(Id(62)));
    release.set_value();
    EXPECT_EQ(fence.get(), Status::OK());
    EXPECT_EQ(queue->Wait(owner.Value()), Status::OK());
    EXPECT_EQ(queue->Wait(waiting.Value()), Status::OK());
    EXPECT_EQ(Send(Describe(Id(63), {0, 1, 2}, true)), Status::OK());
    ASSERT_EQ(trace.writes.size(), 2);
    Verify(Id(62)); Verify(Id(63));
}

TEST_F(UCCacheBlockDumpTest, WaitPolicyAllowsConfiguredNumberOfResidentBlocks)
{
    Start(3, 2, false, false, "wait");
    WriteTrace trace;
    EXPECT_EQ(Send(Describe(Id(64), {0}, false)), Status::OK());
    EXPECT_EQ(Send(Describe(Id(65), {0}, false)), Status::OK());
    auto first = CacheStore::EarlyBufferPins::Acquire(*manager->GetTransBuffer(), Id(64), 0, false);
    auto second = CacheStore::EarlyBufferPins::Acquire(*manager->GetTransBuffer(), Id(65), 0, false);
    EXPECT_TRUE(first && first.Ready());
    EXPECT_TRUE(second && second.Ready());
    EXPECT_TRUE(trace.writes.empty());
    first = {}; second = {};
    EXPECT_EQ(Send(Describe(Id(64), {0, 1, 2}, true)), Status::OK());
    EXPECT_EQ(Send(Describe(Id(65), {0, 1, 2}, true)), Status::OK());
    EXPECT_EQ(trace.writes.size(), 2);
    Verify(Id(64)); Verify(Id(65));
}

TEST_F(UCCacheBlockDumpTest, WaitPolicyFinalizationRepairsOwnerBeforeWaitingBlock)
{
    Start(3, 1, false, false, "wait");
    WriteTrace trace;
    std::array<size_t, 3> copies{};
    CacheStore::TestHooks::blockDumpCopyHook = [&](const auto& shard) {
        if (shard.owner != Id(66)) { return Status::OK(); }
        ++copies[shard.index];
        return shard.index == 1 && copies[1] == 1 ? Status::Error("controlled copy failure") : Status::OK();
    };
    EXPECT_EQ(Send(Describe(Id(66), {0}, false)), Status::OK());
    EXPECT_TRUE(Send(Describe(Id(66), {1}, false)).Failure());
    auto waiting = queue->Submit({TransTask::Type::DUMP, Describe(Id(67), {0}, false)});
    ASSERT_TRUE(waiting);
    auto final = Describe(Id(66), {0, 1, 2}, true);
    auto second = Describe(Id(67), {0, 1, 2}, true);
    final.insert(final.end(), second.begin(), second.end());
    auto task = queue->Submit({TransTask::Type::DUMP, std::move(final)});
    ASSERT_TRUE(task);
    EXPECT_EQ(queue->Wait(waiting.Value()), Status::OK());
    EXPECT_EQ(queue->Wait(task.Value()), Status::OK());
    CacheStore::TestHooks::blockDumpCopyHook = {};
    EXPECT_EQ(copies, (std::array<size_t, 3>{1, 2, 1}));
    EXPECT_EQ(trace.writes.size(), 2);
    Verify(Id(66)); Verify(Id(67));
}

TEST_F(UCCacheBlockDumpTest, WaitPolicyIncompleteFinalizationUsesOnlyTerminalSpill)
{
    Start(3, 1, false, false, "wait");
    WriteTrace trace;
    EXPECT_EQ(Send(Describe(Id(68), {0}, false)), Status::OK());
    auto waiting = queue->Submit({TransTask::Type::DUMP, Describe(Id(69), {0, 1, 2}, true)});
    ASSERT_TRUE(waiting);
    auto closing = queue->Submit({TransTask::Type::DUMP, Describe(Id(68), {0}, true)});
    ASSERT_TRUE(closing);
    EXPECT_EQ(queue->Wait(closing.Value()), Status::NotFound());
    EXPECT_EQ(queue->Wait(waiting.Value()), Status::OK());
    ASSERT_EQ(trace.writes.size(), 2);
    EXPECT_EQ(trace.writes[0], std::make_pair(size_t{0}, kBytes));
    EXPECT_EQ(trace.writes[1], std::make_pair(size_t{0}, 3 * kBytes));
    auto incomplete = Id(68);
    EXPECT_EQ(backend->LookupOnPrefix(&incomplete, 1).Value(), -1);
    Verify(Id(69));
}

TEST_F(UCCacheBlockDumpTest, WaitPolicyRetainsQueuedSourcesAcrossTimeoutAndExternalPressure)
{
    Start(3, 1, false, false, "wait");
    queue.reset();
    cache.timeoutMs = 80;
    queue = std::make_unique<CacheStore::BlockDumpQueue>();
    ASSERT_EQ(queue->Setup(cache, manager->GetTransBuffer()), Status::OK());
    std::vector<CacheStore::TransBuffer::Handle> occupied;
    for (unsigned char i = 70; i < 74; ++i) {
        occupied.push_back(CacheStore::EarlyBufferPins::Acquire(*manager->GetTransBuffer(), Id(i), 0, false));
        ASSERT_TRUE(occupied.back());
    }
    auto task = queue->Submit({TransTask::Type::DUMP, Describe(Id(74), {0, 1, 2}, true)});
    ASSERT_TRUE(task);
    auto fence = std::async(std::launch::async, [&] { return queue->WaitSources(task.Value()); });
    EXPECT_EQ(fence.wait_for(std::chrono::milliseconds(300)), std::future_status::timeout);
    EXPECT_FALSE(queue->Check(task.Value()).Value());
    occupied.clear();
    EXPECT_EQ(fence.get(), Status::OK());
    EXPECT_EQ(queue->Wait(task.Value()), Status::Timeout());
    Verify(Id(74));
}

TEST_F(UCCacheBlockDumpTest, WaitPolicyReservesFinalizationAdmission)
{
    Start(3, 1, false, false, "wait");
    queue.reset();
    cache.waitingQueueDepth = 2;
    queue = std::make_unique<CacheStore::BlockDumpQueue>();
    ASSERT_EQ(queue->Setup(cache, manager->GetTransBuffer()), Status::OK());
    EXPECT_EQ(Send(Describe(Id(75), {0}, false)), Status::OK());
    auto waiting = queue->Submit({TransTask::Type::DUMP, Describe(Id(76), {0}, false)});
    ASSERT_TRUE(waiting);
    auto rejected = queue->Submit({TransTask::Type::DUMP, Describe(Id(77), {0}, false)});
    ASSERT_TRUE(rejected);
    EXPECT_EQ(queue->Wait(rejected.Value()), Status::NoSpace());
    auto final = Describe(Id(75), {0, 1, 2}, true);
    auto second = Describe(Id(76), {0, 1, 2}, true);
    final.insert(final.end(), second.begin(), second.end());
    auto task = queue->Submit({TransTask::Type::DUMP, std::move(final)});
    ASSERT_TRUE(task);
    EXPECT_EQ(queue->Wait(waiting.Value()), Status::OK());
    EXPECT_EQ(queue->Wait(task.Value()), Status::OK());
    Verify(Id(75)); Verify(Id(76));
}

TEST_F(UCCacheBlockDumpTest, WaitPolicyShutdownDrainsAcceptedPartialBlocks)
{
    Start(3, 1, false, false, "wait");
    WriteTrace trace;
    EXPECT_EQ(Send(Describe(Id(78), {0}, false)), Status::OK());
    auto waiting = queue->Submit({TransTask::Type::DUMP, Describe(Id(79), {0}, false)});
    ASSERT_TRUE(waiting);
    queue.reset();
    EXPECT_EQ(trace.writes.size(), 2);
    auto incomplete = Id(79);
    EXPECT_EQ(backend->LookupOnPrefix(&incomplete, 1).Value(), -1);
}

TEST_F(UCCacheBlockDumpTest, WaitPolicyFailedWriterReportsErrorInsteadOfHoldingWaitersForever)
{
    Start(3, 1, false, false, "wait");
    WriteTrace trace;
    trace.fail = true;
    EXPECT_TRUE(Send(Describe(Id(80), {0, 1, 2}, true)).Failure());
    EXPECT_TRUE(Send(Describe(Id(81), {0, 1, 2}, true)).Failure());
    trace.fail = false;
    EXPECT_EQ(Send(Describe(Id(80), {0, 1, 2}, true)), Status::OK());
    EXPECT_EQ(Send(Describe(Id(81), {0, 1, 2}, true)), Status::OK());
    Verify(Id(80)); Verify(Id(81));
}

TEST_F(UCCacheBlockDumpTest, InvalidCapacityPolicyIsRejected)
{
    Start();
    queue.reset();
    cache.cacheDumpCapacityPolicy = "unknown";
    queue = std::make_unique<CacheStore::BlockDumpQueue>();
    EXPECT_TRUE(queue->Setup(cache, manager->GetTransBuffer()).Failure());
    Detail::Dictionary config;
    config.Set("store_backend", backend.get());
    config.Set("cache_dump_capacity_policy", std::string{"unknown"});
    std::unique_ptr<StoreV1> invalid{MakeEarlyCacheStore()};
    EXPECT_TRUE(invalid->Setup(config).Failure());
    config.Set("cache_dump_capacity_policy", std::string{"wait"});
    std::unique_ptr<StoreV1> disabled{MakeEarlyCacheStore()};
    EXPECT_TRUE(disabled->Setup(config).Failure());
}

TEST_F(UCCacheBlockDumpTest, WaitPolicyObservesReadyWithoutRequiringForeignPinRelease)
{
    Start(3, 1, false, false, "wait");
    WriteTrace trace;
    auto& buffer = *manager->GetTransBuffer();
    auto reader = CacheStore::EarlyBufferPins::Acquire(buffer, Id(82), 0, true);
    ASSERT_TRUE(reader && reader.Owner());
    auto full = reader.WholeView();
    auto task = queue->Submit({TransTask::Type::DUMP, Describe(Id(82), {0, 1, 2}, true)});
    ASSERT_TRUE(task);
    auto fence = std::async(std::launch::async, [&] { return queue->WaitSources(task.Value()); });
    EXPECT_EQ(fence.wait_for(std::chrono::milliseconds(100)), std::future_status::timeout);
    const auto epoch = CacheStore::EarlyBufferPins::ReleaseEpoch(buffer);
    for (size_t layer = 0; layer < 3; ++layer) {
        std::memcpy(static_cast<char*>(full.Data()) + layer*kBytes, source[layer].data(), kBytes);
    }
    full.MarkReady();
    EXPECT_EQ(CacheStore::EarlyBufferPins::ReleaseEpoch(buffer), epoch);
    const bool completed = fence.wait_for(std::chrono::milliseconds(500)) == std::future_status::ready;
    EXPECT_TRUE(completed);
    if (!completed) { full = {}; reader = {}; }  // let a failed assertion still drain
    EXPECT_EQ(fence.get(), Status::OK());
    EXPECT_EQ(queue->Wait(task.Value()), Status::OK());
    ASSERT_EQ(trace.writes.size(), 1);
    Verify(Id(82));
}

TEST_F(UCCacheBlockDumpTest, WaitPolicyReportsForeignLoadFailureWithoutWaitingForPinRelease)
{
    Start(3, 1, false, false, "wait");
    WriteTrace trace;
    auto& buffer = *manager->GetTransBuffer();
    auto reader = CacheStore::EarlyBufferPins::Acquire(buffer, Id(83), 0, true);
    ASSERT_TRUE(reader && reader.Owner());
    auto task = queue->Submit({TransTask::Type::DUMP, Describe(Id(83), {0, 1, 2}, true)});
    ASSERT_TRUE(task);
    auto fence = std::async(std::launch::async, [&] { return queue->WaitSources(task.Value()); });
    EXPECT_EQ(fence.wait_for(std::chrono::milliseconds(100)), std::future_status::timeout);
    reader.MarkFailed(Status::Error("controlled foreign read failure"));
    const bool completed = fence.wait_for(std::chrono::milliseconds(500)) == std::future_status::ready;
    EXPECT_TRUE(completed);
    if (!completed) { reader = {}; }
    EXPECT_EQ(fence.get(), Status::OK());
    EXPECT_TRUE(queue->Wait(task.Value()).Failure());
    EXPECT_TRUE(trace.writes.empty());
    auto missing = Id(83);
    EXPECT_EQ(backend->LookupOnPrefix(&missing, 1).Value(), -1);
}

TEST_F(UCCacheBlockDumpTest, ZeroBudgetSpillAllowsMoreThanSixteenResidentBlocks)
{
    CheckZeroBudgetResidency("spill");
}

TEST_F(UCCacheBlockDumpTest, ZeroBudgetWaitAllowsMoreThanSixteenResidentBlocks)
{
    CheckZeroBudgetResidency("wait");
}

TEST_F(UCCacheBlockDumpTest, ZeroBudgetSpillStillReclaimsAtPhysicalL1Capacity)
{
    ASSERT_NO_FATAL_FAILURE(Start(3, 0));
    WriteTrace trace;
    for (size_t layer = 0; layer < 3; ++layer) {
        for (unsigned char block = 120; block < 128; ++block) {
            ASSERT_EQ(Send(Describe(Id(block), {layer}, false)), Status::OK());
        }
    }
    for (unsigned char block = 120; block < 128; ++block) {
        ASSERT_EQ(Send(Describe(Id(block), {0, 1, 2}, true)), Status::OK());
        Verify(Id(block));
    }
    EXPECT_GT(trace.writes.size(), 8);
}

TEST_F(UCCacheBlockDumpTest, ZeroBudgetWaitRetainsSlotUntilPhysicalWriteCompletes)
{
    ASSERT_NO_FATAL_FAILURE(Start(3, 0, false, false, "wait", 1));
    WriteTrace trace;
    std::promise<void> entered, release;
    auto enteredFuture = entered.get_future();
    auto releaseFuture = release.get_future().share();
    std::atomic_bool once{false};
    trace.beforeCompletion = [&] {
        if (!once.exchange(true)) { entered.set_value(); releaseFuture.wait(); }
    };
    ASSERT_EQ(Send(Describe(Id(130), {0}, false)), Status::OK());
    auto waiting = queue->Submit({TransTask::Type::DUMP, Describe(Id(131), {0}, false)});
    ASSERT_TRUE(waiting);
    auto fence = std::async(std::launch::async, [&] {
        return queue->WaitSources(waiting.Value());
    });
    EXPECT_EQ(fence.wait_for(std::chrono::milliseconds(100)), std::future_status::timeout);
    auto owner = queue->Submit({TransTask::Type::DUMP, Describe(Id(130), {1, 2}, true)});
    EXPECT_TRUE(owner);
    if (owner) { EXPECT_EQ(queue->WaitSources(owner.Value()), Status::OK()); }
    EXPECT_EQ(enteredFuture.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    EXPECT_EQ(fence.wait_for(std::chrono::milliseconds(100)), std::future_status::timeout);
    EXPECT_FALSE(backend->AcquireWriteLease(Id(130)));
    release.set_value();
    EXPECT_EQ(fence.get(), Status::OK());
    if (owner) { EXPECT_EQ(queue->Wait(owner.Value()), Status::OK()); }
    EXPECT_EQ(queue->Wait(waiting.Value()), Status::OK());
    EXPECT_EQ(Send(Describe(Id(131), {0, 1, 2}, true)), Status::OK());
    ASSERT_EQ(trace.writes.size(), 2);
    Verify(Id(130));
    Verify(Id(131));
}

TEST_F(UCCacheBlockDumpTest, ZeroBudgetStillBoundsQueueAndReservesFinalizerAdmission)
{
    ASSERT_NO_FATAL_FAILURE(Start(3, 0, false, false, "wait", 1));
    queue.reset();
    cache.waitingQueueDepth = 2;
    queue = std::make_unique<CacheStore::BlockDumpQueue>();
    ASSERT_EQ(queue->Setup(cache, manager->GetTransBuffer()), Status::OK());
    ASSERT_EQ(Send(Describe(Id(132), {0}, false)), Status::OK());
    auto waiting = queue->Submit({TransTask::Type::DUMP, Describe(Id(133), {0}, false)});
    ASSERT_TRUE(waiting);
    auto rejected = queue->Submit({TransTask::Type::DUMP, Describe(Id(134), {0}, false)});
    ASSERT_TRUE(rejected);
    EXPECT_EQ(queue->Wait(rejected.Value()), Status::NoSpace());
    auto final = Describe(Id(132), {0, 1, 2}, true);
    auto second = Describe(Id(133), {0, 1, 2}, true);
    final.insert(final.end(), second.begin(), second.end());
    auto task = queue->Submit({TransTask::Type::DUMP, std::move(final)});
    ASSERT_TRUE(task);
    EXPECT_EQ(queue->Wait(waiting.Value()), Status::OK());
    EXPECT_EQ(queue->Wait(task.Value()), Status::OK());
    Verify(Id(132));
    Verify(Id(133));
}

TEST_F(UCCacheBlockDumpTest, ZeroBudgetWaitReportsFailedWriterAtPhysicalCapacity)
{
    ASSERT_NO_FATAL_FAILURE(Start(3, 0, false, false, "wait", 1));
    WriteTrace trace;
    trace.fail = true;
    ASSERT_TRUE(Send(Describe(Id(135), {0, 1, 2}, true)).Failure());
    trace.fail = false;
    auto waiting = queue->Submit({TransTask::Type::DUMP, Describe(Id(136), {0, 1, 2}, true)});
    ASSERT_TRUE(waiting);
    auto result = std::async(std::launch::async, [&] { return queue->Wait(waiting.Value()); });
    const bool reported =
        result.wait_for(std::chrono::milliseconds(500)) == std::future_status::ready;
    EXPECT_TRUE(reported);
    // Repair frees the slot even if a regression left the other task waiting.
    EXPECT_EQ(Send(Describe(Id(135), {0, 1, 2}, true)), Status::OK());
    EXPECT_TRUE(result.get().Failure());
    EXPECT_EQ(Send(Describe(Id(136), {0, 1, 2}, true)), Status::OK());
    Verify(Id(135));
    Verify(Id(136));
}
