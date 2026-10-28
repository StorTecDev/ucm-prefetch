#include <sys/syscall.h>
#include <unistd.h>
#include <array>
#include <atomic>
#include <cstring>
#include <filesystem>
#include <mutex>
#include "cache/cc/block_dump_queue.h"
#include "cache/cc/buffer_manager.h"
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

    void Start(size_t layers = 3, size_t budget = 16, bool packed = false, bool shared = false)
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
        cache.bufferCapacity = cache.blockSize * 4;
        cache.loadExclusiveBufferNumber = 0;
        cache.shareBufferEnable = shared;
        cache.localRankSize = 1;
        cache.ioDirect = false;
        cache.streamNumber = 1;
        cache.cacheFullBlockL1 = cache.cacheFullBlockLoad = cache.cacheFullBlockDump = true;
        cache.cacheDumpReservedBlocks = budget;
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

TEST_F(UCCacheBlockDumpTest, QueuedTimeoutCancelsBeforeAnySourceAccess)
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
    auto queued = queue->Submit({TransTask::Type::DUMP, Describe(Id(45), {0}, false)});
    ASSERT_TRUE(queued);
    auto wait = std::async(std::launch::async, [&] { return queue->Wait(queued.Value()); });
    EXPECT_EQ(wait.wait_for(std::chrono::milliseconds(400)), std::future_status::ready);
    release.set_value();
    EXPECT_EQ(wait.get(), Status::Timeout());
    EXPECT_EQ(queue->WaitSources(running.Value()), Status::OK());
    (void)queue->Wait(running.Value());
    CacheStore::TestHooks::blockDumpCopyHook = {};
    EXPECT_EQ(laterCopies.load(), 0);
    auto cancelled = Id(45);
    EXPECT_EQ(backend->LookupOnPrefix(&cancelled, 1).Value(), -1);
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
