#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>
#include "cache/cc/load_queue.h"
#include "cache/cc/reserved_prefetch.h"
#include "detail/mock_store.h"
#include "detail/random.h"

namespace {
using namespace UC;
using namespace UC::CacheStore;
using namespace std::chrono_literals;
constexpr size_t kBytes = 8192;
constexpr unsigned char kX = 42;

Detail::BlockId Block(unsigned char value)
{
    Detail::BlockId block{};
    block[0] = std::byte{value};
    return block;
}

size_t Location(const TransBuffer::Handle& pin)
{
    return EarlyBufferPins::Locations({pin}).at(0);
}

class Backend : public testing::NiceMock<UC::Test::Detail::MockStore> {
public:
    bool SupportsFullBlockRead() const override { return true; }
    size_t FullBlockReadBytes() const override { return kBytes; }
    size_t PackedLayerBytes(size_t) const override { return kBytes; }
    Expected<std::shared_ptr<ReadLease>> AcquireReadLease(const Detail::BlockId*, size_t) override
    {
        auto owned = std::make_shared<ReadLease>();
        lease = owned;
        return owned;
    }

    std::weak_ptr<ReadLease> lease;
};

class UCCacheReservedRegionTest : public testing::TestWithParam<bool> {
protected:
    void SetUp() override
    {
        ON_CALL(backend_, LookupOnPrefix).WillByDefault(testing::Invoke(
            [](const Detail::BlockId*, size_t) { return ssize_t{0}; }));
        ON_CALL(backend_, Check).WillByDefault(testing::Invoke([](auto) { return true; }));
        ON_CALL(backend_, Load).WillByDefault(testing::Invoke([this](Detail::TaskDesc task) {
            EXPECT_EQ(task.size(), 1);
            EXPECT_TRUE(task.fullBlockRead);
            for (auto& shard : task) {
                EXPECT_EQ(shard.owner, Block(kX));
                for (auto* destination : shard.addrs) {
                    if (destination) { std::memset(destination, kX, kBytes); }
                }
            }
            return Detail::TaskHandle{++loads_};
        }));
        ON_CALL(backend_, Wait).WillByDefault(testing::Invoke([this](auto) {
            ++waits_;
            return Status::OK();
        }));
        config_.uniqueId = UC::Test::Detail::Random().RandomString(16);
        config_.storeBackend = &backend_;
        config_.deviceId = 0;
        config_.tensorSizes = {kBytes};
        config_.shardSize = config_.blockSize = kBytes;
        config_.bufferCapacity = 3 * kBytes;
        config_.loadExclusiveBufferNumber = 1;
        config_.shareBufferEnable = false;
        config_.localRankSize = 1;
        config_.cacheFullBlockLoad = config_.cacheFullBlockL1 = true;
        config_.streamNumber = 1;
        config_.waitingQueueDepth = config_.runningQueueDepth = 8;
        config_.timeoutMs = 1000;
        ASSERT_EQ(manager_.Setup(config_), Status::OK());
        buffer_ = manager_.GetTransBuffer();
        ASSERT_EQ(EarlyBufferPins::ReservableSlots(*buffer_), 2);
    }

    void TearDown() override { ReleaseReservation(); }

    void ReleaseReservation()
    {
        if (batch_) {
            batch_->Discard();
            auto stopped = batch_->Step(0);
            EXPECT_TRUE(stopped);
            if (stopped) { EXPECT_TRUE(stopped.Value().done); }
        }
        if (early_) { early_->Release(handle_); }
        batch_.reset();
        early_.reset();
    }

    void LoadX()
    {
        // Holding both ordinary slots forces the real dispatcher to use the
        // exclusive slot; no internal metadata or allocation result is mocked.
        std::vector<TransBuffer::Handle> guards;
        for (unsigned char i = 1; i <= (GetParam() ? 2 : 1); ++i) {
            auto pin = buffer_->TryGet(Block(i), 0);
            ASSERT_TRUE(pin);
            ASSERT_LT(Location(pin), 2);
            pin.MarkReady();
            guards.push_back(std::move(pin));
        }
        HashSet<Detail::TaskHandle> failures;
        std::array<unsigned char, kBytes> destination{};
        {
            LoadQueue queue;
            ASSERT_EQ(queue.Setup(config_, &failures, buffer_), Status::OK());
            auto task = std::make_shared<TransTask>(
                TransTask::Type::LOAD, Detail::TaskDesc{{Block(kX), 0, {destination.data()}}});
            auto waiter = std::make_shared<Latch>();
            queue.Submit(task, waiter);
            ASSERT_TRUE(waiter->WaitForDuration(5000));
            EXPECT_FALSE(failures.Contains(task->id));
            EXPECT_EQ(task->failureStatus.load(), Status::OK().Underlying());
        }
        // Joining the native workers drops the last H2D holder before the
        // test checks that X has no references from ordinary loading.
        EXPECT_TRUE(std::all_of(destination.begin(), destination.end(),
                                [](auto value) { return value == kX; }));
        EXPECT_EQ(loads_.load(), 1);
        EXPECT_EQ(waits_.load(), 1);
    }

    Backend backend_;
    Config config_;
    BufferManager manager_;
    TransBuffer* buffer_{nullptr};
    std::atomic<size_t> loads_{0}, waits_{0};
    std::unique_ptr<ReservedPrefetch> early_;
    std::shared_ptr<Early::Batch> batch_;
    uint64_t handle_{0};
};

TEST_P(UCCacheReservedRegionTest, EarlyCompletesForUnreferencedReadyBlock)
{
    ASSERT_NO_FATAL_FAILURE(LoadX());
    const auto x = Block(kX);
    {
        auto pin = EarlyBufferPins::Acquire(*buffer_, x, 0, true);
        ASSERT_TRUE(pin);
        ASSERT_TRUE(pin.Ready());
        ASSERT_TRUE(pin.Owner());
        if (GetParam()) { ASSERT_EQ(Location(pin), 2); }
        else { ASSERT_LT(Location(pin), 2); }
    }
    {
        std::vector<TransBuffer::Handle> available;
        for (unsigned char i = 1; i <= (GetParam() ? 2 : 1); ++i) {
            auto pin = buffer_->TryGet(Block(i), 0);
            ASSERT_TRUE(pin);
            ASSERT_TRUE(pin.Owner());
            ASSERT_LT(Location(pin), 2);
            available.push_back(std::move(pin));
        }
        if (GetParam()) { ASSERT_NE(Location(available[0]), Location(available[1])); }
    }

    early_ = std::make_unique<ReservedPrefetch>(&manager_, &backend_, kBytes, kBytes, 1000, true);
    {
        auto reserved = early_->Reserve(&x, 1, true);
        ASSERT_TRUE(reserved);
        handle_ = reserved.Value().handle;
        batch_ = std::move(reserved.Value().batch);
        ASSERT_EQ(reserved.Value().totalHitBlocks, 1);
    }
    ASSERT_TRUE(batch_->Activate());
    Early::Progress progress;
    const auto deadline = std::chrono::steady_clock::now() + 1s;
    size_t capacityRetries = 0;
    do {
        auto step = batch_->Step(batch_->windowBytes);
        ASSERT_TRUE(step);
        progress = step.Value();
        if (progress.done) { break; }
        if (progress.waitingCapacity) {
            const auto epoch = EarlyBufferPins::ReleaseEpoch(*buffer_);
            {
                auto ordinary = buffer_->TryGet(Block(90), 0);
                ASSERT_TRUE(ordinary);
                ASSERT_LT(Location(ordinary), 2);
                ordinary.MarkReady();
            }
            // A genuine release makes each retry eligible, excluding a lost
            // notification as the explanation for a persistent region conflict.
            ASSERT_NE(EarlyBufferPins::ReleaseEpoch(*buffer_), epoch);
            ++capacityRetries;
        }
        std::this_thread::sleep_for(1ms);
    } while (std::chrono::steady_clock::now() < deadline);

    EXPECT_TRUE(progress.done)
        << "waitingCapacity=" << progress.waitingCapacity
        << " pendingReadTasks=" << progress.pendingReadTasks
        << " inflightBytes=" << progress.inflightBytes
        << " capacityRetries=" << capacityRetries << " backendLoads=" << loads_.load();
    if (progress.done) {
        EXPECT_EQ(progress.result.totalHitBlocks, 1);
        EXPECT_FALSE(progress.result.capacityTruncated);
        Detail::TaskDesc consumer{{x, 0, {nullptr}}};
        consumer.loadReservations = {handle_};
        auto pins = early_->LoadHandles(consumer);
        ASSERT_EQ(pins.size(), 1);
        ASSERT_TRUE(pins[0]);
        ASSERT_TRUE(pins[0].Ready());
        const auto* data = static_cast<const unsigned char*>(pins[0].Data());
        EXPECT_TRUE(std::all_of(data, data + kBytes, [](auto value) { return value == kX; }));
        if (!GetParam()) { EXPECT_EQ(loads_.load(), 1); }
    }
    ReleaseReservation();
    EXPECT_TRUE(backend_.lease.expired());
    EXPECT_EQ(loads_.load(), waits_.load());
    std::vector<TransBuffer::Handle> reused;
    for (unsigned char i = 120; i < 123; ++i) {
        auto pin = EarlyBufferPins::Acquire(*buffer_, Block(i), 0, true);
        ASSERT_TRUE(pin);
        pin.MarkReady();
        reused.push_back(std::move(pin));
    }
}

INSTANTIATE_TEST_SUITE_P(Placement, UCCacheReservedRegionTest, testing::Bool(),
                        [](const auto& info) { return info.param ? "Exclusive" : "Ordinary"; });
}  // namespace
