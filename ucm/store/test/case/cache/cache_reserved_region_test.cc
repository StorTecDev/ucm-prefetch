#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <future>
#include <memory>
#include <ostream>
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

struct RegionCase {
    const char* name;
    bool exclusive;
    bool shared;
    bool fullBlock;
};

void PrintTo(const RegionCase& value, std::ostream* stream) { *stream << value.name; }

class Backend : public testing::NiceMock<UC::Test::Detail::MockStore> {
public:
    bool SupportsFullBlockRead() const override { return true; }
    size_t FullBlockReadBytes() const override { return kBytes; }
    size_t PackedLayerBytes(size_t) const override { return kBytes; }
    Expected<std::shared_ptr<ReadLease>> AcquireReadLease(const Detail::BlockId*, size_t) override
    {
        if (!grantLease) { return std::shared_ptr<ReadLease>{}; }
        auto owned = std::make_shared<ReadLease>();
        lease = owned;
        return owned;
    }

    std::weak_ptr<ReadLease> lease;
    bool grantLease{true};
};

class UCCacheReservedRegionTest : public testing::TestWithParam<RegionCase> {
protected:
    void SetUp() override
    {
        ON_CALL(backend_, LookupOnPrefix).WillByDefault(testing::Invoke(
            [](const Detail::BlockId*, size_t) { return ssize_t{0}; }));
        ON_CALL(backend_, Check).WillByDefault(testing::Invoke([](auto) { return true; }));
        ON_CALL(backend_, Load).WillByDefault(testing::Invoke([this](Detail::TaskDesc task) {
            EXPECT_EQ(task.size(), 1);
            EXPECT_EQ(task.fullBlockRead, config_.cacheFullBlockLoad);
            if (task.brief == "EarlyL2ToL1") { EXPECT_FALSE(backend_.lease.expired()); }
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
        config_.shareBufferEnable = GetParam().shared;
        config_.localRankSize = 1;
        config_.cacheFullBlockLoad = config_.cacheFullBlockL1 = GetParam().fullBlock;
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

    void LoadX() { LoadX(GetParam().exclusive); }

    void LoadX(bool exclusive)
    {
        // Holding both ordinary slots forces the real dispatcher to use the
        // exclusive slot; no internal metadata or allocation result is mocked.
        std::vector<TransBuffer::Handle> guards;
        for (unsigned char i = 1; i <= (exclusive ? 2 : 1); ++i) {
            auto pin = buffer_->TryGet(Block(i), 0);
            ASSERT_TRUE(pin);
            ASSERT_LT(Location(pin), 2);
            pin.MarkReady();
            guards.push_back(std::move(pin));
        }
        const auto priorLoads = loads_.load();
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
        EXPECT_EQ(loads_.load(), priorLoads + 1);
        EXPECT_EQ(waits_.load(), loads_.load());
    }

    void ReserveX()
    {
        const auto x = Block(kX);
        early_ = std::make_unique<ReservedPrefetch>(
            &manager_, &backend_, kBytes, kBytes, 1000, config_.cacheFullBlockLoad);
        auto reserved = early_->Reserve(&x, 1, true);
        ASSERT_TRUE(reserved);
        handle_ = reserved.Value().handle;
        batch_ = std::move(reserved.Value().batch);
    }

    void ExpectPreservedX()
    {
        auto x = EarlyBufferPins::Acquire(*buffer_, Block(kX), 0, true);
        ASSERT_TRUE(x);
        EXPECT_EQ(Location(x), 2);
        EXPECT_TRUE(x.Ready());
        const auto* data = static_cast<const unsigned char*>(x.Data());
        EXPECT_TRUE(std::all_of(data, data + kBytes, [](auto value) { return value == kX; }));
    }

    void ExpectCompletedReread()
    {
        Early::Progress progress;
        for (size_t i = 0; i < 8; ++i) {
            auto step = batch_->Step(batch_->windowBytes);
            ASSERT_TRUE(step);
            progress = step.Value();
            if (progress.done) { break; }
        }
        ASSERT_TRUE(progress.done);
        EXPECT_FALSE(progress.waitingCapacity);
        EXPECT_FALSE(progress.result.capacityTruncated);
        EXPECT_EQ(progress.result.totalHitBlocks, 1);
        EXPECT_EQ(progress.result.l2ReadBytes, kBytes);
        Detail::TaskDesc consumer{{Block(kX), 0, {nullptr}}};
        consumer.loadReservations = {handle_};
        auto pins = early_->LoadHandles(consumer);
        ASSERT_EQ(pins.size(), 1);
        ASSERT_TRUE(pins[0]);
        EXPECT_LT(Location(pins[0]), 2);
        EXPECT_TRUE(pins[0].Ready());
        const auto* data = static_cast<const unsigned char*>(pins[0].Data());
        EXPECT_TRUE(std::all_of(data, data + kBytes, [](auto value) { return value == kX; }));
    }

    void ExpectReleasedResources()
    {
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
        if (GetParam().exclusive) { ASSERT_EQ(Location(pin), 2); }
        else { ASSERT_LT(Location(pin), 2); }
    }
    {
        std::vector<TransBuffer::Handle> available;
        for (unsigned char i = 1; i <= (GetParam().exclusive ? 2 : 1); ++i) {
            auto pin = buffer_->TryGet(Block(i), 0);
            ASSERT_TRUE(pin);
            ASSERT_TRUE(pin.Owner());
            ASSERT_LT(Location(pin), 2);
            available.push_back(std::move(pin));
        }
        if (GetParam().exclusive) { ASSERT_NE(Location(available[0]), Location(available[1])); }
    }

    early_ = std::make_unique<ReservedPrefetch>(
        &manager_, &backend_, kBytes, kBytes, 1000, config_.cacheFullBlockLoad);
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
        EXPECT_LT(Location(pins[0]), 2);
        EXPECT_EQ(loads_.load(), GetParam().exclusive ? 2 : 1);
        EXPECT_EQ(progress.result.l2ReadBytes, GetParam().exclusive ? kBytes : 0);
    }
    ExpectReleasedResources();
}

INSTANTIATE_TEST_SUITE_P(Placement, UCCacheReservedRegionTest,
    testing::Values(RegionCase{"OrdinaryLocalLayers", false, false, false},
                    RegionCase{"ExclusiveLocalLayers", true, false, false},
                    RegionCase{"OrdinaryLocalBlock", false, false, true},
                    RegionCase{"ExclusiveLocalBlock", true, false, true},
                    RegionCase{"OrdinarySharedBlock", false, true, true},
                    RegionCase{"ExclusiveSharedBlock", true, true, true}),
    [](const auto& info) { return info.param.name; });

class UCCacheReservedRecoveryTest : public UCCacheReservedRegionTest {};

TEST_P(UCCacheReservedRecoveryTest, RetainsHeldSourceUntilConsumerReleasesIt)
{
    ASSERT_NO_FATAL_FAILURE(LoadX());
    std::unique_ptr<TransBuffer> peer;
    if (GetParam().shared) {
        peer = std::make_unique<TransBuffer>();
        ASSERT_EQ(peer->Setup(config_), Status::OK());
    }
    auto held = EarlyBufferPins::Acquire(peer ? *peer : *buffer_, Block(kX), 0, true);
    ASSERT_TRUE(held);
    ASSERT_NO_FATAL_FAILURE(ReserveX());
    ASSERT_TRUE(batch_->Activate());
    auto waiting = batch_->Step(batch_->windowBytes);
    ASSERT_TRUE(waiting);
    EXPECT_TRUE(waiting.Value().waitingCapacity);
    EXPECT_EQ(loads_.load(), 1);
    ExpectPreservedX();
    held = {};
    ExpectCompletedReread();
    ExpectReleasedResources();
}

TEST_P(UCCacheReservedRecoveryTest, AcquiresOrdinaryCapacityBeforeEvictingSource)
{
    ASSERT_NO_FATAL_FAILURE(LoadX());
    std::vector<TransBuffer::Handle> held;
    for (unsigned char i = 1; i <= 2; ++i) {
        auto pin = buffer_->TryGet(Block(i), 0);
        ASSERT_TRUE(pin);
        held.push_back(std::move(pin));
    }
    ASSERT_NO_FATAL_FAILURE(ReserveX());
    ASSERT_TRUE(batch_->Activate());
    auto waiting = batch_->Step(batch_->windowBytes);
    ASSERT_TRUE(waiting);
    EXPECT_TRUE(waiting.Value().waitingCapacity);
    EXPECT_EQ(loads_.load(), 1);
    ExpectPreservedX();
    held.clear();
    ExpectCompletedReread();
    ExpectReleasedResources();
}

TEST_P(UCCacheReservedRecoveryTest, MissingL2CopyLeavesSourceUntouched)
{
    ASSERT_NO_FATAL_FAILURE(LoadX());
    EXPECT_CALL(backend_, LookupOnPrefix).WillRepeatedly(testing::Invoke(
        [](const Detail::BlockId*, size_t) { return ssize_t{-1}; }));
    ASSERT_NO_FATAL_FAILURE(ReserveX());
    auto activated = batch_->Activate();
    ASSERT_TRUE(activated);
    EXPECT_TRUE(activated.Value().done);
    EXPECT_EQ(activated.Value().result.totalHitBlocks, 0);
    EXPECT_EQ(loads_.load(), 1);
    ExpectPreservedX();
    ExpectReleasedResources();
}

TEST_P(UCCacheReservedRecoveryTest, PositiveLookupWithoutLeaseCannotEvictSource)
{
    ASSERT_NO_FATAL_FAILURE(LoadX());
    backend_.grantLease = false;
    ASSERT_NO_FATAL_FAILURE(ReserveX());
    ASSERT_TRUE(batch_->Activate());
    auto waiting = batch_->Step(batch_->windowBytes);
    ASSERT_TRUE(waiting);
    EXPECT_TRUE(waiting.Value().waitingCapacity);
    EXPECT_EQ(loads_.load(), 1);
    ExpectPreservedX();
    ExpectReleasedResources();
}

TEST_P(UCCacheReservedRecoveryTest, IncompleteExclusiveSourceCannotBeEvicted)
{
    std::vector<TransBuffer::Handle> guards;
    for (unsigned char i = 1; i <= 2; ++i) {
        auto pin = buffer_->TryGet(Block(i), 0);
        ASSERT_TRUE(pin);
        guards.push_back(std::move(pin));
    }
    {
        auto x = EarlyBufferPins::Acquire(*buffer_, Block(kX), 0, true);
        ASSERT_TRUE(x);
        ASSERT_EQ(Location(x), 2);
        ASSERT_FALSE(x.Ready());
    }
    guards.clear();
    ASSERT_NO_FATAL_FAILURE(ReserveX());
    ASSERT_TRUE(batch_->Activate());
    auto waiting = batch_->Step(batch_->windowBytes);
    ASSERT_TRUE(waiting);
    EXPECT_TRUE(waiting.Value().waitingCapacity);
    EXPECT_EQ(loads_.load(), 0);
    {
        auto x = EarlyBufferPins::Acquire(*buffer_, Block(kX), 0, true);
        ASSERT_TRUE(x);
        EXPECT_EQ(Location(x), 2);
        EXPECT_FALSE(x.Ready());
        x.MarkReady();
    }
    ExpectCompletedReread();
    ExpectReleasedResources();
}

TEST_P(UCCacheReservedRecoveryTest, ActivationRetainsNewL2ProofAfterL1PlacementChanges)
{
    ASSERT_NO_FATAL_FAILURE(LoadX(false));
    EXPECT_CALL(backend_, LookupOnPrefix).Times(0);
    ASSERT_NO_FATAL_FAILURE(ReserveX());
    ASSERT_EQ(batch_->ProtectedBytes(), 0);
    testing::Mock::VerifyAndClearExpectations(&backend_);
    // Deferred lookup holds no L1 pins: a normal load can relocate X before
    // activation. Its new L2 proof must survive into data-stage recovery.
    ASSERT_NO_FATAL_FAILURE(LoadX(true));
    ExpectPreservedX();
    EXPECT_CALL(backend_, LookupOnPrefix).Times(1).WillOnce(testing::Invoke(
        [](const Detail::BlockId*, size_t) { return ssize_t{0}; }));
    ASSERT_TRUE(batch_->Activate());
    ExpectCompletedReread();
    EXPECT_EQ(loads_.load(), 3);
    ExpectReleasedResources();
}

TEST_P(UCCacheReservedRecoveryTest, DiscardDrainsRereadBeforeReleasingDestinationAndLease)
{
    ASSERT_NO_FATAL_FAILURE(LoadX());
    ASSERT_NO_FATAL_FAILURE(ReserveX());
    ASSERT_TRUE(batch_->Activate());
    bool complete = false;
    ON_CALL(backend_, Check).WillByDefault(testing::Invoke([&](auto) { return complete; }));
    auto submitted = batch_->Step(batch_->windowBytes);
    ASSERT_TRUE(submitted);
    ASSERT_EQ(submitted.Value().pendingReadTasks, 1);
    {
        auto follower = buffer_->TryGet(Block(kX), 0);
        ASSERT_TRUE(follower);
        EXPECT_LT(Location(follower), 2);
        EXPECT_FALSE(follower.Ready());
        EXPECT_FALSE(follower.Owner());
    }
    batch_->Discard();
    auto draining = batch_->Step(0);
    ASSERT_TRUE(draining);
    EXPECT_FALSE(draining.Value().done);
    EXPECT_FALSE(backend_.lease.expired());
    EXPECT_EQ(batch_->ProtectedBytes(), kBytes);
    complete = true;
    auto stopped = batch_->Step(0);
    ASSERT_TRUE(stopped);
    EXPECT_TRUE(stopped.Value().done);
    EXPECT_TRUE(stopped.Value().discarded);
    ExpectReleasedResources();
}

TEST_P(UCCacheReservedRecoveryTest, FailedRereadDoesNotPublishReadyAndCanRetry)
{
    ASSERT_NO_FATAL_FAILURE(LoadX());
    ASSERT_NO_FATAL_FAILURE(ReserveX());
    ASSERT_TRUE(batch_->Activate());
    ASSERT_TRUE(batch_->Step(batch_->windowBytes));
    EXPECT_CALL(backend_, Wait).WillOnce(testing::Invoke([this](auto) {
        ++waits_;
        EXPECT_FALSE(backend_.lease.expired());
        return Status::Error("injected reread failure");
    }));
    auto failed = batch_->Step(0);
    ASSERT_FALSE(failed);
    EXPECT_TRUE(batch_->Snapshot().done);
    {
        auto follower = buffer_->TryGet(Block(kX), 0);
        ASSERT_TRUE(follower);
        EXPECT_FALSE(follower.Ready());
        EXPECT_EQ(follower.GetState(), TransBuffer::State::FAILED);
    }
    early_->Release(handle_);
    batch_.reset();
    early_.reset();
    EXPECT_TRUE(backend_.lease.expired());
    testing::Mock::VerifyAndClearExpectations(&backend_);
    ASSERT_NO_FATAL_FAILURE(ReserveX());
    ASSERT_TRUE(batch_->Activate());
    ExpectCompletedReread();
    ExpectReleasedResources();
}

TEST_P(UCCacheReservedRecoveryTest, ConcurrentEarlyAndOrdinaryReadersShareOneReread)
{
    ASSERT_NO_FATAL_FAILURE(LoadX());
    ASSERT_NO_FATAL_FAILURE(ReserveX());
    ASSERT_TRUE(batch_->Activate());
    const auto x = Block(kX);
    auto other = early_->Reserve(&x, 1, true);
    ASSERT_TRUE(other);
    ASSERT_TRUE(other.Value().batch->Activate());
    auto first = std::async(std::launch::async, [&] { return batch_->Step(batch_->windowBytes); });
    auto second = std::async(std::launch::async, [&] {
        return other.Value().batch->Step(other.Value().batch->windowBytes);
    });
    EXPECT_TRUE(first.get());
    EXPECT_TRUE(second.get());
    {
        auto ordinary = EarlyBufferPins::Acquire(*buffer_, x, 0, true);
        ASSERT_TRUE(ordinary);
        EXPECT_LT(Location(ordinary), 2);
        EXPECT_FALSE(ordinary.Owner());
    }
    for (size_t i = 0; i < 4; ++i) {
        ASSERT_TRUE(batch_->Step(batch_->windowBytes));
        ASSERT_TRUE(other.Value().batch->Step(other.Value().batch->windowBytes));
    }
    EXPECT_TRUE(batch_->Snapshot().done);
    EXPECT_TRUE(other.Value().batch->Snapshot().done);
    EXPECT_EQ(loads_.load(), 2);
    early_->Release(other.Value().handle);
    other.Value().batch.reset();
    ExpectReleasedResources();
}

INSTANTIATE_TEST_SUITE_P(Layout, UCCacheReservedRecoveryTest,
    testing::Values(RegionCase{"LocalLayers", true, false, false},
                    RegionCase{"LocalBlock", true, false, true},
                    RegionCase{"SharedBlock", true, true, true}),
    [](const auto& info) { return info.param.name; });
}  // namespace
