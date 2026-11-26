#include <array>
#include <cstring>
#include <thread>
#include <gtest/gtest.h>
#include "cache/cc/early_buffer_pins.h"
#include "detail/mock_store.h"
#include "detail/random.h"

namespace {
using namespace UC;
using CacheStore::EarlyBufferPins;
using CacheStore::TransBuffer;

Detail::BlockId Block(unsigned value)
{
    Detail::BlockId id{};
    id[0] = std::byte(value);
    return id;
}

class UCEarlyConsumerLoadHintsTest : public testing::Test {
protected:
    testing::NiceMock<UC::Test::Detail::MockStore> backend;
    CacheStore::Config config;
    TransBuffer buffer;
    std::unique_ptr<EarlyBufferPins::LoadHints> hints;

    void SetUp() override
    {
        config.uniqueId = UC::Test::Detail::Random().RandomString(16);
        config.deviceId = 0;
        config.tensorSizes = {4096};
        config.shardSize = 4096;
        config.blockSize = config.bufferCapacity = 8192;
        config.loadExclusiveBufferNumber = 0;
        config.shareBufferEnable = false;
        config.cacheFullBlockL1 = true;
        config.cacheSdmaDirect = false;
        config.storeBackend = &backend;
        ASSERT_EQ(buffer.Setup(config), Status::OK());
        hints = EarlyBufferPins::MakeLoadHints(buffer);
        ASSERT_TRUE(hints);
    }

    TransBuffer::Handle Ready()
    {
        auto pin = EarlyBufferPins::Acquire(buffer, Block(1), 0, true);
        if (pin) {
            std::memset(pin.Data(), 0x41, config.shardSize);
            std::memset(static_cast<char*>(pin.Data()) + config.shardSize, 0x42, config.shardSize);
            pin.WholeView().MarkReady();
        }
        return pin;
    }
};
}  // namespace

TEST_F(UCEarlyConsumerLoadHintsTest, ReadyLayerPinsDataUntilLastConsumerReleases)
{
    auto owner = Ready();
    ASSERT_TRUE(owner);
    auto first = EarlyBufferPins::AcquireLoad(buffer, Block(1), 0, hints.get());
    auto second = EarlyBufferPins::AcquireLoad(buffer, Block(1), 1, hints.get());
    ASSERT_TRUE(first);
    ASSERT_TRUE(second);
    EXPECT_FALSE(first.Owner());
    EXPECT_FALSE(second.Owner());
    EXPECT_TRUE(first.Ready());
    EXPECT_TRUE(second.Ready());
    EXPECT_EQ(static_cast<unsigned char*>(first.Data())[0], 0x41);
    EXPECT_EQ(static_cast<unsigned char*>(second.Data())[0], 0x42);
    owner = {};
    first = {};
    EXPECT_FALSE(EarlyBufferPins::AcquireLoad(buffer, Block(2), 0, hints.get()));
    second = {};
    auto replacement = EarlyBufferPins::AcquireLoad(buffer, Block(2), 0, hints.get());
    ASSERT_TRUE(replacement);
    EXPECT_TRUE(replacement.Owner());
    EXPECT_FALSE(replacement.Ready());
}

TEST_F(UCEarlyConsumerLoadHintsTest, ColdOwnerAndFollowerKeepWholeReadProtocol)
{
    auto owner = EarlyBufferPins::AcquireLoad(buffer, Block(1), 0, hints.get());
    auto follower = EarlyBufferPins::AcquireLoad(buffer, Block(1), 1, hints.get());
    ASSERT_TRUE(owner);
    ASSERT_TRUE(follower);
    EXPECT_TRUE(owner.Owner());
    EXPECT_FALSE(follower.Owner());
    EXPECT_FALSE(owner.Ready());
    EXPECT_FALSE(follower.Ready());
    owner.WholeView().MarkReady();
    EXPECT_TRUE(follower.Ready());
    EXPECT_FALSE(EarlyBufferPins::AcquireLoad(buffer, Block(1), 2, hints.get()));
}

TEST_F(UCEarlyConsumerLoadHintsTest, PartialDumpRejectsUnwrittenLayer)
{
    auto writer = EarlyBufferPins::Acquire(buffer, Block(1), 0, false);
    ASSERT_TRUE(writer);
    writer.MarkReady();
    auto completed = EarlyBufferPins::AcquireLoad(buffer, Block(1), 0, hints.get());
    ASSERT_TRUE(completed);
    EXPECT_TRUE(completed.Ready());
    EXPECT_FALSE(completed.Owner());
    EXPECT_FALSE(EarlyBufferPins::AcquireLoad(buffer, Block(1), 1, hints.get()));
    auto lastWriter = EarlyBufferPins::Acquire(buffer, Block(1), 1, false);
    ASSERT_TRUE(lastWriter);
    EXPECT_TRUE(lastWriter.Owner());
    lastWriter.MarkReady();
    auto lastReader = EarlyBufferPins::AcquireLoad(buffer, Block(1), 1, hints.get());
    ASSERT_TRUE(lastReader);
    EXPECT_TRUE(lastReader.Ready());
    EXPECT_FALSE(lastReader.Owner());
}

TEST_F(UCEarlyConsumerLoadHintsTest, FailureStaysVisibleUntilReferencesDrain)
{
    auto owner = Ready();
    ASSERT_TRUE(owner);
    auto reader = EarlyBufferPins::AcquireLoad(buffer, Block(1), 1, hints.get());
    ASSERT_TRUE(reader);
    owner.MarkFailed(Status::Error("injected failure"));
    EXPECT_EQ(reader.GetState(), TransBuffer::State::FAILED);
    auto failed = EarlyBufferPins::AcquireLoad(buffer, Block(1), 1, hints.get());
    ASSERT_TRUE(failed);
    EXPECT_FALSE(failed.Owner());
    EXPECT_EQ(failed.GetState(), TransBuffer::State::FAILED);
    reader = {};
    failed = {};
    owner = {};
    auto retry = EarlyBufferPins::AcquireLoad(buffer, Block(1), 1, hints.get());
    ASSERT_TRUE(retry);
    EXPECT_TRUE(retry.Owner());
    EXPECT_EQ(retry.GetState(), TransBuffer::State::LOADING);
}

TEST_F(UCEarlyConsumerLoadHintsTest, BypassLoadStillReloadsAnIdleReadyBlock)
{
    TransBuffer bypass;
    config.cacheLoadBackendOnly = true;
    ASSERT_EQ(bypass.Setup(config), Status::OK());
    auto owner = EarlyBufferPins::Acquire(bypass, Block(1), 0, true);
    ASSERT_TRUE(owner);
    owner.WholeView().MarkReady();
    auto bypassHints = EarlyBufferPins::MakeLoadHints(bypass);
    EXPECT_FALSE(bypassHints);
    auto active = EarlyBufferPins::AcquireLoad(bypass, Block(1), 1, bypassHints.get());
    ASSERT_TRUE(active);
    EXPECT_TRUE(active.Ready());
    EXPECT_FALSE(active.Owner());
    active = {};
    owner = {};
    auto reload = EarlyBufferPins::AcquireLoad(bypass, Block(1), 1, bypassHints.get());
    ASSERT_TRUE(reload);
    EXPECT_TRUE(reload.Owner());
    EXPECT_FALSE(reload.Ready());
}

TEST_F(UCEarlyConsumerLoadHintsTest, ConcurrentReadersPreserveReferencesAndLayerData)
{
    auto owner = Ready();
    ASSERT_TRUE(owner);
    std::array<std::thread, 4> threads;
    for (size_t i = 0; i < threads.size(); ++i) {
        threads[i] = std::thread([&, i] {
            for (size_t n = 0; n < 1000; ++n) {
                auto pin = EarlyBufferPins::AcquireLoad(buffer, Block(1), i % 2, hints.get());
                ASSERT_TRUE(pin);
                EXPECT_TRUE(pin.Ready());
                EXPECT_FALSE(pin.Owner());
                EXPECT_EQ(static_cast<unsigned char*>(pin.Data())[0], 0x41 + i % 2);
            }
        });
    }
    for (auto& thread : threads) { thread.join(); }
    owner = {};
    auto replacement = EarlyBufferPins::AcquireLoad(buffer, Block(2), 0, hints.get());
    ASSERT_TRUE(replacement);
    EXPECT_TRUE(replacement.Owner());
}

TEST_F(UCEarlyConsumerLoadHintsTest, RecycledPositionCannotAliasAnotherBlock)
{
    auto owner = Ready();
    ASSERT_TRUE(owner);
    auto first = EarlyBufferPins::AcquireLoad(buffer, Block(1), 0, hints.get());
    ASSERT_TRUE(first);
    first = {};
    owner = {};
    auto replacement = EarlyBufferPins::Acquire(buffer, Block(2), 0, true);
    ASSERT_TRUE(replacement);
    replacement.WholeView().MarkReady();
    EXPECT_FALSE(EarlyBufferPins::AcquireLoad(buffer, Block(1), 1, hints.get()));
    replacement = {};
    auto restored = EarlyBufferPins::AcquireLoad(buffer, Block(1), 1, hints.get());
    ASSERT_TRUE(restored);
    EXPECT_TRUE(restored.Owner());
    EXPECT_FALSE(restored.Ready());
}

TEST_F(UCEarlyConsumerLoadHintsTest, HintsDoNotRetainReferencesOrAlterIdleOwnership)
{
    auto owner = Ready();
    ASSERT_TRUE(owner);
    auto first = EarlyBufferPins::AcquireLoad(buffer, Block(1), 0, hints.get());
    ASSERT_TRUE(first);
    first = {};
    owner = {};
    auto reader = EarlyBufferPins::AcquireLoad(buffer, Block(1), 1, hints.get());
    ASSERT_TRUE(reader);
    EXPECT_TRUE(reader.Owner());
    EXPECT_TRUE(reader.Ready());
}

TEST_F(UCEarlyConsumerLoadHintsTest, HintsAreBoundToTheirBuffer)
{
    TransBuffer other;
    config.uniqueId = UC::Test::Detail::Random().RandomString(16);
    ASSERT_EQ(other.Setup(config), Status::OK());
    auto owner = Ready();
    ASSERT_TRUE(owner);
    auto first = EarlyBufferPins::AcquireLoad(buffer, Block(1), 0, hints.get());
    ASSERT_TRUE(first);
    auto cold = EarlyBufferPins::AcquireLoad(other, Block(1), 1, hints.get());
    ASSERT_TRUE(cold);
    EXPECT_TRUE(cold.Owner());
    EXPECT_FALSE(cold.Ready());
}
