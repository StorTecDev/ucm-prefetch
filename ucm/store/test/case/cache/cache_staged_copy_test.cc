/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * */
#include <gtest/gtest.h>

#if defined(UCM_HOST_REGISTRATION_SIMU_TEST)
#include <array>
#include <cerrno>
#include <sys/wait.h>
#include <unistd.h>
#include <cstring>
#include <cstdlib>
#include <deque>
#include <limits>
#include "cache/cc/copy_stream.h"
#include "cache/cc/global_config.h"
#include "cache/cc/staged_copy.h"
#include "cache/cc/trans_buffer.h"
#include "detail/random.h"
#include "trans/simu/simu_stream.h"

namespace UC::CacheStore {
namespace {

class DeferredStream : public Trans::SimuStream {
public:
    std::deque<std::function<void()>> pending;
    size_t submissions{0};
    size_t synchronizations{0};
    size_t failSubmission{0};
    bool failSync{false};

    Status Setup() override { return Status::OK(); }
    Status Submit(void* destination, const void* source, size_t bytes)
    {
        pending.push_back([=] { std::memcpy(destination, source, bytes); });
        ++submissions;
        return submissions == failSubmission ? Status::Error("injected submit failure") : Status::OK();
    }
    Status HostToDeviceAsync(void* host, void* device, size_t bytes) override
    { return Submit(device, host, bytes); }
    Status DeviceToHostAsync(void* device, void* host, size_t bytes) override
    { return Submit(host, device, bytes); }
    Status Synchronized() override
    {
        ++synchronizations;
        if (failSync) { return Status::Error("injected sync failure"); }
        while (!pending.empty()) {
            auto next = std::move(pending.front());
            pending.pop_front();
            next();
        }
        return Status::OK();
    }
};

std::shared_ptr<void> Arena(size_t bytes)
{
    return std::shared_ptr<void>(new char[bytes], [](void* ptr) { delete[] static_cast<char*>(ptr); });
}

TEST(StagedCopyTest, HoldsSnapshotUntilDmaCompletes)
{
    auto runtime = std::make_shared<DeferredStream>();
    StagedCopy copy(runtime, Arena(8), 8);
    std::array<char, 4> source{1, 2, 3, 4}, destination{};
    void* devices[]{destination.data()};
    ASSERT_TRUE(copy.Copy(source.data(), devices, {4}, true).Success());
    source.fill(9);
    EXPECT_EQ(destination[0], 0);
    ASSERT_TRUE(copy.Synchronize().Success());
    EXPECT_EQ(destination, (std::array<char, 4>{1, 2, 3, 4}));
}

TEST(StagedCopyTest, ReusesBoundedArenaOnlyAfterCompletion)
{
    auto runtime = std::make_shared<DeferredStream>();
    StagedCopy copy(runtime, Arena(4), 4);
    std::array<char, 10> source{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, destination{};
    void* devices[]{destination.data()};
    ASSERT_TRUE(copy.Copy(source.data(), devices, {10}, true).Success());
    EXPECT_EQ(runtime->synchronizations, 2);
    EXPECT_EQ(runtime->pending.size(), 1);
    ASSERT_TRUE(copy.Synchronize().Success());
    EXPECT_EQ(source, destination);
}

TEST(StagedCopyTest, WritesSharedDestinationOnlyAfterDmaCompletion)
{
    auto runtime = std::make_shared<DeferredStream>();
    StagedCopy copy(runtime, Arena(8), 8);
    std::array<char, 4> device{1, 2, 3, 4}, host{9, 9, 9, 9};
    void* devices[]{device.data()};
    ASSERT_TRUE(copy.Copy(host.data(), devices, {4}, false).Success());
    EXPECT_EQ(host[0], 9);
    ASSERT_TRUE(copy.Synchronize().Success());
    EXPECT_EQ(host, device);
}

TEST(StagedCopyTest, PreservesGhostOffsetsAndUntouchedPadding)
{
    auto runtime = std::make_shared<DeferredStream>();
    StagedCopy copy(runtime, Arena(16), 16);
    std::array<char, 2> first{1, 2}, last{3, 4};
    std::array<char, 7> host{9, 9, 9, 9, 9, 9, 9};
    void* devices[]{first.data(), nullptr, nullptr, last.data()};
    ASSERT_TRUE(copy.Copy(host.data(), devices, {2, 3, 0, 2}, false).Success());
    ASSERT_TRUE(copy.Synchronize().Success());
    EXPECT_EQ(host, (std::array<char, 7>{1, 2, 9, 9, 9, 3, 4}));
    first.fill(0);
    last.fill(0);
    ASSERT_TRUE(copy.Copy(host.data(), devices, {2, 3, 0, 2}, true).Success());
    ASSERT_TRUE(copy.Synchronize().Success());
    EXPECT_EQ(first, (std::array<char, 2>{1, 2}));
    EXPECT_EQ(last, (std::array<char, 2>{3, 4}));
}

TEST(StagedCopyTest, DirectionChangeCompletesPriorWriteback)
{
    auto runtime = std::make_shared<DeferredStream>();
    StagedCopy copy(runtime, Arena(8), 8);
    char first = 7, host = 0, second = 0;
    void* source[]{&first};
    void* destination[]{&second};
    ASSERT_TRUE(copy.Copy(&host, source, {1}, false).Success());
    ASSERT_TRUE(copy.Copy(&host, destination, {1}, true).Success());
    ASSERT_TRUE(copy.Synchronize().Success());
    EXPECT_EQ(second, 7);
}

TEST(StagedCopyTest, BoundsWritebackMetadataIndependentlyOfBytes)
{
    auto runtime = std::make_shared<DeferredStream>();
    StagedCopy copy(runtime, Arena(1024), 1024);
    std::array<char, 257> device{}, host{};
    device.fill(7);
    std::vector<void*> addresses;
    for (auto& byte : device) { addresses.push_back(&byte); }
    ASSERT_TRUE(copy.Copy(host.data(), addresses.data(), std::vector<size_t>(257, 1), false).Success());
    EXPECT_EQ(runtime->synchronizations, 1);
    ASSERT_TRUE(copy.Synchronize().Success());
    EXPECT_EQ(host, device);
}

TEST(StagedCopyTest, RejectsOverflowBeforeAnySubmission)
{
    auto runtime = std::make_shared<DeferredStream>();
    StagedCopy copy(runtime, Arena(8), 8);
    void* devices[]{nullptr, nullptr};
    EXPECT_TRUE(copy.Copy(nullptr, devices, {std::numeric_limits<size_t>::max(), 1}, true).Failure());
    EXPECT_EQ(runtime->submissions, 0);
}

TEST(StagedCopyTest, RejectsMissingActiveHostAddress)
{
    auto runtime = std::make_shared<DeferredStream>();
    StagedCopy copy(runtime, Arena(8), 8);
    char device = 0;
    void* devices[]{&device};
    EXPECT_TRUE(copy.Copy(nullptr, devices, {1}, true).Failure());
    EXPECT_EQ(runtime->submissions, 0);
}

TEST(StagedCopyTest, EmptyAndGhostOnlyCopiesDoNotSubmit)
{
    auto runtime = std::make_shared<DeferredStream>();
    StagedCopy copy(runtime, Arena(8), 8);
    void* devices[]{nullptr};
    EXPECT_TRUE(copy.Copy(nullptr, nullptr, {}, true).Success());
    EXPECT_TRUE(copy.Copy(nullptr, devices, {4}, false).Success());
    EXPECT_EQ(runtime->submissions, 0);
}

TEST(StagedCopyTest, PartialSubmissionFailureNeverPublishesWritebacks)
{
    auto runtime = std::make_shared<DeferredStream>();
    runtime->failSubmission = 2;
    StagedCopy copy(runtime, Arena(8), 8);
    std::array<char, 2> device{1, 2}, host{9, 9};
    void* devices[]{&device[0], &device[1]};
    EXPECT_TRUE(copy.Copy(host.data(), devices, {1, 1}, false).Failure());
    EXPECT_EQ(runtime->pending.size(), 2);
    EXPECT_TRUE(copy.Synchronize().Failure());
    EXPECT_TRUE(runtime->pending.empty());
    EXPECT_EQ(host, (std::array<char, 2>{9, 9}));
    EXPECT_TRUE(copy.Copy(host.data(), devices, {1, 1}, false).Failure());
    EXPECT_EQ(runtime->submissions, 2);
}

TEST(StagedCopyTest, SyncFailureRetainsArenaAndSuppressesWritebackOnRetry)
{
    auto runtime = std::make_shared<DeferredStream>();
    auto arena = Arena(8);
    std::weak_ptr<void> lifetime = arena;
    char device = 7, host = 9;
    void* devices[]{&device};
    {
        StagedCopy copy(runtime, std::move(arena), 8);
        ASSERT_TRUE(copy.Copy(&host, devices, {1}, false).Success());
        runtime->failSync = true;
        EXPECT_TRUE(copy.Synchronize().Failure());
        EXPECT_FALSE(lifetime.expired());
        EXPECT_EQ(host, 9);
        runtime->failSync = false;
        EXPECT_TRUE(copy.Synchronize().Failure());
        EXPECT_TRUE(runtime->pending.empty());
        EXPECT_EQ(host, 9);
    }
    EXPECT_TRUE(lifetime.expired());
}

TEST(StagedCopyTest, ShutdownDrainsDmaWithoutWritingExpiredQueueDestinations)
{
    auto runtime = std::make_shared<DeferredStream>();
    char device = 7, host = 9;
    void* devices[]{&device};
    auto arena = Arena(8);
    std::weak_ptr<void> lifetime = arena;
    {
        StagedCopy copy(runtime, std::move(arena), 8);
        ASSERT_TRUE(copy.Copy(&host, devices, {1}, false).Success());
    }
    EXPECT_TRUE(runtime->pending.empty());
    EXPECT_EQ(host, 9);
    EXPECT_TRUE(lifetime.expired());
}

TEST(StagedCopyTest, CancelDiscardsPendingWritesAcrossIndependentStreams)
{
    auto first = std::make_shared<DeferredStream>();
    auto second = std::make_shared<DeferredStream>();
    StagedCopy a(first, Arena(8), 8), b(second, Arena(8), 8);
    char device = 7, hostA = 9, hostB = 9;
    void* devices[]{&device};
    ASSERT_TRUE(a.Copy(&hostA, devices, {1}, false).Success());
    ASSERT_TRUE(b.Copy(&hostB, devices, {1}, false).Success());
    auto error = Status::Error("queue task failed");
    a.Cancel(error);
    b.Cancel(error);
    EXPECT_TRUE(a.Synchronize().Failure());
    EXPECT_TRUE(b.Synchronize().Failure());
    EXPECT_EQ(hostA, 9);
    EXPECT_EQ(hostB, 9);
}

TEST(StagingConfig, RequiresSharedOrdinaryCopiesAndBoundedCapacity)
{
    Config config;
    config.cacheSdmaDirect = false;
    config.cacheHostMemoryMode = "staged";
    EXPECT_TRUE(config.ValidateHostMemory().Success());
    EXPECT_EQ(config.EffectiveHostMemoryMode(), "pageable");
    EXPECT_EQ(config.EffectiveStagingBytes(), 8ULL << 20);
    config.cacheStagingBufferBytes = 0;
    EXPECT_TRUE(config.ValidateHostMemory().Failure());
    config.cacheStagingBufferBytes = 8;
    config.streamNumber = std::numeric_limits<size_t>::max();
    EXPECT_TRUE(config.ValidateHostMemory().Failure());
    config.streamNumber = 2;
    config.shareBufferEnable = false;
    EXPECT_TRUE(config.ValidateHostMemory().Failure());
    config.shareBufferEnable = true;
    config.useGdr = true;
    EXPECT_TRUE(config.ValidateHostMemory().Failure());
    config.useGdr = false;
    config.cacheIOAggregation = true;
    EXPECT_TRUE(config.ValidateHostMemory().Failure());
    config.cacheIOAggregation = false;
    config.cacheSdmaDirect = true;
    EXPECT_TRUE(config.ValidateHostMemory().Failure());
    config.cacheSdmaDirect = false;
    config.cachePinAllowPageableFallback = true;
    EXPECT_TRUE(config.ValidateHostMemory().Failure());
    config.cacheHostMemoryMode = "default";
    EXPECT_EQ(config.EffectiveStagingBytes(), 0);
}

TEST(StagedCopyIntegration, SharedCacheRoundTripAndMultistreamCompletion)
{
    Config config;
    config.uniqueId = UC::Test::Detail::Random().RandomString(16);
    config.deviceId = 0;
    config.tensorSizes = {2, 2};
    config.shardSize = config.blockSize = 4;
    config.bufferCapacity = 4 * 8;
    config.loadExclusiveBufferNumber = 0;
    config.localRankSize = 1;
    config.cacheSdmaDirect = false;
    config.cacheHostMemoryMode = "staged";
    config.cacheStagingBufferBytes = 4;
    TransBuffer writer, reader;
    ASSERT_TRUE(writer.Setup(config).Success());
    ASSERT_TRUE(reader.Setup(config).Success());
    CopyStream stream;
    ASSERT_TRUE(stream.Setup(0, 2, false, config.EffectiveStagingBytes()).Success());
    Detail::BlockId id{};
    id[0] = std::byte{1};
    auto handle = writer.Get(id, 0);
    ASSERT_TRUE(handle);
    std::array<char, 2> first{1, 2}, last{3, 4};
    void* devices[]{first.data(), last.data()};
    ASSERT_TRUE(stream.DeviceToHostAsync(devices, handle.Data(), config.tensorSizes).Success());
    EXPECT_FALSE(handle.Ready());
    ASSERT_TRUE(stream.Synchronize().Success());
    handle.MarkReady();
    auto view = reader.TryGet(id, 0);
    ASSERT_TRUE(view);
    ASSERT_TRUE(view.Ready());
    EXPECT_EQ(std::memcmp(view.Data(), "\1\2\3\4", 4), 0);
    std::array<char, 2> outA{}, outB{}, outC{}, outD{};
    void* targetA[]{outA.data(), outB.data()};
    void* targetB[]{outC.data(), outD.data()};
    ASSERT_TRUE(stream.HostToDeviceAsync(view.Data(), targetA, config.tensorSizes).Success());
    ASSERT_TRUE(stream.HostToDeviceAsync(view.Data(), targetB, config.tensorSizes).Success());
    ASSERT_TRUE(stream.Synchronize().Success());
    EXPECT_EQ(outA, first);
    EXPECT_EQ(outB, last);
    EXPECT_EQ(outC, first);
    EXPECT_EQ(outD, last);
}


TEST(StagedCopyIntegration, FailedSubmissionCancelsWritebacksOnEveryStream)
{
    CopyStream stream;
    ASSERT_TRUE(stream.Setup(0, 2, false, 8).Success());
    char device = 7, host = 9;
    void* devices[]{&device};
    ASSERT_TRUE(stream.DeviceToHostAsync(devices, &host, {1}).Success());
    EXPECT_TRUE(stream.DeviceToHostAsync(devices, nullptr, {1}).Failure());
    EXPECT_EQ(host, 9);
    EXPECT_TRUE(stream.Synchronize().Failure());
    EXPECT_EQ(host, 9);
}

TEST(StagedCopyDeathTest, UnconfirmedDmaCompletionDoesNotFreeItsArena)
{
    EXPECT_EXIT({
        auto runtime = std::make_shared<DeferredStream>();
        bool freed = false;
        auto arena = std::shared_ptr<void>(new char[8], [&](void* ptr) {
            freed = true;
            delete[] static_cast<char*>(ptr);
        });
        char source = 7;
        char destination = 0;
        void* devices[]{&destination};
        {
            StagedCopy copy(runtime, std::move(arena), 8);
            if (copy.Copy(&source, devices, {1}, true).Failure()) { std::_Exit(2); }
            runtime->failSync = true;
        }
        std::_Exit(!freed && !runtime->pending.empty() ? 0 : 1);
    }, testing::ExitedWithCode(0), ".*");
}


TEST(StagedCopyIntegration, SharesDataAcrossProcessesWithPrivateStagingArenas)
{
    Config config;
    config.uniqueId = UC::Test::Detail::Random().RandomString(16);
    config.deviceId = 0;
    config.tensorSizes = {2, 2};
    config.shardSize = config.blockSize = 4;
    config.bufferCapacity = 32;
    config.loadExclusiveBufferNumber = 0;
    config.localRankSize = 1;
    config.cacheSdmaDirect = false;
    config.cacheHostMemoryMode = "staged";
    config.cacheStagingBufferBytes = 4;
    TransBuffer writer;
    ASSERT_TRUE(writer.Setup(config).Success());
    Detail::BlockId id{};
    id[0] = std::byte{1};
    auto handle = writer.Get(id, 0);
    ASSERT_TRUE(handle);
    int ready[2];
    ASSERT_EQ(pipe(ready), 0);
    // Fork before any simulated stream starts its worker thread.
    const auto child = fork();
    ASSERT_GE(child, 0);
    if (child == 0) {
        close(ready[1]);
        char signal = 0;
        ssize_t received;
        do { received = read(ready[0], &signal, 1); } while (received < 0 && errno == EINTR);
        close(ready[0]);
        if (received != 1 || signal != 1) { std::_Exit(2); }
        TransBuffer reader;
        if (reader.Setup(config).Failure()) { std::_Exit(3); }
        auto view = reader.TryGet(id, 0);
        if (!view || !view.Ready()) { std::_Exit(4); }
        CopyStream stream;
        if (stream.Setup(0, 1, false, config.EffectiveStagingBytes()).Failure()) { std::_Exit(5); }
        std::array<char, 2> first{}, last{};
        void* devices[]{first.data(), last.data()};
        if (stream.HostToDeviceAsync(view.Data(), devices, config.tensorSizes).Failure() ||
            stream.Synchronize().Failure()) {
            std::_Exit(6);
        }
        const bool correct = first == std::array<char, 2>{1, 2} &&
                             last == std::array<char, 2>{3, 4};
        std::_Exit(correct ? 0 : 7);
    }
    close(ready[0]);
    bool copied = false;
    {
        CopyStream stream;
        auto status = stream.Setup(0, 1, false, config.EffectiveStagingBytes());
        std::array<char, 2> first{1, 2}, last{3, 4};
        void* devices[]{first.data(), last.data()};
        if (status.Success()) {
            status = stream.DeviceToHostAsync(devices, handle.Data(), config.tensorSizes);
        }
        if (status.Success()) { status = stream.Synchronize(); }
        copied = status.Success();
        if (copied) { handle.MarkReady(); }
    }
    const char signal = copied ? 1 : 0;
    ssize_t sent;
    do { sent = write(ready[1], &signal, 1); } while (sent < 0 && errno == EINTR);
    close(ready[1]);
    int result = 0;
    pid_t waited;
    do { waited = waitpid(child, &result, 0); } while (waited < 0 && errno == EINTR);
    EXPECT_TRUE(copied);
    EXPECT_EQ(sent, 1);
    ASSERT_EQ(waited, child);
    ASSERT_TRUE(WIFEXITED(result));
    EXPECT_EQ(WEXITSTATUS(result), 0);
}

}  // namespace
}  // namespace UC::CacheStore
#endif
