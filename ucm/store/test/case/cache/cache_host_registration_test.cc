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
#include <array>
#include <cstring>
#include "cache/cc/trans_buffer.h"
#include "detail/random.h"
#include "cache/cc/global_config.h"
#include "cache/cc/host_registration.h"

namespace UC::CacheStore {
namespace {
class HostRegistrationTest : public testing::Test {
protected:
    static inline int maps;
    static inline int pins;
    static inline int releases;
    static inline bool fail;
    static inline bool missingAlias;
    static inline void* released;
    std::array<char, 4096> storage{};
    void SetUp() override
    {
        maps = pins = releases = 0;
        fail = missingAlias = false;
        released = nullptr;
    }
    static Status Map(void* host, size_t, void** alias)
    {
        ++maps;
        *alias = missingAlias ? nullptr : host;
        return fail ? Status::Error("registration failed") : Status::OK();
    }
    static Status Pin(void*, size_t)
    {
        ++pins;
        return fail ? Status::Error("registration failed") : Status::OK();
    }
    static void Release(void* host) { ++releases; released = host; }
    static HostRegistration::Operations Operations() { return {Map, Pin, Release}; }
};

TEST_F(HostRegistrationTest, PinOnlyOwnsRegistrationWithoutDeviceAlias)
{
    {
        HostRegistration registration(Operations());
        ASSERT_TRUE(registration.Setup(storage.data(), storage.size(), HostMemoryMode::PinOnly, false).Success());
        EXPECT_TRUE(registration.Registered());
        EXPECT_EQ(registration.DevicePointer(), nullptr);
        EXPECT_EQ(pins, 1);
        EXPECT_EQ(maps, 0);
        registration.Reset();
        registration.Reset();
        EXPECT_EQ(releases, 1);
        EXPECT_EQ(released, storage.data());
    }
    EXPECT_EQ(releases, 1);
}

TEST_F(HostRegistrationTest, MappedRegistrationReleasesOnce)
{
    {
        HostRegistration registration(Operations());
        ASSERT_TRUE(registration.Setup(storage.data(), storage.size(), HostMemoryMode::Mapped, false).Success());
        EXPECT_EQ(registration.DevicePointer(), storage.data());
        EXPECT_EQ(maps, 1);
        EXPECT_EQ(pins, 0);
    }
    EXPECT_EQ(releases, 1);
}

TEST_F(HostRegistrationTest, PageableNeverRegistersOrReleases)
{
    {
        HostRegistration registration(Operations());
        EXPECT_TRUE(registration.Setup(storage.data(), storage.size(), HostMemoryMode::Pageable, false).Success());
        EXPECT_FALSE(registration.Registered());
        EXPECT_EQ(registration.DevicePointer(), nullptr);
    }
    EXPECT_EQ(maps + pins + releases, 0);
}

TEST_F(HostRegistrationTest, StrictFailureDoesNotClaimOwnership)
{
    fail = true;
    for (auto mode : {HostMemoryMode::PinOnly, HostMemoryMode::Mapped}) {
        HostRegistration registration(Operations());
        EXPECT_TRUE(registration.Setup(storage.data(), storage.size(), mode, false).Failure());
        EXPECT_TRUE(registration.RegistrationStatus().Failure());
        EXPECT_FALSE(registration.Registered());
        EXPECT_EQ(registration.DevicePointer(), nullptr);
    }
    EXPECT_EQ(releases, 0);
}

TEST_F(HostRegistrationTest, ExplicitFallbackPreservesRegistrationError)
{
    fail = true;
    {
        HostRegistration registration(Operations());
        EXPECT_TRUE(registration.Setup(storage.data(), storage.size(), HostMemoryMode::PinOnly, true).Success());
        EXPECT_TRUE(registration.RegistrationStatus().Failure());
        EXPECT_FALSE(registration.Registered());
        EXPECT_EQ(registration.DevicePointer(), nullptr);
    }
    EXPECT_EQ(pins, 1);
    EXPECT_EQ(maps + releases, 0);
}

TEST_F(HostRegistrationTest, MissingMappingIsReleasedBeforeFailure)
{
    missingAlias = true;
    {
        HostRegistration registration(Operations());
        EXPECT_TRUE(registration.Setup(storage.data(), storage.size(), HostMemoryMode::Mapped, false).Failure());
        EXPECT_FALSE(registration.Registered());
    }
    EXPECT_EQ(releases, 1);
}

TEST_F(HostRegistrationTest, RejectsInvalidRangesAndFallbackBeforeCallingRuntime)
{
    HostRegistration registration(Operations());
    EXPECT_TRUE(registration.Setup(nullptr, storage.size(), HostMemoryMode::PinOnly, false).Failure());
    EXPECT_TRUE(registration.Setup(storage.data(), 0, HostMemoryMode::PinOnly, false).Failure());
    EXPECT_TRUE(registration.Setup(storage.data(), storage.size(), HostMemoryMode::Mapped, true).Failure());
    EXPECT_TRUE(registration.Setup(storage.data(), storage.size(), HostMemoryMode::Pageable, true).Failure());
    EXPECT_EQ(maps + pins + releases, 0);
}

TEST_F(HostRegistrationTest, RejectsSecondSetupWithoutLeakingFirstRegistration)
{
    {
        HostRegistration registration(Operations());
        ASSERT_TRUE(registration.Setup(storage.data(), storage.size(), HostMemoryMode::PinOnly, false).Success());
        EXPECT_TRUE(registration.Setup(storage.data(), storage.size(), HostMemoryMode::PinOnly, false).Failure());
    }
    EXPECT_EQ(pins, 1);
    EXPECT_EQ(releases, 1);
}

TEST(HostMemoryConfig, DefaultIsValidAndExplicitModesRequireSharedBuffer)
{
    Config config;
    EXPECT_TRUE(config.ValidateHostMemory().Success());
    for (const auto* mode : {"pageable", "pin_only"}) {
        config.cacheHostMemoryMode = mode;
        config.shareBufferEnable = false;
        EXPECT_TRUE(config.ValidateHostMemory().Failure());
        config.shareBufferEnable = true;
        EXPECT_TRUE(config.ValidateHostMemory().Success());
        EXPECT_EQ(config.EffectiveHostMemoryMode(), mode);
    }
}

TEST(HostMemoryConfig, DefaultPreservesMappedRegistrationForSharedAndLocalBuffers)
{
    Config config;
    for (bool shared : {false, true}) {
        config.shareBufferEnable = shared;
        ASSERT_TRUE(config.ValidateHostMemory().Success());
        EXPECT_EQ(config.EffectiveHostMemoryMode(), "mapped");
    }
}

TEST(HostMemoryConfig, DefaultPermitsDeviceAliasConsumers)
{
    Config config;
    config.shareBufferEnable = true;
    config.useGdr = true;
    config.cacheIOAggregation = true;
    config.cacheSdmaDirect = true;
    EXPECT_EQ(config.EffectiveHostMemoryMode(), "mapped");
    EXPECT_TRUE(config.ValidateHostMemory().Success());
}

TEST(HostMemoryConfig, RejectsUnknownModeAndImplicitFallback)
{
    Config config;
    config.shareBufferEnable = true;
    config.cacheHostMemoryMode = "pinned";
    EXPECT_TRUE(config.ValidateHostMemory().Failure());
    for (const auto* mode : {"default", "pageable"}) {
        config.cacheHostMemoryMode = mode;
        config.cachePinAllowPageableFallback = true;
        EXPECT_TRUE(config.ValidateHostMemory().Failure());
    }
    config.cacheHostMemoryMode = "pin_only";
    EXPECT_TRUE(config.ValidateHostMemory().Success());
}

TEST(HostMemoryConfig, NonMappedBuffersRejectDeviceAliasConsumers)
{
    for (const auto* mode : {"pageable", "pin_only"}) {
        Config config;
        config.shareBufferEnable = true;
        config.cacheHostMemoryMode = mode;
        config.useGdr = true;
        EXPECT_TRUE(config.ValidateHostMemory().Failure());
        config.useGdr = false;
        config.cacheIOAggregation = true;
        EXPECT_TRUE(config.ValidateHostMemory().Failure());
        config.cacheIOAggregation = false;
        config.cacheSdmaDirect = true;
        EXPECT_TRUE(config.ValidateHostMemory().Failure());
    }
}

#if defined(UCM_HOST_REGISTRATION_SIMU_TEST)
TEST(HostMemoryConfig, SharedPageableAndExplicitFallbackPreserveBytes)
{
    for (const auto* mode : {"pageable", "pin_only"}) {
        Config config;
        config.uniqueId = UC::Test::Detail::Random().RandomString(16);
        config.deviceId = 0;
        config.tensorSizes = {4096};
        config.shardSize = config.blockSize = 4096;
        config.bufferCapacity = 4096 * 4;
        config.loadExclusiveBufferNumber = 0;
        config.localRankSize = 1;
        config.cacheHostMemoryMode = mode;
        config.cachePinAllowPageableFallback = std::string(mode) == "pin_only";
        TransBuffer writer;
        TransBuffer reader;
        ASSERT_TRUE(writer.Setup(config).Success());
        ASSERT_TRUE(reader.Setup(config).Success());
        for (unsigned char i = 1; i <= 3; ++i) {
            Detail::BlockId id{};
            id[0] = std::byte{i};
            auto source = writer.Get(id, 0);
            ASSERT_TRUE(source);
            std::memset(source.Data(), i, 4096);
            source.MarkReady();
            auto destination = reader.TryGet(id, 0);
            ASSERT_TRUE(destination);
            EXPECT_TRUE(destination.Ready());
            EXPECT_EQ(std::memcmp(source.Data(), destination.Data(), 4096), 0);
            EXPECT_EQ(source.DeviceData(), nullptr);
            EXPECT_EQ(destination.DeviceData(), nullptr);
        }
    }
}
#endif

}  // namespace
}  // namespace UC::CacheStore
