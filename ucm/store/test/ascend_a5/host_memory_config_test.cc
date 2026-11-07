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
#include "global_config.h"

namespace UC::CacheStore {
namespace {

TEST(A5HostMemoryConfig, DefaultRegistersSharedMemoryWithoutStagingOrFallback)
{
    Config config;
    EXPECT_EQ(config.cacheHostMemoryMode, "default");
    ASSERT_TRUE(config.shareBufferEnable);
    EXPECT_EQ(config.EffectiveHostMemoryMode(), "pin_only");
    EXPECT_EQ(config.EffectiveStagingBytes(), 0);
    EXPECT_FALSE(config.cachePinAllowPageableFallback);
    EXPECT_TRUE(config.ValidateHostMemory().Success());
}

TEST(A5HostMemoryConfig, PrivateBuffersRetainMappedRegistration)
{
    Config config;
    config.shareBufferEnable = false;
    EXPECT_EQ(config.EffectiveHostMemoryMode(), "mapped");
    EXPECT_TRUE(config.ValidateHostMemory().Success());
}

TEST(A5HostMemoryConfig, ExplicitModesOverrideThePlatformDefault)
{
    Config config;
    for (const auto* mode : {"pin_only", "pageable"}) {
        config.cacheHostMemoryMode = mode;
        EXPECT_EQ(config.EffectiveHostMemoryMode(), mode);
        EXPECT_EQ(config.EffectiveStagingBytes(), 0);
        EXPECT_TRUE(config.ValidateHostMemory().Success());
    }
}

TEST(A5HostMemoryConfig, DefaultRejectsDeviceAliasConsumers)
{
    Config config;
    config.useGdr = true;
    EXPECT_TRUE(config.ValidateHostMemory().Failure());
    config.useGdr = false;
    config.cacheIOAggregation = true;
    EXPECT_TRUE(config.ValidateHostMemory().Failure());
    config.cacheIOAggregation = false;
    config.cacheSdmaDirect = true;
    EXPECT_TRUE(config.ValidateHostMemory().Failure());
}

TEST(A5HostMemoryConfig, FallbackMustBeExplicitlyOptedIntoWithPinOnly)
{
    Config config;
    config.cachePinAllowPageableFallback = true;
    EXPECT_TRUE(config.ValidateHostMemory().Failure());
    config.cacheHostMemoryMode = "pin_only";
    EXPECT_TRUE(config.ValidateHostMemory().Success());
}

}  // namespace
}  // namespace UC::CacheStore
