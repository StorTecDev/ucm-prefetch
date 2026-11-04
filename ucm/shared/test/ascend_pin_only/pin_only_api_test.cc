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
#include <acl/acl.h>
#include <array>
#include "trans/buffer.h"

namespace {
int calls;
void* actualHost;
uint64_t actualSize;
uint32_t actualFlags;
int runtimeResult;
}
extern "C" int aclrtHostRegisterV2(void* host, uint64_t size, uint32_t flags)
{
    ++calls;
    actualHost = host;
    actualSize = size;
    actualFlags = flags;
    return runtimeResult;
}

TEST(PinOnlyApi, RejectsInvalidRangesWithoutCallingRuntime)
{
    calls = 0;
    std::array<char, 4096> storage{};
    EXPECT_TRUE(UC::Trans::Buffer::PinHostBuffer(nullptr, storage.size()).Failure());
    EXPECT_TRUE(UC::Trans::Buffer::PinHostBuffer(storage.data(), 0).Failure());
    EXPECT_EQ(calls, 0);
}

TEST(PinOnlyApi, UsesOnlyPinFlagOrFailsWhenSdkHasNoPinApi)
{
    calls = 0;
    runtimeResult = 0;
    std::array<char, 4096> storage{};
    auto result = UC::Trans::Buffer::PinHostBuffer(storage.data(), storage.size());
#ifdef ACL_HOST_REG_PINNED
    ASSERT_TRUE(result.Success());
    EXPECT_EQ(calls, 1);
    EXPECT_EQ(actualHost, storage.data());
    EXPECT_EQ(actualSize, storage.size());
    EXPECT_EQ(actualFlags, ACL_HOST_REG_PINNED);
    runtimeResult = 507899;
    EXPECT_TRUE(UC::Trans::Buffer::PinHostBuffer(storage.data(), storage.size()).Failure());
    EXPECT_EQ(calls, 2);
#else
    EXPECT_TRUE(result.Failure());
    EXPECT_EQ(calls, 0);
#endif
}
