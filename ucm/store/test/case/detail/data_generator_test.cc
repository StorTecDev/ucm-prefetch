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
#include <cstdint>
#include <memory>
#include <vector>
#include "detail/data_generator.h"

TEST(DataGeneratorAlignment, SupportsDirectIoIndependentOfHeapLayout)
{
    std::vector<std::unique_ptr<UC::Test::Detail::DataGenerator>> buffers;
    for (size_t i = 0; i < 32; ++i) {
        auto data = std::make_unique<UC::Test::Detail::DataGenerator>(1, 32768);
        data->Generate();
        ASSERT_NE(data->Buffer(), nullptr);
        EXPECT_EQ(reinterpret_cast<uintptr_t>(data->Buffer()) % 4096, uintptr_t{0});
        buffers.push_back(std::move(data));
    }
}
