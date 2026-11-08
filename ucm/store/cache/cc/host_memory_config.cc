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
#include "global_config.h"
#include <limits>

namespace UC::CacheStore {

std::string Config::EffectiveHostMemoryMode() const
{
    if (cacheHostMemoryMode == "staged") { return "pageable"; }
    if (cacheHostMemoryMode != "default") { return cacheHostMemoryMode; }
    return "mapped";
}

Status Config::ValidateHostMemory() const
{
    if (cacheHostMemoryMode != "default" && cacheHostMemoryMode != "pageable" &&
        cacheHostMemoryMode != "pin_only" && cacheHostMemoryMode != "staged") {
        return Status::InvalidParam("cache_host_memory_mode must be default, pageable, pin_only or staged");
    }
    if (cacheHostMemoryMode != "default" && !shareBufferEnable) {
        return Status::InvalidParam("cache_host_memory_mode requires share_buffer_enable");
    }
    if (cachePinAllowPageableFallback && cacheHostMemoryMode != "pin_only") {
        return Status::InvalidParam("cache_pin_allow_pageable_fallback requires pin_only");
    }
    if (cacheHostMemoryMode == "staged" &&
        (cacheStagingBufferBytes == 0 || streamNumber == 0 ||
         streamNumber > std::numeric_limits<size_t>::max() / cacheStagingBufferBytes)) {
        return Status::InvalidParam("invalid per-stream staging capacity or stream count");
    }
    if (shareBufferEnable && EffectiveHostMemoryMode() != "mapped" &&
        (useGdr || cacheIOAggregation || cacheSdmaDirect)) {
        return Status::InvalidParam(
            "non-mapped shared buffers require ordinary host-pointer copies");
    }
    return Status::OK();
}

}  // namespace UC::CacheStore
