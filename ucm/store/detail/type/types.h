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
#ifndef UNIFIEDCACHE_STORE_DETAIL_TYPE_TYPES_H
#define UNIFIEDCACHE_STORE_DETAIL_TYPE_TYPES_H

#include <array>
#include <atomic>
#include <memory>
#include <cstdint>
#include <string>
#include <vector>

namespace UC::Detail {

using BlockId = std::array<std::byte, 16>; /* 16-byte block hash */
using TaskHandle = std::size_t;            /* Opaque task token (0 = invalid) */

/**
 * @brief Hasher of BlockId
 */
struct BlockIdHasher {
    size_t operator()(const BlockId& blockId) const noexcept
    {
        std::string_view sv(reinterpret_cast<const char*>(blockId.data()), blockId.size());
        return std::hash<std::string_view>{}(sv);
    }
};

/**
 * @brief Describes one shard (slice) of a block.
 */
struct Shard {
    BlockId owner;            /* Parent block identifier */
    std::size_t index;        /* Shard index inside the block */
    std::vector<void*> addrs; /* Tensor or host-slice addresses */
};

// I/O budget only; retired bytes do not imply success, READY or task reaping.
// Each read retires once after physical completion or a terminal pre-submit failure.
// Backends without progress retain whole-task accounting.
struct ReadProgress {
    std::atomic<size_t> retiredBytes{0};
};

/**
 * @brief Batch descriptor for load or dump operations.
 *
 * Inherits from std::vector<Shard> to store the shard list and reserves
 * room for future extensions (priority, deadline, etc.).
 */
struct TaskDesc : std::vector<Shard> {
    using vector::vector; /* Inherit all ctors */
    std::string brief;    /* Description of Task */
    /** Optional: prerequisite handle for dump. Cache stream waits before D2H. */
    uintptr_t prerequisiteHandle{0};
    /** Optional worker-local reservations supplying already pinned load data. */
    std::vector<uint64_t> loadReservations;
    // Each entry maps a complete file to layer destinations. Null destinations
    // discard already cached/peer-owned slices without overwriting live data.
    // A single non-null address denotes a contiguous complete-file destination.
    bool fullBlockRead{false};
    // Each entry owns one complete contiguous file image, never a partial layer.
    bool fullBlockWrite{false};
    std::shared_ptr<ReadProgress> readProgress;
};

}  // namespace UC::Detail

#endif
