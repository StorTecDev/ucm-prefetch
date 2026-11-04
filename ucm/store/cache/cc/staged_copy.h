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
#ifndef UNIFIEDCACHE_CACHE_STORE_CC_STAGED_COPY_H
#define UNIFIEDCACHE_CACHE_STORE_CC_STAGED_COPY_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>
#include "logger/logger.h"
#include "trans/stream.h"

namespace UC::CacheStore {

// One queue thread owns this arena. A completed stream is the reuse barrier.
class StagedCopy {
    struct Writeback {
        void* destination;
        size_t offset;
        size_t size;
    };
    struct State {
        std::shared_ptr<Trans::Stream> stream;
        std::shared_ptr<void> buffer;
        std::array<Writeback, 256> writebacks{};
        size_t capacity{0};
        size_t used{0};
        size_t writebackCount{0};
        bool inFlight{false};
        bool toDevice{true};
        Status failure{Status::OK()};
    };
    std::unique_ptr<State> state_;

    Status Fail(Status status)
    {
        Cancel(status);
        return status;
    }

    Status CopyOne(void* host, void* device, size_t size, bool toDevice)
    {
        auto& state = *state_;
        auto* hostBytes = static_cast<std::byte*>(host);
        auto* deviceBytes = static_cast<std::byte*>(device);
        auto* staging = static_cast<std::byte*>(state.buffer.get());
        while (size != 0) {
            if (state.used == state.capacity || state.writebackCount == state.writebacks.size()) {
                auto status = Synchronize();
                if (status.Failure()) { return status; }
            }
            const auto bytes = std::min(size, state.capacity - state.used);
            auto* slot = staging + state.used;
            if (toDevice) { std::memcpy(slot, hostBytes, bytes); }
            // A failed submission can still have queued work; retain the arena until drained.
            state.inFlight = true;
            auto status = toDevice ? state.stream->HostToDeviceAsync(slot, deviceBytes, bytes)
                                   : state.stream->DeviceToHostAsync(deviceBytes, slot, bytes);
            if (status.Failure()) { return Fail(status); }
            if (!toDevice) {
                state.writebacks[state.writebackCount++] = {hostBytes, state.used, bytes};
            }
            state.used += bytes;
            hostBytes += bytes;
            deviceBytes += bytes;
            size -= bytes;
        }
        return Status::OK();
    }

public:
    StagedCopy(std::shared_ptr<Trans::Stream> stream, std::shared_ptr<void> buffer, size_t capacity)
        : state_(std::make_unique<State>())
    {
        state_->stream = std::move(stream);
        state_->buffer = std::move(buffer);
        state_->capacity = capacity;
    }
    StagedCopy(const StagedCopy&) = delete;
    StagedCopy& operator=(const StagedCopy&) = delete;

    ~StagedCopy()
    {
        // Queue handles may already be gone at shutdown; discard deferred CPU writes.
        state_->writebackCount = 0;
        if (state_->inFlight && state_->stream->Synchronized().Failure()) {
            // Releasing memory still owned by DMA risks corruption. Retain this bounded
            // arena and its stream until process exit if completion cannot be established.
            UC_ERROR("Failed to drain staged DMA; retaining its arena and stream until process exit.");
            (void)state_.release();
        }
    }

    void Cancel(Status status)
    {
        if (state_->failure.Success()) { state_->failure = status; }
        state_->writebackCount = 0;
    }

    Status Copy(void* host, void** devices, const std::vector<size_t>& sizes, bool toDevice)
    {
        auto& state = *state_;
        if (state.failure.Failure()) { return state.failure; }
        if (!state.stream || !state.buffer || state.capacity == 0) {
            return Fail(Status::InvalidParam("invalid staging arena"));
        }
        if (!devices && !sizes.empty()) {
            return Fail(Status::InvalidParam("missing staging device addresses"));
        }
        size_t total = 0;
        for (size_t i = 0; i < sizes.size(); ++i) {
            if (sizes[i] > std::numeric_limits<size_t>::max() - total) {
                return Fail(Status::InvalidParam("staging layout size overflow"));
            }
            if (sizes[i] && devices[i] && !host) {
                return Fail(Status::InvalidParam("missing staging host address"));
            }
            total += sizes[i];
        }
        if (state.used && state.toDevice != toDevice) {
            auto status = Synchronize();
            if (status.Failure()) { return status; }
        }
        state.toDevice = toDevice;
        size_t offset = 0;
        for (size_t i = 0; i < sizes.size(); ++i) {
            if (sizes[i] && devices[i]) {
                auto status = CopyOne(static_cast<std::byte*>(host) + offset, devices[i], sizes[i],
                                      toDevice);
                if (status.Failure()) { return status; }
            }
            offset += sizes[i];
        }
        return Status::OK();
    }

    Status Synchronize()
    {
        auto& state = *state_;
        if (!state.stream) { return Fail(Status::InvalidParam("missing staging stream")); }
        auto status = state.stream->Synchronized();
        if (status.Failure()) { return Fail(status); }
        state.inFlight = false;
        if (state.failure.Success()) {
            auto* staging = static_cast<std::byte*>(state.buffer.get());
            for (size_t i = 0; i < state.writebackCount; ++i) {
                const auto& write = state.writebacks[i];
                std::memcpy(write.destination, staging + write.offset, write.size);
            }
        }
        state.writebackCount = 0;
        state.used = 0;
        return state.failure;
    }
};

}  // namespace UC::CacheStore

#endif
