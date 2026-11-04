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
#ifndef UNIFIEDCACHE_CACHE_STORE_CC_COPY_STREAM_H
#define UNIFIEDCACHE_CACHE_STORE_CC_COPY_STREAM_H

#include <functional>
#include <memory>
#include <vector>
#include "logger/logger.h"
#include "status/status.h"
#include "trans/device.h"
#include "staged_copy.h"

namespace UC::CacheStore {

class CopyStream {
    int32_t deviceId_{-1};
    size_t streamNumber_{0};
    size_t streamIndex_{0};
    std::vector<std::shared_ptr<Trans::Stream>> streams_;
    std::vector<std::unique_ptr<StagedCopy>> staged_;

    Status CancelStaged(Status status) noexcept
    {
        for (auto& copy : staged_) { copy->Cancel(status); }
        // A failed queue task can release its handles immediately after return.
        // Drop every deferred writeback before draining any remaining stream.
        for (auto& copy : staged_) { (void)copy->Synchronize(); }
        return status;
    }

public:
    Status Setup(const int32_t deviceId, const size_t streamNumber, const bool useGdr,
                 const size_t stagingBytes = 0)
    {
        if (streamNumber == 0) { return Status::InvalidParam("invalid stream number"); }
        if (stagingBytes && useGdr) {
            return Status::InvalidParam("staging requires ordinary host-pointer copies");
        }
        Trans::Device device;
        auto s = device.Setup(deviceId);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to setup device({}).", s, deviceId);
            return s;
        }
        staged_.clear();
        streams_.clear();
        streams_.reserve(streamNumber);
        staged_.reserve(stagingBytes ? streamNumber : 0);
        auto allocator = stagingBytes ? device.MakeBuffer() : nullptr;
        if (stagingBytes && !allocator) { return Status::OutOfMemory(); }
        for (size_t i = 0; i < streamNumber; ++i) {
            std::shared_ptr<Trans::Stream> stream;
            if (useGdr) {
                stream = device.MakeGdrStream();
            } else {
                stream = device.MakeSharedStream();
            }
            if (!stream) [[unlikely]] {
                UC_ERROR("Failed to make stream on device({}).", deviceId);
                return Status::Error();
            }
            if (stagingBytes) {
                auto arena = allocator->MakeHostBuffer(stagingBytes);
                if (!arena) { return Status::OutOfMemory(); }
                staged_.push_back(std::make_unique<StagedCopy>(stream, std::move(arena), stagingBytes));
            }
            streams_.push_back(std::move(stream));
        }
        deviceId_ = deviceId;
        streamNumber_ = streamNumber;
        streamIndex_ = 0;
        return Status::OK();
    }

    Status SetupIoAggregation(const int32_t deviceId, const bool useGdr)
    {
        if (useGdr) {
            return Status::InvalidParam("GDR stream is incompatible with cache IO aggregation");
        }
        Trans::Device device;
        auto s = device.Setup(deviceId);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to setup device({}).", s, deviceId);
            return s;
        }
        auto stream = device.MakeIoAggregationStream();
        if (!stream) [[unlikely]] {
            UC_ERROR("Failed to make cache IO aggregation stream on device({}).", deviceId);
            return Status::Error();
        }
        staged_.clear();
        streams_.clear();
        streams_.push_back(std::move(stream));
        deviceId_ = deviceId;
        streamNumber_ = 1;
        streamIndex_ = 0;
        return Status::OK();
    }

    Status SetupSdmaDirect(const int32_t deviceId, const bool useGdr)
    {
        if (useGdr) {
            return Status::InvalidParam("GDR stream is incompatible with cache SDMA Direct");
        }
        Trans::Device device;
        auto s = device.Setup(deviceId);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to setup device({}).", s, deviceId);
            return s;
        }
        staged_.clear();
        streams_.clear();
        auto stream = device.MakeSdmaDirectStream();
        if (!stream) [[unlikely]] {
            UC_ERROR("Failed to make Cache SDMA Direct stream on device({}).", deviceId);
            return Status::Error();
        }
        // Cache SDMA Direct intentionally uses one stream for stable performance.
        streams_.push_back(std::move(stream));
        deviceId_ = deviceId;
        streamNumber_ = streams_.size();
        streamIndex_ = 0;
        return Status::OK();
    }
    std::shared_ptr<Trans::Stream> NextStream() noexcept
    {
        if (streamNumber_ == 0) [[unlikely]] { return nullptr; }
        auto& stream = streams_[streamIndex_];
        streamIndex_ = (streamIndex_ + 1) % streamNumber_;
        return stream;
    }
    Status AppendCallback(std::function<void(bool)> cb) noexcept
    {
        if (streams_.empty()) [[unlikely]] { return Status::Error(); }
        return streams_.front()->AppendCallback(std::move(cb));
    }
    Status HostToDeviceAsync(void* host, void** device, const std::vector<size_t>& sizes) noexcept
    {
        const auto index = streamIndex_;
        auto stream = NextStream();
        if (!stream) [[unlikely]] { return Status::Error("copy stream is not setup"); }
        if (staged_.empty()) { return stream->HostToDeviceAsync(host, device, sizes); }
        auto status = staged_[index]->Copy(host, device, sizes, true);
        return status.Success() ? status : CancelStaged(status);
    }
    Status DeviceToHostAsync(void** device, void* host, const std::vector<size_t>& sizes) noexcept
    {
        const auto index = streamIndex_;
        auto stream = NextStream();
        if (!stream) [[unlikely]] { return Status::Error("copy stream is not setup"); }
        if (staged_.empty()) { return stream->DeviceToHostAsync(device, host, sizes); }
        auto status = staged_[index]->Copy(host, device, sizes, false);
        return status.Success() ? status : CancelStaged(status);
    }
    Status WaitEvent(const Trans::Event& event) noexcept
    {
        auto status = Status::OK();
        for (auto& stream : streams_) {
            auto s = stream->WaitEvent(event);
            if (s.Success()) { continue; }
            UC_ERROR("Failed({}) to wait event on stream on device({}).", s, deviceId_);
            if (status.Success()) { status = s; }
        }
        return status;
    }
    Status Synchronize() noexcept
    {
        auto status = Status::OK();
        for (size_t i = 0; i < streams_.size(); ++i) {
            auto s = staged_.empty() ? streams_[i]->Synchronized() : staged_[i]->Synchronize();
            if (s.Success()) { continue; }
            UC_ERROR("Failed({}) to synchronize stream on device({}).", s, deviceId_);
            if (status.Success()) { status = s; }
        }
        if (status.Failure() && !staged_.empty()) { return CancelStaged(status); }
        return status;
    }
};

}  // namespace UC::CacheStore

#endif
