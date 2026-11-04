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
#ifndef UNIFIEDCACHE_CACHE_HOST_REGISTRATION_H
#define UNIFIEDCACHE_CACHE_HOST_REGISTRATION_H

#include "trans/buffer.h"

namespace UC::CacheStore {

enum class HostMemoryMode { Mapped, Pageable, PinOnly };

class HostRegistration {
public:
    struct Operations {
        Status (*map)(void*, size_t, void**);
        Status (*pin)(void*, size_t);
        void (*release)(void*);
    };

    HostRegistration()
        : HostRegistration({Trans::Buffer::RegisterHostBuffer, Trans::Buffer::PinHostBuffer,
                            Trans::Buffer::UnregisterHostBuffer})
    {
    }
    explicit HostRegistration(Operations operations) : operations_(operations) {}
    HostRegistration(const HostRegistration&) = delete;
    HostRegistration& operator=(const HostRegistration&) = delete;
    ~HostRegistration() { Reset(); }

    Status Setup(void* host, size_t size, HostMemoryMode mode, bool allowPageableFallback)
    {
        if (configured_) { return Status::InvalidParam("host registration already configured"); }
        if (!host || size == 0) { return Status::InvalidParam("invalid host registration range"); }
        if (allowPageableFallback && mode != HostMemoryMode::PinOnly) {
            return Status::InvalidParam("pageable fallback requires pin-only registration");
        }
        configured_ = true;
        if (mode == HostMemoryMode::Pageable) { return Status::OK(); }
        registrationStatus_ = mode == HostMemoryMode::PinOnly
                                  ? operations_.pin(host, size)
                                  : operations_.map(host, size, &device_);
        if (registrationStatus_.Failure()) {
            device_ = nullptr;
            return allowPageableFallback ? Status::OK() : registrationStatus_;
        }
        host_ = host;
        if (mode == HostMemoryMode::Mapped && !device_) {
            Reset();
            return Status::Error("host registration returned no device mapping");
        }
        return Status::OK();
    }

    void Reset()
    {
        // Only the object that successfully registered this process's mapping
        // may unregister it; pageable and failed setup own no registration.
        auto* host = host_;
        host_ = nullptr;
        device_ = nullptr;
        if (host) { operations_.release(host); }
    }
    void* DevicePointer() const { return device_; }
    bool Registered() const { return host_ != nullptr; }
    const Status& RegistrationStatus() const { return registrationStatus_; }

private:
    Operations operations_;
    void* host_{nullptr};
    void* device_{nullptr};
    bool configured_{false};
    Status registrationStatus_{Status::OK()};
};

}  // namespace UC::CacheStore

#endif
