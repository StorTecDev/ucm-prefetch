/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
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
 */
#ifndef UNIFIEDCACHE_PIPELINE_HEALTH_BREAKER_STORE_H
#define UNIFIEDCACHE_PIPELINE_HEALTH_BREAKER_STORE_H

#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include "health_check_executor.h"
#include "store_health_config.h"
#include "store_health_state.h"
#include "ucmstore_v1.h"
#include "early_batch.h"

namespace UC::PipelineStore {

class HealthBreakerStore : public StoreV1 {
public:
    HealthBreakerStore() = default;
    ~HealthBreakerStore() override;

    Status Setup(StoreV1* store, std::string storeId, const StoreHealthConfig& config);
    Status Start();
    void Stop();
    bool Enabled() const { return !healthState_ || healthState_->Enabled(); }
    size_t FailureCount() const;
    size_t SampleCount() const;

    Status Setup(const Detail::Dictionary& config) override;
    std::string Readme() const override;
    bool SupportsFullBlockRead() const override
    { return store_ && store_->SupportsFullBlockRead(); }
    bool SupportsFullBlockWrite() const override
    { return store_ && store_->SupportsFullBlockWrite(); }
    Expected<std::shared_ptr<void>> AcquireWriteLease(const Detail::BlockId& block) override
    {
        if (!Enabled()) { return Status::StoreUnhealthy(); }
        return store_->AcquireWriteLease(block);
    }
    Status CommitDump(const Detail::BlockId& block) override
    { return store_->CommitDump(block); }
    size_t FullBlockReadBytes() const override
    { return store_ ? store_->FullBlockReadBytes() : 0; }
    size_t PackedLayerBytes(size_t index) const override
    { return store_ ? store_->PackedLayerBytes(index) : 0; }
    Expected<std::vector<uint8_t>> Lookup(const Detail::BlockId* blocks, size_t num) override;
    Expected<ssize_t> LookupOnPrefix(const Detail::BlockId* blocks, size_t num) override;
    Expected<ssize_t> LookupOnReverse(const Detail::BlockId* blocks, size_t num) override;
    void Prefetch(const Detail::BlockId* blocks, size_t num) override;
    Expected<std::shared_ptr<ReadLease>> AcquireReadLease(
        const Detail::BlockId* blocks, size_t num) override
    {
        if (!Enabled()) { return Status::StoreUnhealthy(); }
        return store_->AcquireReadLease(blocks, num);
    }
    Expected<PrefixReservation> ReservePrefix(const Detail::BlockId* blocks, size_t num) override
    {
        if (!Enabled()) { return Status::StoreUnhealthy(); }
        auto result = store_->ReservePrefix(blocks, num);
        if (result && result.Value().batch) {
            result.Value().batch->SetHealthGuard([this] { return Enabled(); });
        }
        return result;
    }
    Expected<PrefetchResult> PrefetchReserved(uint64_t handle) override
    {
        if (!Enabled()) { return Status::StoreUnhealthy(); }
        return store_->PrefetchReserved(handle);
    }
    void ReleaseReservation(uint64_t handle) override { store_->ReleaseReservation(handle); }
    Status CheckHealth() override;
    Expected<Detail::TaskHandle> Load(Detail::TaskDesc task) override;
    Expected<Detail::TaskHandle> Dump(Detail::TaskDesc task) override;
    Expected<bool> Check(Detail::TaskHandle taskId) override;
    Status Wait(Detail::TaskHandle taskId) override;
    Status WaitDumpSources(Detail::TaskHandle taskId) override
    { return store_->WaitDumpSources(taskId); }  // accepted tasks drain even when unhealthy


private:
    void UpdateState(bool changed, const char* source);
    void RecordProbeMetrics(bool healthy);
    void RecordEffectiveHealth();
    void ProbeLoop();

    StoreV1* store_{nullptr};
    std::string storeId_;
    StoreHealthConfig config_{};
    mutable std::mutex healthMutex_;
    std::unique_ptr<StoreHealthState> healthState_;
    std::mutex stopMutex_;
    std::condition_variable stopCv_;
    bool stop_{false};
    std::thread probeThread_;
    std::unique_ptr<Detail::HealthCheckExecutor> healthCheck_;
};

}  // namespace UC::PipelineStore

#endif
