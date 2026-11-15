#ifndef UCM_NATIVE_COMPLETION_EVENT_H
#define UCM_NATIVE_COMPLETION_EVENT_H
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>

namespace UC::CacheStore {
// A retry hint for the native receipt consumer, never proof of DMA completion.
// The generation also covers notifications arriving before the consumer sleeps.
class NativeCompletionEvent {
public:
    uint64_t Snapshot()
    {
        std::lock_guard lock(mutex_);
        return generation_;
    }
    void Notify()
    {
        {
            std::lock_guard lock(mutex_);
            ++generation_;
        }
        wake_.notify_one();
    }
    bool Wait(uint64_t observed, std::chrono::milliseconds timeout)
    {
        std::unique_lock lock(mutex_);
        return wake_.wait_for(lock, timeout, [&] { return generation_ != observed; });
    }

private:
    std::mutex mutex_;
    std::condition_variable wake_;
    uint64_t generation_{0};
};
}  // namespace UC::CacheStore
#endif
