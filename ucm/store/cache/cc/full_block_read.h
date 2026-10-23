#ifndef UCM_CACHE_FULL_BLOCK_READ_H
#define UCM_CACHE_FULL_BLOCK_READ_H

#include "early_buffer_pins.h"
#include "early_read_completion.h"
#include "early_task_pins.h"

namespace UC::CacheStore {

// Keeps every destination alive even when a native task fails before its turn
// in the H2D queue. An accepted read must drain before any pin is released.
class FullBlockRead {
    StoreV1& backend_;
    EarlyTaskPins pins_;
    std::vector<size_t> owners_;
    Detail::TaskHandle handle_{0};
    Status status_{Status::OK()};

public:
    explicit FullBlockRead(StoreV1& backend) : backend_(backend) {}
    ~FullBlockRead() { Wait(); }

    Status Submit(TransBuffer& buffer, const Detail::Shard& shard,
                  const TransBuffer::Handle& requested, size_t shardsPerBlock)
    {
        pins_.resize(shardsPerBlock);
        Detail::Shard block{shard.owner, 0, std::vector<void*>(shardsPerBlock, nullptr)};
        for (size_t i = 0; i < shardsPerBlock; ++i) {
            pins_[i] = i == shard.index ? requested
                : EarlyBufferPins::Acquire(buffer, shard.owner, i, true);
            if (!pins_[i]) { return Status::NoSpace(); }
            if (pins_[i].Owner() && !pins_[i].Ready()) {
                owners_.push_back(i);
                block.addrs[i] = pins_[i].Data();
            }
        }
        Detail::TaskDesc task{std::move(block)};
        task.fullBlockRead = true;
        task.brief = "FullBlockL2ToL1";
        auto read = backend_.Load(std::move(task));
        if (!read) { return read.Error(); }
        handle_ = read.Value();
        return Status::OK();
    }

    Status Wait()
    {
        if (handle_) {
            status_ = CompleteEarlyRead(backend_, handle_, owners_,
                [&](size_t i) -> TransBuffer::Handle& { return pins_[i]; });
            handle_ = 0;
        }
        return status_;
    }
};

}  // namespace UC::CacheStore
#endif
