#include <sys/syscall.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "detail/path_base.h"
#include "posix/cc/aio_impl.h"
#include "ucmstore_v1.h"

extern "C" UC::StoreV1* MakeEarlyPosixStore();

namespace {
using namespace UC;
using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;
constexpr size_t kBytes = 8192;
constexpr auto kTimeout = 2000ms;
constexpr auto kGuardTimeout = 5s;

Detail::BlockId Block(unsigned char value)
{
    Detail::BlockId block{};
    block[0] = std::byte{value};
    return block;
}

class WriteGate {
public:
    struct Operation {
        aio_context_t context{};
        iocb* held{};
        bool accepted{false};
        bool cancelled{false};
        bool completed{false};
        Clock::time_point acceptedAt{};
        Clock::time_point cancelledAt{};
        PosixStore::AioImpl::Result result{};
    };

    WriteGate()
    {
        PosixStore::TestHooks::SetAioSubmitHook(
            [this](aio_context_t context, int64_t count, iocb** requests) {
                auto* request = requests[0];
                if (count != 1 || request->aio_lio_opcode != IOCB_CMD_PWRITE) {
                    return Submit(context, count, requests);
                }
                const auto value = *reinterpret_cast<const unsigned char*>(request->aio_buf);
                if (value < 1 || value > operations_.size()) {
                    return Submit(context, count, requests);
                }
                const auto index = value - 1;
                auto* callback =
                    reinterpret_cast<PosixStore::AioImpl::Callback*>(request->aio_data);
                *callback = [this, index, original = std::move(*callback)](auto result) {
                    original(result);
                    std::lock_guard<std::mutex> lock(mutex_);
                    operations_[index].result = result;
                    operations_[index].completed = true;
                    changed_.notify_all();
                };
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    auto& operation = operations_[index];
                    operation.accepted = true;
                    operation.acceptedAt = Clock::now();
                    if (!draining_) {
                        operation.context = context;
                        operation.held = request;
                        changed_.notify_all();
                        return int32_t{1};
                    }
                    changed_.notify_all();
                }
                return Submit(context, count, requests);
            });
        PosixStore::TestHooks::SetAioCancelHook(
            [this](aio_context_t context, iocb* request, io_event* event) {
                std::lock_guard<std::mutex> lock(mutex_);
                for (auto& operation : operations_) {
                    if (operation.held == request) {
                        operation.cancelled = true;
                        operation.cancelledAt = Clock::now();
                        changed_.notify_all();
                        // A failed cancellation must retain the same ownership as an
                        // accepted device write that cannot yet physically complete.
                        errno = EAGAIN;
                        return int32_t{-1};
                    }
                }
                return static_cast<int32_t>(syscall(SYS_io_cancel, context, request, event));
            });
    }

    ~WriteGate() { PosixStore::TestHooks::ClearAioHooks(); }

    bool Await(size_t index, bool Operation::*state)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        return changed_.wait_for(lock, kGuardTimeout,
                                 [&] { return operations_[index].*state; });
    }

    Operation Snapshot(size_t index)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return operations_[index];
    }

    bool Release(size_t index)
    {
        aio_context_t context;
        iocb* request;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            context = operations_[index].context;
            request = operations_[index].held;
            operations_[index].held = nullptr;
        }
        if (!request) { return true; }
        if (Submit(context, 1, &request) == 1) { return true; }
        std::lock_guard<std::mutex> lock(mutex_);
        operations_[index].held = request;
        return false;
    }

    void Drain()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            // Failed assertions must not leave the next admitted write behind
            // a gate whose owner is already unwinding.
            draining_ = true;
        }
        for (size_t index = 0; index < operations_.size(); ++index) {
            EXPECT_TRUE(Release(index));
        }
    }

private:
    static int32_t Submit(aio_context_t context, int64_t count, iocb** requests)
    {
        return static_cast<int32_t>(syscall(SYS_io_submit, context, count, requests));
    }

    std::mutex mutex_;
    std::condition_variable changed_;
    std::array<Operation, 3> operations_{};
    bool draining_{false};
};

class UCPosixWriteDeadlineTest : public UC::Test::Detail::PathBase {
protected:
    void SetUp() override
    {
        PathBase::SetUp();
        for (size_t index = 0; index < buffers_.size(); ++index) {
            void* data = nullptr;
            ASSERT_EQ(posix_memalign(&data, 4096, kBytes), 0);
            buffers_[index] = std::shared_ptr<void>(data, std::free);
            std::memset(data, index + 1, kBytes);
        }
        Detail::Dictionary config;
        config.Set("storage_backends", std::vector<std::string>{Path()});
        config.SetNumber("device_id", 0);
        config.SetNumber("tensor_size", kBytes);
        config.SetNumber("shard_size", kBytes);
        config.SetNumber("block_size", kBytes);
        config.SetNumber("data_dir_shard_bytes", 0);
        config.SetNumber("posix_open_concurrency", 2);
        config.SetNumber("posix_commit_concurrency", 1);
        config.SetNumber("timeout_ms", kTimeout.count());
        config.SetNumber("posix_write_inflight_limit", 1);
        config.Set("posix_io_engine", std::string{"aio"});
        config.Set("io_direct", true);
        config.Set("posix_gc_enable", false);
        store_.reset(MakeEarlyPosixStore());
        ASSERT_TRUE(store_->Setup(config).Success());
        gate_ = std::make_unique<WriteGate>();
    }

    void TearDown() override
    {
        // Wait may continue draining after its deadline. Release physical I/O
        // before joining those callers, including on a fatal assertion.
        if (gate_) { gate_->Drain(); }
        for (auto& wait : waits_) {
            if (wait.valid()) { wait.wait(); }
        }
        store_.reset();
        gate_.reset();
        PathBase::TearDown();
    }

    Expected<Detail::TaskHandle> Dump(size_t index)
    {
        return store_->Dump({{Block(index + 1), 0, {buffers_[index].get()}}});
    }

    void StartWait(size_t index, Detail::TaskHandle task)
    {
        waits_[index] = std::async(std::launch::async, [this, task] { return store_->Wait(task); });
    }

    std::unique_ptr<StoreV1> store_;
    std::unique_ptr<WriteGate> gate_;
    std::array<std::shared_ptr<void>, 4> buffers_;
    std::array<std::future<Status>, 3> waits_;
};

TEST_F(UCPosixWriteDeadlineTest, QueuedWritesKeepTheirDeadlineAcrossAdmission)
{
    auto first = Dump(0);
    ASSERT_TRUE(first);
    ASSERT_TRUE(gate_->Await(0, &WriteGate::Operation::accepted));
    const auto queuedAt = Clock::now();
    auto second = Dump(1);
    auto third = Dump(2);
    ASSERT_TRUE(second);
    ASSERT_TRUE(third);
    StartWait(0, first.Value());
    StartWait(1, second.Value());
    StartWait(2, third.Value());

    // Most of the later tasks' budget is spent behind one slow write. The
    // second write must not receive a fresh timeout when its permit arrives.
    std::this_thread::sleep_until(queuedAt + kTimeout * 3 / 4);
    EXPECT_FALSE(gate_->Snapshot(1).accepted);
    EXPECT_FALSE(gate_->Snapshot(2).accepted);
    ASSERT_TRUE(gate_->Release(0));
    ASSERT_EQ(waits_[0].wait_for(kGuardTimeout), std::future_status::ready);
    EXPECT_EQ(waits_[0].get(), Status::OK());
    ASSERT_TRUE(gate_->Await(1, &WriteGate::Operation::accepted));
    ASSERT_TRUE(gate_->Await(1, &WriteGate::Operation::cancelled));
    const auto secondIo = gate_->Snapshot(1);
    EXPECT_LT(secondIo.cancelledAt - secondIo.acceptedAt, kTimeout * 3 / 4);
    EXPECT_FALSE(secondIo.completed);
    EXPECT_EQ(waits_[1].wait_for(0ms), std::future_status::timeout);

    ASSERT_EQ(waits_[2].wait_for(kGuardTimeout), std::future_status::ready);
    EXPECT_EQ(waits_[2].get(), Status::Timeout());
    EXPECT_FALSE(gate_->Snapshot(2).accepted);
    ASSERT_TRUE(gate_->Release(1));
    ASSERT_EQ(waits_[1].wait_for(kGuardTimeout), std::future_status::ready);
    EXPECT_EQ(waits_[1].get(), Status::Timeout());
    for (size_t index = 0; index < 2; ++index) {
        ASSERT_TRUE(gate_->Await(index, &WriteGate::Operation::completed));
        const auto result = gate_->Snapshot(index).result;
        EXPECT_EQ(result.error, 0);
        EXPECT_EQ(result.nBytes, static_cast<ssize_t>(kBytes));
    }

    auto recovery = Dump(3);
    ASSERT_TRUE(recovery);
    EXPECT_EQ(store_->Wait(recovery.Value()), Status::OK());
    const auto block = Block(4);
    const auto deadline = Clock::now() + kGuardTimeout;
    bool published = false;
    do {
        auto hit = store_->LookupOnPrefix(&block, 1);
        ASSERT_TRUE(hit);
        published = hit.Value() == 0;
        if (!published) { std::this_thread::sleep_for(1ms); }
    } while (!published && Clock::now() < deadline);
    ASSERT_TRUE(published);
    std::memset(buffers_[0].get(), 0, kBytes);
    auto read = store_->Load({{block, 0, {buffers_[0].get()}}});
    ASSERT_TRUE(read);
    ASSERT_EQ(store_->Wait(read.Value()), Status::OK());
    EXPECT_EQ(std::memcmp(buffers_[0].get(), buffers_[3].get(), kBytes), 0);
}
}  // namespace
