#include <atomic>
#include <cstring>
#include <filesystem>
#include <future>
#include <iostream>
#include <stdexcept>
#include <sys/syscall.h>
#include "posix/cc/posix_store.cc"

using namespace UC;
using namespace UC::PosixStore;

static void Require(bool ok, const char* why)
{
    if (!ok) { throw std::runtime_error(why); }
}

struct Fixture {
    static constexpr size_t layerBytes = 4096;
    static constexpr size_t layers = 14;
    std::string path;
    Detail::BlockId block{};
    std::vector<std::shared_ptr<void>> buffers;
    UC::PosixStore::PosixStore store;
    Fixture(size_t timeout = 5000)
    {
        char pattern[] = "/tmp/ucm-full-block-probe-XXXXXX";
        path = mkdtemp(pattern);
        block[0] = std::byte{42};
        Detail::Dictionary config;
        config.SetNumber("device_id", 0);
        config.Set("storage_backends", std::vector<std::string>{path});
        config.SetNumber("tensor_size", layerBytes);
        config.SetNumber("shard_size", layerBytes);
        config.SetNumber("block_size", layers * layerBytes);
        config.Set("posix_io_engine", std::string("aio"));
        config.SetNumber("timeout_ms", timeout);
        config.SetNumber("posix_open_concurrency", size_t{1});
        config.SetNumber("posix_commit_concurrency", size_t{1});
        config.SetNumber("data_dir_shard_bytes", size_t{0});
        Require(store.Setup(config).Success(), "Posix setup");
        Require(store.SupportsFullBlockRead(), "AIO capability");
        for (size_t i = 0; i < layers; ++i) {
            void* buffer = nullptr;
            Require(posix_memalign(&buffer, 4096, layerBytes) == 0, "aligned allocation");
            buffers.emplace_back(buffer, std::free);
            memset(buffer, i + 1, layerBytes);
            Detail::TaskDesc dump{{block, i, {buffer}}};
            auto task = store.Dump(std::move(dump));
            Require(bool(task) && store.Wait(task.Value()).Success(), "layer dump");
        }
        const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        for (;;) {
            auto hit = store.Lookup(&block, 1);
            if (hit && hit.Value()[0]) { break; }
            Require(std::chrono::steady_clock::now() < end, "block commit");
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    ~Fixture()
    {
        TestHooks::ClearAioHooks();
        std::filesystem::remove_all(path);
    }
    Detail::TaskDesc Read(bool partial = false)
    {
        Detail::TaskDesc read;
        read.fullBlockRead = true;
        read.push_back({block, 0, {}});
        for (size_t i = 0; i < layers; ++i) {
            memset(buffers[i].get(), 0xff, layerBytes);
            read[0].addrs.push_back(partial && i % 2 ? nullptr : buffers[i].get());
        }
        return read;
    }
    void Verify(bool partial = false)
    {
        for (size_t i = 0; i < layers; ++i) {
            auto* data = static_cast<unsigned char*>(buffers[i].get());
            for (size_t j = 0; j < layerBytes; ++j) {
                Require(data[j] == (partial && i % 2 ? 0xff : i + 1), "layer bytes");
            }
        }
    }
};

static void FullRead(bool partial)
{
    Fixture f;
    std::atomic<size_t> reads{0};
    TestHooks::SetAioSubmitHook([&](aio_context_t ctx, int64_t nr, iocb** cb) {
        Require(nr == 1 && cb[0]->aio_lio_opcode == IOCB_CMD_PREADV, "single vector read");
        Require(cb[0]->aio_offset == 0 && cb[0]->aio_nbytes == Fixture::layers, "whole file vectors");
        auto* vectors = reinterpret_cast<iovec*>(cb[0]->aio_buf);
        size_t bytes = 0;
        for (size_t i = 0; i < cb[0]->aio_nbytes; ++i) { bytes += vectors[i].iov_len; }
        Require(bytes == Fixture::layerBytes * Fixture::layers, "whole block bytes");
        ++reads;
        return static_cast<int32_t>(syscall(SYS_io_submit, ctx, nr, cb));
    });
    auto read = f.store.Load(f.Read(partial));
    Require(bool(read) && f.store.Wait(read.Value()).Success(), "full read completion");
    Require(reads == 1, "one IO per block");
    f.Verify(partial);
    auto bad = f.Read();
    bad[0].addrs.pop_back();
    Require(!f.store.Load(std::move(bad)), "invalid vector shape rejected");
    Require(!f.store.Dump(f.Read()), "read descriptor rejected by dump");
}

static void TimeoutRetainsVectors()
{
    Fixture f(50);
    std::promise<std::pair<aio_context_t, iocb*>> accepted;
    TestHooks::SetAioSubmitHook([&](aio_context_t ctx, int64_t nr, iocb** cb) {
        accepted.set_value({ctx, *cb});
        return static_cast<int32_t>(nr);
    });
    TestHooks::SetAioCancelHook([](aio_context_t, iocb*, io_event*) {
        errno = EAGAIN;
        return -1;
    });
    auto read = f.store.Load(f.Read(true));
    Require(bool(read), "delayed read accepted");
    auto [ctx, cb] = accepted.get_future().get();
    auto wait = std::async(std::launch::async, [&] { return f.store.Wait(read.Value()); });
    Require(wait.wait_for(std::chrono::milliseconds(150)) == std::future_status::timeout,
            "timeout must retain live read vectors");
    Require(syscall(SYS_io_submit, ctx, 1, &cb) == 1, "delayed real kernel read");
    Require(wait.get() == Status::Timeout(), "deadline propagated after physical drain");
    f.Verify(true);
}

int main()
{
    try {
        FullRead(false);
        FullRead(true);
        TimeoutRetainsVectors();
        std::cout << "PASS: full block read, partial hits, shape validation, delayed completion\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
