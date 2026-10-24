#define main PackedProbeMain
#include "packed_l2_probe.cc"
#undef main

extern "C" StoreV1* MakeCacheStore();

static void OrdinaryPathsRemainUniform()
{
    PackedFixture packed;
    PosixStore rejected;
    Require(rejected.Setup(packed.config).Failure(), "ordinary Posix rejects packed layout");

    Detail::Dictionary config;
    const auto path = packed.path + "/ordinary";
    std::filesystem::create_directory(path);
    config.SetNumber("device_id", 0);
    config.Set("storage_backends", std::vector<std::string>{path});
    config.SetNumber("tensor_size", PackedFixture::stride);
    config.SetNumber("shard_size", PackedFixture::stride);
    config.SetNumber("block_size", 14 * PackedFixture::stride);
    config.Set("posix_io_engine", std::string("aio"));
    config.Set("posix_gc_enable", false);
    config.SetNumber("posix_open_concurrency", 1);
    config.SetNumber("posix_commit_concurrency", 1);
    config.SetNumber("data_dir_shard_bytes", 0);
    PosixStore ordinary;
    Require(ordinary.Setup(config).Success(), "ordinary store setup");
    Require(!ordinary.SupportsFullBlockRead() && ordinary.PackedLayerBytes(0) == 0,
            "ordinary store has no Early layout capability");
    Require(ordinary.FullBlockReadBytes() == 14 * PackedFixture::stride,
            "ordinary disk block size preserved");
    for (size_t layer = 0; layer < 14; ++layer) {
        memset(packed.buffers[layer].get(), layer + 17, PackedFixture::stride);
        auto write = ordinary.Dump({{packed.block, layer, {packed.buffers[layer].get()}}});
        Require(write && ordinary.Wait(write.Value()).Success(), "ordinary layer dump");
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!ordinary.Lookup(&packed.block, 1).Value()[0]) {
        Require(std::chrono::steady_clock::now() < deadline, "ordinary block publication");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    for (size_t layer = 0; layer < 14; ++layer) {
        memset(packed.buffers[layer].get(), 0, PackedFixture::stride);
        auto read = ordinary.Load({{packed.block, layer, {packed.buffers[layer].get()}}});
        Require(read && ordinary.Wait(read.Value()).Success(), "ordinary layer read");
        auto* bytes = static_cast<unsigned char*>(packed.buffers[layer].get());
        Require(std::all_of(bytes, bytes + PackedFixture::stride,
                           [layer](auto byte) { return byte == layer + 17; }),
                "ordinary payload including original padding preserved");
    }
    Require(!ordinary.Load(packed.Read()), "ordinary path rejects full-block descriptor");
    config.Set<StoreV1*>("store_backend", &ordinary);
    config.Set("tensor_size_list", std::vector<ssize_t>{73728, 8448});
    config.SetNumber("tensor_size", 0);
    config.Set("unique_id", std::string("early-scope-ordinary"));
    config.Set("share_buffer_enable", false);
    config.SetNumber("cache_buffer_capacity_gb", 1);
    config.SetNumber("local_rank_size", 1);
    {
        std::unique_ptr<StoreV1> cache{MakeCacheStore()};
        Require(cache->Setup(config).Success(), "ordinary layerwise cache setup");
    }
    config.Set("cache_full_block_load", true);
    {
        std::unique_ptr<StoreV1> cache{MakeCacheStore()};
        Require(cache->Setup(config).Failure(), "ordinary cache rejects Early-only full-block option");
    }
    for (const auto& entry : std::filesystem::recursive_directory_iterator(path)) {
        Require(entry.path().filename().string().find(".ucm-packed") == std::string::npos,
                "ordinary storage never enters compact namespace");
        Require(entry.path().filename().string().find(".ucm_read_leases") == std::string::npos,
                "ordinary storage does not acquire Early read leases");
    }
    std::cout << "PASS ordinary layerwise bytes and original layout, Early-only option rejection\n";
}

int main()
{
    try {
        OrdinaryPathsRemainUniform();
        return PackedProbeMain();
    } catch (const std::exception& error) {
        std::cerr << "FAIL scope: " << error.what() << '\n';
        return 1;
    }
}
