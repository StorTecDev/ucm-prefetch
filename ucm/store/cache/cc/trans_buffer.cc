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
#include "trans_buffer.h"
#include "buffer_release_event.h"
#include <atomic>
#include <filesystem>
#include <thread>
#include <unistd.h>
#include "logger/logger.h"
#include "posix_shm.h"
#include "trans/buffer.h"
#include "trans/device.h"

namespace UC::CacheStore {

static constexpr size_t nHashTableBucket = 16411;
static constexpr auto invalidIndex = std::numeric_limits<size_t>::max();

static inline size_t Hash(const Detail::BlockId& blockId, size_t shard)
{
    static UC::Detail::BlockIdHasher blockIdHasher;
    static std::hash<size_t> shardHasher;
    constexpr auto goldenSection = 0x9e3779b97f4a7c15ULL;
    size_t h1 = blockIdHasher(blockId);
    size_t h2 = shardHasher(shard);
    return (h1 ^ (h2 + goldenSection + (h1 << 6) + (h1 >> 2))) % nHashTableBucket;
}

struct BufferMetaNode {
    Detail::BlockId block;
    size_t shard;
    size_t reference;
    size_t hash;
    size_t prev;
    size_t next;
    alignas(64) std::atomic<TransBuffer::State> state;
    std::atomic<int32_t> errorCode;
    void Init()
    {
        reference = 0;
        hash = invalidIndex;
        prev = invalidIndex;
        next = invalidIndex;
        state.store(TransBuffer::State::LOADING, std::memory_order_relaxed);
        errorCode.store(Status::OK().Underlying(), std::memory_order_relaxed);
    }
};
static_assert(std::atomic<TransBuffer::State>::is_always_lock_free, "state must be lock-free");
static_assert(std::atomic<int32_t>::is_always_lock_free, "errorCode must be lock-free");

// Optional metadata for one physical block with independently produced layers.
// Claims/counts are protected by the existing native node lock. Layer readiness
// is observed without that lock, exactly as the aggregate native READY state is.
struct BlockFillState {
    enum class Mode : uint8_t { Empty, FullRead, Layers };
    Mode mode{Mode::Empty};
    size_t readyLayers{0};
};

class BufferStrategy {
    friend class TransBuffer;
    friend class EarlyBufferPins;
protected:
    BufferReleaseEvent localRelease_;
    BufferReleaseEvent* releaseEvent_{&localRelease_};
    struct BaseConfig {
        int32_t deviceId{-1};
        size_t nodeSize{0};
        size_t totalSize{0};
        size_t reservedNumber{0};
        size_t layerCount{0};
        uint64_t layoutTag{0};
    };
    BaseConfig base_;

public:
    BufferStrategy(int32_t deviceId, size_t nodeSize, size_t totalSize, size_t reservedNumber,
                   size_t layers = 0, uint64_t layoutTag = 0)
        : base_({deviceId, nodeSize, totalSize, reservedNumber, layers, layoutTag})
    {
    }
    virtual ~BufferStrategy() = default;
    virtual Status Setup() = 0;
    virtual void BucketLock(size_t iBucket) = 0;
    virtual bool BucketTryLock(size_t iBucket) = 0;
    virtual void BucketUnlock(size_t iBucket) = 0;
    virtual void NodeLock(size_t iNode) = 0;
    virtual void NodeUnlock(size_t iNode) = 0;
    virtual size_t& FirstAt(size_t iBucket) = 0;
    virtual size_t FetchNode(bool allowReserved) = 0;
    virtual size_t NodeCount() const = 0;
    bool CanReserve(size_t index) const { return index < NodeCount() - base_.reservedNumber; }
    virtual void* DataAt(size_t iNode) = 0;
    virtual void* DeviceDataAt(size_t iNode) = 0;
    virtual BufferMetaNode* MetaAt(size_t iNode) = 0;
    virtual void MarkAccessed(size_t iNode) = 0;
    virtual BlockFillState* FillAt(size_t iNode) = 0;
    virtual std::atomic<uint8_t>* LayerAt(size_t iNode, size_t layer) = 0;
};

class LocalBufferStrategy : public BufferStrategy {
    struct BufferHeader {
        size_t buckets[nHashTableBucket];
        // Allocations under different bucket locks share this CLOCK cursor.
        std::atomic<size_t> nodeCursor{0};
        size_t nodeSize;
        size_t nNode;
    };
    struct LocalMutex {
        pthread_mutex_t mutex;
        ~LocalMutex() { pthread_mutex_destroy(&mutex); }
        void Init()
        {
            pthread_mutexattr_t attr;
            pthread_mutexattr_init(&attr);
            pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_PRIVATE);
            pthread_mutexattr_setrobust(&attr, PTHREAD_MUTEX_ROBUST);
            pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_ADAPTIVE_NP);
            pthread_mutex_init(&mutex, &attr);
            pthread_mutexattr_destroy(&attr);
        }
        void Lock() { pthread_mutex_lock(&mutex); }
        bool TryLock() { return pthread_mutex_trylock(&mutex) == 0; }
        void Unlock() { pthread_mutex_unlock(&mutex); }
    };
    struct LocalLock {
        pthread_spinlock_t lock;
        ~LocalLock() { pthread_spin_destroy(&lock); }
        void Init() { pthread_spin_init(&lock, PTHREAD_PROCESS_PRIVATE); }
        void Lock() { pthread_spin_lock(&lock); }
        bool TryLock() { return pthread_spin_trylock(&lock) == 0; }
        void Unlock() { pthread_spin_unlock(&lock); }
    };

    bool ioDirect_{false};
    bool mapHostToDevice_{false};
    BufferHeader header_;
    LocalMutex bucketLocks_[nHashTableBucket];
    std::unique_ptr<LocalLock[]> nodeLocks_;
    std::unique_ptr<BufferMetaNode[]> meta_;
    std::unique_ptr<std::atomic<uint8_t>[]> accessed_;
    std::unique_ptr<BlockFillState[]> fills_;
    std::unique_ptr<std::atomic<uint8_t>[]> layers_;
    std::shared_ptr<void> data_;
    std::byte* dataOnDevice_{nullptr};
    bool registeredMappedHost_{false};

public:
    LocalBufferStrategy(int32_t deviceId, size_t nodeSize, size_t totalSize, size_t reservedNumber,
                        bool ioDirect, bool mapHostToDevice, size_t layers = 0, uint64_t layoutTag = 0)
        : BufferStrategy(deviceId, nodeSize, totalSize, reservedNumber, layers, layoutTag),
          ioDirect_(ioDirect),
          mapHostToDevice_(mapHostToDevice)
    {
    }
    ~LocalBufferStrategy() override
    {
        if (registeredMappedHost_ && data_) { Trans::Buffer::UnregisterHostBuffer(data_.get()); }
    }
    Status Setup() override
    {
        const auto deviceId = base_.deviceId;
        const auto totalSize = base_.totalSize;
        const auto nodeSize = base_.nodeSize;
        auto nNode = totalSize / nodeSize;
        try {
            nodeLocks_ = std::make_unique<LocalLock[]>(nNode);
            meta_ = std::make_unique<BufferMetaNode[]>(nNode);
            accessed_ = std::make_unique<std::atomic<uint8_t>[]>(nNode);
            if (base_.layerCount) {
                fills_ = std::make_unique<BlockFillState[]>(nNode);
                layers_ = std::make_unique<std::atomic<uint8_t>[]>(nNode * base_.layerCount);
                for (size_t i = 0; i < nNode * base_.layerCount; ++i) {
                    layers_[i].store(0, std::memory_order_relaxed);
                }
            }
            for (size_t i = 0; i < nHashTableBucket; i++) { bucketLocks_[i].Init(); }
            for (size_t i = 0; i < nNode; i++) {
                nodeLocks_[i].Init();
                accessed_[i].store(0, std::memory_order_relaxed);
            }
        } catch (const std::exception& e) {
            UC_ERROR("Failed({}) to alloc buffer.", e.what());
            return Status::Error(e.what());
        }
        Trans::Device device;
        auto s = device.Setup(deviceId);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to setup device({}).", s, deviceId);
            return s;
        }
        auto buffer = device.MakeBuffer();
        if (!buffer) [[unlikely]] {
            UC_ERROR("Failed to make buffer on device({}).", deviceId);
            return Status::Error();
        }
        data_ = ioDirect_ ? buffer->MakeHostBuffer4DirectIo(nodeSize * nNode)
                          : buffer->MakeHostBuffer(nodeSize * nNode);
        if (!data_) [[unlikely]] {
            UC_ERROR("Failed to make pinned({}) for device({}).", nodeSize * nNode, deviceId);
            return Status::OutOfMemory();
        }
        if (mapHostToDevice_) {
            void* deviceData = nullptr;
            auto s = Status::OK();
            if (ioDirect_) {
                s = Trans::Buffer::GetHostDevicePointer(data_.get(), &deviceData);
            } else {
                s = Trans::Buffer::RegisterHostBuffer(data_.get(), nodeSize * nNode, &deviceData);
                registeredMappedHost_ = s.Success();
            }
            if (s.Failure()) [[unlikely]] {
                UC_ERROR("Failed({}) to map pinned host buffer({}) to device({}).", s,
                         nodeSize * nNode, deviceId);
                return s;
            }
            dataOnDevice_ = static_cast<std::byte*>(deviceData);
        }
        for (size_t i = 0; i < nHashTableBucket; i++) { header_.buckets[i] = invalidIndex; }
        for (size_t i = 0; i < nNode; i++) { meta_[i].Init(); }
        header_.nodeCursor = 0;
        header_.nodeSize = nodeSize;
        header_.nNode = nNode;
        return Status::OK();
    }
    void BucketLock(size_t iBucket) override { bucketLocks_[iBucket].Lock(); }
    bool BucketTryLock(size_t iBucket) override { return bucketLocks_[iBucket].TryLock(); }
    void BucketUnlock(size_t iBucket) override { bucketLocks_[iBucket].Unlock(); }
    void NodeLock(size_t iNode) override { nodeLocks_[iNode].Lock(); }
    void NodeUnlock(size_t iNode) override { nodeLocks_[iNode].Unlock(); }
    size_t& FirstAt(size_t iBucket) override { return header_.buckets[iBucket]; }
    size_t NodeCount() const override { return header_.nNode; }
    size_t FetchNode(bool allowReserved) override
    {
        const auto total = header_.nNode - (allowReserved ? 0 : base_.reservedNumber);
        for (size_t i = 0; i < 2 * total; ++i) {
            auto cur = header_.nodeCursor.fetch_add(1, std::memory_order_relaxed) % total;
            uint8_t expected = 1;
            if (accessed_[cur].compare_exchange_strong(expected, 0, std::memory_order_relaxed,
                                                       std::memory_order_relaxed)) {
                continue;
            }
            return cur;
        }
        return header_.nodeCursor.fetch_add(1, std::memory_order_relaxed) % total;
    }
    void MarkAccessed(size_t iNode) override
    {
        accessed_[iNode].store(1, std::memory_order_relaxed);
    }
    void* DataAt(size_t iNode) override
    {
        return ((std::byte*)data_.get()) + header_.nodeSize * iNode;
    }
    void* DeviceDataAt(size_t iNode) override
    {
        if (dataOnDevice_ == nullptr) { return nullptr; }
        return dataOnDevice_ + header_.nodeSize * iNode;
    }
    BufferMetaNode* MetaAt(size_t iNode) override { return meta_.get() + iNode; }
    BlockFillState* FillAt(size_t iNode) override { return fills_.get() + iNode; }
    std::atomic<uint8_t>* LayerAt(size_t iNode, size_t layer) override
    { return layers_.get() + iNode * base_.layerCount + layer; }
};

class SharedBufferStrategy : public BufferStrategy {
protected:
    struct ShareMutex {
        pthread_mutex_t mutex;
        ~ShareMutex() = delete;
        void Init()
        {
            pthread_mutexattr_t attr;
            pthread_mutexattr_init(&attr);
            pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
            pthread_mutexattr_setrobust(&attr, PTHREAD_MUTEX_ROBUST);
            pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_ADAPTIVE_NP);
            pthread_mutex_init(&mutex, &attr);
            pthread_mutexattr_destroy(&attr);
        }
        void Lock() { pthread_mutex_lock(&mutex); }
        bool TryLock() { return pthread_mutex_trylock(&mutex) == 0; }
        void Unlock() { pthread_mutex_unlock(&mutex); }
    };
    struct ShareLock {
        pthread_spinlock_t lock;
        ~ShareLock() = delete;
        void Init() { pthread_spin_init(&lock, PTHREAD_PROCESS_SHARED); }
        void Lock() { pthread_spin_lock(&lock); }
        bool TryLock() { return pthread_spin_trylock(&lock) == 0; }
        void Unlock() { pthread_spin_unlock(&lock); }
    };
    static constexpr size_t sharedBufferMagic = (('S' << 16) | ('b' << 8) | 3);
    struct BufferHeader {
        std::atomic<size_t> magic;
        size_t nNode;
        alignas(64) std::atomic<size_t> nodeCursor;
        BufferReleaseEvent releaseEvent;
        char nodeCursorPad[64 - sizeof(std::atomic<size_t>) - sizeof(BufferReleaseEvent)];
        size_t buckets[nHashTableBucket];
        ShareMutex bucketLocks[nHashTableBucket];
        ShareLock nodeLocks[0];
    };
    static_assert(std::atomic<size_t>::is_always_lock_free, "nodeCursor must be lock-free");
    static_assert(std::atomic<uint8_t>::is_always_lock_free, "accessed must be lock-free");

    BufferHeader* header_{nullptr};
    BufferMetaNode* meta_{nullptr};
    std::atomic<uint8_t>* accessed_{nullptr};
    std::byte* data_{nullptr};
    std::byte* dataOnDevice_{nullptr};
    const std::string& uuid_;
    std::string shmName_;
    size_t nodeSize_{0};
    size_t nNode_{0};
    void* addrress_{nullptr};
    size_t totalSize_{0};
    bool ownsShm_{false};

    size_t AccessedOffset() const noexcept
    {
        constexpr auto align = 64;
        auto off = sizeof(BufferHeader) + sizeof(ShareLock) * nNode_;
        return (off + align - 1) & ~(align - 1);
    }
    size_t AccessedSize() const noexcept { return sizeof(std::atomic<uint8_t>) * nNode_; }
    size_t MetaOffset() const noexcept
    {
        constexpr auto align = alignof(BufferMetaNode);
        auto off = AccessedOffset() + AccessedSize();
        return (off + align - 1) & ~(align - 1);
    }
    struct BlockLayoutHeader {
        size_t nodeSize, layerCount, reservedNumber;
        uint64_t layoutTag;
    };
    size_t BlockLayoutOffset() const noexcept
    {
        const auto off = MetaOffset() + sizeof(BufferMetaNode) * nNode_;
        constexpr auto align = alignof(BlockLayoutHeader);
        return (off + align - 1) & ~(align - 1);
    }
    size_t FillOffset() const noexcept
    { return BlockLayoutOffset() + sizeof(BlockLayoutHeader); }
    size_t LayerOffset() const noexcept
    { return FillOffset() + sizeof(BlockFillState) * nNode_; }
    BlockLayoutHeader* BlockLayout() const noexcept
    { return reinterpret_cast<BlockLayoutHeader*>(static_cast<std::byte*>(addrress_) + BlockLayoutOffset()); }
    size_t ExpectedMagic() const noexcept { return sharedBufferMagic + (base_.layerCount ? 1 : 0); }
    size_t DataOffset() const noexcept
    {
        static const auto pageSize = sysconf(_SC_PAGESIZE);
        const auto size = base_.layerCount
            ? LayerOffset() + sizeof(std::atomic<uint8_t>) * nNode_ * base_.layerCount
            : MetaOffset() + sizeof(BufferMetaNode) * nNode_;
        return (size + pageSize - 1) & ~(pageSize - 1);
    }
    size_t DataSize() const noexcept { return nodeSize_ * nNode_; }
    static const std::string& ShmPrefix() noexcept
    {
        static std::string prefix{"uc_shm_cache_"};
        return prefix;
    }
    static void CleanUpShmFileExceptMe(const std::string& me)
    {
        namespace fs = std::filesystem;
        std::string_view prefix = ShmPrefix();
        fs::path shmDir = "/dev/shm";
        if (!fs::exists(shmDir)) { return; }
        const auto now = fs::file_time_type::clock::now();
        const auto keepThreshold = std::chrono::minutes(10);
        for (const auto& entry : fs::directory_iterator(shmDir)) {
            const auto& path = entry.path();
            const auto& name = path.filename().string();
            if (!entry.is_regular_file() || name.compare(0, prefix.size(), prefix) != 0 ||
                name == me) {
                continue;
            }
            try {
                const auto lwt = fs::last_write_time(path);
                if (now - lwt <= keepThreshold) { continue; }
                fs::remove(path);
            } catch (...) {
            }
        }
    }
    static Status MmapShmFile(PosixShm& shmFile, const size_t size, void*& addr,
                              bool needTrunc = true)
    {
        auto s = Status::OK();
        if (needTrunc) {
            s = shmFile.Truncate(size);
            if (s.Failure()) [[unlikely]] {
                UC_ERROR("Failed({}) to trunc file({}) with size({}).", s, shmFile.ShmName(), size);
                return s;
            }
        }
        s = shmFile.MMap(addr, size, true, true, true, true);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to mmap file({}) with size({}).", s, shmFile.ShmName(), size);
            return s;
        }
        return Status::OK();
    }
    Status WaitShmHeaderReady(BufferHeader* header) const
    {
        constexpr auto retryInterval = std::chrono::milliseconds(100);
        constexpr auto maxTryTime = 100;
        auto tryTime = 0;
        do {
            const auto magic = header->magic.load(std::memory_order_acquire);
            if (magic == ExpectedMagic()) { break; }
            if (magic != 0) { return Status::InvalidParam("incompatible shared L1 layout"); }
            if (tryTime > maxTryTime) { return Status::Retry(); }
            std::this_thread::sleep_for(retryInterval);
            tryTime++;
        } while (true);
        return Status::OK();
    }
    Status InitShmBuffer(PosixShm& shmFile)
    {
        auto s = MmapShmFile(shmFile, totalSize_, addrress_);
        if (s.Failure()) [[unlikely]] { return s; }
        header_ = static_cast<BufferHeader*>(addrress_);
        releaseEvent_ = &header_->releaseEvent;
        meta_ = (BufferMetaNode*)(static_cast<std::byte*>(addrress_) + MetaOffset());
        accessed_ = reinterpret_cast<std::atomic<uint8_t>*>(static_cast<std::byte*>(addrress_) +
                                                            AccessedOffset());
        header_->nNode = nNode_;
        if (base_.layerCount) {
            *BlockLayout() = {base_.nodeSize, base_.layerCount, base_.reservedNumber, base_.layoutTag};
        }
        header_->nodeCursor.store(0, std::memory_order_relaxed);
        new (&header_->releaseEvent) BufferReleaseEvent;
        releaseEvent_ = &header_->releaseEvent;
        for (size_t i = 0; i < nHashTableBucket; i++) {
            header_->buckets[i] = invalidIndex;
            header_->bucketLocks[i].Init();
        }
        for (size_t i = 0; i < nNode_; i++) {
            header_->nodeLocks[i].Init();
            meta_[i].Init();
            accessed_[i].store(0, std::memory_order_relaxed);
            if (base_.layerCount) {
                new (FillAt(i)) BlockFillState;
                for (size_t layer = 0; layer < base_.layerCount; ++layer) {
                    new (LayerAt(i, layer)) std::atomic<uint8_t>{0};
                }
            }
        }
        header_->magic.store(ExpectedMagic(), std::memory_order_release);
        return Status::OK();
    }
    Status LoadShmBuffer(PosixShm& shmFile)
    {
        auto s = shmFile.ShmOpen(PosixShm::OpenFlag::READ_WRITE);
        if (s.Failure()) {
            UC_ERROR("Failed({}) to open file({}).", s, shmFile.ShmName());
            return s;
        }
        s = MmapShmFile(shmFile, totalSize_, addrress_, false);
        if (s.Failure()) [[unlikely]] { return s; }
        header_ = static_cast<BufferHeader*>(addrress_);
        s = WaitShmHeaderReady(header_);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Shm file({}) not ready.", shmFile.ShmName());
            return s;
        }
        if (header_->nNode != nNode_) { return Status::InvalidParam("shared L1 node count mismatch"); }
        if (base_.layerCount) {
            const auto& layout = *BlockLayout();
            if (layout.nodeSize != base_.nodeSize || layout.layerCount != base_.layerCount ||
                layout.reservedNumber != base_.reservedNumber || layout.layoutTag != base_.layoutTag) {
                return Status::InvalidParam("shared block L1 layout mismatch");
            }
        }
        releaseEvent_ = &header_->releaseEvent;
        meta_ = (BufferMetaNode*)(static_cast<std::byte*>(addrress_) + MetaOffset());
        accessed_ = reinterpret_cast<std::atomic<uint8_t>*>(static_cast<std::byte*>(addrress_) +
                                                            AccessedOffset());
        return Status::OK();
    }
    Status RegisterBuffer(int32_t deviceId)
    {
        data_ = static_cast<std::byte*>(addrress_) + DataOffset();
        Trans::Device device;
        auto s = device.Setup(deviceId);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to setup device({}).", s, deviceId);
            return s;
        }
        const auto dataSize = DataSize();
        s = Trans::Buffer::RegisterHostBuffer((void*)data_, dataSize, (void**)&dataOnDevice_);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to register buffer({}) to device({}).", s, dataSize, deviceId);
            return s;
        }
        return Status::OK();
    }

public:
    SharedBufferStrategy(const std::string& uuid, int32_t deviceId, size_t nodeSize,
                         size_t totalSize, size_t reservedNumber, size_t layers = 0, uint64_t layoutTag = 0)
        : BufferStrategy(deviceId, nodeSize, totalSize, reservedNumber, layers, layoutTag), uuid_(uuid)
    {
    }
    ~SharedBufferStrategy() override
    {
        if (data_) { Trans::Buffer::UnregisterHostBuffer(data_); }
        if (addrress_) { PosixShm::MUnmap(addrress_, totalSize_); }
        if (!base_.layerCount || ownsShm_) { PosixShm{shmName_}.ShmUnlink(); }
    }
    Status Setup() override
    {
        const auto& uuid = uuid_;
        const auto deviceId = base_.deviceId;
        const auto nodeSize = base_.nodeSize;
        const auto totalSize = base_.totalSize;
        shmName_ = ShmPrefix() + uuid;
        nodeSize_ = nodeSize;
        nNode_ = totalSize / nodeSize;
        // Whole-block domains clean up only the backing they created.
        if (!base_.layerCount) { CleanUpShmFileExceptMe(shmName_); }
        PosixShm shmFile{shmName_};
        const auto dataOffset = DataOffset();
        totalSize_ = dataOffset + DataSize();
        const auto flags =
            PosixShm::OpenFlag::CREATE | PosixShm::OpenFlag::EXCL | PosixShm::OpenFlag::READ_WRITE;
        auto s = shmFile.ShmOpen(flags);
        if (s.Success()) {
            ownsShm_ = true;
            s = InitShmBuffer(shmFile);
        } else if (s == Status::DuplicateKey()) {
            s = LoadShmBuffer(shmFile);
        } else {
            UC_ERROR("Failed({}) to open file({}) with flags({}).", s, shmName_, flags);
            return s;
        }
        if (s.Failure()) { return s; }
        return RegisterBuffer(deviceId);
    }
    void BucketLock(size_t iBucket) override { header_->bucketLocks[iBucket].Lock(); }
    bool BucketTryLock(size_t iBucket) override { return header_->bucketLocks[iBucket].TryLock(); }
    void BucketUnlock(size_t iBucket) override { header_->bucketLocks[iBucket].Unlock(); }
    void NodeLock(size_t iNode) override { header_->nodeLocks[iNode].Lock(); }
    void NodeUnlock(size_t iNode) override { header_->nodeLocks[iNode].Unlock(); }
    size_t& FirstAt(size_t iBucket) override { return header_->buckets[iBucket]; }
    size_t NodeCount() const override { return header_->nNode; }
    size_t FetchNode(bool allowReserved) override
    {
        const auto total = header_->nNode - (allowReserved ? 0 : base_.reservedNumber);
        for (size_t i = 0; i < 2 * total; ++i) {
            auto cur = header_->nodeCursor.fetch_add(1, std::memory_order_relaxed) % total;
            uint8_t expected = 1;
            if (accessed_[cur].compare_exchange_strong(expected, 0, std::memory_order_relaxed,
                                                       std::memory_order_relaxed)) {
                continue;
            }
            return cur;
        }
        return header_->nodeCursor.fetch_add(1, std::memory_order_relaxed) % total;
    }
    void MarkAccessed(size_t iNode) override
    {
        accessed_[iNode].store(1, std::memory_order_relaxed);
    }
    void* DataAt(size_t iNode) override { return data_ + nodeSize_ * iNode; }
    void* DeviceDataAt(size_t iNode) override { return dataOnDevice_ + nodeSize_ * iNode; }
    BufferMetaNode* MetaAt(size_t iNode) override { return meta_ + iNode; }
    BlockFillState* FillAt(size_t iNode) override
    { return reinterpret_cast<BlockFillState*>(static_cast<std::byte*>(addrress_) + FillOffset()) + iNode; }
    std::atomic<uint8_t>* LayerAt(size_t iNode, size_t layer) override
    { return reinterpret_cast<std::atomic<uint8_t>*>(static_cast<std::byte*>(addrress_) + LayerOffset())
          + iNode * base_.layerCount + layer; }
};

class SharedBufferWatcherStrategy : public SharedBufferStrategy {
public:
    explicit SharedBufferWatcherStrategy(const std::string& uuid, bool wholeBlock = false)
        : SharedBufferStrategy(uuid, -1, 0, 0, 0, wholeBlock ? 1 : 0)
    {
    }
    Status Setup() override
    {
        shmName_ = ShmPrefix() + uuid_;
        // Whole-block domains clean up only the backing they created.
        if (!base_.layerCount) { CleanUpShmFileExceptMe(shmName_); }
        PosixShm shmFile{shmName_};
        auto s = shmFile.ShmOpen(PosixShm::OpenFlag::READ_WRITE);
        if (s.Failure()) {
            UC_ERROR("Failed({}) to open file({}).", s, shmFile.ShmName());
            return s;
        }
        void* addr = nullptr;
        auto size = sizeof(BufferHeader);
        s = MmapShmFile(shmFile, size, addr, false);
        if (s.Failure()) [[unlikely]] { return s; }
        auto header = static_cast<BufferHeader*>(addr);
        s = WaitShmHeaderReady(header);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Shm file({}) not ready.", shmFile.ShmName());
            return s;
        }
        nNode_ = header->nNode;
        shmFile.MUnmap(addr, size);
        if (base_.layerCount) {
            const auto prefixSize = FillOffset();
            s = MmapShmFile(shmFile, prefixSize, addrress_, false);
            if (s.Failure()) { return s; }
            const auto layout = *BlockLayout();
            shmFile.MUnmap(addrress_, prefixSize);
            addrress_ = nullptr;
            if (!layout.layerCount || !layout.nodeSize || layout.reservedNumber > nNode_) {
                return Status::InvalidParam("invalid shared block L1 layout");
            }
            base_.nodeSize = nodeSize_ = layout.nodeSize;
            base_.layerCount = layout.layerCount;
            base_.reservedNumber = layout.reservedNumber;
            base_.layoutTag = layout.layoutTag;
        }
        totalSize_ = DataOffset();
        s = MmapShmFile(shmFile, totalSize_, addrress_, false);
        if (s.Failure()) [[unlikely]] { return s; }
        header_ = static_cast<BufferHeader*>(addrress_);
        releaseEvent_ = &header_->releaseEvent;
        meta_ = (BufferMetaNode*)(static_cast<std::byte*>(addrress_) + MetaOffset());
        accessed_ = reinterpret_cast<std::atomic<uint8_t>*>(static_cast<std::byte*>(addrress_) +
                                                            AccessedOffset());
        return Status::OK();
    }
    void* DataAt(size_t iNode) override { return nullptr; }
    void* DeviceDataAt(size_t iNode) override { return nullptr; }
    void MarkAccessed(size_t /*iNode*/) override {}
};

Status TransBuffer::Setup(const Config& config)
{
    bypassHitOnLoad_ = config.cacheLoadBackendOnly;
    wholeBlock_ = config.cacheFullBlockL1;
    size_t nodeBytes = config.shardSize;
    uint64_t layoutTag = 1469598103934665603ULL;
    if (wholeBlock_ && config.deviceId >= 0) {
        if (!config.storeBackend || !config.shardSize || config.blockSize % config.shardSize) {
            return Status::InvalidParam("invalid block L1 layout");
        }
        const auto layers = config.blockSize / config.shardSize;
        size_t offset = 0;
        for (size_t layer = 0; layer < layers; ++layer) {
            auto bytes = config.storeBackend->PackedLayerBytes(layer);
            if (!bytes) { bytes = config.shardSize; }
            if (bytes > config.shardSize) { return Status::InvalidParam("invalid block L1 layer size"); }
            layerOffsets_.push_back(offset);
            offset += bytes;
            layoutTag = (layoutTag ^ bytes) * 1099511628211ULL;
        }
        nodeBytes = config.storeBackend->FullBlockReadBytes();
        if (!nodeBytes) { nodeBytes = config.blockSize; }
        if (nodeBytes != offset || layers == 0) { return Status::InvalidParam("block L1 size mismatch"); }
    }
    try {
        if (!config.shareBufferEnable) {
            strategy_ = std::make_shared<LocalBufferStrategy>(
                config.deviceId, nodeBytes, config.bufferCapacity,
                config.loadExclusiveBufferNumber, config.ioDirect, config.cacheSdmaDirect,
                layerOffsets_.size(), layoutTag);
        } else if (config.deviceId >= 0) {
            strategy_ = std::make_shared<SharedBufferStrategy>(
                config.uniqueId, config.deviceId, nodeBytes, config.bufferCapacity,
                config.loadExclusiveBufferNumber, layerOffsets_.size(), layoutTag);
        } else {
            strategy_ = std::make_shared<SharedBufferWatcherStrategy>(config.uniqueId, wholeBlock_);
        }
    } catch (const std::exception& e) {
        return Status::Error(fmt::format("failed({}) to make buffer strategy", e.what()));
    }
    return strategy_->Setup();
}

TransBuffer::Handle TransBuffer::MakeHandle(Index pos, size_t shard, bool owner, bool load, bool whole)
{
    Handle handle{this, pos, owner};
    if (!wholeBlock_ || pos == npos) { return handle; }
    // Callers hold the block bucket across lookup/allocation and this claim.
    strategy_->NodeLock(pos);
    const auto state = strategy_->MetaAt(pos)->state.load(std::memory_order_acquire);
    auto& fill = *strategy_->FillAt(pos);
    bool unavailable = false;
    if (state == State::LOADING) {
        if (whole || load) {
            if (!whole && strategy_->LayerAt(pos, shard)->load(std::memory_order_acquire) == 2) {
                handle.owner_ = false;
            } else if (fill.mode == BlockFillState::Mode::Empty || owner) {
                ResetBlockState(pos);
                fill.mode = BlockFillState::Mode::FullRead;
                handle.owner_ = true;
            } else {
                handle.owner_ = false;
                // Do not pin an incomplete dump as a whole-block follower: if
                // it stops between layers that reference would prevent a safe
                // full read forever. Retry after existing writers/views drain.
                unavailable = fill.mode == BlockFillState::Mode::Layers &&
                    (whole || strategy_->LayerAt(pos, shard)->load(std::memory_order_relaxed) == 0);
            }
        } else {
            auto& layer = *strategy_->LayerAt(pos, shard);
            if (fill.mode != BlockFillState::Mode::FullRead &&
                (layer.load(std::memory_order_relaxed) == 0 || owner)) {
                fill.mode = BlockFillState::Mode::Layers;
                if (layer.load(std::memory_order_relaxed) != 2) {
                    layer.store(1, std::memory_order_relaxed);
                }
                handle.owner_ = true;
            } else {
                handle.owner_ = false;
            }
        }
    } else if (state == State::FAILED) {
        handle.owner_ = false;  // reset/retry is legal only after all references drain
    }
    handle.view_ = whole ? npos : shard;
    strategy_->NodeUnlock(pos);
    if (unavailable) { return {}; }  // handle releases the tentative native reference
    return handle;
}

void TransBuffer::ResetBlockState(Index pos)
{
    if (!wholeBlock_) { return; }
    *strategy_->FillAt(pos) = BlockFillState{};
    for (size_t layer = 0; layer < layerOffsets_.size(); ++layer) {
        strategy_->LayerAt(pos, layer)->store(0, std::memory_order_relaxed);
    }
}

TransBuffer::Handle TransBuffer::Get(const Detail::BlockId& blockId, size_t shardIdx,
                                     bool allowReserved, bool isLoad)
{
    if (wholeBlock_ && shardIdx >= LayerCount()) { return {}; }
    const auto key = wholeBlock_ ? 0 : shardIdx;
    auto iBucket = Hash(blockId, key);
    // Native dispatch expects Get to return a live handle. Early's Acquire and
    // TryGet remain nonblocking, while a missing layer of a partial dump waits
    // here without retaining a reference that could prevent its own reload.
    for (;;) {
        const auto epoch = strategy_->releaseEvent_->epoch.load(std::memory_order_seq_cst);
        bool owner = false;
        strategy_->BucketLock(iBucket);
        auto iNode = FindAt(iBucket, blockId, key, owner);
        if (iNode == invalidIndex) {
            iNode = Alloc(blockId, key, iBucket, allowReserved);
            owner = true;
        } else if (bypassHitOnLoad_ && isLoad && owner && Ready(iNode)) {
            MarkNotReady(iNode);
        }
        auto handle = MakeHandle(iNode, shardIdx, owner, isLoad);
        strategy_->BucketUnlock(iBucket);
        if (handle || !wholeBlock_) { return handle; }
        // A layer may become READY before its producer drops its reference.
        strategy_->releaseEvent_->Wait(epoch, 1);
    }
}

TransBuffer::Handle TransBuffer::TryGet(const Detail::BlockId& blockId, size_t shardIdx)
{
    return TryGetImpl(blockId, shardIdx, false);
}

TransBuffer::Handle TransBuffer::TryGetForPrefetch(const Detail::BlockId& blockId,
                                                size_t shardIdx, bool hasLeasedL2Copy)
{
    return TryGetImpl(blockId, shardIdx, hasLeasedL2Copy);
}

TransBuffer::Handle TransBuffer::TryGetImpl(const Detail::BlockId& blockId, size_t shardIdx,
                                          bool reloadReserved)
{
    if (wholeBlock_ && shardIdx >= LayerCount()) { return {}; }
    // Early's whole-block reservation uses one native key/pin per block.
    const auto key = wholeBlock_ ? 0 : shardIdx;
    auto bucket = Hash(blockId, key);
    bool owner = false;
    strategy_->BucketLock(bucket);
    auto node = FindAt(bucket, blockId, key, owner);
    if (node != invalidIndex && !strategy_->CanReserve(node)) {
        if (reloadReserved) {
            node = ReplaceReservedForPrefetch(node, bucket);
        } else {
            Release(node);
            node = invalidIndex;
        }
        if (node == invalidIndex) {
            strategy_->BucketUnlock(bucket);
            return {};
        }
        owner = true;  // The caller must load the new ordinary destination.
    }
    if (node == invalidIndex && strategy_->CanReserve(0)) {
        node = Alloc(blockId, key, bucket, false, 2 * strategy_->NodeCount());
        owner = true;
    }
    auto handle = MakeHandle(node, shardIdx, owner, true, wholeBlock_);
    strategy_->BucketUnlock(bucket);
    return handle;
}

TransBuffer::Index TransBuffer::ReplaceReservedForPrefetch(Index source, size_t bucket)
{
    // FindAt already acquired our private reference. The bucket serializes
    // identity lookups; the reference prevents concurrent allocator reuse.
    auto* src = strategy_->MetaAt(source);
    strategy_->NodeLock(source);
    const bool eligible = src->hash == bucket && src->reference == 1 &&
                          src->state.load(std::memory_order_acquire) == State::READY &&
                          !strategy_->CanReserve(source) && strategy_->CanReserve(0);
    strategy_->NodeUnlock(source);
    if (!eligible) {
        Release(source);
        return invalidIndex;
    }
    // Acquire capacity before discarding X. Alloc's old-bucket try-lock and
    // finite scan bound are unchanged. Do not hold the source node lock here:
    // hash-chain maintenance may lock it as the destination's neighbour.
    const auto destination = Alloc(src->block, src->shard, bucket, false,
                                   2 * strategy_->NodeCount());
    if (destination == invalidIndex) {
        Release(source);
        return invalidIndex;
    }
    // The ordinary destination is already pinned and LOADING. Both temporary
    // mappings are hidden behind this bucket lock. No data is copied or READY
    // published here: the caller's leased L2 read must do that after completion.
    strategy_->NodeLock(source);
    Remove(bucket, source);
    src->state.store(State::LOADING, std::memory_order_release);
    src->errorCode.store(Status::OK().Underlying(), std::memory_order_relaxed);
    ResetBlockState(source);
    strategy_->NodeUnlock(source);
    Release(source);
    return destination;
}

void TransBuffer::Prealloc(const Detail::BlockId& blockId, size_t shardIdx, bool allowReserved)
{
    const auto key = wholeBlock_ ? 0 : shardIdx;
    auto iBucket = Hash(blockId, key);
    strategy_->BucketLock(iBucket);
    if (!ExistAt(iBucket, blockId, key)) {
        size_t pos = Alloc(blockId, key, iBucket, allowReserved);
        Release(pos);
    }
    strategy_->BucketUnlock(iBucket);
}

bool TransBuffer::Exist(const Detail::BlockId& blockId, size_t shardIdx)
{
    const auto key = wholeBlock_ ? 0 : shardIdx;
    auto iBucket = Hash(blockId, key);
    strategy_->BucketLock(iBucket);
    auto exist = ExistAt(iBucket, blockId, key);
    if (exist && wholeBlock_) {
        exist = false;
        for (auto node = strategy_->FirstAt(iBucket); node != invalidIndex;) {
            const auto* meta = strategy_->MetaAt(node);
            if (meta->block == blockId && meta->shard == key) {
                exist = meta->state.load(std::memory_order_acquire) == State::READY;
                break;
            }
            node = meta->next;
        }
    }
    strategy_->BucketUnlock(iBucket);
    return exist;
}

bool TransBuffer::ExistAt(size_t iBucket, const Detail::BlockId& blockId, size_t shardIdx)
{
    auto iNode = strategy_->FirstAt(iBucket);
    while (iNode != invalidIndex) {
        auto meta = strategy_->MetaAt(iNode);
        if (meta->block == blockId && meta->shard == shardIdx) { return true; }
        iNode = meta->next;
    }
    return false;
}

size_t TransBuffer::FindAt(size_t iBucket, const Detail::BlockId& blockId, size_t shardIdx,
                           bool& owner)
{
    auto iNode = strategy_->FirstAt(iBucket);
    while (iNode != invalidIndex) {
        auto meta = strategy_->MetaAt(iNode);
        if (meta->block == blockId && meta->shard == shardIdx) {
            strategy_->NodeLock(iNode);
            owner = meta->reference == 0;
            if (owner && meta->state.load(std::memory_order_relaxed) == State::FAILED) {
                meta->state.store(State::LOADING, std::memory_order_relaxed);
                meta->errorCode.store(Status::OK().Underlying(), std::memory_order_relaxed);
                ResetBlockState(iNode);
            }
            ++meta->reference;
            strategy_->MarkAccessed(iNode);
            strategy_->NodeUnlock(iNode);
            break;
        }
        iNode = meta->next;
    }
    return iNode;
}

size_t TransBuffer::Alloc(const Detail::BlockId& blockId, size_t shardIdx, size_t iBucket,
                          bool allowReserved, std::optional<size_t> maxAttempts)
{
    for (size_t attempts = 0; !maxAttempts || attempts < *maxAttempts; ++attempts) {
        auto iNode = strategy_->FetchNode(allowReserved);
        auto meta = strategy_->MetaAt(iNode);
        strategy_->NodeLock(iNode);
        if (meta->reference > 0) {
            strategy_->NodeUnlock(iNode);
            continue;
        }
        const auto oldBucket = meta->hash;
        if (oldBucket != iBucket) {
            if (oldBucket != invalidIndex) {
                if (!strategy_->BucketTryLock(oldBucket)) {
                    strategy_->NodeUnlock(iNode);
                    continue;
                }
                Remove(oldBucket, iNode);
                strategy_->BucketUnlock(oldBucket);
            }
            MoveTo(iBucket, iNode);
        }
        ++meta->reference;
        strategy_->MarkAccessed(iNode);
        meta->block = blockId;
        meta->shard = shardIdx;
        meta->state.store(State::LOADING, std::memory_order_relaxed);
        meta->errorCode.store(Status::OK().Underlying(), std::memory_order_relaxed);
        ResetBlockState(iNode);
        strategy_->NodeUnlock(iNode);
        return iNode;
    }
    return invalidIndex;
}

void TransBuffer::MoveTo(size_t iBucket, size_t iNode)
{
    auto meta = strategy_->MetaAt(iNode);
    auto& head = strategy_->FirstAt(iBucket);
    auto n = head;
    meta->next = n;
    if (n != invalidIndex) {
        auto next = strategy_->MetaAt(n);
        strategy_->NodeLock(n);
        next->prev = iNode;
        strategy_->NodeUnlock(n);
    }
    meta->hash = iBucket;
    head = iNode;
}

void TransBuffer::Remove(size_t iBucket, size_t iNode)
{
    auto meta = strategy_->MetaAt(iNode);
    auto p = meta->prev;
    if (p != invalidIndex) {
        auto prev = strategy_->MetaAt(p);
        strategy_->NodeLock(p);
        prev->next = meta->next;
        strategy_->NodeUnlock(p);
    }
    auto n = meta->next;
    if (n != invalidIndex) {
        auto next = strategy_->MetaAt(n);
        strategy_->NodeLock(n);
        next->prev = meta->prev;
        strategy_->NodeUnlock(n);
    }
    if (strategy_->FirstAt(iBucket) == iNode) { strategy_->FirstAt(iBucket) = n; }
    meta->prev = meta->next = invalidIndex;
    meta->hash = invalidIndex;
}

void* TransBuffer::DataAt(Index pos, size_t view)
{
    auto* data = static_cast<std::byte*>(strategy_->DataAt(pos));
    return data && wholeBlock_ && view != npos ? data + layerOffsets_[view] : data;
}

void* TransBuffer::DeviceDataAt(Index pos, size_t view)
{
    auto* data = static_cast<std::byte*>(strategy_->DeviceDataAt(pos));
    return data && wholeBlock_ && view != npos ? data + layerOffsets_[view] : data;
}

void TransBuffer::Acquire(Index pos)
{
    strategy_->NodeLock(pos);
    ++strategy_->MetaAt(pos)->reference;
    strategy_->NodeUnlock(pos);
}

void TransBuffer::Release(Index pos)
{
    strategy_->NodeLock(pos);
    const bool released = --strategy_->MetaAt(pos)->reference == 0;
    strategy_->NodeUnlock(pos);
    if (released) { strategy_->releaseEvent_->Notify(); }
}

bool TransBuffer::Ready(Index pos) { return GetState(pos) == State::READY; }

TransBuffer::State TransBuffer::GetState(Index pos, size_t view)
{
    const auto state = strategy_->MetaAt(pos)->state.load(std::memory_order_acquire);
    if (wholeBlock_ && view != npos && state == State::LOADING &&
        strategy_->LayerAt(pos, view)->load(std::memory_order_acquire) == 2) {
        return State::READY;
    }
    return state;
}

Status TransBuffer::FailureStatus(Index pos)
{
    auto errorCode = strategy_->MetaAt(pos)->errorCode.load(std::memory_order_acquire);
    if (errorCode == Status::OK().Underlying()) {
        return Status::Error("shared buffer failed without an error status");
    }
    return Status{errorCode, {}};
}

void TransBuffer::MarkReady(Index pos, size_t view)
{
    if (!wholeBlock_) {
        strategy_->MetaAt(pos)->state.store(State::READY, std::memory_order_release);
        return;
    }
    strategy_->NodeLock(pos);
    auto& state = strategy_->MetaAt(pos)->state;
    if (state.load(std::memory_order_relaxed) != State::FAILED) {
        if (view == npos) {
            state.store(State::READY, std::memory_order_release);
        } else {
            auto& layer = *strategy_->LayerAt(pos, view);
            if (layer.exchange(2, std::memory_order_release) != 2) {
                ++strategy_->FillAt(pos)->readyLayers;
            }
            if (strategy_->FillAt(pos)->readyLayers == LayerCount()) {
                state.store(State::READY, std::memory_order_release);
            }
        }
    }
    strategy_->NodeUnlock(pos);
}

void TransBuffer::MarkFailed(Index pos, const Status& status)
{
    if (wholeBlock_) { strategy_->NodeLock(pos); }
    auto meta = strategy_->MetaAt(pos);
    meta->errorCode.store(status.Underlying(), std::memory_order_release);
    meta->state.store(State::FAILED, std::memory_order_release);
    if (wholeBlock_) { strategy_->NodeUnlock(pos); }
}

void TransBuffer::MarkNotReady(Index pos)
{
    auto meta = strategy_->MetaAt(pos);
    meta->errorCode.store(Status::OK().Underlying(), std::memory_order_release);
    meta->state.store(State::LOADING, std::memory_order_release);
}

}  // namespace UC::CacheStore

#include "early_buffer_pins_impl.h"
