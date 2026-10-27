#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <sys/syscall.h>
#include "posix/cc/posix_store.cc"
using namespace UC;
using namespace UC::PosixStore;
static void Require(bool value, const char* why) { if (!value) throw std::runtime_error(why); }
static Detail::BlockId Id(unsigned value) { Detail::BlockId id{}; id[0]=std::byte(value); return id; }
template<class Predicate> static void Until(Predicate ready, const char* why) {
    auto end=std::chrono::steady_clock::now()+std::chrono::seconds(3);
    while (!ready()) { Require(std::chrono::steady_clock::now()<end,why); std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
}
struct Fixture {
    std::string path;
    Detail::Dictionary config;
    std::shared_ptr<void> data,readData;
    std::unique_ptr<EarlyPosixStore> store;
    explicit Fixture(size_t timeout=5000, size_t limit=1) {
        auto pattern = (std::filesystem::temp_directory_path() / "ucm-write-window-XXXXXX").string();
        auto* directory = mkdtemp(pattern.data());
        Require(directory != nullptr, "temporary directory");
        path = directory;
        for (auto* dst:{&data,&readData}) { void* p=nullptr; Require(posix_memalign(&p,4096,8192)==0,"aligned allocation"); dst->reset(p,std::free); memset(p,0x5a,8192); }
        config.SetNumber("device_id",0); config.Set("storage_backends",std::vector<std::string>{path});
        config.SetNumber("tensor_size",8192);config.SetNumber("shard_size",8192);config.SetNumber("block_size",8192);
        config.Set("posix_io_engine",std::string("aio"));config.Set("posix_gc_enable",false);
        config.SetNumber("posix_open_concurrency",2);config.SetNumber("posix_commit_concurrency",1);
        config.SetNumber("posix_write_inflight_limit",limit);config.SetNumber("timeout_ms",timeout);config.SetNumber("data_dir_shard_bytes",0);
        store=std::make_unique<EarlyPosixStore>();Require(store->Setup(config).Success(),"setup write window");
        auto t=store->Dump({{Id(1),0,{data.get()}}});Require(t && store->Wait(t.Value()).Success(),"seed dump");
        Until([&]{auto id=Id(1);return store->Lookup(&id,1).Value()[0];},"seed publication");
    }
    ~Fixture() { TestHooks::ClearAioHooks();TestHooks::ClearOpenHook();store.reset();std::filesystem::remove_all(path); }
};
static void ReadBypass(bool cancelQueued) {
    Fixture f(cancelQueued?100:5000);
    std::mutex mutex; aio_context_t heldCtx{}; iocb* held=nullptr;
    std::atomic<unsigned> writes{0};
    TestHooks::SetAioSubmitHook([&](aio_context_t ctx,int64_t nr,iocb** cb) {
        if (cb[0]->aio_lio_opcode==IOCB_CMD_PWRITE && ++writes==1) {
            std::lock_guard<std::mutex> lock(mutex);heldCtx=ctx;held=cb[0];return 1;
        }
        return static_cast<int32_t>(syscall(SYS_io_submit,ctx,nr,cb));
    });
    auto first=f.store->Dump({{Id(2),0,{f.data.get()}}});Require(bool(first),"first accepted");
    Until([&]{std::lock_guard<std::mutex> lock(mutex);return held!=nullptr;},"first write held");
    auto second=f.store->Dump({{Id(3),0,{f.data.get()}}});Require(bool(second),"second accepted queued");
    Detail::TaskDesc desc{{Id(1),0,{f.readData.get()}}};desc.fullBlockRead=true;
    auto read=f.store->Load(std::move(desc));Require(read && f.store->Wait(read.Value()).Success(),"read bypasses capped writes");
    Require(memcmp(f.data.get(),f.readData.get(),8192)==0,"read bytes preserved");
    Require(writes==1,"permit retained until physical write completion");
    Require(!f.store->Check(second.Value()).Value(),"queued write not falsely complete");
    if (cancelQueued) {
        Require(f.store->Wait(second.Value())==Status::Timeout(),"queued cancellation drains without permit release");
        Require(writes==1,"cancelled queued write never submitted");
    }
    { std::lock_guard<std::mutex> lock(mutex);Require(syscall(SYS_io_submit,heldCtx,1,&held)==1,"release real first write"); }
    Until([&]{return f.store->Check(first.Value()).Value();},"first physical callback observed");
    Require(f.store->Wait(first.Value()).Success(),"first physically complete");
    if (!cancelQueued) {
        Require(f.store->Wait(second.Value()).Success(),"completion admits queued write");Require(writes==2,"two writes exactly once");
    }
    auto third=f.store->Dump({{Id(4),0,{f.data.get()}}});Require(third && f.store->Wait(third.Value()).Success(),"write quota reusable after drain");
}
static void FailureReturnsPermit(unsigned mode) {
    Fixture f;std::atomic<unsigned> calls{0};
    if (mode==0) {
        TestHooks::SetOpenHook([&](const std::string& path,int32_t flags,mode_t permissions) {
            if ((flags&O_WRONLY) && ++calls==1) {errno=EIO;return -1;}
            return ::open(path.c_str(),flags,permissions);
        });
    } else {
        TestHooks::SetAioSubmitHook([&](aio_context_t ctx,int64_t nr,iocb** cb) {
            if (cb[0]->aio_lio_opcode==IOCB_CMD_PWRITE && ++calls==1) {
                if (mode==1) {errno=EIO;return -1;}
                cb[0]->aio_nbytes=4096; // real short physical write, expected 8192
            }
            return static_cast<int32_t>(syscall(SYS_io_submit,ctx,nr,cb));
        });
    }
    auto bad=f.store->Dump({{Id(2),0,{f.data.get()}}});auto good=f.store->Dump({{Id(3),0,{f.data.get()}}});
    Require(bad && good,"failure tasks accepted");Require(f.store->Wait(bad.Value()).Failure(),"failed write never succeeds");
    Require(f.store->Wait(good.Value()).Success(),"failed write returns admission permit");
    Require(calls==2,"exactly one attempt for each write");
}
static void OrderAndFairness(size_t limit) {
    Fixture f;Config config;config.storageBackends={f.path};config.dataDirShardBytes=0;
    SpaceLayout layout;Require(layout.Setup(config).Success(),"operator layout");
    BlockOperator op;op.Setup(&layout,1,1,limit);
    std::mutex mutex;std::vector<int> order;std::list<BlockOperator::OpenTask> tasks;
    // Two writes precede 48 reads in enqueue order. Enabled mode must bypass
    // them, but service each write within 16 read dequeues when quota is free.
    for (int id=0;id<50;++id) {
        BlockOperator::OpenTask task;task.id=Id(100+id);task.activated=id<2;
        task.flags=O_CREAT|O_WRONLY;task.callback=[&,id](BlockOperator::OpenResult result) {
            if (result.fd>=0) ::close(result.fd);
            std::lock_guard<std::mutex> lock(mutex);order.push_back(id);
        };tasks.push_back(std::move(task));
    }
    op.Submit(std::move(tasks));Until([&]{std::lock_guard<std::mutex> lock(mutex);return order.size()==50;},"all queue items drained");
    std::lock_guard<std::mutex> lock(mutex);
    if (limit==0) { for(int i=0;i<50;++i)Require(order[i]==i,"disabled preserves FIFO"); }
    else { Require(order[16]==0 && order[33]==1,"eligible writes served after 16 reads");Require(order[0]==2,"reads preferred"); }
}
static void ProductionWindow() {
    Fixture f(5000,16);
    std::mutex mutex;std::vector<std::pair<aio_context_t,iocb*>> held;
    std::atomic<bool> hold{true};std::atomic<unsigned> active{0},peak{0},submitted{0};
    TestHooks::SetAioSubmitHook([&](aio_context_t ctx,int64_t nr,iocb** cb) {
        if (cb[0]->aio_lio_opcode==IOCB_CMD_PWRITE) {
            auto count=++active;auto old=peak.load();while(old<count&&!peak.compare_exchange_weak(old,count)){}
            ++submitted;
            auto* callback=reinterpret_cast<AioImpl::Callback*>(cb[0]->aio_data);
            *callback=[original=std::move(*callback),&active](AioImpl::Result result) { original(result);--active; };
            if (hold) {std::lock_guard<std::mutex> lock(mutex);held.emplace_back(ctx,cb[0]);return 1;}
        }
        return static_cast<int32_t>(syscall(SYS_io_submit,ctx,nr,cb));
    });
    std::vector<Detail::TaskHandle> tasks;
    for(unsigned i=0;i<64;++i) {auto t=f.store->Dump({{Id(10+i),0,{f.data.get()}}});Require(bool(t),"window task accepted");tasks.push_back(t.Value());}
    Until([&]{std::lock_guard<std::mutex> lock(mutex);return held.size()==16;},"production window filled");
    auto read=f.store->Load({{Id(1),0,{f.readData.get()}}});Require(read && f.store->Wait(read.Value()).Success(),"read bypasses full 16-write window");
    Require(active==16 && submitted==16 && peak==16,"no premature permit release at configured cap");
    hold=false;
    {std::lock_guard<std::mutex> lock(mutex);for(auto [ctx,cb]:held)Require(syscall(SYS_io_submit,ctx,1,&cb)==1,"release held window");}
    for(auto t:tasks)Require(f.store->Wait(t).Success(),"all writes drain");
    Until([&]{return active==0;},"all write callbacks finished");
    Require(submitted==64 && peak==16,"cap retained through refill and drain");
}
static void InvalidConfig() {
    Fixture f;
    for (int64_t value:{-1LL,4097LL}) {
        auto config=f.config;config.SetNumber("posix_write_inflight_limit",value);
        EarlyPosixStore store;Require(store.Setup(config).Failure(),"invalid write limit rejected");
    }
    auto config=f.config;config.Set("posix_io_engine",std::string("psync"));
    EarlyPosixStore store;Require(store.Setup(config).Failure(),"psync cannot silently ignore write limit");
}
int main() {
    ReadBypass(false);ReadBypass(true);ProductionWindow();
    for(unsigned i=0;i<3;++i)FailureReturnsPermit(i);
    OrderAndFairness(0);OrderAndFairness(1);InvalidConfig();
    std::cout<<"write window: real AIO bypass/drain, queued cancellation, open/submit/short-write failure, fairness, legacy FIFO and config PASS\n";
}
