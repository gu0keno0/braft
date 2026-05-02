// Reproduces a push_rq deadlock during step_down when many closures are
// pending in ClosureQueue. Submit rate >> FSM rate makes closures pile up;
// an AppendEntries with a higher term then triggers step_down, which calls
// push_rq while holding NodeImpl._mutex. Workers blocked on _mutex cannot
// drain _rq, so push_rq spins forever and the RPC never returns.

#include <gtest/gtest.h>
#include <atomic>
#include <thread>
#include <unistd.h>

#include <braft/raft.h>
#include <braft/util.h>
#include <brpc/channel.h>
#include <brpc/controller.h>
#include <brpc/server.h>
#include <bthread/bthread.h>
#include <butil/endpoint.h>
#include <butil/iobuf.h>
#include "braft/raft.pb.h"

// Slow FSM: creates FSM backpressure by sleeping per entry.
// This makes closures accumulate in ClosureQueue faster than they commit.
class SlowFSM : public braft::StateMachine {
public:
    std::atomic<bool> is_leader{false};
    void on_apply(braft::Iterator& iter) override {
        for (; iter.valid(); iter.next()) {
            ::usleep(500);  // 0.5ms/entry → 2000 entries/sec FSM throughput
        }
    }
    void on_leader_start(int64_t) override  { is_leader = true; }
    void on_leader_stop(const butil::Status&) override { is_leader = false; }
    void on_shutdown() override {}
};

class NopClosure : public braft::Closure {
public:
    void Run() override { delete this; }
};

struct ApplyArg {
    braft::Node* node;
    std::atomic<bool>* stop;
};

static void* apply_fn(void* arg) {
    auto* a = static_cast<ApplyArg*>(arg);
    while (!a->stop->load()) {
        if (!a->node->is_leader()) { bthread_usleep(1000); continue; }
        butil::IOBuf data; data.append("x");
        braft::Task task;
        task.data = &data;
        task.done = new NopClosure;
        a->node->apply(task);
        bthread_usleep(100);
    }
    return nullptr;
}

class LeaderStepDownNoDeadlockTest : public testing::Test {
protected:
    void SetUp()    override { ::system("rm -rf /tmp/cqd_data"); }
    void TearDown() override { ::system("rm -rf /tmp/cqd_data"); }
};

TEST_F(LeaderStepDownNoDeadlockTest, StepDownWithPendingClosures) {
    const int kPort = 8300;
    const std::string self_addr = "127.0.0.1:" + std::to_string(kPort);

    brpc::Server server;
    ASSERT_EQ(0, braft::add_service(&server, kPort));
    ASSERT_EQ(0, server.Start(kPort, nullptr));

    butil::EndPoint addr;
    ASSERT_EQ(0, butil::str2endpoint(self_addr.c_str(), &addr));

    braft::NodeOptions opts;
    opts.election_timeout_ms              = 1000;
    opts.fsm                              = new SlowFSM;
    opts.node_owns_fsm                    = true;
    opts.snapshot_interval_s              = -1;
    ASSERT_EQ(0, opts.initial_conf.parse_from(self_addr + ":0"));
    opts.log_uri       = "local:///tmp/cqd_data/log";
    opts.raft_meta_uri = "local:///tmp/cqd_data/raft_meta";
    opts.snapshot_uri  = "local:///tmp/cqd_data/snapshot";

    braft::Node node("cqd_group", braft::PeerId(addr));
    ASSERT_EQ(0, node.init(opts));

    for (int i = 0; i < 30 && !node.is_leader(); ++i) ::usleep(200000);
    ASSERT_TRUE(node.is_leader()) << "did not become leader";

    bthread_setconcurrency(8);
    const int concurrency = bthread_getconcurrency();
    const int kSubmitters = 2 * concurrency;
    LOG(INFO) << "bthread concurrency=" << concurrency
              << " submitters=" << kSubmitters;

    std::atomic<bool> stop_flag{false};
    ApplyArg apply_arg{&node, &stop_flag};
    std::vector<bthread_t> bthreads;
    for (int i = 0; i < kSubmitters; ++i) {
        bthread_t t; bthread_start_background(&t, nullptr, apply_fn, &apply_arg);
        bthreads.push_back(t);
    }

    // Let closures accumulate: ~80k apply/s vs ~2k FSM/s → ~78k closures/s
    ::sleep(2);

    // Trigger step_down via AppendEntries with higher term
    brpc::Channel chan;
    brpc::ChannelOptions copts;
    copts.protocol = "baidu_std";
    copts.timeout_ms = 30000;
    copts.connect_timeout_ms = 5000;
    ASSERT_EQ(0, chan.Init(self_addr.c_str(), &copts));

    braft::AppendEntriesRequest req;
    req.set_group_id("cqd_group");
    req.set_server_id("127.0.0.1:9999:0:0");
    req.set_peer_id(self_addr + ":0:0");
    req.set_term(99999);
    req.set_prev_log_term(99998);
    req.set_prev_log_index(0);
    req.set_committed_index(0);

    braft::AppendEntriesResponse resp;
    brpc::Controller cntl;
    braft::RaftService_Stub stub(&chan);

    // If deadlocked: push_rq spins forever, RPC never returns, test FAILS.
    std::atomic<bool> rpc_done{false};
    std::thread rpc_thread([&]() {
        stub.append_entries(&cntl, &req, &resp, NULL);
        rpc_done = true;
    });

    for (int i = 0; i < 100 && !rpc_done; ++i) ::usleep(100000);  // wait 10s

    EXPECT_TRUE(rpc_done)
        << "DEADLOCK: step_down did not complete within 10s. "
           "push_rq is spinning with workers blocked on NodeImpl._mutex.";

    stop_flag = true;
    if (!rpc_done) {
        // Deadlocked. Don't wait on threads/node — they all touch _mutex
        // and would hang. Process exit will reap everything.
        rpc_thread.detach();
        return;
    }
    rpc_thread.join();
    node.shutdown(nullptr);
    node.join();
    for (auto t : bthreads) {
        bthread_join(t, nullptr);
    }
    server.Stop(0);
    server.Join();
}
