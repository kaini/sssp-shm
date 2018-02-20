#pragma once
#include "array_slice.hpp"
#include "carray.hpp"
#include "collective_functions.hpp"
#include "graph.hpp"
#include "thread_local_allocator.hpp"
#include <boost/heap/fibonacci_heap.hpp>
#include <boost/thread/barrier.hpp>
#include <mutex>
#include <queue>
#include <tbb/concurrent_priority_queue.h>

namespace sssp {

class shared_queue_sssp {
    struct node;

    struct queued_node {
        queued_node() : queued_node(0.0, nullptr) {}
        queued_node(double value, node* node) : value(value), node(node) {}
        double value;
        shared_queue_sssp::node* node;
        bool operator<(const queued_node& other) const {
            // Use > to create a minimum-heap
            return value > other.value;
        }
    };

    enum class state {
        unexplored,
        fringe,
        settled,
    };

    struct node {
        std::mutex mutex;
        shared_queue_sssp::state state = shared_queue_sssp::state::unexplored;
#if defined(CRAUSER)
        double min_incoming = INFINITY;
#endif
    };

  public:
    shared_queue_sssp(size_t node_count, array_slice<array_slice<const edge>> edges);

    // Runs the parallel SSSP. Has to be called by all participating
    // threads at the same time.
    void run_collective(thread_group& threads, int thread_rank, array_slice<result> out_result);

    double time() const { return m_time.load(std::memory_order_relaxed); }
    double init_time() const { return m_init_time.load(std::memory_order_relaxed); }

  private:
    size_t node_id(const node& node) const;

    // TODO: store result in node and provide an iterator over it

    size_t m_node_count;
    array_slice<array_slice<const edge>> m_edges;
    carray<node> m_nodes;
    tbb::concurrent_priority_queue<queued_node> m_distance_queue;
    std::atomic<bool> m_done;

    std::atomic<double> m_time;
    std::atomic<double> m_init_time;

#if defined(CRAUSER)
    tbb::concurrent_priority_queue<queued_node> m_crauser_in_queue;
    std::atomic<double> m_in_threshold;
#endif
};

} // namespace sssp
