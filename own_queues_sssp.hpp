#pragma once
#include "array_slice.hpp"
#include "carray.hpp"
#include "collective_functions.hpp"
#include "graph.hpp"
#include "perf_counter.hpp"
#include "relaxed_vector.hpp"
#include "thread_local_allocator.hpp"
#include <mutex>
#include <unordered_map>

#if !defined(BY_NODES)
#error Do not include this file in !BY_NODES builds
#endif

namespace sssp {

class own_queues_sssp {
  public:
    // Runs the algorithm.
    // node_count has to be the same on all threads and is the total node count.
    // thread_edges_by_node, out_thread_distances and out_thread_predecessors have
    // be as long as threads.chunk_size(thread_rank, node_count) and contain the information
    // about the nodes starting at threads.chunk_start(thread_rank, node_count).
    // The algorithm does not initialize out_thread_distances or out_thread_predecessors.
    void run_collective(thread_group& threads,
                        int thread_rank,
                        int group_count,
                        int group_rank,
                        thread_group& group_threads,
                        int group_thread_rank,
                        size_t node_count,
                        size_t edge_count,
                        array_slice<const array_slice<const edge>> thread_edges_by_node,
                        array_slice<double> out_thread_distances,
                        array_slice<size_t> out_thread_predecessors);

    array_slice<const perf_counter> perf() const { return m_perf; }

  private:
    struct relaxation {
        size_t node;
        size_t predecessor;
        double distance;
    };

    struct node_cost {
        size_t node;
        double cost;
    };

    size_t m_node_count;
    carray<relaxed_vector<relaxation>> m_relaxations;
    carray<carray<std::atomic<double>>> m_seen_distances; // by group
    carray<perf_counter> m_perf;

#if defined(CRAUSER_IN)
    carray<std::atomic<double>> m_min_incoming;
    std::atomic<double> m_in_threshold;
#endif

#if defined(CRAUSER_INDYN)
    carray<std::atomic<size_t>> m_incoming_at;
    carray<array_slice<edge>> m_incoming_edges;
    std::atomic<double> m_in_threshold;
#endif

#if defined(CRAUSER_OUT) || defined(CRAUSER_OUTDYN)
    std::atomic<double> m_out_threshold;
#endif

    std::atomic<double> m_init_time;
    std::atomic<double> m_local_relax_time;
    std::atomic<double> m_inbox_relax_time;
    std::atomic<double> m_crauser_dyn_time;
};

} // namespace sssp