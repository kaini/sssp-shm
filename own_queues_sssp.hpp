#pragma once
#include "array_slice.hpp"
#include "carray.hpp"
#include "collective_functions.hpp"
#include "graph.hpp"
#include "thread_local_allocator.hpp"
#include <mutex>
#include <tbb/concurrent_vector.h>
#include <unordered_map>

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
                        size_t node_count,
                        array_slice<const array_slice<const edge>> thread_edges_by_node,
                        array_slice<double> out_thread_distances,
                        array_slice<size_t> out_thread_predecessors);

  private:
    struct relaxation {
        size_t node;
        size_t predecessor;
        double distance;
    };

    size_t m_node_count;
    carray<tbb::concurrent_vector<relaxation>> m_relaxations;
    carray<std::atomic<double>> m_seen_distances;

    std::atomic<double> m_time;
    std::atomic<double> m_init_time;

#if defined(CRAUSER) || defined(CRAUSERDYN)
    carray<std::atomic<double>> m_min_incoming;
    std::atomic<double> m_in_threshold;
    std::atomic<double> m_out_threshold;
#endif
};

} // namespace sssp