#pragma once
#include "array_slice.hpp"
#include "collective_functions.hpp"
#include "graph.hpp"
#include "perf_counter.hpp"
#include "relaxed_vector.hpp"
#include <atomic>

#if !defined(DELTASTEPPING)
#error Do not include this file in !DELTASTEPPING builds
#endif

namespace sssp {

class delta_stepping {
  public:
    delta_stepping(double delta) : m_delta(delta) {}
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
    struct request {
        request() = default;
        request(size_t node, double distance, size_t predecessor)
            : node(node), distance(distance), predecessor(predecessor) {}
        size_t node;
        double distance;
        size_t predecessor;
    };

    double m_delta;
    std::atomic<bool> m_done;
    std::atomic<bool> m_inner_done;
    carray<relaxed_vector<request>> m_requests;
    carray<perf_counter> m_perf;
};

} // namespace sssp