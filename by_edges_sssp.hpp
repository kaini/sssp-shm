#pragma once
#include "array_slice.hpp"
#include "carray.hpp"
#include "collective_functions.hpp"
#include "graph.hpp"
#include "perf_counter.hpp"
#include <atomic>

#if !defined(BY_EDGES)
#error Do not define this file in !BY_EDGES builds
#endif

namespace sssp {

class by_edges_sssp {
  public:
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
    carray<std::atomic<double>> m_global_distances;
    carray<size_t> m_global_updated;
    std::atomic<size_t> m_global_updated_at;
    std::vector<perf_counter> m_perf;

#if defined(CRAUSER_OUT)
    std::atomic<double> m_out_threshold;
#endif
};

} // namespace sssp
