#pragma once
#include "array_slice.hpp"
#include "carray.hpp"
#include "collective_functions.hpp"
#include "graph.hpp"
#include <atomic>

namespace sssp {

class allreduce_sssp {
  public:
    // Runs the algorithm on the given thread group.
    // thread_edges_by_node, out_thread_distances and out_thread_predecessors have to be
    // different for each thread. They represent the edges and results for each thread in the group.
    // The lengths of all array_slices have to be equal for all parameters and all threads.
    // The algorithm does not initialize out_thread_distances or out_thread_predecessors.
    void run_collective(thread_group& threads,
                        int thread_rank,
                        array_slice<const array_slice<const edge>> thread_edges_by_node,
                        array_slice<double> out_thread_distances,
                        array_slice<size_t> out_thread_predecessors);

  private:
    carray<std::atomic<double>> m_global_distances;
    std::atomic<bool> m_continue;
};

} // namespace sssp
