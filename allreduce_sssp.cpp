#include "allreduce_sssp.hpp"
#include "carray.hpp"
#include "dijkstra.hpp"
#include <vector>

void sssp::allreduce_sssp::run_collective(thread_group& threads,
                                          int thread_rank,
                                          array_slice<const array_slice<const edge>> thread_edges_by_node,
                                          array_slice<double> thread_distances,
                                          array_slice<size_t> thread_predecessors) {
    size_t node_count = thread_edges_by_node.size();
    carray<bool> invalidated(node_count);

    threads.single_collective([&] { m_global_distances = carray<std::atomic<double>>(node_count); });

    for (size_t n = threads.chunk_start(thread_rank, node_count), end = n + threads.chunk_size(thread_rank, node_count);
         n < end;
         ++n) {
        m_global_distances[n].store(INFINITY, std::memory_order_relaxed);
    }

    for (int phase = 0;; ++phase) {
        if (phase == 0) {
            dijkstra(thread_edges_by_node, thread_distances, thread_predecessors);
        } else {
            dijkstra(thread_edges_by_node, thread_distances, thread_predecessors, invalidated);
        }
        threads.barrier_collective();

        for (size_t n = 0; n < node_count; ++n) {
            atomic_min(m_global_distances[n], thread_distances[n]);
        }
        threads.barrier_collective();

        bool something_invalidated = false;
        for (size_t n = 0; n < node_count; ++n) {
            double global_distance = m_global_distances[n].load(std::memory_order_relaxed);
            invalidated[n] = global_distance < thread_distances[n];
            if (invalidated[n]) {
                thread_distances[n] = global_distance;
                thread_predecessors[n] = -2;
                something_invalidated = true;
            }
        }
        threads.reduce_linear_collective(m_continue, something_invalidated, [](bool a, bool b) { return a || b; });
        if (!m_continue.load(std::memory_order_relaxed)) {
            break;
        }
    }
}
