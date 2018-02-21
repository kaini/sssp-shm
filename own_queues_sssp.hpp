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
    own_queues_sssp(size_t node_count);

    void run_collective(thread_group& threads,
                        int thread_rank,
                        array_slice<array_slice<const edge>> edges,
                        array_slice<result> out_result);

    double time() const { return m_time.load(std::memory_order_relaxed); }
    double init_time() const { return m_init_time.load(std::memory_order_relaxed); }

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