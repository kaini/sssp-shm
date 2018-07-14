#pragma once
#include "array_slice.hpp"
#include "collective_functions.hpp"
#include "relaxed_vector.hpp"
#include <cmath>

namespace sssp {

struct edge {
    edge() {}
    edge(size_t s, size_t d, double c) : source(s), destination(d), cost(c) {}
    size_t source = -1;
    size_t destination = -1;
    double cost = 0.0;
};

void distribute_nodes_generate_uniform_collective(thread_group& threads,
                                                  int thread_rank,
                                                  size_t node_count,
                                                  double edge_chance,
                                                  unsigned int seed,
                                                  std::vector<edge>& out_thread_edges,
                                                  std::vector<array_slice<const edge>>& out_thread_edges_by_node);

void distribute_edges_generate_uniform_collective(thread_group& threads,
                                                  int thread_rank,
                                                  size_t node_count,
                                                  double edge_chance,
                                                  unsigned int seed,
                                                  std::vector<edge>& out_thread_edges,
                                                  std::vector<array_slice<const edge>>& out_thread_edges_by_node);

class distribute_nodes_generate_kronecker {
  public:
    // See the simulation tool for the details of this algorithm.
    void run_collective(thread_group& threads,
                        int thread_rank,
                        int k,
                        unsigned int seed,
                        std::vector<edge>& out_thread_edges,
                        std::vector<array_slice<const edge>>& out_thread_edges_by_node);

  private:
    const double& p1(size_t x, size_t y) const { return m_matrix[x + y * m_start_size]; }

    size_t m_start_size;
    size_t m_edge_count;
    std::vector<double> m_matrix_prefix_sum;
    std::vector<double> m_matrix;
    std::vector<relaxed_vector<edge>*> m_inboxes;
};

} // namespace sssp