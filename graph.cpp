#include "graph.hpp"
#include "thread_local_allocator.hpp"
#include <random>
#include <vector>

void sssp::distribute_nodes_generate_uniform_collective(
    thread_group& threads,
    int thread_rank,
    size_t node_count,
    double edge_chance,
    unsigned int seed,
    std::vector<edge>& out_thread_edges,
    std::vector<array_slice<const edge>>& out_thread_edges_by_node) {
    seed *= (thread_rank + 1234);
    size_t my_nodes_count = threads.chunk_size(thread_rank, node_count);
    size_t my_nodes_start = threads.chunk_start(thread_rank, node_count);
    out_thread_edges_by_node.resize(my_nodes_count);

    std::mt19937 edge_count_rng(seed);
    std::binomial_distribution<size_t> edge_count_dist(node_count, edge_chance);
    size_t edge_count = 0;
    for (size_t n = 0; n < my_nodes_count; ++n) {
        edge_count += edge_count_dist(edge_count_rng);
    };
    out_thread_edges.resize(edge_count);

    edge_count_rng.seed(seed);
    edge_count_dist.reset();
    size_t edge_at = 0;
    std::mt19937 other_rng(seed + 1);
    std::uniform_int_distribution<size_t> node_dist(0, node_count - 1);
    std::uniform_real_distribution<double> edge_cost_dist(0.0, 1.0);
    for (size_t n = 0; n < my_nodes_count; ++n) {
        size_t node_edge_count = edge_count_dist(edge_count_rng);
        array_slice<edge> edges(&out_thread_edges[edge_at], node_edge_count);
        out_thread_edges_by_node[n] = edges.as_const();
        for (edge& e : edges) {
            e.source = n + my_nodes_start;
            e.destination = node_dist(other_rng);
            e.cost = edge_cost_dist(other_rng);
        }
        edge_at += node_edge_count;
    }
    BOOST_ASSERT(edge_count == edge_at);

    threads.barrier_collective(true);
}

void sssp::distribute_edges_generate_uniform_collective(
    thread_group& threads,
    int thread_rank,
    size_t node_count,
    double edge_chance,
    unsigned int seed,
    std::vector<edge>& out_thread_edges,
    std::vector<array_slice<const edge>>& out_thread_edges_by_node) {
    std::mt19937 rng(seed);
    size_t total_edge_count = std::binomial_distribution<size_t>(node_count * node_count, edge_chance)(rng);
    size_t my_edge_count = threads.chunk_size(thread_rank, total_edge_count);
    out_thread_edges.resize(my_edge_count);
    out_thread_edges_by_node.resize(node_count);

    rng.seed(seed * (thread_rank + 1234));
    std::uniform_int_distribution<size_t> node_dist(0, node_count - 1);
    std::uniform_real_distribution<double> cost_dist(0.0, 1.0);
    for (auto& edge : out_thread_edges) {
        edge.source = node_dist(rng);
        edge.destination = node_dist(rng);
        edge.cost = cost_dist(rng);
    }

    std::sort(out_thread_edges.begin(), out_thread_edges.end(), [](const edge& a, const edge& b) {
        return a.source < b.source;
    });

    size_t at = 0;
    for (size_t node = 0; node < node_count; ++node) {
        size_t start = at;
        while (at < my_edge_count && out_thread_edges[at].source == node) {
            at += 1;
        }
        if (start != at) {
            out_thread_edges_by_node[node] = array_slice<const edge>(&out_thread_edges[start], at - start);
        }
    }

    threads.barrier_collective(true);
}

void sssp::distribute_nodes_generate_kronecker::run_collective(
    thread_group& threads,
    int thread_rank,
    int k,
    unsigned int seed,
    std::vector<edge>& out_thread_edges,
    std::vector<array_slice<const edge>>& out_thread_edges_by_node) {

    const size_t start_size = 2;

    size_t final_size = 1;
    for (int i = 0; i < k; ++i) {
        final_size *= start_size;
    }

    // Setup the initiator
    threads.single_collective(
        [&] {
            std::mt19937 rng(seed);

            m_start_size = start_size;
            m_matrix = { 1.425, 0.475, 0.475, 0.125 };

            m_matrix_prefix_sum.resize(start_size * start_size);
            m_matrix_prefix_sum[0] = m_matrix[0];
            for (int i = 1; i < m_matrix.size(); ++i) {
                m_matrix_prefix_sum[i] = m_matrix_prefix_sum[i - 1] + m_matrix[i];
            }

            double edges_expected_value = std::pow(m_matrix_prefix_sum.back(), k);
            m_edge_count = std::poisson_distribution<size_t>(edges_expected_value)(rng);

            m_inboxes.resize(threads.thread_count());
        },
        true);

    std::unique_ptr<relaxed_vector<edge>> my_inbox(new relaxed_vector<edge>(threads, thread_rank, m_edge_count));
    m_inboxes[thread_rank] = my_inbox.get();

    threads.barrier_collective(true);

    // Each thread samples edges
    std::mt19937 rng(seed * (thread_rank + 1234));
    std::uniform_real_distribution<double> cell_dist(0.0, m_matrix_prefix_sum.back());
    std::uniform_real_distribution<double> cost_dist(0.0, 1.0);
    for (size_t i = threads.chunk_start(thread_rank, m_edge_count),
                end = i + threads.chunk_size(thread_rank, m_edge_count);
         i < end;
         ++i) {
        size_t cell = 0;
        size_t granularity = 1;
        for (int i = 0; i < k; ++i) {
            double value = cell_dist(rng);
            size_t index =
                std::distance(m_matrix_prefix_sum.begin(),
                              std::lower_bound(m_matrix_prefix_sum.begin(), m_matrix_prefix_sum.end(), value));
            if (index == m_matrix_prefix_sum.size()) {
                index = m_matrix_prefix_sum.size() - 1;
            }
            cell += index * granularity;
            granularity *= start_size * start_size;
        }

        const size_t source = cell / final_size;
        const size_t dest = cell % final_size;
        m_inboxes[threads.chunk_thread_at(final_size, source)]->push_back(edge(source, dest, cost_dist(rng)));
    }

    threads.barrier_collective(true);

    const size_t my_nodes_start = threads.chunk_start(thread_rank, final_size);
    const size_t my_nodes_size = threads.chunk_size(thread_rank, final_size);

    std::vector<std::vector<edge, thread_local_allocator<edge>>,
                thread_local_allocator<std::vector<edge, thread_local_allocator<edge>>>>
        buckets;
    buckets.resize(my_nodes_size);
    my_inbox->for_each([&](const edge& edge) { buckets[edge.source - my_nodes_start].push_back(edge); });

    out_thread_edges.resize(my_inbox->size());
    auto out_thread_edges_at = out_thread_edges.begin();
    for (const auto& bucket : buckets) {
        out_thread_edges_at = std::move(bucket.begin(), bucket.end(), out_thread_edges_at);
    }

    out_thread_edges_by_node.resize(my_nodes_size);
    size_t at = 0;
    for (size_t node = 0; node < my_nodes_size; ++node) {
        size_t start = at;
        while (at < out_thread_edges.size() && out_thread_edges[at].source == node + my_nodes_start) {
            at += 1;
        }
        if (start != at) {
            out_thread_edges_by_node[node] = array_slice<const edge>(&out_thread_edges[start], at - start);
        }
    }

    threads.barrier_collective(true);
}
