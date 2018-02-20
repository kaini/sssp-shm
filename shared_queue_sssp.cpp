#include "shared_queue_sssp.hpp"
#include "collective_functions.hpp"
#include <chrono>

sssp::shared_queue_sssp::shared_queue_sssp(size_t node_count, const carray<array_slice<const edge>>& edges)
    : m_node_count(node_count), m_edges(edges), m_nodes(node_count) {}

void sssp::shared_queue_sssp::run_collective(thread_group& threads, int thread_rank, array_slice<result> out_result) {
    const auto start_time = std::chrono::steady_clock::now();

#if defined(CRAUSER)
    threads.for_each_collective(thread_rank, m_edges, [&](const array_slice<const edge>& node) {
        for (const edge& edge : node) {
            shared_queue_sssp::node& dest_node = m_nodes[edge.destination];
            std::lock_guard<std::mutex> lock(dest_node.mutex);
            if (edge.cost < dest_node.min_incoming) {
                dest_node.min_incoming = edge.cost;
            }
        }
    });
#endif

    const auto after_init_time = std::chrono::steady_clock::now();

    threads.single_collective([&] {
        node& start_node = m_nodes[0];
        out_result[node_id(start_node)].distance = 0.0;
        out_result[node_id(start_node)].predecessor = -1;
        start_node.state = state::fringe;
        m_distance_queue.push({out_result[node_id(start_node)].distance, &start_node});
#if defined(CRAUSER)
        m_crauser_in_queue.push({out_result[node_id(start_node)].distance - start_node.min_incoming, &start_node});
#endif
    });

    auto relax_node = [&](const node& source) {
        const size_t source_id = node_id(source);
        for (const edge& edge : m_edges[source_id]) {
            node& dest = m_nodes[edge.destination];
            const size_t dest_id = node_id(dest);
            const double distance = out_result[source_id].distance + edge.cost;
            std::lock_guard<std::mutex> dlock(dest.mutex);
            if (dest.state != state::settled && distance < out_result[dest_id].distance) {
                out_result[dest_id].distance = distance;
                out_result[dest_id].predecessor = source_id;
                dest.state = state::fringe;
                m_distance_queue.push({out_result[dest_id].distance, &dest});
#if defined(CRAUSER)
                m_crauser_in_queue.push({out_result[dest_id].distance - dest.min_incoming, &dest});
#endif
            }
        }
    };

    for (;;) {
#if defined(DIJKSTRA)
#error TODO
#elif defined(CRAUSER)
        // Pop all settled nodes
        queued_node qn;
        bool queue_empty = true;
        while (m_distance_queue.try_pop(qn)) {
            std::lock_guard<std::mutex> lock(qn.node->mutex);
            if (qn.node->state != state::settled) {
                m_distance_queue.push(qn);
                queue_empty = false;
                break;
            }
        }

        // Test for termination
        threads.reduce_linear_collective(m_done, queue_empty, [](bool a, bool b) { return a && b; });
        if (m_done.load(std::memory_order_relaxed)) {
            break;
        }

        // Agree on the queue top
        threads.reduce_linear_collective(m_in_threshold, qn.value, [](double a, double b) { return std::min(a, b); });
        double in_threshold = m_in_threshold.load(std::memory_order_relaxed);

        // Relax according to the Crauser IN criteria
        while (m_crauser_in_queue.try_pop(qn)) {
            std::unique_lock<std::mutex> lock(qn.node->mutex);
            if (qn.node->state != state::settled) {
                if (qn.value <= in_threshold) {
                    qn.node->state = state::settled;
                } else {
                    m_crauser_in_queue.push(qn);
                    break;
                }
            } else {
                continue;
            }
            lock.unlock();
            relax_node(*qn.node);
        }

        // Phase end (is this required?)
        threads.barrier_collective();
#endif
    }

    const auto end_time = std::chrono::steady_clock::now();
    threads.reduce_linear_collective(
        m_time, (end_time - start_time).count() / 1000000000.0, [](double a, double b) { return std::max(a, b); });
    threads.reduce_linear_collective(m_init_time,
                                     (after_init_time - start_time).count() / 1000000000.0,
                                     [](double a, double b) { return std::max(a, b); });
}

size_t sssp::shared_queue_sssp::node_id(const node& node) const {
    return &node - &m_nodes[0];
}
