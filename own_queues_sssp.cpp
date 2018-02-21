#include "own_queues_sssp.hpp"
#include "thread_local_allocator.hpp"
#include <boost/heap/fibonacci_heap.hpp>
#include <chrono>
#include <iostream>
#include <unordered_set>

using namespace boost::heap;

sssp::own_queues_sssp::own_queues_sssp(size_t node_count) : m_node_count(node_count), m_seen_distances(node_count) {
    for (auto& d : m_seen_distances) {
        d.store(INFINITY, std::memory_order_relaxed);
    }
#if defined(CRAUSER)
    m_min_incoming = carray<std::atomic<double>>(node_count);
    for (auto& d : m_min_incoming) {
        d.store(INFINITY, std::memory_order_relaxed);
    }
#endif
}

void sssp::own_queues_sssp::run_collective(thread_group& threads,
                                           int thread_rank,
                                           array_slice<array_slice<const edge>> edges,
                                           array_slice<result> out_result) {
    enum class state {
        unexplored,
        fringe,
        settled,
    };

    const size_t my_node_count = threads.for_each_count(thread_rank, m_node_count);
    carray<double> distances(my_node_count, INFINITY);
    carray<size_t> predecessors(my_node_count, -1);
    carray<state> states(my_node_count, state::unexplored);

    auto cmp_distance = [&](size_t a, size_t b) { return distances[a] > distances[b]; };
    using distance_queue_t =
        fibonacci_heap<size_t, compare<decltype(cmp_distance)>, allocator<thread_local_allocator<size_t>>>;
    carray<distance_queue_t::handle_type> distance_queue_handles(my_node_count);

#if defined(CRAUSER)
    carray<double> min_incoming(my_node_count);
    auto cmp_crauser_in = [&](size_t a, size_t b) {
        return distances[a] - min_incoming[a] > distances[b] - min_incoming[b];
    };
    using crauser_in_queue_t =
        fibonacci_heap<size_t, compare<decltype(cmp_crauser_in)>, allocator<thread_local_allocator<size_t>>>;
    carray<crauser_in_queue_t::handle_type> crauser_in_queue_handles(my_node_count);

    carray<double> min_outgoing(my_node_count, INFINITY);
    auto cmp_crauser_out = [&](size_t a, size_t b) {
        return distances[a] + min_outgoing[a] > distances[b] + min_outgoing[b];
    };
    using crauser_out_queue_t =
        fibonacci_heap<size_t, compare<decltype(cmp_crauser_out)>, allocator<thread_local_allocator<size_t>>>;
    carray<crauser_out_queue_t::handle_type> crauser_out_queue_handles(my_node_count);
#endif

    threads.single_collective(
        [&] { m_relaxations = carray<tbb::concurrent_vector<relaxation>>(threads.thread_count()); });
    tbb::concurrent_vector<relaxation>& my_relaxations = m_relaxations[thread_rank];

    const auto start_time = std::chrono::steady_clock::now();

#if defined(CRAUSER)
    threads.for_each_collective(thread_rank, edges, [&](const array_slice<const edge>& node) {
        for (const edge& edge : node) {
            atomic_min(m_min_incoming[edge.destination], edge.cost);
            if (edge.cost < min_outgoing[edge.source / threads.thread_count()]) {
                min_outgoing[edge.source / threads.thread_count()] = edge.cost;
            }
        }
    });
    for (size_t i = 0; i < my_node_count; ++i) {
        min_incoming[i] = m_min_incoming[i * threads.thread_count() + thread_rank].load(std::memory_order_relaxed);
    }
#endif

    const auto after_init_time = std::chrono::steady_clock::now();

    distance_queue_t distance_queue(cmp_distance);
#if defined(CRAUSER)
    crauser_in_queue_t crauser_in_queue(cmp_crauser_in);
    crauser_out_queue_t crauser_out_queue(cmp_crauser_out);
#endif

    auto settle_edge = [&](size_t node, size_t predecessor_id, double distance) {
        if (states[node] != state::settled && distance < distances[node]) {
            distances[node] = distance;
            predecessors[node] = predecessor_id;
            if (states[node] == state::fringe) {
                distance_queue.update(distance_queue_handles[node]);
#if defined(CRAUSER)
                crauser_in_queue.update(crauser_in_queue_handles[node]);
                crauser_out_queue.update(crauser_out_queue_handles[node]);
#endif
            } else {
                states[node] = state::fringe;
                distance_queue_handles[node] = distance_queue.push(node);
#if defined(CRAUSER)
                crauser_in_queue_handles[node] = crauser_in_queue.push(node);
                crauser_out_queue_handles[node] = crauser_out_queue.push(node);
#endif
            }
        }
    };

    auto relax_node = [&](size_t node) {
        distance_queue.erase(distance_queue_handles[node]);
#if defined(CRAUSER)
        crauser_in_queue.erase(crauser_in_queue_handles[node]);
        crauser_out_queue.erase(crauser_out_queue_handles[node]);
#endif
        states[node] = state::settled;

        const size_t node_id = node * threads.thread_count() + thread_rank;
        m_seen_distances[node_id].store(0.0, std::memory_order_relaxed);

        for (const edge& e : edges[node_id]) {
            if (distances[node] + e.cost < m_seen_distances[e.destination].load(std::memory_order_relaxed)) {
                m_seen_distances[e.destination].store(distances[node] + e.cost, std::memory_order_relaxed);
                int dest_thread = e.destination % threads.thread_count();
                if (dest_thread == thread_rank) {
                    settle_edge(e.destination / threads.thread_count(), node_id, distances[node] + e.cost);
                } else {
                    m_relaxations[dest_thread].push_back(relaxation{e.destination, node_id, distances[node] + e.cost});
                }
            }
        }
    };

    if (thread_rank == 0) {
        settle_edge(0, -1, 0.0);
    }

    for (int phase = 0;; ++phase) {
#if defined(CRAUSER)
        threads.reduce_linear_collective(m_in_threshold,
                                         distance_queue.empty() ? INFINITY : distances[distance_queue.top()],
                                         [](auto a, auto b) { return std::min(a, b); });
        const double in_threshold = m_in_threshold.load(std::memory_order_relaxed);
        threads.reduce_linear_collective(m_out_threshold,
                                         crauser_out_queue.empty() ? INFINITY
                                                                   : distances[crauser_out_queue.top()] +
                                                                         min_outgoing[crauser_out_queue.top()],
                                         [](auto a, auto b) { return std::min(a, b); });
        const double out_threshold = m_out_threshold.load(std::memory_order_relaxed);
        if (in_threshold == INFINITY && out_threshold == INFINITY) {
            break;
        }
        while (!crauser_in_queue.empty() &&
               distances[crauser_in_queue.top()] - min_incoming[crauser_in_queue.top()] <= in_threshold) {
            relax_node(crauser_in_queue.top());
        }
        while (!distance_queue.empty() && distances[distance_queue.top()] <= out_threshold) {
            relax_node(distance_queue.top());
        }
#endif

        threads.barrier_collective();

        for (const relaxation& r : my_relaxations) {
            settle_edge(r.node / threads.thread_count(), r.predecessor, r.distance);
        }
        my_relaxations.clear();

        threads.barrier_collective();
    }

    threads.for_each_with_index_collective(thread_rank, out_result, [&](size_t i, result& r) {
        r.distance = distances[i / threads.thread_count()];
        r.predecessor = predecessors[i / threads.thread_count()];
    });

    const auto end_time = std::chrono::steady_clock::now();
    threads.reduce_linear_collective(
        m_time, (end_time - start_time).count() / 1000000000.0, [](double a, double b) { return std::max(a, b); });
    threads.reduce_linear_collective(m_init_time,
                                     (after_init_time - start_time).count() / 1000000000.0,
                                     [](double a, double b) { return std::max(a, b); });
}
