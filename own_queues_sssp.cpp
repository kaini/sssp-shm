#include "own_queues_sssp.hpp"
#include "thread_local_allocator.hpp"
#include <boost/heap/fibonacci_heap.hpp>
#include <chrono>
#include <iostream>
#include <unordered_set>

using namespace boost::heap;

namespace {
enum class state : unsigned char { unexplored, fringe, settled };
}

void sssp::own_queues_sssp::run_collective(thread_group& threads,
                                           int thread_rank,
                                           int group_count,
                                           int group_rank,
                                           thread_group& group_threads,
                                           int group_thread_rank,
                                           size_t node_count,
                                           size_t edge_count,
                                           array_slice<const array_slice<const edge>> thread_edges_by_node,
                                           array_slice<double> out_thread_distances,
                                           array_slice<size_t> out_thread_predecessors) {
    // Local data
    const size_t my_nodes_count = threads.chunk_size(thread_rank, node_count);
    const size_t my_nodes_start = threads.chunk_start(thread_rank, node_count);
    const size_t my_nodes_end = my_nodes_start + my_nodes_count;
    BOOST_ASSERT(thread_edges_by_node.size() == my_nodes_count);
    BOOST_ASSERT(out_thread_distances.size() == my_nodes_count);
    BOOST_ASSERT(out_thread_predecessors.size() == my_nodes_count);

    array_slice<double> distances = out_thread_distances;
    array_slice<size_t> predecessors = out_thread_predecessors;
    carray<state> states(my_nodes_count, state::unexplored);

    auto cmp_distance = [&](size_t a, size_t b) { return distances[a] > distances[b]; };
    using distance_queue_t =
        fibonacci_heap<size_t, compare<decltype(cmp_distance)>, allocator<thread_local_allocator<size_t>>>;
    carray<distance_queue_t::handle_type> distance_queue_handles(my_nodes_count);
    distance_queue_t distance_queue(cmp_distance);

#if defined(CRAUSER) || defined(CRAUSERDYN)
    // Queue for the IN criteria
    carray<double> min_incoming(my_nodes_count);
    auto cmp_crauser_in = [&](size_t a, size_t b) {
        return distances[a] - min_incoming[a] > distances[b] - min_incoming[b];
    };
    using crauser_in_queue_t =
        fibonacci_heap<size_t, compare<decltype(cmp_crauser_in)>, allocator<thread_local_allocator<size_t>>>;
    carray<crauser_in_queue_t::handle_type> crauser_in_queue_handles(my_nodes_count);
    crauser_in_queue_t crauser_in_queue(cmp_crauser_in);

    // Queue for the OUT criteria
#if defined(CRAUSER)
    carray<double> min_outgoing(my_nodes_count, INFINITY);
    auto min_outgoing_value = [&](size_t n) { return min_outgoing[n]; };
#elif defined(CRAUSERDYN)
    carray<edge> min_outgoing(my_nodes_count, edge(-1, -1, INFINITY));
    auto min_outgoing_value = [&](size_t n) { return min_outgoing[n].cost; };
#endif
    auto cmp_crauser_out = [&](size_t a, size_t b) {
        return distances[a] + min_outgoing_value(a) > distances[b] + min_outgoing_value(b);
    };
    using crauser_out_queue_t =
        fibonacci_heap<size_t, compare<decltype(cmp_crauser_out)>, allocator<thread_local_allocator<size_t>>>;
    carray<crauser_out_queue_t::handle_type> crauser_out_queue_handles(my_nodes_count);
    crauser_out_queue_t crauser_out_queue(cmp_crauser_out);
#endif

    // Globally shared data
    threads.single_collective([&] {
        m_seen_distances = carray<carray<std::atomic<double>>>(group_count);
        m_relaxations = carray<relaxed_vector<relaxation>>(threads.thread_count());
#if defined(CRAUSER) || defined(CRAUSERDYN)
        m_min_incoming = carray<std::atomic<double>>(node_count);
#endif
    });

    relaxed_vector<relaxation>& my_relaxations = m_relaxations[thread_rank];
    my_relaxations.init(threads, thread_rank, edge_count);

#if defined(CRAUSER) || defined(CRAUSERDYN)
    for (size_t n = my_nodes_start; n < my_nodes_end; ++n) {
        m_min_incoming[n].store(INFINITY, std::memory_order_relaxed);
    }
#endif

    threads.barrier_collective();

#if defined(CRAUSER) || defined(CRAUSERDYN)
    for (size_t n = 0; n < my_nodes_count; ++n) {
        for (const edge& edge : thread_edges_by_node[n]) {
            atomic_min(m_min_incoming[edge.destination], edge.cost);
#if defined(CRAUSER)
            if (edge.cost < min_outgoing[n]) {
                min_outgoing[n] = edge.cost;
            }
#elif defined(CRAUSERDYN)
            if (edge.cost < min_outgoing[n].cost) {
                min_outgoing[n] = edge;
            }
#endif
        }
    }
    threads.barrier_collective();
    for (size_t n = 0; n < my_nodes_count; ++n) {
        min_incoming[n] = m_min_incoming[n + my_nodes_start].load(std::memory_order_relaxed);
    }
#endif

    // Group shared data
    group_threads.single_collective([&] { m_seen_distances[group_rank] = carray<std::atomic<double>>(node_count); });
    carray<std::atomic<double>>& group_seen_distances = m_seen_distances[group_rank];
    for (size_t n = group_threads.chunk_start(group_thread_rank, node_count),
                end = n + group_threads.chunk_size(group_thread_rank, node_count);
         n < end;
         ++n) {
        group_seen_distances[n].store(INFINITY, std::memory_order_relaxed);
    }
    group_threads.barrier_collective();

    // Helper to settle a single edge.
    auto settle_edge = [&](size_t node, size_t predecessor_id, double distance) {
        if (states[node] != state::settled && distance < distances[node]) {
            distances[node] = distance;
            predecessors[node] = predecessor_id;
            if (states[node] == state::fringe) {
                distance_queue.update(distance_queue_handles[node]);
#if defined(CRAUSER) || defined(CRAUSERDYN)
                crauser_in_queue.update(crauser_in_queue_handles[node]);
                crauser_out_queue.update(crauser_out_queue_handles[node]);
#endif
            } else {
                states[node] = state::fringe;
                distance_queue_handles[node] = distance_queue.push(node);
#if defined(CRAUSER) || defined(CRAUSERDYN)
                crauser_in_queue_handles[node] = crauser_in_queue.push(node);
                crauser_out_queue_handles[node] = crauser_out_queue.push(node);
#endif
            }
        }
    };

    // Helpter to relax a node either locally or remote.
    auto relax_node = [&](size_t node) {
        distance_queue.erase(distance_queue_handles[node]);
#if defined(CRAUSER) || defined(CRAUSERDYN)
        crauser_in_queue.erase(crauser_in_queue_handles[node]);
        crauser_out_queue.erase(crauser_out_queue_handles[node]);
#endif
        states[node] = state::settled;

        group_seen_distances[node + my_nodes_start].store(-INFINITY, std::memory_order_relaxed);

        for (const edge& e : thread_edges_by_node[node]) {
            if (distances[node] + e.cost < group_seen_distances[e.destination].load(std::memory_order_relaxed)) {
                group_seen_distances[e.destination].store(distances[node] + e.cost, std::memory_order_relaxed);
                int dest_thread = threads.chunk_thread_at(node_count, e.destination);
                if (dest_thread == thread_rank) {
                    settle_edge(e.destination - my_nodes_start, node + my_nodes_start, distances[node] + e.cost);
                } else {
                    m_relaxations[dest_thread].push_back(
                        relaxation{e.destination, node + my_nodes_start, distances[node] + e.cost});
                }
            }
        }
    };

    // Start node
    if (thread_rank == 0) {
        settle_edge(0, -1, 0.0);
    }

    for (int phase = 0;; ++phase) {
#if defined(CRAUSER) || defined(CRAUSERDYN)
        threads.reduce_linear_collective(m_in_threshold,
                                         double(INFINITY),
                                         distance_queue.empty() ? INFINITY : distances[distance_queue.top()],
                                         [](auto a, auto b) { return std::min(a, b); });
        const double in_threshold = m_in_threshold.load(std::memory_order_relaxed);
        threads.reduce_linear_collective(m_out_threshold,
                                         double(INFINITY),
                                         crauser_out_queue.empty() ? INFINITY
                                                                   : distances[crauser_out_queue.top()] +
                                                                         min_outgoing_value(crauser_out_queue.top()),
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

        my_relaxations.for_each(
            [&](const relaxation& r) { settle_edge(r.node - my_nodes_start, r.predecessor, r.distance); });
        my_relaxations.clear();

        threads.barrier_collective();

#if defined(CRAUSERDYN)
        for (size_t n = 0; n < my_nodes_count; ++n) {
            if (states[n] != state::settled && min_outgoing[n].cost != INFINITY) {
                bool changed = false;
                if (group_seen_distances[min_outgoing[n].destination].load(std::memory_order_relaxed) < 0.0) {
                    changed = true;
                    edge result(-1, -1, INFINITY);
                    double min_cost = min_outgoing[n].cost;
                    for (const edge& e : thread_edges_by_node[n]) {
                        if (e.cost < result.cost && e.cost >= min_cost &&
                            group_seen_distances[e.destination].load(std::memory_order_relaxed) >= 0.0) {
                            result = e;
                        }
                    }
                    min_outgoing[n] = result;
                }
                if (changed && states[n] == state::fringe) {
                    distance_queue.update(distance_queue_handles[n]);
                    crauser_in_queue.update(crauser_in_queue_handles[n]);
                    crauser_out_queue.update(crauser_out_queue_handles[n]);
                }
            }
        }

        threads.barrier_collective();
#endif
    }
}
