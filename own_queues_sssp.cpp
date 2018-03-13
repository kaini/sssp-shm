#include "own_queues_sssp.hpp"
#include "thread_local_allocator.hpp"
#include <algorithm>
#include <boost/heap/fibonacci_heap.hpp>
#include <iostream>
#include <unordered_set>

#if defined(CRAUSER_IN) && defined(CRAUSER_INDYN)
#error CRAUSER_IN and CRAUSER_INDYN cannot be combined
#endif

#if defined(CRAUSER_OUT) && defined(CRAUSER_OUTDYN)
#error CRAUSER_OUT and CRAUSER_OUTDYN cannot be combined
#endif

#if defined(Q_HEAP) == defined(Q_ARRAY)
#error You have to define either Q_HEAP or Q_ARRAY
#endif

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
    perf_counter perf;

    // Local data
    perf.first_timeblock("init");
    const size_t my_nodes_count = threads.chunk_size(thread_rank, node_count);
    const size_t my_nodes_start = threads.chunk_start(thread_rank, node_count);
    const size_t my_nodes_end = my_nodes_start + my_nodes_count;

    array_slice<double> distances = out_thread_distances;
    array_slice<size_t> predecessors = out_thread_predecessors;
    carray<state> states(my_nodes_count, state::unexplored);
    std::vector<size_t, thread_local_allocator<size_t>> todo;

    auto cmp_distance = [&](size_t a, size_t b) { return distances[a] > distances[b]; };
#if defined(Q_ARRAY)
    std::vector<size_t, thread_local_allocator<size_t>> fringe;
#elif defined(Q_HEAP)
    using distance_queue_t =
        fibonacci_heap<size_t, compare<decltype(cmp_distance)>, allocator<thread_local_allocator<size_t>>>;
    distance_queue_t distance_queue(cmp_distance);
    carray<distance_queue_t::handle_type> distance_queue_handles(my_nodes_count);
#endif

#if defined(CRAUSER_IN)
    carray<double> min_incoming(my_nodes_count);
    auto min_incoming_value = [&](size_t n) { return min_incoming[n]; };
    auto cmp_crauser_in = [&](size_t a, size_t b) {
        return distances[a] - min_incoming_value(a) > distances[b] - min_incoming_value(b);
    };
#endif

#if defined(CRAUSER_INDYN)
    carray<node_cost> min_incoming(my_nodes_count, node_cost{size_t(-1), INFINITY});
    auto min_incoming_value = [&](size_t n) { return min_incoming[n].cost; };
    auto cmp_crauser_in = [&](size_t a, size_t b) {
        return distances[a] - min_incoming_value(a) > distances[b] - min_incoming_value(b);
    };
#endif

#if (defined(CRAUSER_IN) || defined(CRAUSER_INDYN)) && defined(Q_HEAP)
    using crauser_in_queue_t =
        fibonacci_heap<size_t, compare<decltype(cmp_crauser_in)>, allocator<thread_local_allocator<size_t>>>;
    crauser_in_queue_t crauser_in_queue(cmp_crauser_in);
    carray<crauser_in_queue_t::handle_type> crauser_in_queue_handles(my_nodes_count);
#endif

#if defined(CRAUSER_OUT)
    carray<double> min_outgoing(my_nodes_count, INFINITY);
    auto min_outgoing_value = [&](size_t n) { return min_outgoing[n]; };
    auto cmp_crauser_out = [&](size_t a, size_t b) {
        return distances[a] + min_outgoing_value(a) > distances[b] + min_outgoing_value(b);
    };
#endif

#if defined(CRAUSER_OUTDYN)
    carray<node_cost> min_outgoing(my_nodes_count, node_cost{size_t(-1), INFINITY});
    auto min_outgoing_value = [&](size_t n) { return min_outgoing[n].cost; };
    auto cmp_crauser_out = [&](size_t a, size_t b) {
        return distances[a] + min_outgoing_value(a) > distances[b] + min_outgoing_value(b);
    };
#endif

#if (defined(CRAUSER_OUT) || defined(CRAUSER_OUTDYN)) && defined(Q_HEAP)
    using crauser_out_queue_t =
        fibonacci_heap<size_t, compare<decltype(cmp_crauser_out)>, allocator<thread_local_allocator<size_t>>>;
    crauser_out_queue_t crauser_out_queue(cmp_crauser_out);
    carray<crauser_out_queue_t::handle_type> crauser_out_queue_handles(my_nodes_count);
#endif

#if defined(TRAFF) && defined(Q_ARRAY)
    carray<double> min_pred_pred(my_nodes_count, INFINITY);
#endif

#if defined(TRAFF) && defined(Q_HEAP)
    std::vector<size_t, thread_local_allocator<size_t>> traff_candidates;
    carray<double> min_pred_pred(my_nodes_count, INFINITY);
    auto cmp_traff = [&](size_t a, size_t b) {
        return distances[a] - min_pred_pred[a] > distances[b] - min_pred_pred[b];
    };
    using traff_queue_t =
        fibonacci_heap<size_t, compare<decltype(cmp_traff)>, allocator<thread_local_allocator<size_t>>>;
    traff_queue_t traff_queue(cmp_traff);
    carray<traff_queue_t::handle_type> traff_queue_handles(my_nodes_count);
#endif

    // Globally shared data
    threads.single_collective(
        [&] {
            m_seen_distances = carray<carray<std::atomic<double>>>(group_count);
            m_relaxations = carray<relaxed_vector<relaxation>>(threads.thread_count());
#if defined(CRAUSER_IN) || defined(TRAFF)
            m_min_incoming = carray<std::atomic<double>>(node_count);
#endif
#if defined(CRAUSER_INDYN) || defined(TRAFF)
            m_incoming_at = carray<std::atomic<size_t>>(threads.thread_count());
            m_incoming_edges = carray<array_slice<edge>>(threads.thread_count());
#endif
#if defined(TRAFF)
            m_predecessors_in_fringe = carray<std::atomic<size_t>>(node_count);
#endif
        },
        true);

    relaxed_vector<relaxation>& my_relaxations = m_relaxations[thread_rank];
    my_relaxations.init(threads, thread_rank, edge_count);

#if defined(CRAUSER_IN)
    for (size_t n = my_nodes_start; n < my_nodes_end; ++n) {
        m_min_incoming[n].store(INFINITY, std::memory_order_relaxed);
    }
#endif

    threads.barrier_collective(true);

    // Exchange incoming edges
#if defined(CRAUSER_INDYN) || defined(TRAFF)
    carray<size_t> edge_counts(threads.thread_count(), 0);
    for (size_t node = 0; node < my_nodes_count; ++node) {
        for (const edge& edge : thread_edges_by_node[node]) {
            edge_counts[threads.chunk_thread_at(node_count, edge.destination)] += 1;
        }
    }
    for (int t = 0; t < threads.thread_count(); ++t) {
        m_incoming_at[t].fetch_add(edge_counts[t], std::memory_order_relaxed);
    }
    threads.barrier_collective(true);

    carray<edge> thread_in_edges(m_incoming_at[thread_rank].load(std::memory_order_relaxed));
    m_incoming_at[thread_rank].store(0, std::memory_order_relaxed);
    m_incoming_edges[thread_rank] = array_slice<edge>(thread_in_edges.data(), thread_in_edges.size());
    threads.barrier_collective(true);

    for (size_t node = 0; node < my_nodes_count; ++node) {
        for (const edge& edge : thread_edges_by_node[node]) {
            int dest_thread = threads.chunk_thread_at(node_count, edge.destination);
            m_incoming_edges[dest_thread][m_incoming_at[dest_thread].fetch_add(1, std::memory_order_relaxed)] = edge;
        }
    }
    threads.barrier_collective(true);

    std::sort(thread_in_edges.begin(), thread_in_edges.end(), [](const edge& a, const edge& b) {
        return a.destination < b.destination;
    });
    carray<array_slice<const edge>> thread_in_edges_by_node(my_nodes_count);
    for (size_t node = 0, at = 0; node < my_nodes_count; ++node) {
        size_t start = at;
        while (at < thread_in_edges.size() && thread_in_edges[at].destination == node + my_nodes_start) {
            at += 1;
        }
        if (start != at) {
            thread_in_edges_by_node[node] = array_slice<const edge>(&thread_in_edges[start], at - start);
        }
    }
#endif

#if defined(CRAUSER_INDYN)
    for (size_t n = 0; n < my_nodes_count; ++n) {
        for (const edge& edge : thread_in_edges_by_node[n]) {
            if (edge.cost < min_incoming[n].cost) {
                min_incoming[n] = node_cost{edge.source, edge.cost};
            }
        }
    }
#endif

#if defined(CRAUSER_IN) || defined(CRAUSER_OUT) || defined(CRAUSER_OUTDYN) || defined(TRAFF)
    for (size_t n = 0; n < my_nodes_count; ++n) {
        for (const edge& edge : thread_edges_by_node[n]) {
#if defined(CRAUSER_IN) || defined(TRAFF)
            atomic_min(m_min_incoming[edge.destination], edge.cost);
#endif
#if defined(CRAUSER_OUT)
            if (edge.cost < min_outgoing[n]) {
                min_outgoing[n] = edge.cost;
            }
#endif
#if defined(CRAUSER_OUTDYN)
            if (edge.cost < min_outgoing[n].cost) {
                min_outgoing[n] = node_cost{edge.destination, edge.cost};
            }
#endif
        }
    }
#endif

#if defined(CRAUSER_IN) || defined(TRAFF)
    threads.barrier_collective(true);
#endif

#if defined(CRAUSER_IN)
    for (size_t n = 0; n < my_nodes_count; ++n) {
        min_incoming[n] = m_min_incoming[n + my_nodes_start].load(std::memory_order_relaxed);
    }
#endif

#if defined(TRAFF)
    for (size_t n = 0; n < my_nodes_count; ++n) {
        double min = INFINITY;
        for (const auto& edge : thread_in_edges_by_node[n]) {
            double value = edge.cost + m_min_incoming[n + my_nodes_start].load(std::memory_order_relaxed);
            if (value < min) {
                min = value;
            }
        }
        min_pred_pred[n] = min;
    }
#endif

    // Group shared data
    group_threads.single_collective([&] { m_seen_distances[group_rank] = carray<std::atomic<double>>(node_count); },
                                    true);
    carray<std::atomic<double>>& group_seen_distances = m_seen_distances[group_rank];
    for (size_t n = group_threads.chunk_start(group_thread_rank, node_count),
                end = n + group_threads.chunk_size(group_thread_rank, node_count);
         n < end;
         ++n) {
        group_seen_distances[n].store(INFINITY, std::memory_order_relaxed);
    }
    group_threads.barrier_collective(true);

    // Helper to settle a single edge.
    auto settle_edge = [&](size_t node, size_t predecessor_id, double distance) {
        if (states[node] != state::settled && distance < distances[node]) {
            distances[node] = distance;
            predecessors[node] = predecessor_id;

#if defined(TRAFF)
            if (states[node] != state::fringe) {
                for (const edge& edge : thread_edges_by_node[node]) {
                    if (group_seen_distances[edge.destination].load(std::memory_order_relaxed) >= 0.0) {
                        m_predecessors_in_fringe[edge.destination].fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }
#endif

#if defined(Q_ARRAY)
            if (states[node] != state::fringe) {
                states[node] = state::fringe;
                fringe.push_back(node);
            }
#elif defined(Q_HEAP)
            if (states[node] != state::fringe) {
                states[node] = state::fringe;
                distance_queue_handles[node] = distance_queue.push(node);
#if defined(CRAUSER_IN) || defined(CRAUSER_INDYN)
                crauser_in_queue_handles[node] = crauser_in_queue.push(node);
#endif
#if defined(CRAUSER_OUT) || defined(CRAUSER_OUTDYN)
                crauser_out_queue_handles[node] = crauser_out_queue.push(node);
#endif
#if defined(TRAFF)
                traff_queue_handles[node] = traff_queue.push(node);
#endif
            } else {
                distance_queue.update(distance_queue_handles[node]);
#if defined(CRAUSER_IN) || defined(CRAUSER_INDYN)
                crauser_in_queue.update(crauser_in_queue_handles[node]);
#endif
#if defined(CRAUSER_OUT) || defined(CRAUSER_OUTDYN)
                crauser_out_queue.update(crauser_out_queue_handles[node]);
#endif
#if defined(TRAFF)
                traff_queue.update(traff_queue_handles[node]);
#endif
            }
#endif
        }
    };

    // Helper to relax a node either locally or remote.
    auto relax_node = [&](size_t node) {
        states[node] = state::settled;

        group_seen_distances[node + my_nodes_start].store(-INFINITY, std::memory_order_relaxed);

#if defined(TRAFF)
        for (const edge& edge : thread_edges_by_node[node]) {
            if (group_seen_distances[edge.destination].load(std::memory_order_relaxed) >= 0.0) {
                m_predecessors_in_fringe[edge.destination].fetch_sub(1, std::memory_order_relaxed);
            }
        }
#endif

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
        perf.next_timeblock("thresholds");

#if defined(CRAUSER_IN) || defined(CRAUSER_INDYN) || defined(TRAFF)
#if defined(Q_ARRAY)
        const double my_in_threshold =
            fringe.empty() ? INFINITY : distances[*std::max_element(fringe.begin(), fringe.end(), cmp_distance)];
#elif defined(Q_HEAP)
        const double my_in_threshold = distance_queue.empty() ? INFINITY : distances[distance_queue.top()];
#endif

        threads.reduce_linear_collective(
            m_in_threshold, double(INFINITY), my_in_threshold, [](auto a, auto b) { return std::min(a, b); }, false);
        const double in_threshold = m_in_threshold.load(std::memory_order_relaxed);
        if (in_threshold == INFINITY) {
            break;
        }
#endif

#if defined(CRAUSER_OUT) || defined(CRAUSER_OUTDYN)
#if defined(Q_ARRAY)
        const auto out_iter = std::max_element(fringe.begin(), fringe.end(), cmp_crauser_out);
        const double my_out_threshold =
            fringe.empty() ? INFINITY : (distances[*out_iter] + min_outgoing_value(*out_iter));
#elif defined(Q_HEAP)
        const double my_out_threshold =
            crauser_out_queue.empty()
                ? INFINITY
                : (distances[crauser_out_queue.top()] + min_outgoing_value(crauser_out_queue.top()));
#endif

        threads.reduce_linear_collective(
            m_out_threshold, double(INFINITY), my_out_threshold, [](auto a, auto b) { return std::min(a, b); }, false);
        const double out_threshold = m_out_threshold.load(std::memory_order_relaxed);
        if (out_threshold == INFINITY) {
            break;
        }
#endif

        perf.next_timeblock("fill_todo");
        todo.clear();

#if defined(Q_ARRAY)
        // Helper function to check if a node may (not) be settled.
        auto can_be_settled = [&](size_t node) {
#if defined(CRAUSER_IN) || defined(CRAUSER_INDYN)
            if (distances[node] - min_incoming_value(node) <= in_threshold)
                return false;
#endif
#if defined(CRAUSER_OUT) || defined(CRAUSER_OUTDYN)
            if (distances[node] <= out_threshold)
                return false;
#endif
#if defined(TRAFF)
            if (distances[node] - min_pred_pred[node] <= in_threshold &&
                m_predecessors_in_fringe[node + my_nodes_start].load(std::memory_order_relaxed) == 0)
                return false;
#endif
            return true;
        };

        // Move all nodes that can be settled to the end
        auto settle_start = std::partition(fringe.begin(), fringe.end(), can_be_settled);
        todo.insert(todo.begin(), settle_start, fringe.end());
        fringe.erase(settle_start, fringe.end());
#elif defined(Q_HEAP)
#if defined(CRAUSER_IN) || defined(CRAUSER_INDYN)
        while (!crauser_in_queue.empty() &&
               distances[crauser_in_queue.top()] - min_incoming_value(crauser_in_queue.top()) <= in_threshold) {
            size_t node = crauser_in_queue.top();
            crauser_in_queue.pop();
            todo.push_back(node);
            distance_queue.erase(distance_queue_handles[node]);
#if defined(CRAUSER_OUT) || defined(CRAUSER_OUTDYN)
            crauser_out_queue.erase(crauser_out_queue_handles[node]);
#endif
#if defined(TRAFF)
            traff_queue.erase(traff_queue_handles[node]);
#endif
        }
#endif

#if defined(CRAUSER_OUT) || defined(CRAUSER_OUTDYN)
        while (!distance_queue.empty() && distances[distance_queue.top()] <= out_threshold) {
            size_t node = distance_queue.top();
            distance_queue.pop();
            todo.push_back(node);
            crauser_out_queue.erase(crauser_out_queue_handles[node]);
#if defined(CRAUSER_IN) || defined(CRAUSER_INOUT)
            crauser_in_queue.erase(crauser_in_queue_handles[node]);
#endif
#if defined(TRAFF)
            traff_queue.erase(traff_queue_handles[node]);
#endif
        }
#endif

#if defined(TRAFF)
        traff_candidates.clear();
        while (!traff_queue.empty() &&
               distances[traff_queue.top()] - min_pred_pred[traff_queue.top()] <= in_threshold) {
            size_t node = traff_queue.top();
            traff_queue.pop();
            traff_candidates.push_back(node);
            distance_queue.erase(distance_queue_handles[node]);
#if defined(CRAUSER_OUT) || defined(CRAUSER_OUTDYN)
            crauser_out_queue.erase(crauser_out_queue_handles[node]);
#endif
#if defined(CRAUSER_IN) || defined(CRAUSER_INOUT)
            crauser_in_queue.erase(crauser_in_queue_handles[node]);
#endif
        }
        for (size_t node : traff_candidates) {
            if (m_predecessors_in_fringe[node + my_nodes_start].load(std::memory_order_relaxed) == 0) {
                todo.push_back(node);
            } else {
                distance_queue_handles[node] = distance_queue.push(node);
#if defined(CRAUSER_IN) || defined(CRAUSER_INDYN)
                crauser_in_queue_handles[node] = crauser_in_queue.push(node);
#endif
#if defined(CRAUSER_OUT) || defined(CRAUSER_OUTDYN)
                crauser_out_queue_handles[node] = crauser_out_queue.push(node);
#endif
#if defined(TRAFF)
                traff_queue_handles[node] = traff_queue.push(node);
#endif
            }
        }
#endif
#endif

#if defined(TRAFF)
        // Only Träff's criteria needs a barrier at this point in time, because it reads from the
        // global array m_predecessors_in_fringe, which is going to be changed in the relax
        // step.
        perf.next_timeblock("fill_todo_traff_barrier");
        threads.barrier_collective(true);
#endif

        perf.next_timeblock("relax");
        for (size_t node : todo) {
            relax_node(node);
        }

        perf.next_timeblock("relax_barrier");
        threads.barrier_collective(true);

        perf.next_timeblock("relax_inbox");
        my_relaxations.for_each(
            [&](const relaxation& r) { settle_edge(r.node - my_nodes_start, r.predecessor, r.distance); });
        my_relaxations.clear();

        perf.next_timeblock("relax_inbox_barrier");
        threads.barrier_collective(true);

#if defined(CRAUSER_OUTDYN) || defined(CRAUSER_INDYN)
        perf.next_timeblock("update_dynamic");
        for (size_t n = 0; n < my_nodes_count; ++n) {
            if (states[n] != state::settled) {
#if defined(CRAUSER_INDYN)
                if (min_incoming[n].cost != INFINITY &&
                    group_seen_distances[min_incoming[n].node].load(std::memory_order_relaxed) < 0.0) {
                    node_cost result{size_t(-1), INFINITY};
                    double min_cost = min_incoming[n].cost;
                    for (const edge& e : thread_in_edges_by_node[n]) {
                        if (e.cost < result.cost && e.cost >= min_cost &&
                            group_seen_distances[e.source].load(std::memory_order_relaxed) >= 0.0) {
                            result = node_cost{e.source, e.cost};
                        }
                    }
                    min_incoming[n] = result;
#if defined(Q_HEAP)
                    if (states[n] == state::fringe) {
                        crauser_in_queue.update(crauser_in_queue_handles[n]);
                    }
#endif
                }
#endif
#if defined(CRAUSER_OUTDYN)
                if (min_outgoing[n].cost != INFINITY &&
                    group_seen_distances[min_outgoing[n].node].load(std::memory_order_relaxed) < 0.0) {
                    node_cost result{size_t(-1), INFINITY};
                    double min_cost = min_outgoing[n].cost;
                    for (const edge& e : thread_edges_by_node[n]) {
                        if (e.cost < result.cost && e.cost >= min_cost &&
                            group_seen_distances[e.destination].load(std::memory_order_relaxed) >= 0.0) {
                            result = node_cost{e.destination, e.cost};
                        }
                    }
                    min_outgoing[n] = result;
#if defined(Q_HEAP)
                    if (states[n] == state::fringe) {
                        crauser_out_queue.update(crauser_out_queue_handles[n]);
                    }
#endif
                }
#endif
            }
        }

        perf.next_timeblock("update_dynamic_barrier");
        threads.barrier_collective(true);
#endif
    }

    perf.end_timeblock();
    threads.single_collective([&] { m_perf = carray<perf_counter>(threads.thread_count()); }, true);
    m_perf[thread_rank] = std::move(perf);
}
