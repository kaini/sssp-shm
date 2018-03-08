#include "by_edges_sssp.hpp"
#include "thread_local_allocator.hpp"
#include <boost/heap/fibonacci_heap.hpp>
#include <chrono>
#include <iostream>

#if defined(CRAUSER_IN) && defined(CRAUSER_INDYN)
#error CRAUSER_IN and CRAUSER_INDYN cannot be combined
#endif

#if defined(CRAUSER_OUT) && defined(CRAUSER_OUTDYN)
#error CRAUSER_OUT and CRAUSER_OUTDYN cannot be combined
#endif

using namespace boost::heap;

namespace {
enum class state : unsigned char { unexplored, fringe, settled };
}

void sssp::by_edges_sssp::run_collective(thread_group& threads,
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
    BOOST_ASSERT(node_count == thread_edges_by_node.size());
    BOOST_ASSERT(node_count == out_thread_distances.size());
    BOOST_ASSERT(node_count == out_thread_predecessors.size());

    double time = 0;
    auto distances = out_thread_distances;
    auto predecessors = out_thread_predecessors;
    carray<state> states(node_count, state::unexplored);
    carray<int> last_updated(node_count, INT_MIN);
    std::vector<size_t, thread_local_allocator<size_t>> updated;
    std::vector<size_t, thread_local_allocator<size_t>> todo;

    auto cmp_distance = [&](size_t a, size_t b) { return distances[a] > distances[b]; };
    using distance_queue_t =
        fibonacci_heap<size_t, compare<decltype(cmp_distance)>, allocator<thread_local_allocator<size_t>>>;
    distance_queue_t distance_queue(cmp_distance);
    carray<distance_queue_t::handle_type> distance_queue_handles(node_count);

    threads.single_collective([&] {
        m_global_distances = carray<std::atomic<double>>(node_count);
        m_global_updated = carray<size_t>(node_count * threads.thread_count());
        m_global_updated_at.store(0, std::memory_order_relaxed);
    });

#if defined(CRAUSER_OUT)
    carray<double> min_outgoing(node_count);
    auto min_outgoing_value = [&](size_t n) { return min_outgoing[n]; };
    auto cmp_crauser_out = [&](size_t a, size_t b) {
        return distances[a] + min_outgoing_value(a) > distances[b] + min_outgoing_value(b);
    };

    using crauser_out_queue_t =
        fibonacci_heap<size_t, compare<decltype(cmp_crauser_out)>, allocator<thread_local_allocator<size_t>>>;
    crauser_out_queue_t crauser_out_queue(cmp_crauser_out);
    carray<crauser_out_queue_t::handle_type> crauser_out_queue_handles(node_count);

    for (size_t i = threads.chunk_start(thread_rank, node_count), end = i + threads.chunk_size(thread_rank, node_count);
         i < end;
         ++i) {
        m_global_distances[i].store(INFINITY, std::memory_order_relaxed);
    }
    threads.barrier_collective();

    for (size_t node = 0; node < node_count; ++node) {
        double min_outgoing = INFINITY;
        for (const edge& edge : thread_edges_by_node[node]) {
            if (edge.cost < min_outgoing) {
                min_outgoing = edge.cost;
            }
        }
        if (min_outgoing != INFINITY) {
            atomic_min(m_global_distances[node], min_outgoing);
        }
    }
    threads.barrier_collective();

    for (size_t node = 0; node < node_count; ++node) {
        min_outgoing[node] = m_global_distances[node].load(std::memory_order_relaxed);
    }
    threads.barrier_collective();
#endif

    for (size_t i = threads.chunk_start(thread_rank, node_count), end = i + threads.chunk_size(thread_rank, node_count);
         i < end;
         ++i) {
        m_global_distances[i].store(INFINITY, std::memory_order_relaxed);
    }
    threads.barrier_collective();

    distances[0] = 0.0;
    predecessors[0] = -1;
    states[0] = state::fringe;
    distance_queue_handles[0] = distance_queue.push(0);
#if defined(CRAUSER_OUT)
    crauser_out_queue_handles[0] = crauser_out_queue.push(0);
#endif
    m_global_distances[0].store(0.0, std::memory_order_relaxed);
    threads.barrier_collective();

    int phase;
    for (phase = 0;; ++phase) {
        const auto start = std::chrono::steady_clock::now();

        m_global_updated_at.store(0, std::memory_order_relaxed);
        updated.clear();
        todo.clear();

#if defined(CRAUSER_OUT)
        if (crauser_out_queue.empty()) {
            break;
        }
        const double out_threshold = distances[crauser_out_queue.top()] + min_outgoing_value(crauser_out_queue.top());
        while (!distance_queue.empty() && distances[distance_queue.top()] <= out_threshold) {
            size_t node = distance_queue.top();
            todo.push_back(node);
            distance_queue.pop();
            crauser_out_queue.erase(crauser_out_queue_handles[node]);
        }
#endif

        auto end = std::chrono::steady_clock::now();
        time += (end - start).count() / 1000000000.0;

        for (size_t node : todo) {
            BOOST_ASSERT(states[node] == state::fringe);
            states[node] = state::settled;
            for (const edge& e : thread_edges_by_node[node]) {
                if (states[e.destination] != state::settled && distances[node] + e.cost < distances[e.destination]) {
                    distances[e.destination] = distances[node] + e.cost;
                    predecessors[e.destination] = node;
                    if (states[e.destination] != state::fringe) {
                        states[e.destination] = state::fringe;
                        distance_queue_handles[e.destination] = distance_queue.push(e.destination);
#if defined(CRAUSER_OUT)
                        crauser_out_queue_handles[e.destination] = crauser_out_queue.push(e.destination);
#endif
                    } else {
                        distance_queue.update(distance_queue_handles[e.destination]);
#if defined(CRAUSER_OUT)
                        crauser_out_queue.update(crauser_out_queue_handles[e.destination]);
#endif
                    }
                    if (last_updated[e.destination] != phase) {
                        last_updated[e.destination] = phase;
                        updated.push_back(e.destination);
                    }
                }
            }
        }

        threads.barrier_collective();

        for (size_t node : updated) {
            BOOST_ASSERT(states[node] == state::fringe);
            atomic_min(m_global_distances[node], distances[node]);
            m_global_updated[m_global_updated_at.fetch_add(1, std::memory_order_relaxed)] = node;
        }

        threads.barrier_collective();

        for (size_t i = 0, end = m_global_updated_at.load(std::memory_order_relaxed); i < end; ++i) {
            size_t node = m_global_updated[i];
            BOOST_ASSERT(states[node] != state::settled);
            if (last_updated[node] != -phase) {
                last_updated[node] = -phase;
                double global_distance = m_global_distances[node].load(std::memory_order_relaxed);
                if (global_distance < distances[node]) {
                    distances[node] = global_distance;
                    predecessors[node] = -2;
                    if (states[node] != state::fringe) {
                        states[node] = state::fringe;
                        distance_queue_handles[node] = distance_queue.push(node);
#if defined(CRAUSER_OUT)
                        crauser_out_queue_handles[node] = crauser_out_queue.push(node);
#endif
                    } else {
                        distance_queue.update(distance_queue_handles[node]);
#if defined(CRAUSER_OUT)
                        crauser_out_queue.update(crauser_out_queue_handles[node]);
#endif
                    }
                }
            }
        }

        threads.barrier_collective();
    }

    threads.critical_section([&] { std::cout << time << std::endl; });
}
