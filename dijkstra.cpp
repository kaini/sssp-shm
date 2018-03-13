#include "dijkstra.hpp"
#include "carray.hpp"
#include "thread_local_allocator.hpp"
#include <algorithm>
#include <boost/heap/fibonacci_heap.hpp>
#include <cmath>

using namespace boost::heap;

namespace {
enum class state : unsigned char {
    unexplored,
    fringe,
    settled,
};
}

void sssp::dijkstra(array_slice<const array_slice<const edge>> edges_by_node,
                    array_slice<double> distances,
                    array_slice<size_t> predecessors,
                    array_slice<const bool> prefill_fringe) {
    size_t node_count = edges_by_node.size();

    auto cmp_distance = [&](size_t a, size_t b) { return distances[a] > distances[b]; };
    using distance_queue_t =
        fibonacci_heap<size_t, compare<decltype(cmp_distance)>, allocator<thread_local_allocator<size_t>>>;
    distance_queue_t distance_queue(cmp_distance);
    std::vector<distance_queue_t::handle_type> distance_queue_handles(node_count);

    std::vector<state> states(node_count, state::unexplored);
    if (prefill_fringe.size()) {
        for (size_t n = 0; n < node_count; ++n) {
            if (prefill_fringe[n]) {
                states[n] = state::fringe;
                distance_queue_handles[n] = distance_queue.push(n);
            }
        }
    } else {
        states[0] = state::fringe;
        distance_queue_handles[0] = distance_queue.push(0);
    }

    while (!distance_queue.empty()) {
        size_t node = distance_queue.top();
        distance_queue.pop();
        states[node] = state::settled;

        for (const edge& edge : edges_by_node[node]) {
            if (states[edge.destination] != state::settled &&
                distances[node] + edge.cost < distances[edge.destination]) {
                distances[edge.destination] = distances[node] + edge.cost;
                predecessors[edge.destination] = node;
                if (states[edge.destination] == state::unexplored) {
                    states[edge.destination] = state::fringe;
                    distance_queue_handles[edge.destination] = distance_queue.push(edge.destination);
                } else {
                    distance_queue.update(distance_queue_handles[edge.destination]);
                }
            }
        }
    }
}
