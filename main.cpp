#include "array_slice.hpp"
#include "carray.hpp"
#include "thread_local_allocator.hpp"
#include <atomic>
#include <boost/heap/fibonacci_heap.hpp>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <omp.h>
#include <random>
#include <vector>

using namespace sssp;
using namespace boost::heap;

namespace {

struct node;

struct cmp_distance {
    bool operator()(const node* a, const node* b) const;
};
using thread_local_distance_queue =
    fibonacci_heap<node*, compare<cmp_distance>, allocator<thread_local_allocator<node*>>>;

#if defined(CRAUSER)
struct cmp_crauser_in {
    bool operator()(const node* a, const node* b) const;
};
using thread_local_crauser_in_queue =
    fibonacci_heap<node*, compare<cmp_crauser_in>, allocator<thread_local_allocator<node*>>>;
#endif

struct edge {
    size_t source;
    size_t destination;
    double cost;
};

struct relaxation {
    size_t node;
    size_t predecessor;
    double distance;
};

enum class state : unsigned char {
    unexplored,
    fringe,
    settled,
};

struct node {
    size_t id = -1;
    array_slice<const edge> edges;

    double distance = INFINITY;
    size_t predecessor = -1;
    ::state state = ::state::unexplored;
    thread_local_distance_queue::handle_type distance_queue_handle;

#if defined(CRAUSER)
    std::mutex min_incoming_mutex; // TODO this should be low contention, use atomics?
    double min_incoming = INFINITY;
    thread_local_crauser_in_queue::handle_type crauser_in_queue_handle;
#endif
};

bool cmp_distance::operator()(const node* a, const node* b) const {
    return a->distance > b->distance;
}

#if defined(CRAUSER)
bool cmp_crauser_in::operator()(const node* a, const node* b) const {
    return a->distance - a->min_incoming > b->distance - b->min_incoming;
}
#endif

} // namespace

// TODO this is O(t) or something, this can be done better ...
template <typename T, typename Op> static double reduce(std::atomic<T>& storage, T contribution, Op&& op) {
    storage.store(contribution, std::memory_order_relaxed);
#pragma omp barrier
    T current_value = storage.load(std::memory_order_relaxed);
    T wanted_value = op(current_value, contribution);
    while (current_value != wanted_value &&
           !storage.compare_exchange_weak(
               current_value, wanted_value, std::memory_order_relaxed, std::memory_order_relaxed)) {
        wanted_value = op(current_value, contribution);
    }
#pragma omp barrier
    return storage.load(std::memory_order_relaxed);
}

static std::tuple<bool, double> validate(const carray<carray<node>>& nodes, size_t node_count) {
    carray<double> distances(node_count, INFINITY);
    carray<size_t> predecessors(node_count, -1);
    carray<state> states(node_count, state::unexplored);

    auto cmp_distance = [&](size_t a, size_t b) { return distances[a] > distances[b]; };
    using distance_queue_t =
        fibonacci_heap<size_t, compare<decltype(cmp_distance)>, allocator<thread_local_allocator<size_t>>>;
    carray<distance_queue_t::handle_type> distance_queue_handles(node_count);

    const auto start_time = std::chrono::steady_clock::now();

    distance_queue_t distance_queue(cmp_distance);

    distances[0] = 0.0;
    predecessors[0] = -1;
    states[0] = state::fringe;
    distance_queue_handles[0] = distance_queue.push(0);

    while (!distance_queue.empty()) {
        size_t node = distance_queue.top();
        distance_queue.pop();
        states[node] = state::settled;

        for (const edge& edge : nodes[node % nodes.size()][node / nodes.size()].edges) {
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

    const auto end_time = std::chrono::steady_clock::now();

    bool valid = true;
    for (size_t i = 0; i < node_count; ++i) {
        if (nodes[i % nodes.size()][i / nodes.size()].distance != distances[i]) {
            std::cerr << "Invalid distance for node " << i << " " << nodes[i % nodes.size()][i / nodes.size()].distance
                      << "!=" << distances[i] << "\n";
            valid = false;
        }
        if (nodes[i % nodes.size()][i / nodes.size()].predecessor != predecessors[i]) {
            std::cerr << "Invalid predecessor for node " << i << " "
                      << nodes[i % nodes.size()][i / nodes.size()].predecessor << "!=" << predecessors[i] << "\n";
            valid = false;
        }
    }

    return std::make_tuple(valid, (end_time - start_time).count() / 1000000000.0);
}

// This allows double edges and self edges, but it does not matter too much.
static void generate_graph(carray<edge>& edges,
                           carray<node>& nodes,
                           size_t thread_num,
                           size_t thread_count,
                           unsigned int seed,
                           size_t my_node_count,
                           size_t node_count,
                           double edge_chance) {
    nodes = carray<node>(my_node_count);

    std::mt19937 edge_count_rng(seed);
    std::binomial_distribution<size_t> edge_count_dist(node_count, edge_chance);
    size_t edge_count = 0;
    for (size_t node_id = thread_num; node_id < node_count; node_id += thread_count) {
        edge_count += edge_count_dist(edge_count_rng);
    }
    edges = carray<edge>(edge_count);

    edge_count_rng.seed(seed);
    edge_count_dist.reset();
    size_t edge_at = 0;
    std::mt19937 other_rng(seed + 1);
    std::uniform_int_distribution<size_t> node_dist(0, node_count - 1);
    std::uniform_real_distribution<double> edge_cost_dist(0.0, 1.0);
    for (size_t node_id = thread_num; node_id < node_count; node_id += thread_count) {
        node& node = nodes[node_id / thread_count];
        node.id = node_id;

        size_t node_edge_count = edge_count_dist(edge_count_rng);
        node.edges = array_slice<const edge>(&edges[edge_at], node_edge_count);
        for (auto& e : array_slice<edge>(&edges[edge_at], node_edge_count)) {
            e.source = node.id;
            e.destination = node_dist(other_rng);
            e.cost = edge_cost_dist(other_rng);
        }
        edge_at += node_edge_count;
    }
    BOOST_ASSERT(edge_count == edge_at);
}

int main(int argc, char* argv[]) {
    const size_t node_count = 3000001;
    const double edge_chance = 10.0 / node_count;
    const int seed = 42;
    const size_t thread_count = omp_get_max_threads();

    std::cout << "Thread count: " << thread_count << "\n";

    // Maps thread_num -> all edges
    // The nodes contain a pointers into the edges array
    carray<carray<edge>> global_edges(thread_count);
    // Maps thread_num -> local_node_id -> node
    carray<carray<node>> global_nodes(thread_count);
    // Maps source thread_num -> destination thread_num -> node_id
    carray<carray<std::vector<relaxation>>> global_settle_todo(thread_count);

    std::unique_ptr<std::atomic<double>> global_min_distance(new std::atomic<double>());
    std::unique_ptr<std::atomic<double>> global_max_time(new std::atomic<double>(INFINITY));
    std::unique_ptr<std::atomic<double>> global_max_init_time(new std::atomic<double>(INFINITY));

#pragma omp parallel num_threads(int(thread_count))
    {
        const size_t thread_num = omp_get_thread_num();
        const size_t my_node_count = node_count / thread_count + ((thread_num < node_count % thread_count) ? 1 : 0);

        carray<edge>& my_edges = global_edges[thread_num];
        carray<node>& my_nodes = global_nodes[thread_num];
        generate_graph(my_edges,
                       my_nodes,
                       thread_num,
                       thread_count,
                       static_cast<unsigned int>(thread_num * seed),
                       my_node_count,
                       node_count,
                       edge_chance);

        carray<std::vector<relaxation>>& my_settle_todo = global_settle_todo[thread_num];
        my_settle_todo = carray<std::vector<relaxation>>(thread_count);

#pragma omp barrier
        const auto start_time = std::chrono::steady_clock::now();

#if defined(CRAUSER)
        // Preprocessing required for Crauser's criteria
        for (const edge& edge : my_edges) {
            ::node& dest_node = global_nodes[edge.destination % thread_count][edge.destination / thread_count];
            std::lock_guard<std::mutex> lock(dest_node.min_incoming_mutex);
            dest_node.min_incoming = std::min(dest_node.min_incoming, edge.cost);
        }
#pragma omp barrier
#endif

        thread_local_distance_queue distance_queue;
#if defined(CRAUSER)
        thread_local_crauser_in_queue crauser_in_queue;
#endif

        // Helper function to put an edge into the todo list
        auto into_todo = [&](node& node) {
            distance_queue.erase(node.distance_queue_handle);
#if defined(CRAUSER)
            crauser_in_queue.erase(node.crauser_in_queue_handle);
#endif
            node.state = state::settled;
            for (const auto& edge : node.edges) {
                my_settle_todo[edge.destination % thread_count].push_back(
                    relaxation{edge.destination, node.id, node.distance + edge.cost});
            }
        };

        if (thread_num == 0) {
            my_nodes[0].distance = 0.0;
            my_nodes[0].predecessor = -1;
            my_nodes[0].state = state::fringe;
            my_nodes[0].distance_queue_handle = distance_queue.push(&my_nodes[0]);
#if defined(CRAUSER)
            my_nodes[0].crauser_in_queue_handle = crauser_in_queue.push(&my_nodes[0]);
#endif
        }

        const auto init_end = std::chrono::steady_clock::now();

        for (int phase = 0;; ++phase) {
            for (auto& todo : my_settle_todo) {
                todo.clear();
            }

#if defined(DIJKSTRA)
            double min_distance = reduce(*global_min_distance,
                                         distance_queue.empty() ? INFINITY : distance_queue.top()->distance,
                                         [](double a, double b) { return std::min(a, b); });
            if (min_distance == INFINITY) {
                break;
            }
            while (!distance_queue.empty() && distance_queue.top()->distance <= min_distance) {
                into_todo(*distance_queue.top());
            }
#elif defined(CRAUSER)
            double min_distance = reduce(*global_min_distance,
                                         distance_queue.empty() ? INFINITY : distance_queue.top()->distance,
                                         [](double a, double b) { return std::min(a, b); });
            if (min_distance == INFINITY) {
                break;
            }
            while (!crauser_in_queue.empty() &&
                   crauser_in_queue.top()->distance - crauser_in_queue.top()->min_incoming <= min_distance) {
                into_todo(*crauser_in_queue.top());
            }
#endif

#pragma omp barrier

            for (size_t t = 0; t < thread_count; ++t) {
                size_t tt = (t + thread_num) % thread_count;
                for (const auto& todo : global_settle_todo[tt][thread_num]) {
                    node& node = my_nodes[todo.node / thread_count];
                    if (node.state != state::settled && todo.distance < node.distance) {
                        node.distance = todo.distance;
                        node.predecessor = todo.predecessor;
                        if (node.state == state::unexplored) {
                            node.state = state::fringe;
                            node.distance_queue_handle = distance_queue.push(&node);
#if defined(CRAUSER)
                            node.crauser_in_queue_handle = crauser_in_queue.push(&node);
#endif
                        } else {
                            distance_queue.update(node.distance_queue_handle);
#if defined(CRAUSER)
                            crauser_in_queue.update(node.crauser_in_queue_handle);
#endif
                        }
                    }
                }
            }

#pragma omp barrier
        }

        const auto end_time = std::chrono::steady_clock::now();
        reduce(*global_max_time, (end_time - start_time).count() / 1000000000.0, [](double a, double b) {
            return std::max(a, b);
        });
        reduce(*global_max_init_time, (init_end - start_time).count() / 1000000000.0, [](double a, double b) {
            return std::max(a, b);
        });
    }

    bool valid;
    double seq_time;
    std::tie(valid, seq_time) = validate(global_nodes, node_count);

    std::cout << "Par time: " << *global_max_time << " (incl. " << *global_max_init_time << " init time)\n";
    std::cout << "Seq time: " << seq_time << "\n";
    std::cout << "Speedup: " << (seq_time / *global_max_time)
              << "  Efficiency: " << (seq_time / *global_max_time / thread_count) << "\n";

    if (valid) {
        return EXIT_SUCCESS;
    } else {
        return EXIT_FAILURE;
    }
}
