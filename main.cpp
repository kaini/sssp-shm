#include <atomic>
#include <boost/heap/pairing_heap.hpp>
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

namespace {

struct node;

struct cmp_distance {
    bool operator()(const node* a, const node* b) const;
};

using distance_queue_t = boost::heap::pairing_heap<node*, boost::heap::compare<cmp_distance>>;

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
    std::vector<edge> edges;

    double distance = INFINITY;
    size_t predecessor = -1;
    ::state state = ::state::unexplored;

    distance_queue_t::handle_type distance_queue_handle;
};

bool cmp_distance::operator()(const node* a, const node* b) const {
    return a->distance > b->distance;
}

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

static std::tuple<bool, double> validate(const std::vector<std::vector<node>>& nodes, size_t node_count) {
    std::vector<double> distances(node_count, INFINITY);
    std::vector<size_t> predecessors(node_count, -1);
    std::vector<state> states(node_count, state::unexplored);

    auto cmp_distance = [&](size_t a, size_t b) { return distances[a] > distances[b]; };
    using distance_queue_t = boost::heap::pairing_heap<size_t, boost::heap::compare<decltype(cmp_distance)>>;
    std::vector<distance_queue_t::handle_type> distance_queue_handles(node_count);

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
static void generate_graph(std::vector<node>& result,
                           size_t thread_num,
                           size_t thread_count,
                           unsigned int seed,
                           size_t node_count,
                           double edge_chance) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<size_t> node_dist(0, node_count - 1);
    std::binomial_distribution<size_t> edge_count_dist(node_count, edge_chance);
    std::uniform_real_distribution<double> edge_cost_dist(0.0, 1.0);
    for (size_t node_id = thread_num; node_id < node_count; node_id += thread_count) {
        node& node = result[node_id / thread_count];
        node.id = node_id;
        size_t edge_count = edge_count_dist(rng);
        // TODO waaaay to many memory allocations
        node.edges.resize(edge_count);
        for (auto& edge : node.edges) {
            edge.source = node.id;
            edge.destination = node_dist(rng);
            edge.cost = edge_cost_dist(rng);
        }
    }
}

int main(int argc, char* argv[]) {
    const size_t node_count = 1000001;
    const double edge_chance = 10.0 / node_count;
    const int seed = 42;

    // Maps thread_num -> local_node_id -> node
    std::vector<std::vector<node>> global_nodes;
    // Maps source thread_num -> destination thread_num -> node_id
    std::vector<std::vector<std::vector<relaxation>>> global_settle_todo;

    std::unique_ptr<std::atomic<double>> global_min_distance(new std::atomic<double>());
    std::unique_ptr<std::atomic<double>> global_max_time(new std::atomic<double>(INFINITY));

#pragma omp parallel
    {
        const size_t thread_count = omp_get_num_threads();
        const size_t thread_num = omp_get_thread_num();

#pragma omp single
        {
            global_nodes.resize(thread_count);
            global_settle_todo.resize(thread_count);
        }

        const size_t my_node_count = node_count / thread_count + ((thread_num < node_count % thread_count) ? 1 : 0);

        std::vector<node>& my_nodes = global_nodes[thread_num];
        std::vector<std::vector<relaxation>>& my_settle_todo = global_settle_todo[thread_num];
        my_nodes.resize(my_node_count);
        my_settle_todo.resize(thread_count);

        generate_graph(
            my_nodes, thread_num, thread_count, static_cast<unsigned int>(thread_num * seed), node_count, edge_chance);

        const auto start_time = std::chrono::steady_clock::now();

        distance_queue_t distance_queue;

        if (thread_num == 0) {
            my_nodes[0].distance = 0.0;
            my_nodes[0].predecessor = -1;
            my_nodes[0].state = state::fringe;
            my_nodes[0].distance_queue_handle = distance_queue.push(&my_nodes[0]);
        }

        for (int phase = 0;; ++phase) {
            double min_distance = reduce(*global_min_distance,
                                         distance_queue.empty() ? INFINITY : distance_queue.top()->distance,
                                         [](double a, double b) { return std::min(a, b); });
            if (min_distance == INFINITY) {
                break;
            }

#pragma omp barrier

            for (auto& todo : my_settle_todo) {
                todo.clear();
            }
            while (!distance_queue.empty() && distance_queue.top()->distance <= min_distance) {
                node& node = *distance_queue.top();
                distance_queue.pop();
                node.state = state::settled;
                for (const auto& edge : node.edges) {
                    my_settle_todo[edge.destination % thread_count].push_back(
                        relaxation{edge.destination, node.id, node.distance + edge.cost});
                }
            }

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
                        } else {
                            distance_queue.update(node.distance_queue_handle);
                        }
                    }
                }
            }
        }

        const auto end_time = std::chrono::steady_clock::now();
        reduce(*global_max_time, (end_time - start_time).count() / 1000000000.0, [](double a, double b) {
            return std::max(a, b);
        });
    }

    double seq_time;
    bool valid;
    std::tie(valid, seq_time) = validate(global_nodes, node_count);

    std::cout << "Par time: " << *global_max_time << "\n";
    std::cout << "Seq time: " << seq_time << "\n";

    if (valid) {
        return EXIT_SUCCESS;
    } else {
        return EXIT_FAILURE;
    }
}
