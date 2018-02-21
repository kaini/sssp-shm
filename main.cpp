#include "array_slice.hpp"
#include "carray.hpp"
#include "own_queues_sssp.hpp"
#include "thread_local_allocator.hpp"
#include <atomic>
#include <boost/heap/fibonacci_heap.hpp>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <hwloc.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <omp.h>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if HWLOC_API_VERSION < 0x00020000
#error hwloc>=2.0.0 is required
#endif

using namespace sssp;
using namespace boost::heap;

namespace {

static std::string hwloc_type_to_string(hwloc_obj_type_t type) {
    switch (type) {
        case HWLOC_OBJ_MACHINE:
            return "machine";
        case HWLOC_OBJ_PACKAGE:
            return "package";
        case HWLOC_OBJ_CORE:
            return "core";
        case HWLOC_OBJ_PU:
            return "pu";
        case HWLOC_OBJ_L1CACHE:
            return "L1";
        case HWLOC_OBJ_L2CACHE:
            return "L2";
        case HWLOC_OBJ_L3CACHE:
            return "L3";
        case HWLOC_OBJ_L4CACHE:
            return "L4";
        case HWLOC_OBJ_L5CACHE:
            return "L5";
        case HWLOC_OBJ_L1ICACHE:
            return "L1i";
        case HWLOC_OBJ_L2ICACHE:
            return "L2i";
        case HWLOC_OBJ_L3ICACHE:
            return "L3i";
        case HWLOC_OBJ_GROUP:
            return "group";
        case HWLOC_OBJ_NUMANODE:
            return "numanode";
        case HWLOC_OBJ_BRIDGE:
            return "bridge";
        case HWLOC_OBJ_PCI_DEVICE:
            return "pci-device";
        case HWLOC_OBJ_OS_DEVICE:
            return "os-device";
        case HWLOC_OBJ_MISC:
            return "misc";
        default:
            return "???";
    }
}

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
    node() : new_distance(INFINITY) {}

    size_t id = -1;
    array_slice<const edge> edges;

    double distance = INFINITY;
    std::atomic<double> new_distance;
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

static std::tuple<bool, double> validate(array_slice<array_slice<const edge>> nodes,
                                         array_slice<const result> reference) {
    size_t node_count = nodes.size();
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

        for (const edge& edge : nodes[node]) {
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
        if (reference[i].distance != distances[i]) {
            std::cerr << "Invalid distance for node " << i << " " << reference[i].distance << "!=" << distances[i]
                      << "\n";
            valid = false;
        }
        if (reference[i].predecessor != predecessors[i]) {
            std::cerr << "Invalid predecessor for node " << i << " " << reference[i].predecessor
                      << "!=" << predecessors[i] << "\n";
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
                           size_t node_count,
                           double edge_chance) {
    nodes = carray<node>(node_count);

    std::mt19937 edge_count_rng(seed);
    // TODO fix the probability!!!!
    std::binomial_distribution<size_t> edge_count_dist(node_count / thread_count, edge_chance);
    size_t edge_count = 0;
    for (size_t node_id = 0; node_id < node_count; ++node_id) {
        edge_count += edge_count_dist(edge_count_rng);
    }
    edges = carray<edge>(edge_count);

    edge_count_rng.seed(seed);
    edge_count_dist.reset();
    size_t edge_at = 0;
    std::mt19937 other_rng(seed + 1);
    std::uniform_int_distribution<size_t> node_dist(0, node_count - 1);
    std::uniform_real_distribution<double> edge_cost_dist(0.0, 1.0);
    for (size_t node_id = 0; node_id < node_count; ++node_id) {
        node& node = nodes[node_id];
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

// This allows double edges and self edges, but it does not matter too much.
static void generate_edges_collective(thread_group& threads,
                                      int thread_rank,
                                      size_t node_count,
                                      unsigned int seed,
                                      double edge_chance,
                                      carray<carray<edge>>& out_edges_by_rank,
                                      carray<array_slice<const edge>>& out_edges_by_node) {
    threads.single_collective([&] {
        out_edges_by_rank = carray<carray<edge>>(threads.thread_count());
        out_edges_by_node = carray<array_slice<const edge>>(node_count);
    });

    std::mt19937 edge_count_rng(seed);
    std::binomial_distribution<size_t> edge_count_dist(node_count, edge_chance);
    size_t edge_count = 0;
    threads.for_each_collective(thread_rank, out_edges_by_node, [&](array_slice<const edge>&) {
        edge_count += edge_count_dist(edge_count_rng);
    });
    carray<edge>& my_edges = out_edges_by_rank[thread_rank] = carray<edge>(edge_count);

    edge_count_rng.seed(seed);
    edge_count_dist.reset();
    size_t edge_at = 0;
    std::mt19937 other_rng(seed + 1);
    std::uniform_int_distribution<size_t> node_dist(0, node_count - 1);
    std::uniform_real_distribution<double> edge_cost_dist(0.0, 1.0);
    threads.for_each_with_index_collective(
        thread_rank, out_edges_by_node, [&](size_t i, array_slice<const edge>& edges) {
            size_t node_edge_count = edge_count_dist(edge_count_rng);
            edges = array_slice<const edge>(&my_edges[edge_at], node_edge_count);
            for (edge& e : array_slice<edge>(&my_edges[edge_at], node_edge_count)) {
                e.source = i;
                e.destination = node_dist(other_rng);
                e.cost = edge_cost_dist(other_rng);
            }
            edge_at += node_edge_count;
        });
    BOOST_ASSERT(edge_count == edge_at);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "args ...\n";
        return EXIT_FAILURE;
    }
    const int thread_count = atoi(argv[1]); // omp_get_max_threads();

    hwloc_topology_t topo;
    hwloc_topology_init(&topo);
    hwloc_topology_load(topo);

    std::cerr << "Topology: (assuming symmetric topology; * marks NUMA nodes)\n";
    int topo_depth = hwloc_topology_get_depth(topo);
    struct topo_layer {
        int count = 0;
        std::string name = "";
    };
    std::vector<topo_layer> effective_topo;
    for (int d = 0; d < topo_depth; ++d) {
        int depth_count = hwloc_get_nbobjs_by_depth(topo, d);
        if (effective_topo.empty() || depth_count > effective_topo.back().count) {
            effective_topo.emplace_back();
            effective_topo.back().count = depth_count;
        }
        if (!effective_topo.back().name.empty()) {
            effective_topo.back().name += "+";
        }
        effective_topo.back().name += hwloc_type_to_string(hwloc_get_depth_type(topo, d));
        if (hwloc_get_next_obj_by_depth(topo, d, nullptr)->memory_arity > 0) {
            effective_topo.back().name += "*";
        }
    }
    for (const auto& layer : effective_topo) {
        std::cerr << std::setw(4) << layer.count << "x " << layer.name << "\n";
    }

    const size_t node_count = 150000;
    const double edge_chance = 1000.0 / node_count;
    const int seed = 42;

    std::cout << "Thread count: " << thread_count << "\n";

#if 1
    std::vector<std::thread> thread_handles;
    thread_group threads(thread_count);
    carray<carray<edge>> edges_by_rank;
    carray<array_slice<const edge>> edges_by_node;
    carray<result> result_by_node(node_count);
    // shared_queue_sssp algorithm(node_count, edges_by_node);
    own_queues_sssp algorithm(node_count);
    for (int rank = 0; rank < thread_count; ++rank) {
        thread_handles.emplace_back(std::thread([&, rank] {
            hwloc_obj_t pu = hwloc_get_obj_by_type(topo, HWLOC_OBJ_PU, rank);
            if (!pu) {
                throw std::runtime_error("Could not find enough PUs.");
            }
            hwloc_obj_t core = hwloc_get_ancestor_obj_by_type(topo, HWLOC_OBJ_CORE, pu);
            if (!core) {
                throw std::runtime_error("Found a PU without a core.");
            }
            if (hwloc_set_cpubind(topo, core->cpuset, HWLOC_CPUBIND_THREAD | HWLOC_CPUBIND_STRICT) == -1) {
                threads.critical_section([] { std::cerr << "Could not bind a thread, ignoring ...\n"; });
            }

            generate_edges_collective(threads,
                                      rank,
                                      node_count,
                                      static_cast<unsigned int>(rank * seed),
                                      edge_chance,
                                      edges_by_rank,
                                      edges_by_node);
            algorithm.run_collective(threads, rank, edges_by_node, result_by_node);
        }));
    }
    for (auto& t : thread_handles) {
        t.join();
    }

    std::cout << "Par time: " << algorithm.time() << " (incl. " << algorithm.init_time() << " init)\n";
    bool valid = true;
    double seq_time = 0.0;
    std::tie(valid, seq_time) = validate(edges_by_node, result_by_node);

    std::cout << "Seq time: " << seq_time << "\n";
    std::cout << "Speedup: " << (seq_time / algorithm.time())
              << "  Efficiency: " << (seq_time / algorithm.time() / thread_count) << "\n";

    if (valid) {
        return EXIT_SUCCESS;
    } else {
        return EXIT_FAILURE;
    }

#else
    // Maps thread_num -> local edges
    // The nodes contain a pointers into the edges array
    carray<carray<edge>> global_edges(thread_count);
    // Maps thread_num -> node_id -> node
    // Note: *All* threads know *all* nodes, but not all edges! The graph
    // is distributed over the edges, not over the nodes.
    carray<carray<node>> global_nodes(thread_count);

    std::unique_ptr<std::atomic<double>> global_min_distance(new std::atomic<double>());
    std::unique_ptr<std::atomic<double>> global_max_time(new std::atomic<double>(INFINITY));
    std::unique_ptr<std::atomic<double>> global_max_init_time(new std::atomic<double>(INFINITY));
    std::unique_ptr<std::atomic<bool>> global_continue(new std::atomic<bool>());

#pragma omp parallel num_threads(thread_count)
    {
        const size_t thread_num = omp_get_thread_num();
        // TODO: bind stuff

        carray<edge>& my_edges = global_edges[thread_num];
        carray<node>& my_nodes = global_nodes[thread_num];
        generate_graph(my_edges,
                       my_nodes,
                       thread_num,
                       thread_count,
                       static_cast<unsigned int>(thread_num * seed),
                       node_count,
                       edge_chance);

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

        // Helper function to update distance and predecessor of a node locally
        auto settle_edge = [&](node& node, size_t predecessor, double distance) {
            if (distance < node.distance) {
                node.predecessor = predecessor;
                node.distance = distance;
                if (node.state == state::fringe) {
                    distance_queue.update(node.distance_queue_handle);
#if defined(CRAUSER)
                    crauser_in_queue.update(node.crauser_in_queue_handle);
#endif
                } else {
                    node.state = state::fringe;
                    node.distance_queue_handle = distance_queue.push(&node);
#if defined(CRAUSER)
                    node.crauser_in_queue_handle = crauser_in_queue.push(&node);
#endif
                }
            }
        };

        my_nodes[0].distance = 0.0;
        my_nodes[0].predecessor = -1;
        my_nodes[0].state = state::fringe;
        my_nodes[0].distance_queue_handle = distance_queue.push(&my_nodes[0]);
#if defined(CRAUSER)
        my_nodes[0].crauser_in_queue_handle = crauser_in_queue.push(&my_nodes[0]);
#endif

        const auto init_end = std::chrono::steady_clock::now();

        int global_phase = 0;
        while (true) {
            global_phase += 1;
#if defined(CRAUSER)
            double min_distance = reduce(*global_min_distance,
                                         distance_queue.empty() ? INFINITY : distance_queue.top()->distance,
                                         [](double a, double b) { return std::min(a, b); });
            if (min_distance == INFINITY) {
                break;
            }
            while (!crauser_in_queue.empty() &&
                   crauser_in_queue.top()->distance - crauser_in_queue.top()->min_incoming <= min_distance) {
                to_be_relaxed(*crauser_in_queue.top());
            }
#endif

            // Exchange the minimum distances per node
            for (auto& node : my_nodes) {
                node.new_distance.store(node.distance, std::memory_order_relaxed);
            }
#pragma omp barrier
            global_continue->store(false, std::memory_order_relaxed);
            // This is an all-reduce with operator min in log(t) rounds.
            // Fun fact: Because min does not care about duplicate elements, this
            // works with non-powers-of-two as well.
            // Also note that the reads of new_distance and the writes to
            // new_distance actually race, but it does not matter, because
            // a write can only be a smaller value and it is valid to propagate
            // the smaller value.
            for (int offset = 1; offset < thread_count; offset *= 2) {
                int thread = (thread_num + offset) % thread_count;
                // TODO offset n as well (?)
                for (size_t n = 0; n < node_count; ++n) {
                    double my_distance = my_nodes[n].new_distance.load(std::memory_order_relaxed);
                    double other_distance = global_nodes[thread][n].new_distance.load(std::memory_order_relaxed);
                    if (my_distance < other_distance) {
                        global_nodes[thread][n].new_distance.store(my_distance, std::memory_order_relaxed);
                    }
                }
#pragma omp barrier
            }
            BOOST_ASSERT(distance_queue.empty());
            for (auto& node : my_nodes) {
                double new_distance = node.new_distance.load(std::memory_order_relaxed);
                if (new_distance < node.distance) {
                    node.distance = new_distance;
                    node.predecessor = -2;
                    if (node.state == state::fringe) {
                        distance_queue.update(node.distance_queue_handle);
                    } else {
                        node.distance_queue_handle = distance_queue.push(&node);
                        node.state = state::fringe;
                    }
                }
            }
            if (!distance_queue.empty()) {
                global_continue->store(true, std::memory_order_relaxed);
            }
#pragma omp barrier
            if (!global_continue->load(std::memory_order_relaxed)) {
                break;
            }
        }

        const auto end_time = std::chrono::steady_clock::now();
        reduce(*global_max_time, (end_time - start_time).count() / 1000000000.0, [](double a, double b) {
            return std::max(a, b);
        });
        reduce(*global_max_init_time, (init_end - start_time).count() / 1000000000.0, [](double a, double b) {
            return std::max(a, b);
        });

#pragma omp single
        { std::cout << "Global phases: " << global_phase << "\n"; }
    }

    std::cout << "Par time: " << *global_max_time << " (incl. " << *global_max_init_time << " init time)\n";

    bool valid = true;
    double seq_time = 0.0;
    std::tie(valid, seq_time) = validate(global_nodes, node_count);

    std::cout << "Seq time: " << seq_time << "\n";
    std::cout << "Speedup: " << (seq_time / *global_max_time)
              << "  Efficiency: " << (seq_time / *global_max_time / thread_count) << "\n";

    if (valid) {
        return EXIT_SUCCESS;
    } else {
        return EXIT_FAILURE;
    }
#endif
}
