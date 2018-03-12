#include "array_slice.hpp"
#include "carray.hpp"
#include "dijkstra.hpp"
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
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if HWLOC_API_VERSION < 0x00020000
#error hwloc>=2.0.0 is required
#endif

#if (!defined(BY_NODES) && !defined(BY_EDGES) && !defined(DELTASTEPPING)) ||                                           \
    (defined(BY_NODES) && defined(BY_EDGES)) || (defined(BY_NODES) && defined(DELTASTEPPING)) ||                       \
    (defined(BY_EDGES) && defined(DELTASTEPPING))
#error Define either BY_NODES or BY_EDGES or DELTASTEPPING
#endif

#if defined(BY_NODES)
#include "own_queues_sssp.hpp"
#elif defined(BY_EDGES)
#include "by_edges_sssp.hpp"
#elif defined(DELTASTEPPING)
#include "delta_stepping.hpp"
#endif

using namespace sssp;
using namespace boost::heap;

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

// This allows double edges and self edges, but it does not matter too much.
static void distribute_nodes_generate_uniform_collective(thread_group& threads,
                                                         int thread_rank,
                                                         size_t node_count,
                                                         double edge_chance,
                                                         unsigned int seed,
                                                         carray<edge>& out_thread_edges,
                                                         carray<array_slice<const edge>>& out_thread_edges_by_node) {
    seed *= thread_rank;
    size_t my_nodes_count = threads.chunk_size(thread_rank, node_count);
    size_t my_nodes_start = threads.chunk_start(thread_rank, node_count);
    out_thread_edges_by_node = carray<array_slice<const edge>>(my_nodes_count);

    std::mt19937 edge_count_rng(seed);
    std::binomial_distribution<size_t> edge_count_dist(node_count, edge_chance);
    size_t edge_count = 0;
    for (size_t n = 0; n < my_nodes_count; ++n) {
        edge_count += edge_count_dist(edge_count_rng);
    };
    out_thread_edges = carray<edge>(edge_count);

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

    threads.barrier_collective();
}

static void distribute_edges_generate_uniform_collective(thread_group& threads,
                                                         int thread_rank,
                                                         size_t node_count,
                                                         double edge_chance,
                                                         unsigned int seed,
                                                         carray<edge>& out_thread_edges,
                                                         carray<array_slice<const edge>>& out_thread_edges_by_node) {
    std::mt19937 rng(seed);
    size_t total_edge_count = std::binomial_distribution<size_t>(node_count * node_count, edge_chance)(rng);
    size_t my_edge_count = threads.chunk_size(thread_rank, total_edge_count);
    out_thread_edges = carray<edge>(my_edge_count);
    out_thread_edges_by_node = carray<array_slice<const edge>>(node_count);

    rng.seed((seed + 1) * thread_rank);
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

    threads.barrier_collective();
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "thread_count group_layer\n";
        return EXIT_FAILURE;
    }
    const int thread_count = atoi(argv[1]);
    int group_layer = atoi(argv[2]);

    hwloc_topology_t topo;
    hwloc_topology_init(&topo);
    hwloc_topology_load(topo);

    std::cerr << "Topology: (assuming symmetric topology)\n";
    int topo_depth = hwloc_topology_get_depth(topo);
    group_layer = std::max(0, std::min(topo_depth - 1, group_layer));
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
            effective_topo.back().name += " + ";
        }
        effective_topo.back().name += hwloc_type_to_string(hwloc_get_depth_type(topo, d));
        if (d == group_layer) {
            effective_topo.back().name += "(grouping)";
        }
        if (hwloc_get_next_obj_by_depth(topo, d, nullptr)->memory_arity > 0) {
            effective_topo.back().name += "(numa node)";
        }
    }
    for (const auto& layer : effective_topo) {
        std::cerr << std::setw(4) << layer.count << "x " << layer.name << "\n";
    }

    hwloc_obj_t master_pu = hwloc_get_obj_by_type(topo, HWLOC_OBJ_PU, 0);
    if (hwloc_set_cpubind(topo, master_pu->cpuset, HWLOC_CPUBIND_THREAD) == -1) {
        std::cerr << "Could not bind master thread, ignoring ...\n";
    }

    const size_t node_count = 150000;
    const double edge_chance = 400.0 / node_count;
    const unsigned int seed = 55;
    const bool validate = true;
    std::cout << "Thread count: " << thread_count << "\n";

    hwloc_bitmap_t all_threads = hwloc_bitmap_alloc();
    for (int t = 0; t < thread_count; ++t) {
        hwloc_obj_t pu = hwloc_get_obj_by_type(topo, HWLOC_OBJ_PU, t);
        if (!pu) {
            std::cerr << "Not enought PUs!\n";
            return EXIT_FAILURE;
        }
        hwloc_obj_t core = hwloc_get_ancestor_obj_by_type(topo, HWLOC_OBJ_CORE, pu);
        if (!core) {
            core = pu;
        }
        hwloc_bitmap_or(all_threads, all_threads, core->cpuset);
    }

    std::vector<std::thread> thread_handles;
    thread_group threads(thread_count, topo, all_threads);
    carray<carray<edge>> edges_by_thread(thread_count);
    carray<carray<array_slice<const edge>>> edges_by_thread_by_node(thread_count);
    carray<carray<double>> distances_by_thread(thread_count);
    carray<carray<size_t>> predecessors_by_thread(thread_count);
    carray<double> time_by_thread(thread_count, INFINITY);
    std::atomic<double> gen_time;
    std::atomic<size_t> global_edge_count;

#if defined(BY_NODES)
    own_queues_sssp algorithm;
#elif defined(BY_EDGES)
    by_edges_sssp algorithm;
#elif defined(DELTASTEPPING)
    delta_stepping algorithm(0.1 / (edge_chance * node_count)); // delta = 1/d
#endif

    // Split the threads into groups
    struct group {
        int size = 0;
        int master = -1;
        std::unique_ptr<::thread_group> thread_group;
    };
    carray<int> group_by_thread(thread_count);
    carray<int> rank_in_group_by_thread(thread_count);
    carray<group> groups(hwloc_get_nbobjs_by_depth(topo, group_layer));
    for (int rank = 0; rank < thread_count; ++rank) {
        hwloc_obj_t pu = threads.get_pu(rank);
        hwloc_obj_t group_obj = hwloc_get_ancestor_obj_by_depth(topo, group_layer, pu);
        if (!group_obj) {
            std::cerr << "Found a thread that cannot be assigned to the given layer!\n";
            return EXIT_FAILURE;
        }
        group_by_thread[rank] = static_cast<int>(group_obj->logical_index);
        group& g = groups[group_obj->logical_index];
        rank_in_group_by_thread[rank] = g.size;
        g.size += 1;
        if (g.master == -1) {
            g.master = rank;
        }
    }

    for (int rank = 0; rank < thread_count; ++rank) {
        thread_handles.emplace_back(std::thread([&, rank] {
            // Pin the threads
            hwloc_obj_t pu = threads.get_pu(rank);
            hwloc_obj_t core = hwloc_get_ancestor_obj_by_type(topo, HWLOC_OBJ_CORE, pu);
            if (!core) {
                core = pu;
            }
            if (hwloc_set_cpubind(topo, core->cpuset, HWLOC_CPUBIND_THREAD) == -1) {
                threads.critical_section([] { std::cerr << "Could not bind a thread, ignoring ...\n"; });
            }

            // Generate the graph
            auto gen_start = std::chrono::steady_clock::now();

#if defined(BY_NODES) || defined(DELTASTEPPING)
            distribute_nodes_generate_uniform_collective(
                threads, rank, node_count, edge_chance, seed, edges_by_thread[rank], edges_by_thread_by_node[rank]);
#elif defined(BY_EDGES)
            distribute_edges_generate_uniform_collective(
                threads, rank, node_count, edge_chance, seed, edges_by_thread[rank], edges_by_thread_by_node[rank]);
#endif
            threads.reduce_linear_collective(global_edge_count, size_t(0), edges_by_thread[rank].size(), std::plus<>());
            size_t edge_count = global_edge_count.load(std::memory_order_relaxed);

            auto gen_end = std::chrono::steady_clock::now();
            threads.reduce_linear_collective(gen_time,
                                             0.0,
                                             (gen_end - gen_start).count() / 1000000000.0,
                                             [](double a, double b) { return std::max(a, b); });
            threads.single_collective(
                [&] { std::cout << "Gen time: " << gen_time.load(std::memory_order_relaxed) << "\n"; });

            // Form the thread groups
            int group_rank = group_by_thread[rank];
            if (groups[group_rank].master == rank) {
                groups[group_rank].thread_group.reset(new thread_group(
                    groups[group_rank].size, topo, hwloc_get_obj_by_depth(topo, group_layer, group_rank)->cpuset));
            }
            threads.barrier_collective();

            // Time & run the algorithm
            threads.barrier_collective();
            auto start = std::chrono::steady_clock::now();
            carray<array_slice<const edge>>& edges = edges_by_thread_by_node[rank];
            carray<double>& distances = distances_by_thread[rank] = carray<double>(edges.size(), INFINITY);
            carray<size_t>& predecessors = predecessors_by_thread[rank] = carray<size_t>(edges.size(), -1);
            algorithm.run_collective(threads,
                                     rank,
                                     static_cast<int>(groups.size()),
                                     group_rank,
                                     *groups[group_rank].thread_group,
                                     rank_in_group_by_thread[rank],
                                     node_count,
                                     edge_count,
                                     edges,
                                     distances,
                                     predecessors);
            auto end = std::chrono::steady_clock::now();
            time_by_thread[rank] = (end - start).count() / 1000000000.0;
        }));
    }
    for (auto& t : thread_handles) {
        t.join();
    }

    double par_time = *std::max_element(time_by_thread.begin(), time_by_thread.end());
    std::cout << "Par time: " << par_time << "\n";
    std::cout << "Perf counters:\n";
    for (int t = 0; t < thread_count; ++t) {
        std::cout << "Thread " << t << "\n";
        for (const auto& time : algorithm.perf()[t].times()) {
            std::cout << "\t" << time.first << " = " << time.second << "\n";
        }
    }

    double seq_time = 0.0;
    bool valid = true;

    if (validate) {
        size_t edge_count = global_edge_count.load(std::memory_order_relaxed);
        carray<edge> edges(edge_count);
        carray<array_slice<const edge>> edges_by_node(node_count);
        edge* edges_at = edges.begin();
        for (int t = 0; t < threads.thread_count(); ++t) {
            edges_at = std::copy(edges_by_thread[t].begin(), edges_by_thread[t].end(), edges_at);
        }
        BOOST_ASSERT(edges_at == edges.end());
        std::sort(edges.begin(), edges.end(), [&](const edge& a, const edge& b) { return a.source < b.source; });
        edges_at = edges.begin();
        for (size_t node = 0; node < node_count; ++node) {
            edge* edges_start = edges_at;
            while (edges_at != edges.end() && edges_at->source == node) {
                ++edges_at;
            }
            if (edges_start != edges_at) {
                edges_by_node[node] = array_slice<const edge>(edges_start, edges_at - edges_start);
            }
        }

        auto start = std::chrono::steady_clock::now();

        carray<double> distances(node_count, INFINITY);
        distances[0] = 0.0;
        carray<size_t> predecessors(node_count, -1);
        dijkstra(edges_by_node, distances, predecessors);

        auto end = std::chrono::steady_clock::now();
        seq_time = (end - start).count() / 1000000000.0;

        carray<bool> checked(node_count, false);
        for (size_t n = 0; n < node_count; ++n) {
            for (int t = 0; t < threads.thread_count(); ++t) {
#if defined(BY_NODES) || defined(DELTASTEPPING)
                if (t != threads.chunk_thread_at(node_count, n)) {
                    continue;
                }
                const size_t i = n - threads.chunk_start(t, node_count);
#elif defined(BY_EDGES)
                const size_t i = n;
#endif
                if (predecessors_by_thread[t][i] != -2) {
                    checked[n] = true;
                    if (distances_by_thread[t][i] != distances[n]) {
                        std::cerr << "Invalid distance for node " << n << " " << distances_by_thread[t][i]
                                  << "!=" << distances[n] << " (thread " << t << ")\n";
                        valid = false;
                    }
                    if (predecessors_by_thread[t][i] != predecessors[n]) {
                        std::cerr << "Invalid predecessor for node " << n << " " << predecessors_by_thread[t][i]
                                  << "!=" << predecessors[n] << " (thread " << t << ")\n";
                        valid = false;
                    }
                }
            }
        }
        for (size_t n = 0; n < node_count; ++n) {
            if (!checked[n]) {
                std::cout << "Missing result for node " << n << "\n";
            }
        }
    }

    std::cout << "Seq time: " << seq_time << "  Speedup: " << (seq_time / par_time)
              << "  Efficiency: " << (seq_time / par_time / thread_count) << "\n";
    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
