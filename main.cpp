#include "allreduce_sssp.hpp"
#include "array_slice.hpp"
#include "carray.hpp"
#include "dijkstra.hpp"
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
static void generate_edges_collective(thread_group& threads,
                                      int thread_rank,
                                      size_t node_count,
                                      double edge_chance,
                                      unsigned int seed,
                                      carray<edge>& out_thread_edges,
                                      carray<array_slice<const edge>>& out_thread_edges_by_node) {
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

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "args ...\n";
        return EXIT_FAILURE;
    }
    const int thread_count = atoi(argv[1]);

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

    hwloc_obj_t master_pu = hwloc_get_obj_by_type(topo, HWLOC_OBJ_PU, 0);
    if (hwloc_set_cpubind(topo, master_pu->cpuset, HWLOC_CPUBIND_THREAD) == -1) {
        std::cerr << "Could not bind master thread, ignoring ...\n";
    }

    const size_t node_count = 150000;
    const double edge_chance = 400.0 / node_count;
    const unsigned int seed = 42;
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
    own_queues_sssp own_queues_sssp;

    for (int rank = 0; rank < thread_count; ++rank) {
        thread_handles.emplace_back(std::thread([&, rank] {
            // Pin the threads
            hwloc_obj_t pu = hwloc_get_obj_by_type(topo, HWLOC_OBJ_PU, rank);
            hwloc_obj_t core = hwloc_get_ancestor_obj_by_type(topo, HWLOC_OBJ_CORE, pu);
            if (!core) {
                core = pu;
            }
            if (hwloc_set_cpubind(topo, core->cpuset, HWLOC_CPUBIND_THREAD) == -1) {
                threads.critical_section([] { std::cerr << "Could not bind a thread, ignoring ...\n"; });
            }

            // Generate the graph
            auto gen_start = std::chrono::steady_clock::now();

            generate_edges_collective(threads,
                                      rank,
                                      node_count,
                                      edge_chance,
                                      static_cast<unsigned int>(rank) * seed,
                                      edges_by_thread[rank],
                                      edges_by_thread_by_node[rank]);
            threads.reduce_linear_collective(global_edge_count, size_t(0), edges_by_thread[rank].size(), std::plus<>());
            size_t edge_count = global_edge_count.load(std::memory_order_relaxed);

            auto gen_end = std::chrono::steady_clock::now();
            threads.reduce_linear_collective(gen_time,
                                             0.0,
                                             (gen_end - gen_start).count() / 1000000000.0,
                                             [](double a, double b) { return std::max(a, b); });
            threads.single_collective(
                [&] { std::cout << "Gen time: " << gen_time.load(std::memory_order_relaxed) << "\n"; });

            // Time & run the algorithm
            threads.barrier_collective();
            auto start = std::chrono::steady_clock::now();
            carray<array_slice<const edge>>& edges = edges_by_thread_by_node[rank];
            carray<double>& distances = distances_by_thread[rank] = carray<double>(edges.size(), INFINITY);
            carray<size_t>& predecessors = predecessors_by_thread[rank] = carray<size_t>(edges.size(), -1);
            own_queues_sssp.run_collective(threads, rank, node_count, edge_count, edges, distances, predecessors);
            auto end = std::chrono::steady_clock::now();
            time_by_thread[rank] = (end - start).count() / 1000000000.0;
        }));
    }
    for (auto& t : thread_handles) {
        t.join();
    }

    double par_time = *std::max_element(time_by_thread.begin(), time_by_thread.end());
    std::cout << "Par time: " << par_time << "\n";

    double seq_time = 0.0;
    bool valid = true;

    if (validate) {
        size_t edge_count = global_edge_count.load(std::memory_order_relaxed);
        carray<edge> edges(edge_count);
        carray<array_slice<const edge>> edges_by_node(node_count);
        edge* edges_at = edges.begin();
        for (size_t n = 0; n < node_count; ++n) {
            int t = threads.chunk_thread_at(node_count, n);
            size_t i = n - threads.chunk_start(t, node_count);
            edge* edges_at_start = edges_at;
            edges_at = std::copy(edges_by_thread_by_node[t][i].begin(), edges_by_thread_by_node[t][i].end(), edges_at);
            edges_by_node[n] = array_slice<const edge>(edges_at_start, edges_at - edges_at_start);
        }
        BOOST_ASSERT(edges_at == edges.end());

        auto start = std::chrono::steady_clock::now();

        carray<double> distances(node_count, INFINITY);
        distances[0] = 0.0;
        carray<size_t> predecessors(node_count, -1);
        dijkstra(edges_by_node, distances, predecessors);

        auto end = std::chrono::steady_clock::now();
        seq_time = (end - start).count() / 1000000000.0;

        carray<bool> checked(node_count, false);
        for (size_t n = 0; n < node_count; ++n) {
            int t = threads.chunk_thread_at(node_count, n);
            size_t i = n - threads.chunk_start(t, node_count);
            if (predecessors_by_thread[t][i] != -2) {
                checked[n] = true;
                if (distances_by_thread[t][i] != distances[n]) {
                    std::cerr << "Invalid distance for node " << n << " " << distances_by_thread[t][i]
                              << "!=" << distances[n] << "\n";
                    valid = false;
                }
                if (predecessors_by_thread[t][i] != predecessors[n]) {
                    std::cerr << "Invalid predecessor for node " << n << " " << predecessors_by_thread[t][i]
                              << "!=" << predecessors[n] << "\n";
                    valid = false;
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
