#include "array_slice.hpp"
#include "carray.hpp"
#include "dijkstra.hpp"
#include <boost/program_options.hpp>
#include <chrono>
#include <hwloc.h>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>

#if HWLOC_API_VERSION < 0x00020000
#error hwloc>=2.0.0 is required
#endif

#if !defined(BY_NODES) && !defined(BY_EDGES) && !defined(DELTASTEPPING) && !defined(SEQ)
#error Define either BY_NODES or BY_EDGES or DELTASTEPPING or SEQ
#endif

#if defined(BY_NODES)
#include "own_queues_sssp.hpp"
#elif defined(BY_EDGES)
#include "by_edges_sssp.hpp"
#elif defined(DELTASTEPPING)
#include "delta_stepping.hpp"
#endif

using namespace sssp;

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

int main(int argc, char* argv[]) {
    namespace po = boost::program_options;

    size_t node_count = 150000;
    double edge_chance = 400.0 / node_count;
    unsigned int seed = 55;
    bool validate = false;
    int thread_count = -1;
    int group_layer = 0;
    size_t initiator_size = 5;
    int k = 3;
    std::string graph_type = "uniform";
    bool print_header = true;

    // clang-format off
    po::options_description opts("SSSP Benchmarking Tool");
    opts.add_options()
        ("graph", po::value(&graph_type)->default_value(graph_type)->required(),
            "Graph type: uniform or kronecker")
        ("node-count", po::value(&node_count)->default_value(node_count)->required(),
            "The number of nodes for uniform graphs")
        ("edge-chance", po::value(&edge_chance)->default_value(edge_chance)->required(),
            "The chance of a single edge for uniform graphs")
        ("initiator-size", po::value(&initiator_size)->default_value(initiator_size)->required(),
            "The dimension of the initiator matrix for Kronecker graphs")
        ("k", po::value(&k)->default_value(k)->required(),
            "The parameter k for Kronecker graphs")
        ("seed,s", po::value(&seed)->default_value(seed)->required(),
            "The seed for the random number generator")
        ("validate,v", po::bool_switch(&validate)->default_value(validate),
            "Compare the results with a sequential implementation of Dijkstra's algorithm (ignored in the sequential implementation)")
        ("thread-count,t", po::value(&thread_count),
            "The number of threads to use, defaults to all threads as seen by hwloc (used to generate the graph in parallel in the sequential implementation)")
        ("group-layer,g", po::value(&group_layer)->default_value(group_layer)->required(),
            "Some algorithms perform optimizations by grouping threads. The group layer is the layer in the hwloc topology where the grouping should happen")
        ("help,h", "Show this help message")
        ("no-header,0", po::bool_switch(&print_header)->default_value(!print_header),
            "Do not print the CSV header")
        ;
    // clang-format on

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, opts), vm);
        po::notify(vm);
        print_header = !print_header;
#if defined(SEQ)
        validate = true;
#endif
    } catch (const po::error& ex) {
        std::cerr << ex.what() << "\n";
        return EXIT_FAILURE;
    }
    if (vm.count("help")) {
        std::cerr << opts << "\n";
        return EXIT_FAILURE;
    }
    if (!(0.0 <= edge_chance && edge_chance <= 1.0)) {
        std::cerr << "edge_chance has to be between 0 and 1.\n";
        return EXIT_FAILURE;
    }

    hwloc_topology_t topo;
    hwloc_topology_init(&topo);
    hwloc_topology_load(topo);

    if (print_header) {
        std::cout << "# Topology: (assuming symmetric topology)\n";
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
            std::cout << "# " << std::setw(4) << layer.count << "x " << layer.name << "\n";
        }
    }

    hwloc_obj_t master_pu = hwloc_get_obj_by_type(topo, HWLOC_OBJ_PU, 0);
    if (hwloc_set_cpubind(topo, master_pu->cpuset, HWLOC_CPUBIND_THREAD) == -1) {
        std::cerr << "Could not bind master thread, ignoring ...\n";
    }

    if (thread_count < 0) {
        thread_count = hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_PU);
    }

    hwloc_bitmap_t all_threads = hwloc_bitmap_alloc();
    const int hwloc_core_count = hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_CORE);
    if (hwloc_core_count == 0) {
        std::cerr << "The topology does not contain cores!\n";
        return EXIT_FAILURE;
    }
    for (int t = 0; t < thread_count; ++t) {
        hwloc_obj_t core = hwloc_get_obj_by_type(topo, HWLOC_OBJ_CORE, t % hwloc_core_count);
        hwloc_obj_t pu = hwloc_get_obj_inside_cpuset_by_type(topo, core->cpuset, HWLOC_OBJ_PU, t / hwloc_core_count);
        if (!pu) {
            std::cerr << "Not enought PUs (or the system topology is not symmetric)!\n";
            return EXIT_FAILURE;
        }
        hwloc_bitmap_or(all_threads, all_threads, pu->cpuset);
    }

    std::vector<std::thread> thread_handles;
    thread_group threads(thread_count, topo, all_threads);
    std::vector<std::vector<edge>> edges_by_thread(thread_count);
    std::vector<std::vector<array_slice<const edge>>> edges_by_thread_by_node(thread_count);
    std::vector<std::vector<double>> distances_by_thread(thread_count);
    std::vector<std::vector<size_t>> predecessors_by_thread(thread_count);
    std::vector<double> time_by_thread(thread_count, INFINITY);
    std::atomic<size_t> global_edge_count;
    distribute_nodes_generate_kronecker distribute_nodes_generate_kronecker;

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
    std::vector<int> group_by_thread(thread_count);
    std::vector<int> rank_in_group_by_thread(thread_count);
    std::vector<group> groups(hwloc_get_nbobjs_by_depth(topo, group_layer));
    for (int rank = 0; rank < thread_count; ++rank) {
        hwloc_obj_t pu = threads.get_pu(rank);
        hwloc_obj_t group_obj = hwloc_get_ancestor_obj_by_depth(topo, group_layer, pu);
        if (!group_obj) {
            std::cerr
                << "Found a thread that cannot be assigned to the given layer (maybe the topology is not symmetric)!\n";
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
            if (hwloc_set_cpubind(topo, pu->cpuset, HWLOC_CPUBIND_THREAD) == -1) {
                threads.critical_section([] { std::cerr << "Could not bind a thread, ignoring ...\n"; }, false);
            }

            // Generate the graph
            if (graph_type == "uniform") {
#if defined(BY_NODES) || defined(DELTASTEPPING) || defined(SEQ)
                distribute_nodes_generate_uniform_collective(
                    threads, rank, node_count, edge_chance, seed, edges_by_thread[rank], edges_by_thread_by_node[rank]);
#elif defined(BY_EDGES)
                distribute_edges_generate_uniform_collective(
                    threads, rank, node_count, edge_chance, seed, edges_by_thread[rank], edges_by_thread_by_node[rank]);
#endif
            } else if (graph_type == "kronecker") {
#if defined(BY_NODES) || defined(DELTASTEPPING) || defined(SEQ)
                distribute_nodes_generate_kronecker.run_collective(
                    threads, rank, initiator_size, k, seed, edges_by_thread[rank], edges_by_thread_by_node[rank]);
                node_count = 1;
                for (int i = 0; i < k; ++i) {
                    node_count *= initiator_size;
                }
#elif defined(BY_EDGES)
                threads.critical_section(
                    [] {
                        std::cerr << "Kronecker graphs are not supported BY_EDGES\n";
                        std::exit(1);
                    },
                    false);
#endif
            } else {
                threads.critical_section(
                    [] {
                        std::cerr << "Unknown graph type selected\n";
                        std::exit(1);
                    },
                    false);
            }
            threads.reduce_linear_collective(
                global_edge_count, size_t(0), edges_by_thread[rank].size(), std::plus<>(), false);
            size_t edge_count = global_edge_count.load(std::memory_order_relaxed);

#if !defined(SEQ)
            // Form the thread groups
            int group_rank = group_by_thread[rank];
            if (groups[group_rank].master == rank) {
                groups[group_rank].thread_group.reset(new thread_group(
                    groups[group_rank].size, topo, hwloc_get_obj_by_depth(topo, group_layer, group_rank)->cpuset));
            }
            threads.barrier_collective(true);

            // Time & run the algorithm
            auto start = std::chrono::steady_clock::now();
            std::vector<array_slice<const edge>>& edges = edges_by_thread_by_node[rank];
            std::vector<double>& distances = distances_by_thread[rank];
            distances.resize(edges.size(), INFINITY);
            std::vector<size_t>& predecessors = predecessors_by_thread[rank];
            predecessors.resize(edges.size(), -1);
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
#endif
        }));
    }
    for (auto& t : thread_handles) {
        t.join();
    }

    if (validate) {
        // Collect all edges into a single array for the single threaded solution.
        size_t edge_count = global_edge_count.load(std::memory_order_relaxed);
        std::vector<edge> edges(edge_count);
        std::vector<array_slice<const edge>> edges_by_node(node_count);
        auto edges_at = edges.begin();
        for (int t = 0; t < threads.thread_count(); ++t) {
            edges_at = std::copy(edges_by_thread[t].begin(), edges_by_thread[t].end(), edges_at);
        }
        BOOST_ASSERT(edges_at == edges.end());
        std::sort(edges.begin(), edges.end(), [&](const edge& a, const edge& b) { return a.source < b.source; });
        edges_at = edges.begin();
        for (size_t node = 0; node < node_count; ++node) {
            auto edges_start = edges_at;
            while (edges_at != edges.end() && edges_at->source == node) {
                ++edges_at;
            }
            if (edges_start != edges_at) {
                edges_by_node[node] = array_slice<const edge>(&*edges_start, std::distance(edges_start, edges_at));
            }
        }

        // Run the sequential algorithm.
#if defined(SEQ)
        auto start = std::chrono::steady_clock::now();
#endif
        std::vector<double> distances(node_count, INFINITY);
        distances[0] = 0.0;
        std::vector<size_t> predecessors(node_count, -1);
        dijkstra(edges_by_node, distances, predecessors);
#if defined(SEQ)
        auto end = std::chrono::steady_clock::now();
        time_by_thread[0] = (end - start).count() / 1000000000.0;
#endif

        // Compare the multi-threaded solution with the sequential solution.
#if !defined(SEQ)
        bool valid = true;
        std::vector<bool> checked(node_count, false);
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
                std::cerr << "Missing result for node " << n << "\n";
            }
        }
        if (!valid) {
            return EXIT_FAILURE;
        }
#endif
    }

    if (print_header) {
        std::cout
            << "executable,graph,node_count,edge_chance,initiator_size,k,seed,thread_count,group_layer,thread,time";
#if !defined(SEQ)
        for (const auto& perf : algorithm.perf()[0].values()) {
            std::cout << "," << perf.first;
        }
#endif
        std::cout << "\n";
    }
    for (int t = 0; t < thread_count; ++t) {
        std::string executable = argv[0];
        size_t last_slash = executable.rfind('/');
        if (last_slash != std::string::npos) {
            executable = executable.substr(last_slash + 1);
        }
        size_t last_backslash = executable.rfind('\\');
        if (last_backslash != std::string::npos) {
            executable = executable.substr(last_backslash + 1);
        }
        std::cout << executable << ",";
        std::cout << graph_type << ",";
        std::cout << (graph_type == "uniform" ? std::to_string(node_count) : "NA") << ",";
        std::cout << (graph_type == "uniform" ? std::to_string(edge_chance) : "NA") << ",";
        std::cout << (graph_type == "kronecker" ? std::to_string(initiator_size) : "NA") << ",";
        std::cout << (graph_type == "kronecker" ? std::to_string(k) : "NA") << ",";
        std::cout << seed << ",";
        std::cout << thread_count << ",";
        std::cout << group_layer << ",";
        std::cout << t << ",";
        std::cout << time_by_thread[t];
#if !defined(SEQ)
        for (const auto& perf : algorithm.perf()[t].values()) {
            std::cout << "," << perf.second;
        }
#endif
        std::cout << "\n";
#if defined(SEQ)
        // In the sequential implementation only the first thread did something. The others
        // were only used to generate the graph. Therefore there is no need to print them.
        break;
#endif
    }

    return EXIT_SUCCESS;
}
