#include "carray.hpp"
#include "delta_stepping.hpp"
#include "thread_local_allocator.hpp"
#include <iostream>
#include <set>
#include <unordered_set>
#include <vector>

namespace {
using node_set =
    std::unordered_set<size_t, std::hash<size_t>, std::equal_to<size_t>, sssp::thread_local_allocator<size_t>>;
} // namespace

void sssp::delta_stepping::run_collective(thread_group& threads,
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

    perf.first_timeblock("init");
    const size_t my_nodes_count = threads.chunk_size(thread_rank, node_count);
    const size_t my_nodes_start = threads.chunk_start(thread_rank, node_count);
    std::vector<node_set> buckets(static_cast<size_t>(std::ceil(1.0 / m_delta)) + 1);
    node_set current_bucket;
    node_set delayed_nodes;
    size_t nodes_in_buckets = 0;
    array_slice<double> distances = out_thread_distances;
    array_slice<size_t> predecessors = out_thread_predecessors;
    std::vector<size_t> bucket_indices(my_nodes_count, -1);

    // Seperate heavy and light edges as optimization for iteration in heavy/light phases
    using edge_vector = std::vector<edge, thread_local_allocator<edge>>;
    using node_edge_vector = std::vector<edge_vector, thread_local_allocator<edge_vector>>;
    node_edge_vector light_edges(node_count);
    node_edge_vector heavy_edges(node_count);
    for (size_t node = 0; node < my_nodes_count; ++node) {
        std::partition_copy(thread_edges_by_node[node].begin(),
                            thread_edges_by_node[node].end(),
                            std::back_inserter(light_edges[node]),
                            std::back_inserter(heavy_edges[node]),
                            [&](const edge& e) { return e.cost <= m_delta; });
    }

    threads.single_collective([&] { m_requests.resize(threads.thread_count()); }, true);
    m_requests[thread_rank].reset(new relaxed_vector<request>(threads, thread_rank, edge_count));

    auto relax = [&](size_t node, double distance, size_t predecessor) {
        if (distance < distances[node]) {
            const size_t new_bucket = static_cast<size_t>(distance / m_delta) % buckets.size();
            if (new_bucket != bucket_indices[node]) {
                if (bucket_indices[node] != -1) {
                    BOOST_ASSERT(buckets[bucket_indices[node]].find(node) != buckets[bucket_indices[node]].end());
                    buckets[bucket_indices[node]].erase(node);
                    nodes_in_buckets -= 1;
                }
                bucket_indices[node] = new_bucket;
                buckets[new_bucket].insert(node);
                nodes_in_buckets += 1;
            }
            distances[node] = distance;
            predecessors[node] = predecessor;
        }
    };

    if (thread_rank == 0) {
        relax(0, 0.0, -1);
    }

    for (int phase = 0;; ++phase) {
        // Check if all buckets are empty
        perf.next_timeblock("phase_init");
        threads.reduce_linear_collective(
            m_done, true, nodes_in_buckets == 0, [](bool a, bool b) { return a && b; }, false);
        if (m_done.load(std::memory_order_relaxed)) {
            break;
        }

        perf.counter_add("phases", 1);

        auto& bucket = buckets[phase % buckets.size()];
        delayed_nodes.clear();

        // Relax light edges in the bucket until the bucket is empty
        while (true) {
            // Check if the current bucket is empty
            perf.next_timeblock("bucket_init");
            threads.reduce_linear_collective(
                m_inner_done, true, bucket.empty(), [](bool a, bool b) { return a && b; }, false);
            if (m_inner_done.load(std::memory_order_relaxed)) {
                break;
            }

            perf.counter_add("subphases", 1);

            // Clear the bucket because reinsertions are going to happen while
            // iterating over it
            perf.next_timeblock("bucket_relax");
            current_bucket.clear();
            std::swap(current_bucket, bucket);
            nodes_in_buckets -= current_bucket.size();
            delayed_nodes.insert(current_bucket.begin(), current_bucket.end());
            for (size_t node : current_bucket) {
                BOOST_ASSERT(bucket_indices[node] != -1);
                bucket_indices[node] = -1;
            }
            for (size_t node : current_bucket) {
                for (const edge& edge : light_edges[node]) {
                    const int dest_thread = threads.chunk_thread_at(node_count, edge.destination);
                    if (dest_thread == thread_rank) {
                        // local relaxation
                        relax(edge.destination - my_nodes_start, distances[node] + edge.cost, edge.source);
                    } else {
                        // remote relaxation
                        m_requests[dest_thread]->push_back(
                            request(edge.destination, distances[node] + edge.cost, edge.source));
                    }
                }
            }

            perf.next_timeblock("bucket_relax_barrier");
            threads.barrier_collective(true);

            perf.next_timeblock("bucket_inbox");
            auto& my_requests = *m_requests[thread_rank];
            my_requests.for_each(
                [&](const request& req) { relax(req.node - my_nodes_start, req.distance, req.predecessor); });
            my_requests.clear();

            perf.next_timeblock("bucket_inbox_barrier");
            threads.barrier_collective(true);
        }

        // Relax the remaining heavy edges
        perf.next_timeblock("heavy_relax");
        for (size_t node : delayed_nodes) {
            for (const edge& edge : heavy_edges[node]) {
                const int dest_thread = threads.chunk_thread_at(node_count, edge.destination);
                if (dest_thread == thread_rank) {
                    // local relaxation
                    relax(edge.destination - my_nodes_start, distances[node] + edge.cost, edge.source);
                } else {
                    // remote relaxation
                    m_requests[dest_thread]->push_back(
                        request(edge.destination, distances[node] + edge.cost, edge.source));
                }
            }
        }

        perf.next_timeblock("heavy_relax_barrier");
        threads.barrier_collective(true);

        perf.next_timeblock("heavy_inbox");
        auto& my_requests = *m_requests[thread_rank];
        my_requests.for_each(
            [&](const request& req) { relax(req.node - my_nodes_start, req.distance, req.predecessor); });
        my_requests.clear();

        perf.next_timeblock("heavy_inbox_barrier");
        threads.barrier_collective(true);
    }

    perf.end_timeblock();
    threads.single_collective([&] { m_perf.resize(threads.thread_count()); }, true);
    m_perf[thread_rank] = std::move(perf);
}
