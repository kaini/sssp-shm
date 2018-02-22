#pragma once
#include "array_slice.hpp"
#include "graph.hpp"

namespace sssp {

// Uses Dijkstra's algorithm to find the shortest paths, written into out_distances and out_predecessors.
// Note that this function *does not* reset out_distances or out_predecessors to INFINITY/-1.
// Therefore you have to reset these yourself or can prefill them with values to influence
// the algorithm.
// If prefill_fringe is set, the nodes set to true will be put into the fringe state at the
// start of the algorithm. If not set, only the start node (0) will be fringe at the start
// of the algorithm.
void dijkstra(array_slice<const array_slice<const edge>> edges_by_node,
              array_slice<double> out_distances,
              array_slice<size_t> out_predecessors,
              array_slice<const bool> prefill_fringe = array_slice<const bool>());

} // namespace sssp
