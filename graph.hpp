#pragma once
#include <cmath>

namespace sssp {

struct edge {
    edge() {}
    edge(size_t s, size_t d, double c) : source(s), destination(d), cost(c) {}
    size_t source = -1;
    size_t destination = -1;
    double cost = 0.0;
};

struct result {
    result() {}
    result(double distance, size_t predecessor) : distance(distance), predecessor(predecessor) {}
    double distance = INFINITY;
    size_t predecessor = -1;
};

} // namespace sssp