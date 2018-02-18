#pragma once
#include <thread>

#ifdef NUMA
#include <numa.h>
#else

inline int numa_available() {
    return 0;
}

inline int numa_distance(int node1, int node2) {
    return 0;
}

inline int numa_num_task_cpus() {
    return static_cast<int>(std::thread::hardware_concurrency());
}

inline int numa_num_task_nodes() {
    return 1;
}

#endif
