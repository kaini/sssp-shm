#pragma once
#include <algorithm>
#include <atomic>
#include <functional>
#include <hwloc.h>
#include <memory>
#include <mutex>
#include <numeric>
#include <thread>

namespace sssp {

// Shared (!) datastructre to handle a thread group.
class thread_group {
  public:
    thread_group(int thread_count, hwloc_topology_t topo, hwloc_const_bitmap_t cpuset)
        : m_thread_count(thread_count), m_topo(topo), m_cpuset(cpuset), m_barrier_waiting(0),
          m_barrier_generation(false), m_single_do(false) {
        for (int t = 0; t < thread_count; ++t) {
            m_thread_cpusets.push_back(get_pu(t)->cpuset);
        }
    }

    hwloc_obj_t get_pu(int thread) { return hwloc_get_obj_by_type(m_topo, HWLOC_OBJ_PU, thread); }

    int thread_count() const { return m_thread_count; }

    // Returns the start of the chunk for the given thread, such that no chunk
    // of other threads overlap and all the full range is covered.
    size_t chunk_start(int thread_rank, size_t size) const {
        return thread_rank * (size / m_thread_count) + std::min(size_t(thread_rank), size % m_thread_count);
    }

    // Returns the length to accomodate the return value of chunk_start.
    size_t chunk_size(int thread_rank, size_t size) const {
        return (size / m_thread_count) + (thread_rank < size % m_thread_count ? 1 : 0);
    }

    // Returns the thread id of the thread that is responsible for an index in an array according
    // to chunk_start and chunk_size.
    int chunk_thread_at(size_t size, size_t index) const {
        size_t small_chunk_size = size / m_thread_count;
        size_t large_chunks = size % m_thread_count;
        if (index < (small_chunk_size + 1) * large_chunks) {
            return static_cast<int>(index / (small_chunk_size + 1));
        } else {
            return static_cast<int>(large_chunks + (index - (small_chunk_size + 1) * large_chunks) / small_chunk_size);
        }
    }

    std::function<void*(size_t)> alloc_interleaved_fn() {
        return [=](size_t bytes) { return hwloc_alloc_membind(m_topo, bytes, m_cpuset, HWLOC_MEMBIND_INTERLEAVE, 0); };
    }

    std::function<void*(size_t)> alloc_for_rank_fn(int rank) {
        return [=](size_t bytes) {
            return hwloc_alloc_membind(m_topo, bytes, m_thread_cpusets[rank], HWLOC_MEMBIND_BIND, 0);
        };
    }

    std::function<void(void*, size_t)> free_fn() {
        return [=](void* ptr, size_t bytes) { hwloc_free(m_topo, ptr, bytes); };
    }

    // A mutex around the function.
    template <typename F> void critical_section(F&& fun, bool memory_fence) {
        // Memory fence always happens implicitly over the mutex
        std::lock_guard<std::mutex> lock(m_critical_section_mutex);
        fun();
    }

    // Executes the function once in the group.
    template <typename F> void single_collective(F&& fun, bool memory_fence) {
        m_single_do.store(true, std::memory_order_relaxed);
        barrier_collective(memory_fence);
        if (m_single_do.exchange(false, std::memory_order_relaxed)) {
            fun();
        }
        barrier_collective(memory_fence);
    }

    // Barrier. Synchronizes memory.
    // Inspired by the boost barrier implementation but without condition variables and
    // mutexes. Instead I use a spinlock here. Nevertheless I have to enforce a memory
    // ordering here.
    void barrier_collective(bool memory_fence) {
        if (memory_fence) {
            // Synchronize memory
            std::atomic_thread_fence(std::memory_order_release);
        }

        bool current_generation = m_barrier_generation.load(std::memory_order_relaxed);
        int currently_waiting = m_barrier_waiting.fetch_add(1, std::memory_order_relaxed) + 1;
        if (currently_waiting == m_thread_count) {
            m_barrier_generation.store(!current_generation, std::memory_order_relaxed);
            m_barrier_waiting.fetch_sub(m_thread_count, std::memory_order_relaxed);
        } else {
            while (m_barrier_generation.load(std::memory_order_relaxed) == current_generation) {
                // Note: This yield is a *must* on SPARC otherwise the barrier will be extremely
                // slow (in the order of 100 ms) because other strands will starve. On a more
                // modern machine than ceres it might be possible to solve this better using
                // a monitor load and the mwait instruction.
                std::this_thread::yield();
            }
        }

        if (memory_fence) {
            // Synchronize memory
            std::atomic_thread_fence(std::memory_order_acquire);
        }
    }

    // Linear reduction of a single atomic variable. start and reduce has to be the same on each thread.
    template <typename T, typename F>
    void reduce_linear_collective(std::atomic<T>& into,
                                  const T& start,
                                  const T& contribution,
                                  F&& reduce,
                                  bool memory_fence) {
        into.store(start, std::memory_order_relaxed);
        barrier_collective(memory_fence);
        T current_value = into.load(std::memory_order_relaxed);
        T wanted_value = reduce(contribution, current_value);
        while (wanted_value != current_value &&
               !into.compare_exchange_weak(
                   current_value, wanted_value, std::memory_order_relaxed, std::memory_order_relaxed)) {
            wanted_value = reduce(contribution, current_value);
        }
        barrier_collective(memory_fence);
    }

  private:
    int m_thread_count;
    hwloc_topology_t m_topo;
    hwloc_const_bitmap_t m_cpuset;
    std::vector<hwloc_const_bitmap_t> m_thread_cpusets;
    std::atomic<int> m_barrier_waiting;
    std::atomic<bool> m_barrier_generation;
    std::atomic<bool> m_single_do;
    std::mutex m_critical_section_mutex;
};

template <typename T> void atomic_min(std::atomic<T>& destination, const double value) {
    double stored_value = destination.load(std::memory_order_relaxed);
    while (value < stored_value && !destination.compare_exchange_weak(
                                       stored_value, value, std::memory_order_relaxed, std::memory_order_relaxed)) {
    }
}

} // namespace sssp