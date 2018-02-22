#pragma once
#include <boost/thread/barrier.hpp>
#include <mutex>

namespace sssp {

// Shared (!) datastructre to handle a thread group.
class thread_group {
  public:
    thread_group(int thread_count) : m_thread_count(thread_count), m_barrier(thread_count), m_single_do(false) {}

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

    // A mutex around the function.
    template <typename F> void critical_section(F&& fun) {
        std::lock_guard<std::mutex> lock(m_critical_section_mutex);
        fun();
    }

    // Executes the function once in the group.
    template <typename F> void single_collective(F&& fun) {
        m_single_do.store(true, std::memory_order_relaxed);
        barrier_collective();
        if (m_single_do.exchange(false, std::memory_order_relaxed)) {
            fun();
        }
        barrier_collective();
    }

    // Barrier.
    void barrier_collective() { m_barrier.wait(); }

    // Linear reduction of a single atomic variable.
    template <typename T, typename F> void reduce_linear_collective(std::atomic<T>& into, const T& value, F&& reduce) {
        into.store(value, std::memory_order_relaxed);
        barrier_collective();
        T current_value = into.load(std::memory_order_relaxed);
        T wanted_value = reduce(value, current_value);
        while (wanted_value != current_value &&
               !into.compare_exchange_weak(
                   current_value, wanted_value, std::memory_order_relaxed, std::memory_order_relaxed)) {
            wanted_value = reduce(value, current_value);
        }
        barrier_collective();
    }

  private:
    int m_thread_count;
    boost::barrier m_barrier;
    std::atomic<bool> m_single_do;
    std::mutex m_critical_section_mutex;
};

template <typename T> void atomic_min(std::atomic<T>& destination, const double value) {
    while (value < destination.load(std::memory_order_relaxed)) {
        destination.store(value, std::memory_order_relaxed);
    }
}

} // namespace sssp