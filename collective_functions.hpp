#pragma once
#include <boost/thread/barrier.hpp>
#include <mutex>

namespace sssp {

// Shared (!) datastructre to handle a thread group.
class thread_group {
  public:
    thread_group(int thread_count) : m_thread_count(thread_count), m_barrier(thread_count), m_single_do(false) {}

    int thread_count() const { return m_thread_count; }

    template <typename F> void critical_section(F&& fun) {
        std::lock_guard<std::mutex> lock(m_critical_section_mutex);
        fun();
    }

    // Calls fun once on each element of the collection in parallel.
    template <typename Collection, typename F>
    void for_each_collective(int thread_rank, Collection& collection, F&& fun) {
        const size_t size = collection.size();
        for (size_t i = thread_rank; i < size; i += m_thread_count) {
            fun(collection[i]);
        }
        barrier_collective();
    }

    // Calls fun once on each element of the collection in parallel.
    template <typename Collection, typename F>
    void for_each_with_index_collective(int thread_rank, Collection& collection, F&& fun) {
        const size_t size = collection.size();
        for (size_t i = thread_rank; i < size; i += m_thread_count) {
            fun(i, collection[i]);
        }
        barrier_collective();
    }

    size_t for_each_count(int thread_rank, size_t count) const {
        return count / m_thread_count + ((thread_rank < static_cast<int>(count % m_thread_count)) ? 1 : 0);
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

    void barrier_collective() { m_barrier.wait(); }

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