#pragma once
#include <atomic>
#include <deque>
#include <mutex>

namespace sssp {

// This is a very dangerous datastructure without memory consistency. Any number of
// threads is allowed to call push_back in parallel. All other methods may only be
// called by a single thread in a period of quiescence! This datastructure does
// not emit any memory fences, to be able to read the elements from the vector
// you have to enforce a memory fence, e.g., by using threads.barrier_collective().
template <typename T> class relaxed_vector {
  public:
    relaxed_vector() : m_data(1000) {}

    void push_back(const T& item) {
        size_t index = m_at.fetch_add(1, std::memory_order_relaxed);
        if (index > m_data.size()) {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (index > m_data.size()) {
                m_data.resize(m_data.size() * 2);
            }
        }
        m_data[index] = item;
    }

    auto begin() { return m_data.begin(); }
    auto end() { return m_data.end(); }
    size_t size() { return m_at.load(std::memory_order_relaxed); }

  private:
    std::mutex m_mutex;
    std::deque<T> m_data;
    std::atomic<size_t> m_at;
};

} // namespace sssp