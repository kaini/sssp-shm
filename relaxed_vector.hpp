#pragma once
#include "carray.hpp"
#include "collective_functions.hpp"
#include <atomic>
#include <mutex>
#include <type_traits>
#include <vector>

namespace sssp {

// This is a very dangerous datastructure without memory consistency. Any number of
// threads in the group is allowed to call push_back in parallel. All other methods
// may only be called by a single thread in the group in a period of quiescence!
// This datastructure does not emit any memory fences! To be able to read elements
// you have to enforce a memory fence, e.g., by using threads.barrier_collective().
template <typename T> class relaxed_vector {
    static_assert(std::is_trivially_constructible<T>::value, "");
    static_assert(std::is_trivially_assignable<T, T>::value, "");
    static_assert(std::is_trivially_destructible<T>::value, "");

    static constexpr size_t chunk_size = 1024 * 1024;

  public:
    void init(thread_group& threads, int owner_rank, size_t max_size) {
        m_threads = &threads;
        m_owner = owner_rank;
        m_at.store(0, std::memory_order_relaxed);
        m_data = carray<std::atomic<T*>>(max_size / chunk_size + 1);
        m_owned_memory = carray<thread_group::unique_ptr>(m_data.size());
    }

    void push_back(const T& item) {
        size_t index = m_at.fetch_add(1, std::memory_order_relaxed);
        size_t chunk = index / chunk_size;
        size_t position = index % chunk_size;

        T* ptr = m_data[chunk].load(std::memory_order_relaxed);
        if (ptr == nullptr) {
            auto new_chunk = m_threads->alloc_for_rank(m_owner, sizeof(T) * chunk_size);
            if (m_data[chunk].compare_exchange_strong(
                    ptr, static_cast<T*>(new_chunk.get()), std::memory_order_relaxed, std::memory_order_relaxed)) {
                ptr = static_cast<T*>(new_chunk.get());
                m_owned_memory[chunk] = std::move(new_chunk);
            }
        }

        ptr[position] = item;
    }

    template <typename F> void for_each(F&& fun) {
        size_t at = m_at.load(std::memory_order_relaxed);
        for (size_t chunk = 0; chunk < m_data.size(); ++chunk) {
            T* ptr = m_data[chunk].load(std::memory_order_relaxed);
            if (ptr == nullptr) {
                break;
            }

            size_t limit = chunk_size;
            if (at < (chunk + 1) * chunk_size) {
                limit = at - chunk * chunk_size;
            }

            for (size_t position = 0; position < limit; ++position) {
                fun(ptr[position]);
            }
        }
    }

    void clear() { m_at.store(0, std::memory_order_relaxed); }

  private:
    thread_group* m_threads;
    int m_owner;
    carray<std::atomic<T*>> m_data;
    std::atomic<size_t> m_at;
    carray<thread_group::unique_ptr> m_owned_memory;
};

} // namespace sssp