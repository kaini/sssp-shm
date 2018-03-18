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
    static constexpr size_t chunk_size = 1024 * 1024;

  public:
    relaxed_vector(thread_group& threads, int owner_rank, size_t max_size) {
        m_threads = &threads;
        m_owner = owner_rank;
        m_at.store(0, std::memory_order_relaxed);
        m_data = carray<std::atomic<T*>>(max_size / chunk_size + 1);
        for (size_t i = 0; i < m_data.size(); ++i) {
            new (&m_data[i]) std::atomic<T*>(nullptr);
        }
        m_owned_memory.resize(m_data.size());
    }

    void push_back(const T& item) {
        size_t index = m_at.fetch_add(1, std::memory_order_relaxed);
        size_t chunk = index / chunk_size;
        size_t position = index % chunk_size;

        T* ptr = m_data[chunk].load(std::memory_order_relaxed);
        if (ptr == nullptr) {
            std::unique_ptr<T, std::function<void(T*)>> new_chunk(
                static_cast<T*>(m_threads->alloc_for_rank_fn(m_owner)(sizeof(T) * chunk_size)),
                [=](T* ptr) { m_threads->free_fn()(ptr, sizeof(T) * chunk_size); });
            if (m_data[chunk].compare_exchange_strong(
                    ptr, new_chunk.get(), std::memory_order_relaxed, std::memory_order_relaxed)) {
                ptr = new_chunk.get();
                m_owned_memory[chunk] = std::move(new_chunk);
            }
        }

        ptr[position] = item;
    }

    template <typename F> void for_each(F&& fun) {
        const size_t end = m_at.load(std::memory_order_relaxed);
        size_t at = 0;

        for (size_t chunk = 0; chunk < m_data.size(); ++chunk) {
            T* ptr = m_data[chunk].load(std::memory_order_relaxed);
            if (!ptr) {
                break;
            }
            for (size_t i = 0; i < chunk_size; ++i) {
                if (at >= end) {
                    break;
                }
                fun(ptr[i]);
                at += 1;
            }
            if (at >= end) {
                break;
            }
        }

        BOOST_ASSERT(at == end);
    }

    void clear() { m_at.store(0, std::memory_order_relaxed); }

    size_t size() const { return m_at.load(std::memory_order_relaxed); }

  private:
    thread_group* m_threads;
    int m_owner;
    carray<std::atomic<T*>> m_data;
    std::atomic<size_t> m_at;
    std::vector<std::unique_ptr<T, std::function<void(T*)>>> m_owned_memory;
};

} // namespace sssp