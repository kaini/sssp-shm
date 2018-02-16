#pragma once
#include <algorithm>
#include <array>
#include <boost/assert.hpp>
#include <boost/intrusive/list.hpp>
#include <boost/intrusive/set.hpp>
#include <memory>

namespace sssp {

inline int ceil_log2(size_t n) {
    size_t result = 1;
    int bit = 0;
    while (result < n) {
        result *= 2;
        bit += 1;
    }
    return bit;
}

// 128 MiB chunks with 256 byte minimum allocation
static constexpr int buddy_allocator_layers = 19;
static constexpr int buddy_allocator_missing_layers = 8;
static constexpr size_t buddy_allocator_chunk_bytes = 1 << (buddy_allocator_layers + buddy_allocator_missing_layers);
static constexpr size_t buddy_allocator_min_bytes = 1 << buddy_allocator_missing_layers;

// buddy_allocator_nodes form a doubly-linked list and contain a free flag.
struct buddy_allocator_node : boost::intrusive::list_base_hook<> {
    int free_in_layer = buddy_allocator_layers;
};

// A chunk of memory.
struct buddy_allocator_chunk : boost::intrusive::set_base_hook<> {
  public:
    buddy_allocator_chunk() { m_layers.back().push_front(m_nodes[0]); }
    buddy_allocator_chunk(const buddy_allocator_chunk& other) = delete;
    buddy_allocator_chunk(buddy_allocator_chunk&& other) = delete;
    buddy_allocator_chunk& operator=(const buddy_allocator_chunk& other) = delete;
    buddy_allocator_chunk& operator=(buddy_allocator_chunk&& other) = delete;

    size_t alloc_node(int layer);
    void free_node(int layer, size_t offset);
    char* base_pointer() { return &m_memory[0]; }
    const char* base_pointer() const { return &m_memory[0]; }

  private:
    // Preallocate all possibly required nodes
    std::array<buddy_allocator_node, 1 << buddy_allocator_layers> m_nodes;
    // Free list for each layer. The extra layer is the indivisible root layer
    std::array<boost::intrusive::list<buddy_allocator_node>, buddy_allocator_layers + 1> m_layers;
    // The memory
    alignas(max_align_t) char m_memory[buddy_allocator_chunk_bytes];
};

inline bool operator<(const buddy_allocator_chunk& a, const buddy_allocator_chunk& b) {
    return a.base_pointer() < b.base_pointer();
}
inline bool operator==(const buddy_allocator_chunk& a, const buddy_allocator_chunk& b) {
    return a.base_pointer() == b.base_pointer();
}

// This holds all the resources needed by an buddy_allocator. Only
// free this once no more allocators use it!
class buddy_allocator_memory {
  public:
    buddy_allocator_memory() = default;
    buddy_allocator_memory(const buddy_allocator_memory& other) = delete;
    buddy_allocator_memory(buddy_allocator_memory&& other) = delete;
    buddy_allocator_memory& operator=(const buddy_allocator_memory& other) = delete;
    buddy_allocator_memory& operator=(buddy_allocator_memory&& other) = delete;
    ~buddy_allocator_memory() {
        m_chunks.clear_and_dispose([](buddy_allocator_chunk* chunk) { delete chunk; });
    }

    char* alloc(int layer);
    void free(int layer, char* address);

  private:
    boost::intrusive::set<buddy_allocator_chunk> m_chunks;
    buddy_allocator_chunk* m_last_success = nullptr;
};

// This is a *none* thread safe buddy allocator.
template <typename T> class buddy_allocator {
  public:
    template <typename OtherT> friend class buddy_allocator;

    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using void_pointer = void*;
    using const_void_pointer = const void*;
    using value_type = T;
    using size_type = size_t;
    using difference_type = ptrdiff_t;
    template <typename OtherT> struct rebind { using other = buddy_allocator<OtherT>; };

    // Creates a buddy allocator with the given state. Once the state is
    // destroyed, the allocator becomes invalid and all memory is lost.
    // Therefore only free the state once all allocator instances are free'd as well.
    buddy_allocator(buddy_allocator_memory* state) : m_state(state) {}
    // This is the type converting constructor.
    template <typename OtherT> buddy_allocator(const buddy_allocator<OtherT>& other) : m_state(other.m_state) {}
    // Appropriate copy and move semantics.
    buddy_allocator(const buddy_allocator<T>& other) : m_state(other.m_state) {}
    buddy_allocator(const buddy_allocator<T>&& other) : m_state(other.m_state) {}
    buddy_allocator& operator=(const buddy_allocator<T>& other) {
        m_state = other.m_state;
        return *this;
    }
    buddy_allocator& operator=(const buddy_allocator<T>&& other) {
        m_state = other.m_state;
        return *this;
    }

    // Equality operators
    template <typename OtherT> bool operator==(const buddy_allocator<OtherT>& other) {
        return m_state == other.m_state;
    }
    template <typename OtherT> bool operator!=(const buddy_allocator<OtherT>& other) {
        return m_state != other.m_state;
    }

    // Allocate memory.
    T* allocate(size_t n) {
        int layer = std::max(0, ceil_log2(sizeof(T) * n) - buddy_allocator_missing_layers);
        if (layer < buddy_allocator_layers) {
            return reinterpret_cast<T*>(m_state->alloc(layer));
        } else {
            return static_cast<T*>(malloc(sizeof(T) * n));
        }
    }

    // Free memory.
    void deallocate(T* ptr, size_t n) {
        int layer = std::max(0, ceil_log2(sizeof(T) * n) - buddy_allocator_missing_layers);
        if (layer < buddy_allocator_layers) {
            m_state->free(layer, reinterpret_cast<char*>(ptr));
        } else {
            free(ptr);
        }
    }

    void destroy(T* ptr) { ptr->~T(); }

  private:
    buddy_allocator_memory* m_state;
};

} // namespace sssp