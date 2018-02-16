#pragma once
#include "buddy_allocator.hpp"

namespace sssp {

extern thread_local buddy_allocator_memory thread_local_allocator_memory;

template <typename T> class thread_local_allocator {
  public:
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using void_pointer = void*;
    using const_void_pointer = const void*;
    using value_type = T;
    using size_type = size_t;
    using difference_type = ptrdiff_t;
    template <typename OtherT> struct rebind { using other = thread_local_allocator<OtherT>; };

    thread_local_allocator() {}
    template <typename OtherT> thread_local_allocator(const thread_local_allocator<OtherT>& other) {}
    thread_local_allocator(const thread_local_allocator<T>& other) {}
    thread_local_allocator(const thread_local_allocator<T>&& other) {}
    thread_local_allocator& operator=(const thread_local_allocator<T>& other) { return *this; }
    thread_local_allocator& operator=(const thread_local_allocator<T>&& other) { return *this; }

    template <typename OtherT> bool operator==(const thread_local_allocator<OtherT>& other) { return true; }
    template <typename OtherT> bool operator!=(const thread_local_allocator<OtherT>& other) { return false; }

    T* allocate(size_t n) { return buddy_allocator<T>(&thread_local_allocator_memory).allocate(n); }
    void deallocate(T* ptr, size_t n) { return buddy_allocator<T>(&thread_local_allocator_memory).deallocate(ptr, n); }
    void destroy(T* ptr) { buddy_allocator<T>(&thread_local_allocator_memory).destroy(ptr); }
};

} // namespace sssp
