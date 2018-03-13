#pragma once
#include "array_slice.hpp"
#include <algorithm>
#include <boost/assert.hpp>
#include <functional>
#include <memory>

namespace sssp {

using carray_alloc = std::function<void*(size_t)>;
using carray_free = std::function<void(void*, size_t)>;

extern carray_alloc malloc_alloc;
extern carray_free malloc_free;

template <typename T> class carray {
  public:
    static_assert(std::is_trivially_destructible<T>::value, "T has to be trivially destructible");

    carray() : m_size(0), m_data(nullptr) {}
    // WARNING: The memory is *not* initialized! You have to use placement new if the values are not
    // trivially constructible.
    carray(size_t size, carray_alloc alloc = malloc_alloc, carray_free free = malloc_free)
        : m_size(size), m_data(static_cast<T*>(alloc(sizeof(T) * size)), [size, free](T* ptr) { free(ptr, size); }) {}

    T* begin() { return m_data.get(); }
    T* end() { return m_data.get() + m_size; }
    const T* begin() const { return m_data.get(); }
    const T* end() const { return m_data.get() + m_size; }

    T& operator[](size_t index) {
        BOOST_ASSERT(index < m_size);
        return m_data[index];
    }
    const T& operator[](size_t index) const {
        BOOST_ASSERT(index < m_size);
        return m_data[index];
    }

    T* data() { return m_data.get(); }
    const T* data() const { return m_data.get(); }

    size_t size() const { return m_size; }

    operator array_slice<T>() { return {m_data.get(), m_size}; }
    operator array_slice<const T>() const { return {m_data.get(), m_size}; }

  private:
    size_t m_size;
    std::unique_ptr<T[], std::function<void(T*)>> m_data;
};

} // namespace sssp
