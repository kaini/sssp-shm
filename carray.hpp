#pragma once
#include "array_slice.hpp"
#include <algorithm>
#include <boost/assert.hpp>
#include <memory>

namespace sssp {

template <typename T> class carray {
  public:
    carray() : m_size(0), m_data(nullptr) {}
    carray(size_t size) : m_size(size), m_data(new T[size]()) {}
    carray(size_t size, const T& value) : m_size(size), m_data(new T[size]) { std::fill(begin(), end(), value); }

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
    std::unique_ptr<T[]> m_data;
};

} // namespace sssp
