#pragma once
#include <boost/assert.hpp>
#include <cstdlib>
#include <type_traits>
#include <vector>

namespace sssp {

template <typename T> class array_slice {
  public:
    array_slice() : array_slice(nullptr, 0) {}
    array_slice(T* ptr, size_t size) : m_ptr(ptr), m_size(size) {}
    array_slice(std::vector<typename std::remove_const<T>::type>& vector)
        : m_ptr(vector.data()), m_size(vector.size()) {}
    array_slice(const std::vector<typename std::remove_const<T>::type>& vector)
        : m_ptr(vector.data()), m_size(vector.size()) {}

    T* begin() const { return m_ptr; }
    T* end() const { return m_ptr + m_size; }

    size_t size() const { return m_size; }

    T& operator[](size_t index) const {
        BOOST_ASSERT(index < m_size);
        return m_ptr[index];
    }

    template <typename As> array_slice<As> as() const { return array_slice<As>(m_ptr, m_size); }
    array_slice<const T> as_const() const { return as<const T>(); }
    operator array_slice<const T>() const { return as_const(); }

  private:
    T* m_ptr;
    size_t m_size;
};

} // namespace sssp
