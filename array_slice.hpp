#pragma once
#include <cstdlib>

template <typename T> class array_slice {
  public:
    array_slice() : array_slice(nullptr, 0) {}
    array_slice(T* ptr, size_t size) : m_ptr(ptr), m_size(size) {}

    T* begin() const { return m_ptr; }
    T* end() const { return m_ptr + m_size; }

    size_t size() const { return m_size; }

    T& operator[](size_t index) const { return m_ptr[index]; }

  private:
    T* m_ptr;
    size_t m_size;
};
