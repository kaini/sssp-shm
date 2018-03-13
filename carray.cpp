#include "carray.hpp"
#include <memory>

sssp::carray_alloc sssp::malloc_alloc = [](size_t bytes) { return malloc(bytes); };
sssp::carray_free sssp::malloc_free = [](void* ptr, size_t bytes) { free(ptr); };
