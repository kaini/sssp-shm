#include "perf_counter.hpp"
#include <boost/assert.hpp>

sssp::perf_counter::perf_counter() : m_current_timeblock(nullptr) {}

void sssp::perf_counter::first_timeblock(const char* name) {
    BOOST_ASSERT(m_current_timeblock == nullptr);
    m_start = std::chrono::steady_clock::now();
    m_current_timeblock = name;
    BOOST_ASSERT(m_current_timeblock != nullptr);
}

void sssp::perf_counter::next_timeblock(const char* name) {
    BOOST_ASSERT(m_current_timeblock != nullptr);
    auto now = std::chrono::steady_clock::now();
    m_values[m_current_timeblock] += (now - m_start).count() / 1000000000.0;
    m_start = now;
    m_current_timeblock = name;
    BOOST_ASSERT(m_current_timeblock != nullptr);
}

void sssp::perf_counter::end_timeblock() {
    BOOST_ASSERT(m_current_timeblock != nullptr);
    auto now = std::chrono::steady_clock::now();
    m_values[m_current_timeblock] += (now - m_start).count() / 1000000000.0;
    m_current_timeblock = nullptr;
    BOOST_ASSERT(m_current_timeblock == nullptr);
}

void sssp::perf_counter::counter_add(const char* name, double offset) {
    m_values[name] += offset;
}
