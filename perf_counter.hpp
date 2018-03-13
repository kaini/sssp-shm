#pragma once
#include <chrono>
#include <cstring>
#include <map>

namespace sssp {

class perf_counter {
  public:
    perf_counter();

    void first_timeblock(const char* name);
    void next_timeblock(const char* name);
    void end_timeblock();

    void counter_add(const char* name, double offset);

    const auto& values() const { return m_values; }

  private:
    struct cmp_const_char_ptr {
        bool operator()(const char* a, const char* b) const { return strcmp(a, b) < 0; }
    };

    std::chrono::steady_clock::time_point m_start;
    const char* m_current_timeblock;
    std::map<const char*, double, cmp_const_char_ptr> m_values;
};

} // namespace sssp