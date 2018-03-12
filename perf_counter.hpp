#pragma once
#include <chrono>
#include <map>

namespace sssp {

class perf_counter {
  public:
    perf_counter();

    void first_timeblock(const char* name);
    void next_timeblock(const char* name);
    void end_timeblock();

    const auto& times() const { return m_times; }

  private:
    std::chrono::steady_clock::time_point m_start;
    const char* m_current_timeblock;
    std::map<const char*, double> m_times;
};

} // namespace sssp