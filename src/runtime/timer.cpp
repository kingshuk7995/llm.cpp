#include <gpt2lab/runtime/timer.hpp>

namespace gpt2lab {

void Timer::start() { start_ = std::chrono::steady_clock::now(); }

double Timer::elapsed() const {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                       start_)
      .count();
}

} // namespace gpt2lab
