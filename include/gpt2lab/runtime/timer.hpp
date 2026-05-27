#pragma once

#include <chrono>

namespace gpt2lab {

class Timer {
public:
  void start();
  double elapsed() const;

private:
  std::chrono::steady_clock::time_point start_;
};

} // namespace gpt2lab
