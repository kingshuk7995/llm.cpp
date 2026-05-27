#pragma once

#include <string>

namespace gpt2lab {

struct BenchmarkResult {
  std::string name;
  double duration_seconds = 0.0;
};

} // namespace gpt2lab
