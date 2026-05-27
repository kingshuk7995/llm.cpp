#pragma once

#include <gpt2lab/bench/benchmark.hpp>
#include <vector>

namespace gpt2lab {

struct Metrics {
  std::vector<BenchmarkResult> results;
};

} // namespace gpt2lab
