#include <gpt2lab/ops/matmul.hpp>

#include <gpt2lab/backends/cpu_naive.hpp>

namespace gpt2lab {

Tensor matmul(const Tensor &a, const Tensor &b) {
  CpuNaiveBackend backend;
  return backend.matmul(a, b);
}

} // namespace gpt2lab
