#include <gpt2lab/ops/layernorm.hpp>

#include <gpt2lab/backends/cpu_naive.hpp>

namespace gpt2lab {

Tensor layernorm(const Tensor &x) {
  CpuNaiveBackend backend;
  return backend.layernorm(x);
}

} // namespace gpt2lab
