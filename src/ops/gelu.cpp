#include <gpt2lab/ops/gelu.hpp>

#include <gpt2lab/backends/cpu_naive.hpp>

namespace gpt2lab {

Tensor gelu(const Tensor &x) {
  CpuNaiveBackend backend;
  return backend.gelu(x);
}

} // namespace gpt2lab
