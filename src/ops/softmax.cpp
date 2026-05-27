#include <gpt2lab/ops/softmax.hpp>

#include <gpt2lab/backends/cpu_naive.hpp>

namespace gpt2lab {

Tensor softmax(const Tensor &x) {
  CpuNaiveBackend backend;
  return backend.softmax(x);
}

} // namespace gpt2lab
