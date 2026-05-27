#include <gpt2lab/backends/cpu_et.hpp>

namespace gpt2lab {

std::string CpuEtBackend::name() const { return "cpu_et"; }

Tensor CpuEtBackend::matmul(const Tensor &a, const Tensor &b) {
  return Tensor(Shape({a.shape().dim(0), b.shape().dim(1)}), DType::Float32);
}

} // namespace gpt2lab
