#include <gpt2lab/backends/eigen_backend.hpp>

namespace gpt2lab {

std::string EigenBackend::name() const { return "eigen"; }

Tensor EigenBackend::matmul(const Tensor &a, const Tensor &b) {
  return Tensor(Shape({a.shape().dim(0), b.shape().dim(1)}), DType::Float32);
}

} // namespace gpt2lab
