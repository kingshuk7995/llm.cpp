#include <gpt2lab/backends/cudnn_backend.hpp>

namespace gpt2lab {

std::string CudnnBackend::name() const { return "cudnn"; }

Tensor CudnnBackend::matmul(const Tensor &a, const Tensor &b) {
  return Tensor(Shape({a.shape().dim(0), b.shape().dim(1)}), DType::Float32);
}

} // namespace gpt2lab
