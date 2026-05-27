#include <gpt2lab/backends/cublas_backend.hpp>

#include <stdexcept>

namespace gpt2lab {

CublasBackend::CublasBackend() {
  if (cublasCreate(&handle_) != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("Failed to initialize cuBLAS");
  }
}

CublasBackend::~CublasBackend() {
  cublasDestroy(handle_);
}

std::string CublasBackend::name() const { return "cublas"; }

Tensor CublasBackend::matmul(const Tensor &a, const Tensor &b) {
  return Tensor(Shape({a.shape().dim(0), b.shape().dim(1)}), DType::Float32);
}

} // namespace gpt2lab
