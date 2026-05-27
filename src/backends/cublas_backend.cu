#include <gpt2lab/backends/cublas_backend.hpp>

#include <stdexcept>
#include <cuda_runtime.h>

#define CUDA_CHECK(val) { \
  if (val != cudaSuccess) { \
    throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(val))); \
  } \
}

#define CUBLAS_CHECK(val) { \
  if (val != CUBLAS_STATUS_SUCCESS) { \
    throw std::runtime_error("cuBLAS Error"); \
  } \
}

namespace gpt2lab {

CublasBackend::CublasBackend() {
  CUBLAS_CHECK(cublasCreate(&handle_));
}

CublasBackend::~CublasBackend() {
  cublasDestroy(handle_);
}

std::string CublasBackend::name() const { return "cublas"; }

Tensor CublasBackend::matmul(const Tensor &a, const Tensor &b) {
  size_t m = a.shape().dim(0);
  size_t k = a.shape().dim(1);
  size_t n = b.shape().dim(1);

  Tensor out(Shape({m, n}), DType::Float32);

  float *d_a, *d_b, *d_out;
  size_t size_a = m * k * sizeof(float);
  size_t size_b = k * n * sizeof(float);
  size_t size_out = m * n * sizeof(float);

  CUDA_CHECK(cudaMalloc(&d_a, size_a));
  CUDA_CHECK(cudaMalloc(&d_b, size_b));
  CUDA_CHECK(cudaMalloc(&d_out, size_out));

  CUDA_CHECK(cudaMemcpy(d_a, a.data(), size_a, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, b.data(), size_b, cudaMemcpyHostToDevice));

  const float alpha = 1.0f;
  const float beta = 0.0f;

  CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                           n, m, k,
                           &alpha,
                           d_b, n,
                           d_a, k,
                           &beta,
                           d_out, n));

  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(out.data(), d_out, size_out, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_out));

  return out;
}

} // namespace gpt2lab
