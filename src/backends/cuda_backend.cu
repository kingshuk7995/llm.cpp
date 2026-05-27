#include <gpt2lab/backends/cuda_backend.hpp>

#include <cuda_runtime.h>
#include <stdexcept>

#define CUDA_CHECK(val) { \
  if (val != cudaSuccess) { \
    throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(val))); \
  } \
}

namespace gpt2lab {

__global__ void matmul_kernel(const float* a, const float* b, float* out, size_t m, size_t k, size_t n) {
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < m && col < n) {
    float sum = 0.0f;
    for (size_t i = 0; i < k; ++i) {
      sum += a[row * k + i] * b[i * n + col];
    }
    out[row * n + col] = sum;
  }
}

std::string CudaBackend::name() const { return "cuda"; }

Tensor CudaBackend::matmul(const Tensor &a, const Tensor &b) {
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

  dim3 block(16, 16);
  dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);

  matmul_kernel<<<grid, block>>>(d_a, d_b, d_out, m, k, n);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(out.data(), d_out, size_out, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_out));

  return out;
}

} // namespace gpt2lab
