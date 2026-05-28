#include <gpt2lab/ops/ops.hpp>
#include <gpt2lab/backends/cpu_naive.hpp>
#include <gpt2lab/autograd/tape.hpp>
#include <stdexcept>
#include <vector>
#include <cmath>

namespace gpt2lab {

// ---------------------------------------------------------
// Add Op
// ---------------------------------------------------------
Tensor add(const Tensor &a, const Tensor &b) {
  if (a.shape() != b.shape() && !(b.shape().rank() == 1 && b.size() == a.shape().dim(a.shape().rank() - 1))) {
    throw std::invalid_argument("Shapes must match or b must be a 1D tensor matching the last dimension of a");
  }

  Tensor out(a.shape(), a.dtype());
  out.set_requires_grad(a.requires_grad() || b.requires_grad());

  if (a.shape() == b.shape()) {
    for (size_t i = 0; i < a.size(); ++i) {
      out.data()[i] = a.data()[i] + b.data()[i];
    }
  } else {
    // Broadcast b over the last dimension of a
    size_t last_dim = b.size();
    size_t outer_dims = a.size() / last_dim;
    for (size_t i = 0; i < outer_dims; ++i) {
      for (size_t j = 0; j < last_dim; ++j) {
        out.data()[i * last_dim + j] = a.data()[i * last_dim + j] + b.data()[j];
      }
    }
  }

  if (out.requires_grad()) {
    Tape::global().record([a = a, b = b, out = out]() mutable {
      if (out.has_grad()) {
        const float *grad_out = out.grad().data();
        if (a.requires_grad()) {
          float *grad_a = a.grad().data();
          for (size_t i = 0; i < a.size(); ++i) {
            grad_a[i] += grad_out[i];
          }
        }
        if (b.requires_grad()) {
          float *grad_b = b.grad().data();
          for (size_t i = 0; i < b.size(); ++i) {
            grad_b[i] += grad_out[i];
          }
        }
      }
    });
  }

  return out;
}

// ---------------------------------------------------------
// Matmul Op
// ---------------------------------------------------------
Tensor matmul(const Tensor &a, const Tensor &b) {
  CpuNaiveBackend backend;
  return backend.matmul(a, b);
}

// ---------------------------------------------------------
// Batched Matmul Op (for MHA)
// ---------------------------------------------------------
Tensor batched_matmul(const Tensor &a, const Tensor &b, bool trans_b) {
  // Assumes a: [..., M, K], b: [..., K, N] (if trans_b=false) or [..., N, K] (if trans_b=true)
  size_t rank_a = a.shape().rank();
  size_t rank_b = b.shape().rank();
  
  size_t M = a.shape().dim(rank_a - 2);
  size_t K = a.shape().dim(rank_a - 1);
  size_t N = trans_b ? b.shape().dim(rank_b - 2) : b.shape().dim(rank_b - 1);
  
  size_t batch_size = 1;
  for (size_t i = 0; i < rank_a - 2; ++i) batch_size *= a.shape().dim(i);
  
  std::vector<size_t> out_dims = a.shape().dims;
  out_dims[rank_a - 2] = M;
  out_dims[rank_a - 1] = N;
  Tensor out(Shape(out_dims), a.dtype());
  
  const float* a_ptr = a.data();
  const float* b_ptr = b.data();
  float* out_ptr = out.data();
  
  // Simple unoptimized batched matmul (we'll optimize this later if needed)
  for (size_t b_idx = 0; b_idx < batch_size; ++b_idx) {
    const float* a_batch = a_ptr + b_idx * M * K;
    const float* b_batch = b_ptr + b_idx * K * N;
    float* out_batch = out_ptr + b_idx * M * N;
    
    for (size_t i = 0; i < M; ++i) {
      for (size_t j = 0; j < N; ++j) {
        float sum = 0.0f;
        for (size_t k = 0; k < K; ++k) {
          float a_val = a_batch[i * K + k];
          float b_val = trans_b ? b_batch[j * K + k] : b_batch[k * N + j];
          sum += a_val * b_val;
        }
        out_batch[i * N + j] = sum;
      }
    }
  }
  
  return out;
}

// ---------------------------------------------------------
// Gelu Op
// ---------------------------------------------------------
Tensor gelu(const Tensor &x) {
  Tensor out(x.shape(), x.dtype());
  const float* in_data = x.data();
  float* out_data = out.data();
  
  for (size_t i = 0; i < x.size(); ++i) {
    float val = in_data[i];
    out_data[i] = 0.5f * val * (1.0f + std::tanh(0.7978845608f * (val + 0.044715f * val * val * val)));
  }
  return out;
}

// ---------------------------------------------------------
// LayerNorm Op
// ---------------------------------------------------------
Tensor layernorm(const Tensor &x, const Tensor &gamma, const Tensor &beta, float eps) {
  Tensor out(x.shape(), x.dtype());
  
  size_t last_dim = x.shape().dim(x.shape().rank() - 1);
  size_t outer_dims = x.size() / last_dim;
  
  const float* in_data = x.data();
  float* out_data = out.data();
  const float* g_data = gamma.data();
  const float* b_data = beta.data();
  
  for (size_t i = 0; i < outer_dims; ++i) {
    const float* x_row = in_data + i * last_dim;
    float* out_row = out_data + i * last_dim;
    
    float mean = 0.0f;
    for (size_t j = 0; j < last_dim; ++j) {
      mean += x_row[j];
    }
    mean /= last_dim;
    
    float var = 0.0f;
    for (size_t j = 0; j < last_dim; ++j) {
      float diff = x_row[j] - mean;
      var += diff * diff;
    }
    var /= last_dim;
    
    float inv_std = 1.0f / std::sqrt(var + eps);
    
    for (size_t j = 0; j < last_dim; ++j) {
      out_row[j] = (x_row[j] - mean) * inv_std * g_data[j] + b_data[j];
    }
  }
  
  return out;
}

// ---------------------------------------------------------
// Softmax Op
// ---------------------------------------------------------
Tensor softmax(const Tensor &x, int dim) {
  Tensor out(x.shape(), x.dtype());
  
  // For simplicity, assume dim = -1
  size_t last_dim = x.shape().dim(x.shape().rank() - 1);
  size_t outer_dims = x.size() / last_dim;
  
  const float* in_data = x.data();
  float* out_data = out.data();
  
  for (size_t i = 0; i < outer_dims; ++i) {
    const float* x_row = in_data + i * last_dim;
    float* out_row = out_data + i * last_dim;
    
    float max_val = -1e9f;
    for (size_t j = 0; j < last_dim; ++j) {
      if (x_row[j] > max_val) max_val = x_row[j];
    }
    
    float sum_exp = 0.0f;
    for (size_t j = 0; j < last_dim; ++j) {
      float e = std::exp(x_row[j] - max_val);
      out_row[j] = e;
      sum_exp += e;
    }
    
    for (size_t j = 0; j < last_dim; ++j) {
      out_row[j] /= sum_exp;
    }
  }
  
  return out;
}

} // namespace gpt2lab
