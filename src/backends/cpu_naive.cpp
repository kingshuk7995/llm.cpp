#include <gpt2lab/backends/cpu_naive.hpp>
#include <cmath>
namespace gpt2lab {

std::string CpuNaiveBackend::name() const { return "cpu_naive"; }

Tensor CpuNaiveBackend::matmul(const Tensor &a, const Tensor &b) {
  // Simple N-dim to 2D flattening
  size_t m = a.shape().dim(0); // assuming 2D for now in naive matmul
  size_t k = a.shape().dim(1);
  size_t n = b.shape().dim(1);
  
  Shape out_shape({m, n});
  Tensor out(out_shape, DType::Float32);
  out.fill(0.0f);
  
  const float* a_data = a.data();
  const float* b_data = b.data();
  float* out_data = out.data();

  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      float sum = 0.0f;
      for (size_t l = 0; l < k; ++l) {
        sum += a_data[i * k + l] * b_data[l * n + j];
      }
      out_data[i * n + j] = sum;
    }
  }
  return out;
}

Tensor CpuNaiveBackend::add(const Tensor &a, const Tensor &b) {
  Tensor out(a.shape(), DType::Float32);
  const float* a_data = a.data();
  const float* b_data = b.data();
  float* out_data = out.data();
  
  size_t n = a.size();
  for (size_t i = 0; i < n; ++i) {
    out_data[i] = a_data[i] + b_data[i];
  }
  return out;
}

Tensor CpuNaiveBackend::gelu(const Tensor &x) {
  Tensor out(x.shape(), DType::Float32);
  const float* x_data = x.data();
  float* out_data = out.data();
  
  size_t n = x.size();
  for (size_t i = 0; i < n; ++i) {
    float val = x_data[i];
    // Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    float cdf = 0.5f * (1.0f + std::tanh(0.7978845608f * (val + 0.044715f * val * val * val)));
    out_data[i] = val * cdf;
  }
  return out;
}

Tensor CpuNaiveBackend::softmax(const Tensor &x) {
  Tensor out(x.shape(), DType::Float32);
  // Softmax over last dimension
  size_t last_dim = x.shape().dim(x.shape().rank() - 1);
  size_t outer_dims = x.size() / last_dim;
  
  const float* x_data = x.data();
  float* out_data = out.data();

  for (size_t i = 0; i < outer_dims; ++i) {
    const float* x_row = x_data + i * last_dim;
    float* out_row = out_data + i * last_dim;
    
    float max_val = x_row[0];
    for (size_t j = 1; j < last_dim; ++j) {
      if (x_row[j] > max_val) max_val = x_row[j];
    }
    
    float sum = 0.0f;
    for (size_t j = 0; j < last_dim; ++j) {
      out_row[j] = std::exp(x_row[j] - max_val);
      sum += out_row[j];
    }
    
    for (size_t j = 0; j < last_dim; ++j) {
      out_row[j] /= sum;
    }
  }
  return out;
}

Tensor CpuNaiveBackend::layernorm(const Tensor &x) {
  Tensor out(x.shape(), DType::Float32);
  size_t last_dim = x.shape().dim(x.shape().rank() - 1);
  size_t outer_dims = x.size() / last_dim;
  
  const float* x_data = x.data();
  float* out_data = out.data();

  for (size_t i = 0; i < outer_dims; ++i) {
    const float* x_row = x_data + i * last_dim;
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
    
    float stddev_inv = 1.0f / std::sqrt(var + 1e-5f);
    for (size_t j = 0; j < last_dim; ++j) {
      out_row[j] = (x_row[j] - mean) * stddev_inv;
    }
  }
  return out;
}

} // namespace gpt2lab
