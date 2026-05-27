#include <gpt2lab/ops/ops.hpp>
#include <gpt2lab/backends/cpu_naive.hpp>
#include <gpt2lab/autograd/tape.hpp>
#include <stdexcept>
#include <vector>

namespace gpt2lab {

// ---------------------------------------------------------
// Add Op
// ---------------------------------------------------------
Tensor add(const Tensor &a, const Tensor &b) {
  if (a.shape() != b.shape()) {
    throw std::invalid_argument("Shapes must match for addition");
  }

  Tensor out(a.shape(), a.dtype());
  out.set_requires_grad(a.requires_grad() || b.requires_grad());

  for (size_t i = 0; i < a.size(); ++i) {
    out.data()[i] = a.data()[i] + b.data()[i];
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
// Gelu Op
// ---------------------------------------------------------
Tensor gelu(const Tensor &x) {
  return x; // Stubbed for now
}

// ---------------------------------------------------------
// LayerNorm Op
// ---------------------------------------------------------
Tensor layernorm(const Tensor &x) {
  return x; // Stubbed for now
}

// ---------------------------------------------------------
// Softmax Op
// ---------------------------------------------------------
Tensor softmax(const Tensor &x, int /*dim*/) {
  return x; // Stubbed for now
}

} // namespace gpt2lab
