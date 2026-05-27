#include <gpt2lab/ops/add.hpp>

#include <gpt2lab/backends/cpu_naive.hpp>
#include <gpt2lab/autograd/tape.hpp>

namespace gpt2lab {

Tensor add(const Tensor &a, const Tensor &b) {
  CpuNaiveBackend backend;
  Tensor out = backend.add(a, b);
  
  if (a.requires_grad() || b.requires_grad()) {
    out.set_requires_grad(true);
    
    Tape::global().record([out_copy = out, a_copy = a, b_copy = b]() mutable {
      if (a_copy.requires_grad()) {
        a_copy.init_grad();
        CpuNaiveBackend cpu_backend;
        a_copy.grad() = cpu_backend.add(a_copy.grad(), out_copy.grad());
      }
      if (b_copy.requires_grad()) {
        b_copy.init_grad();
        CpuNaiveBackend cpu_backend;
        b_copy.grad() = cpu_backend.add(b_copy.grad(), out_copy.grad());
      }
    });
  }
  
  return out;
}

} // namespace gpt2lab
