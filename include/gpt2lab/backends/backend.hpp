#pragma once

#include <gpt2lab/core/tensor.hpp>
#include <string>
#include <stdexcept>

namespace gpt2lab {

class Backend {
public:
  virtual ~Backend() = default;
  virtual std::string name() const = 0;
  virtual Tensor matmul(const Tensor &a, const Tensor &b) = 0;
  virtual Tensor add(const Tensor &a, const Tensor &b) { throw std::logic_error("Not implemented"); }
  virtual Tensor gelu(const Tensor &x) { throw std::logic_error("Not implemented"); }
  virtual Tensor softmax(const Tensor &x) { throw std::logic_error("Not implemented"); }
  virtual Tensor layernorm(const Tensor &x) { throw std::logic_error("Not implemented"); }
};

} // namespace gpt2lab
