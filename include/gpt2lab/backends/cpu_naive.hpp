#pragma once

#include <gpt2lab/backends/backend.hpp>

namespace gpt2lab {

class CpuNaiveBackend : public Backend {
public:
  std::string name() const override;
  Tensor matmul(const Tensor &a, const Tensor &b) override;
  Tensor add(const Tensor &a, const Tensor &b) override;
  Tensor gelu(const Tensor &x) override;
  Tensor softmax(const Tensor &x) override;
  Tensor layernorm(const Tensor &x) override;
};

} // namespace gpt2lab
