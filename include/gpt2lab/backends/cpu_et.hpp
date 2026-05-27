#pragma once

#include <gpt2lab/backends/backend.hpp>

namespace gpt2lab {

class CpuEtBackend : public Backend {
public:
  std::string name() const override;
  Tensor matmul(const Tensor &a, const Tensor &b) override;
};

} // namespace gpt2lab
