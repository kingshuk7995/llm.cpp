#pragma once

#include <gpt2lab/backends/backend.hpp>
#include <cublas_v2.h>

namespace gpt2lab {

class CublasBackend : public Backend {
public:
  CublasBackend();
  ~CublasBackend();

  std::string name() const override;
  Tensor matmul(const Tensor &a, const Tensor &b) override;

private:
  cublasHandle_t handle_;
};

} // namespace gpt2lab
