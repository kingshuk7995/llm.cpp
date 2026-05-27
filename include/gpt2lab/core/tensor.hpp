#pragma once

#include <gpt2lab/core/dtype.hpp>
#include <gpt2lab/core/shape.hpp>
#include <memory>
#include <vector>

namespace gpt2lab {

class Tensor {
public:
  Tensor() = default;
  Tensor(const Shape &shape, DType dtype = DType::Float32);

  DType dtype() const { return dtype_; }

  float *data() { return data_->data(); }
  const float *data() const { return data_->data(); }
  const Shape &shape() const { return shape_; }
  size_t size() const { return data_->size(); }

  void fill(float value);
  void init_normal(float mean, float stddev, unsigned int seed = 42);

  void set_requires_grad(bool req) {
    requires_grad_ = req;
    if (req) init_grad();
  }
  bool requires_grad() const { return requires_grad_; }
  
  void init_grad();
  Tensor& grad();
  const Tensor& grad() const;
  bool has_grad() const { return grad_ != nullptr; }

private:
  Shape shape_;
  DType dtype_ = DType::Float32;
  std::shared_ptr<std::vector<float>> data_;
  
  bool requires_grad_ = false;
  std::shared_ptr<Tensor> grad_;
};

} // namespace gpt2lab
