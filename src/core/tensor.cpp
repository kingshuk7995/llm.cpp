#include <gpt2lab/core/tensor.hpp>
#include <random>

namespace gpt2lab {

Tensor::Tensor(const Shape &shape, DType dtype)
    : shape_(shape), dtype_(dtype), data_(std::make_shared<std::vector<float>>(shape_size(shape), 0.0f)) {}

void Tensor::fill(float value) {
  for (size_t i = 0; i < data_->size(); ++i) {
    (*data_)[i] = value;
  }
}

void Tensor::init_normal(float mean, float stddev, unsigned int seed) {
  if (!data_) return;
  std::mt19937 gen(seed);
  std::normal_distribution<float> dist(mean, stddev);
  for (size_t i = 0; i < data_->size(); ++i) {
    (*data_)[i] = dist(gen);
  }
}

void Tensor::init_grad() {
  if (!grad_) {
    grad_ = std::make_shared<Tensor>(shape_, dtype_);
    grad_->fill(0.0f);
  }
}

Tensor& Tensor::grad() {
  if (!grad_) init_grad();
  return *grad_;
}

const Tensor& Tensor::grad() const {
  if (!grad_) {
    throw std::runtime_error("Gradient not initialized");
  }
  return *grad_;
}

} // namespace gpt2lab
