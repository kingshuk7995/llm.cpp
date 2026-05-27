#include <gpt2lab/model/init.hpp>
#include <random>

namespace gpt2lab {

void init_normal(Tensor& tensor, float mean, float stddev) {
  // TODO: this function can/should be in the tensor 
  // TODO: make this 42 state configurable at compile time
  std::mt19937 gen(42); // Fixed seed for reproducibility
  std::normal_distribution<float> dist(mean, stddev);
  float* data = tensor.data();
  for (size_t i = 0; i < tensor.size(); ++i) {
    data[i] = dist(gen);
  }
}

void init_zeros(Tensor& tensor) {
  tensor.fill(0.0f);
}

} // namespace gpt2lab
