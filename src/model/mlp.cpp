#include <gpt2lab/model/mlp.hpp>
#include <gpt2lab/model/init.hpp>
#include <gpt2lab/ops/matmul.hpp>
#include <gpt2lab/ops/gelu.hpp>

namespace gpt2lab {

MLP::MLP(const Gpt2Config& config)
    : c_fc_w({(size_t)config.hidden_size, (size_t)(4 * config.hidden_size)}),
      c_fc_b({(size_t)(4 * config.hidden_size)}),
      c_proj_w({(size_t)(4 * config.hidden_size), (size_t)config.hidden_size}),
      c_proj_b({(size_t)config.hidden_size}) {
  
  float stddev = 0.02f;
  init_normal(c_fc_w, 0.0f, stddev); init_zeros(c_fc_b);
  init_normal(c_proj_w, 0.0f, stddev); init_zeros(c_proj_b);
}

Tensor MLP::forward(const Tensor &x) {
  Tensor h = matmul(x, c_fc_w);
  h = gelu(h);
  Tensor out = matmul(h, c_proj_w);
  return out;
}

} // namespace gpt2lab
