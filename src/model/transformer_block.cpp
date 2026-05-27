#include <gpt2lab/model/transformer_block.hpp>
#include <gpt2lab/ops/layernorm.hpp>
#include <gpt2lab/ops/add.hpp>

namespace gpt2lab {

TransformerBlock::TransformerBlock(const Gpt2Config& config)
    : attn(config), mlp(config) {}

Tensor TransformerBlock::forward(const Tensor &x) {
  Tensor ln1 = layernorm(x);
  Tensor a = attn.forward(ln1);
  Tensor h1 = add(x, a);

  Tensor ln2 = layernorm(h1);
  Tensor m = mlp.forward(ln2);
  Tensor h2 = add(h1, m);

  return h2;
}

} // namespace gpt2lab
