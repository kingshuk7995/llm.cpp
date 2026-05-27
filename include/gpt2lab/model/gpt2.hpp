#pragma once

#include <gpt2lab/core/tensor.hpp>
#include <gpt2lab/model/gpt2_config.hpp>
#include <vector>

namespace gpt2lab {

class Embeddings {
public:
  Embeddings(const Gpt2Config& config);
  Tensor forward(const Tensor &tokens);

  Tensor wte;
  Tensor wpe;
};

class MultiHeadAttention {
public:
  MultiHeadAttention(const Gpt2Config& config);
  Tensor forward(const Tensor &x);

  Tensor w_q, b_q;
  Tensor w_k, b_k;
  Tensor w_v, b_v;
  Tensor c_proj_w, c_proj_b;
};

class MLP {
public:
  MLP(const Gpt2Config& config);
  Tensor forward(const Tensor &x);

  Tensor c_fc_w, c_fc_b;
  Tensor c_proj_w, c_proj_b;
};

class TransformerBlock {
public:
  TransformerBlock(const Gpt2Config& config);
  Tensor forward(const Tensor &x);

  MultiHeadAttention attn;
  MLP mlp;
};

class Gpt2Model {
public:
  explicit Gpt2Model(const Gpt2Config &config);
  Tensor forward(const Tensor &input);

private:
  Gpt2Config config_;
  Embeddings embeddings_;
  std::vector<TransformerBlock> blocks_;
  Tensor lm_head_w;
};

} // namespace gpt2lab
