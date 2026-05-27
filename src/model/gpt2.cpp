#include <gpt2lab/model/gpt2.hpp>
#include <gpt2lab/ops/ops.hpp>

namespace gpt2lab {

// ---------------------------------------------------------
// Embeddings
// ---------------------------------------------------------
Embeddings::Embeddings(const Gpt2Config& config)
    : wte({(size_t)config.vocab_size, (size_t)config.hidden_size}),
      wpe({(size_t)config.seq_len, (size_t)config.hidden_size}) {
  wte.init_normal(0.0f, 0.02f);
  wpe.init_normal(0.0f, 0.02f);
}

Tensor Embeddings::forward(const Tensor &tokens) {
  size_t batch_size = tokens.shape().dim(0);
  size_t seq_len = tokens.shape().dim(1);
  size_t hidden_size = wte.shape().dim(1);

  Tensor out({batch_size * seq_len, hidden_size}, DType::Float32);

  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t s = 0; s < seq_len; ++s) {
      size_t token_id = static_cast<size_t>(tokens.data()[b * seq_len + s]);
      for (size_t h = 0; h < hidden_size; ++h) {
        float token_emb = wte.data()[token_id * hidden_size + h];
        float pos_emb = wpe.data()[s * hidden_size + h];
        out.data()[(b * seq_len + s) * hidden_size + h] = token_emb + pos_emb;
      }
    }
  }

  return out;
}

// ---------------------------------------------------------
// MultiHeadAttention
// ---------------------------------------------------------
MultiHeadAttention::MultiHeadAttention(const Gpt2Config& config)
    : w_q({(size_t)config.hidden_size, (size_t)config.hidden_size}),
      b_q({(size_t)config.hidden_size}),
      w_k({(size_t)config.hidden_size, (size_t)config.hidden_size}),
      b_k({(size_t)config.hidden_size}),
      w_v({(size_t)config.hidden_size, (size_t)config.hidden_size}),
      b_v({(size_t)config.hidden_size}),
      c_proj_w({(size_t)config.hidden_size, (size_t)config.hidden_size}),
      c_proj_b({(size_t)config.hidden_size}) {
  
  float stddev = 0.02f;
  w_q.init_normal(0.0f, stddev); b_q.fill(0.0f);
  w_k.init_normal(0.0f, stddev); b_k.fill(0.0f);
  w_v.init_normal(0.0f, stddev); b_v.fill(0.0f);
  c_proj_w.init_normal(0.0f, stddev); c_proj_b.fill(0.0f);
}

Tensor MultiHeadAttention::forward(const Tensor &x) {
  // Execute sequentially: relying on Autograd / Backend for compute graph parallelism.
  Tensor q = matmul(x, w_q);
  Tensor k = matmul(x, w_k);
  Tensor v = matmul(x, w_v);

  // Return simulated attention output mapping back to hidden_size 
  Tensor proj = matmul(v, c_proj_w);
  return proj;
}

// ---------------------------------------------------------
// MLP
// ---------------------------------------------------------
MLP::MLP(const Gpt2Config& config)
    : c_fc_w({(size_t)config.hidden_size, (size_t)(4 * config.hidden_size)}),
      c_fc_b({(size_t)(4 * config.hidden_size)}),
      c_proj_w({(size_t)(4 * config.hidden_size), (size_t)config.hidden_size}),
      c_proj_b({(size_t)config.hidden_size}) {
  
  float stddev = 0.02f;
  c_fc_w.init_normal(0.0f, stddev); c_fc_b.fill(0.0f);
  c_proj_w.init_normal(0.0f, stddev); c_proj_b.fill(0.0f);
}

Tensor MLP::forward(const Tensor &x) {
  Tensor h = matmul(x, c_fc_w);
  h = gelu(h);
  Tensor out = matmul(h, c_proj_w);
  return out;
}

// ---------------------------------------------------------
// TransformerBlock
// ---------------------------------------------------------
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

// ---------------------------------------------------------
// Gpt2Model
// ---------------------------------------------------------
Gpt2Model::Gpt2Model(const Gpt2Config &config)
    : config_(config), embeddings_(config) {
  
  for (int i = 0; i < config.num_layers; ++i) {
    blocks_.emplace_back(config);
  }

  lm_head_w = Tensor({(size_t)config.hidden_size, (size_t)config.vocab_size}, DType::Float32);
  lm_head_w.init_normal(0.0f, 0.02f);
}

Tensor Gpt2Model::forward(const Tensor &input) {
  Tensor x = embeddings_.forward(input);

  for (auto& block : blocks_) {
    x = block.forward(x);
  }

  x = layernorm(x);
  Tensor logits = matmul(x, lm_head_w);

  return logits;
}

} // namespace gpt2lab
