#include <gpt2lab/model/gpt2.hpp>
#include <gpt2lab/ops/ops.hpp>
#include <fstream>
#include <stdexcept>

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
// LayerNorm Module
// ---------------------------------------------------------
LayerNorm::LayerNorm(const Gpt2Config& config)
    : gamma({(size_t)config.hidden_size}),
      beta({(size_t)config.hidden_size}) {
  gamma.fill(1.0f);
  beta.fill(0.0f);
}

Tensor LayerNorm::forward(const Tensor &x) {
  return layernorm(x, gamma, beta, 1e-5f);
}

// ---------------------------------------------------------
// MultiHeadAttention
// ---------------------------------------------------------
MultiHeadAttention::MultiHeadAttention(const Gpt2Config& config)
    : config_(config),
      w_q({(size_t)config.hidden_size, (size_t)config.hidden_size}),
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

static Tensor transpose_for_mha(const Tensor& x, size_t batch, size_t seq, size_t num_heads, size_t head_dim) {
  Tensor out(Shape({batch, num_heads, seq, head_dim}), x.dtype());
  const float* in_ptr = x.data();
  float* out_ptr = out.data();
  for (size_t b = 0; b < batch; ++b) {
    for (size_t s = 0; s < seq; ++s) {
      for (size_t h = 0; h < num_heads; ++h) {
        for (size_t d = 0; d < head_dim; ++d) {
          out_ptr[b * (num_heads * seq * head_dim) + h * (seq * head_dim) + s * head_dim + d] =
              in_ptr[b * (seq * num_heads * head_dim) + s * (num_heads * head_dim) + h * head_dim + d];
        }
      }
    }
  }
  return out;
}

static Tensor merge_heads(const Tensor& x, size_t batch, size_t seq, size_t num_heads, size_t head_dim) {
  Tensor out(Shape({batch, seq, num_heads * head_dim}), x.dtype());
  const float* in_ptr = x.data();
  float* out_ptr = out.data();
  for (size_t b = 0; b < batch; ++b) {
    for (size_t h = 0; h < num_heads; ++h) {
      for (size_t s = 0; s < seq; ++s) {
        for (size_t d = 0; d < head_dim; ++d) {
          out_ptr[b * (seq * num_heads * head_dim) + s * (num_heads * head_dim) + h * head_dim + d] =
              in_ptr[b * (num_heads * seq * head_dim) + h * (seq * head_dim) + s * head_dim + d];
        }
      }
    }
  }
  return out;
}

Tensor MultiHeadAttention::forward(const Tensor &x) {
  size_t batch = x.shape().dim(0);
  size_t seq = x.shape().dim(1);
  size_t num_heads = config_.num_heads;
  size_t head_dim = config_.hidden_size / num_heads;

  Tensor q = add(matmul(x, w_q), b_q);
  Tensor k = add(matmul(x, w_k), b_k);
  Tensor v = add(matmul(x, w_v), b_v);

  Tensor q_t = transpose_for_mha(q, batch, seq, num_heads, head_dim);
  Tensor k_t = transpose_for_mha(k, batch, seq, num_heads, head_dim);
  Tensor v_t = transpose_for_mha(v, batch, seq, num_heads, head_dim);

  Tensor scores = batched_matmul(q_t, k_t, true);

  // Scale and causal mask
  float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  float* s_ptr = scores.data();
  for (size_t b = 0; b < batch; ++b) {
    for (size_t h = 0; h < num_heads; ++h) {
      for (size_t s1 = 0; s1 < seq; ++s1) {
        for (size_t s2 = 0; s2 < seq; ++s2) {
          size_t idx = b * (num_heads * seq * seq) + h * (seq * seq) + s1 * seq + s2;
          if (s2 > s1) {
            s_ptr[idx] = -1e9f;
          } else {
            s_ptr[idx] *= scale;
          }
        }
      }
    }
  }

  Tensor probs = softmax(scores, -1);
  Tensor context = batched_matmul(probs, v_t, false);
  
  Tensor merged = merge_heads(context, batch, seq, num_heads, head_dim);
  
  Tensor proj = add(matmul(merged, c_proj_w), c_proj_b);
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
  Tensor h = add(matmul(x, c_fc_w), c_fc_b);
  h = gelu(h);
  Tensor out = add(matmul(h, c_proj_w), c_proj_b);
  return out;
}

// ---------------------------------------------------------
// TransformerBlock
// ---------------------------------------------------------
TransformerBlock::TransformerBlock(const Gpt2Config& config)
    : ln_1(config), attn(config), ln_2(config), mlp(config) {}

Tensor TransformerBlock::forward(const Tensor &x) {
  Tensor ln1_out = ln_1.forward(x);
  Tensor a = attn.forward(ln1_out);
  Tensor h1 = add(x, a);

  Tensor ln2_out = ln_2.forward(h1);
  Tensor m = mlp.forward(ln2_out);
  Tensor h2 = add(h1, m);

  return h2;
}

// ---------------------------------------------------------
// Gpt2Model
// ---------------------------------------------------------
Gpt2Model::Gpt2Model(const Gpt2Config &config)
    : config_(config), embeddings_(config), ln_f(config) {
  
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

  x = ln_f.forward(x);
  Tensor logits = matmul(x, lm_head_w);

  return logits;
}

void Gpt2Model::load_weights(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f.is_open()) {
    throw std::runtime_error("Failed to open weights file: " + path);
  }

  auto read_tensor = [&f](Tensor& t) {
    f.read(reinterpret_cast<char*>(t.data()), t.size() * sizeof(float));
  };

  read_tensor(embeddings_.wte);
  read_tensor(embeddings_.wpe);

  for (auto& block : blocks_) {
    read_tensor(block.ln_1.gamma);
    read_tensor(block.ln_1.beta);

    read_tensor(block.attn.w_q);
    read_tensor(block.attn.b_q);
    read_tensor(block.attn.w_k);
    read_tensor(block.attn.b_k);
    read_tensor(block.attn.w_v);
    read_tensor(block.attn.b_v);

    read_tensor(block.attn.c_proj_w);
    read_tensor(block.attn.c_proj_b);

    read_tensor(block.ln_2.gamma);
    read_tensor(block.ln_2.beta);

    read_tensor(block.mlp.c_fc_w);
    read_tensor(block.mlp.c_fc_b);
    read_tensor(block.mlp.c_proj_w);
    read_tensor(block.mlp.c_proj_b);
  }

  read_tensor(ln_f.gamma);
  read_tensor(ln_f.beta);
  read_tensor(lm_head_w);
}

} // namespace gpt2lab
