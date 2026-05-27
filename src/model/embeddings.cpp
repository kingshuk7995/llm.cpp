#include <gpt2lab/model/embeddings.hpp>
#include <gpt2lab/model/init.hpp>
#include <gpt2lab/ops/add.hpp>

namespace gpt2lab {

Embeddings::Embeddings(const Gpt2Config& config)
    : wte({(size_t)config.vocab_size, (size_t)config.hidden_size}),
      wpe({(size_t)config.seq_len, (size_t)config.hidden_size}) {
  init_normal(wte, 0.0f, 0.02f);
  init_normal(wpe, 0.0f, 0.02f);
}

Tensor Embeddings::forward(const Tensor &tokens) {
  size_t batch_size = tokens.shape().dim(0);
  size_t seq_len = tokens.shape().dim(1);
  size_t hidden_size = wte.shape().dim(1);

  // Return a flattened 2D tensor [batch_size * seq_len, hidden_size]
  // This allows 2D matmuls in subsequent layers!
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

} // namespace gpt2lab
