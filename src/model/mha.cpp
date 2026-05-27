#include <gpt2lab/model/mha.hpp>
#include <gpt2lab/model/init.hpp>
#include <gpt2lab/ops/matmul.hpp>
#include <thread>

namespace gpt2lab {

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
  init_normal(w_q, 0.0f, stddev); init_zeros(b_q);
  init_normal(w_k, 0.0f, stddev); init_zeros(b_k);
  init_normal(w_v, 0.0f, stddev); init_zeros(b_v);
  init_normal(c_proj_w, 0.0f, stddev); init_zeros(c_proj_b);
}

Tensor MultiHeadAttention::forward(const Tensor &x) {
  Tensor q, k, v;

  // Execute independent Q,K,V projections in parallel
  // FIX: This is a wrong model. Make the autograd generally paralizable not only for Autograd
  // FIX: For CUDA parallizatioon we dont need seperate threads.
  std::thread t1([&]() { q = matmul(x, w_q); });
  std::thread t2([&]() { k = matmul(x, w_k); });
  std::thread t3([&]() { v = matmul(x, w_v); });

  t1.join();
  t2.join();
  t3.join();

  // Return simulated attention output mapping back to hidden_size 
  // (Full attention calculation requires batched 3D matmul)
  Tensor proj = matmul(v, c_proj_w);
  return proj;
}

} // namespace gpt2lab
