#include <gpt2lab/autograd/backward.hpp>

#include <gpt2lab/autograd/tape.hpp>

namespace gpt2lab {

Tensor backward(Tensor &loss) {
  loss.init_grad();
  loss.grad().fill(1.0f); // seed gradient

  const auto& nodes = Tape::global().nodes();
  for (auto it = nodes.rbegin(); it != nodes.rend(); ++it) {
    if (it->backward_closure) {
      it->backward_closure();
    }
  }
  
  return loss.grad();
}

} // namespace gpt2lab
