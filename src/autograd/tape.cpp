#include <gpt2lab/autograd/tape.hpp>

namespace gpt2lab {

Tape& Tape::global() {
  static Tape instance;
  return instance;
}

void Tape::record(std::function<void()> backward_closure) {
  nodes_.push_back({backward_closure});
}

void Tape::clear() {
  nodes_.clear();
}

} // namespace gpt2lab
