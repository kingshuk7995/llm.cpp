#pragma once

#include <gpt2lab/core/tensor.hpp>
#include <vector>
#include <functional>

namespace gpt2lab {

struct Node {
  std::function<void()> backward_closure;
};

class Tape {
public:
  static Tape& global();

  void record(std::function<void()> backward_closure);
  void clear();
  
  const std::vector<Node>& nodes() const { return nodes_; }

private:
  std::vector<Node> nodes_;
};

} // namespace gpt2lab
