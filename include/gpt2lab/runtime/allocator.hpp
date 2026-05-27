#pragma once

#include <cstddef>

namespace gpt2lab {

class Allocator {
public:
  void *allocate(size_t bytes);
  void deallocate(void *ptr);
};

} // namespace gpt2lab
