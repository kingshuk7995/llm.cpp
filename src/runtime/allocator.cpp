#include <gpt2lab/runtime/allocator.hpp>
#include <cstdlib>

namespace gpt2lab {

void *Allocator::allocate(size_t bytes) { return std::malloc(bytes); }

void Allocator::deallocate(void *ptr) { std::free(ptr); }

} // namespace gpt2lab
