#include <gpt2lab/model/transformer_block.hpp>

namespace gpt2lab {

Tensor TransformerBlock::operator()(const Tensor &x) const { return x; }

} // namespace gpt2lab
