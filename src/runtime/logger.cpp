#include <gpt2lab/runtime/logger.hpp>
#include <iostream>

namespace gpt2lab {

void log_info(const std::string &message) {
  std::cout << "[gpt2lab] " << message << std::endl;
}

} // namespace gpt2lab
