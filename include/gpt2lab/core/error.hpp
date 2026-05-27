#pragma once

#include <string>

namespace gpt2lab {

struct Error {
  std::string message;
};

inline Error make_error(const char *message) { return Error{message}; }

} // namespace gpt2lab
