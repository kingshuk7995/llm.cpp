#pragma once

#include <string>

namespace gpt2lab {

class Profiler {
public:
  void begin(const std::string &name);
  void end(const std::string &name);
};

} // namespace gpt2lab
