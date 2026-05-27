#pragma once

#include <gpt2lab/bench/metrics.hpp>
#include <string>

namespace gpt2lab {

std::string format_report(const Metrics &metrics);

} // namespace gpt2lab
