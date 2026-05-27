#pragma once

#include <cstddef>
#include <vector>
#include <initializer_list>
#include <numeric>
#include <string>
#include <sstream>

namespace gpt2lab {

struct Shape {
  std::vector<size_t> dims;

  Shape() = default;
  Shape(std::initializer_list<size_t> d) : dims(d) {}
  explicit Shape(const std::vector<size_t>& d) : dims(d) {}

  size_t rank() const { return dims.size(); }
  size_t dim(size_t i) const { return dims[i]; }
  
  bool operator==(const Shape& other) const {
    return dims == other.dims;
  }
  
  bool operator!=(const Shape& other) const {
    return dims != other.dims;
  }

  std::string to_string() const {
    std::stringstream ss;
    ss << "(";
    for (size_t i = 0; i < dims.size(); ++i) {
      ss << dims[i];
      if (i < dims.size() - 1) ss << ", ";
    }
    ss << ")";
    return ss.str();
  }
};

inline size_t shape_size(const Shape &shape) {
  if (shape.dims.empty()) return 0;
  size_t size = 1;
  for (auto d : shape.dims) size *= d;
  return size;
}

} // namespace gpt2lab
