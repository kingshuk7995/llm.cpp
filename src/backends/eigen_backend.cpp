#include <gpt2lab/backends/eigen_backend.hpp>
#include <Eigen/Dense>

namespace gpt2lab {

std::string EigenBackend::name() const { return "eigen"; }

Tensor EigenBackend::matmul(const Tensor &a, const Tensor &b) {
  size_t m = a.shape().dim(0);
  size_t k = a.shape().dim(1);
  size_t n = b.shape().dim(1);

  Tensor out(Shape({m, n}), DType::Float32);

  using MatrixRM = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  Eigen::Map<const MatrixRM> map_a(a.data(), m, k);
  Eigen::Map<const MatrixRM> map_b(b.data(), k, n);
  Eigen::Map<MatrixRM> map_out(out.data(), m, n);

  map_out.noalias() = map_a * map_b;

  return out;
}

} // namespace gpt2lab
