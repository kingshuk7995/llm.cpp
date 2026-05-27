#include <gpt2lab/backends/cpu_et.hpp>
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <omp.h>
#include <stdexcept>

namespace gpt2lab {

static constexpr int MC = 256;
static constexpr int NC = 256;
static constexpr int KC = 256;

#ifndef N_THREADS
#define N_THREADS omp_get_max_threads()
#endif

static inline int round_up8(int x) { return (x + 7) & ~7; }

static inline float *alloc_aligned_floats(std::size_t count) {
  void *p = nullptr;
  if (posix_memalign(&p, 64, count * sizeof(float)) != 0)
    return nullptr;
  return static_cast<float *>(p);
}

static inline void pack_A_8(const float *__restrict__ A, float *__restrict__ Ap,
                            int lda, int mc, int kc) {
  for (int i = 0; i < mc; i += 8) {
    float *Ablock = Ap + (std::size_t)i * kc;
    for (int p = 0; p < kc; ++p) {
      for (int ii = 0; ii < 8; ++ii) {
        Ablock[p * 8 + ii] = (i + ii < mc) ? A[(i + ii) * lda + p] : 0.0f;
      }
    }
  }
}

static inline void pack_B_8(const float *__restrict__ B, float *__restrict__ Bp,
                            int ldb, int kc, int nc) {
  for (int j = 0; j < nc; j += 8) {
    float *Bblock = Bp + (std::size_t)j * kc;
    for (int p = 0; p < kc; ++p) {
      for (int jj = 0; jj < 8; ++jj) {
        Bblock[p * 8 + jj] = (j + jj < nc) ? B[p * ldb + j + jj] : 0.0f;
      }
    }
  }
}

static inline void kernel_8x8(int kc, const float *__restrict__ Ap,
                              const float *__restrict__ Bp,
                              float *__restrict__ C, int ldc, int mc_actual,
                              int nc_actual) {
  __m256 c0 = _mm256_setzero_ps();
  __m256 c1 = _mm256_setzero_ps();
  __m256 c2 = _mm256_setzero_ps();
  __m256 c3 = _mm256_setzero_ps();
  __m256 c4 = _mm256_setzero_ps();
  __m256 c5 = _mm256_setzero_ps();
  __m256 c6 = _mm256_setzero_ps();
  __m256 c7 = _mm256_setzero_ps();

  const float *a_ptr = Ap;
  const float *b_ptr = Bp;

#define KERNEL_CORE                                                            \
  {                                                                            \
    __m256 b0 = _mm256_load_ps(b_ptr);                                         \
    c0 = _mm256_fmadd_ps(_mm256_broadcast_ss(a_ptr + 0), b0, c0);              \
    c1 = _mm256_fmadd_ps(_mm256_broadcast_ss(a_ptr + 1), b0, c1);              \
    c2 = _mm256_fmadd_ps(_mm256_broadcast_ss(a_ptr + 2), b0, c2);              \
    c3 = _mm256_fmadd_ps(_mm256_broadcast_ss(a_ptr + 3), b0, c3);              \
    c4 = _mm256_fmadd_ps(_mm256_broadcast_ss(a_ptr + 4), b0, c4);              \
    c5 = _mm256_fmadd_ps(_mm256_broadcast_ss(a_ptr + 5), b0, c5);              \
    c6 = _mm256_fmadd_ps(_mm256_broadcast_ss(a_ptr + 6), b0, c6);              \
    c7 = _mm256_fmadd_ps(_mm256_broadcast_ss(a_ptr + 7), b0, c7);              \
    a_ptr += 8;                                                                \
    b_ptr += 8;                                                                \
  }

  int p = 0;
  for (; p + 3 < kc; p += 4) {
    KERNEL_CORE
    KERNEL_CORE
    KERNEL_CORE
    KERNEL_CORE
  }
  for (; p < kc; ++p) {
    KERNEL_CORE
  }

#undef KERNEL_CORE

  if (mc_actual == 8 && nc_actual == 8) {
    _mm256_storeu_ps(C + 0 * ldc,
                     _mm256_add_ps(c0, _mm256_loadu_ps(C + 0 * ldc)));
    _mm256_storeu_ps(C + 1 * ldc,
                     _mm256_add_ps(c1, _mm256_loadu_ps(C + 1 * ldc)));
    _mm256_storeu_ps(C + 2 * ldc,
                     _mm256_add_ps(c2, _mm256_loadu_ps(C + 2 * ldc)));
    _mm256_storeu_ps(C + 3 * ldc,
                     _mm256_add_ps(c3, _mm256_loadu_ps(C + 3 * ldc)));
    _mm256_storeu_ps(C + 4 * ldc,
                     _mm256_add_ps(c4, _mm256_loadu_ps(C + 4 * ldc)));
    _mm256_storeu_ps(C + 5 * ldc,
                     _mm256_add_ps(c5, _mm256_loadu_ps(C + 5 * ldc)));
    _mm256_storeu_ps(C + 6 * ldc,
                     _mm256_add_ps(c6, _mm256_loadu_ps(C + 6 * ldc)));
    _mm256_storeu_ps(C + 7 * ldc,
                     _mm256_add_ps(c7, _mm256_loadu_ps(C + 7 * ldc)));
  } else {
    alignas(32) float tmp[64];
    _mm256_store_ps(tmp + 0 * 8, c0);
    _mm256_store_ps(tmp + 1 * 8, c1);
    _mm256_store_ps(tmp + 2 * 8, c2);
    _mm256_store_ps(tmp + 3 * 8, c3);
    _mm256_store_ps(tmp + 4 * 8, c4);
    _mm256_store_ps(tmp + 5 * 8, c5);
    _mm256_store_ps(tmp + 6 * 8, c6);
    _mm256_store_ps(tmp + 7 * 8, c7);

    for (int i = 0; i < mc_actual; ++i) {
      for (int j = 0; j < nc_actual; ++j) {
        C[i * ldc + j] += tmp[i * 8 + j];
      }
    }
  }
}

void matmul_kp(const float *A, const float *B, float *C, int m, int k, int n) {
  if (m <= 0 || k <= 0 || n <= 0)
    return;

  std::memset(C, 0, (std::size_t)m * n * sizeof(float));

  omp_set_dynamic(0);
  omp_set_num_threads(N_THREADS);

  const int NC_PAD = round_up8(n);

  float *Bp = alloc_aligned_floats((std::size_t)KC * NC_PAD);
  if (!Bp)
    throw std::runtime_error("Allocation failed");

  std::atomic<int> failed{0};

#pragma omp parallel shared(Bp, failed)
  {
    const int MC_PAD = round_up8(MC);
    float *Ap = alloc_aligned_floats((std::size_t)MC_PAD * KC);
    if (!Ap) {
      failed.store(1, std::memory_order_relaxed);
    }

    for (int jc = 0; jc < n; jc += NC) {
      const int nc = std::min(NC, n - jc);

      for (int kk = 0; kk < k; kk += KC) {
        const int kc = std::min(KC, k - kk);

#pragma omp single
        {
          if (!failed.load(std::memory_order_relaxed)) {
            pack_B_8(B + (std::size_t)kk * n + jc, Bp, n, kc, nc);
          }
        }

#pragma omp for schedule(static)
        for (int ic = 0; ic < m; ic += MC) {
          if (failed.load(std::memory_order_relaxed) || !Ap)
            continue;

          const int mc = std::min(MC, m - ic);
          pack_A_8(A + (std::size_t)ic * k + kk, Ap, k, mc, kc);

          for (int i = 0; i < mc; i += 8) {
            const float *Ablock = Ap + (std::size_t)i * kc;
            int mc_actual = std::min(8, mc - i);

            for (int j = 0; j < nc; j += 8) {
              const float *Bblock = Bp + (std::size_t)j * kc;
              int nc_actual = std::min(8, nc - j);

              kernel_8x8(kc, Ablock, Bblock,
                         C + (std::size_t)(ic + i) * n + jc + j, n, mc_actual,
                         nc_actual);
            }
          }
        }
      }
    }

    if (Ap)
      free(Ap);
  }

  free(Bp);
}

std::string CpuEtBackend::name() const { return "cpu_et"; }

Tensor CpuEtBackend::matmul(const Tensor &a, const Tensor &b) {
  size_t m = a.shape().dim(0);
  size_t k = a.shape().dim(1);
  size_t n = b.shape().dim(1);

  Tensor out(Shape({m, n}), DType::Float32);
  matmul_kp(a.data(), b.data(), out.data(), static_cast<int>(m), static_cast<int>(k), static_cast<int>(n));

  return out;
}

} // namespace gpt2lab
