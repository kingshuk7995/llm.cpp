#include <gpt2lab/backends/cpu_naive.hpp>
#include <gpt2lab/backends/cpu_et.hpp>
#include <gpt2lab/backends/eigen_backend.hpp>
#ifdef GPT2LAB_ENABLE_CUDA
#include <gpt2lab/backends/cublas_backend.hpp>
#endif

#include <gpt2lab/core/tensor.hpp>
#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <string>
#include <iomanip>

using namespace gpt2lab;

struct BenchShape {
  int M, K, N;
  std::string name;
};

double run_benchmark(Backend* backend, int M, int K, int N, int iterations = 10) {
  Tensor a({(size_t)M, (size_t)K});
  Tensor b({(size_t)K, (size_t)N});
  a.init_normal(0.0f, 1.0f);
  b.init_normal(0.0f, 1.0f);

  // Warmup
  backend->matmul(a, b);

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; ++i) {
    backend->matmul(a, b);
  }
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> diff = end - start;
  return diff.count() / iterations;
}

int main() {
  std::vector<BenchShape> shapes = {
    {1024, 768, 768, "QKV_Proj_Small"},
    {1024, 768, 3072, "MLP_Expand_Small"},
    {1024, 3072, 768, "MLP_Contract_Small"},
    {1024, 1600, 1600, "QKV_Proj_Large"},
    {1024, 1600, 6400, "MLP_Expand_Large"}
  };

  CpuNaiveBackend cpu_naive;
  CpuEtBackend cpu_et;
  EigenBackend eigen_backend;
#ifdef GPT2LAB_ENABLE_CUDA
  CublasBackend cublas_backend;
#endif

  struct BenchRun {
    std::string name;
    Backend* backend;
  };

  std::vector<BenchRun> backends = {
    {"CpuNaive", &cpu_naive},
    {"CpuET_OpenMP", &cpu_et},
    {"Eigen", &eigen_backend},
#ifdef GPT2LAB_ENABLE_CUDA
    {"cuBLAS", &cublas_backend}
#endif
  };

  std::ofstream out_file("benchmarks/backend_benchmark.csv");
  out_file << "Backend,ShapeName,M,K,N,Latency(s),TFLOPs\n";
  
  std::cout << "Starting Backend Benchmarks...\n";

  for (const auto& run : backends) {
    std::cout << "\nRunning " << run.name << "...\n";
    for (const auto& shape : shapes) {
      double latency = run_benchmark(run.backend, shape.M, shape.K, shape.N);
      
      double flops = 2.0 * shape.M * shape.K * shape.N;
      double tflops = (flops / latency) / 1e12;

      out_file << run.name << "," << shape.name << "," 
               << shape.M << "," << shape.K << "," << shape.N << ","
               << latency << "," << tflops << "\n";
               
      std::cout << "  " << std::setw(20) << shape.name << " | "
                << std::setw(10) << std::fixed << std::setprecision(4) << latency * 1000.0 << " ms | "
                << std::setw(10) << std::fixed << std::setprecision(4) << tflops << " TFLOP/s\n";
    }
  }

  std::cout << "\nResults saved to benchmarks/backend_benchmark.csv\n";
  return 0;
}
