cmake_minimum_required(VERSION 3.16)

project(gpt2lab VERSION 0.1 LANGUAGES CXX)

# =========================================================
# Global configuration
# =========================================================

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(GNUInstallDirs)
include(FetchContent)

# =========================================================
# CUDA
# =========================================================

option(GPT2LAB_ENABLE_CUDA "Enable CUDA" OFF)

if(GPT2LAB_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

# =========================================================
# GoogleTest
# =========================================================

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)

FetchContent_Declare(
    eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
)

FetchContent_MakeAvailable(googletest nlohmann_json eigen)

include(GoogleTest)

# =========================================================
# Core
# =========================================================

add_library(gpt2lab_core
    src/core/dtype.cpp
    src/core/shape.cpp
    src/core/tensor.cpp
    src/core/device.cpp
    src/core/error.cpp
)

target_include_directories(gpt2lab_core PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)


# =========================================================
# Ops
# =========================================================

add_library(gpt2lab_ops
    src/ops/ops.cpp
)

target_include_directories(gpt2lab_ops PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(gpt2lab_ops PUBLIC
    gpt2lab_core
)

# =========================================================
# Autograd
# =========================================================

add_library(gpt2lab_autograd
    src/autograd/tape.cpp
    src/autograd/graph.cpp
    src/autograd/backward.cpp
)

target_include_directories(gpt2lab_autograd PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(gpt2lab_autograd PUBLIC
    gpt2lab_core
    gpt2lab_ops
)

# =========================================================
# Backends
# =========================================================

add_library(gpt2lab_backend_cpu_naive
    src/backends/cpu_naive.cpp
)

add_library(gpt2lab_backend_cpu_et
    src/backends/cpu_et.cpp
)

find_package(OpenMP REQUIRED)
target_link_libraries(gpt2lab_backend_cpu_et PUBLIC OpenMP::OpenMP_CXX)
target_compile_options(gpt2lab_backend_cpu_et PRIVATE -mavx2 -mfma)

add_library(gpt2lab_backend_eigen
    src/backends/eigen_backend.cpp
)
target_link_libraries(gpt2lab_backend_eigen PUBLIC Eigen3::Eigen)

if(GPT2LAB_ENABLE_CUDA)
    add_library(gpt2lab_backend_cuda
        src/backends/cuda_backend.cu
    )

    add_library(gpt2lab_backend_cublas
        src/backends/cublas_backend.cu
    )
    target_link_libraries(gpt2lab_backend_cublas PUBLIC CUDA::cublas)

    add_library(gpt2lab_backend_cudnn
        src/backends/cudnn_backend.cu
    )
endif()

set(BACKENDS
    gpt2lab_backend_cpu_naive
    gpt2lab_backend_cpu_et
    gpt2lab_backend_eigen
)

if(GPT2LAB_ENABLE_CUDA)
    list(APPEND BACKENDS
        gpt2lab_backend_cuda
        gpt2lab_backend_cublas
        gpt2lab_backend_cudnn
    )
endif()

foreach(lib ${BACKENDS})
    target_include_directories(${lib} PUBLIC
        ${CMAKE_SOURCE_DIR}/include
    )

    target_link_libraries(${lib} PUBLIC
        gpt2lab_core
        gpt2lab_ops
    )
endforeach()

if(GPT2LAB_ENABLE_CUDA)
    target_link_libraries(gpt2lab_backend_cuda PUBLIC
        CUDA::cudart
    )

    target_link_libraries(gpt2lab_backend_cublas PUBLIC
        CUDA::cublas
        CUDA::cudart
    )

    # Note: CUDA::cudnn might require a custom FindcuDNN.cmake
    # target_link_libraries(gpt2lab_backend_cudnn PUBLIC
    #     CUDA::cudnn
    #     CUDA::cudart
    # )
endif()

# =========================================================
# Model
# =========================================================

add_library(gpt2lab_model
    src/model/gpt2.cpp
)

target_include_directories(gpt2lab_model PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(gpt2lab_model PUBLIC
    gpt2lab_core
    gpt2lab_ops
    gpt2lab_autograd
    gpt2lab_ops
    nlohmann_json::nlohmann_json
)

# Resolve circular dependency and backend linkage for ops
target_link_libraries(gpt2lab_ops PUBLIC
    gpt2lab_autograd
    gpt2lab_backend_cpu_naive
)

# =========================================================
# Runtime
# =========================================================

add_library(gpt2lab_runtime
    src/runtime/timer.cpp
    src/runtime/profiler.cpp
    src/runtime/logger.cpp
    src/runtime/allocator.cpp
    src/runtime/memory_stats.cpp
    src/runtime/reproducibility.cpp
)

target_include_directories(gpt2lab_runtime PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(gpt2lab_runtime PUBLIC
    gpt2lab_core
)

# =========================================================
# Benchmarking
# =========================================================

add_library(gpt2lab_bench
    src/bench/benchmark.cpp
    src/bench/metrics.cpp
    src/bench/report.cpp
)

target_include_directories(gpt2lab_bench PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(gpt2lab_bench PUBLIC
    gpt2lab_core
    gpt2lab_runtime
)

# =========================================================
# Applications
# =========================================================

add_executable(train_gpt2
    examples/train_gpt2.cpp
)

target_link_libraries(train_gpt2 PRIVATE
    gpt2lab_model
    gpt2lab_runtime
)

add_executable(infer_gpt2
    examples/infer_gpt2.cpp
)

target_link_libraries(infer_gpt2 PRIVATE
    gpt2lab_model
    gpt2lab_runtime
)

add_executable(benchmark_matmul
    benchmarks/benchmark_matmul.cpp
)

target_link_libraries(benchmark_matmul PRIVATE
    gpt2lab_bench
    gpt2lab_ops
    gpt2lab_runtime
)

add_executable(benchmark_attention
    benchmarks/benchmark_attention.cpp
)

target_link_libraries(benchmark_attention PRIVATE
    gpt2lab_bench
    gpt2lab_ops
    gpt2lab_runtime
)

add_executable(benchmark_layernorm
    benchmarks/benchmark_layernorm.cpp
)

target_link_libraries(benchmark_layernorm PRIVATE
    gpt2lab_bench
    gpt2lab_ops
    gpt2lab_runtime
)

add_executable(compare_backends
    benchmarks/compare_backends.cpp
)

target_link_libraries(compare_backends PRIVATE
    gpt2lab_backend_cpu_naive
    gpt2lab_backend_cpu_et
    gpt2lab_backend_eigen
    gpt2lab_runtime
)

if(GPT2LAB_ENABLE_CUDA)
    target_compile_definitions(compare_backends PRIVATE GPT2LAB_ENABLE_CUDA)
    target_link_libraries(compare_backends PRIVATE gpt2lab_backend_cublas)
endif()

add_executable(gradient_check
    tests/tools/gradient_check.cpp
)

target_link_libraries(gradient_check PRIVATE
    gpt2lab_autograd
    gpt2lab_model
    gpt2lab_runtime
)

add_executable(dump_tensor
    tests/tools/dump_tensor.cpp
)

target_link_libraries(dump_tensor PRIVATE
    gpt2lab_core
    gpt2lab_runtime
)

# =========================================================
# Tests
# =========================================================

enable_testing()

add_executable(test_runner
    tests/test_runner.cpp

    tests/unit/test_shape.cpp
    tests/unit/test_tensor.cpp
    tests/unit/test_matmul.cpp
    tests/unit/test_autograd.cpp

    tests/integration/test_gpt2_forward.cpp
)

target_include_directories(test_runner PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(test_runner PRIVATE
    gpt2lab_core
    gpt2lab_ops
    gpt2lab_autograd
    gpt2lab_model
    gpt2lab_runtime
    gpt2lab_backend_cpu_naive

    GTest::gtest
    GTest::gtest_main
)

if(GPT2LAB_ENABLE_CUDA)
    target_sources(test_runner PRIVATE tests/unit/test_cuda.cpp)
    target_link_libraries(test_runner PRIVATE gpt2lab_backend_cuda)
endif()

gtest_discover_tests(test_runner)