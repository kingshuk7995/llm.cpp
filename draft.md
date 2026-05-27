# GPT2Lab: Project Draft

A systems-focused GPT-2 experimentation framework for comparing:

* Handwritten CPU matmul kernels
* CPU expression-template kernels
* Eigen as a CPU baseline
* Handwritten CUDA kernels
* NVIDIA CUDA libraries such as cuBLAS and cuDNN
* Manual backward passes vs autodiff experimentation

The goal is not just to train or run GPT-2, but to measure implementation choices across correctness, speed, memory behavior, and runtime overhead.

---

## 1. Project Goals

### Primary goals

1. Implement a minimal GPT-2 style model from scratch.
2. Support multiple compute backends under a common API.
3. Compare forward-pass and backward-pass performance across backends.
4. Compare hand-written differentiation with autodiff libraries.
5. Build a benchmarking harness for reproducible experiments.
6. Keep the design extensible enough for future fusion, quantization, and kernel specialization experiments.

### Research questions

* How much does expression-template fusion reduce temporary allocations on CPU?
* How close can a hand-written CPU kernel get to Eigen for small/medium matrix sizes?
* How much overhead does a backend abstraction layer introduce?
* How do custom CUDA kernels compare against cuBLAS for GPT-2-shaped matmuls?
* Is autodiff practical for this workload, or does manual backward win clearly?
* How much do layout choices, batching strategy, and precision affect throughput?

---

## 2. High-Level Architecture

The project should be split into layers:

1. **Core tensor layer**: shapes, dtypes, storage, views, indexing, memory management.
2. **Operation layer**: matmul, add, gelu, softmax, layernorm, dropout, attention.
3. **Autograd layer**: graph, tape, backward nodes, gradients.
4. **Backend layer**: CPU naive, CPU ET, Eigen, CUDA custom, cuBLAS/cuDNN.
5. **Model layer**: GPT-2 blocks, embeddings, transformer blocks, LM head.
6. **Runtime layer**: timing, profiling, logging, allocator tracking, reproducibility.
7. **Bench layer**: microbenchmarks and end-to-end benchmarks.

Keep model code independent from backend implementations. The model should call abstract tensor ops, not specific kernel files directly.

---

## 3. Proposed Repository Layout

```text
GPT2Lab/
├── CMakeLists.txt
├── CMakePresets.json
├── README.md
├── LICENSE
├── .gitignore
├── .clang-format
├── .clang-tidy
├── docs/
│   ├── architecture.md
│   ├── benchmarks.md
│   ├── backend-comparison.md
│   ├── autodiff-notes.md
│   └── design-decisions.md
├── cmake/
│   ├── CompilerWarnings.cmake
│   ├── Sanitizers.cmake
│   ├── CUDASettings.cmake
│   ├── Dependencies.cmake
│   └── ProjectOptions.cmake
├── external/
│   ├── eigen/                  # optional submodule or package manager dependency
│   ├── autodiff/               # optional submodule or package manager dependency
│   └── ...
├── include/
│   └── gpt2lab/
│       ├── core/
│       │   ├── dtype.hpp
│       │   ├── shape.hpp
│       │   ├── tensor.hpp
│       │   ├── tensor_view.hpp
│       │   ├── storage.hpp
│       │   ├── layout.hpp
│       │   ├── device.hpp
│       │   ├── status.hpp
│       │   └── error.hpp
│       ├── ops/
│       │   ├── matmul.hpp
│       │   ├── add.hpp
│       │   ├── mul.hpp
│       │   ├── gelu.hpp
│       │   ├── softmax.hpp
│       │   ├── layernorm.hpp
│       │   ├── dropout.hpp
│       │   ├── reshape.hpp
│       │   ├── transpose.hpp
│       │   └── attention.hpp
│       ├── autograd/
│       │   ├── grad_fn.hpp
│       │   ├── tape.hpp
│       │   ├── graph.hpp
│       │   ├── backward.hpp
│       │   └── autodiff_adapter.hpp
│       ├── backends/
│       │   ├── backend.hpp
│       │   ├── backend_registry.hpp
│       │   ├── cpu_naive.hpp
│       │   ├── cpu_et.hpp
│       │   ├── eigen_backend.hpp
│       │   ├── cuda_backend.hpp
│       │   ├── cublas_backend.hpp
│       │   ├── cudnn_backend.hpp
│       │   └── backend_selector.hpp
│       ├── model/
│       │   ├── gpt2_config.hpp
│       │   ├── gpt2.hpp
│       │   ├── transformer_block.hpp
│       │   ├── mha.hpp
│       │   ├── mlp.hpp
│       │   ├── embeddings.hpp
│       │   ├── lm_head.hpp
│       │   └── init.hpp
│       ├── runtime/
│       │   ├── timer.hpp
│       │   ├── profiler.hpp
│       │   ├── logger.hpp
│       │   ├── allocator.hpp
│       │   ├── memory_stats.hpp
│       │   └── reproducibility.hpp
│       └── bench/
│           ├── benchmark.hpp
│           ├── metrics.hpp
│           ├── report.hpp
│           └── cases.hpp
├── src/
│   ├── core/
│   │   ├── dtype.cpp
│   │   ├── shape.cpp
│   │   ├── tensor.cpp
│   │   ├── storage.cpp
│   │   ├── device.cpp
│   │   └── error.cpp
│   ├── ops/
│   │   ├── matmul.cpp
│   │   ├── add.cpp
│   │   ├── gelu.cpp
│   │   ├── softmax.cpp
│   │   ├── layernorm.cpp
│   │   ├── dropout.cpp
│   │   ├── transpose.cpp
│   │   └── attention.cpp
│   ├── autograd/
│   │   ├── tape.cpp
│   │   ├── graph.cpp
│   │   ├── backward.cpp
│   │   └── autodiff_adapter.cpp
│   ├── backends/
│   │   ├── backend_registry.cpp
│   │   ├── cpu_naive.cpp
│   │   ├── cpu_et.cpp
│   │   ├── eigen_backend.cpp
│   │   ├── cuda_backend.cpp
│   │   ├── cublas_backend.cpp
│   │   ├── cudnn_backend.cpp
│   │   └── backend_selector.cpp
│   ├── model/
│   │   ├── gpt2_config.cpp
│   │   ├── gpt2.cpp
│   │   ├── transformer_block.cpp
│   │   ├── mha.cpp
│   │   ├── mlp.cpp
│   │   ├── embeddings.cpp
│   │   ├── lm_head.cpp
│   │   └── init.cpp
│   ├── runtime/
│   │   ├── timer.cpp
│   │   ├── profiler.cpp
│   │   ├── logger.cpp
│   │   ├── allocator.cpp
│   │   ├── memory_stats.cpp
│   │   └── reproducibility.cpp
│   └── bench/
│       ├── benchmark.cpp
│       ├── metrics.cpp
│       └── report.cpp
├── cpu/
│   ├── include/
│   │   └── gpt2lab/
│   │       └── cpu/
│   │           ├── scalar_kernels.hpp
│   │           ├── simd_kernels.hpp
│   │           ├── blocking.hpp
│   │           ├── et_ops.hpp
│   │           ├── et_expr.hpp
│   │           └── et_eval.hpp
│   └── src/
│       ├── scalar_kernels.cpp
│       ├── simd_kernels.cpp
│       ├── blocking.cpp
│       ├── et_ops.cpp
│       └── et_eval.cpp
├── cu/
│   ├── include/
│   │   └── gpt2lab/
│   │       └── cu/
│   │           ├── kernels.cuh
│   │           ├── matmul.cuh
│   │           ├── attention.cuh
│   │           ├── layernorm.cuh
│   │           ├── softmax.cuh
│   │           ├── memory.cuh
│   │           └── device_utils.cuh
│   └── src/
│       ├── matmul.cu
│       ├── attention.cu
│       ├── layernorm.cu
│       ├── softmax.cu
│       ├── memory.cu
│       ├── device_utils.cu
│       └── fused_kernels.cu
├── third_party_wrappers/
│   ├── eigen/
│   │   ├── eigen_backend.hpp
│   │   └── eigen_backend.cpp
│   ├── autodiff/
│   │   ├── autodiff_backend.hpp
│   │   └── autodiff_backend.cpp
│   ├── cublas/
│   │   ├── cublas_wrappers.hpp
│   │   └── cublas_wrappers.cpp
│   └── cudnn/
│       ├── cudnn_wrappers.hpp
│       └── cudnn_wrappers.cpp
├── apps/
│   ├── train_gpt2.cpp
│   ├── infer_gpt2.cpp
│   ├── benchmark_matmul.cpp
│   ├── benchmark_attention.cpp
│   ├── benchmark_layernorm.cpp
│   ├── compare_backends.cpp
│   ├── gradient_check.cpp
│   └── dump_tensor.cpp
├── tests/
│   ├── unit/
│   │   ├── test_shape.cpp
│   │   ├── test_tensor.cpp
│   │   ├── test_matmul.cpp
│   │   ├── test_attention.cpp
│   │   ├── test_layernorm.cpp
│   │   ├── test_softmax.cpp
│   │   ├── test_autograd.cpp
│   │   └── test_backend_registry.cpp
│   ├── integration/
│   │   ├── test_gpt2_forward.cpp
│   │   ├── test_gpt2_backward.cpp
│   │   └── test_cross_backend_consistency.cpp
│   └── golden/
│       ├── golden_forward.json
│       └── golden_gradients.json
├── benchmarks/
│   ├── micro/
│   │   ├── matmul_bench.cpp
│   │   ├── attention_bench.cpp
│   │   ├── layernorm_bench.cpp
│   │   ├── softmax_bench.cpp
│   │   └── alloc_bench.cpp
│   ├── model/
│   │   ├── gpt2_forward_bench.cpp
│   │   ├── gpt2_backward_bench.cpp
│   │   ├── training_step_bench.cpp
│   │   └── inference_latency_bench.cpp
│   └── reports/
│       ├── benchmark_schema.md
│       ├── benchmark_results_template.csv
│       └── benchmark_notes.md
├── scripts/
│   ├── run_benchmarks.sh
│   ├── run_tests.sh
│   ├── format.sh
│   ├── tidy.sh
│   ├── collect_metrics.py
│   └── plot_results.py
├── data/
│   ├── tiny_shakespeare/
│   ├── tokenizer/
│   └── samples/
└── build/
```

---

## 4. Target Breakdown

### Core targets

* `gpt2lab_core`

  * tensor, shape, dtype, storage, device, layout
* `gpt2lab_ops`

  * math and neural-network primitives
* `gpt2lab_autograd`

  * backward graph and gradient logic
* `gpt2lab_backend_cpu_naive`

  * simple CPU implementation
* `gpt2lab_backend_cpu_et`

  * expression-template CPU implementation
* `gpt2lab_backend_eigen`

  * Eigen wrapper backend
* `gpt2lab_backend_cuda`

  * custom CUDA kernels
* `gpt2lab_backend_cublas`

  * cuBLAS-backed implementation
* `gpt2lab_backend_cudnn`

  * cuDNN-backed implementation where applicable
* `gpt2lab_model`

  * GPT-2 model blocks
* `gpt2lab_runtime`

  * timers, profiler, alloc tracking, logging
* `gpt2lab_bench`

  * benchmark helpers and report generation

### Executables

* `train_gpt2`
* `infer_gpt2`
* `benchmark_matmul`
* `benchmark_attention`
* `benchmark_layernorm`
* `compare_backends`
* `gradient_check`
* `dump_tensor`
* `test_*` binaries or a unified test runner

---

## 5. Backend Strategy

Use a backend interface with a common contract.

### Backend responsibilities

Each backend should be able to:

* allocate and free tensors
* move memory between host and device
* execute elementwise ops
* execute matmul
* execute attention-related primitives
* support reductions
* report capabilities such as precision support and preferred layout
* expose timing/profiling hooks

### Backend comparison matrix

#### 5.1 CPU Naive

Purpose:

* simplest correct reference implementation
* baseline for correctness and algorithmic clarity

TODOs:

* [ ] row-major tensor storage
* [ ] scalar matmul
* [ ] scalar add/mul/relu/gelu
* [ ] simple softmax and layernorm
* [ ] unit tests against golden outputs

#### 5.2 CPU Expression Templates
# GPT2Lab: Project Draft

A systems-focused GPT-2 experimentation framework for comparing:

* Handwritten CPU matmul kernels
* CPU expression-template kernels
* Eigen as a CPU baseline
* Handwritten CUDA kernels
* NVIDIA CUDA libraries such as cuBLAS and cuDNN
* Manual backward passes vs autodiff experimentation

The goal is not just to train or run GPT-2, but to measure implementation choices across correctness, speed, memory behavior, and runtime overhead.

---

## 1. Project Goals

### Primary goals

1. Implement a minimal GPT-2 style model from scratch.
2. Support multiple compute backends under a common API.
3. Compare forward-pass and backward-pass performance across backends.
4. Compare hand-written differentiation with autodiff libraries.
5. Build a benchmarking harness for reproducible experiments.
6. Keep the design extensible enough for future fusion, quantization, and kernel specialization experiments.

### Research questions

* How much does expression-template fusion reduce temporary allocations on CPU?
* How close can a hand-written CPU kernel get to Eigen for small/medium matrix sizes?
* How much overhead does a backend abstraction layer introduce?
* How do custom CUDA kernels compare against cuBLAS for GPT-2-shaped matmuls?
* Is autodiff practical for this workload, or does manual backward win clearly?
* How much do layout choices, batching strategy, and precision affect throughput?

---

## 2. High-Level Architecture

The project should be split into layers:

1. **Core tensor layer**: shapes, dtypes, storage, views, indexing, memory management.
2. **Operation layer**: matmul, add, gelu, softmax, layernorm, dropout, attention.
3. **Autograd layer**: graph, tape, backward nodes, gradients.
4. **Backend layer**: CPU naive, CPU ET, Eigen, CUDA custom, cuBLAS/cuDNN.
5. **Model layer**: GPT-2 blocks, embeddings, transformer blocks, LM head.
6. **Runtime layer**: timing, profiling, logging, allocator tracking, reproducibility.
7. **Bench layer**: microbenchmarks and end-to-end benchmarks.

Keep model code independent from backend implementations. The model should call abstract tensor ops, not specific kernel files directly.

---

## 3. Proposed Repository Layout

```text
GPT2Lab/
├── CMakeLists.txt
├── CMakePresets.json
├── README.md
├── LICENSE
├── .gitignore
├── .clang-format
├── .clang-tidy
├── docs/
│   ├── architecture.md
│   ├── benchmarks.md
│   ├── backend-comparison.md
│   ├── autodiff-notes.md
│   └── design-decisions.md
├── cmake/
│   ├── CompilerWarnings.cmake
│   ├── Sanitizers.cmake
│   ├── CUDASettings.cmake
│   ├── Dependencies.cmake
│   └── ProjectOptions.cmake
├── external/
│   ├── eigen/                  # optional submodule or package manager dependency
│   ├── autodiff/               # optional submodule or package manager dependency
│   └── ...
├── include/
│   └── gpt2lab/
│       ├── core/
│       │   ├── dtype.hpp
│       │   ├── shape.hpp
│       │   ├── tensor.hpp
│       │   ├── tensor_view.hpp
│       │   ├── storage.hpp
│       │   ├── layout.hpp
│       │   ├── device.hpp
│       │   ├── status.hpp
│       │   └── error.hpp
│       ├── ops/
│       │   ├── matmul.hpp
│       │   ├── add.hpp
│       │   ├── mul.hpp
│       │   ├── gelu.hpp
│       │   ├── softmax.hpp
│       │   ├── layernorm.hpp
│       │   ├── dropout.hpp
│       │   ├── reshape.hpp
│       │   ├── transpose.hpp
│       │   └── attention.hpp
│       ├── autograd/
│       │   ├── grad_fn.hpp
│       │   ├── tape.hpp
│       │   ├── graph.hpp
│       │   ├── backward.hpp
│       │   └── autodiff_adapter.hpp
│       ├── backends/
│       │   ├── backend.hpp
│       │   ├── backend_registry.hpp
│       │   ├── cpu_naive.hpp
│       │   ├── cpu_et.hpp
│       │   ├── eigen_backend.hpp
│       │   ├── cuda_backend.hpp
│       │   ├── cublas_backend.hpp
│       │   ├── cudnn_backend.hpp
│       │   └── backend_selector.hpp
│       ├── model/
│       │   ├── gpt2_config.hpp
│       │   ├── gpt2.hpp
│       │   ├── transformer_block.hpp
│       │   ├── mha.hpp
│       │   ├── mlp.hpp
│       │   ├── embeddings.hpp
│       │   ├── lm_head.hpp
│       │   └── init.hpp
│       ├── runtime/
│       │   ├── timer.hpp
│       │   ├── profiler.hpp
│       │   ├── logger.hpp
│       │   ├── allocator.hpp
│       │   ├── memory_stats.hpp
│       │   └── reproducibility.hpp
│       └── bench/
│           ├── benchmark.hpp
│           ├── metrics.hpp
│           ├── report.hpp
│           └── cases.hpp
├── src/
│   ├── core/
│   │   ├── dtype.cpp
│   │   ├── shape.cpp
│   │   ├── tensor.cpp
│   │   ├── storage.cpp
│   │   ├── device.cpp
│   │   └── error.cpp
│   ├── ops/
│   │   ├── matmul.cpp
│   │   ├── add.cpp
│   │   ├── gelu.cpp
│   │   ├── softmax.cpp
│   │   ├── layernorm.cpp
│   │   ├── dropout.cpp
│   │   ├── transpose.cpp
│   │   └── attention.cpp
│   ├── autograd/
│   │   ├── tape.cpp
│   │   ├── graph.cpp
│   │   ├── backward.cpp
│   │   └── autodiff_adapter.cpp
│   ├── backends/
│   │   ├── backend_registry.cpp
│   │   ├── cpu_naive.cpp
│   │   ├── cpu_et.cpp
│   │   ├── eigen_backend.cpp
│   │   ├── cuda_backend.cpp
│   │   ├── cublas_backend.cpp
│   │   ├── cudnn_backend.cpp
│   │   └── backend_selector.cpp
│   ├── model/
│   │   ├── gpt2_config.cpp
│   │   ├── gpt2.cpp
│   │   ├── transformer_block.cpp
│   │   ├── mha.cpp
│   │   ├── mlp.cpp
│   │   ├── embeddings.cpp
│   │   ├── lm_head.cpp
│   │   └── init.cpp
│   ├── runtime/
│   │   ├── timer.cpp
│   │   ├── profiler.cpp
│   │   ├── logger.cpp
│   │   ├── allocator.cpp
│   │   ├── memory_stats.cpp
│   │   └── reproducibility.cpp
│   └── bench/
│       ├── benchmark.cpp
│       ├── metrics.cpp
│       └── report.cpp
├── cpu/
│   ├── include/
│   │   └── gpt2lab/
│   │       └── cpu/
│   │           ├── scalar_kernels.hpp
│   │           ├── simd_kernels.hpp
│   │           ├── blocking.hpp
│   │           ├── et_ops.hpp
│   │           ├── et_expr.hpp
│   │           └── et_eval.hpp
│   └── src/
│       ├── scalar_kernels.cpp
│       ├── simd_kernels.cpp
│       ├── blocking.cpp
│       ├── et_ops.cpp
│       └── et_eval.cpp
├── cu/
│   ├── include/
│   │   └── gpt2lab/
│   │       └── cu/
│   │           ├── kernels.cuh
│   │           ├── matmul.cuh
│   │           ├── attention.cuh
│   │           ├── layernorm.cuh
│   │           ├── softmax.cuh
│   │           ├── memory.cuh
│   │           └── device_utils.cuh
│   └── src/
│       ├── matmul.cu
│       ├── attention.cu
│       ├── layernorm.cu
│       ├── softmax.cu
│       ├── memory.cu
│       ├── device_utils.cu
│       └── fused_kernels.cu
├── third_party_wrappers/
│   ├── eigen/
│   │   ├── eigen_backend.hpp
│   │   └── eigen_backend.cpp
│   ├── autodiff/
│   │   ├── autodiff_backend.hpp
│   │   └── autodiff_backend.cpp
│   ├── cublas/
│   │   ├── cublas_wrappers.hpp
│   │   └── cublas_wrappers.cpp
│   └── cudnn/
│       ├── cudnn_wrappers.hpp
│       └── cudnn_wrappers.cpp
├── apps/
│   ├── train_gpt2.cpp
│   ├── infer_gpt2.cpp
│   ├── benchmark_matmul.cpp
│   ├── benchmark_attention.cpp
│   ├── benchmark_layernorm.cpp
│   ├── compare_backends.cpp
│   ├── gradient_check.cpp
│   └── dump_tensor.cpp
├── tests/
│   ├── unit/
│   │   ├── test_shape.cpp
│   │   ├── test_tensor.cpp
│   │   ├── test_matmul.cpp
│   │   ├── test_attention.cpp
│   │   ├── test_layernorm.cpp
│   │   ├── test_softmax.cpp
│   │   ├── test_autograd.cpp
│   │   └── test_backend_registry.cpp
│   ├── integration/
│   │   ├── test_gpt2_forward.cpp
│   │   ├── test_gpt2_backward.cpp
│   │   └── test_cross_backend_consistency.cpp
│   └── golden/
│       ├── golden_forward.json
│       └── golden_gradients.json
├── benchmarks/
│   ├── micro/
│   │   ├── matmul_bench.cpp
│   │   ├── attention_bench.cpp
│   │   ├── layernorm_bench.cpp
│   │   ├── softmax_bench.cpp
│   │   └── alloc_bench.cpp
│   ├── model/
│   │   ├── gpt2_forward_bench.cpp
│   │   ├── gpt2_backward_bench.cpp
│   │   ├── training_step_bench.cpp
│   │   └── inference_latency_bench.cpp
│   └── reports/
│       ├── benchmark_schema.md
│       ├── benchmark_results_template.csv
│       └── benchmark_notes.md
├── scripts/
│   ├── run_benchmarks.sh
│   ├── run_tests.sh
│   ├── format.sh
│   ├── tidy.sh
│   ├── collect_metrics.py
│   └── plot_results.py
├── data/
│   ├── tiny_shakespeare/
│   ├── tokenizer/
│   └── samples/
└── build/
```

---

## 4. Target Breakdown

### Core targets

* `gpt2lab_core`

  * tensor, shape, dtype, storage, device, layout
* `gpt2lab_ops`

  * math and neural-network primitives
* `gpt2lab_autograd`

  * backward graph and gradient logic
* `gpt2lab_backend_cpu_naive`

  * simple CPU implementation
* `gpt2lab_backend_cpu_et`

  * expression-template CPU implementation
* `gpt2lab_backend_eigen`

  * Eigen wrapper backend
* `gpt2lab_backend_cuda`

  * custom CUDA kernels
* `gpt2lab_backend_cublas`

  * cuBLAS-backed implementation
* `gpt2lab_backend_cudnn`

  * cuDNN-backed implementation where applicable
* `gpt2lab_model`

  * GPT-2 model blocks
* `gpt2lab_runtime`

  * timers, profiler, alloc tracking, logging
* `gpt2lab_bench`

  * benchmark helpers and report generation

### Executables

* `train_gpt2`
* `infer_gpt2`
* `benchmark_matmul`
* `benchmark_attention`
* `benchmark_layernorm`
* `compare_backends`
* `gradient_check`
* `dump_tensor`
* `test_*` binaries or a unified test runner

---

## 5. Backend Strategy

Use a backend interface with a common contract.

### Backend responsibilities

Each backend should be able to:

* allocate and free tensors
* move memory between host and device
* execute elementwise ops
* execute matmul
* execute attention-related primitives
* support reductions
* report capabilities such as precision support and preferred layout
* expose timing/profiling hooks

### Backend comparison matrix

#### 5.1 CPU Naive

Purpose:

* simplest correct reference implementation
* baseline for correctness and algorithmic clarity

TODOs:

* [ ] row-major tensor storage
* [ ] scalar matmul
* [ ] scalar add/mul/relu/gelu
* [ ] simple softmax and layernorm
* [ ] unit tests against golden outputs

#### 5.2 CPU Expression Templates

Purpose:

* evaluate fusion and compile-time composition
* remove temporary tensors where possible

TODOs:

* [ ] ET expression nodes
* [ ] lazy evaluation model
* [ ] fused elementwise chains
* [ ] compare temp count and runtime against naive CPU
* [ ] verify compile-time complexity remains manageable

#### 5.3 Eigen Backend

Purpose:

* trusted CPU baseline
* performance and correctness reference

TODOs:

* [ ] backend wrapper around Eigen matrices/tensors
* [ ] map internal layouts to Eigen layouts
* [ ] validate numerics against reference
* [ ] benchmark against hand-written CPU kernels

#### 5.4 CUDA Custom Backend

Purpose:

* experiment with hand-written GPU kernels
* study launch overhead, shared memory, tiling, and memory coalescing

TODOs:

* [ ] device tensor storage
* [ ] cudaMemcpy / async transfers
* [ ] custom matmul kernel
* [ ] custom layernorm kernel
* [ ] custom softmax kernel
* [ ] fused kernels where useful
* [ ] stream and event timing
* [ ] occupancy and memory bandwidth experiments

#### 5.5 cuBLAS / cuDNN Backend

Purpose:

* vendor baseline and optimized comparison point

TODOs:

* [ ] cuBLAS matmul wrapper
* [ ] cuDNN primitives where they actually fit the workload
* [ ] benchmark against custom CUDA kernels
* [ ] document when library overhead is worthwhile
* [ ] compare FP32 / FP16 / BF16 paths if available

#### 5.6 Autodiff Backend / Adapter

Purpose:

* compare manual backward implementation against automated differentiation

TODOs:

* [ ] evaluate `autodiff` library integration
* [ ] define adapter layer for scalar or small tensor components
* [ ] benchmark graph construction cost
* [ ] benchmark backward-pass cost
* [ ] compare gradient correctness against manual backward

---

## 6. Tensor and Storage Design

### Tensor core TODOs

* [ ] shape representation
* [ ] dtype enumeration
* [ ] storage with ownership semantics
* [ ] tensor views and slicing
* [ ] contiguous and strided layout support
* [ ] host/device placement
* [ ] copy and move semantics
* [ ] zero-initialization and uninitialized allocation modes
* [ ] debug bounds checking mode
* [ ] pretty-print and tensor dumping utilities

### Recommended tensor metadata

* shape
* stride
* dtype
* device
* layout
* requires_grad
* grad reference
* storage pointer / handle

### Key design choice

Start with one main contiguous layout. Add strided views later. Do not begin with a generalized tensor algebra monster.

---

## 7. Autograd and Differentiation Plan

### Manual backward path

This should be the primary path.

TODOs:

* [ ] define `Node` / `GradFn` abstraction
* [ ] record operation graph during forward pass
* [ ] implement backward for each op
* [ ] handle broadcasting reductions properly
* [ ] handle matmul gradient formulas
* [ ] handle layernorm backward
* [ ] handle softmax backward
* [ ] handle attention backward if implemented
* [ ] add gradient checking tests

### Autodiff experiment path

Treat this as an evaluation branch, not the core engine.

TODOs:

* [ ] integrate `autodiff` as an optional dependency
* [ ] create adapter benchmarks
* [ ] compare API friction
* [ ] compare runtime and compile-time overhead
* [ ] document whether it is practical for this workload

### Gradient checking

TODOs:

* [ ] finite-difference checker
* [ ] relative and absolute tolerance policy
* [ ] small-shape test suite
* [ ] random-seed controlled reproducibility

---

## 8. GPT-2 Model Scope

Do not start with full-scale GPT-2. Start with a reduced GPT-2 variant.

### Model components

* token embedding
* positional embedding
* layer norm
* causal self-attention
* MLP block
* residual connections
* LM head
* loss function

### Model TODOs

* [ ] define configuration struct
* [ ] build token and position embeddings
* [ ] implement transformer block
* [ ] implement causal mask logic
* [ ] implement inference path
* [ ] implement training path
* [ ] implement loss computation
* [ ] add weight initialization
* [ ] add weight serialization/deserialization
* [ ] add checkpoint save/load

### GPT-2 configuration knobs

* number of layers
* hidden size
* number of attention heads
* head dimension
* sequence length
* vocabulary size
* dropout probability
* precision mode
* backend selection

---

## 9. Benchmark Plan

The benchmark framework is one of the main deliverables.

### Microbenchmarks

* matmul only
* elementwise chains
* layernorm only
* softmax only
* attention only
* allocation/free overhead
* host-device transfer overhead
* fused vs unfused execution

### End-to-end benchmarks

* forward pass latency
* backward pass latency
* training step latency
* inference token latency
* tokens/sec
* memory usage peak
* numerical error compared to baseline

### Benchmark metadata

Every benchmark run should record:

* backend name
* device
* precision
* batch size
* sequence length
* hidden dimension
* number of layers
* compiler flags
* build type
* timestamp
* git commit hash
* average latency
* median latency
* p95/p99 latency where relevant
* throughput
* peak memory usage

### Benchmark TODOs

* [ ] define benchmark config format
* [ ] create benchmark runner
* [ ] capture timing statistics
* [ ] export CSV/JSON reports
* [ ] plot comparisons
* [ ] add warmup and steady-state phases
* [ ] add deterministic seed control

---

## 10. Testing Plan

### Test categories

#### Unit tests

* shape utilities
* tensor creation and ownership
* broadcasting rules
* matmul correctness
* softmax correctness
* layernorm correctness
* activation correctness
* backend registry

#### Integration tests

* forward pass consistency across backends
* backward pass consistency across backends
* checkpoint save/load round trip
* deterministic inference output

#### Numeric checks

* absolute and relative tolerance comparisons
* gradient checks with finite differences
* cross-backend output parity

### Testing TODOs

* [ ] GoogleTest or Catch2 setup
* [ ] fixture for small deterministic tensors
* [ ] golden output generator
* [ ] CI test matrix for CPU builds
* [ ] optional CUDA test jobs

---

## 11. CMake Target Sketch

Suggested targets:

```text
core
ops
autograd
backend_cpu_naive
backend_cpu_et
backend_eigen
backend_cuda
backend_cublas
backend_cudnn
model
runtime
bench
train_gpt2
infer_gpt2
benchmark_matmul
benchmark_attention
compare_backends
gradient_check
```

### CMake TODOs

* [ ] create interface target for common compiler settings
* [ ] split CPU and CUDA compilation cleanly
* [ ] keep Eigen and autodiff optional
* [ ] add build options for each backend
* [ ] add `BUILD_TESTING`
* [ ] add sanitizers for CPU debug builds
* [ ] add CUDA architecture selection
* [ ] add warnings-as-errors option
* [ ] add profiling-friendly release build flags

---

## 12. Build Options to Support

Recommended CMake options:

* `GPT2LAB_BUILD_TESTS`
* `GPT2LAB_BUILD_BENCHMARKS`
* `GPT2LAB_BUILD_APPS`
* `GPT2LAB_ENABLE_CUDA`
* `GPT2LAB_ENABLE_CUBLAS`
* `GPT2LAB_ENABLE_CUDNN`
* `GPT2LAB_ENABLE_EIGEN`
* `GPT2LAB_ENABLE_AUTODIFF`
* `GPT2LAB_ENABLE_SANITIZERS`
* `GPT2LAB_ENABLE_LTO`
* `GPT2LAB_ENABLE_WARNINGS_AS_ERRORS`
* `GPT2LAB_ENABLE_EXPRESSION_TEMPLATES`

---

## 13. TODO Roadmap by Phase

### Phase 1: Core infrastructure

* [ ] create repository skeleton
* [ ] define tensor, dtype, shape, storage
* [ ] implement CPU scalar reference ops
* [ ] implement basic tests
* [ ] build CMake structure

### Phase 2: CPU experimentation

* [ ] add blocked CPU matmul
* [ ] add SIMD CPU kernels
* [ ] add expression-template CPU backend
* [ ] benchmark against Eigen
* [ ] validate all outputs against reference

### Phase 3: Autograd

* [ ] implement manual backward graph
* [ ] add gradient checking
* [ ] optionally wire `autodiff`
* [ ] compare manual vs autodiff tradeoffs

### Phase 4: CUDA

* [ ] implement device storage
* [ ] implement CUDA matmul
* [ ] implement CUDA elementwise kernels
* [ ] implement timing with CUDA events
* [ ] benchmark against CPU backends

### Phase 5: Vendor library comparison

* [ ] add cuBLAS matmul backend
* [ ] add cuDNN where applicable
* [ ] compare against custom CUDA kernels
* [ ] document observations

### Phase 6: GPT-2 model

* [ ] implement transformer block
* [ ] implement inference
* [ ] implement training step
* [ ] add checkpointing
* [ ] run end-to-end backend comparison

### Phase 7: Research-grade benchmarking

* [ ] create repeatable benchmark suite
* [ ] automate result export
* [ ] generate plots and summaries
* [ ] compare precision modes
* [ ] compare batch and sequence scaling

---

## 14. Coding Conventions

### General rules

* Prefer explicit ownership semantics.
* Keep backend-specific code isolated.
* Keep tensor API stable.
* Measure before optimizing.
* Add tests before adding more abstraction.
* Avoid over-abstracting the first version.

### Style rules

* Use `snake_case` for functions and variables.
* Use `PascalCase` for types and backends.
* Avoid macros except for very narrow build/platform cases.
* Prefer `constexpr` only when it improves clarity or performance.
* Treat every benchmark as something that may lie unless carefully controlled.

---

## 15. Important Design Warnings

* Do not let expression templates spread into every layer.
* Do not make the backend interface too wide.
* Do not begin with attention fusion or flash attention.
* Do not mix benchmarking code into core tensor logic.
* Do not rely on autodiff as the main system.
* Do not assume cuDNN will cover all relevant ops.
* Do not use compile-time tricks to hide bad runtime design.

---

## 16. Immediate Next Files to Create

### Root files

* `README.md`
* `CMakeLists.txt`
* `CMakePresets.json`
* `.gitignore`
* `.clang-format`
* `.clang-tidy`

### Core headers

* `include/gpt2lab/core/dtype.hpp`
* `include/gpt2lab/core/shape.hpp`
* `include/gpt2lab/core/tensor.hpp`
* `include/gpt2lab/backends/backend.hpp`
* `include/gpt2lab/model/gpt2_config.hpp`

### First source files

* `src/core/shape.cpp`
* `src/core/tensor.cpp`
* `src/backends/backend_registry.cpp`
* `src/backends/cpu_naive.cpp`
* `src/ops/matmul.cpp`
* `src/model/gpt2_config.cpp`

### First tests

* `tests/unit/test_shape.cpp`
* `tests/unit/test_tensor.cpp`
* `tests/unit/test_matmul.cpp`

### First apps

* `apps/benchmark_matmul.cpp`
* `apps/compare_backends.cpp`
* `apps/dump_tensor.cpp`

---

## 17. Definition of Done for the First Milestone

The first milestone is complete when:

* a tensor can be created and moved between host/device abstractions
* CPU scalar matmul works
* CPU ET matmul works
* Eigen backend matches reference outputs
* at least one CUDA matmul kernel runs successfully
* a benchmark binary can compare all enabled backends
* tests pass for a small deterministic GPT-2-like forward pass
* timing output is reproducible and recorded

---

## 18. Final Note

This project should be treated as a serious backend/runtime laboratory with GPT-2 as the workload. The implementation choices matter more than the model scale. If done well, the final result becomes a strong portfolio piece and a genuine systems-learning project.

Purpose:

* evaluate fusion and compile-time composition
* remove temporary tensors where possible

TODOs:

* [ ] ET expression nodes
* [ ] lazy evaluation model
* [ ] fused elementwise chains
* [ ] compare temp count and runtime against naive CPU
* [ ] verify compile-time complexity remains manageable

#### 5.3 Eigen Backend

Purpose:

* trusted CPU baseline
* performance and correctness reference

TODOs:

* [ ] backend wrapper around Eigen matrices/tensors
* [ ] map internal layouts to Eigen layouts
* [ ] validate numerics against reference
* [ ] benchmark against hand-written CPU kernels

#### 5.4 CUDA Custom Backend

Purpose:

* experiment with hand-written GPU kernels
* study launch overhead, shared memory, tiling, and memory coalescing

TODOs:

* [ ] device tensor storage
* [ ] cudaMemcpy / async transfers
* [ ] custom matmul kernel
* [ ] custom layernorm kernel
* [ ] custom softmax kernel
* [ ] fused kernels where useful
* [ ] stream and event timing
* [ ] occupancy and memory bandwidth experiments

#### 5.5 cuBLAS / cuDNN Backend

Purpose:

* vendor baseline and optimized comparison point

TODOs:

* [ ] cuBLAS matmul wrapper
* [ ] cuDNN primitives where they actually fit the workload
* [ ] benchmark against custom CUDA kernels
* [ ] document when library overhead is worthwhile
* [ ] compare FP32 / FP16 / BF16 paths if available

#### 5.6 Autodiff Backend / Adapter

Purpose:

* compare manual backward implementation against automated differentiation

TODOs:

* [ ] evaluate `autodiff` library integration
* [ ] define adapter layer for scalar or small tensor components
* [ ] benchmark graph construction cost
* [ ] benchmark backward-pass cost
* [ ] compare gradient correctness against manual backward

---

## 6. Tensor and Storage Design

### Tensor core TODOs

* [ ] shape representation
* [ ] dtype enumeration
* [ ] storage with ownership semantics
* [ ] tensor views and slicing
* [ ] contiguous and strided layout support
* [ ] host/device placement
* [ ] copy and move semantics
* [ ] zero-initialization and uninitialized allocation modes
* [ ] debug bounds checking mode
* [ ] pretty-print and tensor dumping utilities

### Recommended tensor metadata

* shape
* stride
* dtype
* device
* layout
* requires_grad
* grad reference
* storage pointer / handle

### Key design choice

Start with one main contiguous layout. Add strided views later. Do not begin with a generalized tensor algebra monster.

---

## 7. Autograd and Differentiation Plan

### Manual backward path

This should be the primary path.

TODOs:

* [ ] define `Node` / `GradFn` abstraction
* [ ] record operation graph during forward pass
* [ ] implement backward for each op
* [ ] handle broadcasting reductions properly
* [ ] handle matmul gradient formulas
* [ ] handle layernorm backward
* [ ] handle softmax backward
* [ ] handle attention backward if implemented
* [ ] add gradient checking tests

### Autodiff experiment path

Treat this as an evaluation branch, not the core engine.

TODOs:

* [ ] integrate `autodiff` as an optional dependency
* [ ] create adapter benchmarks
* [ ] compare API friction
* [ ] compare runtime and compile-time overhead
* [ ] document whether it is practical for this workload

### Gradient checking

TODOs:

* [ ] finite-difference checker
* [ ] relative and absolute tolerance policy
* [ ] small-shape test suite
* [ ] random-seed controlled reproducibility

---

## 8. GPT-2 Model Scope

Do not start with full-scale GPT-2. Start with a reduced GPT-2 variant.

### Model components

* token embedding
* positional embedding
* layer norm
* causal self-attention
* MLP block
* residual connections
* LM head
* loss function

### Model TODOs

* [ ] define configuration struct
* [ ] build token and position embeddings
* [ ] implement transformer block
* [ ] implement causal mask logic
* [ ] implement inference path
* [ ] implement training path
* [ ] implement loss computation
* [ ] add weight initialization
* [ ] add weight serialization/deserialization
* [ ] add checkpoint save/load

### GPT-2 configuration knobs

* number of layers
* hidden size
* number of attention heads
* head dimension
* sequence length
* vocabulary size
* dropout probability
* precision mode
* backend selection

---

## 9. Benchmark Plan

The benchmark framework is one of the main deliverables.

### Microbenchmarks

* matmul only
* elementwise chains
* layernorm only
* softmax only
* attention only
* allocation/free overhead
* host-device transfer overhead
* fused vs unfused execution

### End-to-end benchmarks

* forward pass latency
* backward pass latency
* training step latency
* inference token latency
* tokens/sec
* memory usage peak
* numerical error compared to baseline

### Benchmark metadata

Every benchmark run should record:

* backend name
* device
* precision
* batch size
* sequence length
* hidden dimension
* number of layers
* compiler flags
* build type
* timestamp
* git commit hash
* average latency
* median latency
* p95/p99 latency where relevant
* throughput
* peak memory usage

### Benchmark TODOs

* [ ] define benchmark config format
* [ ] create benchmark runner
* [ ] capture timing statistics
* [ ] export CSV/JSON reports
* [ ] plot comparisons
* [ ] add warmup and steady-state phases
* [ ] add deterministic seed control

---

## 10. Testing Plan

### Test categories

#### Unit tests

* shape utilities
* tensor creation and ownership
* broadcasting rules
* matmul correctness
* softmax correctness
* layernorm correctness
* activation correctness
* backend registry

#### Integration tests

* forward pass consistency across backends
* backward pass consistency across backends
* checkpoint save/load round trip
* deterministic inference output

#### Numeric checks

* absolute and relative tolerance comparisons
* gradient checks with finite differences
* cross-backend output parity

### Testing TODOs

* [ ] GoogleTest or Catch2 setup
* [ ] fixture for small deterministic tensors
* [ ] golden output generator
* [ ] CI test matrix for CPU builds
* [ ] optional CUDA test jobs

---

## 11. CMake Target Sketch

Suggested targets:

```text
core
ops
autograd
backend_cpu_naive
backend_cpu_et
backend_eigen
backend_cuda
backend_cublas
backend_cudnn
model
runtime
bench
train_gpt2
infer_gpt2
benchmark_matmul
benchmark_attention
compare_backends
gradient_check
```

### CMake TODOs

* [ ] create interface target for common compiler settings
* [ ] split CPU and CUDA compilation cleanly
* [ ] keep Eigen and autodiff optional
* [ ] add build options for each backend
* [ ] add `BUILD_TESTING`
* [ ] add sanitizers for CPU debug builds
* [ ] add CUDA architecture selection
* [ ] add warnings-as-errors option
* [ ] add profiling-friendly release build flags

---

## 12. Build Options to Support

Recommended CMake options:

* `GPT2LAB_BUILD_TESTS`
* `GPT2LAB_BUILD_BENCHMARKS`
* `GPT2LAB_BUILD_APPS`
* `GPT2LAB_ENABLE_CUDA`
* `GPT2LAB_ENABLE_CUBLAS`
* `GPT2LAB_ENABLE_CUDNN`
* `GPT2LAB_ENABLE_EIGEN`
* `GPT2LAB_ENABLE_AUTODIFF`
* `GPT2LAB_ENABLE_SANITIZERS`
* `GPT2LAB_ENABLE_LTO`
* `GPT2LAB_ENABLE_WARNINGS_AS_ERRORS`
* `GPT2LAB_ENABLE_EXPRESSION_TEMPLATES`

---

## 13. TODO Roadmap by Phase

### Phase 1: Core infrastructure

* [ ] create repository skeleton
* [ ] define tensor, dtype, shape, storage
* [ ] implement CPU scalar reference ops
* [ ] implement basic tests
* [ ] build CMake structure

### Phase 2: CPU experimentation

* [ ] add blocked CPU matmul
* [ ] add SIMD CPU kernels
* [ ] add expression-template CPU backend
* [ ] benchmark against Eigen
* [ ] validate all outputs against reference

### Phase 3: Autograd

* [ ] implement manual backward graph
* [ ] add gradient checking
* [ ] optionally wire `autodiff`
* [ ] compare manual vs autodiff tradeoffs

### Phase 4: CUDA

* [ ] implement device storage
* [ ] implement CUDA matmul
* [ ] implement CUDA elementwise kernels
* [ ] implement timing with CUDA events
* [ ] benchmark against CPU backends

### Phase 5: Vendor library comparison

* [ ] add cuBLAS matmul backend
* [ ] add cuDNN where applicable
* [ ] compare against custom CUDA kernels
* [ ] document observations

### Phase 6: GPT-2 model

* [ ] implement transformer block
* [ ] implement inference
* [ ] implement training step
* [ ] add checkpointing
* [ ] run end-to-end backend comparison

### Phase 7: Research-grade benchmarking

* [ ] create repeatable benchmark suite
* [ ] automate result export
* [ ] generate plots and summaries
* [ ] compare precision modes
* [ ] compare batch and sequence scaling

---

## 14. Coding Conventions

### General rules

* Prefer explicit ownership semantics.
* Keep backend-specific code isolated.
* Keep tensor API stable.
* Measure before optimizing.
* Add tests before adding more abstraction.
* Avoid over-abstracting the first version.

### Style rules

* Use `snake_case` for functions and variables.
* Use `PascalCase` for types and backends.
* Avoid macros except for very narrow build/platform cases.
* Prefer `constexpr` only when it improves clarity or performance.
* Treat every benchmark as something that may lie unless carefully controlled.

---

## 15. Important Design Warnings

* Do not let expression templates spread into every layer.
* Do not make the backend interface too wide.
* Do not begin with attention fusion or flash attention.
* Do not mix benchmarking code into core tensor logic.
* Do not rely on autodiff as the main system.
* Do not assume cuDNN will cover all relevant ops.
* Do not use compile-time tricks to hide bad runtime design.

---

## 16. Immediate Next Files to Create

### Root files

* `README.md`
* `CMakeLists.txt`
* `CMakePresets.json`
* `.gitignore`
* `.clang-format`
* `.clang-tidy`

### Core headers

* `include/gpt2lab/core/dtype.hpp`
* `include/gpt2lab/core/shape.hpp`
* `include/gpt2lab/core/tensor.hpp`
* `include/gpt2lab/backends/backend.hpp`
* `include/gpt2lab/model/gpt2_config.hpp`

### First source files

* `src/core/shape.cpp`
* `src/core/tensor.cpp`
* `src/backends/backend_registry.cpp`
* `src/backends/cpu_naive.cpp`
* `src/ops/matmul.cpp`
* `src/model/gpt2_config.cpp`

### First tests

* `tests/unit/test_shape.cpp`
* `tests/unit/test_tensor.cpp`
* `tests/unit/test_matmul.cpp`

### First apps

* `apps/benchmark_matmul.cpp`
* `apps/compare_backends.cpp`
* `apps/dump_tensor.cpp`

---

## 17. Definition of Done for the First Milestone

The first milestone is complete when:

* a tensor can be created and moved between host/device abstractions
* CPU scalar matmul works
* CPU ET matmul works
* Eigen backend matches reference outputs
* at least one CUDA matmul kernel runs successfully
* a benchmark binary can compare all enabled backends
* tests pass for a small deterministic GPT-2-like forward pass
* timing output is reproducible and recorded

---

## 18. Final Note

This project should be treated as a serious backend/runtime laboratory with GPT-2 as the workload. The implementation choices matter more than the model scale. If done well, the final result becomes a strong portfolio piece and a genuine systems-learning project.
