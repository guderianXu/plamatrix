# PlaMatrix Vulkan Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional Vulkan Compute float32 PCG backend and a reproducible OpenCL/Vulkan comparison benchmark.

**Architecture:** Mirror the existing OpenCL CPU-owned CSR PCG interface. Keep Vulkan object ownership in a small runtime/execution layer, compile GLSL shaders to SPIR-V at build time, and expose backend availability without changing CUDA/OpenCL behavior.

**Tech Stack:** C++17, Vulkan 1.1, GLSLang/SPIR-V, GoogleTest, CMake, existing OpenCL runtime and benchmark utilities.

**Spec:** `docs/superpowers/specs/2026-09-19-vulkan-backend-design.md`

## Global Constraints

- Vulkan is optional and disabled unless `PLAMATRIX_WITH_VULKAN=ON`.
- Vulkan uses Vulkan 1.1 Compute API only; no window system or vendor-specific extensions.
- Device selection uses `PLAMATRIX_VULKAN_DEVICE_INDEX`.
- GLSL shaders are compiled by `glslangValidator` during the build.
- All Vulkan objects and allocations use RAII ownership.
- First phase supports float32 PCG only and does not add Vulkan Schur/SVD/GEMM.

## Review Focus

- No Vulkan SDK/compiler: auto-disable unless explicitly requested, with a clear CMake diagnostic.
- No usable compute queue or device index out of range: fail with a diagnostic instead of silently using another device.
- Zero-length and malformed CSR inputs: reject before Vulkan allocation.
- Fence/command-buffer/pipeline failures: release all Vulkan objects during exceptions.
- Vulkan and OpenCL results: use identical tolerances and report residual/iteration mismatches.

### Task 1: Vulkan configuration and shader build

**Files:**
- Modify: `CMakeLists.txt`, `src/CMakeLists.txt`
- Create: `src/vulkan/shaders/*.comp`
- Test: CMake configure checks and a shader compilation smoke target

Interfaces:
- Produces `PLAMATRIX_WITH_VULKAN`, `PLAMATRIX_VULKAN_SHADER_DIR`, and compiled `.spv` files for later tasks.

- [ ] Add `PLAMATRIX_WITH_VULKAN` option, `find_package(Vulkan QUIET)`, and `find_program(glslangValidator)` with explicit-on failure.
- [ ] Add custom commands for `spmv.comp`, `dot_reduce.comp`, `vector_update.comp`, and `jacobi.comp`.
- [ ] Add Vulkan sources conditionally and expose status in CMake configuration output.
- [ ] Configure CPU/OpenCL/CUDA builds with Vulkan disabled and verify no new required headers or libraries.

### Task 2: Runtime and RAII execution layer

**Files:**
- Create: `include/plamatrix/vulkan/runtime.h`
- Create: `include/plamatrix/vulkan/execution.h`
- Create: `src/vulkan/runtime.cpp`
- Test: `test/unit/vulkan/runtime_test.cpp`

Interfaces:
- `vulkan::Runtime::instance()`, `hasUsableVulkanDevice()`, `enumerateVulkanDevices()`.
- Move-only `vulkan::Buffer`, `vulkan::CommandContext`, and `vulkan::ShaderPipeline` wrappers.

- [ ] Implement instance/device/compute-queue selection and `PLAMATRIX_VULKAN_DEVICE_INDEX` validation.
- [ ] Implement memory-type selection, host-visible buffer creation, command pool, one-time command submission, and checked fences.
- [ ] Implement shader-module, descriptor-layout, descriptor-pool, pipeline-layout, and compute-pipeline ownership.
- [ ] Add tests for device enumeration, explicit invalid index, zero-byte buffer rejection, and move/destructor behavior when a Vulkan device is usable.

### Task 3: Vulkan PCG primitives and public API

**Files:**
- Create: `include/plamatrix/vulkan/iterative_solver.h`
- Create: `src/vulkan/iterative_solver.cpp`
- Modify: `include/plamatrix/plamatrix.h`, `src/CMakeLists.txt`
- Test: `test/unit/vulkan/iterative_solver_test.cpp`

Interfaces:
- `vulkan::pcg(const CSRMatrix<float, Device::CPU>&, const DenseMatrix<float, Device::CPU>&, DenseMatrix<float, Device::CPU>&, const IterativeSolverOptions&)`.
- `vulkan::selectedVulkanDeviceName()` and `vulkan::hasUsableVulkanDevice()`.

- [ ] Validate square CSR, finite CPU inputs, distinct output, positive diagonal/preconditioner, and nonzero dimensions before allocation.
- [ ] Upload CSR/rhs/state buffers and create cached compute pipelines from Task 1 shaders.
- [ ] Dispatch SpMV, Jacobi, vector update, and deterministic dot reduction kernels with fence synchronization.
- [ ] Implement convergence interval and final solution download matching OpenCL report fields.
- [ ] Add Vulkan-vs-CPU correctness tests and no-Vulkan stub tests.

### Task 4: OpenCL/Vulkan comparison benchmark

**Files:**
- Create: `benchmark/backend_compare.cpp`
- Modify: `benchmark/CMakeLists.txt`, `benchmark/benchmark_cases.h`
- Test: benchmark smoke/CSV schema test

- [ ] Generate one deterministic float32 SPD CSR fixture per size and reuse it for both backends.
- [ ] Measure initialization/upload, kernel-only, download, and end-to-end PCG with equal warmup/trial counts.
- [ ] Emit backend name, device name, size, iterations, initial/final residual, and timing columns.
- [ ] Make unavailable OpenCL/Vulkan backends report `skipped` with a reason rather than failing the benchmark executable.

### Task 5: Documentation and verification matrix

**Files:**
- Modify: `docs/architecture.md`, `README.md`, `docs/build.md`
- Test: CPU-only, OpenCL-only, CUDA/OpenCL, and Vulkan-enabled builds

- [ ] Document Vulkan prerequisites, shader compilation, device selection, and benchmark command.
- [ ] Run CPU full tests, Vulkan runtime/solver tests, and the comparison benchmark on every usable backend.
- [ ] Record exact device availability and any skipped backend in the final report.
- [ ] Commit the implementation, push the branch, fast-forward/merge into `main`, verify remote `main`, then delete local and remote feature branch as requested.
