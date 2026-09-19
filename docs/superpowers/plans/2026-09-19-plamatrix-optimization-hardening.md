# PlaMatrix Optimization Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve PlaMatrix's GPU ownership safety, allocation efficiency, solver hot paths, backend consistency, and regression coverage while preserving existing public APIs where possible.

**Architecture:** Add small internal ownership/context helpers instead of changing matrix storage layout broadly. Reuse uninitialized output paths and workspaces in hot operations, keep CPU/GPU/OpenCL behavior explicit, and pin every new behavior with focused tests.

**Tech Stack:** C++17, CUDA/cuBLAS/cuSOLVER/cuSPARSE, OpenCL 1.2, OpenMP, GoogleTest, CMake.

**Spec:** The ten optimization recommendations approved in the parent conversation.

## Global Constraints

- Preserve column-major dense storage and existing `Device::CPU`/`Device::GPU` template API.
- Keep CPU-only builds compiling without CUDA or OpenCL.
- Do not update the parent repository's submodule pointer automatically.
- Do not rewrite reference data unless a numerical definition changes.
- Every production change gets a focused regression test before implementation.

## Review Focus

- Reusing a GPU allocation after switching CUDA devices must never return a pointer from another device.
- CUDA library handles must not be shared unsafely across threads or devices.
- Allocating output operations must not read uninitialized values or add an unnecessary clear.
- Async host transfers must either use pinned memory or fail clearly.
- Wide GPU SVD, empty dimensions, and integer-size boundaries must retain explicit behavior.

### Task 1: Device-aware bounded GPU memory pool

**Files:** `include/plamatrix/core/allocator.h`, `test/unit/core/allocator_test.cpp`.

Add device identity to pool buckets, synchronize enable/limit state, expose a byte limit and per-device release, and test cross-device behavior when multiple devices exist.

### Task 2: CUDA handle ownership and device restoration

**Files:** `src/ops/gemm.cu`, `src/ops/decomposition.cu`, `src/ops/solver.cu`, `src/optimization/block_schur.cpp`, focused CUDA tests.

Use thread/device-scoped handles, set streams where supported, wrap device selection in a restore guard, and test concurrent same-device calls plus device restoration.

### Task 3: Exception-safe reusable CUDA workspaces

**Files:** `include/plamatrix/core/cuda_buffer.h` (new), `src/ops/decomposition.cu`, `src/ops/solver.cu`, headers and tests.

Introduce a minimal RAII device buffer and optional reusable workspace objects so every allocation is released during exceptions and repeated decomposition calls avoid cold allocations.

### Task 4: Uninitialized outputs and pinned DenseMatrix async transfers

**Files:** `include/plamatrix/dense/dense_matrix.h`, `include/plamatrix/dense/dense_ops.h`, `include/plamatrix/dense/elementwise.h`, `src/dense/*.cu`, tests and docs.

Add internal uninitialized construction for overwrite-all outputs, use it in transfers and elementwise/GEMM allocating overloads, and make `DenseMatrix::toCpuAsync()` return pinned host storage.

### Task 5: CPU GEMM and Schur hot paths

**Files:** `src/ops/gemm_cpu.cpp`, `src/optimization/block_normal_equations.cpp`, `src/optimization/block_schur.cpp`, `include/plamatrix/core/checked_math.h` (new), tests and benchmark notes.

Add checked dimension products, reuse packed buffers where possible, tune parallel thresholds, replace repeated cross-block linear scans with indexed lookup, and reuse PCG output buffers.

### Task 6: OpenCL reduction and runtime cache behavior

**Files:** `src/opencl/iterative_solver.cpp`, `src/opencl/runtime.cpp`, `include/plamatrix/opencl/runtime.h`, tests.

Reduce avoidable host readbacks, hash cache identity without retaining duplicate source strings, and make cache lifecycle explicit without changing selected-device semantics.

### Task 7: Sparse direct and block solver numeric paths

**Files:** `src/optimization/block_sparse_cholesky.*`, `src/optimization/block_schur_sparse_direct.cpp`, tests.

Keep the robust double factor path explicit, add checked size validation, reuse factor work buffers, and cover inactive coordinates and float/double refinement boundaries.

### Task 8: Backend consistency

**Files:** `src/ops/decomposition.cu`, `include/plamatrix/ops/decomposition.h`, tests and docs.

Support wide GPU SVD through a transpose-based path or document and test the deliberate fallback consistently.

### Task 9: API/documentation cleanup

**Files:** `README.md`, `docs/api/*.md`, `docs/architecture.md`, public headers.

Document device affinity, pinned async transfer requirements, memory-pool limits, and backend capability differences.

### Task 10: Verification matrix

**Files:** `test/unit`, `test/integration`, CMake test registration.

Run focused CPU tests, CPU-only build/tests, CUDA build/tests when available, OpenCL tests when available, and benchmark smoke/medium cases. Record exact results without claiming unavailable backends.
