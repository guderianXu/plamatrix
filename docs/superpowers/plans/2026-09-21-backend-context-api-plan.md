# PlaMatrix Backend Context API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the split CUDA/OpenCL/Vulkan execution contracts with one backend-aware context/event API, migrating PCG/CSR and all affected call sites directly without a compatibility layer.

**Architecture:** `Backend`, `DeviceId`, `ExecutionContext`, `Event`, and `Error` become the public execution foundation. Matrix residency is represented by backend-bound device containers, while backend-native stream/queue handles remain private to backend implementation headers. PCG is migrated first as the vertical slice, then old public runtime, workspace, and raw-stream declarations are removed.

**Tech Stack:** C++17, CMake, CUDA, OpenCL, Vulkan, GoogleTest, existing PlaMatrix dense/CSR kernels.

**Spec:** `docs/superpowers/specs/2026-09-21-backend-context-api-design.md`

## Global Constraints

- Do not add a compatibility layer or keep old and new public PCG APIs in parallel.
- Keep C++17 and existing CPU/CUDA/MSVC/GCC support in this migration.
- Do not change numerical algorithms or default convergence policy.
- Do not modify PlaScan files until the new PlaMatrix headers and tests compile.
- Do not expose `cudaStream_t`, `cl_command_queue`, or `VkQueue` from stable core headers.
- Freeze scalar/index/layout/aliasing/sparse/determinism semantics before implementing backend code.
- `Event` destruction must not block; context owns deferred reclamation.
- `ExecutionContext` is non-copyable and non-movable so resident objects can safely retain its identity.
- Preserve existing unrelated dirty changes in the parent repository and submodules.

## Review Focus

- CPU-only consumers include the umbrella header without CUDA/OpenCL/Vulkan installed; contract test in Task 1.
- A context created for one backend rejects a matrix or event owned by another backend; tests in Task 2.
- An async PCG failure is reported by `Event::wait()` and does not require manual allocation cleanup; tests in Task 4.
- Reusing a device-resident CSR matrix for multiple solves performs no implicit host transfer; benchmark/assertion in Task 4.
- Backend-unavailable creation returns `ErrorCode::BackendUnavailable` with backend identity; tests in Task 3.
- Device scalar reads never perform hidden synchronization; compile contract in Task 0.
- CSR duplicate/index-base/sortedness policies are explicit and structure mutation invalidates analysis; tests in Task 2.
- Unsupported scalar/layout/backend combinations fail before submission; tests in Task 1.

---

### Task 0: Freeze public type, layout, allocator, and ABI semantics

**Files:**
- Create: `include/plamatrix/core/scalar_traits.h`
- Create: `include/plamatrix/core/shape.h`
- Create: `include/plamatrix/core/layout.h`
- Create: `include/plamatrix/core/memory_space.h`
- Create: `include/plamatrix/core/memory_resource.h`
- Create: `include/plamatrix/core/api.h`
- Modify: `include/plamatrix/core/types.h`
- Modify: `CMakeLists.txt`
- Test: `test/unit/core/type_contract_test.cpp`

**Interfaces:**
- `ScalarTraits<T>` and `isSupportedScalar<T>` define supported scalar/index types and accumulation type.
- `Shape2`, checked element counts, `Layout`, and explicit stride metadata are shared by dense and sparse APIs.
- `MemoryResource` is the allocation seam used by `ExecutionContext`; default, pinned-host, and device resources are distinct.
- `PLAMATRIX_API`, `PLAMATRIX_VERSION_MAJOR`, and `plamatrix::v1` define the exported ABI boundary.

- [ ] **Step 1: Write compile-only tests for supported/unsupported scalar types, row/column-major layout tags, and exported version macros.**
- [ ] **Step 2: Implement traits, shape/layout validation, memory-resource interface, and visibility/version macros.**
- [ ] **Step 3: Add explicit aliasing documentation and `asConst()` to view contracts; defer owning `clone()`/`copyTo()` until the context-bound containers in Task 2.**
- [ ] **Step 4: Add the type-contract target to CMake and run it in CPU-only mode.**

### Task 1: Add core backend, error, context, and event primitives

**Files:**
- Create: `include/plamatrix/core/backend.h`
- Create: `include/plamatrix/core/error.h`
- Create: `include/plamatrix/core/execution_context.h`
- Create: `include/plamatrix/core/event.h`
- Create: `src/core/execution_context.cpp`
- Create: `src/core/event.cpp`
- Modify: `include/plamatrix/plamatrix.h`
- Modify: `src/CMakeLists.txt`
- Test: `test/unit/core/execution_context_test.cpp`
- Test: `test/unit/core/error_test.cpp`

**Interfaces:**
- Produces `enum class Backend`, `struct DeviceId`, `enum class ErrorCode`, `class Error`, `class ExecutionContext`, and move-only `class Event` exactly as specified.
- `ExecutionContext::create(DeviceId)` creates a CPU context unconditionally and reports unavailable optional backends through `Error`.
- `Event::completed()` creates an already-complete CPU event; `wait()` is idempotent and the destructor never blocks.

- [ ] **Step 1: Write failing tests** for CPU context identity, invalid backend errors, move-only event semantics, and repeated `wait()`.
- [ ] **Step 2: Run `cmake --build build --target plamatrix_tests` and verify the new symbols are absent.**
- [ ] **Step 3: Implement the four public headers and source files with no native backend includes in core headers.**
- [ ] **Step 4: Register sources and include the stable primitives from `plamatrix.h`.**
- [ ] **Step 5: Run `ctest --test-dir build -R 'ExecutionContext|Error' --output-on-failure` and verify all new tests pass.**

### Task 2: Replace device selection and residency foundations

**Files:**
- Create: `include/plamatrix/device/device_matrix.h`
- Create: `include/plamatrix/device/device_vector.h`
- Create: `include/plamatrix/device/device_csr_matrix.h`
- Create: `src/device/device_matrix.cpp`
- Create: `src/device/device_csr_matrix.cpp`
- Modify: `include/plamatrix/core/types.h`
- Modify: `include/plamatrix/dense/dense_matrix.h`
- Modify: `include/plamatrix/sparse/csr_matrix.h`
- Modify: `src/CMakeLists.txt`
- Test: `test/unit/device/device_residency_test.cpp`

**Interfaces:**
- `MemorySpace` distinguishes `Host`, `Cuda`, `OpenCl`, and `Vulkan` storage.
- `ResidentMatrix<T>` and `ResidentVector<T>` carry a non-owning `ExecutionContext*` association and move-only ownership.
- `ResidentCsrMatrix<T>` owns CSR structure/value buffers and exposes `rows()`, `cols()`, `nnz()`, `context()`, and `memorySpace()`.
- Host-to-device construction is explicit: `ResidentCsrMatrix<T>::copyFrom(host, context)`.
- CSR construction options make index type, index base, sortedness and duplicate policy explicit; structure mutation invalidates analysis metadata.

- [ ] **Step 1: Add failing tests for host copy, context binding, shape metadata, and cross-context rejection.**
- [ ] **Step 2: Implement CPU storage first so the tests run without optional backends.**
- [ ] **Step 3: Reserve the backend storage seam without exposing native handles; implement CUDA/OpenCL/Vulkan allocation only after their context factories exist in Task 3.**
- [ ] **Step 4: Make optional backends return `BackendUnavailable` until their context allocators are wired.**
- [ ] **Step 5: Remove public `Device::GPU` residency use from the migrated CSR path and update relevant tests.**
- [ ] **Step 6: Run the device unit tests and CPU-only compile contract.**

### Task 3: Add backend capabilities and context availability

**Files:**
- Modify: `include/plamatrix/opencl/runtime.h`
- Modify: `src/opencl/runtime.cpp`
- Modify: `src/opencl/runtime_stub.cpp`
- Modify: `include/plamatrix/vulkan/runtime.h`
- Modify: `src/vulkan/runtime.cpp`
- Modify: `src/vulkan/iterative_solver_stub.cpp`
- Create: `include/plamatrix/cuda/native.h`
- Create: `include/plamatrix/opencl/native.h`
- Create: `include/plamatrix/vulkan/native.h`
- Create: `include/plamatrix/core/capabilities.h`
- Create: `include/plamatrix/core/diagnostics.h`
- Create: `include/plamatrix/core/cancellation.h`
- Modify: `include/plamatrix/plamatrix.h`
- Test: `test/unit/opencl/opencl_runtime_test.cpp`
- Test: `test/unit/vulkan/vulkan_runtime_test.cpp`

**Interfaces:**
- `ExecutionContext::create({Backend::Cuda/OpenCl/Vulkan, index})` validates explicit device availability and records backend capabilities.
- `ExecutionContext::capabilities()` reports scalar, layout, sparse-format, cooperative-matrix, asynchronous-allocation, and command-buffer support.
- `BackendDiagnostics` and `ExecutionTrace` hold backend-specific timing and kernel-path data; generic solver reports do not grow backend booleans.
- `CancellationToken` is polled by long-running iterative operations and is independent of GUI/task-runtime types.
- Native handles are available only through backend-specific native headers and are never required by generic operations.
- Existing runtime implementations remain internal during this task; their public singleton removal is completed with the operation migration in Task 7.

- [ ] **Step 1: Write failing runtime tests for explicit device index, unavailable backend error, and context lifetime.**
- [ ] **Step 2: Implement backend context factories around existing runtime initialization code.**
- [ ] **Step 3: Move capability and native-handle declarations to backend-specific headers without adding native handles to core headers.**
- [ ] **Step 4: Update stubs to return structured errors rather than defining fake CUDA/OpenCL/Vulkan APIs.**
- [ ] **Step 5: Run CPU-only and available-backend context tests.**

### Task 4: Directly migrate PCG/CSR and remove old solver API

**Files:**
- Modify: `include/plamatrix/sparse/iterative_solver.h`
- Modify: `src/sparse/iterative_solver.cu`
- Modify: `src/sparse/iterative_solver_cpu.cpp`
- Modify: `src/sparse/iterative_solver_workspace.cu`
- Modify: `include/plamatrix/opencl/iterative_solver.h`
- Modify: `src/opencl/iterative_solver.cpp`
- Modify: `include/plamatrix/vulkan/iterative_solver.h`
- Modify: `src/vulkan/iterative_solver.cpp`
- Modify: `src/vulkan/iterative_solver_stub.cpp`
- Modify: `include/plamatrix/sparse/sparse_ops.h`
- Modify: `src/sparse/sparse_ops.cu`
- Test: `test/unit/sparse/iterative_solver_test.cpp`
- Test: `test/unit/opencl/iterative_solver_opencl_test.cpp`
- Test: `test/unit/vulkan/pcg_context_test.cpp`

**Interfaces:**
- Replace old `IterativeSolverWorkspace`, `AsyncIterativeSolverState`, and raw-stream overloads with context-owned scratch and `Event pcgAsync(...)`.
- `IterativeSolverReport` contains only generic convergence fields plus an optional diagnostics object.
- `SolveOptions` records tolerance, accumulation precision, determinism, NaN policy, cancellation token and residual-history policy.
- `pcg` and `blockPcg` consume `ResidentCsrMatrix`/`ResidentVector` for all accelerator backends.
- Remove CPU-owned OpenCL/Vulkan solver declarations and all `cudaStream_t` solver parameters.

- [ ] **Step 1: Add context-based CPU tests and a CUDA numerical regression test before deleting old declarations.**
- [ ] **Step 2: Move existing CUDA workspace allocations behind `ExecutionContext` and return an `Event` for asynchronous submission.**
- [ ] **Step 3: Port OpenCL and Vulkan submission to the same device-resident inputs and report structure.**
- [ ] **Step 4: Delete old solver/workspace public declarations and update all PlaMatrix tests and call sites.**
- [ ] **Step 5: Add a repeated-solve assertion that counts no host transfers after initial `copyFrom`.**
- [ ] **Step 6: Run sparse, OpenCL, Vulkan, and CUDA solver tests.**

### Task 5: Remove manual async allocation APIs and fake CUDA stubs

**Files:**
- Modify: `include/plamatrix/dense/dense_matrix.h`
- Modify: `include/plamatrix/sparse/csr_matrix.h`
- Modify: `include/plamatrix/sparse/sparse_ops.h`
- Modify: `include/plamatrix/ops/indexing.h`
- Modify: `include/plamatrix/ops/reduction.h`
- Modify: `include/plamatrix/ops/small_matrix.h`
- Delete: `include/plamatrix/core/no_cuda_stubs.h`
- Modify: `src/dense/dense_matrix.cu`
- Modify: `src/sparse/csr_matrix.cpp`
- Modify: `src/ops/indexing_workspace.cu`
- Modify: `src/ops/reduction_workspace.cu`
- Test: `test/unit/core/no_backend_header_test.cpp`

**Interfaces:**
- Async operations accept `ExecutionContext&` and return `Event` where applicable.
- No public `closeAsyncAllocation`, `reserveBytesAsync`, or raw stream parameter remains.
- CPU-only headers compile without fake global CUDA names.

- [ ] **Step 1: Add a compile-only test that fails if `cudaStream_t` or stub CUDA functions are visible from `plamatrix.h`.**
- [ ] **Step 2: Replace each migrated async allocation with context-owned storage.**
- [ ] **Step 3: Delete manual close/reset APIs and remove stale stream lifetime documentation.**
- [ ] **Step 4: Run all indexing, reduction, and small-matrix async tests.**

### Task 6: Standardize remaining operation signatures and extension seams

**Files:**
- Modify: `include/plamatrix/dense/dense_ops.h`
- Modify: `include/plamatrix/dense/elementwise.h`
- Modify: `include/plamatrix/ops/gemm.h`
- Modify: `include/plamatrix/ops/reduction.h`
- Modify: `include/plamatrix/ops/statistics.h`
- Modify: `include/plamatrix/ops/vector.h`
- Modify: `include/plamatrix/ops/point_cloud.h`
- Modify: `include/plamatrix/ops/decomposition.h`
- Modify: `include/plamatrix/optimization/block_schur.h`
- Modify: `include/plamatrix/optimization/levenberg_marquardt.h`
- Create: `include/plamatrix/ops/options.h`
- Create: `include/plamatrix/ops/assign.h`
- Test: `test/unit/api/operation_signature_test.cpp`

**Interfaces:**
- Every migrated operation has explicit `ExecutionContext&` and uses the parameter order `inputs, output/options, context`.
- `assign`, `eval`, and `reduce` provide the extension seam for future fusion; fixed-arity fusion overloads stop growing.
- `asConst()`, `clone()`, `copyTo()`, `assign()` and explicit alias checks are available on dense/sparse containers.
- Device scalar access never hides a synchronization or transfer.

- [ ] **Step 1: Write compile-only signature tests for one dense, one sparse, one reduction, and one optimization operation.**
- [ ] **Step 2: Add `OperationOptions` and the assign/eval/reduce declarations without implementing expression templates.**
- [ ] **Step 3: Migrate dense, reduction, statistics, point-cloud, and optimization headers and their direct callers.**
- [ ] **Step 4: Remove global-runtime lookups and hidden device scalar transfers from migrated implementations.**
- [ ] **Step 5: Run the affected unit and integration tests.**

### Task 7: Migrate PlaScan call sites, documentation, and packaging

**Files:**
- Modify: all PlaScan files found by `rg -n 'plamatrix|Device::GPU|cudaStream_t|pcg\(' src tests CMakeLists.txt`
- Modify: `3rdparty/plamatrix/README.md`
- Modify: `3rdparty/plamatrix/docs/api/*.md`
- Modify: `3rdparty/plamatrix/CMakeLists.txt`
- Modify: `3rdparty/plamatrix/docs/architecture.md`

**Interfaces:**
- PlaScan creates one explicit context per backend task and passes it through the solver/operation call graph.
- Documentation shows only the new context-based API and explicit transfer boundaries.
- Package exports install only stable core and selected backend native headers.

- [ ] **Step 1: Search for every old public symbol and update each call site directly.**
- [ ] **Step 2: Add a consumer example that compiles against the installed package using CPU context.**
- [ ] **Step 3: Update API and migration documentation, including the breaking-change notice.**
- [ ] **Step 4: Run `git diff --check` and the complete native test suite.**

### Task 8: Final verification and release notes

**Files:**
- Modify: `3rdparty/plamatrix/CHANGELOG.md`
- Modify: `3rdparty/plamatrix/docs/releases/v1.0.0-api-migration.md`

- [ ] **Step 1: Run the required CPU build and full CTest suite.**
- [ ] **Step 2: Run CUDA/OpenCL/Vulkan tests available on the host and record unavailable backends explicitly.**
- [ ] **Step 3: Verify there are no old public symbols or compatibility headers with `rg`.**
- [ ] **Step 4: Review the complete diff for unrelated changes, temporary files, and accidental parent-repository edits.**
