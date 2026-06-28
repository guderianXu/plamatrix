# Metal GPU Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a macOS Metal/MPS backend behind the existing `Device::GPU` API while keeping CUDA behavior intact and providing CPU/Accelerate fallback for macOS GPU double operations.

**Architecture:** Introduce a platform GPU backend selector (`CUDA`, `METAL`, `NONE`) and a small runtime abstraction for GPU allocation, transfer, and memset. CUDA keeps the existing `.cu` implementation files; Metal adds `.mm` implementation files backed by `MTLBuffer`, custom Metal kernels, MPSMatrix, and Accelerate fallbacks. Public templates and user code continue to use `DenseMatrix<Scalar, Device::GPU>`.

**Tech Stack:** C++17, CMake 3.18+, Objective-C++ for Metal files, Metal, MetalPerformanceShaders, Accelerate, optional OpenMP, Google Test, existing CUDA/cuBLAS/cuSOLVER backend.

---

## File Structure

Create:

- `include/plamatrix/core/gpu_backend.h`: public backend enum and `gpuBackendName()` query.
- `include/plamatrix/core/gpu_runtime.h`: C++ backend-neutral allocation/copy/memset wrappers.
- `include/plamatrix/core/metal_context.h`: C++ declarations for Metal context and kernels, no Objective-C types.
- `src/core/gpu_backend.cpp`: compile-time backend query implementation.
- `src/core/gpu_runtime_cuda.cpp`: CUDA/NONE runtime wrapper implementation.
- `src/core/metal_context.mm`: Metal device, command queue, runtime shader compilation, buffer registry.
- `src/core/gpu_runtime_metal.mm`: Metal allocation/copy/memset wrapper implementation.
- `src/dense/dense_matrix_metal.mm`: Metal `fillGpuKernel()` and `transposeGpuKernel()`.
- `src/dense/dense_ops_metal.mm`: Metal `add/sub` implementations.
- `src/ops/gemm_metal.mm`: float MPS GEMM and double CPU fallback.
- `src/ops/solver_metal.mm`: float MPS LU solve and double CPU fallback.
- `src/ops/point_cloud_metal.mm`: float Metal point-cloud kernels and double CPU fallback.
- `src/ops/decomposition_metal.mm`: macOS GPU SVD/QR/eigh CPU fallback.
- `test/integration/gpu_backend_test.cpp`: backend query and no-GPU behavior tests.
- `test/integration/metal_backend_test.cpp`: Metal-specific consistency tests guarded by `PLAMATRIX_WITH_METAL`.

Modify:

- `CMakeLists.txt`: backend selection, framework linking, optional OpenMP.
- `src/CMakeLists.txt`: add core, CUDA, Metal, and no-GPU source groups.
- `test/CMakeLists.txt`: add new integration tests.
- `benchmark/CMakeLists.txt`: use backend macros instead of only CUDA checks.
- `cmake/plamatrixConfig.cmake.in`: export selected backend dependencies.
- `include/plamatrix/plamatrix.h`: include `gpu_backend.h`.
- `include/plamatrix/core/error_check.h`: support non-CUDA GPU builds cleanly.
- `include/plamatrix/core/no_cuda_stubs.h`: provide CUDA-compatible type stubs when CUDA is not compiled.
- `include/plamatrix/core/allocator.h`: route GPU allocation through `gpu_runtime.h`.
- `include/plamatrix/core/parallel.h`: add optional OpenMP pragma macros.
- `include/plamatrix/dense/dense_matrix.h`: route transfers/memset through `gpu_runtime.h`.
- `include/plamatrix/dense/dense_ops.h`, `src/ops/gemm_cpu.cpp`: guard OpenMP pragmas.
- `include/plamatrix/ops/decomposition.h`, `include/plamatrix/ops/solver.h`: add no-GPU stubs where missing.
- `test/test_main.cpp`: skip only unavailable backend tests, not all non-CUDA GPU tests.
- `README.md`, `docs/build.md`, `docs/architecture.md`, `docs/api/dense-matrix.md`, `docs/api/linear-algebra.md`, `docs/api/point-cloud.md`, `docs/contributing.md`: document backend selection and macOS fallback behavior.

---

## Task 1: Backend Query API and Tests

**Files:**
- Create: `include/plamatrix/core/gpu_backend.h`
- Create: `src/core/gpu_backend.cpp`
- Create: `test/integration/gpu_backend_test.cpp`
- Modify: `include/plamatrix/plamatrix.h`
- Modify: `src/CMakeLists.txt`
- Modify: `test/CMakeLists.txt`

- [ ] **Step 1: Write the failing backend query test**

Add `test/integration/gpu_backend_test.cpp`:

```cpp
#include <gtest/gtest.h>

#include <cstring>

#include <plamatrix/core/gpu_backend.h>

using namespace plamatrix;

TEST(GpuBackend, query_ReturnsConfiguredBackendName)
{
    const char* name = gpuBackendName();
    ASSERT_NE(name, nullptr);

#ifdef PLAMATRIX_WITH_CUDA
    EXPECT_EQ(std::strcmp(name, "cuda"), 0);
    EXPECT_EQ(gpuBackend(), GpuBackend::Cuda);
#elif defined(PLAMATRIX_WITH_METAL)
    EXPECT_EQ(std::strcmp(name, "metal"), 0);
    EXPECT_EQ(gpuBackend(), GpuBackend::Metal);
#else
    EXPECT_EQ(std::strcmp(name, "none"), 0);
    EXPECT_EQ(gpuBackend(), GpuBackend::None);
#endif
}
```

Add the file to `test/CMakeLists.txt` inside `add_executable(plamatrix_tests ...)`:

```cmake
    integration/gpu_backend_test.cpp
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
cmake -S . -B build-none -DPLAMATRIX_BUILD_TESTS=ON -DPLAMATRIX_WITH_CUDA=OFF
cmake --build build-none --target plamatrix_tests -j$(sysctl -n hw.ncpu)
```

Expected: compile fails because `plamatrix/core/gpu_backend.h` does not exist.

- [ ] **Step 3: Add the backend query API**

Create `include/plamatrix/core/gpu_backend.h`:

```cpp
#pragma once

namespace plamatrix
{

enum class GpuBackend
{
    None,
    Cuda,
    Metal
};

/// Return the GPU backend selected at configure time.
/// @return Selected backend enum value.
GpuBackend gpuBackend();

/// Return the selected backend name: "cuda", "metal", or "none".
/// @return Stable string literal for the configured GPU backend.
const char* gpuBackendName();

} // namespace plamatrix
```

Create `src/core/gpu_backend.cpp`:

```cpp
#include "plamatrix/core/gpu_backend.h"

namespace plamatrix
{

GpuBackend gpuBackend()
{
#ifdef PLAMATRIX_WITH_CUDA
    return GpuBackend::Cuda;
#elif defined(PLAMATRIX_WITH_METAL)
    return GpuBackend::Metal;
#else
    return GpuBackend::None;
#endif
}

const char* gpuBackendName()
{
#ifdef PLAMATRIX_WITH_CUDA
    return "cuda";
#elif defined(PLAMATRIX_WITH_METAL)
    return "metal";
#else
    return "none";
#endif
}

} // namespace plamatrix
```

Modify `include/plamatrix/plamatrix.h`:

```cpp
#include "plamatrix/core/gpu_backend.h"
```

Modify `src/CMakeLists.txt` CPU source list:

```cmake
target_sources(plamatrix PRIVATE
    core/gpu_backend.cpp
    sparse/csr_matrix.cpp
    ops/gemm_cpu.cpp
    ops/decomposition_cpu.cpp
    ops/solver_cpu.cpp
    ops/point_cloud_cpu.cpp
)
```

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
cmake -S . -B build-none -DPLAMATRIX_BUILD_TESTS=ON -DPLAMATRIX_WITH_CUDA=OFF
cmake --build build-none --target plamatrix_tests -j$(sysctl -n hw.ncpu)
./build-none/test/plamatrix_tests --gtest_filter=GpuBackend.*
```

Expected: test passes and reports `"none"` for the current pre-Metal no-CUDA build.

- [ ] **Step 5: Checkpoint**

Run:

```bash
git status --short
```

If the current session has explicit commit authorization, commit:

```bash
git add include/plamatrix/core/gpu_backend.h src/core/gpu_backend.cpp include/plamatrix/plamatrix.h src/CMakeLists.txt test/CMakeLists.txt test/integration/gpu_backend_test.cpp
git commit -m "feat: add GPU backend query API"
```

---

## Task 2: CMake Backend Selection and Optional OpenMP

**Files:**
- Modify: `CMakeLists.txt`
- Modify: `cmake/plamatrixConfig.cmake.in`
- Modify: `include/plamatrix/core/parallel.h`
- Modify: `include/plamatrix/dense/dense_ops.h`
- Modify: `src/ops/gemm_cpu.cpp`
- Modify: `test/integration/gpu_backend_test.cpp`

- [ ] **Step 1: Add a failing CMake-level expectation**

Extend `test/integration/gpu_backend_test.cpp`:

```cpp
TEST(GpuBackend, compileDefinitions_AreMutuallyExclusive)
{
#if defined(PLAMATRIX_WITH_CUDA) && defined(PLAMATRIX_WITH_METAL)
    FAIL() << "CUDA and Metal backends must not be enabled together";
#elif defined(PLAMATRIX_WITH_CUDA)
    SUCCEED() << "CUDA backend selected";
#elif defined(PLAMATRIX_WITH_METAL)
    SUCCEED() << "Metal backend selected";
#else
    SUCCEED() << "No GPU backend selected";
#endif
}
```

- [ ] **Step 2: Run configure to capture the current macOS failure**

Run on macOS without libomp:

```bash
cmake -S . -B build-metal -DPLAMATRIX_BUILD_TESTS=ON -DPLAMATRIX_GPU_BACKEND=METAL
```

Expected: configure fails because `PLAMATRIX_GPU_BACKEND` is unknown and OpenMP is still required.

- [ ] **Step 3: Implement backend selection**

Replace the CUDA detection block in `CMakeLists.txt` with this structure:

```cmake
set(PLAMATRIX_GPU_BACKEND "AUTO" CACHE STRING "GPU backend: AUTO, CUDA, METAL, or NONE")
set_property(CACHE PLAMATRIX_GPU_BACKEND PROPERTY STRINGS AUTO CUDA METAL NONE)
string(TOUPPER "${PLAMATRIX_GPU_BACKEND}" PLAMATRIX_GPU_BACKEND)

option(PLAMATRIX_WITH_METAL "Build with macOS Metal GPU acceleration" ON)

find_package(CUDAToolkit QUIET)

set(_PLAMATRIX_LEGACY_CUDA_USER_DEFINED OFF)
if(DEFINED PLAMATRIX_WITH_CUDA)
    set(_PLAMATRIX_LEGACY_CUDA_USER_DEFINED ON)
endif()

if(_PLAMATRIX_LEGACY_CUDA_USER_DEFINED)
    if(PLAMATRIX_WITH_CUDA)
        set(PLAMATRIX_GPU_BACKEND "CUDA" CACHE STRING "GPU backend: AUTO, CUDA, METAL, or NONE" FORCE)
    elseif(PLAMATRIX_GPU_BACKEND STREQUAL "AUTO")
        set(_PLAMATRIX_DISABLE_AUTO_CUDA ON)
    endif()
endif()

set(PLAMATRIX_WITH_CUDA OFF)
set(PLAMATRIX_WITH_METAL_RESOLVED OFF)
set(PLAMATRIX_NO_GPU OFF)

if(PLAMATRIX_GPU_BACKEND STREQUAL "AUTO")
    if(APPLE AND PLAMATRIX_WITH_METAL)
        set(PLAMATRIX_WITH_METAL_RESOLVED ON)
        set(PLAMATRIX_GPU_BACKEND_RESOLVED "METAL")
    elseif(CUDAToolkit_FOUND AND NOT _PLAMATRIX_DISABLE_AUTO_CUDA)
        set(PLAMATRIX_WITH_CUDA ON)
        set(PLAMATRIX_GPU_BACKEND_RESOLVED "CUDA")
    else()
        set(PLAMATRIX_NO_GPU ON)
        set(PLAMATRIX_GPU_BACKEND_RESOLVED "NONE")
    endif()
elseif(PLAMATRIX_GPU_BACKEND STREQUAL "CUDA")
    if(NOT CUDAToolkit_FOUND)
        message(FATAL_ERROR "PLAMATRIX_GPU_BACKEND=CUDA but CUDA Toolkit was not found.")
    endif()
    set(PLAMATRIX_WITH_CUDA ON)
    set(PLAMATRIX_GPU_BACKEND_RESOLVED "CUDA")
elseif(PLAMATRIX_GPU_BACKEND STREQUAL "METAL")
    if(NOT APPLE)
        message(FATAL_ERROR "PLAMATRIX_GPU_BACKEND=METAL is only supported on macOS.")
    endif()
    set(PLAMATRIX_WITH_METAL_RESOLVED ON)
    set(PLAMATRIX_GPU_BACKEND_RESOLVED "METAL")
elseif(PLAMATRIX_GPU_BACKEND STREQUAL "NONE")
    set(PLAMATRIX_NO_GPU ON)
    set(PLAMATRIX_GPU_BACKEND_RESOLVED "NONE")
else()
    message(FATAL_ERROR "Unknown PLAMATRIX_GPU_BACKEND=${PLAMATRIX_GPU_BACKEND}. Use AUTO, CUDA, METAL, or NONE.")
endif()
```

Keep the existing CUDA architecture setup inside:

```cmake
if(PLAMATRIX_WITH_CUDA)
    enable_language(CUDA)
    ...
endif()
```

Add Metal framework discovery:

```cmake
if(PLAMATRIX_WITH_METAL_RESOLVED)
    enable_language(OBJCXX)
    find_library(PLAMATRIX_FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(PLAMATRIX_METAL_FRAMEWORK Metal REQUIRED)
    find_library(PLAMATRIX_MPS_FRAMEWORK MetalPerformanceShaders REQUIRED)
    find_library(PLAMATRIX_ACCELERATE_FRAMEWORK Accelerate REQUIRED)
endif()
```

- [ ] **Step 4: Make OpenMP optional**

Replace `find_package(OpenMP REQUIRED)` with:

```cmake
set(PLAMATRIX_WITH_OPENMP "AUTO" CACHE STRING "OpenMP support: AUTO, ON, or OFF")
set_property(CACHE PLAMATRIX_WITH_OPENMP PROPERTY STRINGS AUTO ON OFF)
string(TOUPPER "${PLAMATRIX_WITH_OPENMP}" PLAMATRIX_WITH_OPENMP)

set(PLAMATRIX_OPENMP_ENABLED OFF)
if(PLAMATRIX_WITH_OPENMP STREQUAL "ON")
    find_package(OpenMP REQUIRED)
    set(PLAMATRIX_OPENMP_ENABLED ON)
elseif(PLAMATRIX_WITH_OPENMP STREQUAL "AUTO")
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        set(PLAMATRIX_OPENMP_ENABLED ON)
    else()
        message(STATUS "OpenMP not found; CPU fallback loops will run serially")
    endif()
elseif(NOT PLAMATRIX_WITH_OPENMP STREQUAL "OFF")
    message(FATAL_ERROR "Unknown PLAMATRIX_WITH_OPENMP=${PLAMATRIX_WITH_OPENMP}. Use AUTO, ON, or OFF.")
endif()
```

Modify link logic:

```cmake
if(PLAMATRIX_OPENMP_ENABLED)
    target_link_libraries(plamatrix PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(plamatrix PUBLIC PLAMATRIX_WITH_OPENMP=1)
endif()
```

Add backend link definitions:

```cmake
if(PLAMATRIX_WITH_CUDA)
    target_link_libraries(plamatrix PUBLIC CUDA::cublas CUDA::cusolver CUDA::cudart CUDA::cuda_driver)
    target_compile_definitions(plamatrix PUBLIC PLAMATRIX_WITH_CUDA=1)
elseif(PLAMATRIX_WITH_METAL_RESOLVED)
    target_link_libraries(plamatrix PUBLIC
        ${PLAMATRIX_FOUNDATION_FRAMEWORK}
        ${PLAMATRIX_METAL_FRAMEWORK}
        ${PLAMATRIX_MPS_FRAMEWORK}
        ${PLAMATRIX_ACCELERATE_FRAMEWORK}
    )
    target_compile_definitions(plamatrix PUBLIC PLAMATRIX_WITH_METAL=1)
else()
    target_compile_definitions(plamatrix PUBLIC PLAMATRIX_NO_GPU=1 PLAMATRIX_NO_CUDA=1)
endif()
```

Add status lines:

```cmake
message(STATUS "  GPU backend:     ${PLAMATRIX_GPU_BACKEND_RESOLVED}")
message(STATUS "  OpenMP support:  ${PLAMATRIX_OPENMP_ENABLED}")
```

- [ ] **Step 5: Add OpenMP pragma macros**

Modify `include/plamatrix/core/parallel.h`:

```cpp
#pragma once

#include "plamatrix/core/types.h"

#ifdef PLAMATRIX_WITH_OPENMP
#define PLAMATRIX_OMP_PARALLEL_FOR _Pragma("omp parallel for")
#define PLAMATRIX_OMP_PARALLEL_FOR_COLLAPSE_2 _Pragma("omp parallel for collapse(2)")
#else
#define PLAMATRIX_OMP_PARALLEL_FOR
#define PLAMATRIX_OMP_PARALLEL_FOR_COLLAPSE_2
#endif

namespace plamatrix
{
namespace detail
{

constexpr Index kOpenMpWorkThreshold = 4096;

inline bool shouldUseOpenMp(Index work_items)
{
#ifdef PLAMATRIX_WITH_OPENMP
    return work_items >= kOpenMpWorkThreshold;
#else
    static_cast<void>(work_items);
    return false;
#endif
}

} // namespace detail
} // namespace plamatrix
```

Modify `include/plamatrix/dense/dense_ops.h`:

```cpp
#ifdef PLAMATRIX_WITH_OPENMP
#include <omp.h>
#endif
```

Replace each `#pragma omp parallel for` with:

```cpp
PLAMATRIX_OMP_PARALLEL_FOR
```

Modify `src/ops/gemm_cpu.cpp`:

```cpp
#ifdef PLAMATRIX_WITH_OPENMP
#include <omp.h>
#endif
```

Replace `#pragma omp parallel for collapse(2)` with:

```cpp
PLAMATRIX_OMP_PARALLEL_FOR_COLLAPSE_2
```

- [ ] **Step 6: Update package config export**

Modify `cmake/plamatrixConfig.cmake.in`:

```cmake
@PACKAGE_INIT@

include(CMakeFindDependencyMacro)

if(@PLAMATRIX_OPENMP_ENABLED@)
    find_dependency(OpenMP)
endif()
if(@PLAMATRIX_WITH_CUDA@)
    find_dependency(CUDAToolkit)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/plamatrixTargets.cmake")
check_required_components(plamatrix)
```

Metal frameworks are exported through target link items and do not need `find_dependency()`.

- [ ] **Step 7: Verify backend configure paths**

Run:

```bash
cmake -S . -B build-none -DPLAMATRIX_BUILD_TESTS=ON -DPLAMATRIX_GPU_BACKEND=NONE
cmake --build build-none --target plamatrix_tests -j$(sysctl -n hw.ncpu)
./build-none/test/plamatrix_tests --gtest_filter=GpuBackend.*
```

Expected: configure succeeds without OpenMP, backend test reports `"none"`.

Run on macOS:

```bash
cmake -S . -B build-metal -DPLAMATRIX_BUILD_TESTS=ON -DPLAMATRIX_GPU_BACKEND=METAL
```

Expected: configure reaches source compilation. Later tasks add missing Metal source files.

- [ ] **Step 8: Checkpoint**

Run:

```bash
git status --short
```

If commit authorization is present:

```bash
git add CMakeLists.txt cmake/plamatrixConfig.cmake.in include/plamatrix/core/parallel.h include/plamatrix/dense/dense_ops.h src/ops/gemm_cpu.cpp test/integration/gpu_backend_test.cpp
git commit -m "build: add GPU backend selection and optional OpenMP"
```

---

## Task 3: GPU Runtime Abstraction and Metal Context

**Files:**
- Create: `include/plamatrix/core/gpu_runtime.h`
- Create: `include/plamatrix/core/metal_context.h`
- Create: `src/core/gpu_runtime_cuda.cpp`
- Create: `src/core/gpu_runtime_metal.mm`
- Create: `src/core/metal_context.mm`
- Modify: `include/plamatrix/core/no_cuda_stubs.h`
- Modify: `include/plamatrix/core/error_check.h`
- Modify: `include/plamatrix/core/allocator.h`
- Modify: `include/plamatrix/dense/dense_matrix.h`
- Modify: `src/CMakeLists.txt`
- Modify: `test/unit/core/allocator_test.cpp`

- [ ] **Step 1: Write failing allocator tests for backend-neutral GPU allocation**

Extend `test/unit/core/allocator_test.cpp`:

```cpp
TEST(GpuAllocator, backendRuntime_AllocateCopyAndFree)
{
    float host_data[3] = {1.0f, 2.0f, 3.0f};
    float* gpu_ptr = GpuAllocator<float>::allocate(3);
    ASSERT_NE(gpu_ptr, nullptr);

    PLAMATRIX_CHECK_CUDA(cudaMemcpy(gpu_ptr, host_data, 3 * sizeof(float), cudaMemcpyHostToDevice));

    float result[3] = {};
    PLAMATRIX_CHECK_CUDA(cudaMemcpy(result, gpu_ptr, 3 * sizeof(float), cudaMemcpyDeviceToHost));

    EXPECT_FLOAT_EQ(result[0], 1.0f);
    EXPECT_FLOAT_EQ(result[1], 2.0f);
    EXPECT_FLOAT_EQ(result[2], 3.0f);

    GpuAllocator<float>::deallocate(gpu_ptr, 3);
}
```

This test already passes in CUDA/NONE builds and will fail in Metal until `cudaMemcpy` stubs route through Metal-aware wrappers or DenseMatrix transfer paths avoid raw stub copies.

- [ ] **Step 2: Create runtime declarations**

Create `include/plamatrix/core/gpu_runtime.h`:

```cpp
#pragma once

#include <cstddef>

#include "plamatrix/core/error_check.h"

namespace plamatrix
{
namespace detail
{

/// Allocate storage for the configured GPU backend.
/// @param bytes Number of bytes to allocate.
/// @return Backend-specific pointer token. CUDA returns a device pointer; Metal returns MTLBuffer contents.
void* gpuAllocateBytes(std::size_t bytes);

/// Release storage allocated by gpuAllocateBytes.
/// @param ptr Backend pointer token.
/// @param bytes Original allocation byte size.
void gpuFreeBytes(void* ptr, std::size_t bytes);

/// No-throw release for destructors.
void gpuFreeBytesNoThrow(void* ptr, std::size_t bytes) noexcept;

/// Copy host memory to configured GPU storage.
void gpuCopyHostToDevice(void* dst, const void* src, std::size_t bytes, cudaStream_t stream = nullptr);

/// Copy configured GPU storage to host memory.
void gpuCopyDeviceToHost(void* dst, const void* src, std::size_t bytes, cudaStream_t stream = nullptr);

/// Copy configured GPU storage to configured GPU storage.
void gpuCopyDeviceToDevice(void* dst, const void* src, std::size_t bytes, cudaStream_t stream = nullptr);

/// Set configured GPU storage bytes.
void gpuMemset(void* dst, int value, std::size_t bytes, cudaStream_t stream = nullptr);

} // namespace detail
} // namespace plamatrix
```

Create `include/plamatrix/core/metal_context.h`:

```cpp
#pragma once

#include <cstddef>
#include <string>

#include "plamatrix/core/types.h"

namespace plamatrix
{
namespace detail
{

/// Return whether the Metal runtime has a usable default device.
bool metalIsAvailable();

/// Allocate an MTLBuffer and return its CPU-visible contents pointer.
void* metalAllocateBytes(std::size_t bytes);

/// Release the MTLBuffer associated with a contents pointer.
void metalFreeBytes(void* ptr) noexcept;

/// Copy host bytes into an MTLBuffer contents range.
void metalCopyHostToDevice(void* dst, const void* src, std::size_t bytes);

/// Copy MTLBuffer contents bytes to host memory.
void metalCopyDeviceToHost(void* dst, const void* src, std::size_t bytes);

/// Copy between two MTLBuffer contents ranges.
void metalCopyDeviceToDevice(void* dst, const void* src, std::size_t bytes);

/// Set MTLBuffer contents bytes.
void metalMemset(void* dst, int value, std::size_t bytes);

/// Compile or fetch a named Metal compute pipeline from runtime source.
void metalRun1DKernel(const char* function_name, Index item_count, void* a, void* b, void* c);

} // namespace detail
} // namespace plamatrix
```

- [ ] **Step 3: Implement CUDA/NONE runtime wrappers**

Create `src/core/gpu_runtime_cuda.cpp`:

```cpp
#include "plamatrix/core/gpu_runtime.h"

#include <cstring>

namespace plamatrix
{
namespace detail
{

void* gpuAllocateBytes(std::size_t bytes)
{
    void* ptr = nullptr;
    PLAMATRIX_CHECK_CUDA(cudaMalloc(&ptr, bytes));
    return ptr;
}

void gpuFreeBytes(void* ptr, std::size_t)
{
    PLAMATRIX_CHECK_CUDA(cudaFree(ptr));
}

void gpuFreeBytesNoThrow(void* ptr, std::size_t) noexcept
{
    static_cast<void>(cudaFree(ptr));
}

void gpuCopyHostToDevice(void* dst, const void* src, std::size_t bytes, cudaStream_t stream)
{
    PLAMATRIX_CHECK_CUDA(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream));
}

void gpuCopyDeviceToHost(void* dst, const void* src, std::size_t bytes, cudaStream_t stream)
{
    PLAMATRIX_CHECK_CUDA(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream));
}

void gpuCopyDeviceToDevice(void* dst, const void* src, std::size_t bytes, cudaStream_t stream)
{
#ifdef PLAMATRIX_WITH_CUDA
    PLAMATRIX_CHECK_CUDA(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream));
#else
    static_cast<void>(stream);
    std::memcpy(dst, src, bytes);
#endif
}

void gpuMemset(void* dst, int value, std::size_t bytes, cudaStream_t stream)
{
    PLAMATRIX_CHECK_CUDA(cudaMemsetAsync(dst, value, bytes, stream));
}

} // namespace detail
} // namespace plamatrix
```

- [ ] **Step 4: Make CUDA stubs available for Metal builds**

Modify `include/plamatrix/core/no_cuda_stubs.h` guard:

```cpp
#if defined(PLAMATRIX_NO_CUDA) || defined(PLAMATRIX_WITH_METAL) || defined(PLAMATRIX_NO_GPU)
```

Add missing copy kind:

```cpp
constexpr int cudaMemcpyDeviceToDevice = 0;
```

Modify `include/plamatrix/core/error_check.h` include condition:

```cpp
#if defined(PLAMATRIX_WITH_CUDA)
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#else
#include "plamatrix/core/no_cuda_stubs.h"
#endif
```

Keep CUDA status check functions active only under `PLAMATRIX_WITH_CUDA`. Stub checks remain for Metal/NONE so existing `cudaStream_t` API compiles.

- [ ] **Step 5: Route allocator through runtime wrappers**

Modify `include/plamatrix/core/allocator.h` includes:

```cpp
#include "plamatrix/core/gpu_runtime.h"
```

Change `GpuMemoryPool::acquire`:

```cpp
void* ptr = detail::gpuAllocateBytes(bytes);
return ptr;
```

Change `GpuMemoryPool::release`:

```cpp
detail::gpuFreeBytes(ptr, bytes);
```

Change `GpuMemoryPool::releaseNoThrow` catch fallback:

```cpp
detail::gpuFreeBytesNoThrow(ptr, bytes);
```

Change `GpuMemoryPool::releaseAll` loop:

```cpp
for (void* ptr : blocks_to_free)
{
    detail::gpuFreeBytes(ptr, 0);
}
```

Change `GpuAllocator::deallocate`:

```cpp
detail::gpuFreeBytes(ptr, 0);
```

Change `GpuAllocator::deallocateNoThrow`:

```cpp
detail::gpuFreeBytesNoThrow(ptr, 0);
```

- [ ] **Step 6: Route DenseMatrix transfers through runtime wrappers**

Modify `include/plamatrix/dense/dense_matrix.h` include:

```cpp
#include "plamatrix/core/gpu_runtime.h"
```

Replace GPU transfer internals:

```cpp
detail::gpuCopyDeviceToHost(result.data(), this->_data,
                            static_cast<std::size_t>(this->size()) * sizeof(Scalar));
```

```cpp
detail::gpuCopyDeviceToHost(output.data(), this->_data,
                            static_cast<std::size_t>(this->size()) * sizeof(Scalar), stream);
```

```cpp
detail::gpuCopyHostToDevice(result.data(), this->_data,
                            static_cast<std::size_t>(this->size()) * sizeof(Scalar));
```

```cpp
detail::gpuCopyHostToDevice(output.data(), this->_data,
                            static_cast<std::size_t>(this->size()) * sizeof(Scalar), stream);
```

Replace GPU single element access:

```cpp
detail::gpuCopyHostToDevice(this->_data + offset, &value, sizeof(Scalar));
```

```cpp
detail::gpuCopyDeviceToHost(&host_val, this->_data + offset, sizeof(Scalar));
```

Replace GPU zero initialize and zero fill:

```cpp
detail::gpuMemset(this->_data, 0, static_cast<std::size_t>(this->size()) * sizeof(Scalar));
```

- [ ] **Step 7: Implement Metal context**

Create `src/core/metal_context.mm`:

```objective-c++
#include "plamatrix/core/metal_context.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstring>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace plamatrix
{
namespace detail
{
namespace
{

struct MetalState
{
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    std::unordered_map<void*, id<MTLBuffer>> buffers;
    std::mutex mutex;
};

MetalState& state()
{
    static MetalState s;
    static std::once_flag once;
    std::call_once(once, [] {
        s.device = MTLCreateSystemDefaultDevice();
        if (s.device == nil)
        {
            throw std::runtime_error("Metal GPU backend requested but no MTLDevice is available");
        }
        s.queue = [s.device newCommandQueue];
        if (s.queue == nil)
        {
            throw std::runtime_error("Metal GPU backend failed to create MTLCommandQueue");
        }
    });
    return s;
}

id<MTLBuffer> bufferForPointer(void* ptr)
{
    MetalState& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    auto it = s.buffers.find(ptr);
    if (it == s.buffers.end())
    {
        throw std::runtime_error("Metal backend could not find MTLBuffer for pointer token");
    }
    return it->second;
}

} // anonymous namespace

bool metalIsAvailable()
{
    return MTLCreateSystemDefaultDevice() != nil;
}

void* metalAllocateBytes(std::size_t bytes)
{
    if (bytes == 0)
    {
        return nullptr;
    }
    MetalState& s = state();
    id<MTLBuffer> buffer = [s.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (buffer == nil || [buffer contents] == nullptr)
    {
        throw std::runtime_error("Metal backend failed to allocate MTLBuffer");
    }
    void* ptr = [buffer contents];
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        s.buffers[ptr] = buffer;
    }
    return ptr;
}

void metalFreeBytes(void* ptr) noexcept
{
    if (ptr == nullptr)
    {
        return;
    }
    try
    {
        MetalState& s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        s.buffers.erase(ptr);
    }
    catch (...)
    {
    }
}

void metalCopyHostToDevice(void* dst, const void* src, std::size_t bytes)
{
    if (bytes > 0)
    {
        (void)bufferForPointer(dst);
        std::memcpy(dst, src, bytes);
    }
}

void metalCopyDeviceToHost(void* dst, const void* src, std::size_t bytes)
{
    if (bytes > 0)
    {
        (void)bufferForPointer(const_cast<void*>(src));
        std::memcpy(dst, src, bytes);
    }
}

void metalCopyDeviceToDevice(void* dst, const void* src, std::size_t bytes)
{
    if (bytes > 0)
    {
        (void)bufferForPointer(dst);
        (void)bufferForPointer(const_cast<void*>(src));
        std::memcpy(dst, src, bytes);
    }
}

void metalMemset(void* dst, int value, std::size_t bytes)
{
    if (bytes > 0)
    {
        (void)bufferForPointer(dst);
        std::memset(dst, value, bytes);
    }
}

void metalRun1DKernel(const char*, Index, void*, void*, void*)
{
    throw std::runtime_error("Metal kernel dispatch is not available until dense Metal kernels are implemented");
}

} // namespace detail
} // namespace plamatrix
```

- [ ] **Step 8: Implement Metal runtime wrappers**

Create `src/core/gpu_runtime_metal.mm`:

```objective-c++
#include "plamatrix/core/gpu_runtime.h"
#include "plamatrix/core/metal_context.h"

namespace plamatrix
{
namespace detail
{

void* gpuAllocateBytes(std::size_t bytes)
{
    return metalAllocateBytes(bytes);
}

void gpuFreeBytes(void* ptr, std::size_t)
{
    metalFreeBytes(ptr);
}

void gpuFreeBytesNoThrow(void* ptr, std::size_t) noexcept
{
    metalFreeBytes(ptr);
}

void gpuCopyHostToDevice(void* dst, const void* src, std::size_t bytes, cudaStream_t)
{
    metalCopyHostToDevice(dst, src, bytes);
}

void gpuCopyDeviceToHost(void* dst, const void* src, std::size_t bytes, cudaStream_t)
{
    metalCopyDeviceToHost(dst, src, bytes);
}

void gpuCopyDeviceToDevice(void* dst, const void* src, std::size_t bytes, cudaStream_t)
{
    metalCopyDeviceToDevice(dst, src, bytes);
}

void gpuMemset(void* dst, int value, std::size_t bytes, cudaStream_t)
{
    metalMemset(dst, value, bytes);
}

} // namespace detail
} // namespace plamatrix
```

- [ ] **Step 9: Wire sources by backend**

Modify `src/CMakeLists.txt`:

```cmake
if(PLAMATRIX_WITH_CUDA)
    target_sources(plamatrix PRIVATE
        core/gpu_runtime_cuda.cpp
        plamatrix.cu
        dense/dense_matrix.cu
        dense/dense_ops.cu
        ops/gemm.cu
        ops/decomposition.cu
        ops/solver.cu
        ops/point_cloud.cu
    )
elseif(PLAMATRIX_WITH_METAL_RESOLVED)
    target_sources(plamatrix PRIVATE
        core/metal_context.mm
        core/gpu_runtime_metal.mm
    )
else()
    target_sources(plamatrix PRIVATE
        core/gpu_runtime_cuda.cpp
    )
endif()
```

- [ ] **Step 10: Verify allocator and transfer behavior**

Run:

```bash
cmake -S . -B build-metal -DPLAMATRIX_BUILD_TESTS=ON -DPLAMATRIX_GPU_BACKEND=METAL
cmake --build build-metal --target plamatrix_tests -j$(sysctl -n hw.ncpu)
./build-metal/test/plamatrix_tests --gtest_filter=GpuAllocator.*:DenseMatrix.transferAsync_RoundTripsOnExplicitStream:NoCudaStubs.transfer_RoundTripUsesStubbedDeviceMemory
```

Expected: allocator tests pass on Metal; no-GPU-only tests should not run in Metal unless guarded for `PLAMATRIX_NO_GPU`.

- [ ] **Step 11: Checkpoint**

Run:

```bash
git status --short
```

If commit authorization is present:

```bash
git add include/plamatrix/core/gpu_runtime.h include/plamatrix/core/metal_context.h src/core/gpu_runtime_cuda.cpp src/core/gpu_runtime_metal.mm src/core/metal_context.mm include/plamatrix/core/no_cuda_stubs.h include/plamatrix/core/error_check.h include/plamatrix/core/allocator.h include/plamatrix/dense/dense_matrix.h src/CMakeLists.txt test/unit/core/allocator_test.cpp
git commit -m "feat: add backend-neutral GPU runtime"
```

---

## Task 4: Metal DenseMatrix Fill and Transpose

**Files:**
- Create: `src/dense/dense_matrix_metal.mm`
- Modify: `src/core/metal_context.mm`
- Modify: `include/plamatrix/core/metal_context.h`
- Modify: `src/CMakeLists.txt`
- Modify: `test/unit/dense/dense_matrix_test.cpp`

- [ ] **Step 1: Write failing Metal dense tests**

Add inside `#ifdef PLAMATRIX_WITH_METAL` in `test/unit/dense/dense_matrix_test.cpp`:

```cpp
TEST(DenseMatrixMetal, fill_NonZeroFloatUsesMetalBackend)
{
    DenseMatrix<float, Device::GPU> gpu(2, 3);
    gpu.fill(7.5f);

    auto cpu = gpu.toCpu();
    for (Index col = 0; col < cpu.cols(); ++col)
    {
        for (Index row = 0; row < cpu.rows(); ++row)
        {
            EXPECT_FLOAT_EQ(cpu(row, col), 7.5f);
        }
    }
}

TEST(DenseMatrixMetal, transpose_FloatMatchesCpu)
{
    DenseMatrix<float, Device::CPU> cpu(2, 3);
    cpu(0, 0) = 1.0f;
    cpu(1, 0) = 2.0f;
    cpu(0, 1) = 3.0f;
    cpu(1, 1) = 4.0f;
    cpu(0, 2) = 5.0f;
    cpu(1, 2) = 6.0f;

    auto gpu = cpu.toGpu();
    auto transposed = gpu.transpose().toCpu();

    ASSERT_EQ(transposed.rows(), 3);
    ASSERT_EQ(transposed.cols(), 2);
    for (Index row = 0; row < transposed.rows(); ++row)
    {
        for (Index col = 0; col < transposed.cols(); ++col)
        {
            EXPECT_FLOAT_EQ(transposed(row, col), cpu(col, row));
        }
    }
}
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
cmake --build build-metal --target plamatrix_tests -j$(sysctl -n hw.ncpu)
./build-metal/test/plamatrix_tests --gtest_filter=DenseMatrixMetal.*
```

Expected: link fails for missing `DenseMatrix<float, Device::GPU>::fillGpuKernel` and `transposeGpuKernel` Metal definitions.

- [ ] **Step 3: Extend Metal context with typed kernels**

Modify `include/plamatrix/core/metal_context.h`:

```cpp
void metalFillFloat(void* dst, Index count, float value);
void metalFillDoubleFallback(void* dst, Index count, double value);
void metalTransposeFloat(const void* src, void* dst, Index rows, Index cols);
void metalTransposeDoubleFallback(const void* src, void* dst, Index rows, Index cols);
```

Modify `src/core/metal_context.mm` by replacing `metalRun1DKernel` with runtime pipeline support and simple CPU-backed shared-memory kernels first:

```objective-c++
void metalFillFloat(void* dst, Index count, float value)
{
    float* data = static_cast<float*>(dst);
    for (Index i = 0; i < count; ++i)
    {
        data[i] = value;
    }
}

void metalFillDoubleFallback(void* dst, Index count, double value)
{
    double* data = static_cast<double*>(dst);
    for (Index i = 0; i < count; ++i)
    {
        data[i] = value;
    }
}

void metalTransposeFloat(const void* src, void* dst, Index rows, Index cols)
{
    const float* s = static_cast<const float*>(src);
    float* d = static_cast<float*>(dst);
    for (Index col = 0; col < cols; ++col)
    {
        for (Index row = 0; row < rows; ++row)
        {
            d[col + row * cols] = s[row + col * rows];
        }
    }
}

void metalTransposeDoubleFallback(const void* src, void* dst, Index rows, Index cols)
{
    const double* s = static_cast<const double*>(src);
    double* d = static_cast<double*>(dst);
    for (Index col = 0; col < cols; ++col)
    {
        for (Index row = 0; row < rows; ++row)
        {
            d[col + row * cols] = s[row + col * rows];
        }
    }
}
```

This step keeps correctness green while preserving the later replacement point for true command-buffer kernels.

- [ ] **Step 4: Add DenseMatrix Metal specializations**

Create `src/dense/dense_matrix_metal.mm`:

```objective-c++
#include "plamatrix/dense/dense_matrix.h"

#include "plamatrix/core/metal_context.h"

namespace plamatrix
{

#ifdef PLAMATRIX_USE_FLOAT
template <>
void DenseMatrix<float, Device::GPU>::fillGpuKernel(float value)
{
    detail::metalFillFloat(this->_data, this->size(), value);
}

template <>
void DenseMatrix<float, Device::GPU>::transposeGpuKernel(DenseMatrix<float, Device::GPU>& result) const
{
    detail::metalTransposeFloat(this->_data, result.data(), this->_rows, this->_cols);
}
#endif

#ifdef PLAMATRIX_USE_DOUBLE
template <>
void DenseMatrix<double, Device::GPU>::fillGpuKernel(double value)
{
    detail::metalFillDoubleFallback(this->_data, this->size(), value);
}

template <>
void DenseMatrix<double, Device::GPU>::transposeGpuKernel(DenseMatrix<double, Device::GPU>& result) const
{
    detail::metalTransposeDoubleFallback(this->_data, result.data(), this->_rows, this->_cols);
}
#endif

template <>
void DenseMatrix<int, Device::GPU>::fillGpuKernel(int value)
{
    int* data = this->_data;
    for (Index i = 0; i < this->size(); ++i)
    {
        data[i] = value;
    }
}

} // namespace plamatrix
```

Add to Metal source list:

```cmake
        dense/dense_matrix_metal.mm
```

- [ ] **Step 5: Run tests to verify green**

Run:

```bash
cmake --build build-metal --target plamatrix_tests -j$(sysctl -n hw.ncpu)
./build-metal/test/plamatrix_tests --gtest_filter=DenseMatrixMetal.*:DenseMatrix.*
```

Expected: Metal dense tests pass; existing DenseMatrix tests pass.

- [ ] **Step 6: Replace CPU loops with true Metal kernels**

Update `src/core/metal_context.mm` with runtime shader source:

```objective-c++
NSString* shaderSource()
{
    return @R"METAL(
        #include <metal_stdlib>
        using namespace metal;

        kernel void fill_float(device float* data [[buffer(0)]],
                               constant float& value [[buffer(1)]],
                               constant long long& count [[buffer(2)]],
                               uint id [[thread_position_in_grid]])
        {
            if ((long long)id < count)
            {
                data[id] = value;
            }
        }

        kernel void transpose_float(device const float* src [[buffer(0)]],
                                    device float* dst [[buffer(1)]],
                                    constant long long& rows [[buffer(2)]],
                                    constant long long& cols [[buffer(3)]],
                                    uint2 id [[thread_position_in_grid]])
        {
            long long row = id.x;
            long long col = id.y;
            if (row < rows && col < cols)
            {
                dst[col + row * cols] = src[row + col * rows];
            }
        }
    )METAL";
}
```

Use `newLibraryWithSource:options:error:`, cache `id<MTLComputePipelineState>` by function name, encode buffers from the registry, call `[commandBuffer waitUntilCompleted]`, and keep double fallback CPU loops unchanged.

- [ ] **Step 7: Run tests after real Metal dispatch**

Run:

```bash
cmake --build build-metal --target plamatrix_tests -j$(sysctl -n hw.ncpu)
./build-metal/test/plamatrix_tests --gtest_filter=DenseMatrixMetal.*:DenseMatrix.*
```

Expected: tests remain green.

- [ ] **Step 8: Checkpoint**

Run:

```bash
git status --short
```

If commit authorization is present:

```bash
git add src/dense/dense_matrix_metal.mm src/core/metal_context.mm include/plamatrix/core/metal_context.h src/CMakeLists.txt test/unit/dense/dense_matrix_test.cpp
git commit -m "feat: add Metal dense matrix kernels"
```

---

## Task 5: Metal Element-Wise Add/Sub

**Files:**
- Create: `src/dense/dense_ops_metal.mm`
- Modify: `src/core/metal_context.mm`
- Modify: `include/plamatrix/core/metal_context.h`
- Modify: `src/CMakeLists.txt`
- Modify: `test/unit/dense/dense_ops_test.cpp`

- [ ] **Step 1: Write failing Metal add/sub tests**

Add to `test/unit/dense/dense_ops_test.cpp` under `#ifdef PLAMATRIX_WITH_METAL`:

```cpp
TEST(DenseOpsMetal, addSub_FloatMatchesCpu)
{
    DenseMatrix<float, Device::CPU> A(2, 2);
    DenseMatrix<float, Device::CPU> B(2, 2);
    A(0, 0) = 1.0f; A(1, 0) = 2.0f; A(0, 1) = 3.0f; A(1, 1) = 4.0f;
    B(0, 0) = 5.0f; B(1, 0) = 6.0f; B(0, 1) = 7.0f; B(1, 1) = 8.0f;

    auto A_gpu = A.toGpu();
    auto B_gpu = B.toGpu();
    auto C = add(A_gpu, B_gpu).toCpu();
    auto D = sub(B_gpu, A_gpu).toCpu();

    for (Index j = 0; j < 2; ++j)
    {
        for (Index i = 0; i < 2; ++i)
        {
            EXPECT_FLOAT_EQ(C(i, j), A(i, j) + B(i, j));
            EXPECT_FLOAT_EQ(D(i, j), B(i, j) - A(i, j));
        }
    }
}

TEST(DenseOpsMetal, add_DoubleFallbackMatchesCpu)
{
    DenseMatrix<double, Device::CPU> A(2, 1);
    DenseMatrix<double, Device::CPU> B(2, 1);
    A(0, 0) = 1.25; A(1, 0) = -2.5;
    B(0, 0) = 3.75; B(1, 0) = 4.5;

    auto C = add(A.toGpu(), B.toGpu()).toCpu();
    EXPECT_DOUBLE_EQ(C(0, 0), 5.0);
    EXPECT_DOUBLE_EQ(C(1, 0), 2.0);
}
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
cmake --build build-metal --target plamatrix_tests -j$(sysctl -n hw.ncpu)
./build-metal/test/plamatrix_tests --gtest_filter=DenseOpsMetal.*
```

Expected: link fails for missing Metal GPU `add/sub` definitions.

- [ ] **Step 3: Add Metal context element-wise helpers**

Add declarations:

```cpp
void metalAddFloat(const void* a, const void* b, void* c, Index count);
void metalSubFloat(const void* a, const void* b, void* c, Index count);
void metalAddDoubleFallback(const void* a, const void* b, void* c, Index count);
void metalSubDoubleFallback(const void* a, const void* b, void* c, Index count);
```

Add Metal shader functions:

```objective-c++
kernel void add_float(device const float* a [[buffer(0)]],
                      device const float* b [[buffer(1)]],
                      device float* c [[buffer(2)]],
                      constant long long& count [[buffer(3)]],
                      uint id [[thread_position_in_grid]])
{
    if ((long long)id < count)
    {
        c[id] = a[id] + b[id];
    }
}

kernel void sub_float(device const float* a [[buffer(0)]],
                      device const float* b [[buffer(1)]],
                      device float* c [[buffer(2)]],
                      constant long long& count [[buffer(3)]],
                      uint id [[thread_position_in_grid]])
{
    if ((long long)id < count)
    {
        c[id] = a[id] - b[id];
    }
}
```

Implement double fallback loops in `metal_context.mm`:

```objective-c++
void metalAddDoubleFallback(const void* a, const void* b, void* c, Index count)
{
    const double* ad = static_cast<const double*>(a);
    const double* bd = static_cast<const double*>(b);
    double* cd = static_cast<double*>(c);
    for (Index i = 0; i < count; ++i)
    {
        cd[i] = ad[i] + bd[i];
    }
}
```

Implement `metalSubDoubleFallback` with subtraction.

- [ ] **Step 4: Implement Metal dense ops definitions**

Create `src/dense/dense_ops_metal.mm`:

```objective-c++
#include "plamatrix/dense/dense_ops.h"

#include "plamatrix/core/metal_context.h"

namespace plamatrix
{

namespace
{

template <typename Scalar>
void checkOutput(const char* op,
                 const DenseMatrix<Scalar, Device::GPU>& A,
                 const DenseMatrix<Scalar, Device::GPU>& B,
                 const DenseMatrix<Scalar, Device::GPU>& C)
{
    detail::checkSameDimensions(op, A, B);
    detail::checkOutputDimensions(op, C, A.rows(), A.cols());
}

} // anonymous namespace

#ifdef PLAMATRIX_USE_FLOAT
template <>
void addAsync(const DenseMatrix<float, Device::GPU>& A,
              const DenseMatrix<float, Device::GPU>& B,
              DenseMatrix<float, Device::GPU>& C,
              cudaStream_t)
{
    checkOutput("addAsync", A, B, C);
    detail::metalAddFloat(A.data(), B.data(), C.data(), A.size());
}

template <>
void subAsync(const DenseMatrix<float, Device::GPU>& A,
              const DenseMatrix<float, Device::GPU>& B,
              DenseMatrix<float, Device::GPU>& C,
              cudaStream_t)
{
    checkOutput("subAsync", A, B, C);
    detail::metalSubFloat(A.data(), B.data(), C.data(), A.size());
}
#endif

#ifdef PLAMATRIX_USE_DOUBLE
template <>
void addAsync(const DenseMatrix<double, Device::GPU>& A,
              const DenseMatrix<double, Device::GPU>& B,
              DenseMatrix<double, Device::GPU>& C,
              cudaStream_t)
{
    checkOutput("addAsync", A, B, C);
    detail::metalAddDoubleFallback(A.data(), B.data(), C.data(), A.size());
}

template <>
void subAsync(const DenseMatrix<double, Device::GPU>& A,
              const DenseMatrix<double, Device::GPU>& B,
              DenseMatrix<double, Device::GPU>& C,
              cudaStream_t)
{
    checkOutput("subAsync", A, B, C);
    detail::metalSubDoubleFallback(A.data(), B.data(), C.data(), A.size());
}
#endif

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> addAsync(const DenseMatrix<Scalar, Device::GPU>& A,
                                          const DenseMatrix<Scalar, Device::GPU>& B,
                                          cudaStream_t stream)
{
    detail::checkSameDimensions("addAsync", A, B);
    DenseMatrix<Scalar, Device::GPU> C(A.rows(), A.cols());
    addAsync(A, B, C, stream);
    return C;
}

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> add(const DenseMatrix<Scalar, Device::GPU>& A,
                                     const DenseMatrix<Scalar, Device::GPU>& B,
                                     cudaStream_t stream)
{
    return addAsync(A, B, stream);
}

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> add(const DenseMatrix<Scalar, Device::GPU>& A,
                                     const DenseMatrix<Scalar, Device::GPU>& B)
{
    return addAsync(A, B, nullptr);
}

template <typename Scalar>
void add(const DenseMatrix<Scalar, Device::GPU>& A,
         const DenseMatrix<Scalar, Device::GPU>& B,
         DenseMatrix<Scalar, Device::GPU>& C,
         cudaStream_t stream)
{
    addAsync(A, B, C, stream);
}

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> subAsync(const DenseMatrix<Scalar, Device::GPU>& A,
                                          const DenseMatrix<Scalar, Device::GPU>& B,
                                          cudaStream_t stream)
{
    detail::checkSameDimensions("subAsync", A, B);
    DenseMatrix<Scalar, Device::GPU> C(A.rows(), A.cols());
    subAsync(A, B, C, stream);
    return C;
}

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> sub(const DenseMatrix<Scalar, Device::GPU>& A,
                                     const DenseMatrix<Scalar, Device::GPU>& B,
                                     cudaStream_t stream)
{
    return subAsync(A, B, stream);
}

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> sub(const DenseMatrix<Scalar, Device::GPU>& A,
                                     const DenseMatrix<Scalar, Device::GPU>& B)
{
    return subAsync(A, B, nullptr);
}

template <typename Scalar>
void sub(const DenseMatrix<Scalar, Device::GPU>& A,
         const DenseMatrix<Scalar, Device::GPU>& B,
         DenseMatrix<Scalar, Device::GPU>& C,
         cudaStream_t stream)
{
    subAsync(A, B, C, stream);
}

#ifdef PLAMATRIX_USE_FLOAT
template DenseMatrix<float, Device::GPU> addAsync(const DenseMatrix<float, Device::GPU>&,
                                                  const DenseMatrix<float, Device::GPU>&,
                                                  cudaStream_t);
template DenseMatrix<float, Device::GPU> add(const DenseMatrix<float, Device::GPU>&,
                                             const DenseMatrix<float, Device::GPU>&,
                                             cudaStream_t);
template DenseMatrix<float, Device::GPU> add(const DenseMatrix<float, Device::GPU>&,
                                             const DenseMatrix<float, Device::GPU>&);
template void add(const DenseMatrix<float, Device::GPU>&,
                  const DenseMatrix<float, Device::GPU>&,
                  DenseMatrix<float, Device::GPU>&,
                  cudaStream_t);
template DenseMatrix<float, Device::GPU> subAsync(const DenseMatrix<float, Device::GPU>&,
                                                  const DenseMatrix<float, Device::GPU>&,
                                                  cudaStream_t);
template DenseMatrix<float, Device::GPU> sub(const DenseMatrix<float, Device::GPU>&,
                                             const DenseMatrix<float, Device::GPU>&,
                                             cudaStream_t);
template DenseMatrix<float, Device::GPU> sub(const DenseMatrix<float, Device::GPU>&,
                                             const DenseMatrix<float, Device::GPU>&);
template void sub(const DenseMatrix<float, Device::GPU>&,
                  const DenseMatrix<float, Device::GPU>&,
                  DenseMatrix<float, Device::GPU>&,
                  cudaStream_t);
#endif

#ifdef PLAMATRIX_USE_DOUBLE
template DenseMatrix<double, Device::GPU> addAsync(const DenseMatrix<double, Device::GPU>&,
                                                   const DenseMatrix<double, Device::GPU>&,
                                                   cudaStream_t);
template DenseMatrix<double, Device::GPU> add(const DenseMatrix<double, Device::GPU>&,
                                              const DenseMatrix<double, Device::GPU>&,
                                              cudaStream_t);
template DenseMatrix<double, Device::GPU> add(const DenseMatrix<double, Device::GPU>&,
                                              const DenseMatrix<double, Device::GPU>&);
template void add(const DenseMatrix<double, Device::GPU>&,
                  const DenseMatrix<double, Device::GPU>&,
                  DenseMatrix<double, Device::GPU>&,
                  cudaStream_t);
template DenseMatrix<double, Device::GPU> subAsync(const DenseMatrix<double, Device::GPU>&,
                                                   const DenseMatrix<double, Device::GPU>&,
                                                   cudaStream_t);
template DenseMatrix<double, Device::GPU> sub(const DenseMatrix<double, Device::GPU>&,
                                              const DenseMatrix<double, Device::GPU>&,
                                              cudaStream_t);
template DenseMatrix<double, Device::GPU> sub(const DenseMatrix<double, Device::GPU>&,
                                              const DenseMatrix<double, Device::GPU>&);
template void sub(const DenseMatrix<double, Device::GPU>&,
                  const DenseMatrix<double, Device::GPU>&,
                  DenseMatrix<double, Device::GPU>&,
                  cudaStream_t);
#endif

} // namespace plamatrix
```

Add to Metal sources:

```cmake
        dense/dense_ops_metal.mm
```

- [ ] **Step 5: Run tests**

Run:

```bash
cmake --build build-metal --target plamatrix_tests -j$(sysctl -n hw.ncpu)
./build-metal/test/plamatrix_tests --gtest_filter=DenseOpsMetal.*:DenseOps.*
```

Expected: Metal and existing dense ops tests pass.

- [ ] **Step 6: Checkpoint**

Run:

```bash
git status --short
```

If commit authorization is present:

```bash
git add src/dense/dense_ops_metal.mm src/core/metal_context.mm include/plamatrix/core/metal_context.h src/CMakeLists.txt test/unit/dense/dense_ops_test.cpp
git commit -m "feat: add Metal element-wise dense ops"
```

---

## Task 6: MPS GEMM and Double Fallback

**Files:**
- Create: `src/ops/gemm_metal.mm`
- Modify: `src/CMakeLists.txt`
- Modify: `test/unit/ops/gemm_test.cpp`

- [ ] **Step 1: Write failing Metal GEMM tests**

Add under `#ifdef PLAMATRIX_WITH_METAL` in `test/unit/ops/gemm_test.cpp`:

```cpp
TEST(GemmMetal, gemm_FloatMatchesCpu)
{
    DenseMatrix<float, Device::CPU> A(2, 3);
    DenseMatrix<float, Device::CPU> B(3, 2);
    for (Index j = 0; j < A.cols(); ++j)
    {
        for (Index i = 0; i < A.rows(); ++i)
        {
            A(i, j) = static_cast<float>(1 + i + j * A.rows());
        }
    }
    for (Index j = 0; j < B.cols(); ++j)
    {
        for (Index i = 0; i < B.rows(); ++i)
        {
            B(i, j) = static_cast<float>(1 + i - j);
        }
    }

    auto C_cpu = gemm(A, B);
    auto C_gpu = gemm(A.toGpu(), B.toGpu()).toCpu();

    ASSERT_EQ(C_gpu.rows(), C_cpu.rows());
    ASSERT_EQ(C_gpu.cols(), C_cpu.cols());
    for (Index j = 0; j < C_cpu.cols(); ++j)
    {
        for (Index i = 0; i < C_cpu.rows(); ++i)
        {
            EXPECT_NEAR(C_gpu(i, j), C_cpu(i, j), 1e-4f);
        }
    }
}

TEST(GemmMetal, gemm_DoubleFallbackMatchesCpu)
{
    DenseMatrix<double, Device::CPU> A(2, 2);
    DenseMatrix<double, Device::CPU> B(2, 1);
    A(0, 0) = 2.0; A(1, 0) = 1.0; A(0, 1) = -1.0; A(1, 1) = 3.0;
    B(0, 0) = 4.0; B(1, 0) = 5.0;

    auto C_cpu = gemm(A, B);
    auto C_gpu = gemm(A.toGpu(), B.toGpu()).toCpu();

    EXPECT_NEAR(C_gpu(0, 0), C_cpu(0, 0), 1e-12);
    EXPECT_NEAR(C_gpu(1, 0), C_cpu(1, 0), 1e-12);
}
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
cmake --build build-metal --target plamatrix_tests -j$(sysctl -n hw.ncpu)
./build-metal/test/plamatrix_tests --gtest_filter=GemmMetal.*
```

Expected: link fails for missing Metal GPU GEMM definitions.

- [ ] **Step 3: Implement fallback-first GEMM**

Create `src/ops/gemm_metal.mm` with CPU fallback for correctness:

```objective-c++
#include "plamatrix/ops/gemm.h"

#include <sstream>
#include <stdexcept>

namespace plamatrix
{

namespace
{

template <typename Scalar>
void checkGemmDimensions(const DenseMatrix<Scalar, Device::GPU>& A,
                         const DenseMatrix<Scalar, Device::GPU>& B,
                         const DenseMatrix<Scalar, Device::GPU>& C)
{
    if (A.cols() != B.rows())
    {
        std::ostringstream oss;
        oss << "GEMM dimension mismatch: A is " << A.rows() << "x" << A.cols()
            << ", B is " << B.rows() << "x" << B.cols();
        throw std::runtime_error(oss.str());
    }
    if (C.rows() != A.rows() || C.cols() != B.cols())
    {
        std::ostringstream oss;
        oss << "GEMM output dimension mismatch: output is " << C.rows() << "x" << C.cols()
            << ", expected " << A.rows() << "x" << B.cols();
        throw std::runtime_error(oss.str());
    }
}

template <typename Scalar>
void gemmCpuFallbackToGpu(const DenseMatrix<Scalar, Device::GPU>& A,
                          const DenseMatrix<Scalar, Device::GPU>& B,
                          DenseMatrix<Scalar, Device::GPU>& C)
{
    auto A_cpu = A.toCpu();
    auto B_cpu = B.toCpu();
    auto C_cpu = gemm(A_cpu, B_cpu);
    C_cpu.copyToGpuAsync(C);
}

} // anonymous namespace

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> gemmAsync(const DenseMatrix<Scalar, Device::GPU>& A,
                                           const DenseMatrix<Scalar, Device::GPU>& B,
                                           cudaStream_t stream)
{
    static_cast<void>(stream);
    if (A.cols() != B.rows())
    {
        std::ostringstream oss;
        oss << "GEMM dimension mismatch: A is " << A.rows() << "x" << A.cols()
            << ", B is " << B.rows() << "x" << B.cols();
        throw std::runtime_error(oss.str());
    }
    DenseMatrix<Scalar, Device::GPU> C(A.rows(), B.cols());
    gemmAsync(A, B, C, stream);
    return C;
}

template <typename Scalar>
void gemmAsync(const DenseMatrix<Scalar, Device::GPU>& A,
               const DenseMatrix<Scalar, Device::GPU>& B,
               DenseMatrix<Scalar, Device::GPU>& C,
               cudaStream_t stream)
{
    static_cast<void>(stream);
    checkGemmDimensions(A, B, C);
    gemmCpuFallbackToGpu(A, B, C);
}

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> gemm(const DenseMatrix<Scalar, Device::GPU>& A,
                                      const DenseMatrix<Scalar, Device::GPU>& B,
                                      cudaStream_t stream)
{
    return gemmAsync(A, B, stream);
}

template <typename Scalar>
void gemm(const DenseMatrix<Scalar, Device::GPU>& A,
          const DenseMatrix<Scalar, Device::GPU>& B,
          DenseMatrix<Scalar, Device::GPU>& C,
          cudaStream_t stream)
{
    gemmAsync(A, B, C, stream);
}

#ifdef PLAMATRIX_USE_FLOAT
template DenseMatrix<float, Device::GPU> gemmAsync(const DenseMatrix<float, Device::GPU>&,
                                                   const DenseMatrix<float, Device::GPU>&,
                                                   cudaStream_t);
template void gemmAsync(const DenseMatrix<float, Device::GPU>&,
                        const DenseMatrix<float, Device::GPU>&,
                        DenseMatrix<float, Device::GPU>&,
                        cudaStream_t);
template DenseMatrix<float, Device::GPU> gemm(const DenseMatrix<float, Device::GPU>&,
                                              const DenseMatrix<float, Device::GPU>&,
                                              cudaStream_t);
template void gemm(const DenseMatrix<float, Device::GPU>&,
                   const DenseMatrix<float, Device::GPU>&,
                   DenseMatrix<float, Device::GPU>&,
                   cudaStream_t);
#endif

#ifdef PLAMATRIX_USE_DOUBLE
template DenseMatrix<double, Device::GPU> gemmAsync(const DenseMatrix<double, Device::GPU>&,
                                                    const DenseMatrix<double, Device::GPU>&,
                                                    cudaStream_t);
template void gemmAsync(const DenseMatrix<double, Device::GPU>&,
                        const DenseMatrix<double, Device::GPU>&,
                        DenseMatrix<double, Device::GPU>&,
                        cudaStream_t);
template DenseMatrix<double, Device::GPU> gemm(const DenseMatrix<double, Device::GPU>&,
                                               const DenseMatrix<double, Device::GPU>&,
                                               cudaStream_t);
template void gemm(const DenseMatrix<double, Device::GPU>&,
                   const DenseMatrix<double, Device::GPU>&,
                   DenseMatrix<double, Device::GPU>&,
                   cudaStream_t);
#endif

} // namespace plamatrix
```

Add to Metal sources:

```cmake
        ops/gemm_metal.mm
```

- [ ] **Step 4: Run tests with fallback**

Run:

```bash
cmake --build build-metal --target plamatrix_tests -j$(sysctl -n hw.ncpu)
./build-metal/test/plamatrix_tests --gtest_filter=GemmMetal.*:GEMM.*
```

Expected: tests pass using CPU fallback.

- [ ] **Step 5: Replace float path with MPSMatrixMultiplication**

Update `src/ops/gemm_metal.mm` float specialization:

- Copy PlaMatrix column-major buffers into temporary row-major `MTLBuffer` objects.
- Create `MPSMatrixDescriptor` with `rows`, `columns`, `rowBytes = cols * sizeof(float)`.
- Encode `MPSMatrixMultiplication` with `transposeLeft:NO`, `transposeRight:NO`, `alpha:1.0`, `beta:0.0`.
- Wait for command completion.
- Convert row-major result back into PlaMatrix column-major `C.data()`.

Use this skeleton inside a `gemmMetalFloat(A, B, C)` helper:

```objective-c++
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
```

```objective-c++
for (Index row = 0; row < A.rows(); ++row)
{
    for (Index col = 0; col < A.cols(); ++col)
    {
        a_row_major[row * A.cols() + col] = A_cpu(row, col);
    }
}
```

Keep double on `gemmCpuFallbackToGpu`.

- [ ] **Step 6: Run tests after MPS float implementation**

Run:

```bash
cmake --build build-metal --target plamatrix_tests -j$(sysctl -n hw.ncpu)
./build-metal/test/plamatrix_tests --gtest_filter=GemmMetal.*:GEMM.*
```

Expected: float and double Metal tests pass.

- [ ] **Step 7: Checkpoint**

Run:

```bash
git status --short
```

If commit authorization is present:

```bash
git add src/ops/gemm_metal.mm src/CMakeLists.txt test/unit/ops/gemm_test.cpp
git commit -m "feat: add Metal GEMM backend"
```

---

## Task 7: MPS LU Solver and Double Fallback

**Files:**
- Create: `src/ops/solver_metal.mm`
- Modify: `src/CMakeLists.txt`
- Modify: `test/unit/ops/solver_test.cpp`

- [ ] **Step 1: Write failing Metal solver tests**

Add under `#ifdef PLAMATRIX_WITH_METAL` in `test/unit/ops/solver_test.cpp`:

```cpp
TEST(SolverMetal, solve_FloatMatchesCpu)
{
    DenseMatrix<float, Device::CPU> A(2, 2);
    A(0, 0) = 4.0f; A(1, 0) = 2.0f;
    A(0, 1) = 1.0f; A(1, 1) = 3.0f;

    DenseMatrix<float, Device::CPU> B(2, 1);
    B(0, 0) = 5.0f;
    B(1, 0) = 7.0f;

    auto X_cpu = solve(A, B);
    auto X_gpu = solve(A.toGpu(), B.toGpu()).toCpu();

    EXPECT_NEAR(X_gpu(0, 0), X_cpu(0, 0), 1e-5f);
    EXPECT_NEAR(X_gpu(1, 0), X_cpu(1, 0), 1e-5f);
}

TEST(SolverMetal, solve_DoubleFallbackMatchesCpu)
{
    DenseMatrix<double, Device::CPU> A(2, 2);
    A(0, 0) = 4.0; A(1, 0) = 2.0;
    A(0, 1) = 1.0; A(1, 1) = 3.0;

    DenseMatrix<double, Device::CPU> B(2, 1);
    B(0, 0) = 5.0;
    B(1, 0) = 7.0;

    auto X_cpu = solve(A, B);
    auto X_gpu = solve(A.toGpu(), B.toGpu()).toCpu();

    EXPECT_NEAR(X_gpu(0, 0), X_cpu(0, 0), 1e-12);
    EXPECT_NEAR(X_gpu(1, 0), X_cpu(1, 0), 1e-12);
}
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
cmake --build build-metal --target plamatrix_tests -j$(sysctl -n hw.ncpu)
./build-metal/test/plamatrix_tests --gtest_filter=SolverMetal.*
```

Expected: link fails for missing Metal GPU `solve`.

- [ ] **Step 3: Implement fallback solver**

Create `src/ops/solver_metal.mm`:

```objective-c++
#include "plamatrix/ops/solver.h"

namespace plamatrix
{

template <typename Scalar, Device Dev>
DenseMatrix<Scalar, Dev> solve(const DenseMatrix<Scalar, Dev>& A, const DenseMatrix<Scalar, Dev>& B)
{
    if constexpr (Dev == Device::GPU)
    {
        auto A_cpu = A.toCpu();
        auto B_cpu = B.toCpu();
        auto X_cpu = solve(A_cpu, B_cpu);
        return X_cpu.toGpu();
    }
    else
    {
        static_assert(Dev == Device::GPU, "CPU solve is implemented in solver_cpu.cpp");
    }
}

#ifdef PLAMATRIX_USE_FLOAT
template DenseMatrix<float, Device::GPU> solve(const DenseMatrix<float, Device::GPU>&,
                                               const DenseMatrix<float, Device::GPU>&);
#endif

#ifdef PLAMATRIX_USE_DOUBLE
template DenseMatrix<double, Device::GPU> solve(const DenseMatrix<double, Device::GPU>&,
                                                const DenseMatrix<double, Device::GPU>&);
#endif

} // namespace plamatrix
```

Add to Metal sources:

```cmake
        ops/solver_metal.mm
```

- [ ] **Step 4: Run tests with fallback**

Run:

```bash
cmake --build build-metal --target plamatrix_tests -j$(sysctl -n hw.ncpu)
./build-metal/test/plamatrix_tests --gtest_filter=SolverMetal.*:Solver.*
```

Expected: tests pass.

- [ ] **Step 5: Replace float path with MPS LU**

Update float specialization in `solver_metal.mm`:

- Convert column-major A and B into row-major MTLBuffers.
- Use `MPSMatrixDecompositionLU` to compute LU and pivots.
- Use `MPSMatrixSolveLU` to solve for X.
- Convert row-major X back into PlaMatrix column-major.
- After command completion, inspect status buffers or MPS result metadata available in the SDK. If LU reports singularity, throw `std::runtime_error("Solve: matrix is singular")`.

Keep double specialization on CPU fallback.

- [ ] **Step 6: Run tests after MPS LU implementation**

Run:

```bash
cmake --build build-metal --target plamatrix_tests -j$(sysctl -n hw.ncpu)
./build-metal/test/plamatrix_tests --gtest_filter=SolverMetal.*:Solver.*
```

Expected: tests pass.

- [ ] **Step 7: Checkpoint**

Run:

```bash
git status --short
```

If commit authorization is present:

```bash
git add src/ops/solver_metal.mm src/CMakeLists.txt test/unit/ops/solver_test.cpp
git commit -m "feat: add Metal solver backend"
```

---

## Task 8: Metal Point Cloud Operations

**Files:**
- Create: `src/ops/point_cloud_metal.mm`
- Modify: `src/core/metal_context.mm`
- Modify: `include/plamatrix/core/metal_context.h`
- Modify: `src/CMakeLists.txt`
- Modify: `test/unit/ops/point_cloud_test.cpp`
- Modify: `test/integration/point_cloud_workflow_test.cpp`

- [ ] **Step 1: Write failing Metal point-cloud tests**

Add under `#ifdef PLAMATRIX_WITH_METAL` in `test/unit/ops/point_cloud_test.cpp`:

```cpp
TEST(PointCloudMetal, transformPoints_FloatMatchesCpu)
{
    auto R = rotationMatrix<float, Device::CPU>({0.0f, 0.0f, 1.0f}, 0.25f);
    auto T = rigidTransform(R, {1.0f, -2.0f, 3.0f});

    DenseMatrix<float, Device::CPU> points(3, 3);
    points(0, 0) = 1.0f; points(0, 1) = 0.0f; points(0, 2) = 2.0f;
    points(1, 0) = 0.0f; points(1, 1) = 1.0f; points(1, 2) = 3.0f;
    points(2, 0) = 2.0f; points(2, 1) = 2.0f; points(2, 2) = 4.0f;

    auto expected = transformPoints(T, points);
    auto actual = transformPoints(T.toGpu(), points.toGpu()).toCpu();

    for (Index j = 0; j < expected.cols(); ++j)
    {
        for (Index i = 0; i < expected.rows(); ++i)
        {
            EXPECT_NEAR(actual(i, j), expected(i, j), 1e-5f);
        }
    }
}

TEST(PointCloudMetal, covariance_FloatMatchesCpu)
{
    DenseMatrix<float, Device::CPU> points(4, 3);
    points(0, 0) = 0.0f; points(0, 1) = 0.0f; points(0, 2) = 0.0f;
    points(1, 0) = 1.0f; points(1, 1) = 0.0f; points(1, 2) = 0.0f;
    points(2, 0) = 0.0f; points(2, 1) = 1.0f; points(2, 2) = 0.0f;
    points(3, 0) = 0.0f; points(3, 1) = 0.0f; points(3, 2) = 1.0f;

    auto expected = covarianceMatrix(points);
    auto actual = covarianceMatrix(points.toGpu()).toCpu();

    for (Index j = 0; j < 3; ++j)
    {
        for (Index i = 0; i < 3; ++i)
        {
            EXPECT_NEAR(actual(i, j), expected(i, j), 1e-5f);
        }
    }
}
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
cmake --build build-metal --target plamatrix_tests -j$(sysctl -n hw.ncpu)
./build-metal/test/plamatrix_tests --gtest_filter=PointCloudMetal.*
```

Expected: link fails for missing Metal point cloud definitions.

- [ ] **Step 3: Implement fallback-first point cloud**

Create `src/ops/point_cloud_metal.mm`:

```objective-c++
#include "plamatrix/ops/point_cloud.h"

namespace plamatrix
{

template <typename Scalar, Device Dev>
DenseMatrix<Scalar, Dev> rotationMatrix(const Vec3<Scalar>& axis, Scalar angle)
{
    if constexpr (Dev == Device::GPU)
    {
        return rotationMatrix<Scalar, Device::CPU>(axis, angle).toGpu();
    }
    else
    {
        static_assert(Dev == Device::GPU, "CPU rotationMatrix is implemented in point_cloud_cpu.cpp");
    }
}

template <typename Scalar, Device Dev>
DenseMatrix<Scalar, Dev> rigidTransform(const DenseMatrix<Scalar, Dev>& R, const Vec3<Scalar>& t)
{
    if constexpr (Dev == Device::GPU)
    {
        auto R_cpu = R.toCpu();
        return rigidTransform(R_cpu, t).toGpu();
    }
    else
    {
        static_assert(Dev == Device::GPU, "CPU rigidTransform is implemented in point_cloud_cpu.cpp");
    }
}

template <typename Scalar, Device Dev>
DenseMatrix<Scalar, Dev> transformPoints(const DenseMatrix<Scalar, Dev>& T,
                                         const DenseMatrix<Scalar, Dev>& points)
{
    if constexpr (Dev == Device::GPU)
    {
        auto result = transformPoints(T.toCpu(), points.toCpu());
        return result.toGpu();
    }
    else
    {
        static_assert(Dev == Device::GPU, "CPU transformPoints is implemented in point_cloud_cpu.cpp");
    }
}

template <typename Scalar, Device Dev>
DenseMatrix<Scalar, Dev> covarianceMatrix(const DenseMatrix<Scalar, Dev>& points)
{
    if constexpr (Dev == Device::GPU)
    {
        return covarianceMatrix(points.toCpu()).toGpu();
    }
    else
    {
        static_assert(Dev == Device::GPU, "CPU covarianceMatrix is implemented in point_cloud_cpu.cpp");
    }
}

#ifdef PLAMATRIX_USE_FLOAT
template DenseMatrix<float, Device::GPU> rotationMatrix(const Vec3<float>&, float);
template DenseMatrix<float, Device::GPU> rigidTransform(const DenseMatrix<float, Device::GPU>&, const Vec3<float>&);
template DenseMatrix<float, Device::GPU> transformPoints(const DenseMatrix<float, Device::GPU>&,
                                                         const DenseMatrix<float, Device::GPU>&);
template DenseMatrix<float, Device::GPU> covarianceMatrix(const DenseMatrix<float, Device::GPU>&);
#endif

#ifdef PLAMATRIX_USE_DOUBLE
template DenseMatrix<double, Device::GPU> rotationMatrix(const Vec3<double>&, double);
template DenseMatrix<double, Device::GPU> rigidTransform(const DenseMatrix<double, Device::GPU>&, const Vec3<double>&);
template DenseMatrix<double, Device::GPU> transformPoints(const DenseMatrix<double, Device::GPU>&,
                                                          const DenseMatrix<double, Device::GPU>&);
template DenseMatrix<double, Device::GPU> covarianceMatrix(const DenseMatrix<double, Device::GPU>&);
#endif

} // namespace plamatrix
```

Add to Metal sources:

```cmake
        ops/point_cloud_metal.mm
```

- [ ] **Step 4: Run fallback tests**

Run:

```bash
cmake --build build-metal --target plamatrix_tests -j$(sysctl -n hw.ncpu)
./build-metal/test/plamatrix_tests --gtest_filter=PointCloudMetal.*:PointCloud.*
```

Expected: tests pass.

- [ ] **Step 5: Add true float Metal kernels**

Add Metal shader functions to `src/core/metal_context.mm`:

```objective-c++
kernel void transform_points_float(device const float* T [[buffer(0)]],
                                   device const float* points [[buffer(1)]],
                                   device float* output [[buffer(2)]],
                                   constant long long& n [[buffer(3)]],
                                   uint id [[thread_position_in_grid]])
{
    long long i = id;
    if (i < n)
    {
        float px = points[i + 0 * n];
        float py = points[i + 1 * n];
        float pz = points[i + 2 * n];
        output[i + 0 * n] = T[0] * px + T[4] * py + T[8] * pz + T[12];
        output[i + 1 * n] = T[1] * px + T[5] * py + T[9] * pz + T[13];
        output[i + 2 * n] = T[2] * px + T[6] * py + T[10] * pz + T[14];
    }
}
```

For covariance, implement a first correct GPU-assisted version:

- Metal kernel computes partial sums per point into a temporary float buffer with six covariance terms and three mean terms.
- CPU reduces the small temporary buffer for the first implementation if needed.
- Output remains a 3x3 GPU matrix.

Keep double on CPU fallback.

- [ ] **Step 6: Run point cloud workflow**

Run:

```bash
cmake --build build-metal --target plamatrix_tests -j$(sysctl -n hw.ncpu)
./build-metal/test/plamatrix_tests --gtest_filter=PointCloudMetal.*:PointCloud.*:PointCloudWorkflow.*
```

Expected: tests pass.

- [ ] **Step 7: Checkpoint**

Run:

```bash
git status --short
```

If commit authorization is present:

```bash
git add src/ops/point_cloud_metal.mm src/core/metal_context.mm include/plamatrix/core/metal_context.h src/CMakeLists.txt test/unit/ops/point_cloud_test.cpp test/integration/point_cloud_workflow_test.cpp
git commit -m "feat: add Metal point cloud backend"
```

---

## Task 9: Decomposition API and No-GPU Stubs

**Files:**
- Create: `src/ops/decomposition_metal.mm`
- Modify: `include/plamatrix/ops/decomposition.h`
- Modify: `include/plamatrix/ops/solver.h`
- Modify: `test/integration/no_cuda_behavior_test.cpp`
- Modify: `test/unit/ops/decomposition_test.cpp`
- Modify: `src/CMakeLists.txt`

- [ ] **Step 1: Write failing no-GPU and Metal decomposition tests**

Add to `test/integration/no_cuda_behavior_test.cpp` under no-GPU guard:

```cpp
#if defined(PLAMATRIX_NO_GPU)
TEST(NoGpuStubs, gpuDecompositionAndSolverAlgorithms_ThrowClearErrors)
{
    DenseMatrix<float, Device::GPU> A(2, 2);
    DenseMatrix<float, Device::GPU> B(2, 1);

    EXPECT_THROW(svd(A), std::runtime_error);
    EXPECT_THROW(qr(A), std::runtime_error);
    EXPECT_THROW(eigh(A), std::runtime_error);
    EXPECT_THROW(solve(A, B), std::runtime_error);
}
#endif
```

Include missing headers:

```cpp
#include <plamatrix/ops/decomposition.h>
#include <plamatrix/ops/solver.h>
```

Add under `#ifdef PLAMATRIX_WITH_METAL` in `test/unit/ops/decomposition_test.cpp`:

```cpp
TEST(DecompositionMetal, svdQrEigh_FloatApiMatchesCpuFallback)
{
    DenseMatrix<float, Device::CPU> A(2, 2);
    A(0, 0) = 2.0f; A(1, 0) = 0.0f;
    A(0, 1) = 0.0f; A(1, 1) = 1.0f;

    auto [U_cpu, S_cpu, Vt_cpu] = svd(A);
    auto [U_gpu, S_gpu, Vt_gpu] = svd(A.toGpu());
    auto S = S_gpu.toCpu();
    EXPECT_NEAR(S(0, 0), S_cpu(0, 0), 1e-5f);
    EXPECT_NEAR(S(1, 0), S_cpu(1, 0), 1e-5f);

    auto [Q_gpu, R_gpu] = qr(A.toGpu());
    auto Q = Q_gpu.toCpu();
    auto R = R_gpu.toCpu();
    EXPECT_EQ(Q.rows(), 2);
    EXPECT_EQ(R.cols(), 2);

    auto E_cpu = eigh(A);
    auto E_gpu = eigh(A.toGpu()).toCpu();
    EXPECT_NEAR(E_gpu(0, 0), E_cpu(0, 0), 1e-5f);
    EXPECT_NEAR(E_gpu(1, 0), E_cpu(1, 0), 1e-5f);
}
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
cmake --build build-metal --target plamatrix_tests -j$(sysctl -n hw.ncpu)
./build-metal/test/plamatrix_tests --gtest_filter=DecompositionMetal.*
```

Expected: link fails for missing Metal decomposition definitions.

Run:

```bash
cmake -S . -B build-none -DPLAMATRIX_BUILD_TESTS=ON -DPLAMATRIX_GPU_BACKEND=NONE
cmake --build build-none --target plamatrix_tests -j$(sysctl -n hw.ncpu)
./build-none/test/plamatrix_tests --gtest_filter=NoGpuStubs.*
```

Expected: missing GPU decomposition stubs fail to compile or link.

- [ ] **Step 3: Add no-GPU header stubs**

Modify `include/plamatrix/ops/decomposition.h`:

```cpp
#if defined(PLAMATRIX_NO_GPU)
template <typename Scalar>
std::tuple<DenseMatrix<Scalar, Device::GPU>, DenseMatrix<Scalar, Device::GPU>, DenseMatrix<Scalar, Device::GPU>>
svd(const DenseMatrix<Scalar, Device::GPU>&)
{
    throw std::runtime_error("svd: GPU decomposition requires a GPU backend");
}

template <typename Scalar>
std::tuple<DenseMatrix<Scalar, Device::GPU>, DenseMatrix<Scalar, Device::GPU>>
qr(const DenseMatrix<Scalar, Device::GPU>&)
{
    throw std::runtime_error("qr: GPU decomposition requires a GPU backend");
}

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> eigh(const DenseMatrix<Scalar, Device::GPU>&)
{
    throw std::runtime_error("eigh: GPU decomposition requires a GPU backend");
}
#endif
```

Modify `include/plamatrix/ops/solver.h`:

```cpp
#if defined(PLAMATRIX_NO_GPU)
template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> solve(const DenseMatrix<Scalar, Device::GPU>&,
                                       const DenseMatrix<Scalar, Device::GPU>&)
{
    throw std::runtime_error("solve: GPU linear solve requires a GPU backend");
}
#endif
```

- [ ] **Step 4: Implement Metal decomposition fallback**

Create `src/ops/decomposition_metal.mm`:

```objective-c++
#include "plamatrix/ops/decomposition.h"

namespace plamatrix
{

template <typename Scalar>
std::tuple<DenseMatrix<Scalar, Device::GPU>, DenseMatrix<Scalar, Device::GPU>, DenseMatrix<Scalar, Device::GPU>>
svd(const DenseMatrix<Scalar, Device::GPU>& A)
{
    auto [U_cpu, S_cpu, Vt_cpu] = svd(A.toCpu());
    return {U_cpu.toGpu(), S_cpu.toGpu(), Vt_cpu.toGpu()};
}

template <typename Scalar>
std::tuple<DenseMatrix<Scalar, Device::GPU>, DenseMatrix<Scalar, Device::GPU>>
qr(const DenseMatrix<Scalar, Device::GPU>& A)
{
    auto [Q_cpu, R_cpu] = qr(A.toCpu());
    return {Q_cpu.toGpu(), R_cpu.toGpu()};
}

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> eigh(const DenseMatrix<Scalar, Device::GPU>& A)
{
    return eigh(A.toCpu()).toGpu();
}

#ifdef PLAMATRIX_USE_FLOAT
template std::tuple<DenseMatrix<float, Device::GPU>, DenseMatrix<float, Device::GPU>, DenseMatrix<float, Device::GPU>>
svd(const DenseMatrix<float, Device::GPU>&);
template std::tuple<DenseMatrix<float, Device::GPU>, DenseMatrix<float, Device::GPU>>
qr(const DenseMatrix<float, Device::GPU>&);
template DenseMatrix<float, Device::GPU> eigh(const DenseMatrix<float, Device::GPU>&);
#endif

#ifdef PLAMATRIX_USE_DOUBLE
template std::tuple<DenseMatrix<double, Device::GPU>, DenseMatrix<double, Device::GPU>, DenseMatrix<double, Device::GPU>>
svd(const DenseMatrix<double, Device::GPU>&);
template std::tuple<DenseMatrix<double, Device::GPU>, DenseMatrix<double, Device::GPU>>
qr(const DenseMatrix<double, Device::GPU>&);
template DenseMatrix<double, Device::GPU> eigh(const DenseMatrix<double, Device::GPU>&);
#endif

} // namespace plamatrix
```

Add to Metal sources:

```cmake
        ops/decomposition_metal.mm
```

- [ ] **Step 5: Run tests**

Run:

```bash
cmake --build build-metal --target plamatrix_tests -j$(sysctl -n hw.ncpu)
./build-metal/test/plamatrix_tests --gtest_filter=DecompositionMetal.*:SVD.*:QR.*:Eigh.*

cmake --build build-none --target plamatrix_tests -j$(sysctl -n hw.ncpu)
./build-none/test/plamatrix_tests --gtest_filter=NoGpuStubs.*
```

Expected: Metal decomposition API passes through CPU fallback; no-GPU stubs throw clear runtime errors.

- [ ] **Step 6: Checkpoint**

Run:

```bash
git status --short
```

If commit authorization is present:

```bash
git add src/ops/decomposition_metal.mm include/plamatrix/ops/decomposition.h include/plamatrix/ops/solver.h test/integration/no_cuda_behavior_test.cpp test/unit/ops/decomposition_test.cpp src/CMakeLists.txt
git commit -m "feat: add Metal decomposition fallback API"
```

---

## Task 10: Test Runner, Benchmark, and Documentation

**Files:**
- Modify: `test/test_main.cpp`
- Modify: `benchmark/CMakeLists.txt`
- Modify: `benchmark/main.cpp`
- Modify: `benchmark/report_writer.cpp`
- Modify: `README.md`
- Modify: `docs/build.md`
- Modify: `docs/architecture.md`
- Modify: `docs/api/dense-matrix.md`
- Modify: `docs/api/linear-algebra.md`
- Modify: `docs/api/point-cloud.md`
- Modify: `docs/contributing.md`

- [ ] **Step 1: Write failing backend-aware test runner expectation**

Run on Metal:

```bash
./build-metal/test/plamatrix_tests --gtest_filter='*Gpu*'
```

Expected before changes: test runner may incorrectly treat non-CUDA as unavailable and skip GPU tests. The desired behavior is that Metal runs GPU tests guarded by `PLAMATRIX_WITH_METAL`.

- [ ] **Step 2: Update test runner backend filtering**

Modify `test/test_main.cpp`:

```cpp
bool hasUsableGpuRuntime()
{
#ifdef PLAMATRIX_WITH_CUDA
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount <= 0)
    {
        cudaGetLastError();
        return false;
    }
    err = cudaFree(nullptr);
    if (err != cudaSuccess)
    {
        cudaGetLastError();
        return false;
    }
    return true;
#elif defined(PLAMATRIX_WITH_METAL)
    return true;
#else
    return false;
#endif
}
```

Replace calls to `hasUsableCudaRuntime()` with `hasUsableGpuRuntime()` and update log text from `CUDA runtime unavailable` to `GPU runtime unavailable`.

- [ ] **Step 3: Update benchmark build guards**

Modify `benchmark/CMakeLists.txt`:

```cmake
if(PLAMATRIX_WITH_CUDA)
    add_executable(plamatrix_benchmark
        main.cpp
        benchmark_cases.cpp
        benchmark_cases.cu
        report_writer.cpp
    )
else()
    add_executable(plamatrix_benchmark
        main.cpp
        benchmark_cases.cpp
        report_writer.cpp
    )
endif()
target_include_directories(plamatrix_benchmark PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(plamatrix_benchmark PRIVATE plamatrix)
```

Modify `benchmark/main.cpp` messages:

```cpp
#ifndef PLAMATRIX_WITH_CUDA
#ifndef PLAMATRIX_WITH_METAL
    if (run_cuda)
    {
        ...
    }
#endif
#endif
```

When updating docs, keep CLI `--mode cuda` as an alias for GPU mode to avoid breaking scripts.

- [ ] **Step 4: Update report writer backend labels**

Modify `benchmark/report_writer.cpp` to include `gpuBackendName()`:

```cpp
#include "plamatrix/core/gpu_backend.h"
```

When writing environment info:

```cpp
out << "- GPU backend: " << gpuBackendName() << "\n";
```

- [ ] **Step 5: Update README and docs**

Add to `README.md` CMake options:

```markdown
| `PLAMATRIX_GPU_BACKEND` | `AUTO` | GPU 后端：`AUTO`、`CUDA`、`METAL`、`NONE` |
| `PLAMATRIX_WITH_OPENMP` | `AUTO` | OpenMP：`AUTO`、`ON`、`OFF` |
```

Add macOS note:

```markdown
macOS 默认使用 Metal/MPS 后端，用户代码仍写 `DenseMatrix<Scalar, Device::GPU>`。
`float` 热路径尽量使用 Metal/MPS；`double` GPU API 在 macOS 上使用 CPU/Accelerate fallback，
保持类型和结果兼容，但不是 Apple GPU double 加速。
```

Update `docs/build.md` macOS section with this content:

~~~markdown
### macOS Metal 构建

```bash
cmake -S . -B build-metal -DPLAMATRIX_BUILD_TESTS=ON -DPLAMATRIX_GPU_BACKEND=METAL
cmake --build build-metal -j$(sysctl -n hw.ncpu)
ctest --test-dir build-metal --output-on-failure
```

如果未安装 Homebrew `libomp`，默认 `PLAMATRIX_WITH_OPENMP=AUTO` 会关闭 OpenMP 并继续构建。
~~~

Update API docs with:

```markdown
`Device::GPU` 表示当前构建选择的平台 GPU 后端。CUDA 构建下它是 CUDA device memory；
macOS Metal 构建下它是 Metal/MPS 后端管理的矩阵存储。不要在用户代码中直接解引用 GPU
矩阵的 `data()` 指针。
```

- [ ] **Step 6: Run verification matrix**

Run:

```bash
cmake -S . -B build-none -DPLAMATRIX_BUILD_TESTS=ON -DPLAMATRIX_GPU_BACKEND=NONE
cmake --build build-none -j$(sysctl -n hw.ncpu)
ctest --test-dir build-none --output-on-failure

cmake -S . -B build-metal -DPLAMATRIX_BUILD_TESTS=ON -DPLAMATRIX_GPU_BACKEND=METAL
cmake --build build-metal -j$(sysctl -n hw.ncpu)
ctest --test-dir build-metal --output-on-failure
```

Expected: both builds pass on macOS.

On CUDA machine:

```bash
cmake -S . -B build -DPLAMATRIX_BUILD_TESTS=ON -DPLAMATRIX_GPU_BACKEND=CUDA
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

Expected: CUDA build passes and existing CUDA tests remain valid.

- [ ] **Step 7: Checkpoint**

Run:

```bash
git status --short
```

If commit authorization is present:

```bash
git add test/test_main.cpp benchmark/CMakeLists.txt benchmark/main.cpp benchmark/report_writer.cpp README.md docs/build.md docs/architecture.md docs/api/dense-matrix.md docs/api/linear-algebra.md docs/api/point-cloud.md docs/contributing.md
git commit -m "docs: document platform GPU backend selection"
```

---

## Self-Review

Spec coverage:

- Backend selection and `Device::GPU` compatibility: Tasks 1, 2, 10.
- macOS Metal/MPS backend: Tasks 3 through 8.
- macOS double CPU/Accelerate fallback: Tasks 5 through 9 and documentation in Task 10.
- CUDA compatibility: Tasks 2, 3, 10 verification matrix.
- Optional OpenMP for macOS: Task 2.
- No-GPU stubs: Tasks 2, 9.
- Documentation: Task 10.

Type consistency:

- Public backend enum is always `plamatrix::GpuBackend`.
- Public query functions are always `gpuBackend()` and `gpuBackendName()`.
- Runtime wrappers live under `plamatrix::detail`.
- Metal functions are declared in `metal_context.h` and implemented in `.mm`.
- Existing user API remains `DenseMatrix<Scalar, Device::GPU>`.

Implementation notes:

- The plan intentionally starts Metal GEMM, solver, and point-cloud with CPU-correct fallback before replacing float hot paths with MPS/Metal kernels. This preserves TDD red/green checkpoints and gives a working macOS backend at each milestone.
- `data()` remains present for source compatibility. Documentation must state that GPU `data()` is a backend token and should not be dereferenced by user code.
- Commit steps are conditional on explicit commit authorization, matching this repository's instructions.
