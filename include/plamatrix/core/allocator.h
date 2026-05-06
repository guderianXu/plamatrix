#pragma once

#include <cstddef>
#include <cstdlib>

#include <cuda_runtime.h>

#include "plamatrix/core/error_check.h"

namespace plamatrix
{

/// CPU memory allocator with 32-byte alignment (suitable for AVX/SSE).
/// @tparam Scalar  Element type (float, double, etc.)
template <typename Scalar>
struct CpuAllocator
{
    /// Allocate aligned memory for `count` elements.
    /// @param count  Number of elements to allocate
    /// @return  Pointer to allocated memory (never null)
    /// @throws std::bad_alloc  if allocation fails
    static Scalar* allocate(std::size_t count)
    {
        void* ptr = nullptr;
        int rc = posix_memalign(&ptr, 32, count * sizeof(Scalar));
        if (rc != 0)
        {
            throw std::bad_alloc();
        }
        return static_cast<Scalar*>(ptr);
    }

    /// Deallocate memory previously allocated by CpuAllocator::allocate.
    /// @param ptr  Pointer to free (nullptr is safe — no-op)
    static void deallocate(Scalar* ptr)
    {
        std::free(ptr);
    }
};

/// GPU memory allocator using CUDA device memory.
/// @tparam Scalar  Element type (float, double, etc.)
template <typename Scalar>
struct GpuAllocator
{
    /// Allocate device memory for `count` elements.
    /// @param count  Number of elements to allocate
    /// @return  Pointer to device memory (never null)
    /// @throws std::runtime_error  if CUDA allocation fails
    static Scalar* allocate(std::size_t count)
    {
        Scalar* ptr = nullptr;
        PLAMATRIX_CHECK_CUDA(cudaMalloc(&ptr, count * sizeof(Scalar)));
        return ptr;
    }

    /// Deallocate device memory previously allocated by GpuAllocator::allocate.
    /// @param ptr  Pointer to free (nullptr is safe — no-op via cudaFree)
    /// @throws std::runtime_error  if CUDA deallocation fails (e.g., double-free)
    static void deallocate(Scalar* ptr)
    {
        PLAMATRIX_CHECK_CUDA(cudaFree(ptr));
    }
};

} // namespace plamatrix
