#pragma once

#include <cstddef>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <stdexcept>

#ifdef PLAMATRIX_NO_CUDA
#include "plamatrix/core/no_cuda_stubs.h"
#else
#include <cuda_runtime.h>
#endif

#include "plamatrix/core/error_check.h"

namespace plamatrix
{

namespace detail
{

template <typename Scalar>
std::size_t checkedAllocationBytes(std::size_t count)
{
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(Scalar))
    {
        std::ostringstream oss;
        oss << "allocation size overflows size_t for " << count << " elements of "
            << sizeof(Scalar) << " bytes";
        throw std::overflow_error(oss.str());
    }
    return count * sizeof(Scalar);
}

} // namespace detail

/// CPU memory allocator with 32-byte alignment (suitable for AVX/SSE).
/// Available in both CPU-only and CUDA builds.
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
        std::size_t bytes = detail::checkedAllocationBytes<Scalar>(count);
        void* ptr = nullptr;
        int rc = posix_memalign(&ptr, 32, bytes);
        if (rc != 0)
        {
            throw std::bad_alloc();
        }
        return static_cast<Scalar*>(ptr);
    }

    /// Deallocate memory previously allocated by CpuAllocator::allocate.
    /// @param ptr  Pointer to free (nullptr is safe)
    static void deallocate(Scalar* ptr)
    {
        std::free(ptr);
    }

    static void deallocateNoThrow(Scalar* ptr) noexcept
    {
        std::free(ptr);
    }
};

/// GPU memory allocator. Uses CUDA device memory when available, falls back to
/// CPU memory (via stubs) when compiled without CUDA (PLAMATRIX_NO_CUDA).
/// @tparam Scalar  Element type (float, double, etc.)
template <typename Scalar>
struct GpuAllocator
{
    /// Allocate device memory for `count` elements.
    /// @throws std::runtime_error  if CUDA allocation fails
    static Scalar* allocate(std::size_t count)
    {
        std::size_t bytes = detail::checkedAllocationBytes<Scalar>(count);
        Scalar* ptr = nullptr;
        PLAMATRIX_CHECK_CUDA(cudaMalloc(&ptr, bytes));
        return ptr;
    }

    /// Deallocate device memory. nullptr is safe (no-op via cudaFree).
    /// @throws std::runtime_error  if deallocation fails (e.g., double-free)
    static void deallocate(Scalar* ptr)
    {
        PLAMATRIX_CHECK_CUDA(cudaFree(ptr));
    }

    static void deallocateNoThrow(Scalar* ptr) noexcept
    {
        if (ptr)
        {
            static_cast<void>(cudaFree(ptr));
        }
    }
};

} // namespace plamatrix
