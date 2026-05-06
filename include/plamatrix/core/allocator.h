#pragma once

#include <cstdlib>
#include <cuda_runtime.h>

#include "plamatrix/core/error_check.h"

namespace plamatrix
{

template <typename Scalar>
struct CpuAllocator
{
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

    static void deallocate(Scalar* ptr)
    {
        std::free(ptr);
    }
};

template <typename Scalar>
struct GpuAllocator
{
    static Scalar* allocate(std::size_t count)
    {
        Scalar* ptr = nullptr;
        PLAMATRIX_CHECK_CUDA(cudaMalloc(&ptr, count * sizeof(Scalar)));
        return ptr;
    }

    static void deallocate(Scalar* ptr)
    {
        PLAMATRIX_CHECK_CUDA(cudaFree(ptr));
    }
};

} // namespace plamatrix
