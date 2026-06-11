#pragma once

#include <cstddef>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

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

class GpuMemoryPool
{
public:
    static void setEnabled(bool enabled)
    {
        enabledFlag() = enabled;
    }

    static bool isEnabled()
    {
        return enabledFlag();
    }

    static void* acquire(std::size_t bytes)
    {
        if (bytes == 0)
        {
            return nullptr;
        }

        if (isEnabled())
        {
            std::lock_guard<std::mutex> lock(mutex());
            auto& blocks = freeBlocks()[bytes];
            if (!blocks.empty())
            {
                void* ptr = blocks.back();
                blocks.pop_back();
                return ptr;
            }
        }

        void* ptr = nullptr;
        PLAMATRIX_CHECK_CUDA(cudaMalloc(&ptr, bytes));
        return ptr;
    }

    static void release(void* ptr, std::size_t bytes)
    {
        if (ptr == nullptr)
        {
            return;
        }

        if (isEnabled() && bytes > 0)
        {
            std::lock_guard<std::mutex> lock(mutex());
            freeBlocks()[bytes].push_back(ptr);
            return;
        }

        PLAMATRIX_CHECK_CUDA(cudaFree(ptr));
    }

    static void releaseNoThrow(void* ptr, std::size_t bytes) noexcept
    {
        try
        {
            release(ptr, bytes);
        }
        catch (...)
        {
            static_cast<void>(cudaFree(ptr));
        }
    }

    static void releaseAll()
    {
        std::vector<void*> blocks_to_free;
        {
            std::lock_guard<std::mutex> lock(mutex());
            for (auto& entry : freeBlocks())
            {
                blocks_to_free.insert(blocks_to_free.end(), entry.second.begin(), entry.second.end());
            }
            freeBlocks().clear();
        }

        for (void* ptr : blocks_to_free)
        {
            PLAMATRIX_CHECK_CUDA(cudaFree(ptr));
        }
    }

    static std::size_t cachedBlockCount()
    {
        std::lock_guard<std::mutex> lock(mutex());
        std::size_t count = 0;
        for (const auto& entry : freeBlocks())
        {
            count += entry.second.size();
        }
        return count;
    }

    static std::size_t cachedBytes()
    {
        std::lock_guard<std::mutex> lock(mutex());
        std::size_t bytes = 0;
        for (const auto& entry : freeBlocks())
        {
            bytes += entry.first * entry.second.size();
        }
        return bytes;
    }

private:
    static bool& enabledFlag()
    {
        static bool enabled = false;
        return enabled;
    }

    static std::mutex& mutex()
    {
        static std::mutex pool_mutex;
        return pool_mutex;
    }

    static std::unordered_map<std::size_t, std::vector<void*>>& freeBlocks()
    {
        static std::unordered_map<std::size_t, std::vector<void*>> blocks;
        return blocks;
    }
};

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

/// Pinned host allocator for faster/true asynchronous CUDA transfers.
/// Falls back to CPU memory stubs when compiled without CUDA.
/// @tparam Scalar  Element type (float, double, etc.)
template <typename Scalar>
struct PinnedCpuAllocator
{
    /// Allocate page-locked host memory for `count` elements.
    /// @param count  Number of elements to allocate
    /// @return  Pointer to allocated memory
    /// @throws std::runtime_error  if CUDA host allocation fails
    static Scalar* allocate(std::size_t count)
    {
        std::size_t bytes = detail::checkedAllocationBytes<Scalar>(count);
        void* ptr = nullptr;
        PLAMATRIX_CHECK_CUDA(cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault));
        return static_cast<Scalar*>(ptr);
    }

    /// Deallocate memory previously allocated by PinnedCpuAllocator::allocate.
    /// @param ptr  Pointer to free (nullptr is safe)
    static void deallocate(Scalar* ptr)
    {
        PLAMATRIX_CHECK_CUDA(cudaFreeHost(ptr));
    }

    static void deallocateNoThrow(Scalar* ptr) noexcept
    {
        if (ptr)
        {
            static_cast<void>(cudaFreeHost(ptr));
        }
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
        return static_cast<Scalar*>(detail::GpuMemoryPool::acquire(bytes));
    }

    /// Deallocate device memory. nullptr is safe (no-op via cudaFree).
    /// @throws std::runtime_error  if deallocation fails (e.g., double-free)
    static void deallocate(Scalar* ptr)
    {
        PLAMATRIX_CHECK_CUDA(cudaFree(ptr));
    }

    /// Deallocate device memory with element count. Uses the memory pool when enabled.
    /// @param ptr    Pointer to release
    /// @param count  Number of elements originally allocated
    static void deallocate(Scalar* ptr, std::size_t count)
    {
        std::size_t bytes = detail::checkedAllocationBytes<Scalar>(count);
        detail::GpuMemoryPool::release(ptr, bytes);
    }

    static void deallocateNoThrow(Scalar* ptr) noexcept
    {
        if (ptr)
        {
            static_cast<void>(cudaFree(ptr));
        }
    }

    static void deallocateNoThrow(Scalar* ptr, std::size_t count) noexcept
    {
        if (ptr)
        {
            std::size_t bytes = 0;
            if (count <= std::numeric_limits<std::size_t>::max() / sizeof(Scalar))
            {
                bytes = count * sizeof(Scalar);
            }
            detail::GpuMemoryPool::releaseNoThrow(ptr, bytes);
        }
    }

    /// Enable or disable the process-local GPU memory pool.
    /// @param enabled  true to cache same-sized freed blocks for reuse
    static void setMemoryPoolEnabled(bool enabled)
    {
        detail::GpuMemoryPool::setEnabled(enabled);
    }

    /// @return whether the process-local GPU memory pool is enabled.
    static bool isMemoryPoolEnabled()
    {
        return detail::GpuMemoryPool::isEnabled();
    }

    /// Free all cached blocks currently held by the process-local GPU memory pool.
    static void releaseMemoryPool()
    {
        detail::GpuMemoryPool::releaseAll();
    }

    /// @return number of cached blocks currently held by the process-local GPU memory pool.
    static std::size_t cachedBlockCount()
    {
        return detail::GpuMemoryPool::cachedBlockCount();
    }

    /// @return total bytes currently cached by the process-local GPU memory pool.
    static std::size_t cachedBytes()
    {
        return detail::GpuMemoryPool::cachedBytes();
    }
};

} // namespace plamatrix
