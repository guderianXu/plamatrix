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
    if (bytes > 0)
    {
        std::memcpy(dst, src, bytes);
    }
#endif
}

void gpuMemset(void* dst, int value, std::size_t bytes, cudaStream_t stream)
{
    PLAMATRIX_CHECK_CUDA(cudaMemsetAsync(dst, value, bytes, stream));
}

} // namespace detail
} // namespace plamatrix
