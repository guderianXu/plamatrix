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
