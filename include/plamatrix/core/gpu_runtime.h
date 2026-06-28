#pragma once

#include <cstddef>

#include "plamatrix/core/error_check.h"

namespace plamatrix
{
namespace detail
{

/// Allocate storage for the configured GPU backend.
/// @param bytes Number of bytes to allocate.
/// @return Backend-specific pointer token.
void* gpuAllocateBytes(std::size_t bytes);

/// Release storage allocated by gpuAllocateBytes.
/// @param ptr    Backend pointer token.
/// @param bytes  Original allocation byte size.
void gpuFreeBytes(void* ptr, std::size_t bytes);

/// Release storage allocated by gpuAllocateBytes without throwing.
/// @param ptr    Backend pointer token.
/// @param bytes  Original allocation byte size.
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
