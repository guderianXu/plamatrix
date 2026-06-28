#pragma once

#include <cstddef>

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

/// Fill a float Metal buffer using a compute kernel.
void metalFillFloat(void* dst, Index count, float value);

/// Fill a double Metal buffer using CPU-visible shared storage fallback.
void metalFillDoubleFallback(void* dst, Index count, double value);

/// Transpose a column-major float matrix using a compute kernel.
void metalTransposeFloat(const void* src, void* dst, Index rows, Index cols);

/// Transpose a column-major double matrix using CPU-visible shared storage fallback.
void metalTransposeDoubleFallback(const void* src, void* dst, Index rows, Index cols);

/// Add two float Metal buffers using a compute kernel.
void metalAddFloat(const void* a, const void* b, void* c, Index count);

/// Subtract two float Metal buffers using a compute kernel.
void metalSubFloat(const void* a, const void* b, void* c, Index count);

/// Add two double Metal buffers using CPU-visible shared storage fallback.
void metalAddDoubleFallback(const void* a, const void* b, void* c, Index count);

/// Subtract two double Metal buffers using CPU-visible shared storage fallback.
void metalSubDoubleFallback(const void* a, const void* b, void* c, Index count);

/// Multiply column-major float GPU matrices using MPS.
void metalGemmFloat(const void* a,
                    const void* b,
                    void* c,
                    Index m,
                    Index n,
                    Index k);

/// Multiply column-major double GPU matrices using CPU-visible shared storage fallback.
void metalGemmDoubleFallback(const void* a,
                             const void* b,
                             void* c,
                             Index m,
                             Index n,
                             Index k);

/// Solve A * X = B for column-major float matrices using MPS LU.
void metalSolveFloat(const void* a,
                     const void* b,
                     void* x,
                     Index n,
                     Index nrhs);

/// Transform column-major Nx3 float point cloud using a Metal compute kernel.
void metalTransformPointsFloat(const void* transform,
                               const void* points,
                               void* output,
                               Index point_count);

/// Transform column-major Nx3 double point cloud using CPU-visible shared storage fallback.
void metalTransformPointsDoubleFallback(const void* transform,
                                        const void* points,
                                        void* output,
                                        Index point_count);

} // namespace detail
} // namespace plamatrix
