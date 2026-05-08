#pragma once

// ============================================================================
// Stub definitions for building without CUDA (PLAMATRIX_NO_CUDA).
//
// All GPU operations are emulated with CPU equivalents:
//   - cudaMalloc  → malloc
//   - cudaFree    → free
//   - cudaMemcpy  → memcpy
//   - cudaMemset  → memset
//   - GPU kernels → no-ops (header-only declarations only)
//
// This allows the library to compile and run without CUDA, using CPU memory
// for everything. Performance is obviously CPU-bound. To get real GPU
// acceleration, rebuild with -DPLAMATRIX_WITH_CUDA=ON.
// ============================================================================

#ifdef PLAMATRIX_NO_CUDA

#include <cstdlib>
#include <cstring>

// ---- Stub CUDA types ----
using cudaError_t = int;
constexpr cudaError_t cudaSuccess = 0;

using cudaStream_t = void*;
inline const char* cudaGetErrorString(int) { return "CUDA not available (PLAMATRIX_NO_CUDA)"; }

// ---- Stub cuBLAS types ----
using cublasStatus_t = int;
constexpr cublasStatus_t CUBLAS_STATUS_SUCCESS = 0;
using cublasHandle_t = void*;
inline cublasStatus_t cublasCreate(cublasHandle_t*) { return CUBLAS_STATUS_SUCCESS; }
inline cublasStatus_t cublasDestroy(cublasHandle_t) { return CUBLAS_STATUS_SUCCESS; }

// ---- Stub cuSOLVER types ----
using cusolverStatus_t = int;
constexpr cusolverStatus_t CUSOLVER_STATUS_SUCCESS = 0;
using cusolverDnHandle_t = void*;
inline cusolverStatus_t cusolverDnCreate(cusolverDnHandle_t*) { return CUSOLVER_STATUS_SUCCESS; }
inline cusolverStatus_t cusolverDnDestroy(cusolverDnHandle_t) { return CUSOLVER_STATUS_SUCCESS; }

// ---- Stub CUDA runtime API — uses CPU memory ----
inline cudaError_t cudaMalloc(void** ptr, std::size_t size)
{
    *ptr = std::malloc(size);
    return (*ptr != nullptr) ? cudaSuccess : 1;
}

inline cudaError_t cudaFree(void* ptr)
{
    std::free(ptr);
    return cudaSuccess;
}

inline cudaError_t cudaMemcpy(void* dst, const void* src, std::size_t count, int)
{
    std::memcpy(dst, src, count);
    return cudaSuccess;
}

inline cudaError_t cudaMemset(void* ptr, int value, std::size_t count)
{
    std::memset(ptr, value, count);
    return cudaSuccess;
}

inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }

constexpr int cudaMemcpyHostToDevice = 0;
constexpr int cudaMemcpyDeviceToHost = 0;

// Note: error-check macros (PLAMATRIX_CHECK_*) are defined in error_check.h.
// This file only provides type/function stubs.

#else
// CUDA is available: this file is a no-op, real CUDA headers are used elsewhere.
#endif // PLAMATRIX_NO_CUDA
