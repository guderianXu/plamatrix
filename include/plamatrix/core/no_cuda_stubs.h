#pragma once

// ============================================================================
// Stub definitions for building without CUDA (PLAMATRIX_NO_CUDA).
//
// CUDA storage/transfer primitives are emulated with CPU equivalents:
//   - cudaMalloc  → malloc
//   - cudaFree    → free
//   - cudaMemcpy  → memcpy
//   - cudaMemset  → memset
//
// This allows matrix storage and transfer APIs to compile without CUDA. CUDA
// algorithm definitions in .cu files are only built when PLAMATRIX_WITH_CUDA=ON.
// ============================================================================

#ifdef PLAMATRIX_NO_CUDA

#include <cstdlib>
#include <cstring>

// ---- Stub CUDA types ----
using cudaError_t = int;
constexpr cudaError_t cudaSuccess = 0;
constexpr unsigned int cudaHostAllocDefault = 0;

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
    if (size == 0)
    {
        *ptr = nullptr;
        return cudaSuccess;
    }
    *ptr = std::malloc(size);
    return (*ptr != nullptr) ? cudaSuccess : 1;
}

template <typename T>
inline cudaError_t cudaMalloc(T** ptr, std::size_t size)
{
    void* raw = nullptr;
    cudaError_t err = cudaMalloc(&raw, size);
    *ptr = static_cast<T*>(raw);
    return err;
}

inline cudaError_t cudaFree(void* ptr)
{
    std::free(ptr);
    return cudaSuccess;
}

inline cudaError_t cudaHostAlloc(void** ptr, std::size_t size, unsigned int)
{
    if (size == 0)
    {
        *ptr = nullptr;
        return cudaSuccess;
    }
    *ptr = std::malloc(size);
    return (*ptr != nullptr) ? cudaSuccess : 1;
}

template <typename T>
inline cudaError_t cudaHostAlloc(T** ptr, std::size_t size, unsigned int flags)
{
    void* raw = nullptr;
    cudaError_t err = cudaHostAlloc(&raw, size, flags);
    *ptr = static_cast<T*>(raw);
    return err;
}

inline cudaError_t cudaFreeHost(void* ptr)
{
    std::free(ptr);
    return cudaSuccess;
}

inline cudaError_t cudaMemcpy(void* dst, const void* src, std::size_t count, int)
{
    if (count == 0)
    {
        return cudaSuccess;
    }
    std::memcpy(dst, src, count);
    return cudaSuccess;
}

inline cudaError_t cudaMemcpyAsync(void* dst, const void* src, std::size_t count, int kind, cudaStream_t)
{
    return cudaMemcpy(dst, src, count, kind);
}

inline cudaError_t cudaMemset(void* ptr, int value, std::size_t count)
{
    if (count == 0)
    {
        return cudaSuccess;
    }
    std::memset(ptr, value, count);
    return cudaSuccess;
}

inline cudaError_t cudaMemsetAsync(void* ptr, int value, std::size_t count, cudaStream_t)
{
    return cudaMemset(ptr, value, count);
}

inline cudaError_t cudaStreamCreate(cudaStream_t* stream)
{
    *stream = nullptr;
    return cudaSuccess;
}

inline cudaError_t cudaStreamDestroy(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }

constexpr int cudaMemcpyHostToDevice = 0;
constexpr int cudaMemcpyDeviceToHost = 0;

// Note: error-check macros (PLAMATRIX_CHECK_*) are defined in error_check.h.
// This file only provides type/function stubs.

#else
// CUDA is available: this file is a no-op, real CUDA headers are used elsewhere.
#endif // PLAMATRIX_NO_CUDA
