#pragma once

#include <sstream>
#include <stdexcept>
#include <string>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

namespace plamatrix
{

/// Check CUDA runtime API error. Throws std::runtime_error with file/line/expression details on failure.
/// @param err  cudaError_t returned by a CUDA runtime API call
/// @param file source file name (typically __FILE__)
/// @param line line number (typically __LINE__)
/// @param expr expression string (typically #call from the macro)
inline void cudaCheck(cudaError_t err, const char* file, int line, const char* expr)
{
    if (err != cudaSuccess)
    {
        std::ostringstream oss;
        oss << "CUDA error at " << file << ":" << line
            << " (" << expr << "): " << cudaGetErrorString(err);
        throw std::runtime_error(oss.str());
    }
}

namespace detail
{

inline const char* cublasStatusString(cublasStatus_t stat)
{
    switch (stat)
    {
    case CUBLAS_STATUS_SUCCESS:          return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:  return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:     return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:    return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:    return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:    return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:   return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:    return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:    return "CUBLAS_STATUS_LICENSE_ERROR";
    default:                             return "UNKNOWN_CUBLAS_STATUS";
    }
}

inline const char* cusolverStatusString(cusolverStatus_t stat)
{
    switch (stat)
    {
    case CUSOLVER_STATUS_SUCCESS:                return "CUSOLVER_STATUS_SUCCESS";
    case CUSOLVER_STATUS_NOT_INITIALIZED:        return "CUSOLVER_STATUS_NOT_INITIALIZED";
    case CUSOLVER_STATUS_ALLOC_FAILED:           return "CUSOLVER_STATUS_ALLOC_FAILED";
    case CUSOLVER_STATUS_INVALID_VALUE:          return "CUSOLVER_STATUS_INVALID_VALUE";
    case CUSOLVER_STATUS_ARCH_MISMATCH:          return "CUSOLVER_STATUS_ARCH_MISMATCH";
    case CUSOLVER_STATUS_MAPPING_ERROR:          return "CUSOLVER_STATUS_MAPPING_ERROR";
    case CUSOLVER_STATUS_EXECUTION_FAILED:       return "CUSOLVER_STATUS_EXECUTION_FAILED";
    case CUSOLVER_STATUS_INTERNAL_ERROR:         return "CUSOLVER_STATUS_INTERNAL_ERROR";
    case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED: return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    case CUSOLVER_STATUS_NOT_SUPPORTED:          return "CUSOLVER_STATUS_NOT_SUPPORTED";
    case CUSOLVER_STATUS_ZERO_PIVOT:             return "CUSOLVER_STATUS_ZERO_PIVOT";
    case CUSOLVER_STATUS_INVALID_LICENSE:        return "CUSOLVER_STATUS_INVALID_LICENSE";
    default:                                     return "UNKNOWN_CUSOLVER_STATUS";
    }
}

} // namespace detail

/// Check cuBLAS API error. Throws std::runtime_error with file/line/expression details on failure.
/// @param stat cublasStatus_t returned by a cuBLAS API call
/// @param file source file name (typically __FILE__)
/// @param line line number (typically __LINE__)
/// @param expr expression string (typically #call from the macro)
inline void cublasCheck(cublasStatus_t stat, const char* file, int line, const char* expr)
{
    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        std::ostringstream oss;
        oss << "cuBLAS error at " << file << ":" << line
            << " (" << expr << "): " << detail::cublasStatusString(stat)
            << " (" << static_cast<int>(stat) << ")";
        throw std::runtime_error(oss.str());
    }
}

/// Check cuSOLVER API error. Throws std::runtime_error with file/line/expression details on failure.
/// @param stat cusolverStatus_t returned by a cuSOLVER API call
/// @param file source file name (typically __FILE__)
/// @param line line number (typically __LINE__)
/// @param expr expression string (typically #call from the macro)
inline void cusolverCheck(cusolverStatus_t stat, const char* file, int line, const char* expr)
{
    if (stat != CUSOLVER_STATUS_SUCCESS)
    {
        std::ostringstream oss;
        oss << "cuSOLVER error at " << file << ":" << line
            << " (" << expr << "): " << detail::cusolverStatusString(stat)
            << " (" << static_cast<int>(stat) << ")";
        throw std::runtime_error(oss.str());
    }
}

} // namespace plamatrix

#define PLAMATRIX_CHECK_CUDA(call)    plamatrix::cudaCheck((call), __FILE__, __LINE__, #call)
#define PLAMATRIX_CHECK_CUBLAS(call)  plamatrix::cublasCheck((call), __FILE__, __LINE__, #call)
#define PLAMATRIX_CHECK_CUSOLVER(call) plamatrix::cusolverCheck((call), __FILE__, __LINE__, #call)
