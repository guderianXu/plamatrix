#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include <sstream>
#include <stdexcept>
#include <string>

namespace plamatrix
{

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

inline void cublasCheck(cublasStatus_t stat, const char* file, int line, const char* expr)
{
    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        std::ostringstream oss;
        oss << "cuBLAS error at " << file << ":" << line
            << " (" << expr << "): status=" << static_cast<int>(stat);
        throw std::runtime_error(oss.str());
    }
}

inline void cusolverCheck(cusolverStatus_t stat, const char* file, int line, const char* expr)
{
    if (stat != CUSOLVER_STATUS_SUCCESS)
    {
        std::ostringstream oss;
        oss << "cuSOLVER error at " << file << ":" << line
            << " (" << expr << "): status=" << static_cast<int>(stat);
        throw std::runtime_error(oss.str());
    }
}

} // namespace plamatrix

#define PLAMATRIX_CHECK_CUDA(call)    plamatrix::cudaCheck((call), __FILE__, __LINE__, #call)
#define PLAMATRIX_CHECK_CUBLAS(call)  plamatrix::cublasCheck((call), __FILE__, __LINE__, #call)
#define PLAMATRIX_CHECK_CUSOLVER(call) plamatrix::cusolverCheck((call), __FILE__, __LINE__, #call)
