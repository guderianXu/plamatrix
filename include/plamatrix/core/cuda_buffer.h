#pragma once

#include <cstddef>

#include <cuda_runtime_api.h>

#include "plamatrix/core/error_check.h"

namespace plamatrix::detail
{

class CudaBuffer
{
public:
    explicit CudaBuffer(std::size_t bytes)
    {
        if (bytes != 0)
        {
            PLAMATRIX_CHECK_CUDA(cudaMalloc(&_data, bytes));
        }
    }

    ~CudaBuffer() noexcept
    {
        if (_data != nullptr)
        {
            static_cast<void>(cudaFree(_data));
        }
    }

    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    template <typename T>
    T* as() const noexcept
    {
        return static_cast<T*>(_data);
    }

private:
    void* _data = nullptr;
};

} // namespace plamatrix::detail
