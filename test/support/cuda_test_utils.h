#pragma once

#include <cstddef>

#include "plamatrix/core/allocator.h"

namespace plamatrix
{
namespace test
{

#ifdef PLAMATRIX_WITH_CUDA

class CudaStreamGuard
{
public:
    CudaStreamGuard()
    {
        PLAMATRIX_CHECK_CUDA(cudaStreamCreate(&_stream));
    }

    ~CudaStreamGuard() noexcept
    {
        if (_stream != nullptr)
        {
            static_cast<void>(cudaStreamSynchronize(_stream));
            static_cast<void>(cudaStreamDestroy(_stream));
        }
    }

    CudaStreamGuard(const CudaStreamGuard&) = delete;
    CudaStreamGuard& operator=(const CudaStreamGuard&) = delete;

    cudaStream_t get() const noexcept
    {
        return _stream;
    }

    void synchronize()
    {
        PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(_stream));
    }

    void destroy()
    {
        PLAMATRIX_CHECK_CUDA(cudaStreamDestroy(_stream));
        _stream = nullptr;
    }

private:
    cudaStream_t _stream = nullptr;
};

template <typename Scalar>
class AsyncGpuPointerGuard
{
public:
    AsyncGpuPointerGuard(std::size_t count, cudaStream_t stream)
        : _stream(stream)
        , _ptr(GpuAllocator<Scalar>::allocateAsync(count, stream))
    {
    }

    ~AsyncGpuPointerGuard() noexcept
    {
        GpuAllocator<Scalar>::deallocateAsyncNoThrow(_ptr, _stream);
    }

    AsyncGpuPointerGuard(const AsyncGpuPointerGuard&) = delete;
    AsyncGpuPointerGuard& operator=(const AsyncGpuPointerGuard&) = delete;

    Scalar* get() const noexcept
    {
        return _ptr;
    }

    void close()
    {
        GpuAllocator<Scalar>::deallocateAsync(_ptr, _stream);
        _ptr = nullptr;
    }

private:
    cudaStream_t _stream = nullptr;
    Scalar* _ptr = nullptr;
};

#endif

template <typename Scalar>
class GpuMemoryPoolGuard
{
public:
    explicit GpuMemoryPoolGuard(bool enabled)
        : _previous_enabled(GpuAllocator<Scalar>::isMemoryPoolEnabled())
    {
        GpuAllocator<Scalar>::releaseMemoryPool();
        GpuAllocator<Scalar>::setMemoryPoolEnabled(enabled);
    }

    ~GpuMemoryPoolGuard() noexcept
    {
        try
        {
            GpuAllocator<Scalar>::releaseMemoryPool();
        }
        catch (...)
        {
        }
        GpuAllocator<Scalar>::setMemoryPoolEnabled(_previous_enabled);
    }

    GpuMemoryPoolGuard(const GpuMemoryPoolGuard&) = delete;
    GpuMemoryPoolGuard& operator=(const GpuMemoryPoolGuard&) = delete;

private:
    bool _previous_enabled = false;
};

} // namespace test
} // namespace plamatrix
