#pragma once

#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "plamatrix/internal/core/allocator.h"
#include "plamatrix/internal/core/error.h"
#include "plamatrix/internal/core/execution_context.h"
#include "plamatrix/internal/device/detail/resident_opencl.h"
#include "plamatrix/internal/device/detail/resident_vulkan.h"

#ifdef PLAMATRIX_WITH_CUDA
#include "plamatrix/internal/cuda/native.h"
#endif

namespace plamatrix::internal
{

inline namespace v1
{

namespace resident_detail
{

template <typename Scalar> class ResidentBuffer
{
public:
    ResidentBuffer() noexcept = default;

    ResidentBuffer(std::size_t count, ExecutionContext& context)
        : _count(count)
        , _context(&context)
    {
        allocate();
    }

    ~ResidentBuffer() noexcept
    {
        release();
    }

    ResidentBuffer(const ResidentBuffer&) = delete;
    ResidentBuffer& operator=(const ResidentBuffer&) = delete;

    ResidentBuffer(ResidentBuffer&& other) noexcept
        : _count(other._count)
        , _context(other._context)
        , _data(other._data)
        , _native(other._native)
    {
        other._count = 0;
        other._context = nullptr;
        other._data = nullptr;
        other._native = nullptr;
    }

    ResidentBuffer& operator=(ResidentBuffer&& other) noexcept
    {
        if (this != &other)
        {
            release();
            _count = other._count;
            _context = other._context;
            _data = other._data;
            _native = other._native;
            other._count = 0;
            other._context = nullptr;
            other._data = nullptr;
            other._native = nullptr;
        }
        return *this;
    }

    std::size_t size() const noexcept { return _count; }
    Scalar* data()
    {
        checkHostAccessiblePointer();
        return _data;
    }
    const Scalar* data() const
    {
        checkHostAccessiblePointer();
        return _data;
    }
    void* nativeHandle() const noexcept { return _native; }

    void copyFromResident(const ResidentBuffer& source)
    {
        if (_context != source._context || _count != source._count)
        {
            throw Error(ErrorCode::InvalidArgument, "ResidentBuffer copy requires matching context and size");
        }
        if (this == &source || _count == 0)
        {
            return;
        }
        const auto bytes = bytesFor(_count);
        switch (_context->backend())
        {
        case Backend::Cpu:
            std::memcpy(_data, source._data, bytes);
            return;
#ifdef PLAMATRIX_WITH_CUDA
        case Backend::Cuda:
            checkCuda(cudaSetDevice(static_cast<int>(_context->device().index)));
            checkCuda(cudaMemcpyAsync(
                _data, source._data, bytes, cudaMemcpyDeviceToDevice, cuda::NativeAccess::stream(*_context)));
            _context->synchronize();
            return;
#endif
#ifdef PLAMATRIX_WITH_OPENCL
        case Backend::OpenCl:
            copyResidentOpenCl(*_context, source._native, _native, bytes);
            return;
#endif
#ifdef PLAMATRIX_WITH_VULKAN
        case Backend::Vulkan:
            copyResidentVulkan(*_context, source._native, _native, bytes);
            return;
#endif
        default:
            throw Error(
                ErrorCode::UnsupportedBackend, "ResidentBuffer copy backend is unavailable", _context->backend());
        }
    }

    void copyFromHost(const Scalar* source, std::size_t count)
    {
        checkCount(count);
        if (count == 0)
        {
            return;
        }
        if (source == nullptr)
        {
            throw Error(ErrorCode::InvalidArgument, "ResidentBuffer source is null", _context->backend());
        }
        if (_context->backend() == Backend::Cpu)
        {
            std::memcpy(_data, source, bytesFor(count));
            return;
        }
#ifdef PLAMATRIX_WITH_CUDA
        if (_context->backend() == Backend::Cuda)
        {
            checkCuda(cudaSetDevice(static_cast<int>(_context->device().index)));
            checkCuda(cudaMemcpyAsync(_data, source, bytesFor(count), cudaMemcpyHostToDevice,
                                      cuda::NativeAccess::stream(*_context)));
            _context->synchronize();
            return;
        }
#endif
        if (_context->backend() == Backend::OpenCl)
        {
#ifdef PLAMATRIX_WITH_OPENCL
            copyFromHostOpenCl(*_context, _native, source, bytesFor(count));
            return;
#endif
        }
        if (_context->backend() == Backend::Vulkan)
        {
#ifdef PLAMATRIX_WITH_VULKAN
            copyFromHostVulkan(*_context, _native, source, bytesFor(count));
            return;
#endif
        }
        throw Error(ErrorCode::UnsupportedBackend,
                    "ResidentBuffer host copy is unsupported for this backend",
                    _context->backend());
    }

    void copyToHost(Scalar* destination, std::size_t count) const
    {
        checkCount(count);
        if (count == 0)
        {
            return;
        }
        if (destination == nullptr)
        {
            throw Error(ErrorCode::InvalidArgument, "ResidentBuffer destination is null", _context->backend());
        }
        if (_context->backend() == Backend::Cpu)
        {
            std::memcpy(destination, _data, bytesFor(count));
            return;
        }
#ifdef PLAMATRIX_WITH_CUDA
        if (_context->backend() == Backend::Cuda)
        {
            checkCuda(cudaSetDevice(static_cast<int>(_context->device().index)));
            checkCuda(cudaMemcpyAsync(destination, _data, bytesFor(count), cudaMemcpyDeviceToHost,
                                      cuda::NativeAccess::stream(*_context)));
            _context->synchronize();
            return;
        }
#endif
        if (_context->backend() == Backend::OpenCl)
        {
#ifdef PLAMATRIX_WITH_OPENCL
            copyToHostOpenCl(*_context, _native, destination, bytesFor(count));
            return;
#endif
        }
        if (_context->backend() == Backend::Vulkan)
        {
#ifdef PLAMATRIX_WITH_VULKAN
            copyToHostVulkan(*_context, _native, destination, bytesFor(count));
            return;
#endif
        }
        throw Error(ErrorCode::UnsupportedBackend,
                    "ResidentBuffer host copy is unsupported for this backend",
                    _context->backend());
    }

private:
    void checkHostAccessiblePointer() const
    {
        if (_context != nullptr &&
            (_context->backend() == Backend::OpenCl || _context->backend() == Backend::Vulkan))
        {
            throw Error(ErrorCode::UnsupportedOperation,
                        "Resident memory has no host-addressable pointer; use explicit copies",
                        _context->backend());
        }
    }

    static std::size_t bytesFor(std::size_t count)
    {
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(Scalar))
        {
            throw Error(ErrorCode::InvalidArgument,
                        "ResidentBuffer allocation size overflows size_t");
        }
        return count * sizeof(Scalar);
    }

    void checkCount(std::size_t count) const
    {
        if (count > _count)
        {
            throw Error(ErrorCode::InvalidArgument,
                        "ResidentBuffer copy exceeds the allocated element count",
                        _context == nullptr ? Backend::Cpu : _context->backend());
        }
    }

    void allocate()
    {
        if (_count == 0)
        {
            return;
        }
        try
        {
            if (_context->backend() == Backend::Cpu)
            {
                _data = CpuAllocator<Scalar>::allocate(_count);
                return;
            }
            if (_context->backend() == Backend::Cuda)
            {
#ifdef PLAMATRIX_WITH_CUDA
                checkCuda(cudaSetDevice(static_cast<int>(_context->device().index)));
                _data = GpuAllocator<Scalar>::allocate(_count);
                return;
#else
                throw Error(
                    ErrorCode::BackendUnavailable, "PlaMatrix was built without CUDA support", _context->backend());
#endif
            }
            if (_context->backend() == Backend::OpenCl)
            {
#ifdef PLAMATRIX_WITH_OPENCL
                _native = allocateOpenCl(*_context, bytesFor(_count));
                return;
#else
                throw Error(ErrorCode::BackendUnavailable,
                            "PlaMatrix was built without OpenCL support",
                            _context->backend());
#endif
            }
            if (_context->backend() == Backend::Vulkan)
            {
#ifdef PLAMATRIX_WITH_VULKAN
                _native = allocateVulkan(*_context, bytesFor(_count));
                return;
#else
                throw Error(ErrorCode::BackendUnavailable,
                            "PlaMatrix was built without Vulkan support",
                            _context->backend());
#endif
            }
            throw Error(ErrorCode::UnsupportedBackend,
                        "ResidentBuffer storage is unsupported for this backend",
                        _context->backend());
        }
        catch (const Error&)
        {
            throw;
        }
        catch (const std::exception& error)
        {
            throw Error(ErrorCode::BackendFailure, error.what(), _context->backend());
        }
    }

    void release() noexcept
    {
        if (_data == nullptr && _native == nullptr)
        {
            return;
        }
        if (_context != nullptr && _context->backend() == Backend::Cuda)
        {
#ifdef PLAMATRIX_WITH_CUDA
            GpuAllocator<Scalar>::deallocateNoThrow(_data, _count);
#endif
        }
        else if (_native != nullptr)
        {
            if (_context->backend() == Backend::Vulkan)
            {
#ifdef PLAMATRIX_WITH_VULKAN
                releaseVulkan(_native);
#endif
            }
            else
            {
#ifdef PLAMATRIX_WITH_OPENCL
                releaseOpenCl(_native);
#endif
            }
        }
        else
        {
            CpuAllocator<Scalar>::deallocateNoThrow(_data);
        }
        _data = nullptr;
        _native = nullptr;
        _count = 0;
        _context = nullptr;
    }

    static void checkCuda(cudaError_t status)
    {
        if (status != cudaSuccess)
        {
#ifdef PLAMATRIX_WITH_CUDA
            throw Error(ErrorCode::BackendFailure,
                        cudaGetErrorString(status),
                        Backend::Cuda,
                        static_cast<int>(status));
#else
            throw Error(ErrorCode::BackendFailure, "CUDA operation failed", Backend::Cuda);
#endif
        }
    }

    std::size_t _count = 0;
    ExecutionContext* _context = nullptr;
    Scalar* _data = nullptr;
    void* _native = nullptr;
};

} // namespace resident_detail

} // namespace v1

} // namespace plamatrix::internal
