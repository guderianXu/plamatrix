#pragma once

#include <limits>
#include <stdexcept>
#include <utility>

#include "plamatrix/core/allocator.h"
#include "plamatrix/core/types.h"

namespace plamatrix
{

namespace detail
{

enum class HostAllocationKind
{
    Pageable,
    Pinned
};

struct AsyncGpuAllocationTag
{
};

enum class GpuAllocationKind
{
    Normal,
    StreamOrderedAsync
};

} // namespace detail

template <typename Scalar, Device Dev>
class DeviceMatrix
{
public:
    DeviceMatrix() = delete;

    /// Construct a matrix with given dimensions. Allocates memory via the device-specific allocator.
    /// @param rows  Number of rows
    /// @param cols  Number of columns
    DeviceMatrix(Index rows, Index cols)
        : _rows(rows)
        , _cols(cols)
        , _data(nullptr)
        , _host_allocation_kind(detail::HostAllocationKind::Pageable)
    {
        allocate(checkedElementCount(rows, cols));
    }

    /// Destructor. Frees allocated memory.
    ~DeviceMatrix() noexcept
    {
        release();
    }

    // Non-copyable
    DeviceMatrix(const DeviceMatrix&) = delete;
    DeviceMatrix& operator=(const DeviceMatrix&) = delete;

    /// Move constructor. Transfers ownership of data from source.
    /// @param other  Source matrix (left in null state)
    DeviceMatrix(DeviceMatrix&& other) noexcept
        : _rows(other._rows)
        , _cols(other._cols)
        , _data(other._data)
        , _host_allocation_kind(other._host_allocation_kind)
        , _gpu_allocation_kind(other._gpu_allocation_kind)
        , _gpu_allocation_stream(other._gpu_allocation_stream)
    {
        other._rows = 0;
        other._cols = 0;
        other._data = nullptr;
        other._host_allocation_kind = detail::HostAllocationKind::Pageable;
        other._gpu_allocation_kind = detail::GpuAllocationKind::Normal;
        other._gpu_allocation_stream = nullptr;
    }

    /// Move assignment. Releases current data and transfers ownership from source.
    /// @param other  Source matrix (left in null state)
    /// @return Reference to this matrix
    DeviceMatrix& operator=(DeviceMatrix&& other) noexcept
    {
        if (this != &other)
        {
            release();
            _rows = other._rows;
            _cols = other._cols;
            _data = other._data;
            _host_allocation_kind = other._host_allocation_kind;
            _gpu_allocation_kind = other._gpu_allocation_kind;
            _gpu_allocation_stream = other._gpu_allocation_stream;
            other._rows = 0;
            other._cols = 0;
            other._data = nullptr;
            other._host_allocation_kind = detail::HostAllocationKind::Pageable;
            other._gpu_allocation_kind = detail::GpuAllocationKind::Normal;
            other._gpu_allocation_stream = nullptr;
        }
        return *this;
    }

    /// @return Number of rows
    Index rows() const { return _rows; }

    /// @return Number of columns
    Index cols() const { return _cols; }

    /// @return Total number of elements (rows * cols)
    Index size() const { return _rows * _cols; }

    /// @return Raw pointer to the underlying data
    Scalar* data() { return _data; }

    /// @return Raw pointer to the underlying data (const)
    const Scalar* data() const { return _data; }

    /// @return Device type for this matrix
    static constexpr Device device() { return Dev; }

    /// @return true when this GPU matrix owns stream-ordered storage.
    bool isAsyncAllocation() const noexcept
    {
        return Dev == Device::GPU
            && _gpu_allocation_kind == detail::GpuAllocationKind::StreamOrderedAsync;
    }

    /// @return stream that owns this stream-ordered allocation, or nullptr for ordinary storage.
    cudaStream_t asyncAllocationStream() const noexcept
    {
        return isAsyncAllocation() ? _gpu_allocation_stream : nullptr;
    }

    /// Explicitly enqueue checked release for a stream-ordered GPU allocation.
    /// The retained allocation stream must remain valid through this call. On success the matrix
    /// becomes 0x0 and no longer retains stream provenance. If cudaFreeAsync fails, ownership and
    /// provenance are unchanged so the caller can retry while the stream remains valid.
    /// Empty matrices are safely normalized to 0x0, making repeated calls a no-op. A non-empty
    /// matrix created through ordinary allocation is rejected with std::logic_error.
    /// Call this before destroying the stream when release errors must be reported; the destructor
    /// is a noexcept fallback that swallows release errors and still requires a valid stream.
    void closeAsyncAllocation()
    {
        if (_data == nullptr)
        {
            _rows = 0;
            _cols = 0;
            _host_allocation_kind = detail::HostAllocationKind::Pageable;
            _gpu_allocation_kind = detail::GpuAllocationKind::Normal;
            _gpu_allocation_stream = nullptr;
            return;
        }

        if constexpr (Dev == Device::CPU)
        {
            throw std::logic_error(
                "DeviceMatrix::closeAsyncAllocation requires a stream-ordered GPU allocation");
        }
        else
        {
            if (_gpu_allocation_kind != detail::GpuAllocationKind::StreamOrderedAsync)
            {
                throw std::logic_error(
                    "DeviceMatrix::closeAsyncAllocation requires a stream-ordered GPU allocation");
            }

            GpuAllocator<Scalar>::deallocateAsync(_data, _gpu_allocation_stream);
            _rows = 0;
            _cols = 0;
            _data = nullptr;
            _host_allocation_kind = detail::HostAllocationKind::Pageable;
            _gpu_allocation_kind = detail::GpuAllocationKind::Normal;
            _gpu_allocation_stream = nullptr;
        }
    }

protected:
    DeviceMatrix(Index rows, Index cols, detail::HostAllocationKind host_allocation_kind)
        : _rows(rows)
        , _cols(cols)
        , _data(nullptr)
        , _host_allocation_kind(host_allocation_kind)
    {
        allocate(checkedElementCount(rows, cols));
    }

    DeviceMatrix(Index rows, Index cols, detail::AsyncGpuAllocationTag, cudaStream_t stream)
        : _rows(rows)
        , _cols(cols)
        , _data(nullptr)
        , _host_allocation_kind(detail::HostAllocationKind::Pageable)
        , _gpu_allocation_kind(detail::GpuAllocationKind::StreamOrderedAsync)
        , _gpu_allocation_stream(stream)
    {
        static_assert(Dev == Device::GPU,
                      "AsyncGpuAllocationTag is only available for GPU matrices");
        _data = GpuAllocator<Scalar>::allocateAsync(checkedElementCount(rows, cols), stream);
    }

    /// Allocate memory for `count` elements using the device-specific allocator.
    /// @param count  Number of elements to allocate
    static std::size_t checkedElementCount(Index rows, Index cols)
    {
        if (rows < 0 || cols < 0)
        {
            throw std::invalid_argument("DeviceMatrix dimensions must be non-negative");
        }
        if (rows == 0 || cols == 0)
        {
            return 0;
        }
        if (cols > std::numeric_limits<Index>::max() / rows)
        {
            throw std::overflow_error("DeviceMatrix element count overflows Index");
        }
        return static_cast<std::size_t>(rows * cols);
    }

    void allocate(std::size_t count)
    {
        if (count == 0)
        {
            _data = nullptr;
            return;
        }
        if constexpr (Dev == Device::CPU)
        {
            if (_host_allocation_kind == detail::HostAllocationKind::Pinned)
            {
                _data = PinnedCpuAllocator<Scalar>::allocate(count);
            }
            else
            {
                _data = CpuAllocator<Scalar>::allocate(count);
            }
        }
        else
        {
            _data = GpuAllocator<Scalar>::allocate(count);
        }
    }

    /// Release allocated memory. Safe to call multiple times (nullptr-safe).
    void release() noexcept
    {
        if (_data != nullptr)
        {
            if constexpr (Dev == Device::CPU)
            {
                if (_host_allocation_kind == detail::HostAllocationKind::Pinned)
                {
                    PinnedCpuAllocator<Scalar>::deallocateNoThrow(_data);
                }
                else
                {
                    CpuAllocator<Scalar>::deallocateNoThrow(_data);
                }
            }
            else
            {
                if (_gpu_allocation_kind == detail::GpuAllocationKind::StreamOrderedAsync)
                {
                    GpuAllocator<Scalar>::deallocateAsyncNoThrow(
                        _data, _gpu_allocation_stream);
                }
                else
                {
                    GpuAllocator<Scalar>::deallocateNoThrow(
                        _data, static_cast<std::size_t>(size()));
                }
            }
            _data = nullptr;
        }
        _gpu_allocation_kind = detail::GpuAllocationKind::Normal;
        _gpu_allocation_stream = nullptr;
    }

    Index _rows = 0;
    Index _cols = 0;
    Scalar* _data = nullptr;
    detail::HostAllocationKind _host_allocation_kind = detail::HostAllocationKind::Pageable;
    detail::GpuAllocationKind _gpu_allocation_kind = detail::GpuAllocationKind::Normal;
    cudaStream_t _gpu_allocation_stream = nullptr;
};

} // namespace plamatrix
