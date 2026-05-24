#pragma once

#include <limits>
#include <stdexcept>
#include <utility>

#include "plamatrix/core/allocator.h"
#include "plamatrix/core/types.h"

namespace plamatrix
{

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
    {
        allocate(checkedElementCount(rows, cols));
    }

    /// Destructor. Frees allocated memory.
    ~DeviceMatrix()
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
    {
        other._rows = 0;
        other._cols = 0;
        other._data = nullptr;
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
            other._rows = 0;
            other._cols = 0;
            other._data = nullptr;
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

protected:
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
            _data = CpuAllocator<Scalar>::allocate(count);
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
                CpuAllocator<Scalar>::deallocateNoThrow(_data);
            }
            else
            {
                GpuAllocator<Scalar>::deallocateNoThrow(_data);
            }
            _data = nullptr;
        }
    }

    Index _rows = 0;
    Index _cols = 0;
    Scalar* _data = nullptr;
};

} // namespace plamatrix
