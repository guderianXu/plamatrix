#pragma once

#include <cstddef>

#include "plamatrix/internal/core/error.h"
#include "plamatrix/internal/core/execution_context.h"
#include "plamatrix/internal/device/device_matrix.h"
#include "plamatrix/internal/device/detail/resident_buffer.h"

namespace plamatrix::internal
{

inline namespace v1
{

template <typename Scalar> class ResidentVector
{
public:
    /// Copy an Eigen-style row or column vector into this context's storage.
    template <int Rows, int Cols>
    static ResidentVector copyFrom(const Matrix<Scalar, Rows, Cols>& host, ExecutionContext& context)
    {
        if (host.rows() != 1 && host.cols() != 1)
        {
            throw Error(ErrorCode::InvalidArgument, "ResidentVector copy requires a vector", context.backend());
        }
        ResidentVector result(host.size(), context);
        result.copyFromHost(host.data(), static_cast<std::size_t>(host.size()));
        return result;
    }

    ResidentVector(Index size, ExecutionContext& context) : _context(&context), _values(checkedSize(size), context)
    {
        if (size < 0)
        {
            throw std::invalid_argument("ResidentVector size must be non-negative");
        }
    }

    ResidentVector(ResidentVector&&) noexcept = default;
    ResidentVector& operator=(ResidentVector&&) noexcept = default;
    ResidentVector(const ResidentVector&) = delete;
    ResidentVector& operator=(const ResidentVector&) = delete;

    Index size() const noexcept { return static_cast<Index>(_values.size()); }
    Scalar* data() { return _values.data(); }
    const Scalar* data() const { return _values.data(); }

    void copyFromHost(const Scalar* source, std::size_t count)
    {
        _values.copyFromHost(source, count);
    }

    void copyToHost(Scalar* destination, std::size_t count) const
    {
        _values.copyToHost(destination, count);
    }

    Matrix<Scalar, Dynamic, 1> toHostVector() const
    {
        Matrix<Scalar, Dynamic, 1> result(size());
        copyToHost(result.data(), static_cast<std::size_t>(size()));
        return result;
    }
    ExecutionContext& context() const noexcept { return *_context; }
    MemorySpace memorySpace() const noexcept { return memorySpaceFor(_context->backend()); }

    void validateContext(const ExecutionContext& context) const
    {
        if (_context != &context)
        {
            throw Error(
                ErrorCode::InvalidState, "ResidentVector belongs to a different ExecutionContext", context.backend());
        }
    }

private:
    friend struct opencl::NativeAccess;
    friend struct vulkan::NativeAccess;
    static std::size_t checkedSize(Index size)
    {
        if (size < 0)
        {
            throw std::invalid_argument("ResidentVector size must be non-negative");
        }
        return static_cast<std::size_t>(size);
    }

    ExecutionContext* _context;
    resident_detail::ResidentBuffer<Scalar> _values;
};

} // namespace v1

} // namespace plamatrix::internal
