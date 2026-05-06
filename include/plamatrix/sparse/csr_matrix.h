#pragma once

#include <cstring>
#include <utility>

#include "plamatrix/core/allocator.h"
#include "plamatrix/core/device_matrix.h"
#include "plamatrix/core/error_check.h"

namespace plamatrix
{

template <typename Scalar, Device Dev>
class CSRMatrix : public DeviceMatrix<Scalar, Dev>
{
public:
    using Base = DeviceMatrix<Scalar, Dev>;

    /// Construct a CSR sparse matrix with given dimensions and non-zero count.
    /// Allocates three arrays: values (nnz), col_indices (nnz), row_offsets (rows+1).
    /// Row offsets are zero-initialized.
    /// @param rows  Number of rows
    /// @param cols  Number of columns
    /// @param nnz   Number of non-zero entries
    CSRMatrix(Index rows, Index cols, Index nnz)
        : DeviceMatrix<Scalar, Dev>(0, 0)
        , _nnz(nnz)
    {
        this->_rows = rows;
        this->_cols = cols;

        if (nnz > 0)
        {
            if constexpr (Dev == Device::CPU)
            {
                _values = CpuAllocator<Scalar>::allocate(static_cast<std::size_t>(nnz));
                _col_indices = CpuAllocator<Index>::allocate(static_cast<std::size_t>(nnz));
                std::memset(_col_indices, 0, static_cast<std::size_t>(nnz) * sizeof(Index));
            }
            else
            {
                _values = GpuAllocator<Scalar>::allocate(static_cast<std::size_t>(nnz));
                _col_indices = GpuAllocator<Index>::allocate(static_cast<std::size_t>(nnz));
                PLAMATRIX_CHECK_CUDA(
                    cudaMemset(_col_indices, 0, static_cast<std::size_t>(nnz) * sizeof(Index)));
            }
        }

        // row_offsets is always allocated (size rows+1), zero-initialized
        if constexpr (Dev == Device::CPU)
        {
            _row_offsets = CpuAllocator<Index>::allocate(static_cast<std::size_t>(rows) + 1);
            std::memset(_row_offsets, 0, (static_cast<std::size_t>(rows) + 1) * sizeof(Index));
        }
        else
        {
            _row_offsets = GpuAllocator<Index>::allocate(static_cast<std::size_t>(rows) + 1);
            PLAMATRIX_CHECK_CUDA(
                cudaMemset(_row_offsets, 0, (static_cast<std::size_t>(rows) + 1) * sizeof(Index)));
        }
    }

    /// Destructor. Frees the three allocated arrays.
    ~CSRMatrix()
    {
        releaseArrays();
    }

    // Non-copyable
    CSRMatrix(const CSRMatrix&) = delete;
    CSRMatrix& operator=(const CSRMatrix&) = delete;

    /// Move constructor. Transfers ownership of CSR arrays from source.
    /// @param other  Source matrix (left in null state)
    CSRMatrix(CSRMatrix&& other) noexcept
        : DeviceMatrix<Scalar, Dev>(std::move(other))
        , _nnz(other._nnz)
        , _values(other._values)
        , _col_indices(other._col_indices)
        , _row_offsets(other._row_offsets)
    {
        other._nnz = 0;
        other._values = nullptr;
        other._col_indices = nullptr;
        other._row_offsets = nullptr;
    }

    /// Move assignment. Releases current arrays and transfers ownership from source.
    /// @param other  Source matrix (left in null state)
    /// @return  Reference to this matrix
    CSRMatrix& operator=(CSRMatrix&& other) noexcept
    {
        if (this != &other)
        {
            releaseArrays();
            DeviceMatrix<Scalar, Dev>::operator=(std::move(other));
            _nnz = other._nnz;
            _values = other._values;
            _col_indices = other._col_indices;
            _row_offsets = other._row_offsets;
            other._nnz = 0;
            other._values = nullptr;
            other._col_indices = nullptr;
            other._row_offsets = nullptr;
        }
        return *this;
    }

    /// @return Number of non-zero entries
    Index nnz() const { return _nnz; }

    /// @return Raw pointer to values array
    Scalar* values() { return _values; }

    /// @return Raw pointer to values array (const)
    const Scalar* values() const { return _values; }

    /// @return Raw pointer to column indices array
    Index* colIndices() { return _col_indices; }

    /// @return Raw pointer to column indices array (const)
    const Index* colIndices() const { return _col_indices; }

    /// @return Raw pointer to row offsets array (size rows+1)
    Index* rowOffsets() { return _row_offsets; }

    /// @return Raw pointer to row offsets array (size rows+1, const)
    const Index* rowOffsets() const { return _row_offsets; }

private:
    /// Release allocated CSR arrays. Safe to call multiple times (nullptr-safe).
    void releaseArrays()
    {
        if (_values != nullptr)
        {
            if constexpr (Dev == Device::CPU)
            {
                CpuAllocator<Scalar>::deallocate(_values);
            }
            else
            {
                GpuAllocator<Scalar>::deallocate(_values);
            }
            _values = nullptr;
        }
        if (_col_indices != nullptr)
        {
            if constexpr (Dev == Device::CPU)
            {
                CpuAllocator<Index>::deallocate(_col_indices);
            }
            else
            {
                GpuAllocator<Index>::deallocate(_col_indices);
            }
            _col_indices = nullptr;
        }
        if (_row_offsets != nullptr)
        {
            if constexpr (Dev == Device::CPU)
            {
                CpuAllocator<Index>::deallocate(_row_offsets);
            }
            else
            {
                GpuAllocator<Index>::deallocate(_row_offsets);
            }
            _row_offsets = nullptr;
        }
    }

    Index _nnz = 0;
    Scalar* _values = nullptr;
    Index* _col_indices = nullptr;
    Index* _row_offsets = nullptr;
};

} // namespace plamatrix
