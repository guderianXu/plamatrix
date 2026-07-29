#pragma once

#include <stdexcept>
#include <vector>

#include "plamatrix/core/device_matrix.h"
#include "plamatrix/sparse/csr_matrix.h"
#include "plamatrix/sparse/sparse_ops.h"

namespace plamatrix
{

template <typename Scalar, Device Dev>
class COOMatrix : public DeviceMatrix<Scalar, Dev>
{
public:
    using Base = DeviceMatrix<Scalar, Dev>;

    /// Construct a COO sparse matrix with given dimensions.
    /// Element storage (triplets) starts empty.
    /// @param rows  Number of rows
    /// @param cols  Number of columns
    COOMatrix(Index rows, Index cols)
        : DeviceMatrix<Scalar, Dev>(0, 0)
    {
        if (rows < 0 || cols < 0)
        {
            throw std::invalid_argument("COOMatrix dimensions must be non-negative");
        }
        this->_rows = rows;
        this->_cols = cols;
    }

    /// Add a non-zero triplet to the matrix (CPU-only, push_back).
    /// Triplets are stored in insertion order and sorted during toCsr().
    /// @param row    Row index (0-based)
    /// @param col    Column index (0-based)
    /// @param value  Non-zero value
    void add(Index row, Index col, Scalar value)
    {
        if (row < 0 || row >= this->_rows || col < 0 || col >= this->_cols)
        {
            throw std::out_of_range("COOMatrix triplet index is out of bounds");
        }
        _row_indices.push_back(row);
        _col_indices.push_back(col);
        _values.push_back(value);
    }

    /// @return Number of non-zero entries
    Index nnz() const { return static_cast<Index>(_values.size()); }

    /// Convert COO to CSR format on the same device.
    /// On CPU: stably sorts triplets by (row, col) and combines duplicate coordinates.
    /// On GPU: computes CSR arrays on the host then copies to the device.
    /// @return CSRMatrix with equivalent sparse representation
    CSRMatrix<Scalar, Dev> toCsr() const
    {
        if constexpr (Dev == Device::CPU)
        {
            return cooToCsr(this->_rows, this->_cols, _row_indices, _col_indices, _values);
        }
        else
        {
            return cooToCsr(
                this->_rows, this->_cols, _row_indices, _col_indices, _values).toGpu();
        }
    }

private:
    std::vector<Index> _row_indices;
    std::vector<Index> _col_indices;
    std::vector<Scalar> _values;
};

} // namespace plamatrix
