#pragma once

#include <algorithm>
#include <numeric>
#include <vector>

#include "plamatrix/core/device_matrix.h"
#include "plamatrix/core/error_check.h"
#include "plamatrix/sparse/csr_matrix.h"

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
        _row_indices.push_back(row);
        _col_indices.push_back(col);
        _values.push_back(value);
    }

    /// @return Number of non-zero entries
    Index nnz() const { return static_cast<Index>(_values.size()); }

    /// Convert COO to CSR format on the same device.
    /// On CPU: sorts triplets by (row, col), builds row offsets and copies values.
    /// On GPU: computes CSR arrays on the host then copies to the device.
    /// @return CSRMatrix with equivalent sparse representation
    CSRMatrix<Scalar, Dev> toCsr() const
    {
        Index rows = this->_rows;
        Index cols = this->_cols;
        Index nnz_val = nnz();

        if (nnz_val == 0)
        {
            CSRMatrix<Scalar, Dev> result(rows, cols, 0);
            // row_offsets is already zero-initialized by constructor
            if constexpr (Dev == Device::CPU)
            {
                for (Index r = 0; r <= rows; ++r)
                {
                    result.rowOffsets()[r] = 0;
                }
            }
            else
            {
                std::vector<Index> host_offsets(rows + 1, 0);
                PLAMATRIX_CHECK_CUDA(cudaMemcpy(result.rowOffsets(), host_offsets.data(),
                                                (rows + 1) * sizeof(Index), cudaMemcpyHostToDevice));
            }
            return result;
        }

        // Sort permutation by (row, col)
        std::vector<Index> perm(static_cast<std::size_t>(nnz_val));
        std::iota(perm.begin(), perm.end(), 0);
        std::sort(perm.begin(), perm.end(), [this](Index a, Index b) {
            if (_row_indices[static_cast<std::size_t>(a)] != _row_indices[static_cast<std::size_t>(b)])
            {
                return _row_indices[static_cast<std::size_t>(a)] < _row_indices[static_cast<std::size_t>(b)];
            }
            return _col_indices[static_cast<std::size_t>(a)] < _col_indices[static_cast<std::size_t>(b)];
        });

        // Count non-zeros per row
        std::vector<Index> row_counts(static_cast<std::size_t>(rows), 0);
        for (Index i = 0; i < nnz_val; ++i)
        {
            row_counts[static_cast<std::size_t>(_row_indices[static_cast<std::size_t>(i)])]++;
        }

        // Build row offsets
        std::vector<Index> offsets(static_cast<std::size_t>(rows) + 1);
        offsets[0] = 0;
        for (Index r = 0; r < rows; ++r)
        {
            offsets[static_cast<std::size_t>(r) + 1] =
                offsets[static_cast<std::size_t>(r)] + row_counts[static_cast<std::size_t>(r)];
        }

        if constexpr (Dev == Device::CPU)
        {
            CSRMatrix<Scalar, Device::CPU> result(rows, cols, nnz_val);

            // Track current write position per row
            std::vector<Index> pos(static_cast<std::size_t>(rows));
            std::copy_n(offsets.data(), static_cast<std::size_t>(rows), pos.data());

            // Scatter values and column indices
            for (Index i = 0; i < nnz_val; ++i)
            {
                Index src = perm[static_cast<std::size_t>(i)];
                Index r = _row_indices[static_cast<std::size_t>(src)];
                Index p = pos[static_cast<std::size_t>(r)]++;
                result.values()[p] = _values[static_cast<std::size_t>(src)];
                result.colIndices()[p] = _col_indices[static_cast<std::size_t>(src)];
            }

            // Copy row offsets
            for (Index r = 0; r <= rows; ++r)
            {
                result.rowOffsets()[r] = offsets[static_cast<std::size_t>(r)];
            }

            return result;
        }
        else
        {
            // GPU path: build CSR arrays on host, then copy to device

            // Build CPU-side value and column-index arrays in sorted order
            std::vector<Scalar> cpu_values(static_cast<std::size_t>(nnz_val));
            std::vector<Index> cpu_col_indices(static_cast<std::size_t>(nnz_val));

            std::vector<Index> pos(static_cast<std::size_t>(rows));
            std::copy_n(offsets.data(), static_cast<std::size_t>(rows), pos.data());

            for (Index i = 0; i < nnz_val; ++i)
            {
                Index src = perm[static_cast<std::size_t>(i)];
                Index r = _row_indices[static_cast<std::size_t>(src)];
                Index p = pos[static_cast<std::size_t>(r)]++;
                cpu_values[static_cast<std::size_t>(p)] = _values[static_cast<std::size_t>(src)];
                cpu_col_indices[static_cast<std::size_t>(p)] = _col_indices[static_cast<std::size_t>(src)];
            }

            CSRMatrix<Scalar, Device::GPU> result(rows, cols, nnz_val);

            PLAMATRIX_CHECK_CUDA(cudaMemcpy(result.values(), cpu_values.data(),
                                            static_cast<std::size_t>(nnz_val) * sizeof(Scalar),
                                            cudaMemcpyHostToDevice));
            PLAMATRIX_CHECK_CUDA(cudaMemcpy(result.colIndices(), cpu_col_indices.data(),
                                            static_cast<std::size_t>(nnz_val) * sizeof(Index),
                                            cudaMemcpyHostToDevice));
            PLAMATRIX_CHECK_CUDA(cudaMemcpy(result.rowOffsets(), offsets.data(),
                                            (static_cast<std::size_t>(rows) + 1) * sizeof(Index),
                                            cudaMemcpyHostToDevice));

            return result;
        }
    }

private:
    std::vector<Index> _row_indices;
    std::vector<Index> _col_indices;
    std::vector<Scalar> _values;
};

} // namespace plamatrix
