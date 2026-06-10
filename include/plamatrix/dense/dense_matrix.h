#pragma once

#include <algorithm>
#include <cstring>
#include <sstream>
#include <stdexcept>

#include "plamatrix/core/device_matrix.h"

namespace plamatrix
{

template <typename Scalar, Device Dev>
class DenseMatrix : public DeviceMatrix<Scalar, Dev>
{
public:
    using Base = DeviceMatrix<Scalar, Dev>;

    /// Default constructor. Creates an empty matrix (0x0).
    DenseMatrix()
        : DeviceMatrix<Scalar, Dev>(0, 0)
    {
    }

    /// Construct a matrix with given dimensions and zero-initialize memory.
    /// Uses std::memset for CPU, cudaMemset for GPU.
    /// @param rows  Number of rows
    /// @param cols  Number of columns
    DenseMatrix(Index rows, Index cols)
        : DeviceMatrix<Scalar, Dev>(rows, cols)
    {
        if (this->size() == 0)
        {
            return;
        }
        if constexpr (Dev == Device::CPU)
        {
            std::memset(this->_data, 0, static_cast<std::size_t>(this->size()) * sizeof(Scalar));
        }
        else
        {
            PLAMATRIX_CHECK_CUDA(
                cudaMemset(this->_data, 0, static_cast<std::size_t>(this->size()) * sizeof(Scalar)));
        }
    }

    /// Move constructor (defaulted).
    /// @param other  Source matrix
    DenseMatrix(DenseMatrix&& other) noexcept = default;

    /// Move assignment (defaulted).
    /// @param other  Source matrix
    /// @return  Reference to this matrix
    DenseMatrix& operator=(DenseMatrix&& other) noexcept = default;

    /// Element access — CPU only, column-major layout: data[row + col * rows].
    /// @param row  Row index (0-based)
    /// @param col  Column index (0-based)
    /// @return  Const reference to the element at (row, col)
    Scalar operator()(Index row, Index col) const
    {
        static_assert(Dev == Device::CPU, "operator() is only available on CPU matrices");
        return this->_data[checkedOffset(row, col)];
    }

    /// Element access — CPU only, column-major layout: data[row + col * rows].
    /// @param row  Row index (0-based)
    /// @param col  Column index (0-based)
    /// @return  Mutable reference to the element at (row, col)
    Scalar& operator()(Index row, Index col)
    {
        static_assert(Dev == Device::CPU, "operator() is only available on CPU matrices");
        return this->_data[checkedOffset(row, col)];
    }

    /// Set a single element value. Works on both CPU and GPU.
    /// @param row    Row index (0-based)
    /// @param col    Column index (0-based)
    /// @param value  Value to set
    void setValue(Index row, Index col, Scalar value)
    {
        Index offset = checkedOffset(row, col);
        if constexpr (Dev == Device::CPU)
        {
            this->_data[offset] = value;
        }
        else
        {
            PLAMATRIX_CHECK_CUDA(
                cudaMemcpy(this->_data + offset, &value, sizeof(Scalar), cudaMemcpyHostToDevice));
        }
    }

    /// Get a single element value. Works on both CPU and GPU.
    /// @param row  Row index (0-based)
    /// @param col  Column index (0-based)
    /// @return  Value at (row, col)
    Scalar getValue(Index row, Index col) const
    {
        Index offset = checkedOffset(row, col);
        if constexpr (Dev == Device::CPU)
        {
            return this->_data[offset];
        }
        else
        {
            Scalar host_val;
            PLAMATRIX_CHECK_CUDA(
                cudaMemcpy(&host_val, this->_data + offset, sizeof(Scalar), cudaMemcpyDeviceToHost));
            return host_val;
        }
    }

    /// Fill all elements with the given value.
    /// Uses std::fill_n for CPU, cudaMemset for zero or GPU kernel for non-zero on GPU.
    /// @param value  Value to fill all elements with
    void fill(Scalar value)
    {
        if (this->size() == 0)
        {
            return;
        }
        if constexpr (Dev == Device::CPU)
        {
            std::fill_n(this->_data, this->size(), value);
        }
        else
        {
            if (value == Scalar(0))
            {
                PLAMATRIX_CHECK_CUDA(
                    cudaMemset(this->_data, 0, static_cast<std::size_t>(this->size()) * sizeof(Scalar)));
            }
            else
            {
                fillGpuKernel(value);
            }
        }
    }

    /// Transfer a GPU matrix to CPU.
    /// @return  CPU copy of this matrix
    DenseMatrix<Scalar, Device::CPU> toCpu() const
    {
        static_assert(Dev == Device::GPU, "toCpu() is only available on GPU matrices");
        DenseMatrix<Scalar, Device::CPU> result(this->_rows, this->_cols);
        if (this->size() == 0)
        {
            return result;
        }
        PLAMATRIX_CHECK_CUDA(
            cudaMemcpy(result.data(), this->_data, static_cast<std::size_t>(this->size()) * sizeof(Scalar),
                       cudaMemcpyDeviceToHost));
        return result;
    }

    /// Transfer a CPU matrix to GPU.
    /// @return  GPU copy of this matrix
    DenseMatrix<Scalar, Device::GPU> toGpu() const
    {
        static_assert(Dev == Device::CPU, "toGpu() is only available on CPU matrices");
        DenseMatrix<Scalar, Device::GPU> result(this->_rows, this->_cols);
        if (this->size() == 0)
        {
            return result;
        }
        PLAMATRIX_CHECK_CUDA(
            cudaMemcpy(result.data(), this->_data, static_cast<std::size_t>(this->size()) * sizeof(Scalar),
                       cudaMemcpyHostToDevice));
        return result;
    }

    /// Compute the transpose of this matrix.
    /// On CPU, uses nested loops. On GPU, launches a 2D transpose kernel.
    /// @return  New matrix with dimensions (cols x rows)
    DenseMatrix transpose() const
    {
        DenseMatrix result(this->_cols, this->_rows);
        if (this->size() == 0)
        {
            return result;
        }
        if constexpr (Dev == Device::CPU)
        {
            for (Index j = 0; j < this->_cols; ++j)
            {
                for (Index i = 0; i < this->_rows; ++i)
                {
                    result(j, i) = (*this)(i, j);
                }
            }
        }
        else
        {
            transposeGpuKernel(result);
        }
        return result;
    }

private:
    Index checkedOffset(Index row, Index col) const
    {
        if (row < 0 || row >= this->_rows || col < 0 || col >= this->_cols)
        {
            std::ostringstream oss;
            oss << "DenseMatrix index out of range: (" << row << ", " << col
                << ") for matrix " << this->_rows << "x" << this->_cols;
            throw std::out_of_range(oss.str());
        }
        return row + col * this->_rows;
    }

#ifdef PLAMATRIX_NO_CUDA
    void fillGpuKernel(Scalar)
    {
        throw std::runtime_error("DenseMatrix::fill: non-zero GPU fill requires PLAMATRIX_WITH_CUDA=ON");
    }

    void transposeGpuKernel(DenseMatrix&) const
    {
        throw std::runtime_error("DenseMatrix::transpose: GPU transpose requires PLAMATRIX_WITH_CUDA=ON");
    }
#else
    /// GPU fill kernel wrapper (implemented in dense_matrix.cu).
    /// @param value  Non-zero fill value
    void fillGpuKernel(Scalar value);

    /// GPU transpose kernel wrapper (implemented in dense_matrix.cu).
    /// @param result  Output matrix (cols x rows dimensions, pre-allocated)
    void transposeGpuKernel(DenseMatrix& result) const;
#endif
};

} // namespace plamatrix
