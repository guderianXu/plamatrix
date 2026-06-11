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
        zeroInitialize();
    }

    /// Construct a CPU matrix backed by pinned/page-locked host memory.
    /// @param rows  Number of rows
    /// @param cols  Number of columns
    /// @return  CPU matrix suitable for asynchronous CUDA transfers
    static DenseMatrix pinned(Index rows, Index cols)
    {
        static_assert(Dev == Device::CPU, "pinned() is only available for CPU matrices");
        return DenseMatrix(rows, cols, detail::HostAllocationKind::Pinned);
    }

    /// Move constructor (defaulted).
    /// @param other  Source matrix
    DenseMatrix(DenseMatrix&& other) noexcept = default;

    /// Move assignment (defaulted).
    /// @param other  Source matrix
    /// @return  Reference to this matrix
    DenseMatrix& operator=(DenseMatrix&& other) noexcept = default;

    /// @return true if this CPU matrix uses pinned/page-locked host memory.
    bool isPinnedHost() const
    {
        if constexpr (Dev == Device::CPU)
        {
            return this->_host_allocation_kind == detail::HostAllocationKind::Pinned;
        }
        else
        {
            return false;
        }
    }

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

    /// Asynchronously transfer a GPU matrix to CPU on the given CUDA stream.
    /// @param stream  CUDA stream used for cudaMemcpyAsync
    /// @return  CPU copy of this matrix; caller must synchronize stream before reading it
    DenseMatrix<Scalar, Device::CPU> toCpuAsync(cudaStream_t stream = nullptr) const
    {
        static_assert(Dev == Device::GPU, "toCpuAsync() is only available on GPU matrices");
        DenseMatrix<Scalar, Device::CPU> result(this->_rows, this->_cols);
        copyToCpuAsync(result, stream);
        return result;
    }

    /// Asynchronously transfer this GPU matrix into an existing CPU matrix.
    /// @param output  CPU output matrix with the same dimensions as this matrix
    /// @param stream  CUDA stream used for cudaMemcpyAsync
    void copyToCpuAsync(DenseMatrix<Scalar, Device::CPU>& output, cudaStream_t stream = nullptr) const
    {
        static_assert(Dev == Device::GPU, "copyToCpuAsync() is only available on GPU matrices");
        checkTransferOutputDimensions("copyToCpuAsync", output.rows(), output.cols());
        if (this->size() == 0)
        {
            return;
        }
        PLAMATRIX_CHECK_CUDA(
            cudaMemcpyAsync(output.data(), this->_data, static_cast<std::size_t>(this->size()) * sizeof(Scalar),
                            cudaMemcpyDeviceToHost, stream));
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

    /// Asynchronously transfer a CPU matrix to GPU on the given CUDA stream.
    /// @param stream  CUDA stream used for cudaMemcpyAsync
    /// @return  GPU copy of this matrix; caller must synchronize stream before reading it on another stream
    DenseMatrix<Scalar, Device::GPU> toGpuAsync(cudaStream_t stream = nullptr) const
    {
        static_assert(Dev == Device::CPU, "toGpuAsync() is only available on CPU matrices");
        DenseMatrix<Scalar, Device::GPU> result(this->_rows, this->_cols);
        copyToGpuAsync(result, stream);
        return result;
    }

    /// Asynchronously transfer this CPU matrix into an existing GPU matrix.
    /// @param output  GPU output matrix with the same dimensions as this matrix
    /// @param stream  CUDA stream used for cudaMemcpyAsync
    void copyToGpuAsync(DenseMatrix<Scalar, Device::GPU>& output, cudaStream_t stream = nullptr) const
    {
        static_assert(Dev == Device::CPU, "copyToGpuAsync() is only available on CPU matrices");
        checkTransferOutputDimensions("copyToGpuAsync", output.rows(), output.cols());
        if (this->size() == 0)
        {
            return;
        }
        PLAMATRIX_CHECK_CUDA(
            cudaMemcpyAsync(output.data(), this->_data, static_cast<std::size_t>(this->size()) * sizeof(Scalar),
                            cudaMemcpyHostToDevice, stream));
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
    DenseMatrix(Index rows, Index cols, detail::HostAllocationKind host_allocation_kind)
        : DeviceMatrix<Scalar, Dev>(rows, cols, host_allocation_kind)
    {
        zeroInitialize();
    }

    void zeroInitialize()
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

    void checkTransferOutputDimensions(const char* op, Index rows, Index cols) const
    {
        if (rows != this->_rows || cols != this->_cols)
        {
            std::ostringstream oss;
            oss << "DenseMatrix::" << op << " output dimension mismatch: output is "
                << rows << "x" << cols << ", expected " << this->_rows << "x" << this->_cols;
            throw std::runtime_error(oss.str());
        }
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
