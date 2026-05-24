#include "plamatrix/dense/dense_matrix.h"

namespace plamatrix
{

template <typename Scalar>
__global__ void fillKernel(Scalar* data, Index count, Scalar value)
{
    Index idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
    {
        data[idx] = value;
    }
}

template <>
void DenseMatrix<float, Device::GPU>::fillGpuKernel(float value)
{
    Index count = this->size();
    if (count == 0)
    {
        return;
    }
    int block_size = 256;
    int grid_size = static_cast<int>((count + block_size - 1) / block_size);
    fillKernel<float><<<grid_size, block_size>>>(this->_data, count, value);
    PLAMATRIX_CHECK_CUDA(cudaGetLastError());
}

template <>
void DenseMatrix<double, Device::GPU>::fillGpuKernel(double value)
{
    Index count = this->size();
    if (count == 0)
    {
        return;
    }
    int block_size = 256;
    int grid_size = static_cast<int>((count + block_size - 1) / block_size);
    fillKernel<double><<<grid_size, block_size>>>(this->_data, count, value);
    PLAMATRIX_CHECK_CUDA(cudaGetLastError());
}

template <typename Scalar>
__global__ void transposeKernel(const Scalar* src, Scalar* dst, Index src_rows, Index src_cols)
{
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    Index j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < src_rows && j < src_cols)
    {
        dst[j + i * src_cols] = src[i + j * src_rows];
    }
}

template <>
void DenseMatrix<float, Device::GPU>::transposeGpuKernel(DenseMatrix<float, Device::GPU>& result) const
{
    if (this->size() == 0)
    {
        return;
    }
    dim3 block(16, 16);
    dim3 grid(
        (static_cast<unsigned int>(this->_rows) + block.x - 1) / block.x,
        (static_cast<unsigned int>(this->_cols) + block.y - 1) / block.y);
    transposeKernel<float><<<grid, block>>>(this->_data, result.data(), this->_rows, this->_cols);
    PLAMATRIX_CHECK_CUDA(cudaGetLastError());
}

template <>
void DenseMatrix<double, Device::GPU>::transposeGpuKernel(DenseMatrix<double, Device::GPU>& result) const
{
    if (this->size() == 0)
    {
        return;
    }
    dim3 block(16, 16);
    dim3 grid(
        (static_cast<unsigned int>(this->_rows) + block.x - 1) / block.x,
        (static_cast<unsigned int>(this->_cols) + block.y - 1) / block.y);
    transposeKernel<double><<<grid, block>>>(this->_data, result.data(), this->_rows, this->_cols);
    PLAMATRIX_CHECK_CUDA(cudaGetLastError());
}

} // namespace plamatrix
