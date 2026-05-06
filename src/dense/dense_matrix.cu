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
    int block_size = 256;
    int grid_size = static_cast<int>((count + block_size - 1) / block_size);
    fillKernel<float><<<grid_size, block_size>>>(this->_data, count, value);
    PLAMATRIX_CHECK_CUDA(cudaGetLastError());
}

template <>
void DenseMatrix<double, Device::GPU>::fillGpuKernel(double value)
{
    Index count = this->size();
    int block_size = 256;
    int grid_size = static_cast<int>((count + block_size - 1) / block_size);
    fillKernel<double><<<grid_size, block_size>>>(this->_data, count, value);
    PLAMATRIX_CHECK_CUDA(cudaGetLastError());
}

} // namespace plamatrix
