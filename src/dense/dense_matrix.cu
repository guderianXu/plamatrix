#include <limits>
#include <sstream>
#include <stdexcept>

#include "plamatrix/dense/dense_matrix.h"

namespace plamatrix
{

template <typename Scalar>
__global__ void fillKernel(Scalar* data, Index count, Scalar value)
{
    Index idx = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < count)
    {
        data[idx] = value;
    }
}

namespace
{

int checkedCudaGrid1D(Index item_count, int block_size, const char* op)
{
    Index block_count = (item_count + block_size - 1) / block_size;
    if (block_count > static_cast<Index>(std::numeric_limits<int>::max()))
    {
        std::ostringstream oss;
        oss << op << ": item count exceeds CUDA grid range";
        throw std::runtime_error(oss.str());
    }
    return static_cast<int>(block_count);
}

unsigned int checkedCudaGridExtent(Index extent, unsigned int block_extent, const char* name)
{
    Index block_count = (extent + static_cast<Index>(block_extent) - 1) / static_cast<Index>(block_extent);
    if (block_count > static_cast<Index>(std::numeric_limits<unsigned int>::max()))
    {
        std::ostringstream oss;
        oss << name << " exceeds CUDA grid extent range: " << block_count;
        throw std::runtime_error(oss.str());
    }
    return static_cast<unsigned int>(block_count);
}

} // anonymous namespace

#ifdef PLAMATRIX_USE_FLOAT
template <>
void DenseMatrix<float, Device::GPU>::fillGpuKernel(float value)
{
    Index count = this->size();
    if (count == 0)
    {
        return;
    }
    constexpr int block_size = 256;
    int grid_size = checkedCudaGrid1D(count, block_size, "DenseMatrix::fill");
    fillKernel<float><<<grid_size, block_size>>>(this->_data, count, value);
    PLAMATRIX_CHECK_CUDA(cudaGetLastError());
}
#endif

#ifdef PLAMATRIX_USE_DOUBLE
template <>
void DenseMatrix<double, Device::GPU>::fillGpuKernel(double value)
{
    Index count = this->size();
    if (count == 0)
    {
        return;
    }
    constexpr int block_size = 256;
    int grid_size = checkedCudaGrid1D(count, block_size, "DenseMatrix::fill");
    fillKernel<double><<<grid_size, block_size>>>(this->_data, count, value);
    PLAMATRIX_CHECK_CUDA(cudaGetLastError());
}
#endif

template <>
void DenseMatrix<int, Device::GPU>::fillGpuKernel(int value)
{
    Index count = this->size();
    if (count == 0)
    {
        return;
    }
    constexpr int block_size = 256;
    int grid_size = checkedCudaGrid1D(count, block_size, "DenseMatrix::fill");
    fillKernel<int><<<grid_size, block_size>>>(this->_data, count, value);
    PLAMATRIX_CHECK_CUDA(cudaGetLastError());
}

template <typename Scalar>
__global__ void transposeKernel(const Scalar* src, Scalar* dst, Index src_rows, Index src_cols)
{
    Index i = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
    Index j = static_cast<Index>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (i < src_rows && j < src_cols)
    {
        dst[j + i * src_cols] = src[i + j * src_rows];
    }
}

#ifdef PLAMATRIX_USE_FLOAT
template <>
void DenseMatrix<float, Device::GPU>::transposeGpuKernel(DenseMatrix<float, Device::GPU>& result) const
{
    if (this->size() == 0)
    {
        return;
    }
    dim3 block(16, 16);
    dim3 grid(
        checkedCudaGridExtent(this->_rows, block.x, "DenseMatrix::transpose rows"),
        checkedCudaGridExtent(this->_cols, block.y, "DenseMatrix::transpose cols"));
    transposeKernel<float><<<grid, block>>>(this->_data, result.data(), this->_rows, this->_cols);
    PLAMATRIX_CHECK_CUDA(cudaGetLastError());
}
#endif

#ifdef PLAMATRIX_USE_DOUBLE
template <>
void DenseMatrix<double, Device::GPU>::transposeGpuKernel(DenseMatrix<double, Device::GPU>& result) const
{
    if (this->size() == 0)
    {
        return;
    }
    dim3 block(16, 16);
    dim3 grid(
        checkedCudaGridExtent(this->_rows, block.x, "DenseMatrix::transpose rows"),
        checkedCudaGridExtent(this->_cols, block.y, "DenseMatrix::transpose cols"));
    transposeKernel<double><<<grid, block>>>(this->_data, result.data(), this->_rows, this->_cols);
    PLAMATRIX_CHECK_CUDA(cudaGetLastError());
}
#endif

} // namespace plamatrix
