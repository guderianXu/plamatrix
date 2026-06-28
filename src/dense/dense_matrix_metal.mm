#include "plamatrix/dense/dense_matrix.h"

#include "plamatrix/core/metal_context.h"

namespace plamatrix
{

#ifdef PLAMATRIX_USE_FLOAT
template <>
void DenseMatrix<float, Device::GPU>::fillGpuKernel(float value)
{
    detail::metalFillFloat(this->_data, this->size(), value);
}

template <>
void DenseMatrix<float, Device::GPU>::transposeGpuKernel(DenseMatrix<float, Device::GPU>& result) const
{
    detail::metalTransposeFloat(this->_data, result.data(), this->_rows, this->_cols);
}
#endif

#ifdef PLAMATRIX_USE_DOUBLE
template <>
void DenseMatrix<double, Device::GPU>::fillGpuKernel(double value)
{
    detail::metalFillDoubleFallback(this->_data, this->size(), value);
}

template <>
void DenseMatrix<double, Device::GPU>::transposeGpuKernel(DenseMatrix<double, Device::GPU>& result) const
{
    detail::metalTransposeDoubleFallback(this->_data, result.data(), this->_rows, this->_cols);
}
#endif

template <>
void DenseMatrix<int, Device::GPU>::fillGpuKernel(int value)
{
    int* data = this->_data;
    for (Index i = 0; i < this->size(); ++i)
    {
        data[i] = value;
    }
}

} // namespace plamatrix
