#include "plamatrix/dense/dense_ops.h"

namespace plamatrix
{

template <typename Scalar>
__global__ void elementWiseAddKernel(const Scalar* A, const Scalar* B, Scalar* C, Index count)
{
    Index idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
    {
        C[idx] = A[idx] + B[idx];
    }
}

template <typename Scalar>
__global__ void elementWiseSubKernel(const Scalar* A, const Scalar* B, Scalar* C, Index count)
{
    Index idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
    {
        C[idx] = A[idx] - B[idx];
    }
}

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> add(const DenseMatrix<Scalar, Device::GPU>& A,
                                     const DenseMatrix<Scalar, Device::GPU>& B,
                                     cudaStream_t stream)
{
    detail::checkSameDimensions("add", A, B);
    DenseMatrix<Scalar, Device::GPU> C(A.rows(), A.cols());
    Index count = A.size();
    if (count == 0)
    {
        return C;
    }
    int block_size = 256;
    int grid_size = static_cast<int>((count + block_size - 1) / block_size);
    elementWiseAddKernel<Scalar><<<grid_size, block_size, 0, stream>>>(A.data(), B.data(), C.data(), count);
    PLAMATRIX_CHECK_CUDA(cudaGetLastError());
    PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
    return C;
}

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> sub(const DenseMatrix<Scalar, Device::GPU>& A,
                                     const DenseMatrix<Scalar, Device::GPU>& B,
                                     cudaStream_t stream)
{
    detail::checkSameDimensions("sub", A, B);
    DenseMatrix<Scalar, Device::GPU> C(A.rows(), A.cols());
    Index count = A.size();
    if (count == 0)
    {
        return C;
    }
    int block_size = 256;
    int grid_size = static_cast<int>((count + block_size - 1) / block_size);
    elementWiseSubKernel<Scalar><<<grid_size, block_size, 0, stream>>>(A.data(), B.data(), C.data(), count);
    PLAMATRIX_CHECK_CUDA(cudaGetLastError());
    PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
    return C;
}

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> add(const DenseMatrix<Scalar, Device::GPU>& A,
                                     const DenseMatrix<Scalar, Device::GPU>& B)
{
    return add(A, B, nullptr);
}

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> sub(const DenseMatrix<Scalar, Device::GPU>& A,
                                     const DenseMatrix<Scalar, Device::GPU>& B)
{
    return sub(A, B, nullptr);
}

// Explicit template instantiations
template DenseMatrix<float, Device::GPU> add(const DenseMatrix<float, Device::GPU>&,
                                             const DenseMatrix<float, Device::GPU>&,
                                             cudaStream_t);
template DenseMatrix<double, Device::GPU> add(const DenseMatrix<double, Device::GPU>&,
                                              const DenseMatrix<double, Device::GPU>&,
                                              cudaStream_t);
template DenseMatrix<float, Device::GPU> add(const DenseMatrix<float, Device::GPU>&,
                                             const DenseMatrix<float, Device::GPU>&);
template DenseMatrix<double, Device::GPU> add(const DenseMatrix<double, Device::GPU>&,
                                              const DenseMatrix<double, Device::GPU>&);
template DenseMatrix<float, Device::GPU> sub(const DenseMatrix<float, Device::GPU>&,
                                             const DenseMatrix<float, Device::GPU>&,
                                             cudaStream_t);
template DenseMatrix<double, Device::GPU> sub(const DenseMatrix<double, Device::GPU>&,
                                              const DenseMatrix<double, Device::GPU>&,
                                              cudaStream_t);
template DenseMatrix<float, Device::GPU> sub(const DenseMatrix<float, Device::GPU>&,
                                             const DenseMatrix<float, Device::GPU>&);
template DenseMatrix<double, Device::GPU> sub(const DenseMatrix<double, Device::GPU>&,
                                              const DenseMatrix<double, Device::GPU>&);

} // namespace plamatrix
