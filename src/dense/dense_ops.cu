#include <limits>
#include <sstream>
#include <stdexcept>

#include "plamatrix/dense/dense_ops.h"

namespace plamatrix
{

template <typename Scalar>
__global__ void elementWiseAddKernel(const Scalar* A, const Scalar* B, Scalar* C, Index count)
{
    Index idx = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < count)
    {
        C[idx] = A[idx] + B[idx];
    }
}

template <typename Scalar>
__global__ void elementWiseSubKernel(const Scalar* A, const Scalar* B, Scalar* C, Index count)
{
    Index idx = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < count)
    {
        C[idx] = A[idx] - B[idx];
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

} // anonymous namespace

template <typename Scalar>
void addAsync(const DenseMatrix<Scalar, Device::GPU>& A,
              const DenseMatrix<Scalar, Device::GPU>& B,
              DenseMatrix<Scalar, Device::GPU>& C,
              cudaStream_t stream)
{
    detail::checkSameDimensions("add", A, B);
    detail::checkOutputDimensions("add", C, A.rows(), A.cols());
    Index count = A.size();
    if (count == 0)
    {
        return;
    }
    constexpr int block_size = 256;
    int grid_size = checkedCudaGrid1D(count, block_size, "add");
    elementWiseAddKernel<Scalar><<<grid_size, block_size, 0, stream>>>(A.data(), B.data(), C.data(), count);
    PLAMATRIX_CHECK_CUDA(cudaGetLastError());
}

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> addAsync(const DenseMatrix<Scalar, Device::GPU>& A,
                                          const DenseMatrix<Scalar, Device::GPU>& B,
                                          cudaStream_t stream)
{
    detail::checkSameDimensions("add", A, B);
    DenseMatrix<Scalar, Device::GPU> C(A.rows(), A.cols());
    addAsync(A, B, C, stream);
    return C;
}

template <typename Scalar>
void subAsync(const DenseMatrix<Scalar, Device::GPU>& A,
              const DenseMatrix<Scalar, Device::GPU>& B,
              DenseMatrix<Scalar, Device::GPU>& C,
              cudaStream_t stream)
{
    detail::checkSameDimensions("sub", A, B);
    detail::checkOutputDimensions("sub", C, A.rows(), A.cols());
    Index count = A.size();
    if (count == 0)
    {
        return;
    }
    constexpr int block_size = 256;
    int grid_size = checkedCudaGrid1D(count, block_size, "sub");
    elementWiseSubKernel<Scalar><<<grid_size, block_size, 0, stream>>>(A.data(), B.data(), C.data(), count);
    PLAMATRIX_CHECK_CUDA(cudaGetLastError());
}

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> subAsync(const DenseMatrix<Scalar, Device::GPU>& A,
                                          const DenseMatrix<Scalar, Device::GPU>& B,
                                          cudaStream_t stream)
{
    detail::checkSameDimensions("sub", A, B);
    DenseMatrix<Scalar, Device::GPU> C(A.rows(), A.cols());
    subAsync(A, B, C, stream);
    return C;
}

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> add(const DenseMatrix<Scalar, Device::GPU>& A,
                                     const DenseMatrix<Scalar, Device::GPU>& B,
                                     cudaStream_t stream)
{
    auto C = addAsync(A, B, stream);
    PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
    return C;
}

template <typename Scalar>
void add(const DenseMatrix<Scalar, Device::GPU>& A,
         const DenseMatrix<Scalar, Device::GPU>& B,
         DenseMatrix<Scalar, Device::GPU>& C,
         cudaStream_t stream)
{
    addAsync(A, B, C, stream);
    PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
}

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> sub(const DenseMatrix<Scalar, Device::GPU>& A,
                                     const DenseMatrix<Scalar, Device::GPU>& B,
                                     cudaStream_t stream)
{
    auto C = subAsync(A, B, stream);
    PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
    return C;
}

template <typename Scalar>
void sub(const DenseMatrix<Scalar, Device::GPU>& A,
         const DenseMatrix<Scalar, Device::GPU>& B,
         DenseMatrix<Scalar, Device::GPU>& C,
         cudaStream_t stream)
{
    subAsync(A, B, C, stream);
    PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
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
#ifdef PLAMATRIX_USE_FLOAT
template DenseMatrix<float, Device::GPU> addAsync(const DenseMatrix<float, Device::GPU>&,
                                                  const DenseMatrix<float, Device::GPU>&,
                                                  cudaStream_t);

template void addAsync(const DenseMatrix<float, Device::GPU>&,
                       const DenseMatrix<float, Device::GPU>&,
                       DenseMatrix<float, Device::GPU>&,
                       cudaStream_t);

template DenseMatrix<float, Device::GPU> add(const DenseMatrix<float, Device::GPU>&,
                                             const DenseMatrix<float, Device::GPU>&,
                                             cudaStream_t);

template void add(const DenseMatrix<float, Device::GPU>&,
                  const DenseMatrix<float, Device::GPU>&,
                  DenseMatrix<float, Device::GPU>&,
                  cudaStream_t);
#endif

#ifdef PLAMATRIX_USE_DOUBLE
template DenseMatrix<double, Device::GPU> addAsync(const DenseMatrix<double, Device::GPU>&,
                                                   const DenseMatrix<double, Device::GPU>&,
                                                   cudaStream_t);

template void addAsync(const DenseMatrix<double, Device::GPU>&,
                       const DenseMatrix<double, Device::GPU>&,
                       DenseMatrix<double, Device::GPU>&,
                       cudaStream_t);

template DenseMatrix<double, Device::GPU> add(const DenseMatrix<double, Device::GPU>&,
                                              const DenseMatrix<double, Device::GPU>&,
                                              cudaStream_t);

template void add(const DenseMatrix<double, Device::GPU>&,
                  const DenseMatrix<double, Device::GPU>&,
                  DenseMatrix<double, Device::GPU>&,
                  cudaStream_t);
#endif

#ifdef PLAMATRIX_USE_FLOAT
template DenseMatrix<float, Device::GPU> add(const DenseMatrix<float, Device::GPU>&,
                                             const DenseMatrix<float, Device::GPU>&);
#endif

#ifdef PLAMATRIX_USE_DOUBLE
template DenseMatrix<double, Device::GPU> add(const DenseMatrix<double, Device::GPU>&,
                                              const DenseMatrix<double, Device::GPU>&);
#endif

#ifdef PLAMATRIX_USE_FLOAT
template DenseMatrix<float, Device::GPU> subAsync(const DenseMatrix<float, Device::GPU>&,
                                                  const DenseMatrix<float, Device::GPU>&,
                                                  cudaStream_t);

template void subAsync(const DenseMatrix<float, Device::GPU>&,
                       const DenseMatrix<float, Device::GPU>&,
                       DenseMatrix<float, Device::GPU>&,
                       cudaStream_t);

template DenseMatrix<float, Device::GPU> sub(const DenseMatrix<float, Device::GPU>&,
                                             const DenseMatrix<float, Device::GPU>&,
                                             cudaStream_t);

template void sub(const DenseMatrix<float, Device::GPU>&,
                  const DenseMatrix<float, Device::GPU>&,
                  DenseMatrix<float, Device::GPU>&,
                  cudaStream_t);
#endif

#ifdef PLAMATRIX_USE_DOUBLE
template DenseMatrix<double, Device::GPU> subAsync(const DenseMatrix<double, Device::GPU>&,
                                                   const DenseMatrix<double, Device::GPU>&,
                                                   cudaStream_t);

template void subAsync(const DenseMatrix<double, Device::GPU>&,
                       const DenseMatrix<double, Device::GPU>&,
                       DenseMatrix<double, Device::GPU>&,
                       cudaStream_t);

template DenseMatrix<double, Device::GPU> sub(const DenseMatrix<double, Device::GPU>&,
                                              const DenseMatrix<double, Device::GPU>&,
                                              cudaStream_t);

template void sub(const DenseMatrix<double, Device::GPU>&,
                  const DenseMatrix<double, Device::GPU>&,
                  DenseMatrix<double, Device::GPU>&,
                  cudaStream_t);
#endif

#ifdef PLAMATRIX_USE_FLOAT
template DenseMatrix<float, Device::GPU> sub(const DenseMatrix<float, Device::GPU>&,
                                             const DenseMatrix<float, Device::GPU>&);
#endif

#ifdef PLAMATRIX_USE_DOUBLE
template DenseMatrix<double, Device::GPU> sub(const DenseMatrix<double, Device::GPU>&,
                                              const DenseMatrix<double, Device::GPU>&);
#endif

} // namespace plamatrix
