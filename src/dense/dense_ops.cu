#include <limits>
#include <sstream>
#include <stdexcept>

#include "plamatrix/internal/dense/dense_ops.h"

namespace plamatrix::internal
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
void addAsync(const DenseStorage<Scalar, Device::GPU>& A,
              const DenseStorage<Scalar, Device::GPU>& B,
              DenseStorage<Scalar, Device::GPU>& C,
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
DenseStorage<Scalar, Device::GPU> addAsync(const DenseStorage<Scalar, Device::GPU>& A,
                                          const DenseStorage<Scalar, Device::GPU>& B,
                                          cudaStream_t stream)
{
    detail::checkSameDimensions("add", A, B);
    auto C = DenseStorage<Scalar, Device::GPU>::uninitializedAsync(A.rows(), A.cols(), stream);
    addAsync(A, B, C, stream);
    return C;
}

template <typename Scalar>
void subAsync(const DenseStorage<Scalar, Device::GPU>& A,
              const DenseStorage<Scalar, Device::GPU>& B,
              DenseStorage<Scalar, Device::GPU>& C,
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
DenseStorage<Scalar, Device::GPU> subAsync(const DenseStorage<Scalar, Device::GPU>& A,
                                          const DenseStorage<Scalar, Device::GPU>& B,
                                          cudaStream_t stream)
{
    detail::checkSameDimensions("sub", A, B);
    auto C = DenseStorage<Scalar, Device::GPU>::uninitializedAsync(A.rows(), A.cols(), stream);
    subAsync(A, B, C, stream);
    return C;
}

template <typename Scalar>
DenseStorage<Scalar, Device::GPU> add(const DenseStorage<Scalar, Device::GPU>& A,
                                     const DenseStorage<Scalar, Device::GPU>& B,
                                     cudaStream_t stream)
{
    auto C = addAsync(A, B, stream);
    PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
    return C;
}

template <typename Scalar>
void add(const DenseStorage<Scalar, Device::GPU>& A,
         const DenseStorage<Scalar, Device::GPU>& B,
         DenseStorage<Scalar, Device::GPU>& C,
         cudaStream_t stream)
{
    addAsync(A, B, C, stream);
    PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
}

template <typename Scalar>
DenseStorage<Scalar, Device::GPU> sub(const DenseStorage<Scalar, Device::GPU>& A,
                                     const DenseStorage<Scalar, Device::GPU>& B,
                                     cudaStream_t stream)
{
    auto C = subAsync(A, B, stream);
    PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
    return C;
}

template <typename Scalar>
void sub(const DenseStorage<Scalar, Device::GPU>& A,
         const DenseStorage<Scalar, Device::GPU>& B,
         DenseStorage<Scalar, Device::GPU>& C,
         cudaStream_t stream)
{
    subAsync(A, B, C, stream);
    PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
}

template <typename Scalar>
DenseStorage<Scalar, Device::GPU> add(const DenseStorage<Scalar, Device::GPU>& A,
                                     const DenseStorage<Scalar, Device::GPU>& B)
{
    return add(A, B, nullptr);
}

template <typename Scalar>
DenseStorage<Scalar, Device::GPU> sub(const DenseStorage<Scalar, Device::GPU>& A,
                                     const DenseStorage<Scalar, Device::GPU>& B)
{
    return sub(A, B, nullptr);
}

// Explicit template instantiations
#ifdef PLAMATRIX_USE_FLOAT
template DenseStorage<float, Device::GPU> addAsync(const DenseStorage<float, Device::GPU>&,
                                                  const DenseStorage<float, Device::GPU>&,
                                                  cudaStream_t);

template void addAsync(const DenseStorage<float, Device::GPU>&,
                       const DenseStorage<float, Device::GPU>&,
                       DenseStorage<float, Device::GPU>&,
                       cudaStream_t);

template DenseStorage<float, Device::GPU> add(const DenseStorage<float, Device::GPU>&,
                                             const DenseStorage<float, Device::GPU>&,
                                             cudaStream_t);

template void add(const DenseStorage<float, Device::GPU>&,
                  const DenseStorage<float, Device::GPU>&,
                  DenseStorage<float, Device::GPU>&,
                  cudaStream_t);
#endif

#ifdef PLAMATRIX_USE_DOUBLE
template DenseStorage<double, Device::GPU> addAsync(const DenseStorage<double, Device::GPU>&,
                                                   const DenseStorage<double, Device::GPU>&,
                                                   cudaStream_t);

template void addAsync(const DenseStorage<double, Device::GPU>&,
                       const DenseStorage<double, Device::GPU>&,
                       DenseStorage<double, Device::GPU>&,
                       cudaStream_t);

template DenseStorage<double, Device::GPU> add(const DenseStorage<double, Device::GPU>&,
                                              const DenseStorage<double, Device::GPU>&,
                                              cudaStream_t);

template void add(const DenseStorage<double, Device::GPU>&,
                  const DenseStorage<double, Device::GPU>&,
                  DenseStorage<double, Device::GPU>&,
                  cudaStream_t);
#endif

#ifdef PLAMATRIX_USE_FLOAT
template DenseStorage<float, Device::GPU> add(const DenseStorage<float, Device::GPU>&,
                                             const DenseStorage<float, Device::GPU>&);
#endif

#ifdef PLAMATRIX_USE_DOUBLE
template DenseStorage<double, Device::GPU> add(const DenseStorage<double, Device::GPU>&,
                                              const DenseStorage<double, Device::GPU>&);
#endif

#ifdef PLAMATRIX_USE_FLOAT
template DenseStorage<float, Device::GPU> subAsync(const DenseStorage<float, Device::GPU>&,
                                                  const DenseStorage<float, Device::GPU>&,
                                                  cudaStream_t);

template void subAsync(const DenseStorage<float, Device::GPU>&,
                       const DenseStorage<float, Device::GPU>&,
                       DenseStorage<float, Device::GPU>&,
                       cudaStream_t);

template DenseStorage<float, Device::GPU> sub(const DenseStorage<float, Device::GPU>&,
                                             const DenseStorage<float, Device::GPU>&,
                                             cudaStream_t);

template void sub(const DenseStorage<float, Device::GPU>&,
                  const DenseStorage<float, Device::GPU>&,
                  DenseStorage<float, Device::GPU>&,
                  cudaStream_t);
#endif

#ifdef PLAMATRIX_USE_DOUBLE
template DenseStorage<double, Device::GPU> subAsync(const DenseStorage<double, Device::GPU>&,
                                                   const DenseStorage<double, Device::GPU>&,
                                                   cudaStream_t);

template void subAsync(const DenseStorage<double, Device::GPU>&,
                       const DenseStorage<double, Device::GPU>&,
                       DenseStorage<double, Device::GPU>&,
                       cudaStream_t);

template DenseStorage<double, Device::GPU> sub(const DenseStorage<double, Device::GPU>&,
                                              const DenseStorage<double, Device::GPU>&,
                                              cudaStream_t);

template void sub(const DenseStorage<double, Device::GPU>&,
                  const DenseStorage<double, Device::GPU>&,
                  DenseStorage<double, Device::GPU>&,
                  cudaStream_t);
#endif

#ifdef PLAMATRIX_USE_FLOAT
template DenseStorage<float, Device::GPU> sub(const DenseStorage<float, Device::GPU>&,
                                             const DenseStorage<float, Device::GPU>&);
#endif

#ifdef PLAMATRIX_USE_DOUBLE
template DenseStorage<double, Device::GPU> sub(const DenseStorage<double, Device::GPU>&,
                                              const DenseStorage<double, Device::GPU>&);
#endif

} // namespace plamatrix::internal
