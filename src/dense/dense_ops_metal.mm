#include "plamatrix/dense/dense_ops.h"

#include "plamatrix/core/metal_context.h"

namespace plamatrix
{

namespace
{

template <typename Scalar>
void checkOutput(const char* op,
                 const DenseMatrix<Scalar, Device::GPU>& A,
                 const DenseMatrix<Scalar, Device::GPU>& B,
                 const DenseMatrix<Scalar, Device::GPU>& C)
{
    detail::checkSameDimensions(op, A, B);
    detail::checkOutputDimensions(op, C, A.rows(), A.cols());
}

} // namespace

#ifdef PLAMATRIX_USE_FLOAT
template <>
void addAsync(const DenseMatrix<float, Device::GPU>& A,
              const DenseMatrix<float, Device::GPU>& B,
              DenseMatrix<float, Device::GPU>& C,
              cudaStream_t)
{
    checkOutput("addAsync", A, B, C);
    detail::metalAddFloat(A.data(), B.data(), C.data(), A.size());
}

template <>
void subAsync(const DenseMatrix<float, Device::GPU>& A,
              const DenseMatrix<float, Device::GPU>& B,
              DenseMatrix<float, Device::GPU>& C,
              cudaStream_t)
{
    checkOutput("subAsync", A, B, C);
    detail::metalSubFloat(A.data(), B.data(), C.data(), A.size());
}
#endif

#ifdef PLAMATRIX_USE_DOUBLE
template <>
void addAsync(const DenseMatrix<double, Device::GPU>& A,
              const DenseMatrix<double, Device::GPU>& B,
              DenseMatrix<double, Device::GPU>& C,
              cudaStream_t)
{
    checkOutput("addAsync", A, B, C);
    detail::metalAddDoubleFallback(A.data(), B.data(), C.data(), A.size());
}

template <>
void subAsync(const DenseMatrix<double, Device::GPU>& A,
              const DenseMatrix<double, Device::GPU>& B,
              DenseMatrix<double, Device::GPU>& C,
              cudaStream_t)
{
    checkOutput("subAsync", A, B, C);
    detail::metalSubDoubleFallback(A.data(), B.data(), C.data(), A.size());
}
#endif

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> addAsync(const DenseMatrix<Scalar, Device::GPU>& A,
                                          const DenseMatrix<Scalar, Device::GPU>& B,
                                          cudaStream_t stream)
{
    detail::checkSameDimensions("addAsync", A, B);
    DenseMatrix<Scalar, Device::GPU> C(A.rows(), A.cols());
    addAsync(A, B, C, stream);
    return C;
}

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> add(const DenseMatrix<Scalar, Device::GPU>& A,
                                     const DenseMatrix<Scalar, Device::GPU>& B,
                                     cudaStream_t stream)
{
    return addAsync(A, B, stream);
}

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> add(const DenseMatrix<Scalar, Device::GPU>& A,
                                     const DenseMatrix<Scalar, Device::GPU>& B)
{
    return addAsync(A, B, nullptr);
}

template <typename Scalar>
void add(const DenseMatrix<Scalar, Device::GPU>& A,
         const DenseMatrix<Scalar, Device::GPU>& B,
         DenseMatrix<Scalar, Device::GPU>& C,
         cudaStream_t stream)
{
    addAsync(A, B, C, stream);
}

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> subAsync(const DenseMatrix<Scalar, Device::GPU>& A,
                                          const DenseMatrix<Scalar, Device::GPU>& B,
                                          cudaStream_t stream)
{
    detail::checkSameDimensions("subAsync", A, B);
    DenseMatrix<Scalar, Device::GPU> C(A.rows(), A.cols());
    subAsync(A, B, C, stream);
    return C;
}

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> sub(const DenseMatrix<Scalar, Device::GPU>& A,
                                     const DenseMatrix<Scalar, Device::GPU>& B,
                                     cudaStream_t stream)
{
    return subAsync(A, B, stream);
}

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> sub(const DenseMatrix<Scalar, Device::GPU>& A,
                                     const DenseMatrix<Scalar, Device::GPU>& B)
{
    return subAsync(A, B, nullptr);
}

template <typename Scalar>
void sub(const DenseMatrix<Scalar, Device::GPU>& A,
         const DenseMatrix<Scalar, Device::GPU>& B,
         DenseMatrix<Scalar, Device::GPU>& C,
         cudaStream_t stream)
{
    subAsync(A, B, C, stream);
}

#ifdef PLAMATRIX_USE_FLOAT
template DenseMatrix<float, Device::GPU> addAsync(const DenseMatrix<float, Device::GPU>&,
                                                  const DenseMatrix<float, Device::GPU>&,
                                                  cudaStream_t);
template DenseMatrix<float, Device::GPU> add(const DenseMatrix<float, Device::GPU>&,
                                             const DenseMatrix<float, Device::GPU>&,
                                             cudaStream_t);
template DenseMatrix<float, Device::GPU> add(const DenseMatrix<float, Device::GPU>&,
                                             const DenseMatrix<float, Device::GPU>&);
template void add(const DenseMatrix<float, Device::GPU>&,
                  const DenseMatrix<float, Device::GPU>&,
                  DenseMatrix<float, Device::GPU>&,
                  cudaStream_t);
template DenseMatrix<float, Device::GPU> subAsync(const DenseMatrix<float, Device::GPU>&,
                                                  const DenseMatrix<float, Device::GPU>&,
                                                  cudaStream_t);
template DenseMatrix<float, Device::GPU> sub(const DenseMatrix<float, Device::GPU>&,
                                             const DenseMatrix<float, Device::GPU>&,
                                             cudaStream_t);
template DenseMatrix<float, Device::GPU> sub(const DenseMatrix<float, Device::GPU>&,
                                             const DenseMatrix<float, Device::GPU>&);
template void sub(const DenseMatrix<float, Device::GPU>&,
                  const DenseMatrix<float, Device::GPU>&,
                  DenseMatrix<float, Device::GPU>&,
                  cudaStream_t);
#endif

#ifdef PLAMATRIX_USE_DOUBLE
template DenseMatrix<double, Device::GPU> addAsync(const DenseMatrix<double, Device::GPU>&,
                                                   const DenseMatrix<double, Device::GPU>&,
                                                   cudaStream_t);
template DenseMatrix<double, Device::GPU> add(const DenseMatrix<double, Device::GPU>&,
                                              const DenseMatrix<double, Device::GPU>&,
                                              cudaStream_t);
template DenseMatrix<double, Device::GPU> add(const DenseMatrix<double, Device::GPU>&,
                                              const DenseMatrix<double, Device::GPU>&);
template void add(const DenseMatrix<double, Device::GPU>&,
                  const DenseMatrix<double, Device::GPU>&,
                  DenseMatrix<double, Device::GPU>&,
                  cudaStream_t);
template DenseMatrix<double, Device::GPU> subAsync(const DenseMatrix<double, Device::GPU>&,
                                                   const DenseMatrix<double, Device::GPU>&,
                                                   cudaStream_t);
template DenseMatrix<double, Device::GPU> sub(const DenseMatrix<double, Device::GPU>&,
                                              const DenseMatrix<double, Device::GPU>&,
                                              cudaStream_t);
template DenseMatrix<double, Device::GPU> sub(const DenseMatrix<double, Device::GPU>&,
                                              const DenseMatrix<double, Device::GPU>&);
template void sub(const DenseMatrix<double, Device::GPU>&,
                  const DenseMatrix<double, Device::GPU>&,
                  DenseMatrix<double, Device::GPU>&,
                  cudaStream_t);
#endif

} // namespace plamatrix
