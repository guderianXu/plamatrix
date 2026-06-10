#pragma once

#include <stdexcept>

#include "plamatrix/dense/dense_matrix.h"

namespace plamatrix
{

/// General matrix multiplication (GEMM): C(m x n) = A(m x k) * B(k x n).
/// Column-major storage assumed for all matrices.
/// @tparam Scalar  Element type (float or double)
/// @param A  Left operand matrix (m rows, k columns)
/// @param B  Right operand matrix (k rows, n columns)
/// @return  New CPU matrix C (m rows, n columns)
template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> gemm(const DenseMatrix<Scalar, Device::CPU>& A,
                                       const DenseMatrix<Scalar, Device::CPU>& B);

/// General matrix multiplication (GEMM) on GPU via cuBLAS.
/// Column-major storage assumed for all matrices.
/// @tparam Scalar  Element type (float or double)
/// @param A      Left operand GPU matrix (m rows, k columns)
/// @param B      Right operand GPU matrix (k rows, n columns)
/// @param stream  Optional CUDA stream (defaults to default stream)
/// @return  New GPU matrix C (m rows, n columns)
template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> gemm(const DenseMatrix<Scalar, Device::GPU>& A,
                                       const DenseMatrix<Scalar, Device::GPU>& B,
                                       cudaStream_t stream = nullptr);

/// General matrix multiplication (GEMM) on GPU into an existing output matrix.
/// Column-major storage assumed for all matrices.
/// @tparam Scalar  Element type (float or double)
/// @param A      Left operand GPU matrix (m rows, k columns)
/// @param B      Right operand GPU matrix (k rows, n columns)
/// @param C      Output GPU matrix (m rows, n columns)
/// @param stream  Optional CUDA stream (defaults to default stream)
template <typename Scalar>
void gemm(const DenseMatrix<Scalar, Device::GPU>& A,
          const DenseMatrix<Scalar, Device::GPU>& B,
          DenseMatrix<Scalar, Device::GPU>& C,
          cudaStream_t stream = nullptr);

/// Asynchronously launch GPU GEMM via cuBLAS.
/// Column-major storage assumed for all matrices.
/// @tparam Scalar  Element type (float or double)
/// @param A      Left operand GPU matrix (m rows, k columns)
/// @param B      Right operand GPU matrix (k rows, n columns)
/// @param stream  Optional CUDA stream (defaults to default stream)
/// @return  New GPU matrix C (m rows, n columns)
template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> gemmAsync(const DenseMatrix<Scalar, Device::GPU>& A,
                                           const DenseMatrix<Scalar, Device::GPU>& B,
                                           cudaStream_t stream = nullptr);

/// Asynchronously launch GPU GEMM into an existing output matrix.
/// Column-major storage assumed for all matrices.
/// @tparam Scalar  Element type (float or double)
/// @param A      Left operand GPU matrix (m rows, k columns)
/// @param B      Right operand GPU matrix (k rows, n columns)
/// @param C      Output GPU matrix (m rows, n columns)
/// @param stream  Optional CUDA stream (defaults to default stream)
template <typename Scalar>
void gemmAsync(const DenseMatrix<Scalar, Device::GPU>& A,
               const DenseMatrix<Scalar, Device::GPU>& B,
               DenseMatrix<Scalar, Device::GPU>& C,
               cudaStream_t stream = nullptr);

#ifdef PLAMATRIX_NO_CUDA
template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> gemm(const DenseMatrix<Scalar, Device::GPU>&,
                                       const DenseMatrix<Scalar, Device::GPU>&,
                                       cudaStream_t)
{
    throw std::runtime_error("gemm: GPU matrix multiplication requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
void gemm(const DenseMatrix<Scalar, Device::GPU>&,
          const DenseMatrix<Scalar, Device::GPU>&,
          DenseMatrix<Scalar, Device::GPU>&,
          cudaStream_t)
{
    throw std::runtime_error("gemm: GPU matrix multiplication requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> gemmAsync(const DenseMatrix<Scalar, Device::GPU>&,
                                           const DenseMatrix<Scalar, Device::GPU>&,
                                           cudaStream_t)
{
    throw std::runtime_error("gemmAsync: GPU matrix multiplication requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
void gemmAsync(const DenseMatrix<Scalar, Device::GPU>&,
               const DenseMatrix<Scalar, Device::GPU>&,
               DenseMatrix<Scalar, Device::GPU>&,
               cudaStream_t)
{
    throw std::runtime_error("gemmAsync: GPU matrix multiplication requires PLAMATRIX_WITH_CUDA=ON");
}
#endif

} // namespace plamatrix
