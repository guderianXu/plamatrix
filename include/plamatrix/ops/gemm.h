#pragma once

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

} // namespace plamatrix
