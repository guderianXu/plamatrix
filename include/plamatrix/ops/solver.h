#pragma once

#include <stdexcept>

#include "plamatrix/dense/dense_matrix.h"

namespace plamatrix
{

/// Solve the linear system A * X = B for X using LU decomposition with partial pivoting.
/// A must be square (n x n). B can be an n x 1 vector or an n x nrhs matrix.
/// On CPU, uses Gaussian elimination with partial pivoting.
/// On GPU, uses cuSOLVER getrf/getrs.
/// @tparam Scalar  Element type (float or double)
/// @tparam Dev     Device (CPU or GPU)
/// @param A  Square coefficient matrix (n x n)
/// @param B  Right-hand side matrix (n x nrhs)
/// @return  Solution matrix X (n x nrhs)
/// @throws std::runtime_error if A is not square, dimensions mismatch, or matrix is singular
template <typename Scalar, Device Dev>
DenseMatrix<Scalar, Dev> solve(const DenseMatrix<Scalar, Dev>& A, const DenseMatrix<Scalar, Dev>& B);

#if defined(PLAMATRIX_NO_GPU)
template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> solve(const DenseMatrix<Scalar, Device::GPU>&,
                                       const DenseMatrix<Scalar, Device::GPU>&)
{
    throw std::runtime_error("solve: GPU linear solve requires a GPU backend");
}
#endif

} // namespace plamatrix
