#pragma once

#include <tuple>

#include "plamatrix/dense/dense_matrix.h"

namespace plamatrix
{

/// Compute the Singular Value Decomposition (SVD) of matrix A.
/// A = U * diag(S) * Vt, where U and Vt are orthogonal and S contains singular values sorted descending.
/// @tparam Scalar  Element type (float or double)
/// @param A  Input matrix (CPU, m rows x n cols)
/// @return  Tuple of (U, S_vector, Vt). U is m×m, S_vector is a min(m,n)×1 column vector, Vt is n×n.
template <typename Scalar>
std::tuple<DenseMatrix<Scalar, Device::CPU>, DenseMatrix<Scalar, Device::CPU>, DenseMatrix<Scalar, Device::CPU>>
svd(const DenseMatrix<Scalar, Device::CPU>& A);

/// GPU accelerated SVD via cuSOLVER gesvd.
/// @tparam Scalar  Element type (float or double)
/// @param A  Input matrix (GPU, m rows x n cols)
/// @return  Tuple of (U, S_vector, Vt). U is m×m, S_vector is min(m,n)×1, Vt is n×n. All on GPU.
template <typename Scalar>
std::tuple<DenseMatrix<Scalar, Device::GPU>, DenseMatrix<Scalar, Device::GPU>, DenseMatrix<Scalar, Device::GPU>>
svd(const DenseMatrix<Scalar, Device::GPU>& A);

/// QR decomposition: A = Q * R. CPU implementation using Householder reflections.
/// Q is m×m orthogonal, R is m×n upper triangular.
/// @tparam Scalar  Element type (float or double)
/// @param A  Input matrix (CPU, m rows x n cols)
/// @return  Tuple of (Q, R). Q is m×m, R is m×n.
template <typename Scalar>
std::tuple<DenseMatrix<Scalar, Device::CPU>, DenseMatrix<Scalar, Device::CPU>>
qr(const DenseMatrix<Scalar, Device::CPU>& A);

/// GPU accelerated QR decomposition via cuSOLVER geqrf + orgqr.
/// Q is m×m orthogonal, R is m×n upper triangular.
/// @tparam Scalar  Element type (float or double)
/// @param A  Input matrix (GPU, m rows x n cols)
/// @return  Tuple of (Q, R). Q is m×m, R is m×n. All on GPU.
template <typename Scalar>
std::tuple<DenseMatrix<Scalar, Device::GPU>, DenseMatrix<Scalar, Device::GPU>>
qr(const DenseMatrix<Scalar, Device::GPU>& A);

/// Symmetric eigenvalue decomposition. CPU implementation using Jacobi algorithm.
/// Returns eigenvalues sorted in descending order as a column vector.
/// @tparam Scalar  Element type (float or double)
/// @param A  Symmetric input matrix (CPU, n rows x n cols)
/// @return  Column vector of eigenvalues (n × 1), sorted descending.
template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> eigh(const DenseMatrix<Scalar, Device::CPU>& A);

/// GPU accelerated symmetric eigenvalue decomposition via cuSOLVER syevd.
/// Returns eigenvalues sorted in descending order as a column vector.
/// @tparam Scalar  Element type (float or double)
/// @param A  Symmetric input matrix (GPU, n rows x n cols)
/// @return  Column vector of eigenvalues (n × 1), sorted descending. On GPU.
template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> eigh(const DenseMatrix<Scalar, Device::GPU>& A);

} // namespace plamatrix
