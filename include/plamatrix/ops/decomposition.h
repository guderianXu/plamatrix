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

} // namespace plamatrix
