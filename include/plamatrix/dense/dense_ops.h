#pragma once

#include <omp.h>

#include "plamatrix/dense/dense_matrix.h"

namespace plamatrix
{

/// Element-wise addition of two CPU matrices.
/// @tparam Scalar  Element type (float or double)
/// @param A  Left operand matrix
/// @param B  Right operand matrix (must have same dimensions as A)
/// @return  New CPU matrix C where C[i] = A[i] + B[i]
template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> add(const DenseMatrix<Scalar, Device::CPU>& A, const DenseMatrix<Scalar, Device::CPU>& B)
{
    DenseMatrix<Scalar, Device::CPU> C(A.rows(), A.cols());
    Index n = A.size();
    #pragma omp parallel for
    for (Index i = 0; i < n; ++i)
    {
        C.data()[i] = A.data()[i] + B.data()[i];
    }
    return C;
}

/// Element-wise subtraction of two CPU matrices.
/// @tparam Scalar  Element type (float or double)
/// @param A  Left operand matrix
/// @param B  Right operand matrix (must have same dimensions as A)
/// @return  New CPU matrix C where C[i] = A[i] - B[i]
template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> sub(const DenseMatrix<Scalar, Device::CPU>& A, const DenseMatrix<Scalar, Device::CPU>& B)
{
    DenseMatrix<Scalar, Device::CPU> C(A.rows(), A.cols());
    Index n = A.size();
    #pragma omp parallel for
    for (Index i = 0; i < n; ++i)
    {
        C.data()[i] = A.data()[i] - B.data()[i];
    }
    return C;
}

/// GPU element-wise addition (declaration only — definition in dense_ops.cu).
/// @tparam Scalar  Element type (float or double)
/// @param A  Left operand GPU matrix
/// @param B  Right operand GPU matrix (must have same dimensions as A)
/// @return  New GPU matrix C where C[i] = A[i] + B[i]
template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> add(const DenseMatrix<Scalar, Device::GPU>& A, const DenseMatrix<Scalar, Device::GPU>& B);

/// GPU element-wise subtraction (declaration only — definition in dense_ops.cu).
/// @tparam Scalar  Element type (float or double)
/// @param A  Left operand GPU matrix
/// @param B  Right operand GPU matrix (must have same dimensions as A)
/// @return  New GPU matrix C where C[i] = A[i] - B[i]
template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> sub(const DenseMatrix<Scalar, Device::GPU>& A, const DenseMatrix<Scalar, Device::GPU>& B);

} // namespace plamatrix
