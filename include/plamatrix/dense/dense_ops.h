#pragma once

#include <sstream>
#include <stdexcept>

#include <omp.h>

#include "plamatrix/core/parallel.h"
#include "plamatrix/dense/dense_matrix.h"

namespace plamatrix
{

namespace detail
{

template <typename Scalar, Device Dev>
void checkSameDimensions(const char* op,
                         const DenseMatrix<Scalar, Dev>& A,
                         const DenseMatrix<Scalar, Dev>& B)
{
    if (A.rows() != B.rows() || A.cols() != B.cols())
    {
        std::ostringstream oss;
        oss << op << " dimension mismatch: A is " << A.rows() << "x" << A.cols()
            << ", B is " << B.rows() << "x" << B.cols();
        throw std::runtime_error(oss.str());
    }
}

template <typename Scalar, Device Dev>
void checkOutputDimensions(const char* op,
                           const DenseMatrix<Scalar, Dev>& output,
                           Index rows,
                           Index cols)
{
    if (output.rows() != rows || output.cols() != cols)
    {
        std::ostringstream oss;
        oss << op << " output dimension mismatch: output is " << output.rows() << "x" << output.cols()
            << ", expected " << rows << "x" << cols;
        throw std::runtime_error(oss.str());
    }
}

} // namespace detail

/// Element-wise addition of two CPU matrices.
/// @tparam Scalar  Element type (float or double)
/// @param A  Left operand matrix
/// @param B  Right operand matrix (must have same dimensions as A)
/// @return  New CPU matrix C where C[i] = A[i] + B[i]
template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> add(const DenseMatrix<Scalar, Device::CPU>& A,
                                     const DenseMatrix<Scalar, Device::CPU>& B)
{
    detail::checkSameDimensions("add", A, B);
    DenseMatrix<Scalar, Device::CPU> C(A.rows(), A.cols());
    Index n = A.size();
    if (detail::shouldUseOpenMp(n))
    {
        #pragma omp parallel for
        for (Index i = 0; i < n; ++i)
        {
            C.data()[i] = A.data()[i] + B.data()[i];
        }
    }
    else
    {
        for (Index i = 0; i < n; ++i)
        {
            C.data()[i] = A.data()[i] + B.data()[i];
        }
    }
    return C;
}

/// Element-wise subtraction of two CPU matrices.
/// @tparam Scalar  Element type (float or double)
/// @param A  Left operand matrix
/// @param B  Right operand matrix (must have same dimensions as A)
/// @return  New CPU matrix C where C[i] = A[i] - B[i]
template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> sub(const DenseMatrix<Scalar, Device::CPU>& A,
                                     const DenseMatrix<Scalar, Device::CPU>& B)
{
    detail::checkSameDimensions("sub", A, B);
    DenseMatrix<Scalar, Device::CPU> C(A.rows(), A.cols());
    Index n = A.size();
    if (detail::shouldUseOpenMp(n))
    {
        #pragma omp parallel for
        for (Index i = 0; i < n; ++i)
        {
            C.data()[i] = A.data()[i] - B.data()[i];
        }
    }
    else
    {
        for (Index i = 0; i < n; ++i)
        {
            C.data()[i] = A.data()[i] - B.data()[i];
        }
    }
    return C;
}

/// GPU element-wise addition (declaration only — definition in dense_ops.cu).
/// @tparam Scalar  Element type (float or double)
/// @param A  Left operand GPU matrix
/// @param B  Right operand GPU matrix (must have same dimensions as A)
/// @param stream  Optional CUDA stream
/// @return  New GPU matrix C where C[i] = A[i] + B[i]
template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> add(const DenseMatrix<Scalar, Device::GPU>& A,
                                      const DenseMatrix<Scalar, Device::GPU>& B,
                                      cudaStream_t stream);

/// GPU element-wise addition into an existing output matrix.
/// @tparam Scalar  Element type (float or double)
/// @param A  Left operand GPU matrix
/// @param B  Right operand GPU matrix (must have same dimensions as A)
/// @param C  Output GPU matrix (must have same dimensions as A)
/// @param stream  Optional CUDA stream
template <typename Scalar>
void add(const DenseMatrix<Scalar, Device::GPU>& A,
         const DenseMatrix<Scalar, Device::GPU>& B,
         DenseMatrix<Scalar, Device::GPU>& C,
         cudaStream_t stream = nullptr);

/// GPU element-wise addition on the default CUDA stream.
template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> add(const DenseMatrix<Scalar, Device::GPU>& A,
                                     const DenseMatrix<Scalar, Device::GPU>& B);

/// Asynchronously launch GPU element-wise addition.
/// @tparam Scalar  Element type (float or double)
/// @param A  Left operand GPU matrix
/// @param B  Right operand GPU matrix (must have same dimensions as A)
/// @param stream  Optional CUDA stream
/// @return  New GPU matrix C where C[i] = A[i] + B[i]
template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> addAsync(const DenseMatrix<Scalar, Device::GPU>& A,
                                          const DenseMatrix<Scalar, Device::GPU>& B,
                                          cudaStream_t stream = nullptr);

/// Asynchronously launch GPU element-wise addition into an existing output matrix.
/// @tparam Scalar  Element type (float or double)
/// @param A  Left operand GPU matrix
/// @param B  Right operand GPU matrix (must have same dimensions as A)
/// @param C  Output GPU matrix (must have same dimensions as A)
/// @param stream  Optional CUDA stream
template <typename Scalar>
void addAsync(const DenseMatrix<Scalar, Device::GPU>& A,
              const DenseMatrix<Scalar, Device::GPU>& B,
              DenseMatrix<Scalar, Device::GPU>& C,
              cudaStream_t stream = nullptr);

/// GPU element-wise subtraction (declaration only — definition in dense_ops.cu).
/// @tparam Scalar  Element type (float or double)
/// @param A  Left operand GPU matrix
/// @param B  Right operand GPU matrix (must have same dimensions as A)
/// @param stream  Optional CUDA stream
/// @return  New GPU matrix C where C[i] = A[i] - B[i]
template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> sub(const DenseMatrix<Scalar, Device::GPU>& A,
                                      const DenseMatrix<Scalar, Device::GPU>& B,
                                      cudaStream_t stream);

/// GPU element-wise subtraction into an existing output matrix.
/// @tparam Scalar  Element type (float or double)
/// @param A  Left operand GPU matrix
/// @param B  Right operand GPU matrix (must have same dimensions as A)
/// @param C  Output GPU matrix (must have same dimensions as A)
/// @param stream  Optional CUDA stream
template <typename Scalar>
void sub(const DenseMatrix<Scalar, Device::GPU>& A,
         const DenseMatrix<Scalar, Device::GPU>& B,
         DenseMatrix<Scalar, Device::GPU>& C,
         cudaStream_t stream = nullptr);

/// GPU element-wise subtraction on the default CUDA stream.
template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> sub(const DenseMatrix<Scalar, Device::GPU>& A,
                                     const DenseMatrix<Scalar, Device::GPU>& B);

/// Asynchronously launch GPU element-wise subtraction.
/// @tparam Scalar  Element type (float or double)
/// @param A  Left operand GPU matrix
/// @param B  Right operand GPU matrix (must have same dimensions as A)
/// @param stream  Optional CUDA stream
/// @return  New GPU matrix C where C[i] = A[i] - B[i]
template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> subAsync(const DenseMatrix<Scalar, Device::GPU>& A,
                                          const DenseMatrix<Scalar, Device::GPU>& B,
                                          cudaStream_t stream = nullptr);

/// Asynchronously launch GPU element-wise subtraction into an existing output matrix.
/// @tparam Scalar  Element type (float or double)
/// @param A  Left operand GPU matrix
/// @param B  Right operand GPU matrix (must have same dimensions as A)
/// @param C  Output GPU matrix (must have same dimensions as A)
/// @param stream  Optional CUDA stream
template <typename Scalar>
void subAsync(const DenseMatrix<Scalar, Device::GPU>& A,
              const DenseMatrix<Scalar, Device::GPU>& B,
              DenseMatrix<Scalar, Device::GPU>& C,
              cudaStream_t stream = nullptr);

#ifdef PLAMATRIX_NO_CUDA
template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> add(const DenseMatrix<Scalar, Device::GPU>&,
                                      const DenseMatrix<Scalar, Device::GPU>&,
                                      cudaStream_t)
{
    throw std::runtime_error("add: GPU element-wise addition requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
void add(const DenseMatrix<Scalar, Device::GPU>&,
         const DenseMatrix<Scalar, Device::GPU>&,
         DenseMatrix<Scalar, Device::GPU>&,
         cudaStream_t)
{
    throw std::runtime_error("add: GPU element-wise addition requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> add(const DenseMatrix<Scalar, Device::GPU>& A,
                                     const DenseMatrix<Scalar, Device::GPU>& B)
{
    return add(A, B, nullptr);
}

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> addAsync(const DenseMatrix<Scalar, Device::GPU>&,
                                           const DenseMatrix<Scalar, Device::GPU>&,
                                           cudaStream_t)
{
    throw std::runtime_error("addAsync: GPU element-wise addition requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
void addAsync(const DenseMatrix<Scalar, Device::GPU>&,
              const DenseMatrix<Scalar, Device::GPU>&,
              DenseMatrix<Scalar, Device::GPU>&,
              cudaStream_t)
{
    throw std::runtime_error("addAsync: GPU element-wise addition requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> sub(const DenseMatrix<Scalar, Device::GPU>&,
                                      const DenseMatrix<Scalar, Device::GPU>&,
                                      cudaStream_t)
{
    throw std::runtime_error("sub: GPU element-wise subtraction requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
void sub(const DenseMatrix<Scalar, Device::GPU>&,
         const DenseMatrix<Scalar, Device::GPU>&,
         DenseMatrix<Scalar, Device::GPU>&,
         cudaStream_t)
{
    throw std::runtime_error("sub: GPU element-wise subtraction requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> sub(const DenseMatrix<Scalar, Device::GPU>& A,
                                     const DenseMatrix<Scalar, Device::GPU>& B)
{
    return sub(A, B, nullptr);
}

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> subAsync(const DenseMatrix<Scalar, Device::GPU>&,
                                           const DenseMatrix<Scalar, Device::GPU>&,
                                           cudaStream_t)
{
    throw std::runtime_error("subAsync: GPU element-wise subtraction requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
void subAsync(const DenseMatrix<Scalar, Device::GPU>&,
              const DenseMatrix<Scalar, Device::GPU>&,
              DenseMatrix<Scalar, Device::GPU>&,
              cudaStream_t)
{
    throw std::runtime_error("subAsync: GPU element-wise subtraction requires PLAMATRIX_WITH_CUDA=ON");
}
#endif

/// Scalar multiplication: alpha * A (element-wise).
/// @tparam Scalar  Element type (float or double)
/// @param alpha  Scalar multiplier
/// @param A      CPU matrix
/// @return  New CPU matrix C where C[i] = alpha * A[i]
template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> operator*(Scalar alpha, const DenseMatrix<Scalar, Device::CPU>& A)
{
    DenseMatrix<Scalar, Device::CPU> C(A.rows(), A.cols());
    Index n = A.size();
    if (detail::shouldUseOpenMp(n))
    {
        #pragma omp parallel for
        for (Index i = 0; i < n; ++i)
        {
            C.data()[i] = alpha * A.data()[i];
        }
    }
    else
    {
        for (Index i = 0; i < n; ++i)
        {
            C.data()[i] = alpha * A.data()[i];
        }
    }
    return C;
}

/// Scalar multiplication: A * alpha (element-wise). Delegates to alpha * A.
/// @tparam Scalar  Element type (float or double)
/// @param A      CPU matrix
/// @param alpha  Scalar multiplier
/// @return  New CPU matrix C where C[i] = A[i] * alpha
template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> operator*(const DenseMatrix<Scalar, Device::CPU>& A, Scalar alpha)
{
    return alpha * A;
}

/// Scalar addition: alpha + A (element-wise).
/// @tparam Scalar  Element type (float or double)
/// @param alpha  Scalar addend
/// @param A      CPU matrix
/// @return  New CPU matrix C where C[i] = alpha + A[i]
template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> operator+(Scalar alpha, const DenseMatrix<Scalar, Device::CPU>& A)
{
    DenseMatrix<Scalar, Device::CPU> C(A.rows(), A.cols());
    Index n = A.size();
    if (detail::shouldUseOpenMp(n))
    {
        #pragma omp parallel for
        for (Index i = 0; i < n; ++i)
        {
            C.data()[i] = alpha + A.data()[i];
        }
    }
    else
    {
        for (Index i = 0; i < n; ++i)
        {
            C.data()[i] = alpha + A.data()[i];
        }
    }
    return C;
}

} // namespace plamatrix
