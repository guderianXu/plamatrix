#pragma once

#include <stdexcept>

#include "plamatrix/dense/dense_matrix.h"

namespace plamatrix
{

/// Unary operations supported by element-wise transforms.
enum class ElementwiseUnaryOp
{
    Abs,
    Sqrt
};

/// Multiply every CPU matrix element by a scalar.
/// @param input  Source matrix
/// @param value  Scalar multiplier
/// @return Newly allocated matrix with the same dimensions as input
template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> scalarMultiply(
    const DenseMatrix<Scalar, Device::CPU>& input,
    Scalar value);

/// Add a scalar to every CPU matrix element.
/// @param input  Source matrix
/// @param value  Scalar addend
/// @return Newly allocated matrix with the same dimensions as input
template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> scalarAdd(
    const DenseMatrix<Scalar, Device::CPU>& input,
    Scalar value);

/// Divide every CPU matrix element by a non-zero scalar.
/// @param input  Source matrix
/// @param value  Scalar divisor
/// @return Newly allocated matrix with the same dimensions as input
/// @throws std::domain_error if value is zero
template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> scalarDivide(
    const DenseMatrix<Scalar, Device::CPU>& input,
    Scalar value);

/// Multiply corresponding elements of two CPU matrices.
/// @param lhs  Left operand
/// @param rhs  Right operand with the same dimensions as lhs
/// @return Newly allocated matrix containing element-wise products
/// @throws std::runtime_error if operand dimensions differ
template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> hadamardMultiply(
    const DenseMatrix<Scalar, Device::CPU>& lhs,
    const DenseMatrix<Scalar, Device::CPU>& rhs);

/// Divide corresponding elements of two CPU matrices using IEEE floating-point semantics.
/// @param lhs  Numerator matrix
/// @param rhs  Denominator matrix with the same dimensions as lhs
/// @return Newly allocated matrix containing element-wise quotients
/// @throws std::runtime_error if operand dimensions differ
template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> hadamardDivide(
    const DenseMatrix<Scalar, Device::CPU>& lhs,
    const DenseMatrix<Scalar, Device::CPU>& rhs);

/// Compute the absolute value of every CPU matrix element.
/// @param input  Source matrix
/// @return Newly allocated matrix with absolute element values
template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> absElements(
    const DenseMatrix<Scalar, Device::CPU>& input);

/// Compute the square root of every CPU matrix element using IEEE floating-point semantics.
/// @param input  Source matrix
/// @return Newly allocated matrix with element-wise square roots
template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> sqrtElements(
    const DenseMatrix<Scalar, Device::CPU>& input);

/// Clamp every CPU matrix element to an inclusive range.
/// @param input  Source matrix
/// @param min_value  Inclusive lower bound
/// @param max_value  Inclusive upper bound
/// @return Newly allocated matrix with clamped element values
/// @throws std::invalid_argument if min_value is greater than max_value
template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> clampElements(
    const DenseMatrix<Scalar, Device::CPU>& input,
    Scalar min_value,
    Scalar max_value);

/// Multiply every GPU matrix element by a scalar.
/// Synchronous overloads wait for the operation on stream to finish before returning.
/// Async overloads enqueue work and return immediately. The caller owns any allocated return value;
/// input and output storage must remain alive until stream work completes.
template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> scalarMultiplyAsync(
    const DenseMatrix<Scalar, Device::GPU>& input,
    Scalar value,
    cudaStream_t stream = nullptr);

template <typename Scalar>
void scalarMultiplyAsync(
    const DenseMatrix<Scalar, Device::GPU>& input,
    Scalar value,
    DenseMatrix<Scalar, Device::GPU>& output,
    cudaStream_t stream = nullptr);

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> scalarMultiply(
    const DenseMatrix<Scalar, Device::GPU>& input,
    Scalar value,
    cudaStream_t stream = nullptr);

template <typename Scalar>
void scalarMultiply(
    const DenseMatrix<Scalar, Device::GPU>& input,
    Scalar value,
    DenseMatrix<Scalar, Device::GPU>& output,
    cudaStream_t stream = nullptr);

/// Add a scalar to every GPU matrix element.
/// Synchronous overloads wait for the operation on stream to finish before returning.
/// Async overloads enqueue work and return immediately. The caller owns any allocated return value;
/// input and output storage must remain alive until stream work completes.
template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> scalarAddAsync(
    const DenseMatrix<Scalar, Device::GPU>& input,
    Scalar value,
    cudaStream_t stream = nullptr);

template <typename Scalar>
void scalarAddAsync(
    const DenseMatrix<Scalar, Device::GPU>& input,
    Scalar value,
    DenseMatrix<Scalar, Device::GPU>& output,
    cudaStream_t stream = nullptr);

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> scalarAdd(
    const DenseMatrix<Scalar, Device::GPU>& input,
    Scalar value,
    cudaStream_t stream = nullptr);

template <typename Scalar>
void scalarAdd(
    const DenseMatrix<Scalar, Device::GPU>& input,
    Scalar value,
    DenseMatrix<Scalar, Device::GPU>& output,
    cudaStream_t stream = nullptr);

/// Divide every GPU matrix element by a non-zero scalar.
/// Synchronous overloads wait for the operation on stream to finish before returning.
/// Async overloads enqueue work and return immediately. The caller owns any allocated return value;
/// input and output storage must remain alive until stream work completes.
/// @throws std::domain_error if value is zero
template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> scalarDivideAsync(
    const DenseMatrix<Scalar, Device::GPU>& input,
    Scalar value,
    cudaStream_t stream = nullptr);

template <typename Scalar>
void scalarDivideAsync(
    const DenseMatrix<Scalar, Device::GPU>& input,
    Scalar value,
    DenseMatrix<Scalar, Device::GPU>& output,
    cudaStream_t stream = nullptr);

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> scalarDivide(
    const DenseMatrix<Scalar, Device::GPU>& input,
    Scalar value,
    cudaStream_t stream = nullptr);

template <typename Scalar>
void scalarDivide(
    const DenseMatrix<Scalar, Device::GPU>& input,
    Scalar value,
    DenseMatrix<Scalar, Device::GPU>& output,
    cudaStream_t stream = nullptr);

/// Multiply corresponding elements of two GPU matrices.
/// Synchronous overloads wait for the operation on stream to finish before returning.
/// Async overloads enqueue work and return immediately. The caller owns any allocated return value;
/// lhs, rhs, and output storage must remain alive until stream work completes.
template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> hadamardMultiplyAsync(
    const DenseMatrix<Scalar, Device::GPU>& lhs,
    const DenseMatrix<Scalar, Device::GPU>& rhs,
    cudaStream_t stream = nullptr);

template <typename Scalar>
void hadamardMultiplyAsync(
    const DenseMatrix<Scalar, Device::GPU>& lhs,
    const DenseMatrix<Scalar, Device::GPU>& rhs,
    DenseMatrix<Scalar, Device::GPU>& output,
    cudaStream_t stream = nullptr);

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> hadamardMultiply(
    const DenseMatrix<Scalar, Device::GPU>& lhs,
    const DenseMatrix<Scalar, Device::GPU>& rhs,
    cudaStream_t stream = nullptr);

template <typename Scalar>
void hadamardMultiply(
    const DenseMatrix<Scalar, Device::GPU>& lhs,
    const DenseMatrix<Scalar, Device::GPU>& rhs,
    DenseMatrix<Scalar, Device::GPU>& output,
    cudaStream_t stream = nullptr);

/// Divide corresponding elements of two GPU matrices using IEEE floating-point semantics.
/// Synchronous overloads wait for the operation on stream to finish before returning.
/// Async overloads enqueue work and return immediately. The caller owns any allocated return value;
/// lhs, rhs, and output storage must remain alive until stream work completes.
template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> hadamardDivideAsync(
    const DenseMatrix<Scalar, Device::GPU>& lhs,
    const DenseMatrix<Scalar, Device::GPU>& rhs,
    cudaStream_t stream = nullptr);

template <typename Scalar>
void hadamardDivideAsync(
    const DenseMatrix<Scalar, Device::GPU>& lhs,
    const DenseMatrix<Scalar, Device::GPU>& rhs,
    DenseMatrix<Scalar, Device::GPU>& output,
    cudaStream_t stream = nullptr);

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> hadamardDivide(
    const DenseMatrix<Scalar, Device::GPU>& lhs,
    const DenseMatrix<Scalar, Device::GPU>& rhs,
    cudaStream_t stream = nullptr);

template <typename Scalar>
void hadamardDivide(
    const DenseMatrix<Scalar, Device::GPU>& lhs,
    const DenseMatrix<Scalar, Device::GPU>& rhs,
    DenseMatrix<Scalar, Device::GPU>& output,
    cudaStream_t stream = nullptr);

/// Compute the absolute value of every GPU matrix element.
/// Synchronous overloads wait for the operation on stream to finish before returning.
/// Async overloads enqueue work and return immediately. The caller owns any allocated return value;
/// input and output storage must remain alive until stream work completes.
template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> absElementsAsync(
    const DenseMatrix<Scalar, Device::GPU>& input,
    cudaStream_t stream = nullptr);

template <typename Scalar>
void absElementsAsync(
    const DenseMatrix<Scalar, Device::GPU>& input,
    DenseMatrix<Scalar, Device::GPU>& output,
    cudaStream_t stream = nullptr);

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> absElements(
    const DenseMatrix<Scalar, Device::GPU>& input,
    cudaStream_t stream = nullptr);

template <typename Scalar>
void absElements(
    const DenseMatrix<Scalar, Device::GPU>& input,
    DenseMatrix<Scalar, Device::GPU>& output,
    cudaStream_t stream = nullptr);

/// Compute the square root of every GPU matrix element using IEEE floating-point semantics.
/// Synchronous overloads wait for the operation on stream to finish before returning.
/// Async overloads enqueue work and return immediately. The caller owns any allocated return value;
/// input and output storage must remain alive until stream work completes.
template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> sqrtElementsAsync(
    const DenseMatrix<Scalar, Device::GPU>& input,
    cudaStream_t stream = nullptr);

template <typename Scalar>
void sqrtElementsAsync(
    const DenseMatrix<Scalar, Device::GPU>& input,
    DenseMatrix<Scalar, Device::GPU>& output,
    cudaStream_t stream = nullptr);

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> sqrtElements(
    const DenseMatrix<Scalar, Device::GPU>& input,
    cudaStream_t stream = nullptr);

template <typename Scalar>
void sqrtElements(
    const DenseMatrix<Scalar, Device::GPU>& input,
    DenseMatrix<Scalar, Device::GPU>& output,
    cudaStream_t stream = nullptr);

/// Clamp every GPU matrix element to an inclusive range.
/// Synchronous overloads wait for the operation on stream to finish before returning.
/// Async overloads enqueue work and return immediately. The caller owns any allocated return value;
/// input and output storage must remain alive until stream work completes.
/// @throws std::invalid_argument if min_value is greater than max_value
template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> clampElementsAsync(
    const DenseMatrix<Scalar, Device::GPU>& input,
    Scalar min_value,
    Scalar max_value,
    cudaStream_t stream = nullptr);

template <typename Scalar>
void clampElementsAsync(
    const DenseMatrix<Scalar, Device::GPU>& input,
    Scalar min_value,
    Scalar max_value,
    DenseMatrix<Scalar, Device::GPU>& output,
    cudaStream_t stream = nullptr);

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> clampElements(
    const DenseMatrix<Scalar, Device::GPU>& input,
    Scalar min_value,
    Scalar max_value,
    cudaStream_t stream = nullptr);

template <typename Scalar>
void clampElements(
    const DenseMatrix<Scalar, Device::GPU>& input,
    Scalar min_value,
    Scalar max_value,
    DenseMatrix<Scalar, Device::GPU>& output,
    cudaStream_t stream = nullptr);

#ifdef PLAMATRIX_NO_CUDA
namespace detail
{

[[noreturn]] inline void throwElementwiseNoCuda(const char* message)
{
    throw std::runtime_error(message);
}

} // namespace detail

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> scalarMultiplyAsync(
    const DenseMatrix<Scalar, Device::GPU>&, Scalar, cudaStream_t)
{
    detail::throwElementwiseNoCuda("scalarMultiplyAsync requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
void scalarMultiplyAsync(const DenseMatrix<Scalar, Device::GPU>&, Scalar,
                         DenseMatrix<Scalar, Device::GPU>&, cudaStream_t)
{
    detail::throwElementwiseNoCuda("scalarMultiplyAsync requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> scalarMultiply(
    const DenseMatrix<Scalar, Device::GPU>&, Scalar, cudaStream_t)
{
    detail::throwElementwiseNoCuda("scalarMultiply requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
void scalarMultiply(const DenseMatrix<Scalar, Device::GPU>&, Scalar,
                    DenseMatrix<Scalar, Device::GPU>&, cudaStream_t)
{
    detail::throwElementwiseNoCuda("scalarMultiply requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> scalarAddAsync(
    const DenseMatrix<Scalar, Device::GPU>&, Scalar, cudaStream_t)
{
    detail::throwElementwiseNoCuda("scalarAddAsync requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
void scalarAddAsync(const DenseMatrix<Scalar, Device::GPU>&, Scalar,
                    DenseMatrix<Scalar, Device::GPU>&, cudaStream_t)
{
    detail::throwElementwiseNoCuda("scalarAddAsync requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> scalarAdd(
    const DenseMatrix<Scalar, Device::GPU>&, Scalar, cudaStream_t)
{
    detail::throwElementwiseNoCuda("scalarAdd requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
void scalarAdd(const DenseMatrix<Scalar, Device::GPU>&, Scalar,
               DenseMatrix<Scalar, Device::GPU>&, cudaStream_t)
{
    detail::throwElementwiseNoCuda("scalarAdd requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> scalarDivideAsync(
    const DenseMatrix<Scalar, Device::GPU>&, Scalar, cudaStream_t)
{
    detail::throwElementwiseNoCuda("scalarDivideAsync requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
void scalarDivideAsync(const DenseMatrix<Scalar, Device::GPU>&, Scalar,
                       DenseMatrix<Scalar, Device::GPU>&, cudaStream_t)
{
    detail::throwElementwiseNoCuda("scalarDivideAsync requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> scalarDivide(
    const DenseMatrix<Scalar, Device::GPU>&, Scalar, cudaStream_t)
{
    detail::throwElementwiseNoCuda("scalarDivide requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
void scalarDivide(const DenseMatrix<Scalar, Device::GPU>&, Scalar,
                  DenseMatrix<Scalar, Device::GPU>&, cudaStream_t)
{
    detail::throwElementwiseNoCuda("scalarDivide requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> hadamardMultiplyAsync(
    const DenseMatrix<Scalar, Device::GPU>&, const DenseMatrix<Scalar, Device::GPU>&, cudaStream_t)
{
    detail::throwElementwiseNoCuda("hadamardMultiplyAsync requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
void hadamardMultiplyAsync(const DenseMatrix<Scalar, Device::GPU>&,
                           const DenseMatrix<Scalar, Device::GPU>&,
                           DenseMatrix<Scalar, Device::GPU>&, cudaStream_t)
{
    detail::throwElementwiseNoCuda("hadamardMultiplyAsync requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> hadamardMultiply(
    const DenseMatrix<Scalar, Device::GPU>&, const DenseMatrix<Scalar, Device::GPU>&, cudaStream_t)
{
    detail::throwElementwiseNoCuda("hadamardMultiply requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
void hadamardMultiply(const DenseMatrix<Scalar, Device::GPU>&,
                      const DenseMatrix<Scalar, Device::GPU>&,
                      DenseMatrix<Scalar, Device::GPU>&, cudaStream_t)
{
    detail::throwElementwiseNoCuda("hadamardMultiply requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> hadamardDivideAsync(
    const DenseMatrix<Scalar, Device::GPU>&, const DenseMatrix<Scalar, Device::GPU>&, cudaStream_t)
{
    detail::throwElementwiseNoCuda("hadamardDivideAsync requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
void hadamardDivideAsync(const DenseMatrix<Scalar, Device::GPU>&,
                         const DenseMatrix<Scalar, Device::GPU>&,
                         DenseMatrix<Scalar, Device::GPU>&, cudaStream_t)
{
    detail::throwElementwiseNoCuda("hadamardDivideAsync requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> hadamardDivide(
    const DenseMatrix<Scalar, Device::GPU>&, const DenseMatrix<Scalar, Device::GPU>&, cudaStream_t)
{
    detail::throwElementwiseNoCuda("hadamardDivide requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
void hadamardDivide(const DenseMatrix<Scalar, Device::GPU>&,
                    const DenseMatrix<Scalar, Device::GPU>&,
                    DenseMatrix<Scalar, Device::GPU>&, cudaStream_t)
{
    detail::throwElementwiseNoCuda("hadamardDivide requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> absElementsAsync(
    const DenseMatrix<Scalar, Device::GPU>&, cudaStream_t)
{
    detail::throwElementwiseNoCuda("absElementsAsync requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
void absElementsAsync(const DenseMatrix<Scalar, Device::GPU>&,
                      DenseMatrix<Scalar, Device::GPU>&, cudaStream_t)
{
    detail::throwElementwiseNoCuda("absElementsAsync requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> absElements(
    const DenseMatrix<Scalar, Device::GPU>&, cudaStream_t)
{
    detail::throwElementwiseNoCuda("absElements requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
void absElements(const DenseMatrix<Scalar, Device::GPU>&,
                 DenseMatrix<Scalar, Device::GPU>&, cudaStream_t)
{
    detail::throwElementwiseNoCuda("absElements requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> sqrtElementsAsync(
    const DenseMatrix<Scalar, Device::GPU>&, cudaStream_t)
{
    detail::throwElementwiseNoCuda("sqrtElementsAsync requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
void sqrtElementsAsync(const DenseMatrix<Scalar, Device::GPU>&,
                       DenseMatrix<Scalar, Device::GPU>&, cudaStream_t)
{
    detail::throwElementwiseNoCuda("sqrtElementsAsync requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> sqrtElements(
    const DenseMatrix<Scalar, Device::GPU>&, cudaStream_t)
{
    detail::throwElementwiseNoCuda("sqrtElements requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
void sqrtElements(const DenseMatrix<Scalar, Device::GPU>&,
                  DenseMatrix<Scalar, Device::GPU>&, cudaStream_t)
{
    detail::throwElementwiseNoCuda("sqrtElements requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> clampElementsAsync(
    const DenseMatrix<Scalar, Device::GPU>&, Scalar, Scalar, cudaStream_t)
{
    detail::throwElementwiseNoCuda("clampElementsAsync requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
void clampElementsAsync(const DenseMatrix<Scalar, Device::GPU>&, Scalar, Scalar,
                        DenseMatrix<Scalar, Device::GPU>&, cudaStream_t)
{
    detail::throwElementwiseNoCuda("clampElementsAsync requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> clampElements(
    const DenseMatrix<Scalar, Device::GPU>&, Scalar, Scalar, cudaStream_t)
{
    detail::throwElementwiseNoCuda("clampElements requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
void clampElements(const DenseMatrix<Scalar, Device::GPU>&, Scalar, Scalar,
                   DenseMatrix<Scalar, Device::GPU>&, cudaStream_t)
{
    detail::throwElementwiseNoCuda("clampElements requires PLAMATRIX_WITH_CUDA=ON");
}
#endif

} // namespace plamatrix
