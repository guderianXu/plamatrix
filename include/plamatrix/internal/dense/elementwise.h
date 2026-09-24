#pragma once

#include <stdexcept>

#include "plamatrix/internal/dense/dense_storage.h"

namespace plamatrix::internal
{

/// Unary operations supported by element-wise transforms.
enum class ElementwiseUnaryOp
{
    Abs,
    Sqrt
};

/// Compute alpha * lhs + beta * rhs in one CPU pass.
/// @return Newly allocated matrix with the same dimensions as the inputs
/// @throws std::runtime_error if operand dimensions differ
template <typename Scalar>
DenseStorage<Scalar, Device::CPU> axpby(
    Scalar alpha,
    const DenseStorage<Scalar, Device::CPU>& lhs,
    Scalar beta,
    const DenseStorage<Scalar, Device::CPU>& rhs);

/// Compute alpha * lhs + beta * rhs into caller-owned CPU storage.
/// Exact aliasing with lhs or rhs is supported; dimensions are validated before output is modified.
template <typename Scalar>
void axpby(
    Scalar alpha,
    const DenseStorage<Scalar, Device::CPU>& lhs,
    Scalar beta,
    const DenseStorage<Scalar, Device::CPU>& rhs,
    DenseStorage<Scalar, Device::CPU>& output);

/// Compute alpha * lhs + beta * rhs for arbitrary-stride CPU views.
/// Exact input/output mappings may alias. Other overlapping mappings are rejected before writing.
/// Output elements must map to distinct addresses.
template <typename Scalar>
void axpby(
    Scalar alpha,
    ConstMatrixView<Scalar, Device::CPU> lhs,
    Scalar beta,
    ConstMatrixView<Scalar, Device::CPU> rhs,
    MatrixView<Scalar, Device::CPU> output);

/// Compute alpha * first + beta * second + gamma * third in one CPU pass.
template <typename Scalar>
DenseStorage<Scalar, Device::CPU> linearCombination(
    Scalar alpha,
    const DenseStorage<Scalar, Device::CPU>& first,
    Scalar beta,
    const DenseStorage<Scalar, Device::CPU>& second,
    Scalar gamma,
    const DenseStorage<Scalar, Device::CPU>& third);

/// Compute a three-term linear combination into caller-owned CPU storage.
/// Exact aliasing with any input is supported; dimensions are validated before writing.
template <typename Scalar>
void linearCombination(
    Scalar alpha,
    const DenseStorage<Scalar, Device::CPU>& first,
    Scalar beta,
    const DenseStorage<Scalar, Device::CPU>& second,
    Scalar gamma,
    const DenseStorage<Scalar, Device::CPU>& third,
    DenseStorage<Scalar, Device::CPU>& output);

/// Compute a three-term linear combination for arbitrary-positive-stride CPU views.
/// Exact input/output mappings may alias. Other overlap and self-overlapping output are rejected.
template <typename Scalar>
void linearCombination(
    Scalar alpha,
    ConstMatrixView<Scalar, Device::CPU> first,
    Scalar beta,
    ConstMatrixView<Scalar, Device::CPU> second,
    Scalar gamma,
    ConstMatrixView<Scalar, Device::CPU> third,
    MatrixView<Scalar, Device::CPU> output);

/// Multiply every CPU matrix element by a scalar.
/// @param input  Source matrix
/// @param value  Scalar multiplier
/// @return Newly allocated matrix with the same dimensions as input
template <typename Scalar>
DenseStorage<Scalar, Device::CPU> scalarMultiply(
    const DenseStorage<Scalar, Device::CPU>& input,
    Scalar value);

/// Add a scalar to every CPU matrix element.
/// @param input  Source matrix
/// @param value  Scalar addend
/// @return Newly allocated matrix with the same dimensions as input
template <typename Scalar>
DenseStorage<Scalar, Device::CPU> scalarAdd(
    const DenseStorage<Scalar, Device::CPU>& input,
    Scalar value);

/// Divide every CPU matrix element by a non-zero scalar.
/// @param input  Source matrix
/// @param value  Scalar divisor
/// @return Newly allocated matrix with the same dimensions as input
/// @throws std::domain_error if value is zero
template <typename Scalar>
DenseStorage<Scalar, Device::CPU> scalarDivide(
    const DenseStorage<Scalar, Device::CPU>& input,
    Scalar value);

/// Multiply corresponding elements of two CPU matrices.
/// @param lhs  Left operand
/// @param rhs  Right operand with the same dimensions as lhs
/// @return Newly allocated matrix containing element-wise products
/// @throws std::runtime_error if operand dimensions differ
template <typename Scalar>
DenseStorage<Scalar, Device::CPU> hadamardMultiply(
    const DenseStorage<Scalar, Device::CPU>& lhs,
    const DenseStorage<Scalar, Device::CPU>& rhs);

/// Divide corresponding elements of two CPU matrices using IEEE floating-point semantics.
/// @param lhs  Numerator matrix
/// @param rhs  Denominator matrix with the same dimensions as lhs
/// @return Newly allocated matrix containing element-wise quotients
/// @throws std::runtime_error if operand dimensions differ
template <typename Scalar>
DenseStorage<Scalar, Device::CPU> hadamardDivide(
    const DenseStorage<Scalar, Device::CPU>& lhs,
    const DenseStorage<Scalar, Device::CPU>& rhs);

/// Compute the absolute value of every CPU matrix element.
/// @param input  Source matrix
/// @return Newly allocated matrix with absolute element values
template <typename Scalar>
DenseStorage<Scalar, Device::CPU> absElements(
    const DenseStorage<Scalar, Device::CPU>& input);

/// Compute the square root of every CPU matrix element using IEEE floating-point semantics.
/// @param input  Source matrix
/// @return Newly allocated matrix with element-wise square roots
template <typename Scalar>
DenseStorage<Scalar, Device::CPU> sqrtElements(
    const DenseStorage<Scalar, Device::CPU>& input);

/// Clamp every CPU matrix element to an inclusive range.
/// @param input  Source matrix
/// @param min_value  Inclusive lower bound
/// @param max_value  Inclusive upper bound
/// @return Newly allocated matrix with clamped element values
/// @throws std::invalid_argument if min_value is greater than max_value
template <typename Scalar>
DenseStorage<Scalar, Device::CPU> clampElements(
    const DenseStorage<Scalar, Device::CPU>& input,
    Scalar min_value,
    Scalar max_value);

/// Multiply every GPU matrix element by a scalar.
/// Synchronous overloads wait for the operation on stream to finish before returning.
/// Async overloads enqueue work and return immediately. The caller owns any allocated return value;
/// input and output storage must remain alive until stream work completes.
template <typename Scalar>
DenseStorage<Scalar, Device::GPU> scalarMultiplyAsync(
    const DenseStorage<Scalar, Device::GPU>& input,
    Scalar value,
    cudaStream_t stream = nullptr);

template <typename Scalar>
void scalarMultiplyAsync(
    const DenseStorage<Scalar, Device::GPU>& input,
    Scalar value,
    DenseStorage<Scalar, Device::GPU>& output,
    cudaStream_t stream = nullptr);

template <typename Scalar>
DenseStorage<Scalar, Device::GPU> scalarMultiply(
    const DenseStorage<Scalar, Device::GPU>& input,
    Scalar value,
    cudaStream_t stream = nullptr);

template <typename Scalar>
void scalarMultiply(
    const DenseStorage<Scalar, Device::GPU>& input,
    Scalar value,
    DenseStorage<Scalar, Device::GPU>& output,
    cudaStream_t stream = nullptr);

/// Add a scalar to every GPU matrix element.
/// Synchronous overloads wait for the operation on stream to finish before returning.
/// Async overloads enqueue work and return immediately. The caller owns any allocated return value;
/// input and output storage must remain alive until stream work completes.
template <typename Scalar>
DenseStorage<Scalar, Device::GPU> scalarAddAsync(
    const DenseStorage<Scalar, Device::GPU>& input,
    Scalar value,
    cudaStream_t stream = nullptr);

template <typename Scalar>
void scalarAddAsync(
    const DenseStorage<Scalar, Device::GPU>& input,
    Scalar value,
    DenseStorage<Scalar, Device::GPU>& output,
    cudaStream_t stream = nullptr);

template <typename Scalar>
DenseStorage<Scalar, Device::GPU> scalarAdd(
    const DenseStorage<Scalar, Device::GPU>& input,
    Scalar value,
    cudaStream_t stream = nullptr);

template <typename Scalar>
void scalarAdd(
    const DenseStorage<Scalar, Device::GPU>& input,
    Scalar value,
    DenseStorage<Scalar, Device::GPU>& output,
    cudaStream_t stream = nullptr);

/// Divide every GPU matrix element by a non-zero scalar.
/// Synchronous overloads wait for the operation on stream to finish before returning.
/// Async overloads enqueue work and return immediately. The caller owns any allocated return value;
/// input and output storage must remain alive until stream work completes.
/// @throws std::domain_error if value is zero
template <typename Scalar>
DenseStorage<Scalar, Device::GPU> scalarDivideAsync(
    const DenseStorage<Scalar, Device::GPU>& input,
    Scalar value,
    cudaStream_t stream = nullptr);

template <typename Scalar>
void scalarDivideAsync(
    const DenseStorage<Scalar, Device::GPU>& input,
    Scalar value,
    DenseStorage<Scalar, Device::GPU>& output,
    cudaStream_t stream = nullptr);

template <typename Scalar>
DenseStorage<Scalar, Device::GPU> scalarDivide(
    const DenseStorage<Scalar, Device::GPU>& input,
    Scalar value,
    cudaStream_t stream = nullptr);

template <typename Scalar>
void scalarDivide(
    const DenseStorage<Scalar, Device::GPU>& input,
    Scalar value,
    DenseStorage<Scalar, Device::GPU>& output,
    cudaStream_t stream = nullptr);

/// Multiply corresponding elements of two GPU matrices.
/// Synchronous overloads wait for the operation on stream to finish before returning.
/// Async overloads enqueue work and return immediately. The caller owns any allocated return value;
/// lhs, rhs, and output storage must remain alive until stream work completes.
template <typename Scalar>
DenseStorage<Scalar, Device::GPU> hadamardMultiplyAsync(
    const DenseStorage<Scalar, Device::GPU>& lhs,
    const DenseStorage<Scalar, Device::GPU>& rhs,
    cudaStream_t stream = nullptr);

template <typename Scalar>
void hadamardMultiplyAsync(
    const DenseStorage<Scalar, Device::GPU>& lhs,
    const DenseStorage<Scalar, Device::GPU>& rhs,
    DenseStorage<Scalar, Device::GPU>& output,
    cudaStream_t stream = nullptr);

template <typename Scalar>
DenseStorage<Scalar, Device::GPU> hadamardMultiply(
    const DenseStorage<Scalar, Device::GPU>& lhs,
    const DenseStorage<Scalar, Device::GPU>& rhs,
    cudaStream_t stream = nullptr);

template <typename Scalar>
void hadamardMultiply(
    const DenseStorage<Scalar, Device::GPU>& lhs,
    const DenseStorage<Scalar, Device::GPU>& rhs,
    DenseStorage<Scalar, Device::GPU>& output,
    cudaStream_t stream = nullptr);

/// Divide corresponding elements of two GPU matrices using IEEE floating-point semantics.
/// Synchronous overloads wait for the operation on stream to finish before returning.
/// Async overloads enqueue work and return immediately. The caller owns any allocated return value;
/// lhs, rhs, and output storage must remain alive until stream work completes.
template <typename Scalar>
DenseStorage<Scalar, Device::GPU> hadamardDivideAsync(
    const DenseStorage<Scalar, Device::GPU>& lhs,
    const DenseStorage<Scalar, Device::GPU>& rhs,
    cudaStream_t stream = nullptr);

template <typename Scalar>
void hadamardDivideAsync(
    const DenseStorage<Scalar, Device::GPU>& lhs,
    const DenseStorage<Scalar, Device::GPU>& rhs,
    DenseStorage<Scalar, Device::GPU>& output,
    cudaStream_t stream = nullptr);

template <typename Scalar>
DenseStorage<Scalar, Device::GPU> hadamardDivide(
    const DenseStorage<Scalar, Device::GPU>& lhs,
    const DenseStorage<Scalar, Device::GPU>& rhs,
    cudaStream_t stream = nullptr);

template <typename Scalar>
void hadamardDivide(
    const DenseStorage<Scalar, Device::GPU>& lhs,
    const DenseStorage<Scalar, Device::GPU>& rhs,
    DenseStorage<Scalar, Device::GPU>& output,
    cudaStream_t stream = nullptr);

/// Compute the absolute value of every GPU matrix element.
/// Synchronous overloads wait for the operation on stream to finish before returning.
/// Async overloads enqueue work and return immediately. The caller owns any allocated return value;
/// input and output storage must remain alive until stream work completes.
template <typename Scalar>
DenseStorage<Scalar, Device::GPU> absElementsAsync(
    const DenseStorage<Scalar, Device::GPU>& input,
    cudaStream_t stream = nullptr);

template <typename Scalar>
void absElementsAsync(
    const DenseStorage<Scalar, Device::GPU>& input,
    DenseStorage<Scalar, Device::GPU>& output,
    cudaStream_t stream = nullptr);

template <typename Scalar>
DenseStorage<Scalar, Device::GPU> absElements(
    const DenseStorage<Scalar, Device::GPU>& input,
    cudaStream_t stream = nullptr);

template <typename Scalar>
void absElements(
    const DenseStorage<Scalar, Device::GPU>& input,
    DenseStorage<Scalar, Device::GPU>& output,
    cudaStream_t stream = nullptr);

/// Compute the square root of every GPU matrix element using IEEE floating-point semantics.
/// Synchronous overloads wait for the operation on stream to finish before returning.
/// Async overloads enqueue work and return immediately. The caller owns any allocated return value;
/// input and output storage must remain alive until stream work completes.
template <typename Scalar>
DenseStorage<Scalar, Device::GPU> sqrtElementsAsync(
    const DenseStorage<Scalar, Device::GPU>& input,
    cudaStream_t stream = nullptr);

template <typename Scalar>
void sqrtElementsAsync(
    const DenseStorage<Scalar, Device::GPU>& input,
    DenseStorage<Scalar, Device::GPU>& output,
    cudaStream_t stream = nullptr);

template <typename Scalar>
DenseStorage<Scalar, Device::GPU> sqrtElements(
    const DenseStorage<Scalar, Device::GPU>& input,
    cudaStream_t stream = nullptr);

template <typename Scalar>
void sqrtElements(
    const DenseStorage<Scalar, Device::GPU>& input,
    DenseStorage<Scalar, Device::GPU>& output,
    cudaStream_t stream = nullptr);

/// Clamp every GPU matrix element to an inclusive range.
/// Synchronous overloads wait for the operation on stream to finish before returning.
/// Async overloads enqueue work and return immediately. The caller owns any allocated return value;
/// input and output storage must remain alive until stream work completes.
/// @throws std::invalid_argument if min_value is greater than max_value
template <typename Scalar>
DenseStorage<Scalar, Device::GPU> clampElementsAsync(
    const DenseStorage<Scalar, Device::GPU>& input,
    Scalar min_value,
    Scalar max_value,
    cudaStream_t stream = nullptr);

template <typename Scalar>
void clampElementsAsync(
    const DenseStorage<Scalar, Device::GPU>& input,
    Scalar min_value,
    Scalar max_value,
    DenseStorage<Scalar, Device::GPU>& output,
    cudaStream_t stream = nullptr);

template <typename Scalar>
DenseStorage<Scalar, Device::GPU> clampElements(
    const DenseStorage<Scalar, Device::GPU>& input,
    Scalar min_value,
    Scalar max_value,
    cudaStream_t stream = nullptr);

template <typename Scalar>
void clampElements(
    const DenseStorage<Scalar, Device::GPU>& input,
    Scalar min_value,
    Scalar max_value,
    DenseStorage<Scalar, Device::GPU>& output,
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
DenseStorage<Scalar, Device::GPU> scalarMultiplyAsync(
    const DenseStorage<Scalar, Device::GPU>&, Scalar, cudaStream_t)
{
    detail::throwElementwiseNoCuda("scalarMultiplyAsync requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
void scalarMultiplyAsync(const DenseStorage<Scalar, Device::GPU>&, Scalar,
                         DenseStorage<Scalar, Device::GPU>&, cudaStream_t)
{
    detail::throwElementwiseNoCuda("scalarMultiplyAsync requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
DenseStorage<Scalar, Device::GPU> scalarMultiply(
    const DenseStorage<Scalar, Device::GPU>&, Scalar, cudaStream_t)
{
    detail::throwElementwiseNoCuda("scalarMultiply requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
void scalarMultiply(const DenseStorage<Scalar, Device::GPU>&, Scalar,
                    DenseStorage<Scalar, Device::GPU>&, cudaStream_t)
{
    detail::throwElementwiseNoCuda("scalarMultiply requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
DenseStorage<Scalar, Device::GPU> scalarAddAsync(
    const DenseStorage<Scalar, Device::GPU>&, Scalar, cudaStream_t)
{
    detail::throwElementwiseNoCuda("scalarAddAsync requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
void scalarAddAsync(const DenseStorage<Scalar, Device::GPU>&, Scalar,
                    DenseStorage<Scalar, Device::GPU>&, cudaStream_t)
{
    detail::throwElementwiseNoCuda("scalarAddAsync requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
DenseStorage<Scalar, Device::GPU> scalarAdd(
    const DenseStorage<Scalar, Device::GPU>&, Scalar, cudaStream_t)
{
    detail::throwElementwiseNoCuda("scalarAdd requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
void scalarAdd(const DenseStorage<Scalar, Device::GPU>&, Scalar,
               DenseStorage<Scalar, Device::GPU>&, cudaStream_t)
{
    detail::throwElementwiseNoCuda("scalarAdd requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
DenseStorage<Scalar, Device::GPU> scalarDivideAsync(
    const DenseStorage<Scalar, Device::GPU>&, Scalar, cudaStream_t)
{
    detail::throwElementwiseNoCuda("scalarDivideAsync requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
void scalarDivideAsync(const DenseStorage<Scalar, Device::GPU>&, Scalar,
                       DenseStorage<Scalar, Device::GPU>&, cudaStream_t)
{
    detail::throwElementwiseNoCuda("scalarDivideAsync requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
DenseStorage<Scalar, Device::GPU> scalarDivide(
    const DenseStorage<Scalar, Device::GPU>&, Scalar, cudaStream_t)
{
    detail::throwElementwiseNoCuda("scalarDivide requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
void scalarDivide(const DenseStorage<Scalar, Device::GPU>&, Scalar,
                  DenseStorage<Scalar, Device::GPU>&, cudaStream_t)
{
    detail::throwElementwiseNoCuda("scalarDivide requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
DenseStorage<Scalar, Device::GPU> hadamardMultiplyAsync(
    const DenseStorage<Scalar, Device::GPU>&, const DenseStorage<Scalar, Device::GPU>&, cudaStream_t)
{
    detail::throwElementwiseNoCuda("hadamardMultiplyAsync requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
void hadamardMultiplyAsync(const DenseStorage<Scalar, Device::GPU>&,
                           const DenseStorage<Scalar, Device::GPU>&,
                           DenseStorage<Scalar, Device::GPU>&, cudaStream_t)
{
    detail::throwElementwiseNoCuda("hadamardMultiplyAsync requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
DenseStorage<Scalar, Device::GPU> hadamardMultiply(
    const DenseStorage<Scalar, Device::GPU>&, const DenseStorage<Scalar, Device::GPU>&, cudaStream_t)
{
    detail::throwElementwiseNoCuda("hadamardMultiply requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
void hadamardMultiply(const DenseStorage<Scalar, Device::GPU>&,
                      const DenseStorage<Scalar, Device::GPU>&,
                      DenseStorage<Scalar, Device::GPU>&, cudaStream_t)
{
    detail::throwElementwiseNoCuda("hadamardMultiply requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
DenseStorage<Scalar, Device::GPU> hadamardDivideAsync(
    const DenseStorage<Scalar, Device::GPU>&, const DenseStorage<Scalar, Device::GPU>&, cudaStream_t)
{
    detail::throwElementwiseNoCuda("hadamardDivideAsync requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
void hadamardDivideAsync(const DenseStorage<Scalar, Device::GPU>&,
                         const DenseStorage<Scalar, Device::GPU>&,
                         DenseStorage<Scalar, Device::GPU>&, cudaStream_t)
{
    detail::throwElementwiseNoCuda("hadamardDivideAsync requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
DenseStorage<Scalar, Device::GPU> hadamardDivide(
    const DenseStorage<Scalar, Device::GPU>&, const DenseStorage<Scalar, Device::GPU>&, cudaStream_t)
{
    detail::throwElementwiseNoCuda("hadamardDivide requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
void hadamardDivide(const DenseStorage<Scalar, Device::GPU>&,
                    const DenseStorage<Scalar, Device::GPU>&,
                    DenseStorage<Scalar, Device::GPU>&, cudaStream_t)
{
    detail::throwElementwiseNoCuda("hadamardDivide requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
DenseStorage<Scalar, Device::GPU> absElementsAsync(
    const DenseStorage<Scalar, Device::GPU>&, cudaStream_t)
{
    detail::throwElementwiseNoCuda("absElementsAsync requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
void absElementsAsync(const DenseStorage<Scalar, Device::GPU>&,
                      DenseStorage<Scalar, Device::GPU>&, cudaStream_t)
{
    detail::throwElementwiseNoCuda("absElementsAsync requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
DenseStorage<Scalar, Device::GPU> absElements(
    const DenseStorage<Scalar, Device::GPU>&, cudaStream_t)
{
    detail::throwElementwiseNoCuda("absElements requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
void absElements(const DenseStorage<Scalar, Device::GPU>&,
                 DenseStorage<Scalar, Device::GPU>&, cudaStream_t)
{
    detail::throwElementwiseNoCuda("absElements requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
DenseStorage<Scalar, Device::GPU> sqrtElementsAsync(
    const DenseStorage<Scalar, Device::GPU>&, cudaStream_t)
{
    detail::throwElementwiseNoCuda("sqrtElementsAsync requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
void sqrtElementsAsync(const DenseStorage<Scalar, Device::GPU>&,
                       DenseStorage<Scalar, Device::GPU>&, cudaStream_t)
{
    detail::throwElementwiseNoCuda("sqrtElementsAsync requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
DenseStorage<Scalar, Device::GPU> sqrtElements(
    const DenseStorage<Scalar, Device::GPU>&, cudaStream_t)
{
    detail::throwElementwiseNoCuda("sqrtElements requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
void sqrtElements(const DenseStorage<Scalar, Device::GPU>&,
                  DenseStorage<Scalar, Device::GPU>&, cudaStream_t)
{
    detail::throwElementwiseNoCuda("sqrtElements requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
DenseStorage<Scalar, Device::GPU> clampElementsAsync(
    const DenseStorage<Scalar, Device::GPU>&, Scalar, Scalar, cudaStream_t)
{
    detail::throwElementwiseNoCuda("clampElementsAsync requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
void clampElementsAsync(const DenseStorage<Scalar, Device::GPU>&, Scalar, Scalar,
                        DenseStorage<Scalar, Device::GPU>&, cudaStream_t)
{
    detail::throwElementwiseNoCuda("clampElementsAsync requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
DenseStorage<Scalar, Device::GPU> clampElements(
    const DenseStorage<Scalar, Device::GPU>&, Scalar, Scalar, cudaStream_t)
{
    detail::throwElementwiseNoCuda("clampElements requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
void clampElements(const DenseStorage<Scalar, Device::GPU>&, Scalar, Scalar,
                   DenseStorage<Scalar, Device::GPU>&, cudaStream_t)
{
    detail::throwElementwiseNoCuda("clampElements requires PLAMATRIX_WITH_CUDA=ON");
}
#endif

} // namespace plamatrix::internal
