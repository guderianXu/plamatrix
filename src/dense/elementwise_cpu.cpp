#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <omp.h>

#include "plamatrix/core/parallel.h"
#include "plamatrix/dense/dense_ops.h"
#include "plamatrix/dense/elementwise.h"

namespace plamatrix
{
namespace
{

template <typename Scalar, typename Operation>
DenseMatrix<Scalar, Device::CPU> transformElements(
    const DenseMatrix<Scalar, Device::CPU>& input,
    Operation operation)
{
    DenseMatrix<Scalar, Device::CPU> output(input.rows(), input.cols());
    const Index count = input.size();
    if (detail::shouldUseOpenMp(count))
    {
        #pragma omp parallel for
        for (Index index = 0; index < count; ++index)
        {
            output.data()[index] = operation(input.data()[index]);
        }
    }
    else
    {
        for (Index index = 0; index < count; ++index)
        {
            output.data()[index] = operation(input.data()[index]);
        }
    }
    return output;
}

template <typename Scalar, typename Operation>
DenseMatrix<Scalar, Device::CPU> transformElements(
    const DenseMatrix<Scalar, Device::CPU>& lhs,
    const DenseMatrix<Scalar, Device::CPU>& rhs,
    Operation operation)
{
    DenseMatrix<Scalar, Device::CPU> output(lhs.rows(), lhs.cols());
    const Index count = lhs.size();
    if (detail::shouldUseOpenMp(count))
    {
        #pragma omp parallel for
        for (Index index = 0; index < count; ++index)
        {
            output.data()[index] = operation(lhs.data()[index], rhs.data()[index]);
        }
    }
    else
    {
        for (Index index = 0; index < count; ++index)
        {
            output.data()[index] = operation(lhs.data()[index], rhs.data()[index]);
        }
    }
    return output;
}

template <ElementwiseUnaryOp Operation, typename Scalar>
DenseMatrix<Scalar, Device::CPU> applyUnary(
    const DenseMatrix<Scalar, Device::CPU>& input)
{
    if constexpr (Operation == ElementwiseUnaryOp::Abs)
    {
        return transformElements(input, [](Scalar value) { return std::abs(value); });
    }
    else
    {
        return transformElements(input, [](Scalar value) { return std::sqrt(value); });
    }
}

} // anonymous namespace

template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> scalarMultiply(
    const DenseMatrix<Scalar, Device::CPU>& input,
    Scalar value)
{
    return transformElements(input, [value](Scalar element) { return element * value; });
}

template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> scalarAdd(
    const DenseMatrix<Scalar, Device::CPU>& input,
    Scalar value)
{
    return transformElements(input, [value](Scalar element) { return element + value; });
}

template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> scalarDivide(
    const DenseMatrix<Scalar, Device::CPU>& input,
    Scalar value)
{
    if (value == Scalar(0))
    {
        throw std::domain_error("scalarDivide: divisor must be non-zero");
    }
    return transformElements(input, [value](Scalar element) { return element / value; });
}

template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> hadamardMultiply(
    const DenseMatrix<Scalar, Device::CPU>& lhs,
    const DenseMatrix<Scalar, Device::CPU>& rhs)
{
    detail::checkSameDimensions("hadamardMultiply", lhs, rhs);
    return transformElements(lhs, rhs, [](Scalar left, Scalar right) { return left * right; });
}

template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> hadamardDivide(
    const DenseMatrix<Scalar, Device::CPU>& lhs,
    const DenseMatrix<Scalar, Device::CPU>& rhs)
{
    detail::checkSameDimensions("hadamardDivide", lhs, rhs);
    return transformElements(lhs, rhs, [](Scalar left, Scalar right) { return left / right; });
}

template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> absElements(
    const DenseMatrix<Scalar, Device::CPU>& input)
{
    return applyUnary<ElementwiseUnaryOp::Abs>(input);
}

template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> sqrtElements(
    const DenseMatrix<Scalar, Device::CPU>& input)
{
    return applyUnary<ElementwiseUnaryOp::Sqrt>(input);
}

template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> clampElements(
    const DenseMatrix<Scalar, Device::CPU>& input,
    Scalar min_value,
    Scalar max_value)
{
    if (min_value > max_value)
    {
        throw std::invalid_argument("clampElements: min_value must not exceed max_value");
    }
    return transformElements(input, [min_value, max_value](Scalar element) {
        return std::clamp(element, min_value, max_value);
    });
}

#ifdef PLAMATRIX_USE_FLOAT
template DenseMatrix<float, Device::CPU> scalarMultiply(const DenseMatrix<float, Device::CPU>&, float);
template DenseMatrix<float, Device::CPU> scalarAdd(const DenseMatrix<float, Device::CPU>&, float);
template DenseMatrix<float, Device::CPU> scalarDivide(const DenseMatrix<float, Device::CPU>&, float);
template DenseMatrix<float, Device::CPU> hadamardMultiply(
    const DenseMatrix<float, Device::CPU>&,
    const DenseMatrix<float, Device::CPU>&);
template DenseMatrix<float, Device::CPU> hadamardDivide(
    const DenseMatrix<float, Device::CPU>&,
    const DenseMatrix<float, Device::CPU>&);
template DenseMatrix<float, Device::CPU> absElements(const DenseMatrix<float, Device::CPU>&);
template DenseMatrix<float, Device::CPU> sqrtElements(const DenseMatrix<float, Device::CPU>&);
template DenseMatrix<float, Device::CPU> clampElements(
    const DenseMatrix<float, Device::CPU>&,
    float,
    float);
#endif

#ifdef PLAMATRIX_USE_DOUBLE
template DenseMatrix<double, Device::CPU> scalarMultiply(const DenseMatrix<double, Device::CPU>&, double);
template DenseMatrix<double, Device::CPU> scalarAdd(const DenseMatrix<double, Device::CPU>&, double);
template DenseMatrix<double, Device::CPU> scalarDivide(const DenseMatrix<double, Device::CPU>&, double);
template DenseMatrix<double, Device::CPU> hadamardMultiply(
    const DenseMatrix<double, Device::CPU>&,
    const DenseMatrix<double, Device::CPU>&);
template DenseMatrix<double, Device::CPU> hadamardDivide(
    const DenseMatrix<double, Device::CPU>&,
    const DenseMatrix<double, Device::CPU>&);
template DenseMatrix<double, Device::CPU> absElements(const DenseMatrix<double, Device::CPU>&);
template DenseMatrix<double, Device::CPU> sqrtElements(const DenseMatrix<double, Device::CPU>&);
template DenseMatrix<double, Device::CPU> clampElements(
    const DenseMatrix<double, Device::CPU>&,
    double,
    double);
#endif

} // namespace plamatrix
