#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

#include <omp.h>

#include "plamatrix/internal/core/parallel.h"
#include "plamatrix/internal/dense/dense_ops.h"
#include "plamatrix/internal/dense/elementwise.h"

namespace plamatrix::internal
{
namespace
{

template <typename Scalar, bool LeftConst, bool RightConst>
void checkSameViewDimensions(
    const char* operation,
    BasicMatrixView<Scalar, Device::CPU, LeftConst> lhs,
    BasicMatrixView<Scalar, Device::CPU, RightConst> rhs)
{
    if (lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols())
    {
        throw std::runtime_error(std::string(operation) + ": matrix dimensions must match");
    }
}

template <typename Scalar, bool IsConst>
std::pair<std::uintptr_t, std::uintptr_t> viewAddressRange(
    BasicMatrixView<Scalar, Device::CPU, IsConst> view)
{
    if (view.size() == 0)
    {
        return {0, 0};
    }
    const Index maximum_offset =
        (view.rows() - 1) * view.rowStride() + (view.cols() - 1) * view.colStride();
    const auto begin = reinterpret_cast<std::uintptr_t>(view.data());
    const auto element_size = static_cast<std::uintptr_t>(sizeof(Scalar));
    const auto maximum_address = std::numeric_limits<std::uintptr_t>::max();
    if (static_cast<std::uintmax_t>(maximum_offset) >
        static_cast<std::uintmax_t>((maximum_address - begin) / element_size))
    {
        throw std::overflow_error("elementwise fusion: matrix view address range overflows uintptr_t");
    }
    const auto last = begin + static_cast<std::uintptr_t>(maximum_offset) * element_size;
    if (last > maximum_address - element_size)
    {
        throw std::overflow_error("elementwise fusion: matrix view address range overflows uintptr_t");
    }
    return {begin, last + element_size};
}

template <typename Scalar, bool LeftConst, bool RightConst>
bool viewsOverlap(
    BasicMatrixView<Scalar, Device::CPU, LeftConst> lhs,
    BasicMatrixView<Scalar, Device::CPU, RightConst> rhs)
{
    const auto lhs_range = viewAddressRange(lhs);
    const auto rhs_range = viewAddressRange(rhs);
    return lhs_range.first < rhs_range.second && rhs_range.first < lhs_range.second;
}

template <typename Scalar, bool LeftConst, bool RightConst>
bool viewsHaveSameMapping(
    BasicMatrixView<Scalar, Device::CPU, LeftConst> lhs,
    BasicMatrixView<Scalar, Device::CPU, RightConst> rhs)
{
    return lhs.data() == rhs.data() && lhs.rows() == rhs.rows() && lhs.cols() == rhs.cols() &&
           lhs.rowStride() == rhs.rowStride() && lhs.colStride() == rhs.colStride();
}

template <typename Scalar>
bool viewHasUniqueOutputMapping(MatrixView<Scalar, Device::CPU> view)
{
    if (view.size() <= 1 || view.rows() <= 1 || view.cols() <= 1)
    {
        return true;
    }
    const Index stride_gcd = std::gcd(view.rowStride(), view.colStride());
    const Index minimum_colliding_row_delta = view.colStride() / stride_gcd;
    const Index minimum_colliding_col_delta = view.rowStride() / stride_gcd;
    return minimum_colliding_row_delta >= view.rows() || minimum_colliding_col_delta >= view.cols();
}

template <typename Scalar, bool IsConst>
bool viewAliasesOutputSafely(
    BasicMatrixView<Scalar, Device::CPU, IsConst> input,
    MatrixView<Scalar, Device::CPU> output)
{
    return !viewsOverlap(input, output) || viewsHaveSameMapping(input, output);
}

template <typename Scalar, typename... Inputs>
void validateOutputView(
    const char* operation,
    MatrixView<Scalar, Device::CPU> output,
    Inputs... inputs)
{
    if (!viewHasUniqueOutputMapping(output))
    {
        throw std::invalid_argument(
            std::string(operation) + ": output view elements must not overlap each other");
    }
    if (!(viewAliasesOutputSafely(inputs, output) && ...))
    {
        throw std::invalid_argument(
            std::string(operation) + ": output may only alias an input with the same view mapping");
    }
}

template <typename Scalar, typename Operation>
DenseStorage<Scalar, Device::CPU> transformElements(
    const DenseStorage<Scalar, Device::CPU>& input,
    Operation operation)
{
    DenseStorage<Scalar, Device::CPU> output(input.rows(), input.cols());
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
DenseStorage<Scalar, Device::CPU> transformElements(
    const DenseStorage<Scalar, Device::CPU>& lhs,
    const DenseStorage<Scalar, Device::CPU>& rhs,
    Operation operation)
{
    DenseStorage<Scalar, Device::CPU> output(lhs.rows(), lhs.cols());
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
DenseStorage<Scalar, Device::CPU> applyUnary(
    const DenseStorage<Scalar, Device::CPU>& input)
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
DenseStorage<Scalar, Device::CPU> axpby(
    Scalar alpha,
    const DenseStorage<Scalar, Device::CPU>& lhs,
    Scalar beta,
    const DenseStorage<Scalar, Device::CPU>& rhs)
{
    detail::checkSameDimensions("axpby", lhs, rhs);
    auto output = DenseStorage<Scalar, Device::CPU>::uninitialized(lhs.rows(), lhs.cols());
    axpby(alpha, lhs.view(), beta, rhs.view(), output.view());
    return output;
}

template <typename Scalar>
void axpby(
    Scalar alpha,
    const DenseStorage<Scalar, Device::CPU>& lhs,
    Scalar beta,
    const DenseStorage<Scalar, Device::CPU>& rhs,
    DenseStorage<Scalar, Device::CPU>& output)
{
    detail::checkSameDimensions("axpby", lhs, rhs);
    detail::checkSameDimensions("axpby", lhs, output);
    axpby(alpha, lhs.view(), beta, rhs.view(), output.view());
}

template <typename Scalar>
void axpby(
    Scalar alpha,
    ConstMatrixView<Scalar, Device::CPU> lhs,
    Scalar beta,
    ConstMatrixView<Scalar, Device::CPU> rhs,
    MatrixView<Scalar, Device::CPU> output)
{
    checkSameViewDimensions("axpby", lhs, rhs);
    checkSameViewDimensions("axpby", lhs, output);
    validateOutputView("axpby", output, lhs, rhs);

    const Index count = lhs.size();
    const bool contiguous = lhs.isContiguousColumnMajor() && rhs.isContiguousColumnMajor() &&
                            output.isContiguousColumnMajor();
    if (contiguous)
    {
        if (detail::shouldUseOpenMp(count))
        {
            #pragma omp parallel for simd
            for (Index index = 0; index < count; ++index)
            {
                output.data()[index] = alpha * lhs.data()[index] + beta * rhs.data()[index];
            }
        }
        else
        {
            #pragma omp simd
            for (Index index = 0; index < count; ++index)
            {
                output.data()[index] = alpha * lhs.data()[index] + beta * rhs.data()[index];
            }
        }
        return;
    }

    if (detail::shouldUseOpenMp(count))
    {
        #pragma omp parallel for
        for (Index index = 0; index < count; ++index)
        {
            const Index row = index % lhs.rows();
            const Index col = index / lhs.rows();
            output.data()[row * output.rowStride() + col * output.colStride()] =
                alpha * lhs.data()[row * lhs.rowStride() + col * lhs.colStride()] +
                beta * rhs.data()[row * rhs.rowStride() + col * rhs.colStride()];
        }
    }
    else
    {
        for (Index index = 0; index < count; ++index)
        {
            const Index row = index % lhs.rows();
            const Index col = index / lhs.rows();
            output.data()[row * output.rowStride() + col * output.colStride()] =
                alpha * lhs.data()[row * lhs.rowStride() + col * lhs.colStride()] +
                beta * rhs.data()[row * rhs.rowStride() + col * rhs.colStride()];
        }
    }
}

template <typename Scalar>
DenseStorage<Scalar, Device::CPU> linearCombination(
    Scalar alpha,
    const DenseStorage<Scalar, Device::CPU>& first,
    Scalar beta,
    const DenseStorage<Scalar, Device::CPU>& second,
    Scalar gamma,
    const DenseStorage<Scalar, Device::CPU>& third)
{
    detail::checkSameDimensions("linearCombination", first, second);
    detail::checkSameDimensions("linearCombination", first, third);
    auto output = DenseStorage<Scalar, Device::CPU>::uninitialized(first.rows(), first.cols());
    linearCombination(
        alpha, first.view(), beta, second.view(), gamma, third.view(), output.view());
    return output;
}

template <typename Scalar>
void linearCombination(
    Scalar alpha,
    const DenseStorage<Scalar, Device::CPU>& first,
    Scalar beta,
    const DenseStorage<Scalar, Device::CPU>& second,
    Scalar gamma,
    const DenseStorage<Scalar, Device::CPU>& third,
    DenseStorage<Scalar, Device::CPU>& output)
{
    detail::checkSameDimensions("linearCombination", first, second);
    detail::checkSameDimensions("linearCombination", first, third);
    detail::checkSameDimensions("linearCombination", first, output);
    linearCombination(
        alpha, first.view(), beta, second.view(), gamma, third.view(), output.view());
}

template <typename Scalar>
void linearCombination(
    Scalar alpha,
    ConstMatrixView<Scalar, Device::CPU> first,
    Scalar beta,
    ConstMatrixView<Scalar, Device::CPU> second,
    Scalar gamma,
    ConstMatrixView<Scalar, Device::CPU> third,
    MatrixView<Scalar, Device::CPU> output)
{
    checkSameViewDimensions("linearCombination", first, second);
    checkSameViewDimensions("linearCombination", first, third);
    checkSameViewDimensions("linearCombination", first, output);
    validateOutputView("linearCombination", output, first, second, third);

    const Index count = first.size();
    const bool contiguous = first.isContiguousColumnMajor() && second.isContiguousColumnMajor() &&
                            third.isContiguousColumnMajor() && output.isContiguousColumnMajor();
    if (contiguous)
    {
        if (detail::shouldUseOpenMp(count))
        {
            #pragma omp parallel for simd
            for (Index index = 0; index < count; ++index)
            {
                output.data()[index] = alpha * first.data()[index] + beta * second.data()[index] +
                                       gamma * third.data()[index];
            }
        }
        else
        {
            #pragma omp simd
            for (Index index = 0; index < count; ++index)
            {
                output.data()[index] = alpha * first.data()[index] + beta * second.data()[index] +
                                       gamma * third.data()[index];
            }
        }
        return;
    }

    if (detail::shouldUseOpenMp(count))
    {
        #pragma omp parallel for
        for (Index index = 0; index < count; ++index)
        {
            const Index row = index % first.rows();
            const Index col = index / first.rows();
            output.data()[row * output.rowStride() + col * output.colStride()] =
                alpha * first.data()[row * first.rowStride() + col * first.colStride()] +
                beta * second.data()[row * second.rowStride() + col * second.colStride()] +
                gamma * third.data()[row * third.rowStride() + col * third.colStride()];
        }
    }
    else
    {
        for (Index index = 0; index < count; ++index)
        {
            const Index row = index % first.rows();
            const Index col = index / first.rows();
            output.data()[row * output.rowStride() + col * output.colStride()] =
                alpha * first.data()[row * first.rowStride() + col * first.colStride()] +
                beta * second.data()[row * second.rowStride() + col * second.colStride()] +
                gamma * third.data()[row * third.rowStride() + col * third.colStride()];
        }
    }
}

template <typename Scalar>
DenseStorage<Scalar, Device::CPU> scalarMultiply(
    const DenseStorage<Scalar, Device::CPU>& input,
    Scalar value)
{
    return transformElements(input, [value](Scalar element) { return element * value; });
}

template <typename Scalar>
DenseStorage<Scalar, Device::CPU> scalarAdd(
    const DenseStorage<Scalar, Device::CPU>& input,
    Scalar value)
{
    return transformElements(input, [value](Scalar element) { return element + value; });
}

template <typename Scalar>
DenseStorage<Scalar, Device::CPU> scalarDivide(
    const DenseStorage<Scalar, Device::CPU>& input,
    Scalar value)
{
    if (value == Scalar(0))
    {
        throw std::domain_error("scalarDivide: divisor must be non-zero");
    }
    return transformElements(input, [value](Scalar element) { return element / value; });
}

template <typename Scalar>
DenseStorage<Scalar, Device::CPU> hadamardMultiply(
    const DenseStorage<Scalar, Device::CPU>& lhs,
    const DenseStorage<Scalar, Device::CPU>& rhs)
{
    detail::checkSameDimensions("hadamardMultiply", lhs, rhs);
    return transformElements(lhs, rhs, [](Scalar left, Scalar right) { return left * right; });
}

template <typename Scalar>
DenseStorage<Scalar, Device::CPU> hadamardDivide(
    const DenseStorage<Scalar, Device::CPU>& lhs,
    const DenseStorage<Scalar, Device::CPU>& rhs)
{
    detail::checkSameDimensions("hadamardDivide", lhs, rhs);
    return transformElements(lhs, rhs, [](Scalar left, Scalar right) { return left / right; });
}

template <typename Scalar>
DenseStorage<Scalar, Device::CPU> absElements(
    const DenseStorage<Scalar, Device::CPU>& input)
{
    return applyUnary<ElementwiseUnaryOp::Abs>(input);
}

template <typename Scalar>
DenseStorage<Scalar, Device::CPU> sqrtElements(
    const DenseStorage<Scalar, Device::CPU>& input)
{
    return applyUnary<ElementwiseUnaryOp::Sqrt>(input);
}

template <typename Scalar>
DenseStorage<Scalar, Device::CPU> clampElements(
    const DenseStorage<Scalar, Device::CPU>& input,
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
template DenseStorage<float, Device::CPU> axpby(
    float,
    const DenseStorage<float, Device::CPU>&,
    float,
    const DenseStorage<float, Device::CPU>&);
template void axpby(
    float,
    const DenseStorage<float, Device::CPU>&,
    float,
    const DenseStorage<float, Device::CPU>&,
    DenseStorage<float, Device::CPU>&);
template void axpby(
    float,
    ConstMatrixView<float, Device::CPU>,
    float,
    ConstMatrixView<float, Device::CPU>,
    MatrixView<float, Device::CPU>);
template DenseStorage<float, Device::CPU> linearCombination(
    float,
    const DenseStorage<float, Device::CPU>&,
    float,
    const DenseStorage<float, Device::CPU>&,
    float,
    const DenseStorage<float, Device::CPU>&);
template void linearCombination(
    float,
    const DenseStorage<float, Device::CPU>&,
    float,
    const DenseStorage<float, Device::CPU>&,
    float,
    const DenseStorage<float, Device::CPU>&,
    DenseStorage<float, Device::CPU>&);
template void linearCombination(
    float,
    ConstMatrixView<float, Device::CPU>,
    float,
    ConstMatrixView<float, Device::CPU>,
    float,
    ConstMatrixView<float, Device::CPU>,
    MatrixView<float, Device::CPU>);
template DenseStorage<float, Device::CPU> scalarMultiply(const DenseStorage<float, Device::CPU>&, float);
template DenseStorage<float, Device::CPU> scalarAdd(const DenseStorage<float, Device::CPU>&, float);
template DenseStorage<float, Device::CPU> scalarDivide(const DenseStorage<float, Device::CPU>&, float);
template DenseStorage<float, Device::CPU> hadamardMultiply(
    const DenseStorage<float, Device::CPU>&,
    const DenseStorage<float, Device::CPU>&);
template DenseStorage<float, Device::CPU> hadamardDivide(
    const DenseStorage<float, Device::CPU>&,
    const DenseStorage<float, Device::CPU>&);
template DenseStorage<float, Device::CPU> absElements(const DenseStorage<float, Device::CPU>&);
template DenseStorage<float, Device::CPU> sqrtElements(const DenseStorage<float, Device::CPU>&);
template DenseStorage<float, Device::CPU> clampElements(
    const DenseStorage<float, Device::CPU>&,
    float,
    float);
#endif

#ifdef PLAMATRIX_USE_DOUBLE
template DenseStorage<double, Device::CPU> axpby(
    double,
    const DenseStorage<double, Device::CPU>&,
    double,
    const DenseStorage<double, Device::CPU>&);
template void axpby(
    double,
    const DenseStorage<double, Device::CPU>&,
    double,
    const DenseStorage<double, Device::CPU>&,
    DenseStorage<double, Device::CPU>&);
template void axpby(
    double,
    ConstMatrixView<double, Device::CPU>,
    double,
    ConstMatrixView<double, Device::CPU>,
    MatrixView<double, Device::CPU>);
template DenseStorage<double, Device::CPU> linearCombination(
    double,
    const DenseStorage<double, Device::CPU>&,
    double,
    const DenseStorage<double, Device::CPU>&,
    double,
    const DenseStorage<double, Device::CPU>&);
template void linearCombination(
    double,
    const DenseStorage<double, Device::CPU>&,
    double,
    const DenseStorage<double, Device::CPU>&,
    double,
    const DenseStorage<double, Device::CPU>&,
    DenseStorage<double, Device::CPU>&);
template void linearCombination(
    double,
    ConstMatrixView<double, Device::CPU>,
    double,
    ConstMatrixView<double, Device::CPU>,
    double,
    ConstMatrixView<double, Device::CPU>,
    MatrixView<double, Device::CPU>);
template DenseStorage<double, Device::CPU> scalarMultiply(const DenseStorage<double, Device::CPU>&, double);
template DenseStorage<double, Device::CPU> scalarAdd(const DenseStorage<double, Device::CPU>&, double);
template DenseStorage<double, Device::CPU> scalarDivide(const DenseStorage<double, Device::CPU>&, double);
template DenseStorage<double, Device::CPU> hadamardMultiply(
    const DenseStorage<double, Device::CPU>&,
    const DenseStorage<double, Device::CPU>&);
template DenseStorage<double, Device::CPU> hadamardDivide(
    const DenseStorage<double, Device::CPU>&,
    const DenseStorage<double, Device::CPU>&);
template DenseStorage<double, Device::CPU> absElements(const DenseStorage<double, Device::CPU>&);
template DenseStorage<double, Device::CPU> sqrtElements(const DenseStorage<double, Device::CPU>&);
template DenseStorage<double, Device::CPU> clampElements(
    const DenseStorage<double, Device::CPU>&,
    double,
    double);
#endif

} // namespace plamatrix::internal
