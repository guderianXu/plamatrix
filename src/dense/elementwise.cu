#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "plamatrix/internal/dense/dense_ops.h"
#include "plamatrix/internal/dense/elementwise.h"

namespace plamatrix::internal
{
namespace
{

enum class ElementwiseOperation
{
    ScalarMultiply,
    ScalarAdd,
    ScalarDivide,
    HadamardMultiply,
    HadamardDivide,
    Abs,
    Sqrt,
    Clamp
};

constexpr int kBlockSize = 256;
constexpr Index kMaxGridSize = 65535;

unsigned int checkedCudaGrid1D(Index item_count, const char* operation)
{
    if (item_count <= 0)
    {
        std::ostringstream oss;
        oss << operation << ": CUDA grid requires a positive item count";
        throw std::invalid_argument(oss.str());
    }

    const Index block_count = item_count / kBlockSize + (item_count % kBlockSize != 0 ? 1 : 0);
    const Index grid_size = std::min(block_count, kMaxGridSize);
    if (grid_size > static_cast<Index>(std::numeric_limits<unsigned int>::max()))
    {
        std::ostringstream oss;
        oss << operation << ": CUDA grid size exceeds unsigned int range";
        throw std::runtime_error(oss.str());
    }
    return static_cast<unsigned int>(grid_size);
}

template <ElementwiseOperation Operation, typename Scalar>
__device__ Scalar applyElementwise(Scalar lhs, Scalar rhs, Scalar first, Scalar second)
{
    if constexpr (Operation == ElementwiseOperation::ScalarMultiply)
    {
        return lhs * first;
    }
    else if constexpr (Operation == ElementwiseOperation::ScalarAdd)
    {
        return lhs + first;
    }
    else if constexpr (Operation == ElementwiseOperation::ScalarDivide)
    {
        return lhs / first;
    }
    else if constexpr (Operation == ElementwiseOperation::HadamardMultiply)
    {
        return lhs * rhs;
    }
    else if constexpr (Operation == ElementwiseOperation::HadamardDivide)
    {
        return lhs / rhs;
    }
    else if constexpr (Operation == ElementwiseOperation::Abs)
    {
        return fabs(lhs);
    }
    else if constexpr (Operation == ElementwiseOperation::Sqrt)
    {
        return sqrt(lhs);
    }
    else
    {
        return lhs < first ? first : (lhs > second ? second : lhs);
    }
}

template <ElementwiseOperation Operation, typename Scalar>
__global__ void elementwiseKernel(const Scalar* lhs,
                                  const Scalar* rhs,
                                  Scalar first,
                                  Scalar second,
                                  Scalar* output,
                                  Index item_count)
{
    Index index = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
    const Index stride = static_cast<Index>(blockDim.x) * gridDim.x;
    for (; index < item_count; index += stride)
    {
        const Scalar rhs_value = rhs == nullptr ? Scalar(0) : rhs[index];
        output[index] = applyElementwise<Operation>(lhs[index], rhs_value, first, second);
    }
}

template <ElementwiseOperation Operation, typename Scalar>
void launchElementwise(const char* operation,
                       const DenseStorage<Scalar, Device::GPU>& input,
                       const DenseStorage<Scalar, Device::GPU>* rhs,
                       Scalar first,
                       Scalar second,
                       DenseStorage<Scalar, Device::GPU>& output,
                       cudaStream_t stream)
{
    detail::checkOutputDimensions(operation, output, input.rows(), input.cols());
    const Index item_count = input.size();
    if (item_count == 0)
    {
        return;
    }

    const unsigned int grid_size = checkedCudaGrid1D(item_count, operation);
    const Scalar* rhs_data = rhs == nullptr ? nullptr : rhs->data();
    elementwiseKernel<Operation><<<grid_size, kBlockSize, 0, stream>>>(
        input.data(), rhs_data, first, second, output.data(), item_count);
    PLAMATRIX_CHECK_CUDA(cudaGetLastError());
}

template <typename Scalar>
void validateScalarDivisor(Scalar value)
{
    if (value == Scalar(0))
    {
        throw std::domain_error("scalarDivide: divisor must be non-zero");
    }
}

template <typename Scalar>
void validateClampBounds(Scalar min_value, Scalar max_value)
{
    if (min_value > max_value)
    {
        throw std::invalid_argument("clampElements: min_value must not exceed max_value");
    }
}

} // anonymous namespace

template <typename Scalar>
void scalarMultiplyAsync(const DenseStorage<Scalar, Device::GPU>& input,
                         Scalar value,
                         DenseStorage<Scalar, Device::GPU>& output,
                         cudaStream_t stream)
{
    launchElementwise<ElementwiseOperation::ScalarMultiply, Scalar>(
        "scalarMultiply", input, nullptr, value, Scalar(0), output, stream);
}

template <typename Scalar>
DenseStorage<Scalar, Device::GPU> scalarMultiplyAsync(
    const DenseStorage<Scalar, Device::GPU>& input, Scalar value, cudaStream_t stream)
{
    auto output = DenseStorage<Scalar, Device::GPU>::uninitializedAsync(
        input.rows(), input.cols(), stream);
    scalarMultiplyAsync(input, value, output, stream);
    return output;
}

template <typename Scalar>
void scalarMultiply(const DenseStorage<Scalar, Device::GPU>& input,
                    Scalar value,
                    DenseStorage<Scalar, Device::GPU>& output,
                    cudaStream_t stream)
{
    scalarMultiplyAsync(input, value, output, stream);
    PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
}

template <typename Scalar>
DenseStorage<Scalar, Device::GPU> scalarMultiply(
    const DenseStorage<Scalar, Device::GPU>& input, Scalar value, cudaStream_t stream)
{
    DenseStorage<Scalar, Device::GPU> output(input.rows(), input.cols());
    scalarMultiplyAsync(input, value, output, stream);
    PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
    return output;
}

template <typename Scalar>
void scalarAddAsync(const DenseStorage<Scalar, Device::GPU>& input,
                    Scalar value,
                    DenseStorage<Scalar, Device::GPU>& output,
                    cudaStream_t stream)
{
    launchElementwise<ElementwiseOperation::ScalarAdd, Scalar>(
        "scalarAdd", input, nullptr, value, Scalar(0), output, stream);
}

template <typename Scalar>
DenseStorage<Scalar, Device::GPU> scalarAddAsync(
    const DenseStorage<Scalar, Device::GPU>& input, Scalar value, cudaStream_t stream)
{
    auto output = DenseStorage<Scalar, Device::GPU>::uninitializedAsync(
        input.rows(), input.cols(), stream);
    scalarAddAsync(input, value, output, stream);
    return output;
}

template <typename Scalar>
void scalarAdd(const DenseStorage<Scalar, Device::GPU>& input,
               Scalar value,
               DenseStorage<Scalar, Device::GPU>& output,
               cudaStream_t stream)
{
    scalarAddAsync(input, value, output, stream);
    PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
}

template <typename Scalar>
DenseStorage<Scalar, Device::GPU> scalarAdd(
    const DenseStorage<Scalar, Device::GPU>& input, Scalar value, cudaStream_t stream)
{
    DenseStorage<Scalar, Device::GPU> output(input.rows(), input.cols());
    scalarAddAsync(input, value, output, stream);
    PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
    return output;
}

template <typename Scalar>
void scalarDivideAsync(const DenseStorage<Scalar, Device::GPU>& input,
                       Scalar value,
                       DenseStorage<Scalar, Device::GPU>& output,
                       cudaStream_t stream)
{
    validateScalarDivisor(value);
    launchElementwise<ElementwiseOperation::ScalarDivide, Scalar>(
        "scalarDivide", input, nullptr, value, Scalar(0), output, stream);
}

template <typename Scalar>
DenseStorage<Scalar, Device::GPU> scalarDivideAsync(
    const DenseStorage<Scalar, Device::GPU>& input, Scalar value, cudaStream_t stream)
{
    validateScalarDivisor(value);
    auto output = DenseStorage<Scalar, Device::GPU>::uninitializedAsync(
        input.rows(), input.cols(), stream);
    scalarDivideAsync(input, value, output, stream);
    return output;
}

template <typename Scalar>
void scalarDivide(const DenseStorage<Scalar, Device::GPU>& input,
                  Scalar value,
                  DenseStorage<Scalar, Device::GPU>& output,
                  cudaStream_t stream)
{
    scalarDivideAsync(input, value, output, stream);
    PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
}

template <typename Scalar>
DenseStorage<Scalar, Device::GPU> scalarDivide(
    const DenseStorage<Scalar, Device::GPU>& input, Scalar value, cudaStream_t stream)
{
    validateScalarDivisor(value);
    DenseStorage<Scalar, Device::GPU> output(input.rows(), input.cols());
    scalarDivideAsync(input, value, output, stream);
    PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
    return output;
}

template <typename Scalar>
void hadamardMultiplyAsync(const DenseStorage<Scalar, Device::GPU>& lhs,
                           const DenseStorage<Scalar, Device::GPU>& rhs,
                           DenseStorage<Scalar, Device::GPU>& output,
                           cudaStream_t stream)
{
    detail::checkSameDimensions("hadamardMultiply", lhs, rhs);
    launchElementwise<ElementwiseOperation::HadamardMultiply>(
        "hadamardMultiply", lhs, &rhs, Scalar(0), Scalar(0), output, stream);
}

template <typename Scalar>
DenseStorage<Scalar, Device::GPU> hadamardMultiplyAsync(
    const DenseStorage<Scalar, Device::GPU>& lhs,
    const DenseStorage<Scalar, Device::GPU>& rhs,
    cudaStream_t stream)
{
    detail::checkSameDimensions("hadamardMultiply", lhs, rhs);
    auto output = DenseStorage<Scalar, Device::GPU>::uninitializedAsync(
        lhs.rows(), lhs.cols(), stream);
    hadamardMultiplyAsync(lhs, rhs, output, stream);
    return output;
}

template <typename Scalar>
void hadamardMultiply(const DenseStorage<Scalar, Device::GPU>& lhs,
                      const DenseStorage<Scalar, Device::GPU>& rhs,
                      DenseStorage<Scalar, Device::GPU>& output,
                      cudaStream_t stream)
{
    hadamardMultiplyAsync(lhs, rhs, output, stream);
    PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
}

template <typename Scalar>
DenseStorage<Scalar, Device::GPU> hadamardMultiply(
    const DenseStorage<Scalar, Device::GPU>& lhs,
    const DenseStorage<Scalar, Device::GPU>& rhs,
    cudaStream_t stream)
{
    detail::checkSameDimensions("hadamardMultiply", lhs, rhs);
    DenseStorage<Scalar, Device::GPU> output(lhs.rows(), lhs.cols());
    hadamardMultiplyAsync(lhs, rhs, output, stream);
    PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
    return output;
}

template <typename Scalar>
void hadamardDivideAsync(const DenseStorage<Scalar, Device::GPU>& lhs,
                         const DenseStorage<Scalar, Device::GPU>& rhs,
                         DenseStorage<Scalar, Device::GPU>& output,
                         cudaStream_t stream)
{
    detail::checkSameDimensions("hadamardDivide", lhs, rhs);
    launchElementwise<ElementwiseOperation::HadamardDivide>(
        "hadamardDivide", lhs, &rhs, Scalar(0), Scalar(0), output, stream);
}

template <typename Scalar>
DenseStorage<Scalar, Device::GPU> hadamardDivideAsync(
    const DenseStorage<Scalar, Device::GPU>& lhs,
    const DenseStorage<Scalar, Device::GPU>& rhs,
    cudaStream_t stream)
{
    detail::checkSameDimensions("hadamardDivide", lhs, rhs);
    auto output = DenseStorage<Scalar, Device::GPU>::uninitializedAsync(
        lhs.rows(), lhs.cols(), stream);
    hadamardDivideAsync(lhs, rhs, output, stream);
    return output;
}

template <typename Scalar>
void hadamardDivide(const DenseStorage<Scalar, Device::GPU>& lhs,
                    const DenseStorage<Scalar, Device::GPU>& rhs,
                    DenseStorage<Scalar, Device::GPU>& output,
                    cudaStream_t stream)
{
    hadamardDivideAsync(lhs, rhs, output, stream);
    PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
}

template <typename Scalar>
DenseStorage<Scalar, Device::GPU> hadamardDivide(
    const DenseStorage<Scalar, Device::GPU>& lhs,
    const DenseStorage<Scalar, Device::GPU>& rhs,
    cudaStream_t stream)
{
    detail::checkSameDimensions("hadamardDivide", lhs, rhs);
    DenseStorage<Scalar, Device::GPU> output(lhs.rows(), lhs.cols());
    hadamardDivideAsync(lhs, rhs, output, stream);
    PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
    return output;
}

template <typename Scalar>
void absElementsAsync(const DenseStorage<Scalar, Device::GPU>& input,
                      DenseStorage<Scalar, Device::GPU>& output,
                      cudaStream_t stream)
{
    launchElementwise<ElementwiseOperation::Abs, Scalar>(
        "absElements", input, nullptr, Scalar(0), Scalar(0), output, stream);
}

template <typename Scalar>
DenseStorage<Scalar, Device::GPU> absElementsAsync(
    const DenseStorage<Scalar, Device::GPU>& input, cudaStream_t stream)
{
    auto output = DenseStorage<Scalar, Device::GPU>::uninitializedAsync(
        input.rows(), input.cols(), stream);
    absElementsAsync(input, output, stream);
    return output;
}

template <typename Scalar>
void absElements(const DenseStorage<Scalar, Device::GPU>& input,
                 DenseStorage<Scalar, Device::GPU>& output,
                 cudaStream_t stream)
{
    absElementsAsync(input, output, stream);
    PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
}

template <typename Scalar>
DenseStorage<Scalar, Device::GPU> absElements(
    const DenseStorage<Scalar, Device::GPU>& input, cudaStream_t stream)
{
    DenseStorage<Scalar, Device::GPU> output(input.rows(), input.cols());
    absElementsAsync(input, output, stream);
    PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
    return output;
}

template <typename Scalar>
void sqrtElementsAsync(const DenseStorage<Scalar, Device::GPU>& input,
                       DenseStorage<Scalar, Device::GPU>& output,
                       cudaStream_t stream)
{
    launchElementwise<ElementwiseOperation::Sqrt, Scalar>(
        "sqrtElements", input, nullptr, Scalar(0), Scalar(0), output, stream);
}

template <typename Scalar>
DenseStorage<Scalar, Device::GPU> sqrtElementsAsync(
    const DenseStorage<Scalar, Device::GPU>& input, cudaStream_t stream)
{
    auto output = DenseStorage<Scalar, Device::GPU>::uninitializedAsync(
        input.rows(), input.cols(), stream);
    sqrtElementsAsync(input, output, stream);
    return output;
}

template <typename Scalar>
void sqrtElements(const DenseStorage<Scalar, Device::GPU>& input,
                  DenseStorage<Scalar, Device::GPU>& output,
                  cudaStream_t stream)
{
    sqrtElementsAsync(input, output, stream);
    PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
}

template <typename Scalar>
DenseStorage<Scalar, Device::GPU> sqrtElements(
    const DenseStorage<Scalar, Device::GPU>& input, cudaStream_t stream)
{
    DenseStorage<Scalar, Device::GPU> output(input.rows(), input.cols());
    sqrtElementsAsync(input, output, stream);
    PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
    return output;
}

template <typename Scalar>
void clampElementsAsync(const DenseStorage<Scalar, Device::GPU>& input,
                        Scalar min_value,
                        Scalar max_value,
                        DenseStorage<Scalar, Device::GPU>& output,
                        cudaStream_t stream)
{
    validateClampBounds(min_value, max_value);
    launchElementwise<ElementwiseOperation::Clamp, Scalar>(
        "clampElements", input, nullptr, min_value, max_value, output, stream);
}

template <typename Scalar>
DenseStorage<Scalar, Device::GPU> clampElementsAsync(
    const DenseStorage<Scalar, Device::GPU>& input,
    Scalar min_value,
    Scalar max_value,
    cudaStream_t stream)
{
    validateClampBounds(min_value, max_value);
    auto output = DenseStorage<Scalar, Device::GPU>::uninitializedAsync(
        input.rows(), input.cols(), stream);
    clampElementsAsync(input, min_value, max_value, output, stream);
    return output;
}

template <typename Scalar>
void clampElements(const DenseStorage<Scalar, Device::GPU>& input,
                   Scalar min_value,
                   Scalar max_value,
                   DenseStorage<Scalar, Device::GPU>& output,
                   cudaStream_t stream)
{
    clampElementsAsync(input, min_value, max_value, output, stream);
    PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
}

template <typename Scalar>
DenseStorage<Scalar, Device::GPU> clampElements(
    const DenseStorage<Scalar, Device::GPU>& input,
    Scalar min_value,
    Scalar max_value,
    cudaStream_t stream)
{
    validateClampBounds(min_value, max_value);
    DenseStorage<Scalar, Device::GPU> output(input.rows(), input.cols());
    clampElementsAsync(input, min_value, max_value, output, stream);
    PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
    return output;
}

#define PLAMATRIX_INSTANTIATE_ELEMENTWISE(Scalar)                                                   \
    template DenseStorage<Scalar, Device::GPU> scalarMultiplyAsync(                                 \
        const DenseStorage<Scalar, Device::GPU>&, Scalar, cudaStream_t);                             \
    template void scalarMultiplyAsync(const DenseStorage<Scalar, Device::GPU>&, Scalar,              \
                                      DenseStorage<Scalar, Device::GPU>&, cudaStream_t);              \
    template DenseStorage<Scalar, Device::GPU> scalarMultiply(                                      \
        const DenseStorage<Scalar, Device::GPU>&, Scalar, cudaStream_t);                             \
    template void scalarMultiply(const DenseStorage<Scalar, Device::GPU>&, Scalar,                   \
                                 DenseStorage<Scalar, Device::GPU>&, cudaStream_t);                   \
    template DenseStorage<Scalar, Device::GPU> scalarAddAsync(                                      \
        const DenseStorage<Scalar, Device::GPU>&, Scalar, cudaStream_t);                             \
    template void scalarAddAsync(const DenseStorage<Scalar, Device::GPU>&, Scalar,                   \
                                 DenseStorage<Scalar, Device::GPU>&, cudaStream_t);                   \
    template DenseStorage<Scalar, Device::GPU> scalarAdd(                                           \
        const DenseStorage<Scalar, Device::GPU>&, Scalar, cudaStream_t);                             \
    template void scalarAdd(const DenseStorage<Scalar, Device::GPU>&, Scalar,                        \
                            DenseStorage<Scalar, Device::GPU>&, cudaStream_t);                        \
    template DenseStorage<Scalar, Device::GPU> scalarDivideAsync(                                   \
        const DenseStorage<Scalar, Device::GPU>&, Scalar, cudaStream_t);                             \
    template void scalarDivideAsync(const DenseStorage<Scalar, Device::GPU>&, Scalar,                \
                                    DenseStorage<Scalar, Device::GPU>&, cudaStream_t);                \
    template DenseStorage<Scalar, Device::GPU> scalarDivide(                                        \
        const DenseStorage<Scalar, Device::GPU>&, Scalar, cudaStream_t);                             \
    template void scalarDivide(const DenseStorage<Scalar, Device::GPU>&, Scalar,                     \
                               DenseStorage<Scalar, Device::GPU>&, cudaStream_t);                     \
    template DenseStorage<Scalar, Device::GPU> hadamardMultiplyAsync(                               \
        const DenseStorage<Scalar, Device::GPU>&, const DenseStorage<Scalar, Device::GPU>&,           \
        cudaStream_t);                                                                              \
    template void hadamardMultiplyAsync(                                                            \
        const DenseStorage<Scalar, Device::GPU>&, const DenseStorage<Scalar, Device::GPU>&,           \
        DenseStorage<Scalar, Device::GPU>&, cudaStream_t);                                           \
    template DenseStorage<Scalar, Device::GPU> hadamardMultiply(                                    \
        const DenseStorage<Scalar, Device::GPU>&, const DenseStorage<Scalar, Device::GPU>&,           \
        cudaStream_t);                                                                              \
    template void hadamardMultiply(                                                                 \
        const DenseStorage<Scalar, Device::GPU>&, const DenseStorage<Scalar, Device::GPU>&,           \
        DenseStorage<Scalar, Device::GPU>&, cudaStream_t);                                           \
    template DenseStorage<Scalar, Device::GPU> hadamardDivideAsync(                                 \
        const DenseStorage<Scalar, Device::GPU>&, const DenseStorage<Scalar, Device::GPU>&,           \
        cudaStream_t);                                                                              \
    template void hadamardDivideAsync(                                                              \
        const DenseStorage<Scalar, Device::GPU>&, const DenseStorage<Scalar, Device::GPU>&,           \
        DenseStorage<Scalar, Device::GPU>&, cudaStream_t);                                           \
    template DenseStorage<Scalar, Device::GPU> hadamardDivide(                                      \
        const DenseStorage<Scalar, Device::GPU>&, const DenseStorage<Scalar, Device::GPU>&,           \
        cudaStream_t);                                                                              \
    template void hadamardDivide(                                                                   \
        const DenseStorage<Scalar, Device::GPU>&, const DenseStorage<Scalar, Device::GPU>&,           \
        DenseStorage<Scalar, Device::GPU>&, cudaStream_t);                                           \
    template DenseStorage<Scalar, Device::GPU> absElementsAsync(                                    \
        const DenseStorage<Scalar, Device::GPU>&, cudaStream_t);                                    \
    template void absElementsAsync(const DenseStorage<Scalar, Device::GPU>&,                         \
                                   DenseStorage<Scalar, Device::GPU>&, cudaStream_t);                 \
    template DenseStorage<Scalar, Device::GPU> absElements(                                         \
        const DenseStorage<Scalar, Device::GPU>&, cudaStream_t);                                    \
    template void absElements(const DenseStorage<Scalar, Device::GPU>&,                              \
                              DenseStorage<Scalar, Device::GPU>&, cudaStream_t);                      \
    template DenseStorage<Scalar, Device::GPU> sqrtElementsAsync(                                   \
        const DenseStorage<Scalar, Device::GPU>&, cudaStream_t);                                    \
    template void sqrtElementsAsync(const DenseStorage<Scalar, Device::GPU>&,                        \
                                    DenseStorage<Scalar, Device::GPU>&, cudaStream_t);                \
    template DenseStorage<Scalar, Device::GPU> sqrtElements(                                        \
        const DenseStorage<Scalar, Device::GPU>&, cudaStream_t);                                    \
    template void sqrtElements(const DenseStorage<Scalar, Device::GPU>&,                             \
                               DenseStorage<Scalar, Device::GPU>&, cudaStream_t);                     \
    template DenseStorage<Scalar, Device::GPU> clampElementsAsync(                                  \
        const DenseStorage<Scalar, Device::GPU>&, Scalar, Scalar, cudaStream_t);                     \
    template void clampElementsAsync(const DenseStorage<Scalar, Device::GPU>&, Scalar, Scalar,       \
                                     DenseStorage<Scalar, Device::GPU>&, cudaStream_t);               \
    template DenseStorage<Scalar, Device::GPU> clampElements(                                       \
        const DenseStorage<Scalar, Device::GPU>&, Scalar, Scalar, cudaStream_t);                     \
    template void clampElements(const DenseStorage<Scalar, Device::GPU>&, Scalar, Scalar,            \
                                DenseStorage<Scalar, Device::GPU>&, cudaStream_t)

#ifdef PLAMATRIX_USE_FLOAT
PLAMATRIX_INSTANTIATE_ELEMENTWISE(float);
#endif

#ifdef PLAMATRIX_USE_DOUBLE
PLAMATRIX_INSTANTIATE_ELEMENTWISE(double);
#endif

#undef PLAMATRIX_INSTANTIATE_ELEMENTWISE

} // namespace plamatrix::internal
