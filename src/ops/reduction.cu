#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

#include <cub/device/device_reduce.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include "reduction_detail.h"
#include "reduction_kernels.cuh"

namespace plamatrix::internal
{
namespace
{

std::size_t alignedViewOffset(std::size_t offset, std::size_t alignment)
{
    if (offset > std::numeric_limits<std::size_t>::max() - (alignment - 1))
    {
        throw std::overflow_error("reduction: workspace alignment overflows size_t");
    }
    return (offset + alignment - 1) & ~(alignment - 1);
}

template <typename Stage>
Stage* reserveViewStage(ReductionWorkspace& workspace, std::size_t temporary_bytes, cudaStream_t stream)
{
    const std::size_t offset = alignedViewOffset(temporary_bytes, alignof(Stage));
    if (offset > std::numeric_limits<std::size_t>::max() - sizeof(Stage))
    {
        throw std::overflow_error("reduction: workspace size overflows size_t");
    }
    workspace.reserveBytesAsync(offset + sizeof(Stage), stream);
    auto* bytes = static_cast<unsigned char*>(workspace.data());
    return reinterpret_cast<Stage*>(bytes + offset);
}

template <typename Scalar>
bool viewStorageOverlaps(ConstMatrixView<Scalar, Device::GPU> input, MatrixView<Scalar, Device::GPU> output)
{
    if (input.size() == 0 || output.size() == 0)
    {
        return false;
    }
    const std::size_t input_bytes = detail::checkedAllocationBytes<Scalar>(input.size());
    const std::size_t output_bytes = detail::checkedAllocationBytes<Scalar>(output.size());
    const auto input_begin = reinterpret_cast<std::uintptr_t>(input.data());
    const auto output_begin = reinterpret_cast<std::uintptr_t>(output.data());
    if (input_begin > std::numeric_limits<std::uintptr_t>::max() - input_bytes ||
        output_begin > std::numeric_limits<std::uintptr_t>::max() - output_bytes)
    {
        throw std::overflow_error("reduction: view storage range overflows uintptr_t");
    }
    return input_begin < output_begin + output_bytes && output_begin < input_begin + input_bytes;
}

template <typename Scalar>
void launchViewValueReduction(reduction_detail::ValueOperation operation,
                              const char* operation_name,
                              ConstMatrixView<Scalar, Device::GPU> input,
                              ReductionAxis axis,
                              MatrixView<Scalar, Device::GPU> output,
                              ReductionWorkspace& workspace,
                              cudaStream_t stream)
{
    const auto plan = reduction_detail::validatedValuePlan(
        operation_name, operation, input.rows(), input.cols(), input.size(), axis);
    if (output.rows() != plan.output_rows || output.cols() != plan.output_cols)
    {
        std::ostringstream message;
        message << operation_name << ": output dimensions must be " << plan.output_rows << 'x'
                << plan.output_cols;
        throw std::invalid_argument(message.str());
    }
    if (!input.isContiguousColumnMajor() || !output.isContiguousColumnMajor())
    {
        throw std::invalid_argument(std::string(operation_name) +
                                    ": GPU views must be contiguous column-major");
    }
    if (viewStorageOverlaps(input, output))
    {
        throw std::invalid_argument(std::string(operation_name) + ": input and output views must not overlap");
    }
    if (plan.lane_count == 0)
    {
        workspace.reserveBytesAsync(0, stream);
        return;
    }
    if (plan.reduction_length == 0)
    {
        workspace.reserveBytesAsync(0, stream);
        PLAMATRIX_CHECK_CUDA(
            cudaMemsetAsync(output.data(), 0, detail::checkedAllocationBytes<Scalar>(output.size()), stream));
        return;
    }

    if (axis == ReductionAxis::All)
    {
        if (operation == reduction_detail::ValueOperation::Sum)
        {
            const auto values =
                thrust::make_transform_iterator(input.data(), reduction_detail::CastToDouble<Scalar>{});
            std::size_t temporary_bytes = 0;
            PLAMATRIX_CHECK_CUDA(cub::DeviceReduce::Sum(
                nullptr, temporary_bytes, values, static_cast<double*>(nullptr), input.size(), stream));
            double* accumulated = reserveViewStage<double>(workspace, temporary_bytes, stream);
            PLAMATRIX_CHECK_CUDA(cub::DeviceReduce::Sum(
                workspace.data(), temporary_bytes, values, accumulated, input.size(), stream));
            reduction_detail::storeAccumulatedKernel<<<1, 1, 0, stream>>>(accumulated, output.data());
        }
        else
        {
            using Pair = reduction_detail::ExtremePair<Scalar>;
            const auto pairs = thrust::make_transform_iterator(
                thrust::counting_iterator<Index>(0), reduction_detail::LoadExtremePair<Scalar>{input.data()});
            const Pair initial{Scalar(0), Index(-1)};
            std::size_t temporary_bytes = 0;
            if (operation == reduction_detail::ValueOperation::Minimum)
            {
                const reduction_detail::ExtremeReducer<true, Scalar> reducer;
                PLAMATRIX_CHECK_CUDA(cub::DeviceReduce::Reduce(nullptr,
                                                               temporary_bytes,
                                                               pairs,
                                                               static_cast<Pair*>(nullptr),
                                                               input.size(),
                                                               reducer,
                                                               initial,
                                                               stream));
                Pair* selected = reserveViewStage<Pair>(workspace, temporary_bytes, stream);
                PLAMATRIX_CHECK_CUDA(cub::DeviceReduce::Reduce(workspace.data(),
                                                               temporary_bytes,
                                                               pairs,
                                                               selected,
                                                               input.size(),
                                                               reducer,
                                                               initial,
                                                               stream));
                reduction_detail::storeExtremeKernel<<<1, 1, 0, stream>>>(selected, output.data(), nullptr);
            }
            else
            {
                const reduction_detail::ExtremeReducer<false, Scalar> reducer;
                PLAMATRIX_CHECK_CUDA(cub::DeviceReduce::Reduce(nullptr,
                                                               temporary_bytes,
                                                               pairs,
                                                               static_cast<Pair*>(nullptr),
                                                               input.size(),
                                                               reducer,
                                                               initial,
                                                               stream));
                Pair* selected = reserveViewStage<Pair>(workspace, temporary_bytes, stream);
                PLAMATRIX_CHECK_CUDA(cub::DeviceReduce::Reduce(workspace.data(),
                                                               temporary_bytes,
                                                               pairs,
                                                               selected,
                                                               input.size(),
                                                               reducer,
                                                               initial,
                                                               stream));
                reduction_detail::storeExtremeKernel<<<1, 1, 0, stream>>>(selected, output.data(), nullptr);
            }
        }
        PLAMATRIX_CHECK_CUDA(cudaGetLastError());
        return;
    }

    workspace.reserveBytesAsync(0, stream);
    if (plan.lane_count > static_cast<Index>(std::numeric_limits<int>::max()))
    {
        throw std::overflow_error(std::string(operation_name) + ": lane count is outside the CUDA grid range");
    }
    const unsigned int grid = static_cast<unsigned int>(plan.lane_count);
    if (operation == reduction_detail::ValueOperation::Sum)
    {
        reduction_detail::laneSumKernel<<<grid, reduction_detail::kBlockSize, 0, stream>>>(
            input.data(), output.data(), plan);
    }
    else if (operation == reduction_detail::ValueOperation::Minimum)
    {
        reduction_detail::laneExtremeKernel<true>
            <<<grid, reduction_detail::kBlockSize, 0, stream>>>(input.data(), output.data(), nullptr, plan);
    }
    else
    {
        reduction_detail::laneExtremeKernel<false>
            <<<grid, reduction_detail::kBlockSize, 0, stream>>>(input.data(), output.data(), nullptr, plan);
    }
    PLAMATRIX_CHECK_CUDA(cudaGetLastError());
}

template <typename Scalar>
DenseStorage<Scalar, Device::GPU> valueReductionAsync(
    reduction_detail::ValueOperation operation,
    const char* operation_name,
    const DenseStorage<Scalar, Device::GPU>& input,
    ReductionAxis axis,
    ReductionWorkspace& workspace,
    cudaStream_t stream)
{
    const auto plan = reduction_detail::validatedValuePlan(
        operation_name, operation, input.rows(), input.cols(), input.size(), axis);
    auto output = DenseStorage<Scalar, Device::GPU>::uninitializedAsync(
        plan.output_rows, plan.output_cols, stream);
    reduction_detail::launchValueReduction(
        operation, operation_name, input, axis, output, workspace, stream);
    return output;
}

template <typename Scalar>
DenseStorage<Scalar, Device::GPU> valueReduction(
    reduction_detail::ValueOperation operation,
    const char* operation_name,
    const DenseStorage<Scalar, Device::GPU>& input,
    ReductionAxis axis,
    ReductionWorkspace& workspace,
    cudaStream_t stream)
{
    const auto plan = reduction_detail::validatedValuePlan(
        operation_name, operation, input.rows(), input.cols(), input.size(), axis);
    auto output = DenseStorage<Scalar, Device::GPU>::uninitialized(
        plan.output_rows, plan.output_cols);
    reduction_detail::launchValueReduction(
        operation, operation_name, input, axis, output, workspace, stream);
    PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
    return output;
}

template <typename Scalar>
void valueReduction(
    reduction_detail::ValueOperation operation,
    const char* operation_name,
    const DenseStorage<Scalar, Device::GPU>& input,
    ReductionAxis axis,
    DenseStorage<Scalar, Device::GPU>& output,
    ReductionWorkspace& workspace,
    cudaStream_t stream,
    bool synchronize)
{
    reduction_detail::launchValueReduction(
        operation, operation_name, input, axis, output, workspace, stream);
    if (synchronize)
    {
        PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
    }
}

template <bool FindMinimum, typename Scalar>
IndexedReductionResult<Scalar, Device::GPU> indexedReductionAsync(
    const char* operation_name,
    const DenseStorage<Scalar, Device::GPU>& input,
    ReductionAxis axis,
    ReductionWorkspace& workspace,
    cudaStream_t stream)
{
    const auto plan = reduction_detail::validatedIndexedPlan(
        operation_name, input.rows(), input.cols(), input.size(), axis);
    IndexedReductionResult<Scalar, Device::GPU> result{
        DenseStorage<Scalar, Device::GPU>::uninitializedAsync(
            plan.output_rows, plan.output_cols, stream),
        DenseStorage<Index, Device::GPU>::uninitializedAsync(
            plan.output_rows, plan.output_cols, stream)
    };
    reduction_detail::launchIndexedReduction<FindMinimum>(
        operation_name, input, axis, result.values, result.indices, workspace, stream);
    return result;
}

template <bool FindMinimum, typename Scalar>
IndexedReductionResult<Scalar, Device::GPU> indexedReduction(
    const char* operation_name,
    const DenseStorage<Scalar, Device::GPU>& input,
    ReductionAxis axis,
    ReductionWorkspace& workspace,
    cudaStream_t stream)
{
    const auto plan = reduction_detail::validatedIndexedPlan(
        operation_name, input.rows(), input.cols(), input.size(), axis);
    IndexedReductionResult<Scalar, Device::GPU> result{
        DenseStorage<Scalar, Device::GPU>::uninitialized(
            plan.output_rows, plan.output_cols),
        DenseStorage<Index, Device::GPU>::uninitialized(
            plan.output_rows, plan.output_cols)
    };
    reduction_detail::launchIndexedReduction<FindMinimum>(
        operation_name, input, axis, result.values, result.indices, workspace, stream);
    PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
    return result;
}

template <bool FindMinimum, typename Scalar>
void indexedReduction(
    const char* operation_name,
    const DenseStorage<Scalar, Device::GPU>& input,
    ReductionAxis axis,
    DenseStorage<Scalar, Device::GPU>& values,
    DenseStorage<Index, Device::GPU>& indices,
    ReductionWorkspace& workspace,
    cudaStream_t stream,
    bool synchronize)
{
    reduction_detail::launchIndexedReduction<FindMinimum>(
        operation_name, input, axis, values, indices, workspace, stream);
    if (synchronize)
    {
        PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
    }
}

} // anonymous namespace

#define PLAMATRIX_DEFINE_VALUE_REDUCTION(OP, ASYNC_OP, KIND)                           \
    template <typename Scalar>                                                         \
    DenseStorage<Scalar, Device::GPU> OP(                                               \
        const DenseStorage<Scalar, Device::GPU>& input, ReductionAxis axis)             \
    {                                                                                  \
        ReductionWorkspace workspace;                                                  \
        return valueReduction(KIND, #OP, input, axis, workspace, nullptr);             \
    }                                                                                  \
                                                                                       \
    template <typename Scalar>                                                         \
    DenseStorage<Scalar, Device::GPU> OP(                                               \
        const DenseStorage<Scalar, Device::GPU>& input, ReductionAxis axis,             \
        ReductionWorkspace& workspace, cudaStream_t stream)                            \
    {                                                                                  \
        return valueReduction(KIND, #OP, input, axis, workspace, stream);              \
    }                                                                                  \
                                                                                       \
    template <typename Scalar>                                                         \
    void OP(const DenseStorage<Scalar, Device::GPU>& input, ReductionAxis axis,         \
            DenseStorage<Scalar, Device::GPU>& output, ReductionWorkspace& workspace,   \
            cudaStream_t stream)                                                       \
    {                                                                                  \
        valueReduction(KIND, #OP, input, axis, output, workspace, stream, true);       \
    }                                                                                  \
                                                                                       \
    template <typename Scalar>                                                         \
    DenseStorage<Scalar, Device::GPU> ASYNC_OP(                                         \
        const DenseStorage<Scalar, Device::GPU>& input, ReductionAxis axis,             \
        ReductionWorkspace& workspace, cudaStream_t stream)                            \
    {                                                                                  \
        return valueReductionAsync(KIND, #OP, input, axis, workspace, stream);         \
    }                                                                                  \
                                                                                       \
    template <typename Scalar>                                                         \
    void ASYNC_OP(                                                                     \
        const DenseStorage<Scalar, Device::GPU>& input, ReductionAxis axis,             \
        DenseStorage<Scalar, Device::GPU>& output, ReductionWorkspace& workspace,       \
        cudaStream_t stream)                                                           \
    {                                                                                  \
        valueReduction(KIND, #OP, input, axis, output, workspace, stream, false);      \
    }

PLAMATRIX_DEFINE_VALUE_REDUCTION(sum, sumAsync, reduction_detail::ValueOperation::Sum)
PLAMATRIX_DEFINE_VALUE_REDUCTION(mean, meanAsync, reduction_detail::ValueOperation::Mean)
PLAMATRIX_DEFINE_VALUE_REDUCTION(min, minAsync, reduction_detail::ValueOperation::Minimum)
PLAMATRIX_DEFINE_VALUE_REDUCTION(max, maxAsync, reduction_detail::ValueOperation::Maximum)

#undef PLAMATRIX_DEFINE_VALUE_REDUCTION

#define PLAMATRIX_DEFINE_VIEW_VALUE_REDUCTION(OP, ASYNC_OP, KIND)                   \
    template <typename Scalar>                                                       \
    void OP(ConstMatrixView<Scalar, Device::GPU> input, ReductionAxis axis,            \
            MatrixView<Scalar, Device::GPU> output, ReductionWorkspace& workspace,    \
            cudaStream_t stream)                                                      \
    {                                                                                 \
        launchViewValueReduction(KIND, #OP, input, axis, output, workspace, stream);   \
        PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));                          \
    }                                                                                 \
    template <typename Scalar>                                                       \
    void ASYNC_OP(ConstMatrixView<Scalar, Device::GPU> input, ReductionAxis axis,      \
                  MatrixView<Scalar, Device::GPU> output, ReductionWorkspace& workspace, \
                  cudaStream_t stream)                                                \
    {                                                                                 \
        launchViewValueReduction(KIND, #OP, input, axis, output, workspace, stream);   \
    }

PLAMATRIX_DEFINE_VIEW_VALUE_REDUCTION(sum, sumAsync, reduction_detail::ValueOperation::Sum)
PLAMATRIX_DEFINE_VIEW_VALUE_REDUCTION(min, minAsync, reduction_detail::ValueOperation::Minimum)
PLAMATRIX_DEFINE_VIEW_VALUE_REDUCTION(max, maxAsync, reduction_detail::ValueOperation::Maximum)

#undef PLAMATRIX_DEFINE_VIEW_VALUE_REDUCTION

#define PLAMATRIX_DEFINE_INDEXED_REDUCTION(OP, ASYNC_OP, FIND_MINIMUM)                  \
    template <typename Scalar>                                                         \
    IndexedReductionResult<Scalar, Device::GPU> OP(                                    \
        const DenseStorage<Scalar, Device::GPU>& input, ReductionAxis axis)             \
    {                                                                                  \
        ReductionWorkspace workspace;                                                  \
        return indexedReduction<FIND_MINIMUM>(#OP, input, axis, workspace, nullptr);  \
    }                                                                                  \
                                                                                       \
    template <typename Scalar>                                                         \
    IndexedReductionResult<Scalar, Device::GPU> OP(                                    \
        const DenseStorage<Scalar, Device::GPU>& input, ReductionAxis axis,             \
        ReductionWorkspace& workspace, cudaStream_t stream)                            \
    {                                                                                  \
        return indexedReduction<FIND_MINIMUM>(#OP, input, axis, workspace, stream);   \
    }                                                                                  \
                                                                                       \
    template <typename Scalar>                                                         \
    void OP(const DenseStorage<Scalar, Device::GPU>& input, ReductionAxis axis,         \
            DenseStorage<Scalar, Device::GPU>& values,                                  \
            DenseStorage<Index, Device::GPU>& indices, ReductionWorkspace& workspace,   \
            cudaStream_t stream)                                                       \
    {                                                                                  \
        indexedReduction<FIND_MINIMUM>(                                                \
            #OP, input, axis, values, indices, workspace, stream, true);               \
    }                                                                                  \
                                                                                       \
    template <typename Scalar>                                                         \
    IndexedReductionResult<Scalar, Device::GPU> ASYNC_OP(                              \
        const DenseStorage<Scalar, Device::GPU>& input, ReductionAxis axis,             \
        ReductionWorkspace& workspace, cudaStream_t stream)                            \
    {                                                                                  \
        return indexedReductionAsync<FIND_MINIMUM>(                                   \
            #OP, input, axis, workspace, stream);                                      \
    }                                                                                  \
                                                                                       \
    template <typename Scalar>                                                         \
    void ASYNC_OP(                                                                     \
        const DenseStorage<Scalar, Device::GPU>& input, ReductionAxis axis,             \
        DenseStorage<Scalar, Device::GPU>& values,                                      \
        DenseStorage<Index, Device::GPU>& indices, ReductionWorkspace& workspace,       \
        cudaStream_t stream)                                                           \
    {                                                                                  \
        indexedReduction<FIND_MINIMUM>(                                                \
            #OP, input, axis, values, indices, workspace, stream, false);              \
    }

PLAMATRIX_DEFINE_INDEXED_REDUCTION(argMin, argMinAsync, true)
PLAMATRIX_DEFINE_INDEXED_REDUCTION(argMax, argMaxAsync, false)

#undef PLAMATRIX_DEFINE_INDEXED_REDUCTION

#define PLAMATRIX_INSTANTIATE_VALUE_REDUCTION(OP, ASYNC_OP, Scalar)                    \
    template DenseStorage<Scalar, Device::GPU> OP(                                      \
        const DenseStorage<Scalar, Device::GPU>&, ReductionAxis);                       \
    template DenseStorage<Scalar, Device::GPU> OP(                                      \
        const DenseStorage<Scalar, Device::GPU>&, ReductionAxis,                        \
        ReductionWorkspace&, cudaStream_t);                                            \
    template void OP(                                                                  \
        const DenseStorage<Scalar, Device::GPU>&, ReductionAxis,                        \
        DenseStorage<Scalar, Device::GPU>&, ReductionWorkspace&, cudaStream_t);         \
    template DenseStorage<Scalar, Device::GPU> ASYNC_OP(                                \
        const DenseStorage<Scalar, Device::GPU>&, ReductionAxis,                        \
        ReductionWorkspace&, cudaStream_t);                                            \
    template void ASYNC_OP(                                                            \
        const DenseStorage<Scalar, Device::GPU>&, ReductionAxis,                        \
        DenseStorage<Scalar, Device::GPU>&, ReductionWorkspace&, cudaStream_t)

#define PLAMATRIX_INSTANTIATE_INDEXED_REDUCTION(OP, ASYNC_OP, Scalar)                  \
    template IndexedReductionResult<Scalar, Device::GPU> OP(                           \
        const DenseStorage<Scalar, Device::GPU>&, ReductionAxis);                       \
    template IndexedReductionResult<Scalar, Device::GPU> OP(                           \
        const DenseStorage<Scalar, Device::GPU>&, ReductionAxis,                        \
        ReductionWorkspace&, cudaStream_t);                                            \
    template void OP(                                                                  \
        const DenseStorage<Scalar, Device::GPU>&, ReductionAxis,                        \
        DenseStorage<Scalar, Device::GPU>&, DenseStorage<Index, Device::GPU>&,           \
        ReductionWorkspace&, cudaStream_t);                                            \
    template IndexedReductionResult<Scalar, Device::GPU> ASYNC_OP(                     \
        const DenseStorage<Scalar, Device::GPU>&, ReductionAxis,                        \
        ReductionWorkspace&, cudaStream_t);                                            \
    template void ASYNC_OP(                                                            \
        const DenseStorage<Scalar, Device::GPU>&, ReductionAxis,                        \
        DenseStorage<Scalar, Device::GPU>&, DenseStorage<Index, Device::GPU>&,           \
        ReductionWorkspace&, cudaStream_t)

#define PLAMATRIX_INSTANTIATE_VIEW_VALUE_REDUCTION(OP, ASYNC_OP, Scalar)            \
    template void OP(ConstMatrixView<Scalar, Device::GPU>, ReductionAxis,             \
                     MatrixView<Scalar, Device::GPU>, ReductionWorkspace&, cudaStream_t); \
    template void ASYNC_OP(ConstMatrixView<Scalar, Device::GPU>, ReductionAxis,       \
                           MatrixView<Scalar, Device::GPU>, ReductionWorkspace&, cudaStream_t)

#define PLAMATRIX_INSTANTIATE_REDUCTIONS(Scalar)                                       \
    PLAMATRIX_INSTANTIATE_VALUE_REDUCTION(sum, sumAsync, Scalar);                      \
    PLAMATRIX_INSTANTIATE_VALUE_REDUCTION(mean, meanAsync, Scalar);                    \
    PLAMATRIX_INSTANTIATE_VALUE_REDUCTION(min, minAsync, Scalar);                      \
    PLAMATRIX_INSTANTIATE_VALUE_REDUCTION(max, maxAsync, Scalar);                      \
    PLAMATRIX_INSTANTIATE_VIEW_VALUE_REDUCTION(sum, sumAsync, Scalar);                 \
    PLAMATRIX_INSTANTIATE_VIEW_VALUE_REDUCTION(min, minAsync, Scalar);                 \
    PLAMATRIX_INSTANTIATE_VIEW_VALUE_REDUCTION(max, maxAsync, Scalar);                 \
    PLAMATRIX_INSTANTIATE_INDEXED_REDUCTION(argMin, argMinAsync, Scalar);              \
    PLAMATRIX_INSTANTIATE_INDEXED_REDUCTION(argMax, argMaxAsync, Scalar)

#ifdef PLAMATRIX_USE_FLOAT
PLAMATRIX_INSTANTIATE_REDUCTIONS(float);
#endif

#ifdef PLAMATRIX_USE_DOUBLE
PLAMATRIX_INSTANTIATE_REDUCTIONS(double);
#endif

#undef PLAMATRIX_INSTANTIATE_REDUCTIONS
#undef PLAMATRIX_INSTANTIATE_INDEXED_REDUCTION
#undef PLAMATRIX_INSTANTIATE_VIEW_VALUE_REDUCTION
#undef PLAMATRIX_INSTANTIATE_VALUE_REDUCTION

} // namespace plamatrix::internal
