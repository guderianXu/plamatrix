#include <algorithm>
#include <cstddef>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

#include <cub/device/device_reduce.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include "reduction_detail.h"
#include "reduction_kernels.cuh"

namespace plamatrix
{
namespace reduction_detail
{
namespace
{

ReductionPlan makeReductionPlan(Index rows,
                                Index cols,
                                Index size,
                                ReductionAxis axis)
{
    switch (axis)
    {
    case ReductionAxis::All:
        return {1, 1, 1, size, 0, 1};
    case ReductionAxis::Rows:
        return {rows, 1, rows, cols, 1, rows};
    case ReductionAxis::Columns:
        return {1, cols, cols, rows, rows, 1};
    default:
        throw std::invalid_argument("reduction: invalid axis");
    }
}

void requireNonEmptyLanes(const char* operation, const ReductionPlan& plan)
{
    if (plan.lane_count > 0 && plan.reduction_length == 0)
    {
        throw std::invalid_argument(
            std::string(operation) + ": cannot reduce an empty lane");
    }
}

template <typename Scalar>
void checkOutput(const char* operation,
                 const DenseMatrix<Scalar, Device::GPU>& output,
                 const ReductionPlan& plan)
{
    if (output.rows() != plan.output_rows || output.cols() != plan.output_cols)
    {
        std::ostringstream oss;
        oss << operation << ": output dimensions must be "
            << plan.output_rows << "x" << plan.output_cols;
        throw std::invalid_argument(oss.str());
    }
}

unsigned int checkedLaneGrid(Index lane_count, const char* operation)
{
    if (lane_count <= 0 || lane_count > static_cast<Index>(std::numeric_limits<int>::max()))
    {
        std::ostringstream oss;
        oss << operation << ": lane count is outside the CUDA grid range: " << lane_count;
        throw std::overflow_error(oss.str());
    }
    return static_cast<unsigned int>(lane_count);
}

std::size_t alignedOffset(std::size_t offset, std::size_t alignment)
{
    if (offset > std::numeric_limits<std::size_t>::max() - (alignment - 1))
    {
        throw std::overflow_error("reduction: workspace alignment overflows size_t");
    }
    return (offset + alignment - 1) & ~(alignment - 1);
}

std::size_t checkedAppend(std::size_t offset,
                          std::size_t alignment,
                          std::size_t bytes)
{
    const std::size_t aligned = alignedOffset(offset, alignment);
    if (aligned > std::numeric_limits<std::size_t>::max() - bytes)
    {
        throw std::overflow_error("reduction: workspace size overflows size_t");
    }
    return aligned + bytes;
}

template <typename Stage>
Stage* reserveCubWorkspace(ReductionWorkspace& workspace,
                           std::size_t temporary_bytes,
                           cudaStream_t stream)
{
    const std::size_t total_bytes = checkedAppend(
        temporary_bytes, alignof(Stage), sizeof(Stage));
    workspace.reserveBytesAsync(total_bytes, stream);
    auto* bytes = static_cast<unsigned char*>(workspace.data());
    return reinterpret_cast<Stage*>(bytes + alignedOffset(temporary_bytes, alignof(Stage)));
}

template <typename Scalar, typename Iterator>
void launchCubSum(Iterator input,
                  Index item_count,
                  DenseMatrix<Scalar, Device::GPU>& output,
                  ReductionWorkspace& workspace,
                  cudaStream_t stream)
{
    std::size_t temporary_bytes = 0;
    PLAMATRIX_CHECK_CUDA(cub::DeviceReduce::Sum(
        nullptr, temporary_bytes, input, static_cast<double*>(nullptr), item_count, stream));
    double* accumulated = reserveCubWorkspace<double>(workspace, temporary_bytes, stream);
    PLAMATRIX_CHECK_CUDA(cub::DeviceReduce::Sum(
        workspace.data(), temporary_bytes, input, accumulated, item_count, stream));
    storeAccumulatedKernel<<<1, 1, 0, stream>>>(accumulated, output.data());
    PLAMATRIX_CHECK_CUDA(cudaGetLastError());
}

template <typename Scalar>
void launchCubMean(const DenseMatrix<Scalar, Device::GPU>& input,
                   DenseMatrix<Scalar, Device::GPU>& output,
                   ReductionWorkspace& workspace,
                   cudaStream_t stream)
{
    const auto summaries = thrust::make_transform_iterator(
        input.data(), MakeMeanSummary<Scalar>{});
    const MeanSummaryReducer summary_reducer;
    const MeanSummary initial_summary{0.0, 0U};
    std::size_t summary_temporary_bytes = 0;
    PLAMATRIX_CHECK_CUDA(cub::DeviceReduce::Reduce(
        nullptr, summary_temporary_bytes, summaries,
        static_cast<MeanSummary*>(nullptr), input.size(),
        summary_reducer, initial_summary, stream));

    const auto normalized_query = thrust::make_transform_iterator(
        input.data(), NormalizeForMean<Scalar>{nullptr});
    const CompensatedSumReducer sum_reducer;
    const CompensatedSum initial_sum{0.0, 0.0};
    std::size_t sum_temporary_bytes = 0;
    PLAMATRIX_CHECK_CUDA(cub::DeviceReduce::Reduce(
        nullptr, sum_temporary_bytes, normalized_query,
        static_cast<CompensatedSum*>(nullptr), input.size(),
        sum_reducer, initial_sum, stream));

    const std::size_t temporary_bytes = std::max(
        summary_temporary_bytes, sum_temporary_bytes);
    const std::size_t summary_offset = alignedOffset(
        temporary_bytes, alignof(MeanSummary));
    const std::size_t after_summary = checkedAppend(
        temporary_bytes, alignof(MeanSummary), sizeof(MeanSummary));
    const std::size_t sum_offset = alignedOffset(
        after_summary, alignof(CompensatedSum));
    const std::size_t total_bytes = checkedAppend(
        after_summary, alignof(CompensatedSum), sizeof(CompensatedSum));
    workspace.reserveBytesAsync(total_bytes, stream);

    auto* bytes = static_cast<unsigned char*>(workspace.data());
    auto* summary = reinterpret_cast<MeanSummary*>(bytes + summary_offset);
    auto* normalized = reinterpret_cast<CompensatedSum*>(bytes + sum_offset);
    PLAMATRIX_CHECK_CUDA(cub::DeviceReduce::Reduce(
        workspace.data(), summary_temporary_bytes, summaries, summary, input.size(),
        summary_reducer, initial_summary, stream));
    const auto normalized_input = thrust::make_transform_iterator(
        input.data(), NormalizeForMean<Scalar>{summary});
    PLAMATRIX_CHECK_CUDA(cub::DeviceReduce::Reduce(
        workspace.data(), sum_temporary_bytes, normalized_input, normalized, input.size(),
        sum_reducer, initial_sum, stream));
    storeMeanKernel<<<1, 1, 0, stream>>>(
        summary, normalized, input.size(), output.data());
    PLAMATRIX_CHECK_CUDA(cudaGetLastError());
}

template <bool FindMinimum, typename Scalar>
void launchCubExtreme(const DenseMatrix<Scalar, Device::GPU>& input,
                      DenseMatrix<Scalar, Device::GPU>& values,
                      Index* indices,
                      ReductionWorkspace& workspace,
                      cudaStream_t stream)
{
    using Pair = ExtremePair<Scalar>;
    const auto pairs = thrust::make_transform_iterator(
        thrust::counting_iterator<Index>(0), LoadExtremePair<Scalar>{input.data()});
    const ExtremeReducer<FindMinimum, Scalar> reducer;
    const Pair initial{Scalar(0), Index(-1)};
    std::size_t temporary_bytes = 0;
    PLAMATRIX_CHECK_CUDA(cub::DeviceReduce::Reduce(
        nullptr, temporary_bytes, pairs, static_cast<Pair*>(nullptr), input.size(),
        reducer, initial, stream));
    Pair* selected = reserveCubWorkspace<Pair>(workspace, temporary_bytes, stream);
    PLAMATRIX_CHECK_CUDA(cub::DeviceReduce::Reduce(
        workspace.data(), temporary_bytes, pairs, selected, input.size(),
        reducer, initial, stream));
    storeExtremeKernel<<<1, 1, 0, stream>>>(selected, values.data(), indices);
    PLAMATRIX_CHECK_CUDA(cudaGetLastError());
}

} // anonymous namespace

ReductionPlan validatedValuePlan(const char* operation,
                                 ValueOperation value_operation,
                                 Index rows,
                                 Index cols,
                                 Index size,
                                 ReductionAxis axis)
{
    const ReductionPlan plan = makeReductionPlan(rows, cols, size, axis);
    if (value_operation != ValueOperation::Sum)
    {
        requireNonEmptyLanes(operation, plan);
    }
    return plan;
}

ReductionPlan validatedIndexedPlan(const char* operation,
                                   Index rows,
                                   Index cols,
                                   Index size,
                                   ReductionAxis axis)
{
    const ReductionPlan plan = makeReductionPlan(rows, cols, size, axis);
    requireNonEmptyLanes(operation, plan);
    return plan;
}

template <typename Scalar>
void launchValueReduction(ValueOperation operation,
                          const char* operation_name,
                          const DenseMatrix<Scalar, Device::GPU>& input,
                          ReductionAxis axis,
                          DenseMatrix<Scalar, Device::GPU>& output,
                          ReductionWorkspace& workspace,
                          cudaStream_t stream)
{
    const ReductionPlan plan = validatedValuePlan(
        operation_name, operation, input.rows(), input.cols(), input.size(), axis);
    checkOutput(operation_name, output, plan);
    if (plan.lane_count == 0)
    {
        workspace.reserveBytesAsync(0, stream);
        return;
    }
    if (plan.reduction_length == 0)
    {
        workspace.reserveBytesAsync(0, stream);
        PLAMATRIX_CHECK_CUDA(cudaMemsetAsync(
            output.data(), 0, detail::checkedAllocationBytes<Scalar>(output.size()), stream));
        return;
    }

    if (axis == ReductionAxis::All)
    {
        if (operation == ValueOperation::Sum)
        {
            launchCubSum<Scalar>(
                thrust::make_transform_iterator(input.data(), CastToDouble<Scalar>{}), input.size(),
                output, workspace, stream);
        }
        else if (operation == ValueOperation::Mean)
        {
            launchCubMean(input, output, workspace, stream);
        }
        else if (operation == ValueOperation::Minimum)
        {
            launchCubExtreme<true>(input, output, nullptr, workspace, stream);
        }
        else
        {
            launchCubExtreme<false>(input, output, nullptr, workspace, stream);
        }
        return;
    }

    workspace.reserveBytesAsync(0, stream);
    const unsigned int grid = checkedLaneGrid(plan.lane_count, operation_name);
    if (operation == ValueOperation::Sum)
    {
        laneSumKernel<<<grid, kBlockSize, 0, stream>>>(input.data(), output.data(), plan);
    }
    else if (operation == ValueOperation::Mean)
    {
        laneMeanKernel<<<grid, kBlockSize, 0, stream>>>(input.data(), output.data(), plan);
    }
    else if (operation == ValueOperation::Minimum)
    {
        laneExtremeKernel<true><<<grid, kBlockSize, 0, stream>>>(
            input.data(), output.data(), nullptr, plan);
    }
    else
    {
        laneExtremeKernel<false><<<grid, kBlockSize, 0, stream>>>(
            input.data(), output.data(), nullptr, plan);
    }
    PLAMATRIX_CHECK_CUDA(cudaGetLastError());
}

template <bool FindMinimum, typename Scalar>
void launchIndexedReduction(const char* operation,
                            const DenseMatrix<Scalar, Device::GPU>& input,
                            ReductionAxis axis,
                            DenseMatrix<Scalar, Device::GPU>& values,
                            DenseMatrix<Index, Device::GPU>& indices,
                            ReductionWorkspace& workspace,
                            cudaStream_t stream)
{
    const ReductionPlan plan = validatedIndexedPlan(
        operation, input.rows(), input.cols(), input.size(), axis);
    checkOutput(operation, values, plan);
    checkOutput(operation, indices, plan);
    if (plan.lane_count == 0)
    {
        workspace.reserveBytesAsync(0, stream);
        return;
    }
    if (axis == ReductionAxis::All)
    {
        launchCubExtreme<FindMinimum>(
            input, values, indices.data(), workspace, stream);
        return;
    }

    workspace.reserveBytesAsync(0, stream);
    const unsigned int grid = checkedLaneGrid(plan.lane_count, operation);
    laneExtremeKernel<FindMinimum><<<grid, kBlockSize, 0, stream>>>(
        input.data(), values.data(), indices.data(), plan);
    PLAMATRIX_CHECK_CUDA(cudaGetLastError());
}

#define PLAMATRIX_INSTANTIATE_DISPATCH(Scalar)                                          \
    template void launchValueReduction<Scalar>(                                        \
        ValueOperation, const char*, const DenseMatrix<Scalar, Device::GPU>&,           \
        ReductionAxis, DenseMatrix<Scalar, Device::GPU>&, ReductionWorkspace&,          \
        cudaStream_t);                                                                  \
    template void launchIndexedReduction<true, Scalar>(                                 \
        const char*, const DenseMatrix<Scalar, Device::GPU>&, ReductionAxis,            \
        DenseMatrix<Scalar, Device::GPU>&, DenseMatrix<Index, Device::GPU>&,            \
        ReductionWorkspace&, cudaStream_t);                                             \
    template void launchIndexedReduction<false, Scalar>(                                \
        const char*, const DenseMatrix<Scalar, Device::GPU>&, ReductionAxis,            \
        DenseMatrix<Scalar, Device::GPU>&, DenseMatrix<Index, Device::GPU>&,            \
        ReductionWorkspace&, cudaStream_t)

#ifdef PLAMATRIX_USE_FLOAT
PLAMATRIX_INSTANTIATE_DISPATCH(float);
#endif

#ifdef PLAMATRIX_USE_DOUBLE
PLAMATRIX_INSTANTIATE_DISPATCH(double);
#endif

#undef PLAMATRIX_INSTANTIATE_DISPATCH

} // namespace reduction_detail
} // namespace plamatrix
