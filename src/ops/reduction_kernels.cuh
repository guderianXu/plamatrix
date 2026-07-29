#pragma once

#include <cfloat>

#include <math_constants.h>

#include "reduction_detail.h"

namespace plamatrix
{
namespace reduction_detail
{

constexpr int kBlockSize = 256;
constexpr unsigned int kMeanHasNan = 1U;
constexpr unsigned int kMeanHasPositiveInfinity = 2U;
constexpr unsigned int kMeanHasNegativeInfinity = 4U;

template <typename Scalar>
struct CastToDouble
{
    __host__ __device__ double operator()(Scalar value) const
    {
        return static_cast<double>(value);
    }
};

template <typename Scalar>
struct ExtremePair
{
    Scalar value;
    Index source_index;
};

template <typename Scalar>
struct LoadExtremePair
{
    const Scalar* input;

    __host__ __device__ ExtremePair<Scalar> operator()(Index index) const
    {
        return {input[index], index};
    }
};

template <bool FindMinimum, typename Scalar>
struct ExtremeReducer
{
    __host__ __device__ ExtremePair<Scalar> operator()(
        const ExtremePair<Scalar>& left,
        const ExtremePair<Scalar>& right) const
    {
        if (left.source_index < 0)
        {
            return right;
        }
        if (right.source_index < 0)
        {
            return left;
        }

        const bool left_nan = left.value != left.value;
        const bool right_nan = right.value != right.value;
        if (left_nan != right_nan)
        {
            return right_nan ? right : left;
        }

        const bool right_better = FindMinimum
                                    ? right.value < left.value
                                    : right.value > left.value;
        if (right_better || (!(left.value < right.value)
                             && !(right.value < left.value)
                             && right.source_index < left.source_index))
        {
            return right;
        }
        return left;
    }
};

struct MeanSummary
{
    double max_abs;
    unsigned int flags;
};

struct MeanSummaryReducer
{
    __host__ __device__ MeanSummary operator()(
        const MeanSummary& left,
        const MeanSummary& right) const
    {
        return {
            left.max_abs > right.max_abs ? left.max_abs : right.max_abs,
            left.flags | right.flags
        };
    }
};

template <typename Scalar>
struct MakeMeanSummary
{
    __host__ __device__ MeanSummary operator()(Scalar source) const
    {
        const double value = static_cast<double>(source);
        if (value != value)
        {
            return {0.0, kMeanHasNan};
        }
        if (value > DBL_MAX)
        {
            return {0.0, kMeanHasPositiveInfinity};
        }
        if (value < -DBL_MAX)
        {
            return {0.0, kMeanHasNegativeInfinity};
        }
        return {value < 0.0 ? -value : value, 0U};
    }
};

struct CompensatedSum
{
    double sum;
    double compensation;
};

__host__ __device__ inline double absoluteDouble(double value)
{
    return value < 0.0 ? -value : value;
}

__host__ __device__ inline CompensatedSum addCompensated(
    CompensatedSum accumulated,
    double value)
{
    const double next = accumulated.sum + value;
    if (absoluteDouble(accumulated.sum) >= absoluteDouble(value))
    {
        accumulated.compensation += (accumulated.sum - next) + value;
    }
    else
    {
        accumulated.compensation += (value - next) + accumulated.sum;
    }
    accumulated.sum = next;
    return accumulated;
}

struct CompensatedSumReducer
{
    __host__ __device__ CompensatedSum operator()(
        const CompensatedSum& left,
        const CompensatedSum& right) const
    {
        CompensatedSum combined = addCompensated(left, right.sum);
        return addCompensated(combined, right.compensation);
    }
};

template <typename Scalar>
struct NormalizeForMean
{
    const MeanSummary* summary;

    __host__ __device__ CompensatedSum operator()(Scalar source) const
    {
        if (summary->flags != 0U || summary->max_abs == 0.0)
        {
            return {0.0, 0.0};
        }
        return {static_cast<double>(source) / summary->max_abs, 0.0};
    }
};

template <typename Scalar>
__device__ Scalar finalizeMean(const MeanSummary& summary,
                               const CompensatedSum& normalized,
                               Index reduction_length)
{
    if ((summary.flags & kMeanHasNan) != 0U
        || (summary.flags & kMeanHasPositiveInfinity) != 0U
           && (summary.flags & kMeanHasNegativeInfinity) != 0U)
    {
        return static_cast<Scalar>(CUDART_NAN);
    }
    if ((summary.flags & kMeanHasPositiveInfinity) != 0U)
    {
        return static_cast<Scalar>(CUDART_INF);
    }
    if ((summary.flags & kMeanHasNegativeInfinity) != 0U)
    {
        return static_cast<Scalar>(-CUDART_INF);
    }
    if (summary.max_abs == 0.0)
    {
        return Scalar(0);
    }

    double normalized_mean = (normalized.sum + normalized.compensation)
                           / static_cast<double>(reduction_length);
    normalized_mean = normalized_mean > 1.0 ? 1.0 : normalized_mean;
    normalized_mean = normalized_mean < -1.0 ? -1.0 : normalized_mean;
    return static_cast<Scalar>(normalized_mean * summary.max_abs);
}

template <typename Scalar>
__global__ void storeAccumulatedKernel(const double* accumulated, Scalar* output)
{
    output[0] = static_cast<Scalar>(accumulated[0]);
}

template <typename Scalar>
__global__ void storeExtremeKernel(const ExtremePair<Scalar>* selected,
                                   Scalar* values,
                                   Index* indices)
{
    values[0] = selected[0].value;
    if (indices != nullptr)
    {
        indices[0] = selected[0].source_index;
    }
}

template <typename Scalar>
__global__ void storeMeanKernel(const MeanSummary* summary,
                                const CompensatedSum* normalized,
                                Index reduction_length,
                                Scalar* output)
{
    output[0] = finalizeMean<Scalar>(summary[0], normalized[0], reduction_length);
}

template <typename Scalar>
__global__ void laneSumKernel(const Scalar* input,
                              Scalar* output,
                              ReductionPlan plan)
{
    __shared__ double partial[kBlockSize];
    const Index lane = static_cast<Index>(blockIdx.x);
    const Index lane_base = lane * plan.lane_stride;
    double accumulated = 0.0;
    for (Index index = threadIdx.x; index < plan.reduction_length; index += blockDim.x)
    {
        accumulated += static_cast<double>(
            input[lane_base + index * plan.reduction_stride]);
    }
    partial[threadIdx.x] = accumulated;
    __syncthreads();

    for (int offset = kBlockSize / 2; offset > 0; offset /= 2)
    {
        if (threadIdx.x < offset)
        {
            partial[threadIdx.x] += partial[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        output[lane] = static_cast<Scalar>(partial[0]);
    }
}

template <typename Scalar>
__global__ void laneMeanKernel(const Scalar* input,
                               Scalar* output,
                               ReductionPlan plan)
{
    __shared__ MeanSummary summaries[kBlockSize];
    __shared__ CompensatedSum normalized[kBlockSize];
    const Index lane = static_cast<Index>(blockIdx.x);
    const Index lane_base = lane * plan.lane_stride;
    const MeanSummaryReducer summary_reducer;
    const CompensatedSumReducer sum_reducer;
    MeanSummary local_summary{0.0, 0U};
    for (Index index = threadIdx.x; index < plan.reduction_length; index += blockDim.x)
    {
        local_summary = summary_reducer(
            local_summary,
            MakeMeanSummary<Scalar>{}(
                input[lane_base + index * plan.reduction_stride]));
    }
    summaries[threadIdx.x] = local_summary;
    __syncthreads();

    for (int offset = kBlockSize / 2; offset > 0; offset /= 2)
    {
        if (threadIdx.x < offset)
        {
            summaries[threadIdx.x] = summary_reducer(
                summaries[threadIdx.x], summaries[threadIdx.x + offset]);
        }
        __syncthreads();
    }

    CompensatedSum local_sum{0.0, 0.0};
    const MeanSummary summary = summaries[0];
    if (summary.flags == 0U && summary.max_abs != 0.0)
    {
        for (Index index = threadIdx.x; index < plan.reduction_length; index += blockDim.x)
        {
            const double value = static_cast<double>(
                input[lane_base + index * plan.reduction_stride]) / summary.max_abs;
            local_sum = addCompensated(local_sum, value);
        }
    }
    normalized[threadIdx.x] = local_sum;
    __syncthreads();

    for (int offset = kBlockSize / 2; offset > 0; offset /= 2)
    {
        if (threadIdx.x < offset)
        {
            normalized[threadIdx.x] = sum_reducer(
                normalized[threadIdx.x], normalized[threadIdx.x + offset]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        output[lane] = finalizeMean<Scalar>(summary, normalized[0], plan.reduction_length);
    }
}

template <bool FindMinimum, typename Scalar>
__global__ void laneExtremeKernel(const Scalar* input,
                                  Scalar* values,
                                  Index* indices,
                                  ReductionPlan plan)
{
    __shared__ ExtremePair<Scalar> partial[kBlockSize];
    const Index lane = static_cast<Index>(blockIdx.x);
    const Index lane_base = lane * plan.lane_stride;
    ExtremePair<Scalar> selected{Scalar(0), Index(-1)};
    const ExtremeReducer<FindMinimum, Scalar> reducer;
    for (Index index = threadIdx.x; index < plan.reduction_length; index += blockDim.x)
    {
        selected = reducer(
            selected,
            {input[lane_base + index * plan.reduction_stride], index});
    }
    partial[threadIdx.x] = selected;
    __syncthreads();

    for (int offset = kBlockSize / 2; offset > 0; offset /= 2)
    {
        if (threadIdx.x < offset)
        {
            partial[threadIdx.x] = reducer(
                partial[threadIdx.x], partial[threadIdx.x + offset]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        values[lane] = partial[0].value;
        if (indices != nullptr)
        {
            indices[lane] = partial[0].source_index;
        }
    }
}

} // namespace reduction_detail
} // namespace plamatrix
