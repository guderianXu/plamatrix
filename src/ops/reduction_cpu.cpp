#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "plamatrix/core/parallel.h"
#include "plamatrix/ops/reduction.h"

namespace plamatrix
{
namespace
{

struct ReductionPlan
{
    Index output_rows;
    Index output_cols;
    Index lane_count;
    Index reduction_length;
    Index lane_stride;
    Index reduction_stride;
};

template <typename Scalar>
ReductionPlan makeReductionPlan(
    const DenseMatrix<Scalar, Device::CPU>& input,
    ReductionAxis axis)
{
    switch (axis)
    {
    case ReductionAxis::All:
        return {1, 1, 1, input.size(), 0, 1};
    case ReductionAxis::Rows:
        return {input.rows(), 1, input.rows(), input.cols(), 1, input.rows()};
    case ReductionAxis::Columns:
        return {1, input.cols(), input.cols(), input.rows(), input.rows(), 1};
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

template <typename Operation>
void forEachLane(const ReductionPlan& plan, Operation operation)
{
    if (detail::shouldUseOpenMp(plan.lane_count))
    {
        #pragma omp parallel for
        for (Index lane = 0; lane < plan.lane_count; ++lane)
        {
            operation(lane);
        }
    }
    else
    {
        for (Index lane = 0; lane < plan.lane_count; ++lane)
        {
            operation(lane);
        }
    }
}

template <typename Scalar>
using Accumulator = std::conditional_t<std::is_same_v<Scalar, float>, double, Scalar>;

template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> sumLanes(
    const DenseMatrix<Scalar, Device::CPU>& input,
    const ReductionPlan& plan)
{
    DenseMatrix<Scalar, Device::CPU> output(plan.output_rows, plan.output_cols);
    forEachLane(plan, [&](Index lane) {
        Accumulator<Scalar> accumulated = Accumulator<Scalar>(0);
        const Index lane_base = lane * plan.lane_stride;
        for (Index reduced_index = 0; reduced_index < plan.reduction_length; ++reduced_index)
        {
            accumulated += static_cast<Accumulator<Scalar>>(
                input.data()[lane_base + reduced_index * plan.reduction_stride]);
        }
        output.data()[lane] = static_cast<Scalar>(accumulated);
    });
    return output;
}

template <typename Scalar>
Scalar meanLane(
    const DenseMatrix<Scalar, Device::CPU>& input,
    const ReductionPlan& plan,
    Index lane) noexcept
{
    using Accum = Accumulator<Scalar>;
    const Index lane_base = lane * plan.lane_stride;
    Accum scale = Accum(0);
    bool has_positive_infinity = false;
    bool has_negative_infinity = false;

    for (Index reduced_index = 0; reduced_index < plan.reduction_length; ++reduced_index)
    {
        const Scalar source = input.data()[lane_base + reduced_index * plan.reduction_stride];
        const Accum value = static_cast<Accum>(source);
        if (std::isnan(value))
        {
            return source;
        }
        if (std::isinf(value))
        {
            has_positive_infinity = has_positive_infinity || value > Accum(0);
            has_negative_infinity = has_negative_infinity || value < Accum(0);
        }
        else
        {
            scale = std::max(scale, std::abs(value));
        }
    }

    if (has_positive_infinity && has_negative_infinity)
    {
        return std::numeric_limits<Scalar>::quiet_NaN();
    }
    if (has_positive_infinity)
    {
        return std::numeric_limits<Scalar>::infinity();
    }
    if (has_negative_infinity)
    {
        return -std::numeric_limits<Scalar>::infinity();
    }
    if (scale == Accum(0))
    {
        return Scalar(0);
    }

    const Accum highest = std::numeric_limits<Accum>::max();
    const Accum lowest = std::numeric_limits<Accum>::lowest();
    Accum ordinary_sum = Accum(0);
    bool requires_scaled_accumulation = false;
    for (Index reduced_index = 0; reduced_index < plan.reduction_length; ++reduced_index)
    {
        const Accum value = static_cast<Accum>(
            input.data()[lane_base + reduced_index * plan.reduction_stride]);
        const bool positive_overflow_risk = value > Accum(0)
                                          && ordinary_sum > Accum(0)
                                          && value > highest - ordinary_sum;
        const bool negative_overflow_risk = value < Accum(0)
                                          && ordinary_sum < Accum(0)
                                          && value < lowest - ordinary_sum;
        if (positive_overflow_risk || negative_overflow_risk)
        {
            requires_scaled_accumulation = true;
            break;
        }

        const Accum next = ordinary_sum + value;
        const Accum addition_residual = std::abs(ordinary_sum) >= std::abs(value)
                                          ? (ordinary_sum - next) + value
                                          : (value - next) + ordinary_sum;
        if (addition_residual != Accum(0))
        {
            requires_scaled_accumulation = true;
            break;
        }
        ordinary_sum = next;
    }
    if (!requires_scaled_accumulation)
    {
        return static_cast<Scalar>(
            ordinary_sum / static_cast<Accum>(plan.reduction_length));
    }

    Accum accumulated = Accum(0);
    Accum compensation = Accum(0);
    for (Index reduced_index = 0; reduced_index < plan.reduction_length; ++reduced_index)
    {
        const Accum normalized = static_cast<Accum>(
            input.data()[lane_base + reduced_index * plan.reduction_stride]) / scale;
        const Accum next = accumulated + normalized;
        if (std::abs(accumulated) >= std::abs(normalized))
        {
            compensation += (accumulated - next) + normalized;
        }
        else
        {
            compensation += (normalized - next) + accumulated;
        }
        accumulated = next;
    }

    const Accum normalized_mean = (accumulated + compensation)
                                / static_cast<Accum>(plan.reduction_length);
    return static_cast<Scalar>(normalized_mean * scale);
}

template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> meanLanes(
    const DenseMatrix<Scalar, Device::CPU>& input,
    const ReductionPlan& plan)
{
    DenseMatrix<Scalar, Device::CPU> output(plan.output_rows, plan.output_cols);
    forEachLane(plan, [&](Index lane) {
        output.data()[lane] = meanLane(input, plan, lane);
    });
    return output;
}

template <typename Scalar>
struct LaneExtreme
{
    Scalar value;
    Index index;
};

template <bool FindMinimum, typename Scalar>
LaneExtreme<Scalar> reduceExtremeLane(
    const DenseMatrix<Scalar, Device::CPU>& input,
    const ReductionPlan& plan,
    Index lane) noexcept
{
    const Index lane_base = lane * plan.lane_stride;
    Scalar selected = input.data()[lane_base];
    Index selected_index = 0;
    for (Index reduced_index = 1; reduced_index < plan.reduction_length; ++reduced_index)
    {
        const Scalar candidate = input.data()[lane_base + reduced_index * plan.reduction_stride];
        const bool selected_nan = std::isnan(selected);
        const bool candidate_nan = std::isnan(candidate);
        const bool better = FindMinimum ? candidate < selected : candidate > selected;
        if ((!selected_nan && candidate_nan)
            || (!selected_nan && !candidate_nan && better))
        {
            selected = candidate;
            selected_index = reduced_index;
        }
    }
    return {selected, selected_index};
}

template <bool FindMinimum, typename Scalar>
DenseMatrix<Scalar, Device::CPU> extremeValues(
    const DenseMatrix<Scalar, Device::CPU>& input,
    const ReductionPlan& plan)
{
    DenseMatrix<Scalar, Device::CPU> output(plan.output_rows, plan.output_cols);
    forEachLane(plan, [&](Index lane) {
        output.data()[lane] = reduceExtremeLane<FindMinimum>(input, plan, lane).value;
    });
    return output;
}

template <bool FindMinimum, typename Scalar>
IndexedReductionResult<Scalar, Device::CPU> indexedExtreme(
    const DenseMatrix<Scalar, Device::CPU>& input,
    const ReductionPlan& plan)
{
    IndexedReductionResult<Scalar, Device::CPU> output{
        DenseMatrix<Scalar, Device::CPU>(plan.output_rows, plan.output_cols),
        DenseMatrix<Index, Device::CPU>(plan.output_rows, plan.output_cols)
    };
    forEachLane(plan, [&](Index lane) {
        const auto selected = reduceExtremeLane<FindMinimum>(input, plan, lane);
        output.values.data()[lane] = selected.value;
        output.indices.data()[lane] = selected.index;
    });
    return output;
}

} // anonymous namespace

template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> sum(
    const DenseMatrix<Scalar, Device::CPU>& input,
    ReductionAxis axis)
{
    const ReductionPlan plan = makeReductionPlan(input, axis);
    return sumLanes(input, plan);
}

template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> mean(
    const DenseMatrix<Scalar, Device::CPU>& input,
    ReductionAxis axis)
{
    const ReductionPlan plan = makeReductionPlan(input, axis);
    requireNonEmptyLanes("mean", plan);
    return meanLanes(input, plan);
}

template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> min(
    const DenseMatrix<Scalar, Device::CPU>& input,
    ReductionAxis axis)
{
    const ReductionPlan plan = makeReductionPlan(input, axis);
    requireNonEmptyLanes("min", plan);
    return extremeValues<true>(input, plan);
}

template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> max(
    const DenseMatrix<Scalar, Device::CPU>& input,
    ReductionAxis axis)
{
    const ReductionPlan plan = makeReductionPlan(input, axis);
    requireNonEmptyLanes("max", plan);
    return extremeValues<false>(input, plan);
}

template <typename Scalar>
IndexedReductionResult<Scalar, Device::CPU> argMin(
    const DenseMatrix<Scalar, Device::CPU>& input,
    ReductionAxis axis)
{
    const ReductionPlan plan = makeReductionPlan(input, axis);
    requireNonEmptyLanes("argMin", plan);
    return indexedExtreme<true>(input, plan);
}

template <typename Scalar>
IndexedReductionResult<Scalar, Device::CPU> argMax(
    const DenseMatrix<Scalar, Device::CPU>& input,
    ReductionAxis axis)
{
    const ReductionPlan plan = makeReductionPlan(input, axis);
    requireNonEmptyLanes("argMax", plan);
    return indexedExtreme<false>(input, plan);
}

#ifdef PLAMATRIX_USE_FLOAT
template DenseMatrix<float, Device::CPU> sum(const DenseMatrix<float, Device::CPU>&, ReductionAxis);
template DenseMatrix<float, Device::CPU> mean(const DenseMatrix<float, Device::CPU>&, ReductionAxis);
template DenseMatrix<float, Device::CPU> min(const DenseMatrix<float, Device::CPU>&, ReductionAxis);
template DenseMatrix<float, Device::CPU> max(const DenseMatrix<float, Device::CPU>&, ReductionAxis);
template IndexedReductionResult<float, Device::CPU> argMin(
    const DenseMatrix<float, Device::CPU>&,
    ReductionAxis);
template IndexedReductionResult<float, Device::CPU> argMax(
    const DenseMatrix<float, Device::CPU>&,
    ReductionAxis);
#endif

#ifdef PLAMATRIX_USE_DOUBLE
template DenseMatrix<double, Device::CPU> sum(const DenseMatrix<double, Device::CPU>&, ReductionAxis);
template DenseMatrix<double, Device::CPU> mean(const DenseMatrix<double, Device::CPU>&, ReductionAxis);
template DenseMatrix<double, Device::CPU> min(const DenseMatrix<double, Device::CPU>&, ReductionAxis);
template DenseMatrix<double, Device::CPU> max(const DenseMatrix<double, Device::CPU>&, ReductionAxis);
template IndexedReductionResult<double, Device::CPU> argMin(
    const DenseMatrix<double, Device::CPU>&,
    ReductionAxis);
template IndexedReductionResult<double, Device::CPU> argMax(
    const DenseMatrix<double, Device::CPU>&,
    ReductionAxis);
#endif

} // namespace plamatrix
