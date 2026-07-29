#pragma once

#include "plamatrix/ops/reduction.h"

namespace plamatrix
{
namespace reduction_detail
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

enum class ValueOperation
{
    Sum,
    Mean,
    Minimum,
    Maximum
};

ReductionPlan validatedValuePlan(const char* operation,
                                 ValueOperation value_operation,
                                 Index rows,
                                 Index cols,
                                 Index size,
                                 ReductionAxis axis);

ReductionPlan validatedIndexedPlan(const char* operation,
                                   Index rows,
                                   Index cols,
                                   Index size,
                                   ReductionAxis axis);

template <typename Scalar>
void launchValueReduction(ValueOperation operation,
                          const char* operation_name,
                          const DenseMatrix<Scalar, Device::GPU>& input,
                          ReductionAxis axis,
                          DenseMatrix<Scalar, Device::GPU>& output,
                          ReductionWorkspace& workspace,
                          cudaStream_t stream);

template <bool FindMinimum, typename Scalar>
void launchIndexedReduction(const char* operation,
                            const DenseMatrix<Scalar, Device::GPU>& input,
                            ReductionAxis axis,
                            DenseMatrix<Scalar, Device::GPU>& values,
                            DenseMatrix<Index, Device::GPU>& indices,
                            ReductionWorkspace& workspace,
                            cudaStream_t stream);

} // namespace reduction_detail
} // namespace plamatrix
