#include "reduction_detail.h"

namespace plamatrix
{
namespace
{

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> valueReductionAsync(
    reduction_detail::ValueOperation operation,
    const char* operation_name,
    const DenseMatrix<Scalar, Device::GPU>& input,
    ReductionAxis axis,
    ReductionWorkspace& workspace,
    cudaStream_t stream)
{
    const auto plan = reduction_detail::validatedValuePlan(
        operation_name, operation, input.rows(), input.cols(), input.size(), axis);
    auto output = DenseMatrix<Scalar, Device::GPU>::uninitializedAsync(
        plan.output_rows, plan.output_cols, stream);
    reduction_detail::launchValueReduction(
        operation, operation_name, input, axis, output, workspace, stream);
    return output;
}

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> valueReduction(
    reduction_detail::ValueOperation operation,
    const char* operation_name,
    const DenseMatrix<Scalar, Device::GPU>& input,
    ReductionAxis axis,
    ReductionWorkspace& workspace,
    cudaStream_t stream)
{
    const auto plan = reduction_detail::validatedValuePlan(
        operation_name, operation, input.rows(), input.cols(), input.size(), axis);
    auto output = DenseMatrix<Scalar, Device::GPU>::uninitialized(
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
    const DenseMatrix<Scalar, Device::GPU>& input,
    ReductionAxis axis,
    DenseMatrix<Scalar, Device::GPU>& output,
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
    const DenseMatrix<Scalar, Device::GPU>& input,
    ReductionAxis axis,
    ReductionWorkspace& workspace,
    cudaStream_t stream)
{
    const auto plan = reduction_detail::validatedIndexedPlan(
        operation_name, input.rows(), input.cols(), input.size(), axis);
    IndexedReductionResult<Scalar, Device::GPU> result{
        DenseMatrix<Scalar, Device::GPU>::uninitializedAsync(
            plan.output_rows, plan.output_cols, stream),
        DenseMatrix<Index, Device::GPU>::uninitializedAsync(
            plan.output_rows, plan.output_cols, stream)
    };
    reduction_detail::launchIndexedReduction<FindMinimum>(
        operation_name, input, axis, result.values, result.indices, workspace, stream);
    return result;
}

template <bool FindMinimum, typename Scalar>
IndexedReductionResult<Scalar, Device::GPU> indexedReduction(
    const char* operation_name,
    const DenseMatrix<Scalar, Device::GPU>& input,
    ReductionAxis axis,
    ReductionWorkspace& workspace,
    cudaStream_t stream)
{
    const auto plan = reduction_detail::validatedIndexedPlan(
        operation_name, input.rows(), input.cols(), input.size(), axis);
    IndexedReductionResult<Scalar, Device::GPU> result{
        DenseMatrix<Scalar, Device::GPU>::uninitialized(
            plan.output_rows, plan.output_cols),
        DenseMatrix<Index, Device::GPU>::uninitialized(
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
    const DenseMatrix<Scalar, Device::GPU>& input,
    ReductionAxis axis,
    DenseMatrix<Scalar, Device::GPU>& values,
    DenseMatrix<Index, Device::GPU>& indices,
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
    DenseMatrix<Scalar, Device::GPU> OP(                                               \
        const DenseMatrix<Scalar, Device::GPU>& input, ReductionAxis axis)             \
    {                                                                                  \
        ReductionWorkspace workspace;                                                  \
        return valueReduction(KIND, #OP, input, axis, workspace, nullptr);             \
    }                                                                                  \
                                                                                       \
    template <typename Scalar>                                                         \
    DenseMatrix<Scalar, Device::GPU> OP(                                               \
        const DenseMatrix<Scalar, Device::GPU>& input, ReductionAxis axis,             \
        ReductionWorkspace& workspace, cudaStream_t stream)                            \
    {                                                                                  \
        return valueReduction(KIND, #OP, input, axis, workspace, stream);              \
    }                                                                                  \
                                                                                       \
    template <typename Scalar>                                                         \
    void OP(const DenseMatrix<Scalar, Device::GPU>& input, ReductionAxis axis,         \
            DenseMatrix<Scalar, Device::GPU>& output, ReductionWorkspace& workspace,   \
            cudaStream_t stream)                                                       \
    {                                                                                  \
        valueReduction(KIND, #OP, input, axis, output, workspace, stream, true);       \
    }                                                                                  \
                                                                                       \
    template <typename Scalar>                                                         \
    DenseMatrix<Scalar, Device::GPU> ASYNC_OP(                                         \
        const DenseMatrix<Scalar, Device::GPU>& input, ReductionAxis axis,             \
        ReductionWorkspace& workspace, cudaStream_t stream)                            \
    {                                                                                  \
        return valueReductionAsync(KIND, #OP, input, axis, workspace, stream);         \
    }                                                                                  \
                                                                                       \
    template <typename Scalar>                                                         \
    void ASYNC_OP(                                                                     \
        const DenseMatrix<Scalar, Device::GPU>& input, ReductionAxis axis,             \
        DenseMatrix<Scalar, Device::GPU>& output, ReductionWorkspace& workspace,       \
        cudaStream_t stream)                                                           \
    {                                                                                  \
        valueReduction(KIND, #OP, input, axis, output, workspace, stream, false);      \
    }

PLAMATRIX_DEFINE_VALUE_REDUCTION(sum, sumAsync, reduction_detail::ValueOperation::Sum)
PLAMATRIX_DEFINE_VALUE_REDUCTION(mean, meanAsync, reduction_detail::ValueOperation::Mean)
PLAMATRIX_DEFINE_VALUE_REDUCTION(min, minAsync, reduction_detail::ValueOperation::Minimum)
PLAMATRIX_DEFINE_VALUE_REDUCTION(max, maxAsync, reduction_detail::ValueOperation::Maximum)

#undef PLAMATRIX_DEFINE_VALUE_REDUCTION

#define PLAMATRIX_DEFINE_INDEXED_REDUCTION(OP, ASYNC_OP, FIND_MINIMUM)                  \
    template <typename Scalar>                                                         \
    IndexedReductionResult<Scalar, Device::GPU> OP(                                    \
        const DenseMatrix<Scalar, Device::GPU>& input, ReductionAxis axis)             \
    {                                                                                  \
        ReductionWorkspace workspace;                                                  \
        return indexedReduction<FIND_MINIMUM>(#OP, input, axis, workspace, nullptr);  \
    }                                                                                  \
                                                                                       \
    template <typename Scalar>                                                         \
    IndexedReductionResult<Scalar, Device::GPU> OP(                                    \
        const DenseMatrix<Scalar, Device::GPU>& input, ReductionAxis axis,             \
        ReductionWorkspace& workspace, cudaStream_t stream)                            \
    {                                                                                  \
        return indexedReduction<FIND_MINIMUM>(#OP, input, axis, workspace, stream);   \
    }                                                                                  \
                                                                                       \
    template <typename Scalar>                                                         \
    void OP(const DenseMatrix<Scalar, Device::GPU>& input, ReductionAxis axis,         \
            DenseMatrix<Scalar, Device::GPU>& values,                                  \
            DenseMatrix<Index, Device::GPU>& indices, ReductionWorkspace& workspace,   \
            cudaStream_t stream)                                                       \
    {                                                                                  \
        indexedReduction<FIND_MINIMUM>(                                                \
            #OP, input, axis, values, indices, workspace, stream, true);               \
    }                                                                                  \
                                                                                       \
    template <typename Scalar>                                                         \
    IndexedReductionResult<Scalar, Device::GPU> ASYNC_OP(                              \
        const DenseMatrix<Scalar, Device::GPU>& input, ReductionAxis axis,             \
        ReductionWorkspace& workspace, cudaStream_t stream)                            \
    {                                                                                  \
        return indexedReductionAsync<FIND_MINIMUM>(                                   \
            #OP, input, axis, workspace, stream);                                      \
    }                                                                                  \
                                                                                       \
    template <typename Scalar>                                                         \
    void ASYNC_OP(                                                                     \
        const DenseMatrix<Scalar, Device::GPU>& input, ReductionAxis axis,             \
        DenseMatrix<Scalar, Device::GPU>& values,                                      \
        DenseMatrix<Index, Device::GPU>& indices, ReductionWorkspace& workspace,       \
        cudaStream_t stream)                                                           \
    {                                                                                  \
        indexedReduction<FIND_MINIMUM>(                                                \
            #OP, input, axis, values, indices, workspace, stream, false);              \
    }

PLAMATRIX_DEFINE_INDEXED_REDUCTION(argMin, argMinAsync, true)
PLAMATRIX_DEFINE_INDEXED_REDUCTION(argMax, argMaxAsync, false)

#undef PLAMATRIX_DEFINE_INDEXED_REDUCTION

#define PLAMATRIX_INSTANTIATE_VALUE_REDUCTION(OP, ASYNC_OP, Scalar)                    \
    template DenseMatrix<Scalar, Device::GPU> OP(                                      \
        const DenseMatrix<Scalar, Device::GPU>&, ReductionAxis);                       \
    template DenseMatrix<Scalar, Device::GPU> OP(                                      \
        const DenseMatrix<Scalar, Device::GPU>&, ReductionAxis,                        \
        ReductionWorkspace&, cudaStream_t);                                            \
    template void OP(                                                                  \
        const DenseMatrix<Scalar, Device::GPU>&, ReductionAxis,                        \
        DenseMatrix<Scalar, Device::GPU>&, ReductionWorkspace&, cudaStream_t);         \
    template DenseMatrix<Scalar, Device::GPU> ASYNC_OP(                                \
        const DenseMatrix<Scalar, Device::GPU>&, ReductionAxis,                        \
        ReductionWorkspace&, cudaStream_t);                                            \
    template void ASYNC_OP(                                                            \
        const DenseMatrix<Scalar, Device::GPU>&, ReductionAxis,                        \
        DenseMatrix<Scalar, Device::GPU>&, ReductionWorkspace&, cudaStream_t)

#define PLAMATRIX_INSTANTIATE_INDEXED_REDUCTION(OP, ASYNC_OP, Scalar)                  \
    template IndexedReductionResult<Scalar, Device::GPU> OP(                           \
        const DenseMatrix<Scalar, Device::GPU>&, ReductionAxis);                       \
    template IndexedReductionResult<Scalar, Device::GPU> OP(                           \
        const DenseMatrix<Scalar, Device::GPU>&, ReductionAxis,                        \
        ReductionWorkspace&, cudaStream_t);                                            \
    template void OP(                                                                  \
        const DenseMatrix<Scalar, Device::GPU>&, ReductionAxis,                        \
        DenseMatrix<Scalar, Device::GPU>&, DenseMatrix<Index, Device::GPU>&,           \
        ReductionWorkspace&, cudaStream_t);                                            \
    template IndexedReductionResult<Scalar, Device::GPU> ASYNC_OP(                     \
        const DenseMatrix<Scalar, Device::GPU>&, ReductionAxis,                        \
        ReductionWorkspace&, cudaStream_t);                                            \
    template void ASYNC_OP(                                                            \
        const DenseMatrix<Scalar, Device::GPU>&, ReductionAxis,                        \
        DenseMatrix<Scalar, Device::GPU>&, DenseMatrix<Index, Device::GPU>&,           \
        ReductionWorkspace&, cudaStream_t)

#define PLAMATRIX_INSTANTIATE_REDUCTIONS(Scalar)                                       \
    PLAMATRIX_INSTANTIATE_VALUE_REDUCTION(sum, sumAsync, Scalar);                      \
    PLAMATRIX_INSTANTIATE_VALUE_REDUCTION(mean, meanAsync, Scalar);                    \
    PLAMATRIX_INSTANTIATE_VALUE_REDUCTION(min, minAsync, Scalar);                      \
    PLAMATRIX_INSTANTIATE_VALUE_REDUCTION(max, maxAsync, Scalar);                      \
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
#undef PLAMATRIX_INSTANTIATE_VALUE_REDUCTION

} // namespace plamatrix
