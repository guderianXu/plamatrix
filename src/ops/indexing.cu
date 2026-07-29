#include <cstddef>
#include <stdexcept>

#include "indexing_detail.h"

namespace plamatrix
{
namespace
{

void synchronizeAndCheck(
    cudaStream_t stream,
    IndexingWorkspace& workspace,
    const char* operation)
{
    PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
    workspace.checkStatus(operation);
}

void requireIndexVector(
    const char* operation,
    const DenseMatrix<Index, Device::GPU>& indices)
{
    if (indices.cols() != 1)
    {
        throw std::invalid_argument(std::string(operation) + ": indices must have shape K x 1");
    }
}

template <typename Scalar>
void requireCompactMask(
    const DenseMatrix<Scalar, Device::GPU>& input,
    const DenseMatrix<std::uint8_t, Device::GPU>& keep_mask)
{
    if (keep_mask.rows() != input.rows() || keep_mask.cols() != 1)
    {
        throw std::invalid_argument(
            "compactRows: keep_mask must have shape input.rows() x 1");
    }
}

} // namespace

DenseMatrix<Index, Device::GPU> exclusiveScan(
    const DenseMatrix<Index, Device::GPU>& counts)
{
    IndexingWorkspace workspace;
    return exclusiveScan(counts, workspace, nullptr);
}

DenseMatrix<Index, Device::GPU> exclusiveScan(
    const DenseMatrix<Index, Device::GPU>& counts,
    IndexingWorkspace& workspace,
    cudaStream_t stream)
{
    DenseMatrix<Index, Device::GPU> output(counts.rows(), counts.cols());
    indexing_detail::launchExclusiveScan(counts, output, workspace, stream);
    synchronizeAndCheck(stream, workspace, "exclusiveScan");
    return output;
}

void exclusiveScan(
    const DenseMatrix<Index, Device::GPU>& counts,
    DenseMatrix<Index, Device::GPU>& output,
    IndexingWorkspace& workspace,
    cudaStream_t stream)
{
    indexing_detail::launchExclusiveScan(counts, output, workspace, stream);
    synchronizeAndCheck(stream, workspace, "exclusiveScan");
}

DenseMatrix<Index, Device::GPU> exclusiveScanAsync(
    const DenseMatrix<Index, Device::GPU>& counts,
    IndexingWorkspace& workspace,
    cudaStream_t stream)
{
    auto output = DenseMatrix<Index, Device::GPU>::uninitializedAsync(
        counts.rows(), counts.cols(), stream);
    indexing_detail::launchExclusiveScan(counts, output, workspace, stream);
    return output;
}

void exclusiveScanAsync(
    const DenseMatrix<Index, Device::GPU>& counts,
    DenseMatrix<Index, Device::GPU>& output,
    IndexingWorkspace& workspace,
    cudaStream_t stream)
{
    indexing_detail::launchExclusiveScan(counts, output, workspace, stream);
}

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> gatherRows(
    const DenseMatrix<Scalar, Device::GPU>& input,
    const DenseMatrix<Index, Device::GPU>& indices)
{
    IndexingWorkspace workspace;
    return gatherRows(input, indices, workspace, nullptr);
}

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> gatherRows(
    const DenseMatrix<Scalar, Device::GPU>& input,
    const DenseMatrix<Index, Device::GPU>& indices,
    IndexingWorkspace& workspace,
    cudaStream_t stream)
{
    requireIndexVector("gatherRows", indices);
    DenseMatrix<Scalar, Device::GPU> output(indices.rows(), input.cols());
    indexing_detail::launchGatherRows(input, indices, output, workspace, stream);
    synchronizeAndCheck(stream, workspace, "gatherRows");
    return output;
}

template <typename Scalar>
void gatherRows(
    const DenseMatrix<Scalar, Device::GPU>& input,
    const DenseMatrix<Index, Device::GPU>& indices,
    DenseMatrix<Scalar, Device::GPU>& output,
    IndexingWorkspace& workspace,
    cudaStream_t stream)
{
    indexing_detail::launchGatherRows(input, indices, output, workspace, stream);
    synchronizeAndCheck(stream, workspace, "gatherRows");
}

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> gatherRowsAsync(
    const DenseMatrix<Scalar, Device::GPU>& input,
    const DenseMatrix<Index, Device::GPU>& indices,
    IndexingWorkspace& workspace,
    cudaStream_t stream)
{
    requireIndexVector("gatherRowsAsync", indices);
    auto output = DenseMatrix<Scalar, Device::GPU>::uninitializedAsync(
        indices.rows(), input.cols(), stream);
    indexing_detail::launchGatherRows(input, indices, output, workspace, stream);
    return output;
}

template <typename Scalar>
void gatherRowsAsync(
    const DenseMatrix<Scalar, Device::GPU>& input,
    const DenseMatrix<Index, Device::GPU>& indices,
    DenseMatrix<Scalar, Device::GPU>& output,
    IndexingWorkspace& workspace,
    cudaStream_t stream)
{
    indexing_detail::launchGatherRows(input, indices, output, workspace, stream);
}

template <typename Scalar>
void scatterRows(
    const DenseMatrix<Scalar, Device::GPU>& values,
    const DenseMatrix<Index, Device::GPU>& indices,
    DenseMatrix<Scalar, Device::GPU>& output)
{
    IndexingWorkspace workspace;
    scatterRows(values, indices, output, workspace, nullptr);
}

template <typename Scalar>
void scatterRows(
    const DenseMatrix<Scalar, Device::GPU>& values,
    const DenseMatrix<Index, Device::GPU>& indices,
    DenseMatrix<Scalar, Device::GPU>& output,
    IndexingWorkspace& workspace,
    cudaStream_t stream)
{
    indexing_detail::launchScatterRows(values, indices, output, workspace, stream);
    synchronizeAndCheck(stream, workspace, "scatterRows");
}

template <typename Scalar>
void scatterRowsAsync(
    const DenseMatrix<Scalar, Device::GPU>& values,
    const DenseMatrix<Index, Device::GPU>& indices,
    DenseMatrix<Scalar, Device::GPU>& output,
    IndexingWorkspace& workspace,
    cudaStream_t stream)
{
    indexing_detail::launchScatterRows(values, indices, output, workspace, stream);
}

template <typename Scalar>
CompactRowsResult<Scalar, Device::GPU> compactRows(
    const DenseMatrix<Scalar, Device::GPU>& input,
    const DenseMatrix<std::uint8_t, Device::GPU>& keep_mask)
{
    IndexingWorkspace workspace;
    return compactRows(input, keep_mask, workspace, nullptr);
}

template <typename Scalar>
CompactRowsResult<Scalar, Device::GPU> compactRows(
    const DenseMatrix<Scalar, Device::GPU>& input,
    const DenseMatrix<std::uint8_t, Device::GPU>& keep_mask,
    IndexingWorkspace& workspace,
    cudaStream_t stream)
{
    requireCompactMask(input, keep_mask);
    DenseMatrix<Scalar, Device::GPU> capacity_output(input.rows(), input.cols());
    DenseMatrix<Index, Device::GPU> capacity_sources(input.rows(), 1);
    DenseMatrix<Index, Device::GPU> selected_count(1, 1);
    indexing_detail::launchCompactRows(
        input, keep_mask, capacity_output, capacity_sources, selected_count, workspace, stream);
    synchronizeAndCheck(stream, workspace, "compactRows");

    Index selected = 0;
    PLAMATRIX_CHECK_CUDA(cudaMemcpy(
        &selected, selected_count.data(), sizeof(Index), cudaMemcpyDeviceToHost));
    if (selected < 0 || selected > input.rows())
    {
        throw std::runtime_error("compactRows: CUB returned an invalid selected count");
    }
    CompactRowsResult<Scalar, Device::GPU> result{
        DenseMatrix<Scalar, Device::GPU>(selected, input.cols()),
        DenseMatrix<Index, Device::GPU>(selected, 1)
    };
    if (selected != 0)
    {
        const std::size_t selected_size = static_cast<std::size_t>(selected);
        PLAMATRIX_CHECK_CUDA(cudaMemcpyAsync(
            result.sourceIndices.data(), capacity_sources.data(),
            detail::checkedAllocationBytes<Index>(selected_size),
            cudaMemcpyDeviceToDevice, stream));
        if (input.cols() != 0)
        {
            const std::size_t selected_bytes =
                detail::checkedAllocationBytes<Scalar>(selected_size);
            const std::size_t capacity_pitch = detail::checkedAllocationBytes<Scalar>(
                static_cast<std::size_t>(input.rows()));
            PLAMATRIX_CHECK_CUDA(cudaMemcpy2DAsync(
                result.values.data(), selected_bytes,
                capacity_output.data(), capacity_pitch,
                selected_bytes, static_cast<std::size_t>(input.cols()),
                cudaMemcpyDeviceToDevice, stream));
        }
        PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
    }
    return result;
}

template <typename Scalar>
void compactRows(
    const DenseMatrix<Scalar, Device::GPU>& input,
    const DenseMatrix<std::uint8_t, Device::GPU>& keep_mask,
    DenseMatrix<Scalar, Device::GPU>& capacity_output,
    DenseMatrix<Index, Device::GPU>& capacity_source_indices,
    DenseMatrix<Index, Device::GPU>& selected_count,
    IndexingWorkspace& workspace,
    cudaStream_t stream)
{
    indexing_detail::launchCompactRows(
        input, keep_mask, capacity_output, capacity_source_indices,
        selected_count, workspace, stream);
    synchronizeAndCheck(stream, workspace, "compactRows");
}

template <typename Scalar>
void compactRowsAsync(
    const DenseMatrix<Scalar, Device::GPU>& input,
    const DenseMatrix<std::uint8_t, Device::GPU>& keep_mask,
    DenseMatrix<Scalar, Device::GPU>& capacity_output,
    DenseMatrix<Index, Device::GPU>& capacity_source_indices,
    DenseMatrix<Index, Device::GPU>& selected_count,
    IndexingWorkspace& workspace,
    cudaStream_t stream)
{
    indexing_detail::launchCompactRows(
        input, keep_mask, capacity_output, capacity_source_indices,
        selected_count, workspace, stream);
}

#define PLAMATRIX_INSTANTIATE_GPU_INDEXING(Scalar)                                     \
    template DenseMatrix<Scalar, Device::GPU> gatherRows(                              \
        const DenseMatrix<Scalar, Device::GPU>&, const DenseMatrix<Index, Device::GPU>&);\
    template DenseMatrix<Scalar, Device::GPU> gatherRows(                              \
        const DenseMatrix<Scalar, Device::GPU>&, const DenseMatrix<Index, Device::GPU>&,\
        IndexingWorkspace&, cudaStream_t);                                              \
    template void gatherRows(                                                          \
        const DenseMatrix<Scalar, Device::GPU>&, const DenseMatrix<Index, Device::GPU>&,\
        DenseMatrix<Scalar, Device::GPU>&, IndexingWorkspace&, cudaStream_t);           \
    template DenseMatrix<Scalar, Device::GPU> gatherRowsAsync(                         \
        const DenseMatrix<Scalar, Device::GPU>&, const DenseMatrix<Index, Device::GPU>&,\
        IndexingWorkspace&, cudaStream_t);                                              \
    template void gatherRowsAsync(                                                     \
        const DenseMatrix<Scalar, Device::GPU>&, const DenseMatrix<Index, Device::GPU>&,\
        DenseMatrix<Scalar, Device::GPU>&, IndexingWorkspace&, cudaStream_t);           \
    template void scatterRows(                                                         \
        const DenseMatrix<Scalar, Device::GPU>&, const DenseMatrix<Index, Device::GPU>&,\
        DenseMatrix<Scalar, Device::GPU>&);                                             \
    template void scatterRows(                                                         \
        const DenseMatrix<Scalar, Device::GPU>&, const DenseMatrix<Index, Device::GPU>&,\
        DenseMatrix<Scalar, Device::GPU>&, IndexingWorkspace&, cudaStream_t);           \
    template void scatterRowsAsync(                                                    \
        const DenseMatrix<Scalar, Device::GPU>&, const DenseMatrix<Index, Device::GPU>&,\
        DenseMatrix<Scalar, Device::GPU>&, IndexingWorkspace&, cudaStream_t);           \
    template CompactRowsResult<Scalar, Device::GPU> compactRows(                       \
        const DenseMatrix<Scalar, Device::GPU>&,                                       \
        const DenseMatrix<std::uint8_t, Device::GPU>&);                                \
    template CompactRowsResult<Scalar, Device::GPU> compactRows(                       \
        const DenseMatrix<Scalar, Device::GPU>&,                                       \
        const DenseMatrix<std::uint8_t, Device::GPU>&, IndexingWorkspace&,             \
        cudaStream_t);                                                                 \
    template void compactRows(                                                         \
        const DenseMatrix<Scalar, Device::GPU>&,                                       \
        const DenseMatrix<std::uint8_t, Device::GPU>&,                                 \
        DenseMatrix<Scalar, Device::GPU>&, DenseMatrix<Index, Device::GPU>&,           \
        DenseMatrix<Index, Device::GPU>&, IndexingWorkspace&, cudaStream_t);            \
    template void compactRowsAsync(                                                    \
        const DenseMatrix<Scalar, Device::GPU>&,                                       \
        const DenseMatrix<std::uint8_t, Device::GPU>&,                                 \
        DenseMatrix<Scalar, Device::GPU>&, DenseMatrix<Index, Device::GPU>&,           \
        DenseMatrix<Index, Device::GPU>&, IndexingWorkspace&, cudaStream_t)

#ifdef PLAMATRIX_USE_FLOAT
PLAMATRIX_INSTANTIATE_GPU_INDEXING(float);
#endif
#ifdef PLAMATRIX_USE_DOUBLE
PLAMATRIX_INSTANTIATE_GPU_INDEXING(double);
#endif

#undef PLAMATRIX_INSTANTIATE_GPU_INDEXING

} // namespace plamatrix
