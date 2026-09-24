#include <cstddef>
#include <stdexcept>

#include "indexing_detail.h"

namespace plamatrix::internal
{
    namespace
    {

        void synchronizeAndCheck(cudaStream_t stream, IndexingWorkspace& workspace, const char* operation)
        {
            PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
            workspace.checkStatus(operation);
        }

        void requireIndexVector(const char* operation, const DenseStorage<Index, Device::GPU>& indices)
        {
            if (indices.cols() != 1)
            {
                throw std::invalid_argument(std::string(operation) + ": indices must have shape K x 1");
            }
        }

        template <typename Scalar>
        void requireCompactMask(const DenseStorage<Scalar, Device::GPU>& input,
                                const DenseStorage<std::uint8_t, Device::GPU>& keep_mask)
        {
            if (keep_mask.rows() != input.rows() || keep_mask.cols() != 1)
            {
                throw std::invalid_argument("compactRows: keep_mask must have shape input.rows() x 1");
            }
        }

    } // namespace

    DenseStorage<Index, Device::GPU> exclusiveScan(const DenseStorage<Index, Device::GPU>& counts)
    {
        IndexingWorkspace workspace;
        return exclusiveScan(counts, workspace, nullptr);
    }

    DenseStorage<Index, Device::GPU>
    exclusiveScan(const DenseStorage<Index, Device::GPU>& counts, IndexingWorkspace& workspace, cudaStream_t stream)
    {
        DenseStorage<Index, Device::GPU> output(counts.rows(), counts.cols());
        indexing_detail::launchExclusiveScan(counts, output, workspace, stream);
        synchronizeAndCheck(stream, workspace, "exclusiveScan");
        return output;
    }

    void exclusiveScan(const DenseStorage<Index, Device::GPU>& counts,
                       DenseStorage<Index, Device::GPU>& output,
                       IndexingWorkspace& workspace,
                       cudaStream_t stream)
    {
        indexing_detail::launchExclusiveScan(counts, output, workspace, stream);
        synchronizeAndCheck(stream, workspace, "exclusiveScan");
    }

    DenseStorage<Index, Device::GPU>
    exclusiveScanAsync(const DenseStorage<Index, Device::GPU>& counts, IndexingWorkspace& workspace, cudaStream_t stream)
    {
        auto output = DenseStorage<Index, Device::GPU>::uninitializedAsync(counts.rows(), counts.cols(), stream);
        indexing_detail::launchExclusiveScan(counts, output, workspace, stream);
        return output;
    }

    void exclusiveScanAsync(const DenseStorage<Index, Device::GPU>& counts,
                            DenseStorage<Index, Device::GPU>& output,
                            IndexingWorkspace& workspace,
                            cudaStream_t stream)
    {
        indexing_detail::launchExclusiveScan(counts, output, workspace, stream);
    }

    void exclusiveScan(ConstMatrixView<Index, Device::GPU> counts,
                       MatrixView<Index, Device::GPU> output,
                       IndexingWorkspace& workspace,
                       cudaStream_t stream)
    {
        indexing_detail::launchExclusiveScan(counts, output, workspace, stream);
        synchronizeAndCheck(stream, workspace, "exclusiveScan");
    }

    void exclusiveScanAsync(ConstMatrixView<Index, Device::GPU> counts,
                            MatrixView<Index, Device::GPU> output,
                            IndexingWorkspace& workspace,
                            cudaStream_t stream)
    {
        indexing_detail::launchExclusiveScan(counts, output, workspace, stream);
    }

    template <typename Scalar>
    DenseStorage<Scalar, Device::GPU> gatherRows(const DenseStorage<Scalar, Device::GPU>& input,
                                                const DenseStorage<Index, Device::GPU>& indices)
    {
        IndexingWorkspace workspace;
        return gatherRows(input, indices, workspace, nullptr);
    }

    template <typename Scalar>
    DenseStorage<Scalar, Device::GPU> gatherRows(const DenseStorage<Scalar, Device::GPU>& input,
                                                const DenseStorage<Index, Device::GPU>& indices,
                                                IndexingWorkspace& workspace,
                                                cudaStream_t stream)
    {
        requireIndexVector("gatherRows", indices);
        DenseStorage<Scalar, Device::GPU> output(indices.rows(), input.cols());
        indexing_detail::launchGatherRows(input.view(), indices.view(), output.view(), workspace, stream);
        synchronizeAndCheck(stream, workspace, "gatherRows");
        return output;
    }

    template <typename Scalar>
    void gatherRows(const DenseStorage<Scalar, Device::GPU>& input,
                    const DenseStorage<Index, Device::GPU>& indices,
                    DenseStorage<Scalar, Device::GPU>& output,
                    IndexingWorkspace& workspace,
                    cudaStream_t stream)
    {
        indexing_detail::launchGatherRows(input.view(), indices.view(), output.view(), workspace, stream);
        synchronizeAndCheck(stream, workspace, "gatherRows");
    }

    template <typename Scalar>
    DenseStorage<Scalar, Device::GPU> gatherRowsAsync(const DenseStorage<Scalar, Device::GPU>& input,
                                                     const DenseStorage<Index, Device::GPU>& indices,
                                                     IndexingWorkspace& workspace,
                                                     cudaStream_t stream)
    {
        requireIndexVector("gatherRowsAsync", indices);
        auto output = DenseStorage<Scalar, Device::GPU>::uninitializedAsync(indices.rows(), input.cols(), stream);
        indexing_detail::launchGatherRows(input.view(), indices.view(), output.view(), workspace, stream);
        return output;
    }

    template <typename Scalar>
    void gatherRowsAsync(const DenseStorage<Scalar, Device::GPU>& input,
                         const DenseStorage<Index, Device::GPU>& indices,
                         DenseStorage<Scalar, Device::GPU>& output,
                         IndexingWorkspace& workspace,
                         cudaStream_t stream)
    {
        indexing_detail::launchGatherRows(input.view(), indices.view(), output.view(), workspace, stream);
    }

    template <typename Scalar>
    void scatterRows(const DenseStorage<Scalar, Device::GPU>& values,
                     const DenseStorage<Index, Device::GPU>& indices,
                     DenseStorage<Scalar, Device::GPU>& output)
    {
        IndexingWorkspace workspace;
        scatterRows(values, indices, output, workspace, nullptr);
    }

    template <typename Scalar>
    void scatterRows(const DenseStorage<Scalar, Device::GPU>& values,
                     const DenseStorage<Index, Device::GPU>& indices,
                     DenseStorage<Scalar, Device::GPU>& output,
                     IndexingWorkspace& workspace,
                     cudaStream_t stream)
    {
        indexing_detail::launchScatterRows(values.view(), indices.view(), output.view(), workspace, stream);
        synchronizeAndCheck(stream, workspace, "scatterRows");
    }

    template <typename Scalar>
    void scatterRowsAsync(const DenseStorage<Scalar, Device::GPU>& values,
                          const DenseStorage<Index, Device::GPU>& indices,
                          DenseStorage<Scalar, Device::GPU>& output,
                          IndexingWorkspace& workspace,
                          cudaStream_t stream)
    {
        indexing_detail::launchScatterRows(values.view(), indices.view(), output.view(), workspace, stream);
    }

    template <typename Scalar>
    CompactRowsResult<Scalar, Device::GPU> compactRows(const DenseStorage<Scalar, Device::GPU>& input,
                                                       const DenseStorage<std::uint8_t, Device::GPU>& keep_mask)
    {
        IndexingWorkspace workspace;
        return compactRows(input, keep_mask, workspace, nullptr);
    }

    template <typename Scalar>
    CompactRowsResult<Scalar, Device::GPU> compactRows(const DenseStorage<Scalar, Device::GPU>& input,
                                                       const DenseStorage<std::uint8_t, Device::GPU>& keep_mask,
                                                       IndexingWorkspace& workspace,
                                                       cudaStream_t stream)
    {
        requireCompactMask(input, keep_mask);
        auto capacity_output = DenseStorage<Scalar, Device::GPU>::uninitialized(input.rows(), input.cols());
        auto capacity_sources = DenseStorage<Index, Device::GPU>::uninitialized(input.rows(), 1);
        auto selected_count = DenseStorage<Index, Device::GPU>::uninitialized(1, 1);
        indexing_detail::launchCompactRows(input.view(),
                                           keep_mask.view(),
                                           capacity_output.view(),
                                           capacity_sources.view(),
                                           selected_count.view(),
                                           workspace,
                                           stream);
        synchronizeAndCheck(stream, workspace, "compactRows");

        Index selected = 0;
        PLAMATRIX_CHECK_CUDA(cudaMemcpy(&selected, selected_count.data(), sizeof(Index), cudaMemcpyDeviceToHost));
        if (selected < 0 || selected > input.rows())
        {
            throw std::runtime_error("compactRows: CUB returned an invalid selected count");
        }
        CompactRowsResult<Scalar, Device::GPU> result{
            DenseStorage<Scalar, Device::GPU>::uninitialized(selected, input.cols()),
            DenseStorage<Index, Device::GPU>::uninitialized(selected, 1)};
        if (selected != 0)
        {
            const std::size_t selected_size = static_cast<std::size_t>(selected);
            PLAMATRIX_CHECK_CUDA(cudaMemcpyAsync(result.sourceIndices.data(),
                                                 capacity_sources.data(),
                                                 detail::checkedAllocationBytes<Index>(selected_size),
                                                 cudaMemcpyDeviceToDevice,
                                                 stream));
            if (input.cols() != 0)
            {
                const std::size_t selected_bytes = detail::checkedAllocationBytes<Scalar>(selected_size);
                const std::size_t capacity_pitch =
                    detail::checkedAllocationBytes<Scalar>(static_cast<std::size_t>(input.rows()));
                PLAMATRIX_CHECK_CUDA(cudaMemcpy2DAsync(result.values.data(),
                                                       selected_bytes,
                                                       capacity_output.data(),
                                                       capacity_pitch,
                                                       selected_bytes,
                                                       static_cast<std::size_t>(input.cols()),
                                                       cudaMemcpyDeviceToDevice,
                                                       stream));
            }
            PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
        }
        return result;
    }

    template <typename Scalar>
    void compactRows(const DenseStorage<Scalar, Device::GPU>& input,
                     const DenseStorage<std::uint8_t, Device::GPU>& keep_mask,
                     DenseStorage<Scalar, Device::GPU>& capacity_output,
                     DenseStorage<Index, Device::GPU>& capacity_source_indices,
                     DenseStorage<Index, Device::GPU>& selected_count,
                     IndexingWorkspace& workspace,
                     cudaStream_t stream)
    {
        indexing_detail::launchCompactRows(input.view(),
                                           keep_mask.view(),
                                           capacity_output.view(),
                                           capacity_source_indices.view(),
                                           selected_count.view(),
                                           workspace,
                                           stream);
        synchronizeAndCheck(stream, workspace, "compactRows");
    }

    template <typename Scalar>
    void compactRowsAsync(const DenseStorage<Scalar, Device::GPU>& input,
                          const DenseStorage<std::uint8_t, Device::GPU>& keep_mask,
                          DenseStorage<Scalar, Device::GPU>& capacity_output,
                          DenseStorage<Index, Device::GPU>& capacity_source_indices,
                          DenseStorage<Index, Device::GPU>& selected_count,
                          IndexingWorkspace& workspace,
                          cudaStream_t stream)
    {
        indexing_detail::launchCompactRows(input.view(),
                                           keep_mask.view(),
                                           capacity_output.view(),
                                           capacity_source_indices.view(),
                                           selected_count.view(),
                                           workspace,
                                           stream);
    }

    template <typename Scalar>
    void gatherRowsAsync(ConstMatrixView<Scalar, Device::GPU> input,
                         ConstMatrixView<Index, Device::GPU> indices,
                         MatrixView<Scalar, Device::GPU> output,
                         IndexingWorkspace& workspace,
                         cudaStream_t stream)
    {
        indexing_detail::launchGatherRows(input, indices, output, workspace, stream);
    }

    template <typename Scalar>
    void scatterRowsAsync(ConstMatrixView<Scalar, Device::GPU> values,
                          ConstMatrixView<Index, Device::GPU> indices,
                          MatrixView<Scalar, Device::GPU> output,
                          IndexingWorkspace& workspace,
                          cudaStream_t stream)
    {
        indexing_detail::launchScatterRows(values, indices, output, workspace, stream);
    }

    template <typename Scalar>
    void compactRowsAsync(ConstMatrixView<Scalar, Device::GPU> input,
                          ConstMatrixView<std::uint8_t, Device::GPU> keep_mask,
                          MatrixView<Scalar, Device::GPU> capacity_output,
                          MatrixView<Index, Device::GPU> capacity_source_indices,
                          MatrixView<Index, Device::GPU> selected_count,
                          IndexingWorkspace& workspace,
                          cudaStream_t stream)
    {
        indexing_detail::launchCompactRows(
            input, keep_mask, capacity_output, capacity_source_indices, selected_count, workspace, stream);
    }

#define PLAMATRIX_INSTANTIATE_GPU_INDEXING(Scalar)                                                                     \
    template DenseStorage<Scalar, Device::GPU> gatherRows(const DenseStorage<Scalar, Device::GPU>&,                      \
                                                         const DenseStorage<Index, Device::GPU>&);                      \
    template DenseStorage<Scalar, Device::GPU> gatherRows(const DenseStorage<Scalar, Device::GPU>&,                      \
                                                         const DenseStorage<Index, Device::GPU>&,                       \
                                                         IndexingWorkspace&,                                           \
                                                         cudaStream_t);                                                \
    template void gatherRows(const DenseStorage<Scalar, Device::GPU>&,                                                  \
                             const DenseStorage<Index, Device::GPU>&,                                                   \
                             DenseStorage<Scalar, Device::GPU>&,                                                        \
                             IndexingWorkspace&,                                                                       \
                             cudaStream_t);                                                                            \
    template DenseStorage<Scalar, Device::GPU> gatherRowsAsync(const DenseStorage<Scalar, Device::GPU>&,                 \
                                                              const DenseStorage<Index, Device::GPU>&,                  \
                                                              IndexingWorkspace&,                                      \
                                                              cudaStream_t);                                           \
    template void gatherRowsAsync(const DenseStorage<Scalar, Device::GPU>&,                                             \
                                  const DenseStorage<Index, Device::GPU>&,                                              \
                                  DenseStorage<Scalar, Device::GPU>&,                                                   \
                                  IndexingWorkspace&,                                                                  \
                                  cudaStream_t);                                                                       \
    template void scatterRows(const DenseStorage<Scalar, Device::GPU>&,                                                 \
                              const DenseStorage<Index, Device::GPU>&,                                                  \
                              DenseStorage<Scalar, Device::GPU>&);                                                      \
    template void scatterRows(const DenseStorage<Scalar, Device::GPU>&,                                                 \
                              const DenseStorage<Index, Device::GPU>&,                                                  \
                              DenseStorage<Scalar, Device::GPU>&,                                                       \
                              IndexingWorkspace&,                                                                      \
                              cudaStream_t);                                                                           \
    template void scatterRowsAsync(const DenseStorage<Scalar, Device::GPU>&,                                            \
                                   const DenseStorage<Index, Device::GPU>&,                                             \
                                   DenseStorage<Scalar, Device::GPU>&,                                                  \
                                   IndexingWorkspace&,                                                                 \
                                   cudaStream_t);                                                                      \
    template CompactRowsResult<Scalar, Device::GPU> compactRows(const DenseStorage<Scalar, Device::GPU>&,               \
                                                                const DenseStorage<std::uint8_t, Device::GPU>&);        \
    template CompactRowsResult<Scalar, Device::GPU> compactRows(const DenseStorage<Scalar, Device::GPU>&,               \
                                                                const DenseStorage<std::uint8_t, Device::GPU>&,         \
                                                                IndexingWorkspace&,                                    \
                                                                cudaStream_t);                                         \
    template void compactRows(const DenseStorage<Scalar, Device::GPU>&,                                                 \
                              const DenseStorage<std::uint8_t, Device::GPU>&,                                           \
                              DenseStorage<Scalar, Device::GPU>&,                                                       \
                              DenseStorage<Index, Device::GPU>&,                                                        \
                              DenseStorage<Index, Device::GPU>&,                                                        \
                              IndexingWorkspace&,                                                                      \
                              cudaStream_t);                                                                           \
    template void compactRowsAsync(const DenseStorage<Scalar, Device::GPU>&,                                            \
                                   const DenseStorage<std::uint8_t, Device::GPU>&,                                      \
                                   DenseStorage<Scalar, Device::GPU>&,                                                  \
                                   DenseStorage<Index, Device::GPU>&,                                                   \
                                   DenseStorage<Index, Device::GPU>&,                                                   \
                                   IndexingWorkspace&,                                                                 \
                                   cudaStream_t)

#ifdef PLAMATRIX_USE_FLOAT
    PLAMATRIX_INSTANTIATE_GPU_INDEXING(float);
#endif
#ifdef PLAMATRIX_USE_DOUBLE
    PLAMATRIX_INSTANTIATE_GPU_INDEXING(double);
#endif
    PLAMATRIX_INSTANTIATE_GPU_INDEXING(std::uint8_t);
    PLAMATRIX_INSTANTIATE_GPU_INDEXING(std::uint16_t);
    PLAMATRIX_INSTANTIATE_GPU_INDEXING(Index);

#undef PLAMATRIX_INSTANTIATE_GPU_INDEXING

#define PLAMATRIX_INSTANTIATE_VIEW_INDEXING(Scalar)                                                                    \
    template void gatherRowsAsync(ConstMatrixView<Scalar, Device::GPU>,                                                \
                                  ConstMatrixView<Index, Device::GPU>,                                                 \
                                  MatrixView<Scalar, Device::GPU>,                                                     \
                                  IndexingWorkspace&,                                                                  \
                                  cudaStream_t);                                                                       \
    template void scatterRowsAsync(ConstMatrixView<Scalar, Device::GPU>,                                               \
                                   ConstMatrixView<Index, Device::GPU>,                                                \
                                   MatrixView<Scalar, Device::GPU>,                                                    \
                                   IndexingWorkspace&,                                                                 \
                                   cudaStream_t);                                                                      \
    template void compactRowsAsync(ConstMatrixView<Scalar, Device::GPU>,                                               \
                                   ConstMatrixView<std::uint8_t, Device::GPU>,                                         \
                                   MatrixView<Scalar, Device::GPU>,                                                    \
                                   MatrixView<Index, Device::GPU>,                                                     \
                                   MatrixView<Index, Device::GPU>,                                                     \
                                   IndexingWorkspace&,                                                                 \
                                   cudaStream_t);

#ifdef PLAMATRIX_USE_FLOAT
    PLAMATRIX_INSTANTIATE_VIEW_INDEXING(float)
#endif
#ifdef PLAMATRIX_USE_DOUBLE
    PLAMATRIX_INSTANTIATE_VIEW_INDEXING(double)
#endif
    PLAMATRIX_INSTANTIATE_VIEW_INDEXING(std::uint8_t)
    PLAMATRIX_INSTANTIATE_VIEW_INDEXING(std::uint16_t)
    PLAMATRIX_INSTANTIATE_VIEW_INDEXING(Index)

#undef PLAMATRIX_INSTANTIATE_VIEW_INDEXING

} // namespace plamatrix::internal
