#include <cstddef>
#include <limits>
#include <stdexcept>

#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>
#include <cuda/iterator>

#include "indexing_detail.h"

namespace plamatrix
{
namespace indexing_detail
{
namespace
{

constexpr Index kIndexMax = static_cast<Index>(9223372036854775807LL);
constexpr Index kIndexMin = -kIndexMax - 1;
// Keep CUB temporary storage at the alignment provided by CUDA allocations.
constexpr std::size_t kTemporaryAlignment = 256;

struct WorkspaceSlices
{
    void* temporary;
    DeviceStatusBatch* statusBatch;
    Index* status;
    Index* owners;
};

void initializeStatusBatch(DeviceStatusBatch* status, cudaStream_t stream);

WorkspaceSlices reserveWorkspace(
    std::size_t temporary_bytes,
    Index owner_count,
    StatusCategory category,
    IndexingWorkspace& workspace,
    cudaStream_t stream)
{
    const std::size_t temporary_offset = alignedOffset(
        sizeof(DeviceStatusBatch), kTemporaryAlignment);
    const std::size_t after_temporary = checkedAppend(
        temporary_offset, 1, temporary_bytes);
    const std::size_t owner_offset = alignedOffset(after_temporary, alignof(Index));
    const std::size_t owner_bytes = checkedIndexBytes(owner_count, "indexing");
    const std::size_t total_bytes = checkedAppend(
        after_temporary, alignof(Index), owner_bytes);
    workspace.reserveBytesAsync(total_bytes, stream);

    auto* base = static_cast<unsigned char*>(workspace.data());
    auto* status_batch = reinterpret_cast<DeviceStatusBatch*>(base);
    WorkspaceSlices slices{
        base + temporary_offset,
        status_batch,
        selectStatus(status_batch, category),
        owner_count == 0 ? nullptr : reinterpret_cast<Index*>(base + owner_offset)
    };
    if (temporary_offset < sizeof(DeviceStatusBatch))
    {
        throw std::logic_error("indexing: CUB temporary storage overlaps device status");
    }
    if (owner_count != 0 && owner_offset < after_temporary)
    {
        throw std::logic_error("indexing: owner storage overlaps temporary storage");
    }
    if (IndexingWorkspaceAccess::beginStatusBatch(workspace))
    {
        initializeStatusBatch(status_batch, stream);
    }
    return slices;
}

__global__ void initializeIndexKernel(Index* values, Index count, Index value)
{
    const Index offset = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (offset < count)
    {
        values[offset] = value;
    }
}

__global__ void validateIndicesKernel(
    const Index* indices,
    Index source_count,
    Index row_count,
    Index* bad_offset)
{
    const Index source_row = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (source_row < source_count)
    {
        const Index row = indices[source_row];
        if (row < 0 || row >= row_count)
        {
            atomicMin(
                reinterpret_cast<unsigned long long*>(bad_offset),
                static_cast<unsigned long long>(source_row));
        }
    }
}

__global__ void validateScanKernel(
    const Index* counts,
    const Index* prefixes,
    Index count,
    Index* bad_offset)
{
    const Index offset = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (offset >= count)
    {
        return;
    }
    const Index prefix = prefixes[offset];
    const Index value = counts[offset];
    const bool overflow =
        (value > 0 && prefix > kIndexMax - value) ||
        (value < 0 && prefix < kIndexMin - value);
    if (overflow)
    {
        atomicMin(
            reinterpret_cast<unsigned long long*>(bad_offset),
            static_cast<unsigned long long>(offset));
    }
}

template <typename Scalar>
__global__ void gatherRowsKernel(
    const Scalar* input,
    Index input_rows,
    const Index* indices,
    Index output_rows,
    Index element_count,
    const Index* bad_offset,
    Scalar* output)
{
    const Index offset = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (offset >= element_count || *bad_offset != kIndexMax)
    {
        return;
    }
    const Index output_row = offset % output_rows;
    const Index col = offset / output_rows;
    output[offset] = input[indices[output_row] + col * input_rows];
}

__global__ void assignOwnersKernel(
    const Index* indices,
    Index source_count,
    const Index* bad_offset,
    Index* owners)
{
    const Index source_row = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (source_row >= source_count || *bad_offset != kIndexMax)
    {
        return;
    }
    atomicMin(
        reinterpret_cast<unsigned long long*>(owners + indices[source_row]),
        static_cast<unsigned long long>(source_row));
}

template <typename Scalar>
__global__ void scatterRowsKernel(
    const Scalar* values,
    const Index* indices,
    Index source_rows,
    Index element_count,
    const Index* bad_offset,
    const Index* owners,
    Index output_rows,
    Scalar* output)
{
    const Index offset = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (offset >= element_count || *bad_offset != kIndexMax)
    {
        return;
    }
    const Index source_row = offset % source_rows;
    const Index col = offset / source_rows;
    const Index destination_row = indices[source_row];
    if (owners[destination_row] == source_row)
    {
        output[destination_row + col * output_rows] = values[offset];
    }
}

template <typename Scalar>
__global__ void compactRowsKernel(
    const Scalar* input,
    Index input_rows,
    Index capacity_size,
    const Index* source_indices,
    const Index* selected_count,
    Scalar* output)
{
    const Index offset = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (offset >= capacity_size)
    {
        return;
    }
    const Index output_row = offset % input_rows;
    if (output_row < *selected_count)
    {
        const Index col = offset / input_rows;
        output[offset] = input[source_indices[output_row] + col * input_rows];
    }
}

void initializeStatusBatch(DeviceStatusBatch* status, cudaStream_t stream)
{
    initializeIndexKernel<<<1, 2, 0, stream>>>(
        reinterpret_cast<Index*>(status), 2, std::numeric_limits<Index>::max());
    PLAMATRIX_CHECK_CUDA(cudaGetLastError());
}

void launchIndexValidation(
    const DenseMatrix<Index, Device::GPU>& indices,
    Index row_count,
    Index* status,
    cudaStream_t stream,
    const char* operation)
{
    const unsigned int grid = checkedGrid(indices.rows(), operation);
    if (grid != 0)
    {
        validateIndicesKernel<<<grid, kIndexingBlockSize, 0, stream>>>(
            indices.data(), indices.rows(), row_count, status);
        PLAMATRIX_CHECK_CUDA(cudaGetLastError());
    }
}

} // namespace

void launchExclusiveScan(
    const DenseMatrix<Index, Device::GPU>& counts,
    DenseMatrix<Index, Device::GPU>& output,
    IndexingWorkspace& workspace,
    cudaStream_t stream)
{
    requireShape("exclusiveScan", output, counts.rows(), counts.cols(), "output");
    std::size_t temporary_bytes = 0;
    if (counts.size() != 0)
    {
        PLAMATRIX_CHECK_CUDA(cub::DeviceScan::ExclusiveSum(
            nullptr, temporary_bytes, counts.data(), output.data(), counts.size(), stream));
    }
    const WorkspaceSlices slices = reserveWorkspace(
        temporary_bytes, 0, StatusCategory::Overflow, workspace, stream);
    if (counts.size() == 0)
    {
        return;
    }

    PLAMATRIX_CHECK_CUDA(cub::DeviceScan::ExclusiveSum(
        slices.temporary, temporary_bytes, counts.data(), output.data(), counts.size(), stream));
    const unsigned int grid = checkedGrid(counts.size(), "exclusiveScan");
    validateScanKernel<<<grid, kIndexingBlockSize, 0, stream>>>(
        counts.data(), output.data(), counts.size(), slices.status);
    PLAMATRIX_CHECK_CUDA(cudaGetLastError());
}

template <typename Scalar>
void launchGatherRows(
    const DenseMatrix<Scalar, Device::GPU>& input,
    const DenseMatrix<Index, Device::GPU>& indices,
    DenseMatrix<Scalar, Device::GPU>& output,
    IndexingWorkspace& workspace,
    cudaStream_t stream)
{
    requireIndexVector("gatherRows", indices);
    requireShape("gatherRows", output, indices.rows(), input.cols(), "output");
    const WorkspaceSlices slices = reserveWorkspace(
        0, 0, StatusCategory::OutOfRange, workspace, stream);
    launchIndexValidation(indices, input.rows(), slices.status, stream, "gatherRows");

    const unsigned int grid = checkedGrid(output.size(), "gatherRows");
    if (grid != 0)
    {
        gatherRowsKernel<<<grid, kIndexingBlockSize, 0, stream>>>(
            input.data(), input.rows(), indices.data(), output.rows(), output.size(),
            slices.status, output.data());
        PLAMATRIX_CHECK_CUDA(cudaGetLastError());
    }
}

template <typename Scalar>
void launchScatterRows(
    const DenseMatrix<Scalar, Device::GPU>& values,
    const DenseMatrix<Index, Device::GPU>& indices,
    DenseMatrix<Scalar, Device::GPU>& output,
    IndexingWorkspace& workspace,
    cudaStream_t stream)
{
    requireIndexVector("scatterRows", indices);
    if (values.rows() != indices.rows() || values.cols() != output.cols())
    {
        throw std::invalid_argument("scatterRows: values and output shapes are incompatible");
    }
    const Index owner_count =
        values.rows() != 0 && values.cols() != 0 ? output.rows() : 0;
    const WorkspaceSlices slices = reserveWorkspace(
        0, owner_count, StatusCategory::OutOfRange, workspace, stream);
    launchIndexValidation(indices, output.rows(), slices.status, stream, "scatterRows");
    if (owner_count == 0)
    {
        return;
    }

    const unsigned int owner_grid = checkedGrid(owner_count, "scatterRows owners");
    initializeIndexKernel<<<owner_grid, kIndexingBlockSize, 0, stream>>>(
        slices.owners, owner_count, std::numeric_limits<Index>::max());
    PLAMATRIX_CHECK_CUDA(cudaGetLastError());
    const unsigned int source_grid = checkedGrid(values.rows(), "scatterRows sources");
    assignOwnersKernel<<<source_grid, kIndexingBlockSize, 0, stream>>>(
        indices.data(), values.rows(), slices.status, slices.owners);
    PLAMATRIX_CHECK_CUDA(cudaGetLastError());
    const unsigned int value_grid = checkedGrid(values.size(), "scatterRows values");
    scatterRowsKernel<<<value_grid, kIndexingBlockSize, 0, stream>>>(
        values.data(), indices.data(), values.rows(), values.size(), slices.status,
        slices.owners, output.rows(), output.data());
    PLAMATRIX_CHECK_CUDA(cudaGetLastError());
}

template <typename Scalar>
void launchCompactRows(
    const DenseMatrix<Scalar, Device::GPU>& input,
    const DenseMatrix<std::uint8_t, Device::GPU>& keep_mask,
    DenseMatrix<Scalar, Device::GPU>& capacity_output,
    DenseMatrix<Index, Device::GPU>& capacity_source_indices,
    DenseMatrix<Index, Device::GPU>& selected_count,
    IndexingWorkspace& workspace,
    cudaStream_t stream)
{
    requireShape("compactRows", keep_mask, input.rows(), 1, "keep_mask");
    requireShape("compactRows", capacity_output, input.rows(), input.cols(), "capacity_output");
    requireShape(
        "compactRows", capacity_source_indices, input.rows(), 1, "capacity_source_indices");
    requireShape("compactRows", selected_count, 1, 1, "selected_count");

    std::size_t temporary_bytes = 0;
    const auto source_rows = cuda::counting_iterator<Index>(0);
    if (input.rows() != 0)
    {
        PLAMATRIX_CHECK_CUDA(cub::DeviceSelect::Flagged(
            nullptr, temporary_bytes, source_rows, keep_mask.data(),
            capacity_source_indices.data(), selected_count.data(), input.rows(), stream));
    }
    const WorkspaceSlices slices = reserveWorkspace(
        temporary_bytes, 0, StatusCategory::None, workspace, stream);
    PLAMATRIX_CHECK_CUDA(cudaMemsetAsync(
        selected_count.data(), 0, sizeof(Index), stream));
    if (input.rows() == 0)
    {
        return;
    }

    PLAMATRIX_CHECK_CUDA(cub::DeviceSelect::Flagged(
        slices.temporary, temporary_bytes, source_rows, keep_mask.data(),
        capacity_source_indices.data(), selected_count.data(), input.rows(), stream));
    const unsigned int grid = checkedGrid(capacity_output.size(), "compactRows");
    if (grid != 0)
    {
        compactRowsKernel<<<grid, kIndexingBlockSize, 0, stream>>>(
            input.data(), input.rows(), capacity_output.size(), capacity_source_indices.data(),
            selected_count.data(), capacity_output.data());
        PLAMATRIX_CHECK_CUDA(cudaGetLastError());
    }
}

#define PLAMATRIX_INSTANTIATE_INDEXING_DISPATCH(Scalar)                                \
    template void launchGatherRows<Scalar>(                                            \
        const DenseMatrix<Scalar, Device::GPU>&, const DenseMatrix<Index, Device::GPU>&,\
        DenseMatrix<Scalar, Device::GPU>&, IndexingWorkspace&, cudaStream_t);           \
    template void launchScatterRows<Scalar>(                                           \
        const DenseMatrix<Scalar, Device::GPU>&, const DenseMatrix<Index, Device::GPU>&,\
        DenseMatrix<Scalar, Device::GPU>&, IndexingWorkspace&, cudaStream_t);           \
    template void launchCompactRows<Scalar>(                                           \
        const DenseMatrix<Scalar, Device::GPU>&,                                       \
        const DenseMatrix<std::uint8_t, Device::GPU>&,                                 \
        DenseMatrix<Scalar, Device::GPU>&, DenseMatrix<Index, Device::GPU>&,           \
        DenseMatrix<Index, Device::GPU>&, IndexingWorkspace&, cudaStream_t)

#ifdef PLAMATRIX_USE_FLOAT
PLAMATRIX_INSTANTIATE_INDEXING_DISPATCH(float);
#endif
#ifdef PLAMATRIX_USE_DOUBLE
PLAMATRIX_INSTANTIATE_INDEXING_DISPATCH(double);
#endif

#undef PLAMATRIX_INSTANTIATE_INDEXING_DISPATCH

} // namespace indexing_detail
} // namespace plamatrix
