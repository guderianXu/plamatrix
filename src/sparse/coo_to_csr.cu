#include <algorithm>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>

#include "plamatrix/core/error_check.h"
#include "plamatrix/sparse/sparse_ops.h"

namespace plamatrix
{
namespace
{

constexpr int kBlockSize = 256;
constexpr std::uint64_t kInvalidKey = std::numeric_limits<std::uint64_t>::max();
constexpr Index kIndexMax = INT64_C(0x7fffffffffffffff);

struct CooStatus
{
    Index invalidSource;
    Index invalidValue;
    Index mismatchActual;
    Index mismatchExpected;
    Index actualNnz;
};

struct CooSlices
{
    CooStatus* status = nullptr;
    std::uint64_t* keysIn = nullptr;
    std::uint64_t* keysOut = nullptr;
    Index* orderIn = nullptr;
    Index* orderOut = nullptr;
    Index* flags = nullptr;
    Index* positions = nullptr;
    Index* rowCounts = nullptr;
    void* sortTemporary = nullptr;
    void* flagScanTemporary = nullptr;
    void* rowScanTemporary = nullptr;
    std::size_t sortTemporaryBytes = 0;
    std::size_t flagScanTemporaryBytes = 0;
    std::size_t rowScanTemporaryBytes = 0;
};

std::size_t checkedAppend(std::size_t offset, std::size_t count, std::size_t element_size)
{
    if (count != 0 && element_size > (std::numeric_limits<std::size_t>::max() - offset) / count)
    {
        throw std::overflow_error("cooToCsr workspace size overflow");
    }
    return offset + count * element_size;
}

std::size_t alignedOffset(std::size_t offset, std::size_t alignment)
{
    const std::size_t remainder = offset % alignment;
    return remainder == 0 ? offset : checkedAppend(offset, alignment - remainder, 1);
}

unsigned int checkedGrid(Index count)
{
    if (count <= 0)
    {
        return 0;
    }
    const Index blocks = (count + kBlockSize - 1) / kBlockSize;
    if (blocks > static_cast<Index>(std::numeric_limits<unsigned int>::max()))
    {
        throw std::overflow_error("cooToCsr launch grid exceeds CUDA limits");
    }
    return static_cast<unsigned int>(blocks);
}

void checkMetadata(Index rows,
                   Index cols,
                   const DenseMatrix<Index, Device::GPU>& row_indices,
                   const DenseMatrix<Index, Device::GPU>& col_indices,
                   Index value_rows,
                   Index value_cols)
{
    if (rows < 0 || cols < 0)
    {
        throw std::invalid_argument("cooToCsr GPU dimensions must be non-negative");
    }
    if (row_indices.cols() != 1 || col_indices.cols() != 1 || value_cols != 1)
    {
        throw std::invalid_argument("cooToCsr GPU triplets must be column vectors");
    }
    if (row_indices.rows() != col_indices.rows() || row_indices.rows() != value_rows)
    {
        throw std::invalid_argument("cooToCsr GPU triplet lengths must match");
    }
    const Index cub_limit = static_cast<Index>(std::numeric_limits<int>::max());
    if (row_indices.rows() > cub_limit || rows >= cub_limit
        || cols > static_cast<Index>(std::numeric_limits<std::uint32_t>::max()))
    {
        throw std::overflow_error(
            "cooToCsr GPU dimensions exceed the supported packed-key/CUB range");
    }
}

template <typename Matrix>
void checkStorageStream(const char* name, const Matrix& matrix, cudaStream_t stream)
{
    if (matrix.isAsyncAllocation() && matrix.asyncAllocationStream() != stream)
    {
        throw std::logic_error(
            std::string(name) + " must use the stream that owns its async allocation");
    }
}

__global__ void initializeStatusKernel(CooStatus* status)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        status->invalidSource = kIndexMax;
        status->invalidValue = kIndexMax;
        status->mismatchActual = kIndexMax;
        status->mismatchExpected = kIndexMax;
        status->actualNnz = 0;
    }
}

template <typename Scalar>
__global__ void packKeysKernel(const Index* rows,
                               const Index* cols,
                               const Scalar* values,
                               Index count,
                               Index row_count,
                               Index col_count,
                               std::uint64_t* keys,
                               Index* order,
                               CooStatus* status)
{
    const Index offset = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (offset >= count)
    {
        return;
    }
    order[offset] = offset;
    const Index row = rows[offset];
    const Index col = cols[offset];
    if (!isfinite(static_cast<double>(values[offset])))
    {
        atomicMin(reinterpret_cast<unsigned long long*>(&status->invalidValue),
                  static_cast<unsigned long long>(offset));
    }
    if (row < 0 || row >= row_count || col < 0 || col >= col_count)
    {
        keys[offset] = kInvalidKey;
        atomicMin(reinterpret_cast<unsigned long long*>(&status->invalidSource),
                  static_cast<unsigned long long>(offset));
        return;
    }
    keys[offset] = (static_cast<std::uint64_t>(row) << 32)
        | static_cast<std::uint32_t>(col);
}

__global__ void markRunsKernel(
    const std::uint64_t* keys, Index count, Index* flags)
{
    const Index offset = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (offset < count)
    {
        const std::uint64_t key = keys[offset];
        flags[offset] = key != kInvalidKey && (offset == 0 || key != keys[offset - 1]) ? 1 : 0;
    }
}

__global__ void finishStatusKernel(const Index* flags,
                                   const Index* positions,
                                   Index count,
                                   Index expected_nnz,
                                   CooStatus* status)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        const Index actual = count == 0 ? 0 : positions[count - 1] + flags[count - 1];
        status->actualNnz = actual;
        if (expected_nnz >= 0 && actual != expected_nnz)
        {
            status->mismatchActual = actual;
            status->mismatchExpected = expected_nnz;
        }
    }
}

template <typename Scalar>
__global__ void reduceRunsKernel(const std::uint64_t* keys,
                                 const Index* order,
                                 const Index* flags,
                                 const Index* positions,
                                 Index count,
                                 const Scalar* input_values,
                                 Index output_capacity,
                                 Index* output_columns,
                                 Scalar* output_values,
                                 Index* row_counts,
                                 CooStatus* status)
{
    const Index offset = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (offset >= count || flags[offset] == 0)
    {
        return;
    }
    const std::uint64_t key = keys[offset];
    Scalar sum = Scalar{0};
    for (Index current = offset; current < count && keys[current] == key; ++current)
    {
        sum += input_values[order[current]];
        if (!isfinite(static_cast<double>(sum)))
        {
            atomicMin(reinterpret_cast<unsigned long long*>(&status->invalidValue),
                      static_cast<unsigned long long>(order[current]));
        }
    }
    const Index destination = positions[offset];
    if (destination < output_capacity)
    {
        output_columns[destination] = static_cast<Index>(key & 0xffffffffULL);
        output_values[destination] = sum;
    }
    const Index row = static_cast<Index>(key >> 32);
    atomicAdd(reinterpret_cast<unsigned long long*>(&row_counts[row]), 1ULL);
}

} // anonymous namespace

struct SparseCooWorkspaceAccess
{
    static void reserve(SparseOpsWorkspace& workspace, std::size_t bytes, cudaStream_t stream)
    {
        if (workspace._hasStatusBatch)
        {
            throw std::logic_error(
                "SparseOpsWorkspace status must be checked before another COO conversion");
        }
        if (workspace._hasReuseStream && workspace._reuseStream != stream)
        {
            throw std::logic_error(
                "SparseOpsWorkspace cannot be reused on a different stream; close it first");
        }
        workspace._reuseStream = stream;
        workspace._hasReuseStream = true;
        if (bytes <= workspace._capacityBytes)
        {
            return;
        }
        if (workspace._buffer != nullptr
            && (!workspace._streamOrderedAllocation || workspace._allocationStream != stream))
        {
            throw std::logic_error("SparseOpsWorkspace temporary buffer has incompatible ownership");
        }
        void* replacement = nullptr;
        PLAMATRIX_CHECK_CUDA(cudaMallocAsync(&replacement, bytes, stream));
        try
        {
            if (workspace._buffer != nullptr)
            {
                PLAMATRIX_CHECK_CUDA(cudaFreeAsync(workspace._buffer, stream));
            }
        }
        catch (...)
        {
            static_cast<void>(cudaFreeAsync(replacement, stream));
            throw;
        }
        workspace._buffer = replacement;
        workspace._capacityBytes = bytes;
        workspace._allocationStream = stream;
        workspace._streamOrderedAllocation = true;
    }

    template <typename Scalar>
    static void finalizeOutput(void* output, bool trusted) noexcept
    {
        detail::CSRMatrixAccess::completeAsyncWrite(
            *static_cast<CSRMatrix<Scalar, Device::GPU>*>(output), trusted);
    }

    template <typename Scalar>
    static void beginStatus(
        SparseOpsWorkspace& workspace,
        CSRMatrix<Scalar, Device::GPU>* output) noexcept
    {
        workspace._hasStatusBatch = true;
        workspace._statusOutput = output;
        workspace._statusFinalize = output == nullptr ? nullptr : &finalizeOutput<Scalar>;
    }

    static void* data(SparseOpsWorkspace& workspace) noexcept
    {
        return workspace._buffer;
    }

    static CooStatus consumeStatus(SparseOpsWorkspace& workspace, const char* operation)
    {
        if (!workspace._hasReuseStream)
        {
            return {std::numeric_limits<Index>::max(), std::numeric_limits<Index>::max(),
                    std::numeric_limits<Index>::max(), std::numeric_limits<Index>::max(), 0};
        }
        const cudaError_t query = cudaStreamQuery(workspace._reuseStream);
        if (query == cudaErrorNotReady)
        {
            throw std::logic_error(
                "SparseOpsWorkspace::checkStatus requires stream synchronization first");
        }
        PLAMATRIX_CHECK_CUDA(query);
        if (!workspace._hasStatusBatch)
        {
            return {std::numeric_limits<Index>::max(), std::numeric_limits<Index>::max(),
                    std::numeric_limits<Index>::max(), std::numeric_limits<Index>::max(), 0};
        }
        CooStatus status{};
        PLAMATRIX_CHECK_CUDA(cudaMemcpyAsync(
            &status, workspace._buffer, sizeof(status), cudaMemcpyDeviceToHost,
            workspace._reuseStream));
        PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(workspace._reuseStream));
        workspace._hasStatusBatch = false;
        const auto finalize_output = [&workspace](bool trusted) noexcept {
            if (workspace._statusOutput != nullptr && workspace._statusFinalize != nullptr)
            {
                workspace._statusFinalize(workspace._statusOutput, trusted);
            }
            workspace._statusOutput = nullptr;
            workspace._statusFinalize = nullptr;
        };
        const char* name = operation == nullptr ? "cooToCsrAsync" : operation;
        try
        {
            if (status.invalidSource != std::numeric_limits<Index>::max())
            {
                std::ostringstream message;
                message << name << ": coordinate out of range at source offset "
                        << status.invalidSource;
                throw std::out_of_range(message.str());
            }
            if (status.invalidValue != std::numeric_limits<Index>::max())
            {
                std::ostringstream message;
                message << name << ": non-finite value at source offset "
                        << status.invalidValue;
                throw std::invalid_argument(message.str());
            }
            if (status.mismatchActual != std::numeric_limits<Index>::max())
            {
                std::ostringstream message;
                message << name << ": output nnz is " << status.mismatchExpected
                        << ", but duplicate-combined nnz is " << status.mismatchActual;
                throw std::runtime_error(message.str());
            }
        }
        catch (...)
        {
            finalize_output(false);
            throw;
        }
        finalize_output(true);
        return status;
    }
};

namespace
{

CooSlices reserveCooSlices(Index count,
                           Index rows,
                           SparseOpsWorkspace& workspace,
                           cudaStream_t stream)
{
    const int item_count = static_cast<int>(count);
    const int row_items = static_cast<int>(rows + 1);
    std::size_t sort_bytes = 0;
    std::size_t flag_scan_bytes = 0;
    std::size_t row_scan_bytes = 0;
    cub::DoubleBuffer<std::uint64_t> query_keys(nullptr, nullptr);
    cub::DoubleBuffer<Index> query_order(nullptr, nullptr);
    PLAMATRIX_CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
        nullptr, sort_bytes, query_keys, query_order, item_count, 0, 64, stream));
    PLAMATRIX_CHECK_CUDA(cub::DeviceScan::ExclusiveSum(
        nullptr, flag_scan_bytes, static_cast<Index*>(nullptr),
        static_cast<Index*>(nullptr), item_count, stream));
    PLAMATRIX_CHECK_CUDA(cub::DeviceScan::ExclusiveSum(
        nullptr, row_scan_bytes, static_cast<Index*>(nullptr),
        static_cast<Index*>(nullptr), row_items, stream));
    std::size_t offset = sizeof(CooStatus);
    const auto append = [&offset](std::size_t count_value, std::size_t element_size,
                                  std::size_t alignment) {
        offset = alignedOffset(offset, alignment);
        const std::size_t result = offset;
        offset = checkedAppend(offset, count_value, element_size);
        return result;
    };
    const std::size_t keys_in = append(static_cast<std::size_t>(count), sizeof(std::uint64_t),
                                       alignof(std::uint64_t));
    const std::size_t keys_out = append(static_cast<std::size_t>(count), sizeof(std::uint64_t),
                                        alignof(std::uint64_t));
    const std::size_t order_in = append(static_cast<std::size_t>(count), sizeof(Index), alignof(Index));
    const std::size_t order_out = append(static_cast<std::size_t>(count), sizeof(Index), alignof(Index));
    const std::size_t flags = append(static_cast<std::size_t>(count), sizeof(Index), alignof(Index));
    const std::size_t positions = append(static_cast<std::size_t>(count), sizeof(Index), alignof(Index));
    const std::size_t row_counts = append(static_cast<std::size_t>(rows) + 1, sizeof(Index), alignof(Index));
    const std::size_t sort_temporary = append(sort_bytes, 1, 256);
    const std::size_t flag_scan_temporary = append(flag_scan_bytes, 1, 256);
    const std::size_t row_scan_temporary = append(row_scan_bytes, 1, 256);
    SparseCooWorkspaceAccess::reserve(workspace, offset, stream);

    auto* base = static_cast<unsigned char*>(SparseCooWorkspaceAccess::data(workspace));
    return {reinterpret_cast<CooStatus*>(base),
            reinterpret_cast<std::uint64_t*>(base + keys_in),
            reinterpret_cast<std::uint64_t*>(base + keys_out),
            reinterpret_cast<Index*>(base + order_in),
            reinterpret_cast<Index*>(base + order_out),
            reinterpret_cast<Index*>(base + flags),
            reinterpret_cast<Index*>(base + positions),
            reinterpret_cast<Index*>(base + row_counts),
            base + sort_temporary,
            base + flag_scan_temporary,
            base + row_scan_temporary,
            sort_bytes,
            flag_scan_bytes,
            row_scan_bytes};
}

template <typename Scalar>
void launchCooConversion(Index rows,
                         Index cols,
                         const DenseMatrix<Index, Device::GPU>& row_indices,
                         const DenseMatrix<Index, Device::GPU>& col_indices,
                         const DenseMatrix<Scalar, Device::GPU>& values,
                         CSRMatrix<Scalar, Device::GPU>* output,
                         SparseOpsWorkspace& workspace,
                         cudaStream_t stream)
{
    checkMetadata(rows, cols, row_indices, col_indices, values.rows(), values.cols());
    checkStorageStream("cooToCsrAsync rows", row_indices, stream);
    checkStorageStream("cooToCsrAsync columns", col_indices, stream);
    checkStorageStream("cooToCsrAsync values", values, stream);
    if (output != nullptr)
    {
        checkStorageStream("cooToCsrAsync output", *output, stream);
        if (output->rows() != rows || output->cols() != cols)
        {
            throw std::runtime_error("cooToCsrAsync output dimensions do not match");
        }
        if (values.data() != nullptr
            && values.data() == detail::CSRMatrixAccess::values(*output))
        {
            throw std::invalid_argument("cooToCsrAsync values and output must not alias");
        }
    }
    const Index count = row_indices.rows();
    if (count == 0)
    {
        SparseCooWorkspaceAccess::reserve(workspace, sizeof(CooStatus), stream);
        if (output != nullptr)
        {
            if (output->nnz() != 0)
            {
                throw std::runtime_error("cooToCsrAsync empty input requires output nnz=0");
            }
            detail::CSRMatrixAccess::beginAsyncWrite(
                *output, stream, "cooToCsrAsync");
            initializeStatusKernel<<<1, 1, 0, stream>>>(
                static_cast<CooStatus*>(SparseCooWorkspaceAccess::data(workspace)));
            SparseCooWorkspaceAccess::beginStatus(workspace, output);
            PLAMATRIX_CHECK_CUDA(cudaMemsetAsync(
                detail::CSRMatrixAccess::rowOffsets(*output), 0,
                static_cast<std::size_t>(rows + 1) * sizeof(Index), stream));
        }
        return;
    }

    CooSlices slices = reserveCooSlices(count, rows, workspace, stream);
    if (output != nullptr)
    {
        detail::CSRMatrixAccess::beginAsyncWrite(
            *output, stream, "cooToCsrAsync");
    }
    initializeStatusKernel<<<1, 1, 0, stream>>>(slices.status);
    SparseCooWorkspaceAccess::beginStatus(workspace, output);
    const unsigned int grid = checkedGrid(count);
    packKeysKernel<<<grid, kBlockSize, 0, stream>>>(
        row_indices.data(), col_indices.data(), values.data(), count, rows, cols,
        slices.keysIn, slices.orderIn, slices.status);
    PLAMATRIX_CHECK_CUDA(cudaGetLastError());
    std::size_t operation_temporary_bytes = slices.sortTemporaryBytes;
    cub::DoubleBuffer<std::uint64_t> sorted_keys(slices.keysIn, slices.keysOut);
    cub::DoubleBuffer<Index> sorted_order(slices.orderIn, slices.orderOut);
    PLAMATRIX_CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
        slices.sortTemporary, operation_temporary_bytes, sorted_keys, sorted_order,
        static_cast<int>(count), 0, 64, stream));
    const std::uint64_t* sorted_key_data = sorted_keys.Current();
    const Index* sorted_order_data = sorted_order.Current();
    markRunsKernel<<<grid, kBlockSize, 0, stream>>>(sorted_key_data, count, slices.flags);
    PLAMATRIX_CHECK_CUDA(cudaGetLastError());
    operation_temporary_bytes = slices.flagScanTemporaryBytes;
    PLAMATRIX_CHECK_CUDA(cub::DeviceScan::ExclusiveSum(
        slices.flagScanTemporary, operation_temporary_bytes, slices.flags, slices.positions,
        static_cast<int>(count), stream));
    const Index expected_nnz = output == nullptr ? -1 : output->nnz();
    finishStatusKernel<<<1, 1, 0, stream>>>(
        slices.flags, slices.positions, count, expected_nnz, slices.status);
    PLAMATRIX_CHECK_CUDA(cudaGetLastError());
    if (output == nullptr)
    {
        return;
    }

    PLAMATRIX_CHECK_CUDA(cudaMemsetAsync(
        slices.rowCounts, 0, static_cast<std::size_t>(rows + 1) * sizeof(Index), stream));
    reduceRunsKernel<<<grid, kBlockSize, 0, stream>>>(
        sorted_key_data, sorted_order_data, slices.flags, slices.positions, count, values.data(),
        output->nnz(), detail::CSRMatrixAccess::colIndices(*output),
        detail::CSRMatrixAccess::values(*output), slices.rowCounts, slices.status);
    PLAMATRIX_CHECK_CUDA(cudaGetLastError());
    operation_temporary_bytes = slices.rowScanTemporaryBytes;
    PLAMATRIX_CHECK_CUDA(cub::DeviceScan::ExclusiveSum(
        slices.rowScanTemporary, operation_temporary_bytes, slices.rowCounts,
        detail::CSRMatrixAccess::rowOffsets(*output),
        static_cast<int>(rows + 1), stream));
}

} // anonymous namespace

void SparseOpsWorkspace::checkStatus(const char* operation)
{
    static_cast<void>(SparseCooWorkspaceAccess::consumeStatus(*this, operation));
}

template <typename Scalar>
CSRMatrix<Scalar, Device::GPU> cooToCsr(
    Index rows,
    Index cols,
    const DenseMatrix<Index, Device::GPU>& row_indices,
    const DenseMatrix<Index, Device::GPU>& col_indices,
    const DenseMatrix<Scalar, Device::GPU>& values,
    SparseOpsWorkspace& workspace)
{
    launchCooConversion(
        rows, cols, row_indices, col_indices, values,
        static_cast<CSRMatrix<Scalar, Device::GPU>*>(nullptr), workspace, nullptr);
    if (row_indices.rows() == 0)
    {
        return CSRMatrix<Scalar, Device::GPU>(rows, cols, 0);
    }
    PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(nullptr));
    const CooStatus status = SparseCooWorkspaceAccess::consumeStatus(workspace, "cooToCsr");
    CSRMatrix<Scalar, Device::GPU> output(rows, cols, status.actualNnz);
    launchCooConversion(
        rows, cols, row_indices, col_indices, values, &output, workspace, nullptr);
    PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(nullptr));
    workspace.checkStatus("cooToCsr");
    detail::CSRMatrixAccess::completeAsyncWrite(output, true);
    return output;
}

template <typename Scalar>
void cooToCsrAsync(
    Index rows,
    Index cols,
    const DenseMatrix<Index, Device::GPU>& row_indices,
    const DenseMatrix<Index, Device::GPU>& col_indices,
    const DenseMatrix<Scalar, Device::GPU>& values,
    CSRMatrix<Scalar, Device::GPU>& output,
    SparseOpsWorkspace& workspace,
    cudaStream_t stream)
{
    launchCooConversion(
        rows, cols, row_indices, col_indices, values, &output, workspace, stream);
}

#ifdef PLAMATRIX_USE_FLOAT
template CSRMatrix<float, Device::GPU> cooToCsr<float>(
    Index, Index, const DenseMatrix<Index, Device::GPU>&,
    const DenseMatrix<Index, Device::GPU>&, const DenseMatrix<float, Device::GPU>&,
    SparseOpsWorkspace&);
template void cooToCsrAsync<float>(
    Index, Index, const DenseMatrix<Index, Device::GPU>&,
    const DenseMatrix<Index, Device::GPU>&, const DenseMatrix<float, Device::GPU>&,
    CSRMatrix<float, Device::GPU>&, SparseOpsWorkspace&, cudaStream_t);
#endif

#ifdef PLAMATRIX_USE_DOUBLE
template CSRMatrix<double, Device::GPU> cooToCsr<double>(
    Index, Index, const DenseMatrix<Index, Device::GPU>&,
    const DenseMatrix<Index, Device::GPU>&, const DenseMatrix<double, Device::GPU>&,
    SparseOpsWorkspace&);
template void cooToCsrAsync<double>(
    Index, Index, const DenseMatrix<Index, Device::GPU>&,
    const DenseMatrix<Index, Device::GPU>&, const DenseMatrix<double, Device::GPU>&,
    CSRMatrix<double, Device::GPU>&, SparseOpsWorkspace&, cudaStream_t);
#endif

} // namespace plamatrix
