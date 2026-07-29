#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "indexing_detail.h"

namespace plamatrix
{
namespace
{

const char* kCloseBeforeSyncReserve =
    "IndexingWorkspace::reserveBytes cannot replace a stream-ordered allocation; "
    "call closeAsyncAllocation first";

const char* kSyncGrowBeforeLaunch =
    "IndexingWorkspace::reserveBytesAsync cannot grow an ordinary allocation; "
    "synchronize the owning stream and call reserveBytes synchronously before launching";

const char* kCrossStreamReuse =
    "IndexingWorkspace cannot be reused on a different stream; synchronize the owning stream "
    "and explicitly reset or close the workspace first";

const char* kPendingStreamReset =
    "IndexingWorkspace::reserveBytes cannot reset a workspace while its owning stream has "
    "pending work; synchronize the owning stream first";

const char* kCheckStatusBeforeReset =
    "IndexingWorkspace::reserveBytes cannot reset a workspace with an unconsumed status batch; "
    "call checkStatus first";

const char* kCheckStatusBeforeClose =
    "IndexingWorkspace::closeAsyncAllocation cannot close a workspace with an unconsumed status "
    "batch; synchronize the owning stream and call checkStatus first";

} // namespace

IndexingWorkspace::~IndexingWorkspace() noexcept
{
    release();
}

IndexingWorkspace::IndexingWorkspace(IndexingWorkspace&& other) noexcept
    : _capacityBytes(other._capacityBytes)
    , _data(other._data)
    , _allocationKind(other._allocationKind)
    , _allocationStream(other._allocationStream)
    , _reuseStream(other._reuseStream)
    , _hasReuseStream(other._hasReuseStream)
    , _hasStatusBatch(other._hasStatusBatch)
{
    other._capacityBytes = 0;
    other._data = nullptr;
    other._allocationKind = AllocationKind::Normal;
    other._allocationStream = nullptr;
    other._reuseStream = nullptr;
    other._hasReuseStream = false;
    other._hasStatusBatch = false;
}

IndexingWorkspace& IndexingWorkspace::operator=(IndexingWorkspace&& other) noexcept
{
    if (this != &other)
    {
        release();
        _capacityBytes = other._capacityBytes;
        _data = other._data;
        _allocationKind = other._allocationKind;
        _allocationStream = other._allocationStream;
        _reuseStream = other._reuseStream;
        _hasReuseStream = other._hasReuseStream;
        _hasStatusBatch = other._hasStatusBatch;

        other._capacityBytes = 0;
        other._data = nullptr;
        other._allocationKind = AllocationKind::Normal;
        other._allocationStream = nullptr;
        other._reuseStream = nullptr;
        other._hasReuseStream = false;
        other._hasStatusBatch = false;
    }
    return *this;
}

void IndexingWorkspace::reserveBytes(std::size_t bytes)
{
    if (_hasReuseStream)
    {
        const cudaError_t query = cudaStreamQuery(_reuseStream);
        if (query == cudaErrorNotReady)
        {
            throw std::logic_error(kPendingStreamReset);
        }
        PLAMATRIX_CHECK_CUDA(query);
    }
    if (_hasStatusBatch)
    {
        throw std::logic_error(kCheckStatusBeforeReset);
    }
    if (_allocationKind == AllocationKind::StreamOrderedAsync && _data != nullptr)
    {
        throw std::logic_error(kCloseBeforeSyncReserve);
    }
    if (bytes <= _capacityBytes)
    {
        _reuseStream = nullptr;
        _hasReuseStream = false;
        _hasStatusBatch = false;
        return;
    }

    void* replacement = nullptr;
    PLAMATRIX_CHECK_CUDA(cudaMalloc(&replacement, bytes));
    if (_data != nullptr)
    {
        try
        {
            PLAMATRIX_CHECK_CUDA(cudaFree(_data));
        }
        catch (...)
        {
            static_cast<void>(cudaFree(replacement));
            throw;
        }
    }

    _data = replacement;
    _capacityBytes = bytes;
    _allocationKind = AllocationKind::Normal;
    _allocationStream = nullptr;
    _reuseStream = nullptr;
    _hasReuseStream = false;
    _hasStatusBatch = false;
}

void IndexingWorkspace::reserveBytesAsync(std::size_t bytes, cudaStream_t stream)
{
    if (_hasReuseStream && _reuseStream != stream)
    {
        throw std::logic_error(kCrossStreamReuse);
    }
    if (_data == nullptr)
    {
        if (bytes != 0)
        {
            void* replacement = nullptr;
            PLAMATRIX_CHECK_CUDA(cudaMallocAsync(&replacement, bytes, stream));
            _data = replacement;
            _capacityBytes = bytes;
            _allocationKind = AllocationKind::StreamOrderedAsync;
            _allocationStream = stream;
        }
        _reuseStream = stream;
        _hasReuseStream = true;
        return;
    }
    if (_allocationKind == AllocationKind::Normal)
    {
        if (bytes > _capacityBytes)
        {
            throw std::logic_error(kSyncGrowBeforeLaunch);
        }
        _reuseStream = stream;
        _hasReuseStream = true;
        return;
    }
    if (_allocationStream != stream)
    {
        throw std::logic_error(kCrossStreamReuse);
    }
    if (bytes <= _capacityBytes)
    {
        _reuseStream = stream;
        _hasReuseStream = true;
        return;
    }

    void* replacement = nullptr;
    PLAMATRIX_CHECK_CUDA(cudaMallocAsync(&replacement, bytes, stream));
    try
    {
        if (_hasStatusBatch)
        {
            PLAMATRIX_CHECK_CUDA(cudaMemcpyAsync(
                replacement, _data, sizeof(indexing_detail::DeviceStatusBatch),
                cudaMemcpyDeviceToDevice, stream));
        }
        PLAMATRIX_CHECK_CUDA(cudaFreeAsync(_data, stream));
    }
    catch (...)
    {
        static_cast<void>(cudaFreeAsync(replacement, stream));
        throw;
    }
    _data = replacement;
    _capacityBytes = bytes;
    _reuseStream = stream;
    _hasReuseStream = true;
}

void IndexingWorkspace::closeAsyncAllocation()
{
    if (_hasStatusBatch)
    {
        throw std::logic_error(kCheckStatusBeforeClose);
    }
    if (_data == nullptr)
    {
        _capacityBytes = 0;
        _allocationKind = AllocationKind::Normal;
        _allocationStream = nullptr;
        _reuseStream = nullptr;
        _hasReuseStream = false;
        _hasStatusBatch = false;
        return;
    }
    if (_allocationKind != AllocationKind::StreamOrderedAsync)
    {
        throw std::logic_error(
            "IndexingWorkspace::closeAsyncAllocation requires a stream-ordered allocation");
    }

    PLAMATRIX_CHECK_CUDA(cudaFreeAsync(_data, _allocationStream));
    _capacityBytes = 0;
    _data = nullptr;
    _allocationKind = AllocationKind::Normal;
    _allocationStream = nullptr;
    _reuseStream = nullptr;
    _hasReuseStream = false;
    _hasStatusBatch = false;
}

void IndexingWorkspace::checkStatus(const char* operation)
{
    if (!_hasReuseStream)
    {
        return;
    }
    const cudaError_t query = cudaStreamQuery(_reuseStream);
    if (query == cudaErrorNotReady)
    {
        throw std::logic_error(
            "IndexingWorkspace::checkStatus requires the owning stream to be synchronized first");
    }
    PLAMATRIX_CHECK_CUDA(query);
    if (!_hasStatusBatch || _data == nullptr)
    {
        return;
    }

    indexing_detail::DeviceStatusBatch status{
        std::numeric_limits<Index>::max(), std::numeric_limits<Index>::max()
    };
    PLAMATRIX_CHECK_CUDA(cudaMemcpy(
        &status, _data, sizeof(status), cudaMemcpyDeviceToHost));
    _hasStatusBatch = false;
    if (status.overflow == std::numeric_limits<Index>::max() &&
        status.out_of_range == std::numeric_limits<Index>::max())
    {
        return;
    }

    std::ostringstream message;
    message << (operation == nullptr ? "indexing" : operation) << ": ";
    if (status.overflow != std::numeric_limits<Index>::max())
    {
        message << "Index overflow at source offset " << status.overflow;
        throw std::overflow_error(message.str());
    }
    message << "row index out of range at source offset " << status.out_of_range;
    throw std::out_of_range(message.str());
}

void IndexingWorkspace::release() noexcept
{
    if (_data != nullptr)
    {
        if (_allocationKind == AllocationKind::StreamOrderedAsync)
        {
            static_cast<void>(cudaFreeAsync(_data, _allocationStream));
        }
        else
        {
            static_cast<void>(cudaFree(_data));
        }
    }
    _capacityBytes = 0;
    _data = nullptr;
    _allocationKind = AllocationKind::Normal;
    _allocationStream = nullptr;
    _reuseStream = nullptr;
    _hasReuseStream = false;
    _hasStatusBatch = false;
}

bool indexing_detail::IndexingWorkspaceAccess::beginStatusBatch(
    IndexingWorkspace& workspace) noexcept
{
    if (workspace._hasStatusBatch)
    {
        return false;
    }
    workspace._hasStatusBatch = true;
    return true;
}

} // namespace plamatrix
