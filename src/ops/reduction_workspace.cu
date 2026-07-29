#include <stdexcept>
#include <utility>

#include "plamatrix/ops/reduction.h"

namespace plamatrix
{
namespace
{

const char* kCloseBeforeSyncReserve =
    "ReductionWorkspace::reserveBytes cannot replace a stream-ordered allocation; "
    "call closeAsyncAllocation first";

const char* kSyncGrowBeforeLaunch =
    "ReductionWorkspace::reserveBytesAsync cannot grow an ordinary allocation; "
    "synchronize the owning stream and call reserveBytes synchronously before launching";

const char* kCrossStreamReuse =
    "ReductionWorkspace cannot be reused on a different stream; synchronize the owning stream "
    "and explicitly reset or close the workspace first";

} // anonymous namespace

ReductionWorkspace::~ReductionWorkspace() noexcept
{
    release();
}

ReductionWorkspace::ReductionWorkspace(ReductionWorkspace&& other) noexcept
    : _capacityBytes(other._capacityBytes)
    , _data(other._data)
    , _allocationKind(other._allocationKind)
    , _allocationStream(other._allocationStream)
    , _reuseStream(other._reuseStream)
    , _hasReuseStream(other._hasReuseStream)
{
    other._capacityBytes = 0;
    other._data = nullptr;
    other._allocationKind = AllocationKind::Normal;
    other._allocationStream = nullptr;
    other._reuseStream = nullptr;
    other._hasReuseStream = false;
}

ReductionWorkspace& ReductionWorkspace::operator=(ReductionWorkspace&& other) noexcept
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

        other._capacityBytes = 0;
        other._data = nullptr;
        other._allocationKind = AllocationKind::Normal;
        other._allocationStream = nullptr;
        other._reuseStream = nullptr;
        other._hasReuseStream = false;
    }
    return *this;
}

void ReductionWorkspace::reserveBytes(std::size_t bytes)
{
    if (_allocationKind == AllocationKind::StreamOrderedAsync && _data != nullptr)
    {
        throw std::logic_error(kCloseBeforeSyncReserve);
    }

    if (bytes <= _capacityBytes)
    {
        _reuseStream = nullptr;
        _hasReuseStream = false;
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
}

void ReductionWorkspace::reserveBytesAsync(std::size_t bytes, cudaStream_t stream)
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

void ReductionWorkspace::closeAsyncAllocation()
{
    if (_data == nullptr)
    {
        _capacityBytes = 0;
        _allocationKind = AllocationKind::Normal;
        _allocationStream = nullptr;
        _reuseStream = nullptr;
        _hasReuseStream = false;
        return;
    }
    if (_allocationKind != AllocationKind::StreamOrderedAsync)
    {
        throw std::logic_error(
            "ReductionWorkspace::closeAsyncAllocation requires a stream-ordered allocation");
    }

    PLAMATRIX_CHECK_CUDA(cudaFreeAsync(_data, _allocationStream));
    _capacityBytes = 0;
    _data = nullptr;
    _allocationKind = AllocationKind::Normal;
    _allocationStream = nullptr;
    _reuseStream = nullptr;
    _hasReuseStream = false;
}

void ReductionWorkspace::release() noexcept
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
}

} // namespace plamatrix
