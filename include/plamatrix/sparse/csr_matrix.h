#pragma once

#include <cmath>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "plamatrix/core/allocator.h"
#include "plamatrix/core/device_matrix.h"
#include "plamatrix/core/error_check.h"

namespace plamatrix
{

namespace detail
{
struct CSRMatrixAccess;
}

template <typename Scalar, Device Dev>
class CSRMatrix : public DeviceMatrix<Scalar, Dev>
{
public:
    using Base = DeviceMatrix<Scalar, Dev>;

    /// Construct a CSR sparse matrix with given dimensions and non-zero count.
    /// Allocates three arrays: values (nnz), col_indices (nnz), row_offsets (rows+1).
    /// Row offsets are zero-initialized.
    /// @param rows  Number of rows
    /// @param cols  Number of columns
    /// @param nnz   Number of non-zero entries
    CSRMatrix(Index rows, Index cols, Index nnz)
        : CSRMatrix(rows, cols, nnz, AllocationKind::Regular, nullptr, true)
    {
    }

    /// Construct a CPU CSR matrix backed by page-locked host memory.
    template <Device D = Dev, std::enable_if_t<D == Dev && D == Device::CPU, int> = 0>
    static CSRMatrix pinned(Index rows, Index cols, Index nnz)
    {
#ifdef PLAMATRIX_NO_CUDA
        static_cast<void>(rows);
        static_cast<void>(cols);
        static_cast<void>(nnz);
        throw std::runtime_error("CSRMatrix::pinned requires PLAMATRIX_WITH_CUDA=ON");
#else
        return CSRMatrix(rows, cols, nnz, AllocationKind::PinnedHost, nullptr, true);
#endif
    }

    /// Construct an uninitialized GPU CSR matrix using stream-ordered allocations.
    /// The stream must remain valid until closeAsyncAllocation() is called or the matrix is
    /// destroyed. Call closeAsyncAllocation() before destroying a non-default stream so release
    /// errors are reported instead of handled by the synchronous destructor fallback.
    template <Device D = Dev, std::enable_if_t<D == Dev && D == Device::GPU, int> = 0>
    static CSRMatrix uninitializedAsync(
        Index rows, Index cols, Index nnz, cudaStream_t stream = nullptr)
    {
#ifdef PLAMATRIX_NO_CUDA
        static_cast<void>(rows);
        static_cast<void>(cols);
        static_cast<void>(nnz);
        static_cast<void>(stream);
        throw std::runtime_error(
            "CSRMatrix::uninitializedAsync requires PLAMATRIX_WITH_CUDA=ON");
#else
        return CSRMatrix(
            rows, cols, nnz, AllocationKind::StreamOrderedGpu, stream, false);
#endif
    }

    /// Destructor. Frees the three allocated arrays.
    ~CSRMatrix()
    {
        releaseArrays();
    }

    // Non-copyable
    CSRMatrix(const CSRMatrix&) = delete;
    CSRMatrix& operator=(const CSRMatrix&) = delete;

    /// Move constructor. Transfers ownership of CSR arrays from source.
    /// @param other  Source matrix (left in null state)
    CSRMatrix(CSRMatrix&& other) noexcept
        : DeviceMatrix<Scalar, Dev>(std::move(other))
        , _nnz(other._nnz)
        , _values(other._values)
        , _col_indices(other._col_indices)
        , _row_offsets(other._row_offsets)
        , _allocationKind(other._allocationKind)
        , _allocationStream(other._allocationStream)
        , _structureValidated(other._structureValidated)
        , _mutablePointersEscaped(other._mutablePointersEscaped)
        , _pendingWriteStream(other._pendingWriteStream)
        , _hasPendingWrite(other._hasPendingWrite)
    {
        other._nnz = 0;
        other._values = nullptr;
        other._col_indices = nullptr;
        other._row_offsets = nullptr;
        other._allocationKind = AllocationKind::Regular;
        other._allocationStream = nullptr;
        other._structureValidated = true;
        other._mutablePointersEscaped = false;
        other._pendingWriteStream = nullptr;
        other._hasPendingWrite = false;
    }

    /// Move assignment. Releases current arrays and transfers ownership from source.
    /// @param other  Source matrix (left in null state)
    /// @return  Reference to this matrix
    CSRMatrix& operator=(CSRMatrix&& other) noexcept
    {
        if (this != &other)
        {
            releaseArrays();
            DeviceMatrix<Scalar, Dev>::operator=(std::move(other));
            _nnz = other._nnz;
            _values = other._values;
            _col_indices = other._col_indices;
            _row_offsets = other._row_offsets;
            _allocationKind = other._allocationKind;
            _allocationStream = other._allocationStream;
            _structureValidated = other._structureValidated;
            _mutablePointersEscaped = other._mutablePointersEscaped;
            _pendingWriteStream = other._pendingWriteStream;
            _hasPendingWrite = other._hasPendingWrite;
            other._nnz = 0;
            other._values = nullptr;
            other._col_indices = nullptr;
            other._row_offsets = nullptr;
            other._allocationKind = AllocationKind::Regular;
            other._allocationStream = nullptr;
            other._structureValidated = true;
            other._mutablePointersEscaped = false;
            other._pendingWriteStream = nullptr;
            other._hasPendingWrite = false;
        }
        return *this;
    }

    /// @return Number of non-zero entries
    Index nnz() const { return _nnz; }

    /// @return Raw pointer to values array. Escaping mutable storage permanently disables trusted
    /// fixed-async use; adaptive consumers revalidate every time.
    Scalar* values()
    {
        _structureValidated = false;
        _mutablePointersEscaped = true;
        return _values;
    }

    /// @return Raw pointer to values array (const)
    const Scalar* values() const { return _values; }

    /// @return Raw pointer to column indices array, with the same escaped-storage rule as values().
    Index* colIndices()
    {
        _structureValidated = false;
        _mutablePointersEscaped = true;
        return _col_indices;
    }

    /// @return Raw pointer to column indices array (const)
    const Index* colIndices() const { return _col_indices; }

    /// @return Raw pointer to row offsets array (size rows+1), with the same escaped-storage rule.
    Index* rowOffsets()
    {
        _structureValidated = false;
        _mutablePointersEscaped = true;
        return _row_offsets;
    }

    /// @return Raw pointer to row offsets array (size rows+1, const)
    const Index* rowOffsets() const { return _row_offsets; }

    /// Return whether CSR structure is trusted and no mutable pointer or async write is outstanding.
    bool hasValidatedStructure() const noexcept
    {
        return _structureValidated && !_mutablePointersEscaped && !_hasPendingWrite;
    }

    /// Return whether trusted CSR storage may be consumed in-order on the given stream.
    bool isStructureUsableOnStream(cudaStream_t stream) const noexcept
    {
        return _structureValidated && !_mutablePointersEscaped
            && (!_hasPendingWrite || _pendingWriteStream == stream);
    }

    /// Validate row offsets, column indices, and finite values.
    /// GPU validation synchronizes stream once; fixed-iteration async solvers therefore require
    /// callers to validate before submission when mutable GPU array pointers have escaped.
    void validateStructure(cudaStream_t stream = nullptr) const
    {
        if (_hasPendingWrite && _pendingWriteStream != stream)
        {
            throw std::logic_error(
                "CSRMatrix::validateStructure must use the stream of the pending async copy");
        }
        if (_structureValidated && !_mutablePointersEscaped && !_hasPendingWrite)
        {
            return;
        }
        if constexpr (Dev == Device::CPU)
        {
#ifdef PLAMATRIX_WITH_CUDA
            if (_hasPendingWrite)
            {
                PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
                _pendingWriteStream = nullptr;
                _hasPendingWrite = false;
            }
#endif
            validateHostStructure();
            _structureValidated = true;
        }
        else
        {
#ifdef PLAMATRIX_NO_CUDA
            static_cast<void>(stream);
            throw std::runtime_error(
                "CSRMatrix::validateStructure requires PLAMATRIX_WITH_CUDA=ON");
#else
            checkAsyncStream("validateStructure", stream);
            auto host = toCpuAsync(stream);
            PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
            host.validateStructure(stream);
            _structureValidated = true;
            _pendingWriteStream = nullptr;
            _hasPendingWrite = false;
#endif
        }
    }

    /// @return true when this CPU CSR matrix owns page-locked host arrays.
    bool isPinnedHost() const noexcept
    {
        return Dev == Device::CPU && _allocationKind == AllocationKind::PinnedHost;
    }

    /// @return true when this GPU CSR matrix owns stream-ordered arrays.
    bool isAsyncAllocation() const noexcept
    {
        return Dev == Device::GPU && _allocationKind == AllocationKind::StreamOrderedGpu;
    }

    /// @return stream that owns this stream-ordered allocation, or nullptr for ordinary storage.
    cudaStream_t asyncAllocationStream() const noexcept
    {
        return isAsyncAllocation() ? _allocationStream : nullptr;
    }

    /// Enqueue checked release of stream-ordered GPU arrays.
    template <Device D = Dev, std::enable_if_t<D == Dev && D == Device::GPU, int> = 0>
    void closeAsyncAllocation()
    {
        if (_values == nullptr && _col_indices == nullptr && _row_offsets == nullptr)
        {
            this->_rows = 0;
            this->_cols = 0;
            _nnz = 0;
            _allocationKind = AllocationKind::Regular;
            _allocationStream = nullptr;
            _structureValidated = true;
            _mutablePointersEscaped = false;
            _pendingWriteStream = nullptr;
            _hasPendingWrite = false;
            return;
        }
        if (_allocationKind != AllocationKind::StreamOrderedGpu)
        {
            throw std::logic_error(
                "CSRMatrix::closeAsyncAllocation requires stream-ordered storage");
        }
#ifdef PLAMATRIX_NO_CUDA
        throw std::runtime_error(
            "CSRMatrix::closeAsyncAllocation requires PLAMATRIX_WITH_CUDA=ON");
#else
        GpuAllocator<Scalar>::deallocateAsync(_values, _allocationStream);
        _values = nullptr;
        GpuAllocator<Index>::deallocateAsync(_col_indices, _allocationStream);
        _col_indices = nullptr;
        GpuAllocator<Index>::deallocateAsync(_row_offsets, _allocationStream);
        _row_offsets = nullptr;
        this->_rows = 0;
        this->_cols = 0;
        _nnz = 0;
        _allocationKind = AllocationKind::Regular;
        _allocationStream = nullptr;
        _structureValidated = true;
        _mutablePointersEscaped = false;
        _pendingWriteStream = nullptr;
        _hasPendingWrite = false;
#endif
    }

    /// Transfer a GPU CSR matrix to CPU.
    /// @return CPU copy of this matrix
    /// @throws std::runtime_error if CUDA support is disabled or a CUDA copy fails
    /// @throws std::overflow_error if a transfer byte count overflows size_t
    template <Device D = Dev, std::enable_if_t<D == Dev && D == Device::GPU, int> = 0>
    CSRMatrix<Scalar, Device::CPU> toCpu() const
    {
        if (isAsyncAllocation())
        {
            throw std::logic_error(
                "CSRMatrix::toCpu cannot synchronously read stream-ordered storage; "
                "use toCpuAsync on the owning stream");
        }
#ifdef PLAMATRIX_NO_CUDA
        throw std::runtime_error("CSRMatrix::toCpu requires PLAMATRIX_WITH_CUDA=ON");
#else
        if (_hasPendingWrite)
        {
            PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(_pendingWriteStream));
            _pendingWriteStream = nullptr;
            _hasPendingWrite = false;
        }
        CSRMatrix<Scalar, Device::CPU> result(this->_rows, this->_cols, _nnz);
        if (_nnz > 0)
        {
            PLAMATRIX_CHECK_CUDA(cudaMemcpy(
                result._values, _values, valueBytes(), cudaMemcpyDeviceToHost));
            PLAMATRIX_CHECK_CUDA(cudaMemcpy(
                result._col_indices, _col_indices, columnBytes(), cudaMemcpyDeviceToHost));
        }
        PLAMATRIX_CHECK_CUDA(cudaMemcpy(
            result._row_offsets, _row_offsets, rowOffsetBytes(), cudaMemcpyDeviceToHost));
        result._structureValidated = hasValidatedStructure();
        result._mutablePointersEscaped = false;
        return result;
#endif
    }

    /// Asynchronously transfer a GPU CSR matrix to CPU on the caller's stream.
    /// The caller must synchronize the stream before reading the returned matrix.
    /// @param stream CUDA stream used for all three CSR array copies
    /// @return CPU copy of this matrix
    /// @throws std::runtime_error if CUDA support is disabled or a CUDA copy fails
    /// @throws std::overflow_error if a transfer byte count overflows size_t
    template <Device D = Dev, std::enable_if_t<D == Dev && D == Device::GPU, int> = 0>
    CSRMatrix<Scalar, Device::CPU> toCpuAsync(cudaStream_t stream = nullptr) const
    {
#ifdef PLAMATRIX_NO_CUDA
        static_cast<void>(stream);
        throw std::runtime_error("CSRMatrix::toCpuAsync requires PLAMATRIX_WITH_CUDA=ON");
#else
        checkAsyncStream("toCpuAsync", stream);
        auto result = CSRMatrix<Scalar, Device::CPU>::pinned(
            this->_rows, this->_cols, _nnz);
        copyToCpuAsync(result, stream);
        return result;
#endif
    }

    /// Asynchronously transfer this GPU CSR matrix into an existing CPU matrix.
    /// @param output CPU output matrix with matching rows, columns, and nnz
    /// @param stream CUDA stream used for all three CSR array copies
    /// @throws std::runtime_error if output shape differs, CUDA is disabled, or a copy fails
    /// @throws std::overflow_error if a transfer byte count overflows size_t
    template <Device D = Dev, std::enable_if_t<D == Dev && D == Device::GPU, int> = 0>
    void copyToCpuAsync(
        CSRMatrix<Scalar, Device::CPU>& output, cudaStream_t stream = nullptr) const
    {
        checkTransferOutput("copyToCpuAsync", output.rows(), output.cols(), output.nnz());
#ifdef PLAMATRIX_NO_CUDA
        static_cast<void>(stream);
        throw std::runtime_error("CSRMatrix::copyToCpuAsync requires PLAMATRIX_WITH_CUDA=ON");
#else
        checkAsyncStream("copyToCpuAsync", stream);
        checkPendingWriteStream("copyToCpuAsync", stream);
        if (!output.isPinnedHost())
        {
            throw std::invalid_argument(
                "CSRMatrix::copyToCpuAsync requires pinned CPU output storage");
        }
        output.beginAsyncWrite("copyToCpuAsync", stream);
        if (_nnz > 0)
        {
            PLAMATRIX_CHECK_CUDA(cudaMemcpyAsync(
                output._values, _values, valueBytes(), cudaMemcpyDeviceToHost, stream));
            PLAMATRIX_CHECK_CUDA(cudaMemcpyAsync(
                output._col_indices, _col_indices, columnBytes(), cudaMemcpyDeviceToHost, stream));
        }
        PLAMATRIX_CHECK_CUDA(cudaMemcpyAsync(
            output._row_offsets, _row_offsets, rowOffsetBytes(), cudaMemcpyDeviceToHost, stream));
#endif
    }

    /// Transfer a CPU CSR matrix to GPU.
    /// @return GPU copy of this matrix
    /// @throws std::runtime_error if CUDA support is disabled or a CUDA copy fails
    /// @throws std::overflow_error if a transfer byte count overflows size_t
    template <Device D = Dev, std::enable_if_t<D == Dev && D == Device::CPU, int> = 0>
    CSRMatrix<Scalar, Device::GPU> toGpu() const
    {
#ifdef PLAMATRIX_NO_CUDA
        throw std::runtime_error("CSRMatrix::toGpu requires PLAMATRIX_WITH_CUDA=ON");
#else
        validateStructure();
        CSRMatrix<Scalar, Device::GPU> result(this->_rows, this->_cols, _nnz);
        if (_nnz > 0)
        {
            PLAMATRIX_CHECK_CUDA(cudaMemcpy(
                result._values, _values, valueBytes(), cudaMemcpyHostToDevice));
            PLAMATRIX_CHECK_CUDA(cudaMemcpy(
                result._col_indices, _col_indices, columnBytes(), cudaMemcpyHostToDevice));
        }
        PLAMATRIX_CHECK_CUDA(cudaMemcpy(
            result._row_offsets, _row_offsets, rowOffsetBytes(), cudaMemcpyHostToDevice));
        result._structureValidated = true;
        result._mutablePointersEscaped = false;
        result._pendingWriteStream = nullptr;
        result._hasPendingWrite = false;
        return result;
#endif
    }

    /// Asynchronously transfer a CPU CSR matrix to GPU on the caller's stream.
    /// The source must use pinned host storage. The returned allocation retains the stream; call
    /// closeAsyncAllocation() before destroying a non-default stream.
    /// The caller must synchronize the stream before using the result on another stream.
    /// @param stream CUDA stream used for all three CSR array copies
    /// @return GPU copy of this matrix
    /// @throws std::runtime_error if CUDA support is disabled or a CUDA copy fails
    /// @throws std::overflow_error if a transfer byte count overflows size_t
    template <Device D = Dev, std::enable_if_t<D == Dev && D == Device::CPU, int> = 0>
    CSRMatrix<Scalar, Device::GPU> toGpuAsync(cudaStream_t stream = nullptr) const
    {
#ifdef PLAMATRIX_NO_CUDA
        static_cast<void>(stream);
        throw std::runtime_error("CSRMatrix::toGpuAsync requires PLAMATRIX_WITH_CUDA=ON");
#else
        if (!isPinnedHost())
        {
            throw std::invalid_argument(
                "CSRMatrix::toGpuAsync requires pinned CPU input storage");
        }
        validateStructure();
        auto result = CSRMatrix<Scalar, Device::GPU>::uninitializedAsync(
            this->_rows, this->_cols, _nnz, stream);
        copyToGpuAsync(result, stream);
        return result;
#endif
    }

    /// Asynchronously transfer this CPU CSR matrix into an existing GPU matrix.
    /// @param output GPU output matrix with matching rows, columns, and nnz
    /// @param stream CUDA stream used for all three CSR array copies
    /// @throws std::runtime_error if output shape differs, CUDA is disabled, or a copy fails
    /// @throws std::overflow_error if a transfer byte count overflows size_t
    template <Device D = Dev, std::enable_if_t<D == Dev && D == Device::CPU, int> = 0>
    void copyToGpuAsync(
        CSRMatrix<Scalar, Device::GPU>& output, cudaStream_t stream = nullptr) const
    {
        checkTransferOutput("copyToGpuAsync", output.rows(), output.cols(), output.nnz());
#ifdef PLAMATRIX_NO_CUDA
        static_cast<void>(stream);
        throw std::runtime_error("CSRMatrix::copyToGpuAsync requires PLAMATRIX_WITH_CUDA=ON");
#else
        if (!isPinnedHost())
        {
            throw std::invalid_argument(
                "CSRMatrix::copyToGpuAsync requires pinned CPU input storage");
        }
        validateStructure();
        output.checkAsyncStream("copyToGpuAsync", stream);
        output.beginAsyncWrite("copyToGpuAsync", stream);
        if (_nnz > 0)
        {
            PLAMATRIX_CHECK_CUDA(cudaMemcpyAsync(
                output._values, _values, valueBytes(), cudaMemcpyHostToDevice, stream));
            PLAMATRIX_CHECK_CUDA(cudaMemcpyAsync(
                output._col_indices, _col_indices, columnBytes(), cudaMemcpyHostToDevice, stream));
        }
        PLAMATRIX_CHECK_CUDA(cudaMemcpyAsync(
            output._row_offsets, _row_offsets, rowOffsetBytes(), cudaMemcpyHostToDevice, stream));
        output._structureValidated = true;
#endif
    }

private:
    template <typename, Device>
    friend class CSRMatrix;
    friend struct detail::CSRMatrixAccess;

    enum class AllocationKind
    {
        Regular,
        PinnedHost,
        StreamOrderedGpu
    };

    CSRMatrix(Index rows,
              Index cols,
              Index nnz,
              AllocationKind allocation_kind,
              cudaStream_t allocation_stream,
              bool zero_indices)
        : DeviceMatrix<Scalar, Dev>(0, 0)
        , _nnz(nnz)
        , _allocationKind(allocation_kind)
        , _allocationStream(allocation_stream)
        , _structureValidated(nnz == 0)
        , _mutablePointersEscaped(false)
    {
        if (rows < 0 || cols < 0)
        {
            throw std::invalid_argument("CSRMatrix dimensions must be non-negative");
        }
        if (nnz < 0)
        {
            throw std::invalid_argument("CSRMatrix nnz must be non-negative");
        }
        if (nnz > 0 && (rows == 0 || cols == 0))
        {
            throw std::invalid_argument(
                "CSRMatrix cannot store non-zero entries with zero rows or columns");
        }
        this->_rows = rows;
        this->_cols = cols;

        try
        {
            if (nnz > 0)
            {
                if constexpr (Dev == Device::CPU)
                {
                    if (_allocationKind == AllocationKind::PinnedHost)
                    {
                        _values = PinnedCpuAllocator<Scalar>::allocate(
                            static_cast<std::size_t>(nnz));
                        _col_indices = PinnedCpuAllocator<Index>::allocate(
                            static_cast<std::size_t>(nnz));
                    }
                    else
                    {
                        _values = CpuAllocator<Scalar>::allocate(static_cast<std::size_t>(nnz));
                        _col_indices = CpuAllocator<Index>::allocate(
                            static_cast<std::size_t>(nnz));
                    }
                    if (zero_indices)
                    {
                        std::memset(_col_indices, 0, columnBytes());
                    }
                }
                else
                {
                    if (_allocationKind == AllocationKind::StreamOrderedGpu)
                    {
                        _values = GpuAllocator<Scalar>::allocateAsync(
                            static_cast<std::size_t>(nnz), _allocationStream);
                        _col_indices = GpuAllocator<Index>::allocateAsync(
                            static_cast<std::size_t>(nnz), _allocationStream);
                    }
                    else
                    {
                        _values = GpuAllocator<Scalar>::allocate(static_cast<std::size_t>(nnz));
                        _col_indices = GpuAllocator<Index>::allocate(
                            static_cast<std::size_t>(nnz));
                    }
                    if (zero_indices)
                    {
                        PLAMATRIX_CHECK_CUDA(cudaMemset(_col_indices, 0, columnBytes()));
                    }
                }
            }

            const auto row_offset_count = static_cast<std::size_t>(rows) + 1;
            if constexpr (Dev == Device::CPU)
            {
                if (_allocationKind == AllocationKind::PinnedHost)
                {
                    _row_offsets = PinnedCpuAllocator<Index>::allocate(row_offset_count);
                }
                else
                {
                    _row_offsets = CpuAllocator<Index>::allocate(row_offset_count);
                }
                if (zero_indices)
                {
                    std::memset(_row_offsets, 0, rowOffsetBytes());
                }
            }
            else
            {
                if (_allocationKind == AllocationKind::StreamOrderedGpu)
                {
                    _row_offsets = GpuAllocator<Index>::allocateAsync(
                        row_offset_count, _allocationStream);
                }
                else
                {
                    _row_offsets = GpuAllocator<Index>::allocate(row_offset_count);
                }
                if (zero_indices)
                {
                    PLAMATRIX_CHECK_CUDA(cudaMemset(_row_offsets, 0, rowOffsetBytes()));
                    if (_allocationKind == AllocationKind::Regular)
                    {
                        PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(nullptr));
                    }
                }
            }
        }
        catch (...)
        {
            releaseArrays();
            throw;
        }
    }

    std::size_t valueBytes() const
    {
        return detail::checkedAllocationBytes<Scalar>(static_cast<std::size_t>(_nnz));
    }

    std::size_t columnBytes() const
    {
        return detail::checkedAllocationBytes<Index>(static_cast<std::size_t>(_nnz));
    }

    std::size_t rowOffsetBytes() const
    {
        const auto count = static_cast<std::size_t>(this->_rows) + 1;
        return detail::checkedAllocationBytes<Index>(count);
    }

    void validateHostStructure() const
    {
        if (_row_offsets == nullptr || _row_offsets[0] != 0
            || _row_offsets[this->_rows] != _nnz)
        {
            throw std::invalid_argument(
                "CSRMatrix row offsets must start at zero and end at nnz");
        }
        for (Index row = 0; row < this->_rows; ++row)
        {
            const Index begin = _row_offsets[row];
            const Index end = _row_offsets[row + 1];
            if (begin < 0 || end < begin || end > _nnz)
            {
                std::ostringstream message;
                message << "CSRMatrix invalid row offsets at row " << row;
                throw std::invalid_argument(message.str());
            }
        }
        for (Index position = 0; position < _nnz; ++position)
        {
            if (_col_indices[position] < 0 || _col_indices[position] >= this->_cols)
            {
                std::ostringstream message;
                message << "CSRMatrix column index out of range at position " << position;
                throw std::invalid_argument(message.str());
            }
            if (!std::isfinite(static_cast<double>(_values[position])))
            {
                std::ostringstream message;
                message << "CSRMatrix value is not finite at position " << position;
                throw std::invalid_argument(message.str());
            }
        }
    }

    void checkTransferOutput(const char* operation, Index rows, Index cols, Index nnz) const
    {
        if (rows != this->_rows || cols != this->_cols || nnz != _nnz)
        {
            std::ostringstream message;
            message << "CSRMatrix::" << operation << " output mismatch: output is "
                    << rows << "x" << cols << " with nnz=" << nnz << ", expected "
                    << this->_rows << "x" << this->_cols << " with nnz=" << _nnz;
            throw std::runtime_error(message.str());
        }
    }

    void checkAsyncStream(const char* operation, cudaStream_t stream) const
    {
        if (_allocationKind == AllocationKind::StreamOrderedGpu
            && stream != _allocationStream)
        {
            std::ostringstream message;
            message << "CSRMatrix::" << operation
                    << " must use the stream that owns the async allocation";
            throw std::logic_error(message.str());
        }
    }

    void checkPendingWriteStream(const char* operation, cudaStream_t stream) const
    {
        if (_hasPendingWrite && _pendingWriteStream != stream)
        {
            std::ostringstream message;
            message << "CSRMatrix::" << operation
                    << " cannot overlap a pending async write from another stream";
            throw std::logic_error(message.str());
        }
    }

    void beginAsyncWrite(const char* operation, cudaStream_t stream) const
    {
        checkPendingWriteStream(operation, stream);
        _structureValidated = false;
        _pendingWriteStream = stream;
        _hasPendingWrite = true;
    }

    void markAsyncWriteTrusted() const noexcept
    {
        _structureValidated = true;
    }

    void completeAsyncWrite(bool trusted) const noexcept
    {
        _structureValidated = trusted;
        _pendingWriteStream = nullptr;
        _hasPendingWrite = false;
    }

    template <typename Value>
    static void releaseGpuPointerNoThrow(
        Value* pointer, AllocationKind allocation_kind, cudaStream_t stream) noexcept
    {
        static_cast<void>(stream);
        if (pointer == nullptr)
        {
            return;
        }
        if (allocation_kind == AllocationKind::StreamOrderedGpu)
        {
#ifdef PLAMATRIX_WITH_CUDA
            const cudaError_t synchronize_status = cudaDeviceSynchronize();
            if (synchronize_status == cudaSuccess
                && cudaFreeAsync(pointer, nullptr) == cudaSuccess)
            {
                static_cast<void>(cudaStreamSynchronize(nullptr));
                return;
            }
            static_cast<void>(cudaGetLastError());
#endif
            GpuAllocator<Value>::deallocateNoThrow(pointer);
            return;
        }
        GpuAllocator<Value>::deallocateNoThrow(pointer);
    }

    /// Release allocated CSR arrays. Safe to call multiple times (nullptr-safe).
    void releaseArrays() noexcept
    {
        if (_values != nullptr)
        {
            if constexpr (Dev == Device::CPU)
            {
                if (_allocationKind == AllocationKind::PinnedHost)
                {
                    PinnedCpuAllocator<Scalar>::deallocateNoThrow(_values);
                }
                else
                {
                    CpuAllocator<Scalar>::deallocateNoThrow(_values);
                }
            }
            else
            {
                releaseGpuPointerNoThrow(_values, _allocationKind, _allocationStream);
            }
            _values = nullptr;
        }
        if (_col_indices != nullptr)
        {
            if constexpr (Dev == Device::CPU)
            {
                if (_allocationKind == AllocationKind::PinnedHost)
                {
                    PinnedCpuAllocator<Index>::deallocateNoThrow(_col_indices);
                }
                else
                {
                    CpuAllocator<Index>::deallocateNoThrow(_col_indices);
                }
            }
            else
            {
                releaseGpuPointerNoThrow(
                    _col_indices, _allocationKind, _allocationStream);
            }
            _col_indices = nullptr;
        }
        if (_row_offsets != nullptr)
        {
            if constexpr (Dev == Device::CPU)
            {
                if (_allocationKind == AllocationKind::PinnedHost)
                {
                    PinnedCpuAllocator<Index>::deallocateNoThrow(_row_offsets);
                }
                else
                {
                    CpuAllocator<Index>::deallocateNoThrow(_row_offsets);
                }
            }
            else
            {
                releaseGpuPointerNoThrow(
                    _row_offsets, _allocationKind, _allocationStream);
            }
            _row_offsets = nullptr;
        }
    }

    Index _nnz = 0;
    Scalar* _values = nullptr;
    Index* _col_indices = nullptr;
    Index* _row_offsets = nullptr;
    AllocationKind _allocationKind = AllocationKind::Regular;
    cudaStream_t _allocationStream = nullptr;
    mutable bool _structureValidated = true;
    mutable bool _mutablePointersEscaped = false;
    mutable cudaStream_t _pendingWriteStream = nullptr;
    mutable bool _hasPendingWrite = false;
};

namespace detail
{

struct CSRMatrixAccess
{
    template <typename Scalar, Device Dev>
    static Scalar* values(CSRMatrix<Scalar, Dev>& matrix) noexcept
    {
        return matrix._values;
    }

    template <typename Scalar, Device Dev>
    static Index* colIndices(CSRMatrix<Scalar, Dev>& matrix) noexcept
    {
        return matrix._col_indices;
    }

    template <typename Scalar, Device Dev>
    static Index* rowOffsets(CSRMatrix<Scalar, Dev>& matrix) noexcept
    {
        return matrix._row_offsets;
    }

    template <typename Scalar, Device Dev>
    static void beginAsyncWrite(
        CSRMatrix<Scalar, Dev>& matrix,
        cudaStream_t stream,
        const char* operation = "internalAsyncWrite")
    {
        matrix.beginAsyncWrite(operation, stream);
    }

    template <typename Scalar, Device Dev>
    static void markAsyncWriteTrusted(CSRMatrix<Scalar, Dev>& matrix) noexcept
    {
        matrix.markAsyncWriteTrusted();
    }

    template <typename Scalar, Device Dev>
    static void completeAsyncWrite(CSRMatrix<Scalar, Dev>& matrix, bool trusted) noexcept
    {
        matrix.completeAsyncWrite(trusted);
    }
};

} // namespace detail

} // namespace plamatrix
