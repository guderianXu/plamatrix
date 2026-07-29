#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "plamatrix/dense/dense_matrix.h"

namespace plamatrix
{

/// Result of stable row compaction.
/// @tparam Scalar  Matrix element type
/// @tparam Dev     Matrix device
template <typename Scalar, Device Dev>
struct CompactRowsResult
{
    DenseMatrix<Scalar, Dev> values;
    DenseMatrix<Index, Dev> sourceIndices;
};

namespace indexing_detail
{
struct IndexingWorkspaceAccess;
}

/// Caller-owned temporary storage for GPU indexing operations.
///
/// The workspace is move-only, grow-only, and not thread-safe. Async operations may reuse it
/// sequentially on one stream. Reuse on another stream is rejected until the owner synchronizes
/// and resets an ordinary allocation with reserveBytes(), or closes a stream-ordered allocation.
/// Async callers must synchronize their stream and then call checkStatus() before consuming results.
/// An unconsumed status batch must be checked before resetting or closing the workspace.
class IndexingWorkspace
{
public:
    IndexingWorkspace() noexcept = default;

#ifdef PLAMATRIX_WITH_CUDA
    ~IndexingWorkspace() noexcept;
    IndexingWorkspace(IndexingWorkspace&& other) noexcept;
    IndexingWorkspace& operator=(IndexingWorkspace&& other) noexcept;

    /// Grow ordinary storage, or reset its stream reuse binding when capacity is sufficient.
    /// @param bytes  Required byte capacity
    /// @throws std::logic_error if stream work is pending or checkStatus() has not consumed status
    void reserveBytes(std::size_t bytes);

    /// Grow or bind storage for sequential use on stream.
    /// @param bytes   Required byte capacity
    /// @param stream  Stream that owns subsequent asynchronous reuse
    void reserveBytesAsync(std::size_t bytes, cudaStream_t stream);

    /// Enqueue checked release of a stream-ordered allocation.
    /// @throws std::logic_error if checkStatus() has not consumed the current status batch
    void closeAsyncAllocation();

    /// Consume the aggregated device status after the owning stream has been synchronized.
    /// @param operation  Operation name included in any diagnostic
    /// @throws std::logic_error if work is still pending
    /// @throws std::overflow_error for the lowest scan overflow source offset
    /// @throws std::out_of_range for the lowest invalid row-index source offset
    /// If both categories occurred in one batch, overflow has deterministic priority.
    void checkStatus(const char* operation);
#else
    ~IndexingWorkspace() noexcept = default;
    IndexingWorkspace(IndexingWorkspace&& other) noexcept = default;
    IndexingWorkspace& operator=(IndexingWorkspace&& other) noexcept = default;

    void reserveBytes(std::size_t)
    {
        throw std::runtime_error(
            "IndexingWorkspace::reserveBytes requires PLAMATRIX_WITH_CUDA=ON");
    }

    void reserveBytesAsync(std::size_t, cudaStream_t)
    {
        throw std::runtime_error(
            "IndexingWorkspace::reserveBytesAsync requires PLAMATRIX_WITH_CUDA=ON");
    }

    void closeAsyncAllocation()
    {
    }

    void checkStatus(const char*)
    {
        throw std::runtime_error(
            "IndexingWorkspace::checkStatus requires PLAMATRIX_WITH_CUDA=ON");
    }
#endif

    IndexingWorkspace(const IndexingWorkspace&) = delete;
    IndexingWorkspace& operator=(const IndexingWorkspace&) = delete;

    /// @return Current byte capacity
    std::size_t capacityBytes() const noexcept
    {
        return _capacityBytes;
    }

    /// @return Mutable base pointer of the workspace allocation
    void* data() noexcept
    {
        return _data;
    }

    /// @return Const base pointer of the workspace allocation
    const void* data() const noexcept
    {
        return _data;
    }

private:
    enum class AllocationKind
    {
        Normal,
        StreamOrderedAsync
    };

#ifdef PLAMATRIX_WITH_CUDA
    void release() noexcept;
#endif

    friend struct indexing_detail::IndexingWorkspaceAccess;

    std::size_t _capacityBytes = 0;
    void* _data = nullptr;
    AllocationKind _allocationKind = AllocationKind::Normal;
    cudaStream_t _allocationStream = nullptr;
    cudaStream_t _reuseStream = nullptr;
    bool _hasReuseStream = false;
    bool _hasStatusBatch = false;
};

/// Compute an exclusive scan in column-major linear storage order.
/// @throws std::overflow_error if any prefix addition exceeds the Index range
DenseMatrix<Index, Device::CPU> exclusiveScan(
    const DenseMatrix<Index, Device::CPU>& counts);

/// Gather CPU input rows in index order; duplicate indices are preserved.
/// @throws std::invalid_argument if indices is not K x 1
/// @throws std::out_of_range if an index is outside input rows
template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> gatherRows(
    const DenseMatrix<Scalar, Device::CPU>& input,
    const DenseMatrix<Index, Device::CPU>& indices);

/// Scatter CPU rows with the lowest source row winning duplicate destinations.
/// @throws std::invalid_argument if matrix shapes are incompatible
/// @throws std::out_of_range if an index is invalid; output remains unchanged
template <typename Scalar>
void scatterRows(
    const DenseMatrix<Scalar, Device::CPU>& values,
    const DenseMatrix<Index, Device::CPU>& indices,
    DenseMatrix<Scalar, Device::CPU>& output);

/// Stably compact CPU rows selected by a nonzero byte mask.
/// @throws std::invalid_argument if keep_mask is not input.rows() x 1
template <typename Scalar>
CompactRowsResult<Scalar, Device::CPU> compactRows(
    const DenseMatrix<Scalar, Device::CPU>& input,
    const DenseMatrix<std::uint8_t, Device::CPU>& keep_mask);

#ifdef PLAMATRIX_WITH_CUDA

/// Synchronously scan GPU counts using ordinary output allocation and a local workspace.
DenseMatrix<Index, Device::GPU> exclusiveScan(
    const DenseMatrix<Index, Device::GPU>& counts);

/// Synchronously scan GPU counts on stream using caller-owned reusable workspace.
DenseMatrix<Index, Device::GPU> exclusiveScan(
    const DenseMatrix<Index, Device::GPU>& counts,
    IndexingWorkspace& workspace,
    cudaStream_t stream = nullptr);

/// Synchronously scan into same-shaped output and check overflow status.
void exclusiveScan(
    const DenseMatrix<Index, Device::GPU>& counts,
    DenseMatrix<Index, Device::GPU>& output,
    IndexingWorkspace& workspace,
    cudaStream_t stream = nullptr);

/// Enqueue scan into stream-ordered output. Input, output, workspace, and stream must remain alive
/// until synchronization; close output and async workspace allocations before destroying stream.
DenseMatrix<Index, Device::GPU> exclusiveScanAsync(
    const DenseMatrix<Index, Device::GPU>& counts,
    IndexingWorkspace& workspace,
    cudaStream_t stream);

/// Enqueue scan into caller-owned same-shaped output without host synchronization.
void exclusiveScanAsync(
    const DenseMatrix<Index, Device::GPU>& counts,
    DenseMatrix<Index, Device::GPU>& output,
    IndexingWorkspace& workspace,
    cudaStream_t stream);

/// Synchronously gather GPU rows using ordinary output allocation and a local workspace.
template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> gatherRows(
    const DenseMatrix<Scalar, Device::GPU>& input,
    const DenseMatrix<Index, Device::GPU>& indices);

/// Synchronously gather GPU rows with reusable workspace and explicit stream.
template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> gatherRows(
    const DenseMatrix<Scalar, Device::GPU>& input,
    const DenseMatrix<Index, Device::GPU>& indices,
    IndexingWorkspace& workspace,
    cudaStream_t stream = nullptr);

/// Synchronously gather into K x input.cols() output; invalid indices leave output unchanged.
template <typename Scalar>
void gatherRows(
    const DenseMatrix<Scalar, Device::GPU>& input,
    const DenseMatrix<Index, Device::GPU>& indices,
    DenseMatrix<Scalar, Device::GPU>& output,
    IndexingWorkspace& workspace,
    cudaStream_t stream = nullptr);

/// Enqueue gather into stream-ordered output without host synchronization.
template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> gatherRowsAsync(
    const DenseMatrix<Scalar, Device::GPU>& input,
    const DenseMatrix<Index, Device::GPU>& indices,
    IndexingWorkspace& workspace,
    cudaStream_t stream);

/// Enqueue gather into caller-owned output; call workspace.checkStatus() after synchronization.
template <typename Scalar>
void gatherRowsAsync(
    const DenseMatrix<Scalar, Device::GPU>& input,
    const DenseMatrix<Index, Device::GPU>& indices,
    DenseMatrix<Scalar, Device::GPU>& output,
    IndexingWorkspace& workspace,
    cudaStream_t stream);

/// Synchronously scatter GPU rows with a local workspace.
template <typename Scalar>
void scatterRows(
    const DenseMatrix<Scalar, Device::GPU>& values,
    const DenseMatrix<Index, Device::GPU>& indices,
    DenseMatrix<Scalar, Device::GPU>& output);

/// Synchronously scatter GPU rows and check device index status.
template <typename Scalar>
void scatterRows(
    const DenseMatrix<Scalar, Device::GPU>& values,
    const DenseMatrix<Index, Device::GPU>& indices,
    DenseMatrix<Scalar, Device::GPU>& output,
    IndexingWorkspace& workspace,
    cudaStream_t stream = nullptr);

/// Enqueue deterministic scatter; the lowest source row wins duplicate destinations.
template <typename Scalar>
void scatterRowsAsync(
    const DenseMatrix<Scalar, Device::GPU>& values,
    const DenseMatrix<Index, Device::GPU>& indices,
    DenseMatrix<Scalar, Device::GPU>& output,
    IndexingWorkspace& workspace,
    cudaStream_t stream);

/// Synchronously compact GPU rows and return exact-size ordinary GPU matrices.
template <typename Scalar>
CompactRowsResult<Scalar, Device::GPU> compactRows(
    const DenseMatrix<Scalar, Device::GPU>& input,
    const DenseMatrix<std::uint8_t, Device::GPU>& keep_mask);

/// Synchronously compact GPU rows on stream and return exact-size ordinary GPU matrices.
template <typename Scalar>
CompactRowsResult<Scalar, Device::GPU> compactRows(
    const DenseMatrix<Scalar, Device::GPU>& input,
    const DenseMatrix<std::uint8_t, Device::GPU>& keep_mask,
    IndexingWorkspace& workspace,
    cudaStream_t stream = nullptr);

/// Synchronously compact into R-row capacity outputs and a 1 x 1 selected count.
template <typename Scalar>
void compactRows(
    const DenseMatrix<Scalar, Device::GPU>& input,
    const DenseMatrix<std::uint8_t, Device::GPU>& keep_mask,
    DenseMatrix<Scalar, Device::GPU>& capacity_output,
    DenseMatrix<Index, Device::GPU>& capacity_source_indices,
    DenseMatrix<Index, Device::GPU>& selected_count,
    IndexingWorkspace& workspace,
    cudaStream_t stream = nullptr);

/// Enqueue stable compaction into R x C values, R x 1 source indices, and 1 x 1 selected count.
template <typename Scalar>
void compactRowsAsync(
    const DenseMatrix<Scalar, Device::GPU>& input,
    const DenseMatrix<std::uint8_t, Device::GPU>& keep_mask,
    DenseMatrix<Scalar, Device::GPU>& capacity_output,
    DenseMatrix<Index, Device::GPU>& capacity_source_indices,
    DenseMatrix<Index, Device::GPU>& selected_count,
    IndexingWorkspace& workspace,
    cudaStream_t stream);

#else

namespace indexing_detail
{
[[noreturn]] inline void throwNoCuda(const char* operation)
{
    throw std::runtime_error(
        std::string(operation) + " requires PLAMATRIX_WITH_CUDA=ON");
}
}

inline DenseMatrix<Index, Device::GPU> exclusiveScan(
    const DenseMatrix<Index, Device::GPU>&)
{
    indexing_detail::throwNoCuda("exclusiveScan");
}

inline DenseMatrix<Index, Device::GPU> exclusiveScan(
    const DenseMatrix<Index, Device::GPU>&, IndexingWorkspace&, cudaStream_t = nullptr)
{
    indexing_detail::throwNoCuda("exclusiveScan");
}

inline void exclusiveScan(
    const DenseMatrix<Index, Device::GPU>&, DenseMatrix<Index, Device::GPU>&,
    IndexingWorkspace&, cudaStream_t = nullptr)
{
    indexing_detail::throwNoCuda("exclusiveScan");
}

inline DenseMatrix<Index, Device::GPU> exclusiveScanAsync(
    const DenseMatrix<Index, Device::GPU>&, IndexingWorkspace&, cudaStream_t)
{
    indexing_detail::throwNoCuda("exclusiveScanAsync");
}

inline void exclusiveScanAsync(
    const DenseMatrix<Index, Device::GPU>&, DenseMatrix<Index, Device::GPU>&,
    IndexingWorkspace&, cudaStream_t)
{
    indexing_detail::throwNoCuda("exclusiveScanAsync");
}

#define PLAMATRIX_DEFINE_NO_CUDA_INDEXING(Scalar)                                      \
    inline DenseMatrix<Scalar, Device::GPU> gatherRows(                                \
        const DenseMatrix<Scalar, Device::GPU>&,                                       \
        const DenseMatrix<Index, Device::GPU>&)                                        \
    { indexing_detail::throwNoCuda("gatherRows"); }                                   \
    inline DenseMatrix<Scalar, Device::GPU> gatherRows(                                \
        const DenseMatrix<Scalar, Device::GPU>&, const DenseMatrix<Index, Device::GPU>&,\
        IndexingWorkspace&, cudaStream_t = nullptr)                                    \
    { indexing_detail::throwNoCuda("gatherRows"); }                                   \
    inline void gatherRows(                                                            \
        const DenseMatrix<Scalar, Device::GPU>&, const DenseMatrix<Index, Device::GPU>&,\
        DenseMatrix<Scalar, Device::GPU>&, IndexingWorkspace&, cudaStream_t = nullptr) \
    { indexing_detail::throwNoCuda("gatherRows"); }                                   \
    inline DenseMatrix<Scalar, Device::GPU> gatherRowsAsync(                           \
        const DenseMatrix<Scalar, Device::GPU>&, const DenseMatrix<Index, Device::GPU>&,\
        IndexingWorkspace&, cudaStream_t)                                               \
    { indexing_detail::throwNoCuda("gatherRowsAsync"); }                              \
    inline void gatherRowsAsync(                                                       \
        const DenseMatrix<Scalar, Device::GPU>&, const DenseMatrix<Index, Device::GPU>&,\
        DenseMatrix<Scalar, Device::GPU>&, IndexingWorkspace&, cudaStream_t)            \
    { indexing_detail::throwNoCuda("gatherRowsAsync"); }                              \
    inline void scatterRows(                                                           \
        const DenseMatrix<Scalar, Device::GPU>&, const DenseMatrix<Index, Device::GPU>&,\
        DenseMatrix<Scalar, Device::GPU>&)                                              \
    { indexing_detail::throwNoCuda("scatterRows"); }                                  \
    inline void scatterRows(                                                           \
        const DenseMatrix<Scalar, Device::GPU>&, const DenseMatrix<Index, Device::GPU>&,\
        DenseMatrix<Scalar, Device::GPU>&, IndexingWorkspace&, cudaStream_t = nullptr) \
    { indexing_detail::throwNoCuda("scatterRows"); }                                  \
    inline void scatterRowsAsync(                                                      \
        const DenseMatrix<Scalar, Device::GPU>&, const DenseMatrix<Index, Device::GPU>&,\
        DenseMatrix<Scalar, Device::GPU>&, IndexingWorkspace&, cudaStream_t)            \
    { indexing_detail::throwNoCuda("scatterRowsAsync"); }                             \
    inline CompactRowsResult<Scalar, Device::GPU> compactRows(                         \
        const DenseMatrix<Scalar, Device::GPU>&,                                       \
        const DenseMatrix<std::uint8_t, Device::GPU>&)                                 \
    { indexing_detail::throwNoCuda("compactRows"); }                                  \
    inline CompactRowsResult<Scalar, Device::GPU> compactRows(                         \
        const DenseMatrix<Scalar, Device::GPU>&,                                       \
        const DenseMatrix<std::uint8_t, Device::GPU>&, IndexingWorkspace&,             \
        cudaStream_t = nullptr)                                                        \
    { indexing_detail::throwNoCuda("compactRows"); }                                  \
    inline void compactRows(                                                           \
        const DenseMatrix<Scalar, Device::GPU>&,                                       \
        const DenseMatrix<std::uint8_t, Device::GPU>&,                                 \
        DenseMatrix<Scalar, Device::GPU>&, DenseMatrix<Index, Device::GPU>&,           \
        DenseMatrix<Index, Device::GPU>&, IndexingWorkspace&, cudaStream_t = nullptr)  \
    { indexing_detail::throwNoCuda("compactRows"); }                                  \
    inline void compactRowsAsync(                                                      \
        const DenseMatrix<Scalar, Device::GPU>&,                                       \
        const DenseMatrix<std::uint8_t, Device::GPU>&,                                 \
        DenseMatrix<Scalar, Device::GPU>&, DenseMatrix<Index, Device::GPU>&,           \
        DenseMatrix<Index, Device::GPU>&, IndexingWorkspace&, cudaStream_t)            \
    { indexing_detail::throwNoCuda("compactRowsAsync"); }

#ifdef PLAMATRIX_USE_FLOAT
PLAMATRIX_DEFINE_NO_CUDA_INDEXING(float)
#endif
#ifdef PLAMATRIX_USE_DOUBLE
PLAMATRIX_DEFINE_NO_CUDA_INDEXING(double)
#endif

#undef PLAMATRIX_DEFINE_NO_CUDA_INDEXING

#endif

} // namespace plamatrix
