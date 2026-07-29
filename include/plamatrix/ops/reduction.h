#pragma once

#include <cstddef>
#include <stdexcept>

#include "plamatrix/dense/dense_matrix.h"

namespace plamatrix
{

/// Reduction shape: All -> 1x1; Rows -> rows x 1 by reducing columns;
/// Columns -> 1 x columns by reducing rows.
/// CPU overloads support the enabled float and double explicit instantiations.
enum class ReductionAxis
{
    All,
    Rows,
    Columns
};

/// Values and source offsets returned by an indexed reduction.
template <typename Scalar, Device Dev>
struct IndexedReductionResult
{
    DenseMatrix<Scalar, Dev> values;
    DenseMatrix<Index, Dev> indices;
};

/// Caller-owned temporary storage reused by GPU reductions. It is move-only, grow-only, and not
/// thread-safe. Enqueued operations may reuse it sequentially on one stream without synchronizing.
/// A different stream is rejected until the caller synchronizes and explicitly resets an ordinary
/// allocation with reserveBytes(), or closes a stream-ordered allocation.
///
/// Synchronous reservation uses ordinary GPU allocation and resets that allocation's stream reuse
/// binding. Asynchronous reservation on an empty workspace uses stream-ordered allocation. Such an
/// allocation may grow only on its owning stream and must be closed before synchronous reservation
/// or use on another stream. Moving transfers storage and stream provenance. Destruction is a
/// noexcept release fallback on the retained stream.
class ReductionWorkspace
{
public:
    ReductionWorkspace() noexcept = default;

#ifdef PLAMATRIX_WITH_CUDA
    ~ReductionWorkspace() noexcept;
    ReductionWorkspace(ReductionWorkspace&& other) noexcept;
    ReductionWorkspace& operator=(ReductionWorkspace&& other) noexcept;

    void reserveBytes(std::size_t bytes);
    void reserveBytesAsync(std::size_t bytes, cudaStream_t stream);
    void closeAsyncAllocation();
#else
    ~ReductionWorkspace() noexcept = default;
    ReductionWorkspace(ReductionWorkspace&& other) noexcept = default;
    ReductionWorkspace& operator=(ReductionWorkspace&& other) noexcept = default;

    void reserveBytes(std::size_t)
    {
        throw std::runtime_error(
            "ReductionWorkspace::reserveBytes requires PLAMATRIX_WITH_CUDA=ON");
    }

    void reserveBytesAsync(std::size_t, cudaStream_t)
    {
        throw std::runtime_error(
            "ReductionWorkspace::reserveBytesAsync requires PLAMATRIX_WITH_CUDA=ON");
    }

    void closeAsyncAllocation()
    {
    }
#endif

    ReductionWorkspace(const ReductionWorkspace&) = delete;
    ReductionWorkspace& operator=(const ReductionWorkspace&) = delete;

    std::size_t capacityBytes() const noexcept
    {
        return _capacityBytes;
    }

    void* data() noexcept
    {
        return _data;
    }

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

    std::size_t _capacityBytes = 0;
    void* _data = nullptr;
    AllocationKind _allocationKind = AllocationKind::Normal;
    cudaStream_t _allocationStream = nullptr;
    cudaStream_t _reuseStream = nullptr;
    bool _hasReuseStream = false;
};

/// Sum each CPU reduction lane. Float inputs accumulate in double; empty lanes produce zero.
/// @throws std::invalid_argument if axis is invalid
template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> sum(
    const DenseMatrix<Scalar, Device::CPU>& input,
    ReductionAxis axis);

/// Compute a deterministic, overflow-resistant arithmetic mean of each CPU reduction lane.
/// NaNs propagate; mixed positive and negative infinity produces NaN.
/// @throws std::invalid_argument if a non-empty output contains an empty lane or axis is invalid
template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> mean(
    const DenseMatrix<Scalar, Device::CPU>& input,
    ReductionAxis axis);

/// Select the minimum value in each CPU reduction lane.
/// NaNs propagate and ties select the lowest source offset.
/// @throws std::invalid_argument if a non-empty output contains an empty lane or axis is invalid
template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> min(
    const DenseMatrix<Scalar, Device::CPU>& input,
    ReductionAxis axis);

/// Select the maximum value in each CPU reduction lane.
/// NaNs propagate and ties select the lowest source offset.
/// @throws std::invalid_argument if a non-empty output contains an empty lane or axis is invalid
template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> max(
    const DenseMatrix<Scalar, Device::CPU>& input,
    ReductionAxis axis);

/// Return minimum values and offsets along the reduced dimension.
/// For All, offsets are column-major linear source indices.
/// @throws std::invalid_argument if a non-empty output contains an empty lane or axis is invalid
template <typename Scalar>
IndexedReductionResult<Scalar, Device::CPU> argMin(
    const DenseMatrix<Scalar, Device::CPU>& input,
    ReductionAxis axis);

/// Return maximum values and offsets along the reduced dimension.
/// For All, offsets are column-major linear source indices.
/// @throws std::invalid_argument if a non-empty output contains an empty lane or axis is invalid
template <typename Scalar>
IndexedReductionResult<Scalar, Device::CPU> argMax(
    const DenseMatrix<Scalar, Device::CPU>& input,
    ReductionAxis axis);

/// GPU value reductions. Synchronous forms wait for stream completion. Async forms enqueue work
/// and require input, output, and workspace storage to remain alive until stream completion.
/// Allocating async results use stream-ordered storage and must be explicitly closed before the
/// stream is destroyed when release errors need to be reported.
template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> sum(
    const DenseMatrix<Scalar, Device::GPU>& input,
    ReductionAxis axis);

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> sum(
    const DenseMatrix<Scalar, Device::GPU>& input,
    ReductionAxis axis,
    ReductionWorkspace& workspace,
    cudaStream_t stream = nullptr);

template <typename Scalar>
void sum(
    const DenseMatrix<Scalar, Device::GPU>& input,
    ReductionAxis axis,
    DenseMatrix<Scalar, Device::GPU>& output,
    ReductionWorkspace& workspace,
    cudaStream_t stream = nullptr);

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> sumAsync(
    const DenseMatrix<Scalar, Device::GPU>& input,
    ReductionAxis axis,
    ReductionWorkspace& workspace,
    cudaStream_t stream);

template <typename Scalar>
void sumAsync(
    const DenseMatrix<Scalar, Device::GPU>& input,
    ReductionAxis axis,
    DenseMatrix<Scalar, Device::GPU>& output,
    ReductionWorkspace& workspace,
    cudaStream_t stream);

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> mean(
    const DenseMatrix<Scalar, Device::GPU>& input,
    ReductionAxis axis);

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> mean(
    const DenseMatrix<Scalar, Device::GPU>& input,
    ReductionAxis axis,
    ReductionWorkspace& workspace,
    cudaStream_t stream = nullptr);

template <typename Scalar>
void mean(
    const DenseMatrix<Scalar, Device::GPU>& input,
    ReductionAxis axis,
    DenseMatrix<Scalar, Device::GPU>& output,
    ReductionWorkspace& workspace,
    cudaStream_t stream = nullptr);

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> meanAsync(
    const DenseMatrix<Scalar, Device::GPU>& input,
    ReductionAxis axis,
    ReductionWorkspace& workspace,
    cudaStream_t stream);

template <typename Scalar>
void meanAsync(
    const DenseMatrix<Scalar, Device::GPU>& input,
    ReductionAxis axis,
    DenseMatrix<Scalar, Device::GPU>& output,
    ReductionWorkspace& workspace,
    cudaStream_t stream);

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> min(
    const DenseMatrix<Scalar, Device::GPU>& input,
    ReductionAxis axis);

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> min(
    const DenseMatrix<Scalar, Device::GPU>& input,
    ReductionAxis axis,
    ReductionWorkspace& workspace,
    cudaStream_t stream = nullptr);

template <typename Scalar>
void min(
    const DenseMatrix<Scalar, Device::GPU>& input,
    ReductionAxis axis,
    DenseMatrix<Scalar, Device::GPU>& output,
    ReductionWorkspace& workspace,
    cudaStream_t stream = nullptr);

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> minAsync(
    const DenseMatrix<Scalar, Device::GPU>& input,
    ReductionAxis axis,
    ReductionWorkspace& workspace,
    cudaStream_t stream);

template <typename Scalar>
void minAsync(
    const DenseMatrix<Scalar, Device::GPU>& input,
    ReductionAxis axis,
    DenseMatrix<Scalar, Device::GPU>& output,
    ReductionWorkspace& workspace,
    cudaStream_t stream);

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> max(
    const DenseMatrix<Scalar, Device::GPU>& input,
    ReductionAxis axis);

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> max(
    const DenseMatrix<Scalar, Device::GPU>& input,
    ReductionAxis axis,
    ReductionWorkspace& workspace,
    cudaStream_t stream = nullptr);

template <typename Scalar>
void max(
    const DenseMatrix<Scalar, Device::GPU>& input,
    ReductionAxis axis,
    DenseMatrix<Scalar, Device::GPU>& output,
    ReductionWorkspace& workspace,
    cudaStream_t stream = nullptr);

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> maxAsync(
    const DenseMatrix<Scalar, Device::GPU>& input,
    ReductionAxis axis,
    ReductionWorkspace& workspace,
    cudaStream_t stream);

template <typename Scalar>
void maxAsync(
    const DenseMatrix<Scalar, Device::GPU>& input,
    ReductionAxis axis,
    DenseMatrix<Scalar, Device::GPU>& output,
    ReductionWorkspace& workspace,
    cudaStream_t stream);

/// GPU indexed reductions. Indices are offsets within the reduced dimension; All uses the
/// column-major linear source offset. NaNs propagate and ties select the lowest source offset.
template <typename Scalar>
IndexedReductionResult<Scalar, Device::GPU> argMin(
    const DenseMatrix<Scalar, Device::GPU>& input,
    ReductionAxis axis);

template <typename Scalar>
IndexedReductionResult<Scalar, Device::GPU> argMin(
    const DenseMatrix<Scalar, Device::GPU>& input,
    ReductionAxis axis,
    ReductionWorkspace& workspace,
    cudaStream_t stream = nullptr);

template <typename Scalar>
void argMin(
    const DenseMatrix<Scalar, Device::GPU>& input,
    ReductionAxis axis,
    DenseMatrix<Scalar, Device::GPU>& values,
    DenseMatrix<Index, Device::GPU>& indices,
    ReductionWorkspace& workspace,
    cudaStream_t stream = nullptr);

template <typename Scalar>
IndexedReductionResult<Scalar, Device::GPU> argMinAsync(
    const DenseMatrix<Scalar, Device::GPU>& input,
    ReductionAxis axis,
    ReductionWorkspace& workspace,
    cudaStream_t stream);

template <typename Scalar>
void argMinAsync(
    const DenseMatrix<Scalar, Device::GPU>& input,
    ReductionAxis axis,
    DenseMatrix<Scalar, Device::GPU>& values,
    DenseMatrix<Index, Device::GPU>& indices,
    ReductionWorkspace& workspace,
    cudaStream_t stream);

template <typename Scalar>
IndexedReductionResult<Scalar, Device::GPU> argMax(
    const DenseMatrix<Scalar, Device::GPU>& input,
    ReductionAxis axis);

template <typename Scalar>
IndexedReductionResult<Scalar, Device::GPU> argMax(
    const DenseMatrix<Scalar, Device::GPU>& input,
    ReductionAxis axis,
    ReductionWorkspace& workspace,
    cudaStream_t stream = nullptr);

template <typename Scalar>
void argMax(
    const DenseMatrix<Scalar, Device::GPU>& input,
    ReductionAxis axis,
    DenseMatrix<Scalar, Device::GPU>& values,
    DenseMatrix<Index, Device::GPU>& indices,
    ReductionWorkspace& workspace,
    cudaStream_t stream = nullptr);

template <typename Scalar>
IndexedReductionResult<Scalar, Device::GPU> argMaxAsync(
    const DenseMatrix<Scalar, Device::GPU>& input,
    ReductionAxis axis,
    ReductionWorkspace& workspace,
    cudaStream_t stream);

template <typename Scalar>
void argMaxAsync(
    const DenseMatrix<Scalar, Device::GPU>& input,
    ReductionAxis axis,
    DenseMatrix<Scalar, Device::GPU>& values,
    DenseMatrix<Index, Device::GPU>& indices,
    ReductionWorkspace& workspace,
    cudaStream_t stream);

#ifdef PLAMATRIX_NO_CUDA
namespace detail
{

[[noreturn]] inline void throwReductionNoCuda(const char* message)
{
    throw std::runtime_error(message);
}

} // namespace detail

#define PLAMATRIX_NO_CUDA_VALUE_REDUCTION(OP, ASYNC_OP)                                  \
    template <typename Scalar>                                                           \
    DenseMatrix<Scalar, Device::GPU> OP(                                                 \
        const DenseMatrix<Scalar, Device::GPU>&, ReductionAxis)                          \
    {                                                                                    \
        detail::throwReductionNoCuda(#OP " requires PLAMATRIX_WITH_CUDA=ON");            \
    }                                                                                    \
    template <typename Scalar>                                                           \
    DenseMatrix<Scalar, Device::GPU> OP(                                                 \
        const DenseMatrix<Scalar, Device::GPU>&, ReductionAxis,                          \
        ReductionWorkspace&, cudaStream_t)                                               \
    {                                                                                    \
        detail::throwReductionNoCuda(#OP " requires PLAMATRIX_WITH_CUDA=ON");            \
    }                                                                                    \
    template <typename Scalar>                                                           \
    void OP(const DenseMatrix<Scalar, Device::GPU>&, ReductionAxis,                      \
            DenseMatrix<Scalar, Device::GPU>&, ReductionWorkspace&, cudaStream_t)        \
    {                                                                                    \
        detail::throwReductionNoCuda(#OP " requires PLAMATRIX_WITH_CUDA=ON");            \
    }                                                                                    \
    template <typename Scalar>                                                           \
    DenseMatrix<Scalar, Device::GPU> ASYNC_OP(                                           \
        const DenseMatrix<Scalar, Device::GPU>&, ReductionAxis,                          \
        ReductionWorkspace&, cudaStream_t)                                               \
    {                                                                                    \
        detail::throwReductionNoCuda(#ASYNC_OP " requires PLAMATRIX_WITH_CUDA=ON");      \
    }                                                                                    \
    template <typename Scalar>                                                           \
    void ASYNC_OP(const DenseMatrix<Scalar, Device::GPU>&, ReductionAxis,                \
                  DenseMatrix<Scalar, Device::GPU>&, ReductionWorkspace&, cudaStream_t)  \
    {                                                                                    \
        detail::throwReductionNoCuda(#ASYNC_OP " requires PLAMATRIX_WITH_CUDA=ON");      \
    }

PLAMATRIX_NO_CUDA_VALUE_REDUCTION(sum, sumAsync)
PLAMATRIX_NO_CUDA_VALUE_REDUCTION(mean, meanAsync)
PLAMATRIX_NO_CUDA_VALUE_REDUCTION(min, minAsync)
PLAMATRIX_NO_CUDA_VALUE_REDUCTION(max, maxAsync)

#undef PLAMATRIX_NO_CUDA_VALUE_REDUCTION

#define PLAMATRIX_NO_CUDA_INDEXED_REDUCTION(OP, ASYNC_OP)                                \
    template <typename Scalar>                                                           \
    IndexedReductionResult<Scalar, Device::GPU> OP(                                      \
        const DenseMatrix<Scalar, Device::GPU>&, ReductionAxis)                          \
    {                                                                                    \
        detail::throwReductionNoCuda(#OP " requires PLAMATRIX_WITH_CUDA=ON");            \
    }                                                                                    \
    template <typename Scalar>                                                           \
    IndexedReductionResult<Scalar, Device::GPU> OP(                                      \
        const DenseMatrix<Scalar, Device::GPU>&, ReductionAxis,                          \
        ReductionWorkspace&, cudaStream_t)                                               \
    {                                                                                    \
        detail::throwReductionNoCuda(#OP " requires PLAMATRIX_WITH_CUDA=ON");            \
    }                                                                                    \
    template <typename Scalar>                                                           \
    void OP(const DenseMatrix<Scalar, Device::GPU>&, ReductionAxis,                      \
            DenseMatrix<Scalar, Device::GPU>&, DenseMatrix<Index, Device::GPU>&,         \
            ReductionWorkspace&, cudaStream_t)                                           \
    {                                                                                    \
        detail::throwReductionNoCuda(#OP " requires PLAMATRIX_WITH_CUDA=ON");            \
    }                                                                                    \
    template <typename Scalar>                                                           \
    IndexedReductionResult<Scalar, Device::GPU> ASYNC_OP(                                \
        const DenseMatrix<Scalar, Device::GPU>&, ReductionAxis,                          \
        ReductionWorkspace&, cudaStream_t)                                               \
    {                                                                                    \
        detail::throwReductionNoCuda(#ASYNC_OP " requires PLAMATRIX_WITH_CUDA=ON");      \
    }                                                                                    \
    template <typename Scalar>                                                           \
    void ASYNC_OP(const DenseMatrix<Scalar, Device::GPU>&, ReductionAxis,                \
                  DenseMatrix<Scalar, Device::GPU>&, DenseMatrix<Index, Device::GPU>&,   \
                  ReductionWorkspace&, cudaStream_t)                                     \
    {                                                                                    \
        detail::throwReductionNoCuda(#ASYNC_OP " requires PLAMATRIX_WITH_CUDA=ON");      \
    }

PLAMATRIX_NO_CUDA_INDEXED_REDUCTION(argMin, argMinAsync)
PLAMATRIX_NO_CUDA_INDEXED_REDUCTION(argMax, argMaxAsync)

#undef PLAMATRIX_NO_CUDA_INDEXED_REDUCTION
#endif

} // namespace plamatrix
