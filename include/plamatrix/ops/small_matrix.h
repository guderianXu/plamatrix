#pragma once

#include <cstddef>
#include <stdexcept>

#include "plamatrix/dense/dense_matrix.h"

namespace plamatrix
{

/// Result of a batched symmetric 3x3 eigendecomposition.
/// Each row of eigenvalues contains the three eigenvalues in ascending order.
/// Each row of eigenvectors packs the corresponding unit eigenvectors column-major as
/// [v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z].
template <typename Scalar, Device Dev>
struct SymmetricEigh3x3Result
{
    DenseMatrix<Scalar, Dev> eigenvalues;
    DenseMatrix<Scalar, Dev> eigenvectors;
};

namespace small_matrix_detail
{
struct SymmetricEigh3x3WorkspaceAccess;
}

/// Caller-owned status storage for asynchronous GPU 3x3 eigendecompositions.
///
/// The workspace is move-only, grow-only, and not thread-safe. Sequential async calls may reuse
/// it on one stream. Reuse on another stream is rejected until the owning stream is synchronized,
/// status is consumed, and an ordinary allocation is reset with reserveBytes(), or a
/// stream-ordered allocation is closed. Async callers must synchronize and call checkStatus().
class SymmetricEigh3x3Workspace
{
public:
    SymmetricEigh3x3Workspace() noexcept = default;

#ifdef PLAMATRIX_WITH_CUDA
    ~SymmetricEigh3x3Workspace() noexcept;
    SymmetricEigh3x3Workspace(SymmetricEigh3x3Workspace&& other) noexcept;
    SymmetricEigh3x3Workspace& operator=(SymmetricEigh3x3Workspace&& other);

    void reserveBytes(std::size_t bytes);
    void reserveBytesAsync(std::size_t bytes, cudaStream_t stream);
    void closeAsyncAllocation();
    void checkStatus(const char* operation);
#else
    ~SymmetricEigh3x3Workspace() noexcept = default;
    SymmetricEigh3x3Workspace(SymmetricEigh3x3Workspace&& other) noexcept = default;
    SymmetricEigh3x3Workspace& operator=(SymmetricEigh3x3Workspace&& other) noexcept = default;

    void reserveBytes(std::size_t)
    {
        throw std::runtime_error(
            "SymmetricEigh3x3Workspace::reserveBytes requires PLAMATRIX_WITH_CUDA=ON");
    }

    void reserveBytesAsync(std::size_t, cudaStream_t)
    {
        throw std::runtime_error(
            "SymmetricEigh3x3Workspace::reserveBytesAsync requires PLAMATRIX_WITH_CUDA=ON");
    }

    void closeAsyncAllocation()
    {
    }

    void checkStatus(const char*)
    {
        throw std::runtime_error(
            "SymmetricEigh3x3Workspace::checkStatus requires PLAMATRIX_WITH_CUDA=ON");
    }
#endif

    SymmetricEigh3x3Workspace(const SymmetricEigh3x3Workspace&) = delete;
    SymmetricEigh3x3Workspace& operator=(const SymmetricEigh3x3Workspace&) = delete;

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

    friend struct small_matrix_detail::SymmetricEigh3x3WorkspaceAccess;

    std::size_t _capacityBytes = 0;
    void* _data = nullptr;
    AllocationKind _allocationKind = AllocationKind::Normal;
    cudaStream_t _allocationStream = nullptr;
    cudaStream_t _reuseStream = nullptr;
    bool _hasReuseStream = false;
    bool _hasStatusBatch = false;
};

/// Decompose a CPU batch of compact symmetric 3x3 matrices.
/// Input must have shape N x 6. Each row is [xx, xy, xz, yy, yz, zz]. The complete input is
/// validated before any output is allocated or any decomposition is computed. Empty N=0 batches
/// return eigenvalues with shape 0 x 3 and eigenvectors with shape 0 x 9.
/// Eigenvalues are ascending. Eigenvectors use the packed layout documented above. Every vector is
/// normalized and signed so its largest-absolute component is nonnegative; ties use the lowest
/// component index. Repeated eigenspaces use a deterministic fixed-axis projection basis.
/// @throws std::invalid_argument if the input does not have six columns or contains NaN or infinity.
template <typename Scalar>
SymmetricEigh3x3Result<Scalar, Device::CPU> symmetricEigh3x3Batched(
    const DenseMatrix<Scalar, Device::CPU>& compact_matrices);

#ifdef PLAMATRIX_WITH_CUDA

/// Synchronously decompose a GPU batch using ordinary output allocation.
template <typename Scalar>
SymmetricEigh3x3Result<Scalar, Device::GPU> symmetricEigh3x3Batched(
    const DenseMatrix<Scalar, Device::GPU>& compact_matrices);

/// Synchronously decompose into caller-owned N x 3 and N x 9 outputs.
template <typename Scalar>
void symmetricEigh3x3Batched(
    const DenseMatrix<Scalar, Device::GPU>& compact_matrices,
    DenseMatrix<Scalar, Device::GPU>& eigenvalues,
    DenseMatrix<Scalar, Device::GPU>& eigenvectors,
    SymmetricEigh3x3Workspace& workspace,
    cudaStream_t stream = nullptr);

/// Enqueue decomposition into caller-owned outputs without host synchronization.
/// Shape errors are reported before any workspace reservation or kernel launch. Device-detected
/// non-finite rows are zeroed and aggregated; synchronize stream and call workspace.checkStatus().
template <typename Scalar>
void symmetricEigh3x3BatchedAsync(
    const DenseMatrix<Scalar, Device::GPU>& compact_matrices,
    DenseMatrix<Scalar, Device::GPU>& eigenvalues,
    DenseMatrix<Scalar, Device::GPU>& eigenvectors,
    SymmetricEigh3x3Workspace& workspace,
    cudaStream_t stream);

#else

template <typename Scalar>
SymmetricEigh3x3Result<Scalar, Device::GPU> symmetricEigh3x3Batched(
    const DenseMatrix<Scalar, Device::GPU>&)
{
    throw std::runtime_error(
        "symmetricEigh3x3Batched GPU overload requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
void symmetricEigh3x3Batched(
    const DenseMatrix<Scalar, Device::GPU>&,
    DenseMatrix<Scalar, Device::GPU>&,
    DenseMatrix<Scalar, Device::GPU>&,
    SymmetricEigh3x3Workspace&,
    cudaStream_t = nullptr)
{
    throw std::runtime_error(
        "symmetricEigh3x3Batched GPU overload requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
void symmetricEigh3x3BatchedAsync(
    const DenseMatrix<Scalar, Device::GPU>&,
    DenseMatrix<Scalar, Device::GPU>&,
    DenseMatrix<Scalar, Device::GPU>&,
    SymmetricEigh3x3Workspace&,
    cudaStream_t)
{
    throw std::runtime_error(
        "symmetricEigh3x3BatchedAsync requires PLAMATRIX_WITH_CUDA=ON");
}

#endif

} // namespace plamatrix
