#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "plamatrix/dense/dense_matrix.h"

namespace plamatrix
{

/**
 * @brief Solve a fixed-size linear system with partial pivoting.
 *
 * The matrix is row-major and all workspace remains on the stack.
 * Returns false for non-finite input or a relatively singular system.
 */
template <typename Scalar, std::size_t N>
bool solveSmallLinearSystem(
    std::array<Scalar, N * N> matrix,
    std::array<Scalar, N> rhs,
    std::array<Scalar, N> *solution,
    Scalar relative_singular_tolerance =
        std::numeric_limits<Scalar>::epsilon() * Scalar(16 * N))
{
    static_assert(N > 0, "solveSmallLinearSystem requires N > 0");
    static_assert(std::is_floating_point_v<Scalar>,
                  "solveSmallLinearSystem requires a floating-point scalar");
    if (!solution ||
        !std::isfinite(relative_singular_tolerance) ||
        relative_singular_tolerance < Scalar(0))
    {
        return false;
    }

    for (const Scalar value : matrix)
    {
        if (!std::isfinite(value))
        {
            return false;
        }
    }
    for (const Scalar value : rhs)
    {
        if (!std::isfinite(value))
        {
            return false;
        }
    }
    // BA normal equations can mix rotation, translation, and intrinsics
    // with very different scales. Row equilibration avoids false singularity.
    for (std::size_t row = 0; row < N; ++row)
    {
        Scalar row_scale = Scalar(0);
        for (std::size_t column = 0; column < N; ++column)
        {
            row_scale = std::max(
                row_scale,
                std::abs(matrix[row * N + column]));
        }
        if (!(row_scale > Scalar(0)) || !std::isfinite(row_scale))
        {
            return false;
        }
        for (std::size_t column = 0; column < N; ++column)
        {
            matrix[row * N + column] /= row_scale;
        }
        rhs[row] /= row_scale;
        if (!std::isfinite(rhs[row]))
        {
            return false;
        }
    }
    const Scalar singular_threshold = relative_singular_tolerance;

    for (std::size_t column = 0; column < N; ++column)
    {
        std::size_t pivot_row = column;
        Scalar pivot_magnitude =
            std::abs(matrix[column * N + column]);
        for (std::size_t row = column + 1; row < N; ++row)
        {
            const Scalar candidate =
                std::abs(matrix[row * N + column]);
            if (candidate > pivot_magnitude)
            {
                pivot_magnitude = candidate;
                pivot_row = row;
            }
        }
        if (!(pivot_magnitude > singular_threshold))
        {
            return false;
        }

        if (pivot_row != column)
        {
            for (std::size_t entry = column; entry < N; ++entry)
            {
                std::swap(matrix[column * N + entry],
                          matrix[pivot_row * N + entry]);
            }
            std::swap(rhs[column], rhs[pivot_row]);
        }

        const Scalar pivot = matrix[column * N + column];
        for (std::size_t row = column + 1; row < N; ++row)
        {
            const Scalar factor = matrix[row * N + column] / pivot;
            matrix[row * N + column] = Scalar(0);
            for (std::size_t entry = column + 1; entry < N; ++entry)
            {
                matrix[row * N + entry] -=
                    factor * matrix[column * N + entry];
            }
            rhs[row] -= factor * rhs[column];
        }
    }

    solution->fill(Scalar(0));
    for (std::size_t offset = 0; offset < N; ++offset)
    {
        const std::size_t row = N - 1 - offset;
        Scalar value = rhs[row];
        for (std::size_t column = row + 1; column < N; ++column)
        {
            value -= matrix[row * N + column] * (*solution)[column];
        }
        const Scalar diagonal = matrix[row * N + row];
        if (!(std::abs(diagonal) > singular_threshold))
        {
            return false;
        }
        (*solution)[row] = value / diagonal;
        if (!std::isfinite((*solution)[row]))
        {
            return false;
        }
    }
    return true;
}

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

/// Allocation-free symmetric 3x3 eigendecomposition for one compact matrix.
/// Input is `[xx, xy, xz, yy, yz, zz]`; eigenvalues are ascending and eigenvectors are packed
/// column-major with the same deterministic sign/repeated-space convention as the batched API.
template <typename Scalar>
void symmetricEigh3x3(const std::array<Scalar, 6>& compact_matrix,
                      std::array<Scalar, 3>* eigenvalues,
                      std::array<Scalar, 9>* eigenvectors);

/// Allocation-free full SVD of one row-major 3x3 matrix.
/// Output satisfies `matrix = U * diag(singularValues) * Vt`; singular values are descending.
template <typename Scalar>
void svd3x3(const std::array<Scalar, 9>& matrix,
            std::array<Scalar, 9>* u,
            std::array<Scalar, 3>* singular_values,
            std::array<Scalar, 9>* vt);

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
