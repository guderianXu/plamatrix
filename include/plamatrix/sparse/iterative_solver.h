#pragma once

#include "plamatrix/dense/dense_matrix.h"
#include "plamatrix/sparse/csr_matrix.h"
#include "plamatrix/sparse/sparse_ops.h"

namespace plamatrix
{

struct IterativeSolverOptions
{
    int maxIterations = 1000;
    /// Relative residual tolerance in [0, 1].
    double relativeTolerance = 1.0e-6;
    double absoluteTolerance = 0.0;
    bool useJacobiPreconditioner = true;
    bool requireConvergence = false;
};

struct IterativeSolverReport
{
    bool converged = false;
    int iterations = 0;
    double initialResidual = 0.0;
    double finalResidual = 0.0;
};

/// Move-only reusable storage for CUDA CG/PCG solves.
/// The bound CUDA stream must remain alive until closeAsyncAllocation() or destruction completes.
/// closeAsyncAllocation() enqueues stream-ordered frees; synchronize once more before destroying a
/// non-default stream.
template <typename Scalar>
class IterativeSolverWorkspace
{
public:
    IterativeSolverWorkspace() noexcept = default;
    ~IterativeSolverWorkspace() noexcept;
    IterativeSolverWorkspace(IterativeSolverWorkspace&& other) noexcept;
    IterativeSolverWorkspace& operator=(IterativeSolverWorkspace&& other) noexcept;

    IterativeSolverWorkspace(const IterativeSolverWorkspace&) = delete;
    IterativeSolverWorkspace& operator=(const IterativeSolverWorkspace&) = delete;

    /// Release stream-ordered storage after the owning stream has completed.
    /// The owning stream must remain valid through this call and the subsequent free completion.
    void closeAsyncAllocation();
    Index capacitySize() const noexcept { return _capacitySize; }

private:
    friend struct IterativeSolverWorkspaceAccess;

    DenseMatrix<Scalar, Device::GPU> _residual;
    DenseMatrix<Scalar, Device::GPU> _direction;
    DenseMatrix<Scalar, Device::GPU> _transformed;
    DenseMatrix<Scalar, Device::GPU> _matrixDirection;
    DenseMatrix<Scalar, Device::GPU> _inverseDiagonal;
    DenseMatrix<Scalar, Device::GPU> _scalars;
    SparseOpsWorkspace _sparseWorkspace;
    Index _capacitySize = 0;
    void* _blasHandle = nullptr;
    cudaStream_t _reuseStream = nullptr;
    bool _hasReuseStream = false;
};

/// Device residuals and completion boundary returned by a fixed-iteration asynchronous solve.
/// Its stream must remain alive until closeAsyncAllocation() or destruction completes; synchronize
/// after closeAsyncAllocation() before destroying a non-default stream.
struct AsyncIterativeSolverState
{
    AsyncIterativeSolverState() noexcept = default;
    ~AsyncIterativeSolverState() noexcept;
    AsyncIterativeSolverState(AsyncIterativeSolverState&& other) noexcept;
    AsyncIterativeSolverState& operator=(AsyncIterativeSolverState&& other) noexcept;

    AsyncIterativeSolverState(const AsyncIterativeSolverState&) = delete;
    AsyncIterativeSolverState& operator=(const AsyncIterativeSolverState&) = delete;

    void closeAsyncAllocation();

    int submittedIterations = 0;
    DenseMatrix<double, Device::GPU> initialResidualSquared;
    DenseMatrix<double, Device::GPU> finalResidualSquared;

private:
    friend struct IterativeSolverStateAccess;
    friend IterativeSolverReport finalizeIterativeSolverReport(
        const AsyncIterativeSolverState&, const IterativeSolverOptions&);

    void* _completionEvent = nullptr;
    cudaStream_t _stream = nullptr;
};

/// Solve an SPD CPU system with conjugate gradients, using solution as the initial guess.
template <typename Scalar>
IterativeSolverReport cg(
    const CSRMatrix<Scalar, Device::CPU>& matrix,
    const DenseMatrix<Scalar, Device::CPU>& rhs,
    DenseMatrix<Scalar, Device::CPU>& solution,
    const IterativeSolverOptions& options = {});

/// Solve an SPD CPU system with optional Jacobi-preconditioned conjugate gradients.
template <typename Scalar>
IterativeSolverReport pcg(
    const CSRMatrix<Scalar, Device::CPU>& matrix,
    const DenseMatrix<Scalar, Device::CPU>& rhs,
    DenseMatrix<Scalar, Device::CPU>& solution,
    const IterativeSolverOptions& options = {});

/// Adaptively solve an SPD GPU system with conjugate gradients.
/// Unknown CSR structure is validated once on stream before cuSPARSE is launched.
template <typename Scalar>
IterativeSolverReport cg(
    const CSRMatrix<Scalar, Device::GPU>& matrix,
    const DenseMatrix<Scalar, Device::GPU>& rhs,
    DenseMatrix<Scalar, Device::GPU>& solution,
    IterativeSolverWorkspace<Scalar>& workspace,
    const IterativeSolverOptions& options = {},
    cudaStream_t stream = nullptr);

/// Adaptively solve an SPD GPU system with optional Jacobi preconditioning.
template <typename Scalar>
IterativeSolverReport pcg(
    const CSRMatrix<Scalar, Device::GPU>& matrix,
    const DenseMatrix<Scalar, Device::GPU>& rhs,
    DenseMatrix<Scalar, Device::GPU>& solution,
    IterativeSolverWorkspace<Scalar>& workspace,
    const IterativeSolverOptions& options = {},
    cudaStream_t stream = nullptr);

/// Submit exactly iterations CG steps without host convergence checks.
/// The matrix must have trusted CSR structure. Synchronous CPU-to-GPU transfer establishes trust.
/// Async transfer requires validateStructure() on its copy stream before submission. Once mutable
/// values(), colIndices(), or rowOffsets() escapes, aliases make trust unenforceable: use adaptive
/// cg/pcg (which revalidates) or transfer into a fresh GPU CSR object.
template <typename Scalar>
AsyncIterativeSolverState cgFixedIterationsAsync(
    const CSRMatrix<Scalar, Device::GPU>& matrix,
    const DenseMatrix<Scalar, Device::GPU>& rhs,
    DenseMatrix<Scalar, Device::GPU>& solution,
    int iterations,
    IterativeSolverWorkspace<Scalar>& workspace,
    cudaStream_t stream = nullptr);

/// Submit exactly iterations Jacobi-PCG steps without host convergence checks.
/// The matrix must have validated CSR structure under the same rules as cgFixedIterationsAsync().
template <typename Scalar>
AsyncIterativeSolverState pcgFixedIterationsAsync(
    const CSRMatrix<Scalar, Device::GPU>& matrix,
    const DenseMatrix<Scalar, Device::GPU>& rhs,
    DenseMatrix<Scalar, Device::GPU>& solution,
    int iterations,
    IterativeSolverWorkspace<Scalar>& workspace,
    cudaStream_t stream = nullptr);

/// Read a completed fixed-iteration state and apply normal convergence/reporting policy.
IterativeSolverReport finalizeIterativeSolverReport(
    const AsyncIterativeSolverState& state,
    const IterativeSolverOptions& options = {});

} // namespace plamatrix
