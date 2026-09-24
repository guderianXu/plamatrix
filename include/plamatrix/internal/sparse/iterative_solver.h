#pragma once

#include <cstdint>
#include <vector>

#include "plamatrix/internal/dense/dense_storage.h"
#include "plamatrix/dense/matrix.h"
#include "plamatrix/internal/device/device_csr_matrix.h"
#include "plamatrix/internal/device/device_vector.h"
#include "plamatrix/internal/core/cancellation.h"
#include "plamatrix/internal/core/diagnostics.h"
#include "plamatrix/internal/core/event.h"
#include "plamatrix/internal/core/execution_context.h"
#include "plamatrix/internal/sparse/csr_storage.h"
#include "plamatrix/internal/sparse/sparse_ops.h"

namespace plamatrix::internal
{

    struct IterativeSolverOptions
    {
        int maxIterations = 1000;
        /// Relative residual tolerance in [0, 1].
        double relativeTolerance = 1.0e-6;
        double absoluteTolerance = 0.0;
        bool useJacobiPreconditioner = true;
        bool requireConvergence = false;
        /// CUDA adaptive solves may batch this many device iterations between host checks.
        /// Device-side convergence freezes the solution at the first converged iteration.
        int convergenceCheckInterval = 1;
    };

    struct SolveOptions
    {
        int maxIterations = 1000;
        double relativeTolerance = 1.0e-6;
        double absoluteTolerance = 0.0;
        bool useJacobiPreconditioner = true;
        bool requireConvergence = false;
        bool deterministic = false;
        bool recordResidualHistory = false;
        CancellationToken* cancellation = nullptr;
    };

    struct IterativeSolverReport
    {
        bool converged = false;
        int iterations = 0;
        double initialResidual = 0.0;
        double finalResidual = 0.0;
        /// Number of backend command submissions when the backend exposes this diagnostic.
        std::uint32_t commandSubmissions = 0;
        /// Number of Vulkan command-buffer recordings performed during the solve.
        std::uint32_t commandBufferRecordings = 0;
        /// Number of descriptor sets allocated during the solve when the backend exposes this diagnostic.
        std::uint32_t descriptorSetAllocations = 0;
        /// GPU execution time accumulated by Vulkan when timestamp queries are available.
        double gpuMilliseconds = 0.0;
        /// Number of Vulkan buffer dependency barriers executed for the solve.
        std::uint32_t barrierCount = 0;
        /// Whether Vulkan selected the subgroup-per-row CSR SpMV kernel.
        bool subgroupSpmv = false;
        /// Whether Vulkan selected the dense-block sparse SpMV kernel.
        bool blockSpmv = false;
        /// Whether Vulkan generated block-Jacobi inverse blocks from device-resident matrix values.
        bool deviceBlockJacobi = false;
        BackendDiagnostics diagnostics;
        std::vector<double> residualHistory;
    };

    /// Move-only reusable storage for CUDA CG/PCG solves.
    /// The bound CUDA stream must remain alive until closeAsyncAllocation() or destruction completes.
    /// closeAsyncAllocation() enqueues stream-ordered frees; synchronize once more before destroying a
    /// non-default stream.
    template <typename Scalar> class IterativeSolverWorkspace
    {
    public:
        IterativeSolverWorkspace() noexcept
            : _residual()
            , _direction()
            , _transformed()
            , _matrixDirection()
            , _inverseDiagonal()
            , _scalars()
            , _sparseWorkspace()
        {
        }
        ~IterativeSolverWorkspace() noexcept;
        IterativeSolverWorkspace(IterativeSolverWorkspace&& other) noexcept;
        IterativeSolverWorkspace& operator=(IterativeSolverWorkspace&& other) noexcept;

        IterativeSolverWorkspace(const IterativeSolverWorkspace&) = delete;
        IterativeSolverWorkspace& operator=(const IterativeSolverWorkspace&) = delete;

        /// Release stream-ordered storage after the owning stream has completed.
        /// The owning stream must remain valid through this call and the subsequent free completion.
        void closeAsyncAllocation();
        Index capacitySize() const noexcept
        {
            return _capacitySize;
        }

    private:
        friend struct IterativeSolverWorkspaceAccess;

        DenseStorage<Scalar, Device::GPU> _residual;
        DenseStorage<Scalar, Device::GPU> _direction;
        DenseStorage<Scalar, Device::GPU> _transformed;
        DenseStorage<Scalar, Device::GPU> _matrixDirection;
        DenseStorage<Scalar, Device::GPU> _inverseDiagonal;
        DenseStorage<Scalar, Device::GPU> _scalars;
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
        AsyncIterativeSolverState() = default;
        ~AsyncIterativeSolverState() noexcept;
        AsyncIterativeSolverState(AsyncIterativeSolverState&& other) noexcept;
        AsyncIterativeSolverState& operator=(AsyncIterativeSolverState&& other) noexcept;

        AsyncIterativeSolverState(const AsyncIterativeSolverState&) = delete;
        AsyncIterativeSolverState& operator=(const AsyncIterativeSolverState&) = delete;

        void closeAsyncAllocation();

        int submittedIterations = 0;
        DenseStorage<double, Device::GPU> initialResidualSquared;
        DenseStorage<double, Device::GPU> finalResidualSquared;

    private:
        friend struct IterativeSolverStateAccess;
        friend IterativeSolverReport finalizeIterativeSolverReport(const AsyncIterativeSolverState&,
                                                                   const IterativeSolverOptions&);

        void* _completionEvent = nullptr;
        cudaStream_t _stream = nullptr;
    };

    /// Solve an SPD CPU system with conjugate gradients, using solution as the initial guess.
    template <typename Scalar>
    IterativeSolverReport cg(const CsrStorage<Scalar, Device::CPU>& matrix,
                             const DenseStorage<Scalar, Device::CPU>& rhs,
                             DenseStorage<Scalar, Device::CPU>& solution,
                             const IterativeSolverOptions& options = {});

    /// Solve an SPD CPU system with optional Jacobi-preconditioned conjugate gradients.
    template <typename Scalar>
    IterativeSolverReport pcg(const CsrStorage<Scalar, Device::CPU>& matrix,
                              const DenseStorage<Scalar, Device::CPU>& rhs,
                              DenseStorage<Scalar, Device::CPU>& solution,
                              const IterativeSolverOptions& options = {});

    /// Solve directly into Eigen-style vectors without converting their storage to DenseStorage.
    template <typename Scalar>
    IterativeSolverReport cg(const CsrStorage<Scalar, Device::CPU>& matrix,
                             const Matrix<Scalar, Dynamic, 1>& rhs,
                             Matrix<Scalar, Dynamic, 1>& solution,
                             const IterativeSolverOptions& options = {});

    template <typename Scalar>
    IterativeSolverReport pcg(const CsrStorage<Scalar, Device::CPU>& matrix,
                              const Matrix<Scalar, Dynamic, 1>& rhs,
                              Matrix<Scalar, Dynamic, 1>& solution,
                              const IterativeSolverOptions& options = {});

    /// Adaptively solve an SPD GPU system with conjugate gradients.
    /// Unknown CSR structure is validated once on stream before cuSPARSE is launched.
    template <typename Scalar>
    IterativeSolverReport cg(const CsrStorage<Scalar, Device::GPU>& matrix,
                             const DenseStorage<Scalar, Device::GPU>& rhs,
                             DenseStorage<Scalar, Device::GPU>& solution,
                             IterativeSolverWorkspace<Scalar>& workspace,
                             const IterativeSolverOptions& options = {},
                             cudaStream_t stream = nullptr);

    /// Adaptively solve an SPD GPU system with optional Jacobi preconditioning.
    template <typename Scalar>
    IterativeSolverReport pcg(const CsrStorage<Scalar, Device::GPU>& matrix,
                              const DenseStorage<Scalar, Device::GPU>& rhs,
                              DenseStorage<Scalar, Device::GPU>& solution,
                              IterativeSolverWorkspace<Scalar>& workspace,
                              const IterativeSolverOptions& options = {},
                              cudaStream_t stream = nullptr);

    /// Adaptively solve an SPD GPU system with caller-supplied inverse diagonal blocks.
    /// inverse_blocks stores row-major blocks as a contiguous (matrix.rows() * block_size) x 1 vector.
    template <typename Scalar>
    IterativeSolverReport blockPcg(const CsrStorage<Scalar, Device::GPU>& matrix,
                                   const DenseStorage<Scalar, Device::GPU>& rhs,
                                   DenseStorage<Scalar, Device::GPU>& solution,
                                   const DenseStorage<Scalar, Device::GPU>& inverse_blocks,
                                   Index block_size,
                                   IterativeSolverWorkspace<Scalar>& workspace,
                                   const IterativeSolverOptions& options = {},
                                   cudaStream_t stream = nullptr);

    /// Submit exactly iterations CG steps without host convergence checks.
    /// The matrix must have trusted CSR structure. Synchronous CPU-to-GPU transfer establishes trust.
    /// Async transfer requires validateStructure() on its copy stream before submission. Once mutable
    /// values(), colIndices(), or rowOffsets() escapes, aliases make trust unenforceable: use adaptive
    /// cg/pcg (which revalidates) or transfer into a fresh GPU CSR object.
    template <typename Scalar>
    AsyncIterativeSolverState cgFixedIterationsAsync(const CsrStorage<Scalar, Device::GPU>& matrix,
                                                     const DenseStorage<Scalar, Device::GPU>& rhs,
                                                     DenseStorage<Scalar, Device::GPU>& solution,
                                                     int iterations,
                                                     IterativeSolverWorkspace<Scalar>& workspace,
                                                     cudaStream_t stream = nullptr);

    /// Submit exactly iterations Jacobi-PCG steps without host convergence checks.
    /// The matrix must have validated CSR structure under the same rules as cgFixedIterationsAsync().
    template <typename Scalar>
    AsyncIterativeSolverState pcgFixedIterationsAsync(const CsrStorage<Scalar, Device::GPU>& matrix,
                                                      const DenseStorage<Scalar, Device::GPU>& rhs,
                                                      DenseStorage<Scalar, Device::GPU>& solution,
                                                      int iterations,
                                                      IterativeSolverWorkspace<Scalar>& workspace,
                                                      cudaStream_t stream = nullptr);

    /// Read a completed fixed-iteration state and apply normal convergence/reporting policy.
    IterativeSolverReport finalizeIterativeSolverReport(const AsyncIterativeSolverState& state,
                                                        const IterativeSolverOptions& options = {});

    template <typename Scalar>
    IterativeSolverReport pcg(const ResidentCsrMatrix<Scalar>& matrix,
                              const ResidentVector<Scalar>& rhs,
                              ResidentVector<Scalar>& solution,
                              const SolveOptions& options,
                              ExecutionContext& context);

    template <typename Scalar>
    Event pcgAsync(const ResidentCsrMatrix<Scalar>& matrix,
                   const ResidentVector<Scalar>& rhs,
                   ResidentVector<Scalar>& solution,
                   const SolveOptions& options,
                   ExecutionContext& context,
                   IterativeSolverReport* report = nullptr);

} // namespace plamatrix::internal
