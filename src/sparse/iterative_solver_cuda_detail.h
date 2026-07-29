#pragma once

#include <climits>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "plamatrix/core/error_check.h"
#include "plamatrix/sparse/iterative_solver.h"

namespace plamatrix
{
namespace iterative_solver_detail
{

constexpr int kBlockSize = 256;
constexpr int kRho = 0;
constexpr int kNextRho = 1;
constexpr int kDenominator = 2;
constexpr int kAlpha = 3;
constexpr int kNegativeAlpha = 4;
constexpr int kBeta = 5;
constexpr int kResidualSquared = 6;
constexpr int kBreakdown = 7;
constexpr int kScalarCount = 8;

inline void validateOptions(const IterativeSolverOptions& options)
{
    if (options.maxIterations < 0 || !std::isfinite(options.relativeTolerance)
        || options.relativeTolerance < 0.0 || options.relativeTolerance > 1.0
        || !std::isfinite(options.absoluteTolerance) || options.absoluteTolerance < 0.0)
    {
        throw std::invalid_argument("invalid CUDA iterative solver options");
    }
}

template <typename Matrix>
void checkStreamStorage(const char* name, const Matrix& matrix, cudaStream_t stream)
{
    if (matrix.isAsyncAllocation() && matrix.asyncAllocationStream() != stream)
    {
        std::ostringstream message;
        message << name << " must use the stream that owns its async allocation";
        throw std::logic_error(message.str());
    }
}

template <typename Scalar>
void validateSystem(const CSRMatrix<Scalar, Device::GPU>& matrix,
                    const DenseMatrix<Scalar, Device::GPU>& rhs,
                    const DenseMatrix<Scalar, Device::GPU>& solution,
                    cudaStream_t stream,
                    bool require_prevalidated_structure)
{
    if (matrix.rows() != matrix.cols() || rhs.rows() != matrix.rows() || rhs.cols() != 1
        || solution.rows() != matrix.cols() || solution.cols() != 1)
    {
        throw std::invalid_argument("CUDA iterative solver requires square A and Nx1 rhs/solution");
    }
    if (matrix.rows() > INT_MAX)
    {
        throw std::overflow_error("CUDA iterative solver size exceeds cuBLAS int range");
    }
    if (rhs.data() != nullptr && rhs.data() == solution.data())
    {
        throw std::invalid_argument("CUDA iterative solver rhs and solution must not alias");
    }
    checkStreamStorage("CUDA iterative solver matrix", matrix, stream);
    checkStreamStorage("CUDA iterative solver rhs", rhs, stream);
    checkStreamStorage("CUDA iterative solver solution", solution, stream);
    if (!matrix.isStructureUsableOnStream(stream))
    {
        if (require_prevalidated_structure)
        {
            throw std::logic_error(
                "fixed CUDA iterative solver requires prevalidated CSR structure; "
                "call matrix.validateStructure(stream) after mutable array access");
        }
        matrix.validateStructure(stream);
    }
}

inline int gridSize(Index size)
{
    return static_cast<int>((size + kBlockSize - 1) / kBlockSize);
}

} // namespace iterative_solver_detail

struct IterativeSolverWorkspaceAccess
{
#ifdef PLAMATRIX_ITERATIVE_SOLVER_TEST_HOOKS
    static DenseMatrix<float, Device::GPU> allocateFloatForTest(
        Index rows, cudaStream_t stream);
    static DenseMatrix<double, Device::GPU> allocateDoubleForTest(
        Index rows, cudaStream_t stream);
#endif

    template <typename Scalar>
    static DenseMatrix<Scalar, Device::GPU> allocate(Index rows, cudaStream_t stream)
    {
#ifdef PLAMATRIX_ITERATIVE_SOLVER_TEST_HOOKS
        if constexpr (std::is_same_v<Scalar, float>)
        {
            return allocateFloatForTest(rows, stream);
        }
        else
        {
            return allocateDoubleForTest(rows, stream);
        }
#else
        return DenseMatrix<Scalar, Device::GPU>::uninitializedAsync(rows, 1, stream);
#endif
    }

    template <typename Scalar>
    static DenseMatrix<Scalar, Device::GPU>& residual(
        IterativeSolverWorkspace<Scalar>& workspace) { return workspace._residual; }
    template <typename Scalar>
    static DenseMatrix<Scalar, Device::GPU>& direction(
        IterativeSolverWorkspace<Scalar>& workspace) { return workspace._direction; }
    template <typename Scalar>
    static DenseMatrix<Scalar, Device::GPU>& transformed(
        IterativeSolverWorkspace<Scalar>& workspace) { return workspace._transformed; }
    template <typename Scalar>
    static DenseMatrix<Scalar, Device::GPU>& matrixDirection(
        IterativeSolverWorkspace<Scalar>& workspace) { return workspace._matrixDirection; }
    template <typename Scalar>
    static DenseMatrix<Scalar, Device::GPU>& inverseDiagonal(
        IterativeSolverWorkspace<Scalar>& workspace) { return workspace._inverseDiagonal; }
    template <typename Scalar>
    static DenseMatrix<Scalar, Device::GPU>& scalars(
        IterativeSolverWorkspace<Scalar>& workspace) { return workspace._scalars; }
    template <typename Scalar>
    static SparseOpsWorkspace& sparse(
        IterativeSolverWorkspace<Scalar>& workspace) { return workspace._sparseWorkspace; }
    template <typename Scalar>
    static Index size(const IterativeSolverWorkspace<Scalar>& workspace)
    {
        return workspace._capacitySize;
    }

    template <typename Scalar>
    static void bind(IterativeSolverWorkspace<Scalar>& workspace, Index size,
                     cudaStream_t stream)
    {
        if (workspace._hasReuseStream && workspace._reuseStream != stream)
        {
            throw std::logic_error(
                "IterativeSolverWorkspace cannot be reused on another stream; close it first");
        }
        if (workspace._residual.rows() != size || workspace._residual.cols() != 1)
        {
            auto residual = allocate<Scalar>(size, stream);
            auto direction = allocate<Scalar>(size, stream);
            auto transformed = allocate<Scalar>(size, stream);
            auto matrix_direction = allocate<Scalar>(size, stream);
            auto inverse_diagonal = allocate<Scalar>(size, stream);
            auto scalars = allocate<Scalar>(iterative_solver_detail::kScalarCount, stream);
            auto old_residual = std::move(workspace._residual);
            auto old_direction = std::move(workspace._direction);
            auto old_transformed = std::move(workspace._transformed);
            auto old_matrix_direction = std::move(workspace._matrixDirection);
            auto old_inverse_diagonal = std::move(workspace._inverseDiagonal);
            auto old_scalars = std::move(workspace._scalars);
            workspace._residual = std::move(residual);
            workspace._direction = std::move(direction);
            workspace._transformed = std::move(transformed);
            workspace._matrixDirection = std::move(matrix_direction);
            workspace._inverseDiagonal = std::move(inverse_diagonal);
            workspace._scalars = std::move(scalars);
            workspace._capacitySize = size;
        }
        workspace._reuseStream = stream;
        workspace._hasReuseStream = true;
    }

    template <typename Scalar>
    static cublasHandle_t handle(IterativeSolverWorkspace<Scalar>& workspace,
                                 cudaStream_t stream)
    {
        if (workspace._blasHandle == nullptr)
        {
            cublasHandle_t handle = nullptr;
            PLAMATRIX_CHECK_CUBLAS(cublasCreate(&handle));
            workspace._blasHandle = handle;
        }
        auto handle = static_cast<cublasHandle_t>(workspace._blasHandle);
        PLAMATRIX_CHECK_CUBLAS(cublasSetStream(handle, stream));
        PLAMATRIX_CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
        return handle;
    }
};

struct IterativeSolverStateAccess
{
    static AsyncIterativeSolverState create(cudaStream_t stream, int iterations)
    {
        AsyncIterativeSolverState state;
        state.submittedIterations = iterations;
        state.initialResidualSquared =
            DenseMatrix<double, Device::GPU>::uninitializedAsync(1, 1, stream);
        state.finalResidualSquared =
            DenseMatrix<double, Device::GPU>::uninitializedAsync(1, 1, stream);
        cudaEvent_t event = nullptr;
        PLAMATRIX_CHECK_CUDA(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        state._completionEvent = event;
        state._stream = stream;
        return state;
    }

    static void record(AsyncIterativeSolverState& state)
    {
        PLAMATRIX_CHECK_CUDA(cudaEventRecord(
            static_cast<cudaEvent_t>(state._completionEvent), state._stream));
    }
};

} // namespace plamatrix
