#include "iterative_solver_cuda_detail.h"
#include "iterative_solver_kernels.cuh"
#include "iterative_solver_test_hooks.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace plamatrix
{
using namespace iterative_solver_detail;

namespace
{

#ifdef PLAMATRIX_ITERATIVE_SOLVER_TEST_HOOKS
void* fixed_solver_completion_gate = nullptr;

__global__ void waitForFixedSolverCompletionGate(int* gate)
{
    while (atomicAdd(gate, 0) == 0)
    {
        __nanosleep(1000);
    }
}
#endif

template <typename Scalar>
void initialize(const CSRMatrix<Scalar, Device::GPU>& matrix,
                const DenseMatrix<Scalar, Device::GPU>& rhs,
                DenseMatrix<Scalar, Device::GPU>& solution,
                IterativeSolverWorkspace<Scalar>& workspace,
                bool preconditioned,
                cudaStream_t stream)
{
    const Index size = matrix.rows();
    IterativeSolverWorkspaceAccess::bind(workspace, size, stream);
    if (size == 0)
    {
        return;
    }
    auto handle = IterativeSolverWorkspaceAccess::handle(workspace, stream);
    auto& matrix_direction = IterativeSolverWorkspaceAccess::matrixDirection(workspace);
    auto& residual = IterativeSolverWorkspaceAccess::residual(workspace);
    auto& transformed = IterativeSolverWorkspaceAccess::transformed(workspace);
    auto& direction = IterativeSolverWorkspaceAccess::direction(workspace);
    auto& scalars = IterativeSolverWorkspaceAccess::scalars(workspace);
    PLAMATRIX_CHECK_CUDA(cudaMemsetAsync(
        scalars.data(), 0, static_cast<std::size_t>(kScalarCount) * sizeof(Scalar), stream));
    spmvAsync(matrix, solution, matrix_direction,
              IterativeSolverWorkspaceAccess::sparse(workspace), stream);
    subtractKernel<<<gridSize(size), kBlockSize, 0, stream>>>(
        rhs.data(), matrix_direction.data(), residual.data(), size);
    PLAMATRIX_CHECK_CUDA(cudaGetLastError());
    if (preconditioned)
    {
        jacobiKernel<<<gridSize(size), kBlockSize, 0, stream>>>(
            matrix.rowOffsets(), matrix.colIndices(), matrix.values(),
            IterativeSolverWorkspaceAccess::inverseDiagonal(workspace).data(), size);
        applyJacobiKernel<<<gridSize(size), kBlockSize, 0, stream>>>(
            IterativeSolverWorkspaceAccess::inverseDiagonal(workspace).data(), residual.data(),
            transformed.data(), size);
        PLAMATRIX_CHECK_CUDA(cudaGetLastError());
    }
    else
    {
        PLAMATRIX_CHECK_CUBLAS(Blas<Scalar>::copy(
            handle, static_cast<int>(size), residual.data(), transformed.data()));
    }
    PLAMATRIX_CHECK_CUBLAS(Blas<Scalar>::copy(
        handle, static_cast<int>(size), transformed.data(), direction.data()));
    PLAMATRIX_CHECK_CUBLAS(Blas<Scalar>::dot(
        handle, static_cast<int>(size), residual.data(), transformed.data(),
        scalars.data() + kRho));
}

template <typename Scalar>
void submitStep(const CSRMatrix<Scalar, Device::GPU>& matrix,
                DenseMatrix<Scalar, Device::GPU>& solution,
                IterativeSolverWorkspace<Scalar>& workspace,
                cudaStream_t stream)
{
    const int size = static_cast<int>(matrix.rows());
    auto handle = IterativeSolverWorkspaceAccess::handle(workspace, stream);
    auto& direction = IterativeSolverWorkspaceAccess::direction(workspace);
    auto& matrix_direction = IterativeSolverWorkspaceAccess::matrixDirection(workspace);
    auto& residual = IterativeSolverWorkspaceAccess::residual(workspace);
    auto& scalars = IterativeSolverWorkspaceAccess::scalars(workspace);
    spmvAsync(matrix, direction, matrix_direction,
              IterativeSolverWorkspaceAccess::sparse(workspace), stream);
    PLAMATRIX_CHECK_CUBLAS(Blas<Scalar>::dot(
        handle, size, direction.data(), matrix_direction.data(),
        scalars.data() + kDenominator));
    alphaKernel<Scalar><<<1, 1, 0, stream>>>(scalars.data());
    PLAMATRIX_CHECK_CUDA(cudaGetLastError());
    PLAMATRIX_CHECK_CUBLAS(Blas<Scalar>::axpy(
        handle, size, scalars.data() + kAlpha, direction.data(), solution.data()));
    PLAMATRIX_CHECK_CUBLAS(Blas<Scalar>::axpy(
        handle, size, scalars.data() + kNegativeAlpha,
        matrix_direction.data(), residual.data()));
}

template <typename Scalar>
void updateDirection(IterativeSolverWorkspace<Scalar>& workspace,
                     bool preconditioned,
                     cudaStream_t stream)
{
    const int size = static_cast<int>(IterativeSolverWorkspaceAccess::size(workspace));
    auto handle = IterativeSolverWorkspaceAccess::handle(workspace, stream);
    auto& inverse = IterativeSolverWorkspaceAccess::inverseDiagonal(workspace);
    auto& residual = IterativeSolverWorkspaceAccess::residual(workspace);
    auto& transformed = IterativeSolverWorkspaceAccess::transformed(workspace);
    auto& direction = IterativeSolverWorkspaceAccess::direction(workspace);
    auto& scalars = IterativeSolverWorkspaceAccess::scalars(workspace);
    if (preconditioned)
    {
        applyJacobiKernel<<<gridSize(size), kBlockSize, 0, stream>>>(
            inverse.data(), residual.data(), transformed.data(), size);
        PLAMATRIX_CHECK_CUDA(cudaGetLastError());
    }
    else
    {
        PLAMATRIX_CHECK_CUBLAS(Blas<Scalar>::copy(
            handle, size, residual.data(), transformed.data()));
    }
    PLAMATRIX_CHECK_CUBLAS(Blas<Scalar>::dot(
        handle, size, residual.data(), transformed.data(), scalars.data() + kNextRho));
    betaKernel<Scalar><<<1, 1, 0, stream>>>(scalars.data());
    PLAMATRIX_CHECK_CUDA(cudaGetLastError());
    PLAMATRIX_CHECK_CUBLAS(Blas<Scalar>::scal(
        handle, size, scalars.data() + kBeta, direction.data()));
    PLAMATRIX_CHECK_CUBLAS(Blas<Scalar>::axpy(
        handle, size, scalars.data() + kAlpha, transformed.data(), direction.data()));
}

template <typename Scalar>
void residualSquared(IterativeSolverWorkspace<Scalar>& workspace, cudaStream_t stream)
{
    auto handle = IterativeSolverWorkspaceAccess::handle(workspace, stream);
    auto& residual = IterativeSolverWorkspaceAccess::residual(workspace);
    auto& scalars = IterativeSolverWorkspaceAccess::scalars(workspace);
    PLAMATRIX_CHECK_CUBLAS(Blas<Scalar>::dot(
        handle, static_cast<int>(IterativeSolverWorkspaceAccess::size(workspace)),
        residual.data(), residual.data(), scalars.data() + kResidualSquared));
}

template <typename Scalar>
AsyncIterativeSolverState fixedSolve(
    const CSRMatrix<Scalar, Device::GPU>& matrix,
    const DenseMatrix<Scalar, Device::GPU>& rhs,
    DenseMatrix<Scalar, Device::GPU>& solution,
    int iterations,
    IterativeSolverWorkspace<Scalar>& workspace,
    cudaStream_t stream,
    bool preconditioned)
{
    if (iterations < 0)
    {
        throw std::invalid_argument("fixed CUDA iterative solver iterations must be non-negative");
    }
    validateSystem(matrix, rhs, solution, stream, true);
    auto state = IterativeSolverStateAccess::create(stream, iterations);
    initialize(matrix, rhs, solution, workspace, preconditioned, stream);
    if (matrix.rows() == 0)
    {
        PLAMATRIX_CHECK_CUDA(cudaMemsetAsync(
            state.initialResidualSquared.data(), 0, sizeof(double), stream));
        PLAMATRIX_CHECK_CUDA(cudaMemsetAsync(
            state.finalResidualSquared.data(), 0, sizeof(double), stream));
    }
    else
    {
        residualSquared(workspace, stream);
        toDoubleKernel<<<1, 1, 0, stream>>>(
            IterativeSolverWorkspaceAccess::scalars(workspace).data() + kResidualSquared,
            state.initialResidualSquared.data());
        for (int iteration = 0; iteration < iterations; ++iteration)
        {
            submitStep(matrix, solution, workspace, stream);
            if (iteration + 1 < iterations)
            {
                updateDirection(workspace, preconditioned, stream);
            }
        }
        residualSquared(workspace, stream);
        toDoubleCheckedKernel<<<1, 1, 0, stream>>>(
            IterativeSolverWorkspaceAccess::scalars(workspace).data() + kResidualSquared,
            IterativeSolverWorkspaceAccess::scalars(workspace).data() + kBreakdown,
            state.finalResidualSquared.data());
        PLAMATRIX_CHECK_CUDA(cudaGetLastError());
    }
#ifdef PLAMATRIX_ITERATIVE_SOLVER_TEST_HOOKS
    if (fixed_solver_completion_gate != nullptr)
    {
        waitForFixedSolverCompletionGate<<<1, 1, 0, stream>>>(
            static_cast<int*>(fixed_solver_completion_gate));
        PLAMATRIX_CHECK_CUDA(cudaGetLastError());
    }
#endif
    IterativeSolverStateAccess::record(state);
    return state;
}

template <typename Scalar>
IterativeSolverReport adaptiveSolve(
    const CSRMatrix<Scalar, Device::GPU>& matrix,
    const DenseMatrix<Scalar, Device::GPU>& rhs,
    DenseMatrix<Scalar, Device::GPU>& solution,
    IterativeSolverWorkspace<Scalar>& workspace,
    const IterativeSolverOptions& options,
    cudaStream_t stream,
    bool preconditioned)
{
    validateOptions(options);
    validateSystem(matrix, rhs, solution, stream, false);
    initialize(matrix, rhs, solution, workspace, preconditioned, stream);
    IterativeSolverReport report;
    if (matrix.rows() == 0)
    {
        report.converged = true;
        return report;
    }
    Scalar host_squared = Scalar{0};
    Scalar host_breakdown = Scalar{0};
    residualSquared(workspace, stream);
    PLAMATRIX_CHECK_CUDA(cudaMemcpyAsync(
        &host_squared,
        IterativeSolverWorkspaceAccess::scalars(workspace).data() + kResidualSquared,
        sizeof(Scalar), cudaMemcpyDeviceToHost, stream));
    PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
    if (!std::isfinite(static_cast<double>(host_squared)) || host_squared < Scalar{0})
    {
        throw std::runtime_error("CUDA iterative solver initial residual is not finite");
    }
    report.initialResidual = std::sqrt(static_cast<double>(host_squared));
    report.finalResidual = report.initialResidual;
    const double tolerance = std::max(
        options.absoluteTolerance, options.relativeTolerance * report.initialResidual);
    if (report.finalResidual <= tolerance)
    {
        report.converged = true;
        return report;
    }
    for (int iteration = 0; iteration < options.maxIterations; ++iteration)
    {
        submitStep(matrix, solution, workspace, stream);
        residualSquared(workspace, stream);
        PLAMATRIX_CHECK_CUDA(cudaMemcpyAsync(
            &host_squared,
            IterativeSolverWorkspaceAccess::scalars(workspace).data() + kResidualSquared,
            sizeof(Scalar), cudaMemcpyDeviceToHost, stream));
        PLAMATRIX_CHECK_CUDA(cudaMemcpyAsync(
            &host_breakdown,
            IterativeSolverWorkspaceAccess::scalars(workspace).data() + kBreakdown,
            sizeof(Scalar), cudaMemcpyDeviceToHost, stream));
        PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
        if (host_breakdown != Scalar{0})
        {
            throw std::runtime_error(
                "CUDA iterative solver breakdown: matrix is not numerically SPD");
        }
        if (!std::isfinite(static_cast<double>(host_squared)) || host_squared < Scalar{0})
        {
            throw std::runtime_error(
                "CUDA iterative solver breakdown: matrix is not numerically SPD");
        }
        report.iterations = iteration + 1;
        report.finalResidual = std::sqrt(static_cast<double>(host_squared));
        if (report.finalResidual <= tolerance)
        {
            report.converged = true;
            break;
        }
        if (iteration + 1 < options.maxIterations)
        {
            updateDirection(workspace, preconditioned, stream);
        }
    }
    if (!report.converged && options.requireConvergence)
    {
        std::ostringstream message;
        message << "CUDA iterative solver did not converge in " << report.iterations
                << " iterations; final residual=" << report.finalResidual;
        throw std::runtime_error(message.str());
    }
    return report;
}

} // anonymous namespace

#ifdef PLAMATRIX_ITERATIVE_SOLVER_TEST_HOOKS
void iterative_solver_detail::setFixedSolverCompletionGate(void* event) noexcept
{
    fixed_solver_completion_gate = event;
}
#endif

template <typename Scalar>
IterativeSolverReport cg(
    const CSRMatrix<Scalar, Device::GPU>& matrix,
    const DenseMatrix<Scalar, Device::GPU>& rhs,
    DenseMatrix<Scalar, Device::GPU>& solution,
    IterativeSolverWorkspace<Scalar>& workspace,
    const IterativeSolverOptions& options,
    cudaStream_t stream)
{
    return adaptiveSolve(matrix, rhs, solution, workspace, options, stream, false);
}

template <typename Scalar>
IterativeSolverReport pcg(
    const CSRMatrix<Scalar, Device::GPU>& matrix,
    const DenseMatrix<Scalar, Device::GPU>& rhs,
    DenseMatrix<Scalar, Device::GPU>& solution,
    IterativeSolverWorkspace<Scalar>& workspace,
    const IterativeSolverOptions& options,
    cudaStream_t stream)
{
    return adaptiveSolve(
        matrix, rhs, solution, workspace, options, stream,
        options.useJacobiPreconditioner);
}

template <typename Scalar>
AsyncIterativeSolverState cgFixedIterationsAsync(
    const CSRMatrix<Scalar, Device::GPU>& matrix,
    const DenseMatrix<Scalar, Device::GPU>& rhs,
    DenseMatrix<Scalar, Device::GPU>& solution,
    int iterations,
    IterativeSolverWorkspace<Scalar>& workspace,
    cudaStream_t stream)
{
    return fixedSolve(matrix, rhs, solution, iterations, workspace, stream, false);
}

template <typename Scalar>
AsyncIterativeSolverState pcgFixedIterationsAsync(
    const CSRMatrix<Scalar, Device::GPU>& matrix,
    const DenseMatrix<Scalar, Device::GPU>& rhs,
    DenseMatrix<Scalar, Device::GPU>& solution,
    int iterations,
    IterativeSolverWorkspace<Scalar>& workspace,
    cudaStream_t stream)
{
    return fixedSolve(matrix, rhs, solution, iterations, workspace, stream, true);
}

IterativeSolverReport finalizeIterativeSolverReport(
    const AsyncIterativeSolverState& state,
    const IterativeSolverOptions& options)
{
    validateOptions(options);
    if (state._completionEvent == nullptr)
    {
        throw std::logic_error("fixed CUDA iterative solver state is empty or closed");
    }
    const cudaError_t status = cudaEventQuery(static_cast<cudaEvent_t>(state._completionEvent));
    if (status == cudaErrorNotReady)
    {
        throw std::logic_error(
            "finalizeIterativeSolverReport requires caller stream synchronization");
    }
    PLAMATRIX_CHECK_CUDA(status);
    double initial_squared = 0.0;
    double final_squared = 0.0;
    PLAMATRIX_CHECK_CUDA(cudaMemcpyAsync(
        &initial_squared, state.initialResidualSquared.data(), sizeof(double),
        cudaMemcpyDeviceToHost, state._stream));
    PLAMATRIX_CHECK_CUDA(cudaMemcpyAsync(
        &final_squared, state.finalResidualSquared.data(), sizeof(double),
        cudaMemcpyDeviceToHost, state._stream));
    PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(state._stream));
    if (!std::isfinite(initial_squared) || !std::isfinite(final_squared)
        || initial_squared < 0.0 || final_squared < 0.0)
    {
        throw std::runtime_error(
            "fixed CUDA iterative solver breakdown: residual is not finite");
    }
    IterativeSolverReport report;
    report.iterations = state.submittedIterations;
    report.initialResidual = std::sqrt(initial_squared);
    report.finalResidual = std::sqrt(final_squared);
    const double tolerance = std::max(
        options.absoluteTolerance, options.relativeTolerance * report.initialResidual);
    report.converged = report.finalResidual <= tolerance;
    if (!report.converged && options.requireConvergence)
    {
        std::ostringstream message;
        message << "fixed CUDA iterative solver did not converge in " << report.iterations
                << " iterations; final residual=" << report.finalResidual;
        throw std::runtime_error(message.str());
    }
    return report;
}

#define PLAMATRIX_INSTANTIATE_GPU_SOLVER(Scalar)                                     \
    template IterativeSolverReport cg<Scalar>(                                      \
        const CSRMatrix<Scalar, Device::GPU>&, const DenseMatrix<Scalar, Device::GPU>&, \
        DenseMatrix<Scalar, Device::GPU>&, IterativeSolverWorkspace<Scalar>&,        \
        const IterativeSolverOptions&, cudaStream_t);                               \
    template IterativeSolverReport pcg<Scalar>(                                     \
        const CSRMatrix<Scalar, Device::GPU>&, const DenseMatrix<Scalar, Device::GPU>&, \
        DenseMatrix<Scalar, Device::GPU>&, IterativeSolverWorkspace<Scalar>&,        \
        const IterativeSolverOptions&, cudaStream_t);                               \
    template AsyncIterativeSolverState cgFixedIterationsAsync<Scalar>(              \
        const CSRMatrix<Scalar, Device::GPU>&, const DenseMatrix<Scalar, Device::GPU>&, \
        DenseMatrix<Scalar, Device::GPU>&, int, IterativeSolverWorkspace<Scalar>&,   \
        cudaStream_t);                                                              \
    template AsyncIterativeSolverState pcgFixedIterationsAsync<Scalar>(             \
        const CSRMatrix<Scalar, Device::GPU>&, const DenseMatrix<Scalar, Device::GPU>&, \
        DenseMatrix<Scalar, Device::GPU>&, int, IterativeSolverWorkspace<Scalar>&,   \
        cudaStream_t)

#ifdef PLAMATRIX_USE_FLOAT
PLAMATRIX_INSTANTIATE_GPU_SOLVER(float);
#endif
#ifdef PLAMATRIX_USE_DOUBLE
PLAMATRIX_INSTANTIATE_GPU_SOLVER(double);
#endif

#undef PLAMATRIX_INSTANTIATE_GPU_SOLVER

} // namespace plamatrix
