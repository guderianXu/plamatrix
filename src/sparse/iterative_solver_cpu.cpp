#include "plamatrix/sparse/iterative_solver.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace plamatrix
{
namespace
{

void validateOptions(const IterativeSolverOptions& options)
{
    if (options.maxIterations < 0)
    {
        throw std::invalid_argument("iterative solver maxIterations must be non-negative");
    }
    if (!std::isfinite(options.relativeTolerance) || options.relativeTolerance < 0.0
        || options.relativeTolerance > 1.0
        || !std::isfinite(options.absoluteTolerance) || options.absoluteTolerance < 0.0)
    {
        throw std::invalid_argument(
            "iterative solver relative tolerance must be in [0, 1] and absolute "
            "tolerance must be finite and non-negative");
    }
}

template <typename Scalar>
void validateSystem(const CSRMatrix<Scalar, Device::CPU>& matrix,
                    const DenseMatrix<Scalar, Device::CPU>& rhs,
                    const DenseMatrix<Scalar, Device::CPU>& solution)
{
    if (matrix.rows() != matrix.cols())
    {
        throw std::invalid_argument("iterative solver matrix must be square");
    }
    if (rhs.rows() != matrix.rows() || rhs.cols() != 1)
    {
        throw std::invalid_argument("iterative solver rhs must be matrix.rows() x 1");
    }
    if (solution.rows() != matrix.cols() || solution.cols() != 1)
    {
        throw std::invalid_argument("iterative solver solution must be matrix.cols() x 1");
    }
    if (rhs.data() != nullptr && rhs.data() == solution.data())
    {
        throw std::invalid_argument("iterative solver rhs and solution must not alias");
    }

    const Index* row_offsets = matrix.rowOffsets();
    if (row_offsets[0] != 0 || row_offsets[matrix.rows()] != matrix.nnz())
    {
        throw std::invalid_argument(
            "iterative solver requires CSR row offsets to start at zero and end at nnz");
    }
    for (Index row = 0; row < matrix.rows(); ++row)
    {
        const Index begin = row_offsets[row];
        const Index end = row_offsets[row + 1];
        if (begin < 0 || end < begin || end > matrix.nnz())
        {
            std::ostringstream message;
            message << "iterative solver invalid CSR row offsets at row " << row;
            throw std::invalid_argument(message.str());
        }
    }
    for (Index position = 0; position < matrix.nnz(); ++position)
    {
        if (matrix.colIndices()[position] < 0
            || matrix.colIndices()[position] >= matrix.cols())
        {
            std::ostringstream message;
            message << "iterative solver CSR column index out of range at position "
                    << position;
            throw std::invalid_argument(message.str());
        }
        if (!std::isfinite(static_cast<double>(matrix.values()[position])))
        {
            std::ostringstream message;
            message << "iterative solver CSR value is not finite at position " << position;
            throw std::invalid_argument(message.str());
        }
    }
    for (Index row = 0; row < matrix.rows(); ++row)
    {
        if (!std::isfinite(static_cast<double>(rhs.data()[row]))
            || !std::isfinite(static_cast<double>(solution.data()[row])))
        {
            std::ostringstream message;
            message << "iterative solver rhs and solution must be finite at row " << row;
            throw std::invalid_argument(message.str());
        }
    }
}

double dot(const std::vector<double>& left, const std::vector<double>& right)
{
    double result = 0.0;
    for (std::size_t i = 0; i < left.size(); ++i)
    {
        result += left[i] * right[i];
    }
    return result;
}

template <typename Scalar>
void multiply(const CSRMatrix<Scalar, Device::CPU>& matrix,
              const std::vector<double>& input,
              std::vector<double>& output)
{
    for (Index row = 0; row < matrix.rows(); ++row)
    {
        double sum = 0.0;
        for (Index position = matrix.rowOffsets()[row];
             position < matrix.rowOffsets()[row + 1]; ++position)
        {
            sum += static_cast<double>(matrix.values()[position])
                * input[static_cast<std::size_t>(matrix.colIndices()[position])];
        }
        output[static_cast<std::size_t>(row)] = sum;
    }
}

template <typename Scalar>
std::vector<double> jacobiInverse(const CSRMatrix<Scalar, Device::CPU>& matrix)
{
    std::vector<double> inverse(static_cast<std::size_t>(matrix.rows()));
    for (Index row = 0; row < matrix.rows(); ++row)
    {
        bool found = false;
        double diagonal = 0.0;
        double row_scale = 0.0;
        for (Index position = matrix.rowOffsets()[row];
             position < matrix.rowOffsets()[row + 1]; ++position)
        {
            const double value = static_cast<double>(matrix.values()[position]);
            row_scale = std::max(row_scale, std::abs(value));
            if (matrix.colIndices()[position] == row)
            {
                diagonal += value;
                found = true;
            }
        }
        const double threshold = static_cast<double>(std::numeric_limits<Scalar>::epsilon())
            * row_scale * 16.0;
        if (!found || !std::isfinite(diagonal) || diagonal <= threshold
            || !std::isfinite(1.0 / diagonal))
        {
            std::ostringstream message;
            message << "pcg Jacobi diagonal is missing, non-positive, or near zero at row "
                    << row;
            throw std::runtime_error(message.str());
        }
        inverse[static_cast<std::size_t>(row)] = 1.0 / diagonal;
    }
    return inverse;
}

template <typename Scalar>
IterativeSolverReport solve(const CSRMatrix<Scalar, Device::CPU>& matrix,
                            const DenseMatrix<Scalar, Device::CPU>& rhs,
                            DenseMatrix<Scalar, Device::CPU>& solution,
                            const IterativeSolverOptions& options,
                            bool preconditioned)
{
    validateOptions(options);
    validateSystem(matrix, rhs, solution);
    const std::size_t size = static_cast<std::size_t>(matrix.rows());
    std::vector<double> x(size);
    std::vector<double> residual(size);
    std::vector<double> direction(size);
    std::vector<double> transformed(size);
    std::vector<double> matrix_direction(size);
    for (std::size_t i = 0; i < size; ++i)
    {
        x[i] = static_cast<double>(solution.data()[i]);
    }
    multiply(matrix, x, matrix_direction);
    for (std::size_t i = 0; i < size; ++i)
    {
        residual[i] = static_cast<double>(rhs.data()[i]) - matrix_direction[i];
    }

    IterativeSolverReport report;
    const double initial_squared = dot(residual, residual);
    if (!std::isfinite(initial_squared) || initial_squared < 0.0)
    {
        throw std::runtime_error("iterative solver initial residual is not finite");
    }
    report.initialResidual = std::sqrt(initial_squared);
    report.finalResidual = report.initialResidual;
    const double tolerance = std::max(
        options.absoluteTolerance, options.relativeTolerance * report.initialResidual);
    if (report.finalResidual <= tolerance)
    {
        report.converged = true;
        return report;
    }

    const bool use_jacobi = preconditioned && options.useJacobiPreconditioner;
    const std::vector<double> inverse_diagonal = use_jacobi
        ? jacobiInverse(matrix) : std::vector<double>{};
    for (std::size_t i = 0; i < size; ++i)
    {
        transformed[i] = use_jacobi ? inverse_diagonal[i] * residual[i] : residual[i];
        direction[i] = transformed[i];
    }
    double rho = dot(residual, transformed);
    if (!std::isfinite(rho) || rho <= 0.0)
    {
        throw std::runtime_error(
            "iterative solver breakdown: preconditioned residual is not positive");
    }

    for (int iteration = 0; iteration < options.maxIterations; ++iteration)
    {
        multiply(matrix, direction, matrix_direction);
        const double denominator = dot(direction, matrix_direction);
        if (!std::isfinite(denominator) || denominator <= 0.0)
        {
            throw std::runtime_error(
                "iterative solver breakdown: matrix is not numerically SPD");
        }
        const double alpha = rho / denominator;
        if (!std::isfinite(alpha))
        {
            throw std::runtime_error("iterative solver breakdown: step size is not finite");
        }
        for (std::size_t i = 0; i < size; ++i)
        {
            x[i] += alpha * direction[i];
            residual[i] -= alpha * matrix_direction[i];
        }
        report.iterations = iteration + 1;
        const double residual_squared = dot(residual, residual);
        if (!std::isfinite(residual_squared) || residual_squared < 0.0)
        {
            throw std::runtime_error("iterative solver residual is not finite");
        }
        report.finalResidual = std::sqrt(residual_squared);
        if (report.finalResidual <= tolerance)
        {
            report.converged = true;
            break;
        }

        for (std::size_t i = 0; i < size; ++i)
        {
            transformed[i] = use_jacobi ? inverse_diagonal[i] * residual[i] : residual[i];
        }
        const double next_rho = dot(residual, transformed);
        if (!std::isfinite(next_rho) || next_rho <= 0.0)
        {
            throw std::runtime_error("iterative solver breakdown in direction update");
        }
        const double beta = next_rho / rho;
        if (!std::isfinite(beta))
        {
            throw std::runtime_error(
                "iterative solver breakdown: direction scale is not finite");
        }
        for (std::size_t i = 0; i < size; ++i)
        {
            direction[i] = transformed[i] + beta * direction[i];
        }
        rho = next_rho;
    }

    for (std::size_t i = 0; i < size; ++i)
    {
        solution.data()[i] = static_cast<Scalar>(x[i]);
    }
    if (!report.converged && options.requireConvergence)
    {
        std::ostringstream message;
        message << "iterative solver did not converge in " << report.iterations
                << " iterations; final residual=" << report.finalResidual;
        throw std::runtime_error(message.str());
    }
    return report;
}

} // anonymous namespace

template <typename Scalar>
IterativeSolverReport cg(const CSRMatrix<Scalar, Device::CPU>& matrix,
                         const DenseMatrix<Scalar, Device::CPU>& rhs,
                         DenseMatrix<Scalar, Device::CPU>& solution,
                         const IterativeSolverOptions& options)
{
    return solve(matrix, rhs, solution, options, false);
}

template <typename Scalar>
IterativeSolverReport pcg(const CSRMatrix<Scalar, Device::CPU>& matrix,
                          const DenseMatrix<Scalar, Device::CPU>& rhs,
                          DenseMatrix<Scalar, Device::CPU>& solution,
                          const IterativeSolverOptions& options)
{
    return solve(matrix, rhs, solution, options, true);
}

#ifdef PLAMATRIX_USE_FLOAT
template IterativeSolverReport cg<float>(
    const CSRMatrix<float, Device::CPU>&, const DenseMatrix<float, Device::CPU>&,
    DenseMatrix<float, Device::CPU>&, const IterativeSolverOptions&);
template IterativeSolverReport pcg<float>(
    const CSRMatrix<float, Device::CPU>&, const DenseMatrix<float, Device::CPU>&,
    DenseMatrix<float, Device::CPU>&, const IterativeSolverOptions&);
#endif

#ifdef PLAMATRIX_USE_DOUBLE
template IterativeSolverReport cg<double>(
    const CSRMatrix<double, Device::CPU>&, const DenseMatrix<double, Device::CPU>&,
    DenseMatrix<double, Device::CPU>&, const IterativeSolverOptions&);
template IterativeSolverReport pcg<double>(
    const CSRMatrix<double, Device::CPU>&, const DenseMatrix<double, Device::CPU>&,
    DenseMatrix<double, Device::CPU>&, const IterativeSolverOptions&);
#endif

#ifdef PLAMATRIX_NO_CUDA

template <typename Scalar>
IterativeSolverWorkspace<Scalar>::~IterativeSolverWorkspace() noexcept = default;

template <typename Scalar>
IterativeSolverWorkspace<Scalar>::IterativeSolverWorkspace(
    IterativeSolverWorkspace&& other) noexcept = default;

template <typename Scalar>
IterativeSolverWorkspace<Scalar>& IterativeSolverWorkspace<Scalar>::operator=(
    IterativeSolverWorkspace&& other) noexcept = default;

template <typename Scalar>
void IterativeSolverWorkspace<Scalar>::closeAsyncAllocation()
{
}

AsyncIterativeSolverState::~AsyncIterativeSolverState() noexcept = default;
AsyncIterativeSolverState::AsyncIterativeSolverState(
    AsyncIterativeSolverState&& other) noexcept = default;
AsyncIterativeSolverState& AsyncIterativeSolverState::operator=(
    AsyncIterativeSolverState&& other) noexcept = default;

void AsyncIterativeSolverState::closeAsyncAllocation()
{
}

template <typename Scalar>
IterativeSolverReport cg(
    const CSRMatrix<Scalar, Device::GPU>&,
    const DenseMatrix<Scalar, Device::GPU>&,
    DenseMatrix<Scalar, Device::GPU>&,
    IterativeSolverWorkspace<Scalar>&,
    const IterativeSolverOptions&,
    cudaStream_t)
{
    throw std::runtime_error("CUDA cg requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
IterativeSolverReport pcg(
    const CSRMatrix<Scalar, Device::GPU>&,
    const DenseMatrix<Scalar, Device::GPU>&,
    DenseMatrix<Scalar, Device::GPU>&,
    IterativeSolverWorkspace<Scalar>&,
    const IterativeSolverOptions&,
    cudaStream_t)
{
    throw std::runtime_error("CUDA pcg requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
IterativeSolverReport blockPcg(
    const CSRMatrix<Scalar, Device::GPU>&,
    const DenseMatrix<Scalar, Device::GPU>&,
    DenseMatrix<Scalar, Device::GPU>&,
    const DenseMatrix<Scalar, Device::GPU>&,
    Index,
    IterativeSolverWorkspace<Scalar>&,
    const IterativeSolverOptions&,
    cudaStream_t)
{
    throw std::runtime_error("CUDA block PCG requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
AsyncIterativeSolverState cgFixedIterationsAsync(
    const CSRMatrix<Scalar, Device::GPU>&,
    const DenseMatrix<Scalar, Device::GPU>&,
    DenseMatrix<Scalar, Device::GPU>&,
    int,
    IterativeSolverWorkspace<Scalar>&,
    cudaStream_t)
{
    throw std::runtime_error(
        "cgFixedIterationsAsync requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
AsyncIterativeSolverState pcgFixedIterationsAsync(
    const CSRMatrix<Scalar, Device::GPU>&,
    const DenseMatrix<Scalar, Device::GPU>&,
    DenseMatrix<Scalar, Device::GPU>&,
    int,
    IterativeSolverWorkspace<Scalar>&,
    cudaStream_t)
{
    throw std::runtime_error(
        "pcgFixedIterationsAsync requires PLAMATRIX_WITH_CUDA=ON");
}

IterativeSolverReport finalizeIterativeSolverReport(
    const AsyncIterativeSolverState&,
    const IterativeSolverOptions&)
{
    throw std::runtime_error(
        "finalizeIterativeSolverReport requires PLAMATRIX_WITH_CUDA=ON");
}

#define PLAMATRIX_INSTANTIATE_NO_CUDA_SOLVER(Scalar)                                \
    template class IterativeSolverWorkspace<Scalar>;                               \
    template IterativeSolverReport cg<Scalar>(                                     \
        const CSRMatrix<Scalar, Device::GPU>&, const DenseMatrix<Scalar, Device::GPU>&, \
        DenseMatrix<Scalar, Device::GPU>&, IterativeSolverWorkspace<Scalar>&,       \
        const IterativeSolverOptions&, cudaStream_t);                              \
    template IterativeSolverReport pcg<Scalar>(                                    \
        const CSRMatrix<Scalar, Device::GPU>&, const DenseMatrix<Scalar, Device::GPU>&, \
        DenseMatrix<Scalar, Device::GPU>&, IterativeSolverWorkspace<Scalar>&,       \
        const IterativeSolverOptions&, cudaStream_t);                              \
    template IterativeSolverReport blockPcg<Scalar>(                               \
        const CSRMatrix<Scalar, Device::GPU>&, const DenseMatrix<Scalar, Device::GPU>&, \
        DenseMatrix<Scalar, Device::GPU>&, const DenseMatrix<Scalar, Device::GPU>&, \
        Index, IterativeSolverWorkspace<Scalar>&, const IterativeSolverOptions&,    \
        cudaStream_t);                                                             \
    template AsyncIterativeSolverState cgFixedIterationsAsync<Scalar>(             \
        const CSRMatrix<Scalar, Device::GPU>&, const DenseMatrix<Scalar, Device::GPU>&, \
        DenseMatrix<Scalar, Device::GPU>&, int, IterativeSolverWorkspace<Scalar>&,  \
        cudaStream_t);                                                             \
    template AsyncIterativeSolverState pcgFixedIterationsAsync<Scalar>(            \
        const CSRMatrix<Scalar, Device::GPU>&, const DenseMatrix<Scalar, Device::GPU>&, \
        DenseMatrix<Scalar, Device::GPU>&, int, IterativeSolverWorkspace<Scalar>&,  \
        cudaStream_t)

#ifdef PLAMATRIX_USE_FLOAT
PLAMATRIX_INSTANTIATE_NO_CUDA_SOLVER(float);
#endif
#ifdef PLAMATRIX_USE_DOUBLE
PLAMATRIX_INSTANTIATE_NO_CUDA_SOLVER(double);
#endif

#undef PLAMATRIX_INSTANTIATE_NO_CUDA_SOLVER
#endif

} // namespace plamatrix
