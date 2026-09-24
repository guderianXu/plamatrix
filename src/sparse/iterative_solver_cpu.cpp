#include "plamatrix/internal/sparse/iterative_solver.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace plamatrix::internal
{

#ifdef PLAMATRIX_WITH_CUDA
namespace resident_solver_detail
{
template <typename Scalar>
IterativeSolverReport pcgCudaResident(const ResidentCsrMatrix<Scalar>& matrix,
                                      const ResidentVector<Scalar>& rhs,
                                      ResidentVector<Scalar>& solution,
                                      const SolveOptions& options,
                                      ExecutionContext& context);
}
#endif

#ifdef PLAMATRIX_WITH_OPENCL
namespace resident_solver_detail
{
template <typename Scalar>
IterativeSolverReport pcgOpenClResident(const ResidentCsrMatrix<Scalar>& matrix,
                                        const ResidentVector<Scalar>& rhs,
                                        ResidentVector<Scalar>& solution,
                                        const SolveOptions& options,
                                        ExecutionContext& context);
}
#endif

#ifdef PLAMATRIX_WITH_VULKAN
namespace resident_solver_detail
{
IterativeSolverReport pcgVulkanResident(const ResidentCsrMatrix<float>& matrix,
                                        const ResidentVector<float>& rhs,
                                        ResidentVector<float>& solution,
                                        const SolveOptions& options,
                                        ExecutionContext& context);
}
#endif

    namespace
    {

        void validateOptions(const IterativeSolverOptions& options)
        {
            if (options.maxIterations < 0)
            {
                throw std::invalid_argument("iterative solver maxIterations must be non-negative");
            }
            if (!std::isfinite(options.relativeTolerance) || options.relativeTolerance < 0.0 ||
                options.relativeTolerance > 1.0 || !std::isfinite(options.absoluteTolerance) ||
                options.absoluteTolerance < 0.0 || options.convergenceCheckInterval <= 0)
            {
                throw std::invalid_argument("iterative solver relative tolerance must be in [0, 1] and absolute "
                                            "tolerance must be finite and non-negative");
            }
        }

        template <typename Scalar, typename Right, typename Solution>
        void validateSystem(const CsrStorage<Scalar, Device::CPU>& matrix, const Right& rhs, const Solution& solution)
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
                if (matrix.colIndices()[position] < 0 || matrix.colIndices()[position] >= matrix.cols())
                {
                    std::ostringstream message;
                    message << "iterative solver CSR column index out of range at position " << position;
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
                if (!std::isfinite(static_cast<double>(rhs.data()[row])) ||
                    !std::isfinite(static_cast<double>(solution.data()[row])))
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
        void multiply(const CsrStorage<Scalar, Device::CPU>& matrix,
                      const std::vector<double>& input,
                      std::vector<double>& output)
        {
            for (Index row = 0; row < matrix.rows(); ++row)
            {
                double sum = 0.0;
                for (Index position = matrix.rowOffsets()[row]; position < matrix.rowOffsets()[row + 1]; ++position)
                {
                    sum += static_cast<double>(matrix.values()[position]) *
                           input[static_cast<std::size_t>(matrix.colIndices()[position])];
                }
                output[static_cast<std::size_t>(row)] = sum;
            }
        }

        template <typename Scalar> std::vector<double> jacobiInverse(const CsrStorage<Scalar, Device::CPU>& matrix)
        {
            std::vector<double> inverse(static_cast<std::size_t>(matrix.rows()));
            for (Index row = 0; row < matrix.rows(); ++row)
            {
                bool found = false;
                double diagonal = 0.0;
                double row_scale = 0.0;
                for (Index position = matrix.rowOffsets()[row]; position < matrix.rowOffsets()[row + 1]; ++position)
                {
                    const double value = static_cast<double>(matrix.values()[position]);
                    row_scale = std::max(row_scale, std::abs(value));
                    if (matrix.colIndices()[position] == row)
                    {
                        diagonal += value;
                        found = true;
                    }
                }
                const double threshold = static_cast<double>(std::numeric_limits<Scalar>::epsilon()) * row_scale * 16.0;
                if (!found || !std::isfinite(diagonal) || diagonal <= threshold || !std::isfinite(1.0 / diagonal))
                {
                    std::ostringstream message;
                    message << "pcg Jacobi diagonal is missing, non-positive, or near zero at row " << row;
                    throw std::runtime_error(message.str());
                }
                inverse[static_cast<std::size_t>(row)] = 1.0 / diagonal;
            }
            return inverse;
        }

        template <typename Scalar, typename Right, typename Solution>
        IterativeSolverReport solve(const CsrStorage<Scalar, Device::CPU>& matrix,
                                    const Right& rhs,
                                    Solution& solution,
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
            const double tolerance =
                std::max(options.absoluteTolerance, options.relativeTolerance * report.initialResidual);
            if (report.finalResidual <= tolerance)
            {
                report.converged = true;
                return report;
            }

            const bool use_jacobi = preconditioned && options.useJacobiPreconditioner;
            const std::vector<double> inverse_diagonal = use_jacobi ? jacobiInverse(matrix) : std::vector<double>{};
            for (std::size_t i = 0; i < size; ++i)
            {
                transformed[i] = use_jacobi ? inverse_diagonal[i] * residual[i] : residual[i];
                direction[i] = transformed[i];
            }
            double rho = dot(residual, transformed);
            if (!std::isfinite(rho) || rho <= 0.0)
            {
                throw std::runtime_error("iterative solver breakdown: preconditioned residual is not positive");
            }

            for (int iteration = 0; iteration < options.maxIterations; ++iteration)
            {
                multiply(matrix, direction, matrix_direction);
                const double denominator = dot(direction, matrix_direction);
                if (!std::isfinite(denominator) || denominator <= 0.0)
                {
                    throw std::runtime_error("iterative solver breakdown: matrix is not numerically SPD");
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
                    throw std::runtime_error("iterative solver breakdown: direction scale is not finite");
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
    IterativeSolverReport cg(const CsrStorage<Scalar, Device::CPU>& matrix,
                             const DenseStorage<Scalar, Device::CPU>& rhs,
                             DenseStorage<Scalar, Device::CPU>& solution,
                             const IterativeSolverOptions& options)
    {
        return solve(matrix, rhs, solution, options, false);
    }

    template <typename Scalar>
    IterativeSolverReport pcg(const CsrStorage<Scalar, Device::CPU>& matrix,
                              const DenseStorage<Scalar, Device::CPU>& rhs,
                              DenseStorage<Scalar, Device::CPU>& solution,
                              const IterativeSolverOptions& options)
    {
        return solve(matrix, rhs, solution, options, true);
    }

    template <typename Scalar>
    IterativeSolverReport cg(const CsrStorage<Scalar, Device::CPU>& matrix,
                             const Matrix<Scalar, Dynamic, 1>& rhs,
                             Matrix<Scalar, Dynamic, 1>& solution,
                             const IterativeSolverOptions& options)
    {
        if (currentExecutionSettings().policy == ExecutionPolicy::GpuRequired)
        {
            throw Error(ErrorCode::UnsupportedOperation, "CSR CG with Matrix vectors has no GPU implementation");
        }
        return solve(matrix, rhs, solution, options, false);
    }

    template <typename Scalar>
    IterativeSolverReport pcg(const CsrStorage<Scalar, Device::CPU>& matrix,
                              const Matrix<Scalar, Dynamic, 1>& rhs,
                              Matrix<Scalar, Dynamic, 1>& solution,
                              const IterativeSolverOptions& options)
    {
        if (currentExecutionSettings().policy == ExecutionPolicy::GpuRequired)
        {
            throw Error(ErrorCode::UnsupportedOperation, "CSR PCG with Matrix vectors has no GPU implementation");
        }
        return solve(matrix, rhs, solution, options, true);
    }

#ifdef PLAMATRIX_USE_FLOAT
    template IterativeSolverReport cg<float>(const CsrStorage<float, Device::CPU>&,
                                             const DenseStorage<float, Device::CPU>&,
                                             DenseStorage<float, Device::CPU>&,
                                             const IterativeSolverOptions&);
    template IterativeSolverReport pcg<float>(const CsrStorage<float, Device::CPU>&,
                                              const DenseStorage<float, Device::CPU>&,
                                              DenseStorage<float, Device::CPU>&,
                                              const IterativeSolverOptions&);
    template IterativeSolverReport cg<float>(const CsrStorage<float, Device::CPU>&,
                                             const Matrix<float, Dynamic, 1>&,
                                             Matrix<float, Dynamic, 1>&,
                                             const IterativeSolverOptions&);
    template IterativeSolverReport pcg<float>(const CsrStorage<float, Device::CPU>&,
                                              const Matrix<float, Dynamic, 1>&,
                                              Matrix<float, Dynamic, 1>&,
                                              const IterativeSolverOptions&);
#endif

#ifdef PLAMATRIX_USE_DOUBLE
    template IterativeSolverReport cg<double>(const CsrStorage<double, Device::CPU>&,
                                              const DenseStorage<double, Device::CPU>&,
                                              DenseStorage<double, Device::CPU>&,
                                              const IterativeSolverOptions&);
    template IterativeSolverReport pcg<double>(const CsrStorage<double, Device::CPU>&,
                                               const DenseStorage<double, Device::CPU>&,
                                               DenseStorage<double, Device::CPU>&,
                                               const IterativeSolverOptions&);
    template IterativeSolverReport cg<double>(const CsrStorage<double, Device::CPU>&,
                                              const Matrix<double, Dynamic, 1>&,
                                              Matrix<double, Dynamic, 1>&,
                                              const IterativeSolverOptions&);
    template IterativeSolverReport pcg<double>(const CsrStorage<double, Device::CPU>&,
                                               const Matrix<double, Dynamic, 1>&,
                                               Matrix<double, Dynamic, 1>&,
                                               const IterativeSolverOptions&);
#endif

    namespace
    {

        template <typename Scalar>
        IterativeSolverReport solveResidentPcg(const ResidentCsrMatrix<Scalar>& matrix,
                                               const ResidentVector<Scalar>& rhs,
                                               ResidentVector<Scalar>& solution,
                                               const SolveOptions& options,
                                               ExecutionContext& context)
        {
            matrix.validateContext(context);
            rhs.validateContext(context);
            solution.validateContext(context);
            if (context.backend() == Backend::Cuda)
            {
#ifdef PLAMATRIX_WITH_CUDA
                return resident_solver_detail::pcgCudaResident(matrix, rhs, solution, options, context);
#else
                throw Error(ErrorCode::BackendUnavailable,
                            "PlaMatrix was built without CUDA support",
                            context.backend());
#endif
            }
            if (context.backend() == Backend::OpenCl)
            {
#ifdef PLAMATRIX_WITH_OPENCL
                return resident_solver_detail::pcgOpenClResident(matrix, rhs, solution, options, context);
#else
                throw Error(ErrorCode::BackendUnavailable,
                            "PlaMatrix was built without OpenCL support",
                            context.backend());
#endif
            }
            if (context.backend() == Backend::Vulkan)
            {
#ifdef PLAMATRIX_WITH_VULKAN
                if constexpr (std::is_same_v<Scalar, float>)
                {
                    return resident_solver_detail::pcgVulkanResident(matrix, rhs, solution, options, context);
                }
                else
                {
                    throw Error(ErrorCode::UnsupportedOperation,
                                "Vulkan resident PCG currently supports only float32",
                                context.backend());
                }
#else
                throw Error(ErrorCode::BackendUnavailable,
                            "PlaMatrix was built without Vulkan support",
                            context.backend());
#endif
            }
            if (context.backend() != Backend::Cpu)
            {
                throw Error(ErrorCode::UnsupportedOperation,
                            "Resident PCG is not implemented for this backend yet",
                            context.backend());
            }
            if (matrix.rows() != matrix.cols() || rhs.size() != matrix.rows() || solution.size() != matrix.cols())
            {
                throw Error(ErrorCode::InvalidArgument,
                            "Resident PCG requires a square CSR matrix and matching vectors",
                            context.backend());
            }
            if (options.maxIterations < 0 || !std::isfinite(options.relativeTolerance) ||
                options.relativeTolerance < 0.0 || options.relativeTolerance > 1.0 ||
                !std::isfinite(options.absoluteTolerance) || options.absoluteTolerance < 0.0)
            {
                throw Error(ErrorCode::InvalidArgument,
                            "Resident PCG options are invalid",
                            context.backend());
            }

            const std::size_t size = static_cast<std::size_t>(matrix.rows());
            std::vector<double> x(size);
            std::vector<double> residual(size);
            std::vector<double> direction(size);
            std::vector<double> transformed(size);
            std::vector<double> matrix_direction(size);
            const auto multiply = [&](const std::vector<double>& input, std::vector<double>& output)
            {
                for (Index row = 0; row < matrix.rows(); ++row)
                {
                    double sum = 0.0;
                    for (Index position = matrix.rowOffsets()[row];
                         position < matrix.rowOffsets()[row + 1];
                         ++position)
                    {
                        const Index column = matrix.colIndices()[position];
                        if (column < 0 || column >= matrix.cols())
                        {
                            throw Error(ErrorCode::InvalidArgument,
                                        "Resident PCG CSR column index is out of range",
                                        context.backend());
                        }
                        sum += static_cast<double>(matrix.values()[position]) *
                               input[static_cast<std::size_t>(column)];
                    }
                    output[static_cast<std::size_t>(row)] = sum;
                }
            };

            for (std::size_t index = 0; index < size; ++index)
            {
                x[index] = static_cast<double>(solution.data()[index]);
            }
            multiply(x, matrix_direction);
            for (std::size_t index = 0; index < size; ++index)
            {
                residual[index] = static_cast<double>(rhs.data()[index]) - matrix_direction[index];
            }

            IterativeSolverReport report;
            const auto dot = [](const std::vector<double>& left, const std::vector<double>& right)
            {
                double result = 0.0;
                for (std::size_t index = 0; index < left.size(); ++index)
                {
                    result += left[index] * right[index];
                }
                return result;
            };
            const double initial_squared = dot(residual, residual);
            report.initialResidual = std::sqrt(initial_squared);
            report.finalResidual = report.initialResidual;
            if (options.recordResidualHistory)
            {
                report.residualHistory.push_back(report.initialResidual);
            }
            const double tolerance = std::max(options.absoluteTolerance,
                                              options.relativeTolerance * report.initialResidual);
            if (report.finalResidual <= tolerance)
            {
                report.converged = true;
                return report;
            }

            std::vector<double> inverse_diagonal;
            if (options.useJacobiPreconditioner)
            {
                inverse_diagonal.resize(size);
                for (Index row = 0; row < matrix.rows(); ++row)
                {
                    double diagonal = 0.0;
                    bool found = false;
                    for (Index position = matrix.rowOffsets()[row];
                         position < matrix.rowOffsets()[row + 1];
                         ++position)
                    {
                        if (matrix.colIndices()[position] == row)
                        {
                            diagonal += static_cast<double>(matrix.values()[position]);
                            found = true;
                        }
                    }
                    if (!found || diagonal <= 0.0 || !std::isfinite(diagonal))
                    {
                        throw Error(ErrorCode::NumericalFailure,
                                    "Resident PCG Jacobi diagonal is invalid",
                                    context.backend());
                    }
                    inverse_diagonal[static_cast<std::size_t>(row)] = 1.0 / diagonal;
                }
            }

            for (std::size_t index = 0; index < size; ++index)
            {
                transformed[index] = inverse_diagonal.empty() ? residual[index]
                                                               : inverse_diagonal[index] * residual[index];
                direction[index] = transformed[index];
            }
            double rho = dot(residual, transformed);
            for (int iteration = 0; iteration < options.maxIterations; ++iteration)
            {
                if (options.cancellation && options.cancellation->isCancellationRequested())
                {
                    throw Error(ErrorCode::InvalidState, "Resident PCG was cancelled", context.backend());
                }
                multiply(direction, matrix_direction);
                const double denominator = dot(direction, matrix_direction);
                if (!std::isfinite(denominator) || denominator <= 0.0)
                {
                    throw Error(ErrorCode::NumericalFailure,
                                "Resident PCG matrix is not numerically SPD",
                                context.backend());
                }
                const double alpha = rho / denominator;
                for (std::size_t index = 0; index < size; ++index)
                {
                    x[index] += alpha * direction[index];
                    residual[index] -= alpha * matrix_direction[index];
                }
                report.iterations = iteration + 1;
                report.finalResidual = std::sqrt(dot(residual, residual));
                if (options.recordResidualHistory)
                {
                    report.residualHistory.push_back(report.finalResidual);
                }
                if (report.finalResidual <= tolerance)
                {
                    report.converged = true;
                    break;
                }
                for (std::size_t index = 0; index < size; ++index)
                {
                    transformed[index] = inverse_diagonal.empty() ? residual[index]
                                                                   : inverse_diagonal[index] * residual[index];
                }
                const double next_rho = dot(residual, transformed);
                const double beta = next_rho / rho;
                for (std::size_t index = 0; index < size; ++index)
                {
                    direction[index] = transformed[index] + beta * direction[index];
                }
                rho = next_rho;
            }
            for (std::size_t index = 0; index < size; ++index)
            {
                solution.data()[index] = static_cast<Scalar>(x[index]);
            }
            if (!report.converged && options.requireConvergence)
            {
                throw Error(ErrorCode::NumericalFailure,
                            "Resident PCG did not converge",
                            context.backend());
            }
            report.diagnostics.deterministic = options.deterministic;
            return report;
        }

    } // namespace

    template <typename Scalar>
    IterativeSolverReport pcg(const ResidentCsrMatrix<Scalar>& matrix,
                              const ResidentVector<Scalar>& rhs,
                              ResidentVector<Scalar>& solution,
                              const SolveOptions& options,
                              ExecutionContext& context)
    {
        return solveResidentPcg(matrix, rhs, solution, options, context);
    }

    template <typename Scalar>
    Event pcgAsync(const ResidentCsrMatrix<Scalar>& matrix,
                   const ResidentVector<Scalar>& rhs,
                   ResidentVector<Scalar>& solution,
                   const SolveOptions& options,
                   ExecutionContext& context,
                   IterativeSolverReport* report)
    {
        const auto result = solveResidentPcg(matrix, rhs, solution, options, context);
        if (report != nullptr)
        {
            *report = result;
        }
        return Event::completed();
    }

#ifdef PLAMATRIX_USE_FLOAT
    template IterativeSolverReport pcg<float>(const ResidentCsrMatrix<float>&,
                                              const ResidentVector<float>&,
                                              ResidentVector<float>&,
                                              const SolveOptions&,
                                              ExecutionContext&);
    template Event pcgAsync<float>(const ResidentCsrMatrix<float>&,
                                   const ResidentVector<float>&,
                                   ResidentVector<float>&,
                                   const SolveOptions&,
                                   ExecutionContext&,
                                   IterativeSolverReport*);
#endif
#ifdef PLAMATRIX_USE_DOUBLE
    template IterativeSolverReport pcg<double>(const ResidentCsrMatrix<double>&,
                                               const ResidentVector<double>&,
                                               ResidentVector<double>&,
                                               const SolveOptions&,
                                               ExecutionContext&);
    template Event pcgAsync<double>(const ResidentCsrMatrix<double>&,
                                    const ResidentVector<double>&,
                                    ResidentVector<double>&,
                                    const SolveOptions&,
                                    ExecutionContext&,
                                    IterativeSolverReport*);
#endif

#ifdef PLAMATRIX_NO_CUDA

    template <typename Scalar> IterativeSolverWorkspace<Scalar>::~IterativeSolverWorkspace() noexcept = default;

    template <typename Scalar>
    IterativeSolverWorkspace<Scalar>::IterativeSolverWorkspace(IterativeSolverWorkspace&& other) noexcept = default;

    template <typename Scalar>
    IterativeSolverWorkspace<Scalar>&
    IterativeSolverWorkspace<Scalar>::operator=(IterativeSolverWorkspace&& other) noexcept = default;

    template <typename Scalar> void IterativeSolverWorkspace<Scalar>::closeAsyncAllocation()
    {
    }

    AsyncIterativeSolverState::~AsyncIterativeSolverState() noexcept = default;
    AsyncIterativeSolverState::AsyncIterativeSolverState(AsyncIterativeSolverState&& other) noexcept = default;
    AsyncIterativeSolverState&
    AsyncIterativeSolverState::operator=(AsyncIterativeSolverState&& other) noexcept = default;

    void AsyncIterativeSolverState::closeAsyncAllocation()
    {
    }

    template <typename Scalar>
    IterativeSolverReport cg(const CsrStorage<Scalar, Device::GPU>&,
                             const DenseStorage<Scalar, Device::GPU>&,
                             DenseStorage<Scalar, Device::GPU>&,
                             IterativeSolverWorkspace<Scalar>&,
                             const IterativeSolverOptions&,
                             cudaStream_t)
    {
        throw std::runtime_error("CUDA cg requires PLAMATRIX_WITH_CUDA=ON");
    }

    template <typename Scalar>
    IterativeSolverReport pcg(const CsrStorage<Scalar, Device::GPU>&,
                              const DenseStorage<Scalar, Device::GPU>&,
                              DenseStorage<Scalar, Device::GPU>&,
                              IterativeSolverWorkspace<Scalar>&,
                              const IterativeSolverOptions&,
                              cudaStream_t)
    {
        throw std::runtime_error("CUDA pcg requires PLAMATRIX_WITH_CUDA=ON");
    }

    template <typename Scalar>
    IterativeSolverReport blockPcg(const CsrStorage<Scalar, Device::GPU>&,
                                   const DenseStorage<Scalar, Device::GPU>&,
                                   DenseStorage<Scalar, Device::GPU>&,
                                   const DenseStorage<Scalar, Device::GPU>&,
                                   Index,
                                   IterativeSolverWorkspace<Scalar>&,
                                   const IterativeSolverOptions&,
                                   cudaStream_t)
    {
        throw std::runtime_error("CUDA block PCG requires PLAMATRIX_WITH_CUDA=ON");
    }

    template <typename Scalar>
    AsyncIterativeSolverState cgFixedIterationsAsync(const CsrStorage<Scalar, Device::GPU>&,
                                                     const DenseStorage<Scalar, Device::GPU>&,
                                                     DenseStorage<Scalar, Device::GPU>&,
                                                     int,
                                                     IterativeSolverWorkspace<Scalar>&,
                                                     cudaStream_t)
    {
        throw std::runtime_error("cgFixedIterationsAsync requires PLAMATRIX_WITH_CUDA=ON");
    }

    template <typename Scalar>
    AsyncIterativeSolverState pcgFixedIterationsAsync(const CsrStorage<Scalar, Device::GPU>&,
                                                      const DenseStorage<Scalar, Device::GPU>&,
                                                      DenseStorage<Scalar, Device::GPU>&,
                                                      int,
                                                      IterativeSolverWorkspace<Scalar>&,
                                                      cudaStream_t)
    {
        throw std::runtime_error("pcgFixedIterationsAsync requires PLAMATRIX_WITH_CUDA=ON");
    }

    IterativeSolverReport finalizeIterativeSolverReport(const AsyncIterativeSolverState&, const IterativeSolverOptions&)
    {
        throw std::runtime_error("finalizeIterativeSolverReport requires PLAMATRIX_WITH_CUDA=ON");
    }

#define PLAMATRIX_INSTANTIATE_NO_CUDA_SOLVER(Scalar)                                                                   \
    template class IterativeSolverWorkspace<Scalar>;                                                                   \
    template IterativeSolverReport cg<Scalar>(const CsrStorage<Scalar, Device::GPU>&,                                   \
                                              const DenseStorage<Scalar, Device::GPU>&,                                 \
                                              DenseStorage<Scalar, Device::GPU>&,                                       \
                                              IterativeSolverWorkspace<Scalar>&,                                       \
                                              const IterativeSolverOptions&,                                           \
                                              cudaStream_t);                                                           \
    template IterativeSolverReport pcg<Scalar>(const CsrStorage<Scalar, Device::GPU>&,                                  \
                                               const DenseStorage<Scalar, Device::GPU>&,                                \
                                               DenseStorage<Scalar, Device::GPU>&,                                      \
                                               IterativeSolverWorkspace<Scalar>&,                                      \
                                               const IterativeSolverOptions&,                                          \
                                               cudaStream_t);                                                          \
    template IterativeSolverReport blockPcg<Scalar>(const CsrStorage<Scalar, Device::GPU>&,                             \
                                                    const DenseStorage<Scalar, Device::GPU>&,                           \
                                                    DenseStorage<Scalar, Device::GPU>&,                                 \
                                                    const DenseStorage<Scalar, Device::GPU>&,                           \
                                                    Index,                                                             \
                                                    IterativeSolverWorkspace<Scalar>&,                                 \
                                                    const IterativeSolverOptions&,                                     \
                                                    cudaStream_t);                                                     \
    template AsyncIterativeSolverState cgFixedIterationsAsync<Scalar>(const CsrStorage<Scalar, Device::GPU>&,           \
                                                                      const DenseStorage<Scalar, Device::GPU>&,         \
                                                                      DenseStorage<Scalar, Device::GPU>&,               \
                                                                      int,                                             \
                                                                      IterativeSolverWorkspace<Scalar>&,               \
                                                                      cudaStream_t);                                   \
    template AsyncIterativeSolverState pcgFixedIterationsAsync<Scalar>(const CsrStorage<Scalar, Device::GPU>&,          \
                                                                       const DenseStorage<Scalar, Device::GPU>&,        \
                                                                       DenseStorage<Scalar, Device::GPU>&,              \
                                                                       int,                                            \
                                                                       IterativeSolverWorkspace<Scalar>&,              \
                                                                       cudaStream_t)

#ifdef PLAMATRIX_USE_FLOAT
    PLAMATRIX_INSTANTIATE_NO_CUDA_SOLVER(float);
#endif
#ifdef PLAMATRIX_USE_DOUBLE
    PLAMATRIX_INSTANTIATE_NO_CUDA_SOLVER(double);
#endif

#undef PLAMATRIX_INSTANTIATE_NO_CUDA_SOLVER
#endif

} // namespace plamatrix::internal
