#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "plamatrix/internal/core/execution_context.h"
#include "plamatrix/internal/device/device_csr_matrix.h"
#include "plamatrix/internal/device/device_vector.h"
#include "plamatrix/internal/sparse/iterative_solver.h"
#include "plamatrix/sparse/detail/sparse_direct_common.h"
#include "plamatrix/sparse/preconditioners.h"

namespace plamatrix::v1
{
    /// Eigen-style conjugate gradient with reusable analysis, composable preconditioners, and resident GPU PCG.
    template <typename MatrixType, int UpLo, typename Preconditioner>
    class ConjugateGradient : public SparseSolverBase<ConjugateGradient<MatrixType, UpLo, Preconditioner>>
    {
        using Base = SparseSolverBase<ConjugateGradient<MatrixType, UpLo, Preconditioner>>;
        using Scalar = typename MatrixType::Scalar;
        static_assert(std::is_floating_point_v<Scalar>, "ConjugateGradient requires floating-point coefficients");
        static_assert(UpLo == Lower || UpLo == Upper || UpLo == (Lower | Upper),
                      "ConjugateGradient UpLo must select Lower, Upper, or both");

    public:
        using Vector = Matrix<Scalar, Dynamic, 1>;

        ConjugateGradient() = default;
        explicit ConjugateGradient(const MatrixType& matrix)
        {
            compute(matrix);
        }

        ConjugateGradient& compute(const MatrixType& matrix)
        {
            analyzePattern(matrix);
            if (_analysisIsOk)
            {
                factorize(matrix);
            }
            return *this;
        }

        ConjugateGradient& analyzePattern(const MatrixType& matrix)
        {
            _csr.reset();
            Base::_isInitialized = false;
            _analysisIsOk = false;
            _dimension = 0;
            _info = InvalidInput;
            _backend = internal::Backend::Cpu;
            if (matrix.rows() != matrix.cols())
            {
                throw std::invalid_argument("ConjugateGradient requires a square matrix");
            }
            _dimension = matrix.cols();
            _pattern = relevantPattern(matrix);
            _preconditioner.analyzePattern(matrix);
            if (_preconditioner.info() != Success)
            {
                _info = _preconditioner.info();
                return *this;
            }
            _analysisIsOk = true;
            _info = Success;
            _iterations = 0;
            _error = Scalar{};
            return *this;
        }

        ConjugateGradient& factorize(const MatrixType& matrix)
        {
            _csr.reset();
            Base::_isInitialized = false;
            if (!_analysisIsOk)
            {
                _info = InvalidInput;
                return *this;
            }
            if (matrix.rows() != _dimension || matrix.cols() != _dimension || relevantPattern(matrix) != _pattern)
            {
                _info = InvalidInput;
                return *this;
            }

            _csr = symmetricSnapshot(matrix);
            _preconditioner.factorize(matrix);
            if (_preconditioner.info() != Success)
            {
                _info = _preconditioner.info();
                return *this;
            }
            Base::_isInitialized = true;
            _info = Success;
            _iterations = 0;
            _error = Scalar{};
            _backend = internal::Backend::Cpu;
            return *this;
        }

        ConjugateGradient& setMaxIterations(Index count)
        {
            if (count < 0 || count > static_cast<Index>(std::numeric_limits<int>::max()))
            {
                throw std::invalid_argument("ConjugateGradient max iterations are out of range");
            }
            _maxIterations = count;
            return *this;
        }

        ConjugateGradient& setTolerance(Scalar tolerance)
        {
            if (!std::isfinite(tolerance) || tolerance < Scalar{})
            {
                throw std::invalid_argument("ConjugateGradient tolerance must be finite and non-negative");
            }
            _tolerance = tolerance;
            return *this;
        }

        Index maxIterations() const noexcept
        {
            return _maxIterations >= 0 ? _maxIterations
                                       : std::min<Index>(2 * _dimension, std::numeric_limits<int>::max());
        }
        Scalar tolerance() const noexcept
        {
            return _tolerance;
        }
        Index iterations() const noexcept
        {
            return _iterations;
        }
        Scalar error() const noexcept
        {
            return _error;
        }
        ComputationInfo info() const noexcept
        {
            return _info;
        }
        Index rows() const noexcept
        {
            return _dimension;
        }
        Index cols() const noexcept
        {
            return _dimension;
        }
        internal::Backend backend() const noexcept
        {
            return _backend;
        }
        Preconditioner& preconditioner() noexcept
        {
            return _preconditioner;
        }
        const Preconditioner& preconditioner() const noexcept
        {
            return _preconditioner;
        }

        template <typename Rhs> auto solve(const MatrixBase<Rhs>& rhs) const
        {
            constexpr int RightCols = detail::DenseTraits<Rhs>::ColsAtCompileTime;
            Matrix<Scalar, Dynamic, RightCols> guess = Matrix<Scalar, Dynamic, RightCols>::Zero(rhs.rows(), rhs.cols());
            return solveWithGuess(rhs, guess);
        }

        template <typename Rhs, typename Guess>
        auto solveWithGuess(const MatrixBase<Rhs>& rhs_base, const MatrixBase<Guess>& guess_base) const
        {
            const auto& rhs = rhs_base.derived();
            const auto& guess = guess_base.derived();
            using RhsScalar = typename detail::DenseTraits<Rhs>::ScalarType;
            using GuessScalar = typename detail::DenseTraits<Guess>::ScalarType;
            static_assert(std::is_same_v<RhsScalar, Scalar> && std::is_same_v<GuessScalar, Scalar>,
                          "ConjugateGradient rhs and guess scalars must match the matrix");
            constexpr int RightCols = detail::DenseTraits<Rhs>::ColsAtCompileTime;
            if (!_csr)
            {
                _info = InvalidInput;
                throw std::logic_error("ConjugateGradient::compute must be called before solve");
            }
            if (rhs.rows() != _dimension || guess.rows() != _dimension || rhs.cols() != guess.cols())
            {
                _info = InvalidInput;
                throw std::invalid_argument("ConjugateGradient rhs or guess has the wrong size");
            }
            Matrix<Scalar, Dynamic, RightCols> result(_dimension, rhs.cols());
            _iterations = 0;
            _error = Scalar{};
            for (Index column = 0; column < rhs.cols(); ++column)
            {
                Vector rhs_column(_dimension);
                Vector guess_column(_dimension);
                for (Index row = 0; row < _dimension; ++row)
                {
                    rhs_column(row) = rhs(row, column);
                    guess_column(row) = guess(row, column);
                }
                const Vector solution = solveVector(rhs_column, guess_column);
                for (Index row = 0; row < _dimension; ++row)
                {
                    result(row, column) = solution(row);
                }
            }
            return result;
        }

    private:
        static constexpr bool supportsResidentGpu = std::is_same_v<Preconditioner, IdentityPreconditioner> ||
                                                    std::is_same_v<Preconditioner, DiagonalPreconditioner<Scalar>>;

        static std::vector<sparse_detail::Coordinate> relevantPattern(const MatrixType& matrix)
        {
            return sparse_detail::patternOf(matrix,
                                            [](Index row, Index col)
                                            {
                                                if constexpr (UpLo == Lower)
                                                {
                                                    return row >= col;
                                                }
                                                if constexpr (UpLo == Upper)
                                                {
                                                    return row <= col;
                                                }
                                                return true;
                                            });
        }

        static std::unique_ptr<internal::CsrStorage<Scalar, internal::Device::CPU>>
        symmetricSnapshot(const MatrixType& matrix)
        {
            if constexpr (UpLo == (Lower | Upper))
            {
                return std::make_unique<internal::CsrStorage<Scalar, internal::Device::CPU>>(
                    internal::SparseAccess::csrSnapshot(matrix));
            }
            std::vector<Index> rows;
            std::vector<Index> cols;
            std::vector<Scalar> values;
            rows.reserve(internal::SparseAccess::entries(matrix).size() * 2U);
            cols.reserve(internal::SparseAccess::entries(matrix).size() * 2U);
            values.reserve(internal::SparseAccess::entries(matrix).size() * 2U);
            for (const auto& [coordinate, value] : internal::SparseAccess::entries(matrix))
            {
                const Index row = coordinate.first;
                const Index col = coordinate.second;
                if ((UpLo == Lower && row < col) || (UpLo == Upper && row > col))
                {
                    continue;
                }
                rows.push_back(row);
                cols.push_back(col);
                values.push_back(value);
                if (row != col)
                {
                    rows.push_back(col);
                    cols.push_back(row);
                    values.push_back(value);
                }
            }
            return std::make_unique<internal::CsrStorage<Scalar, internal::Device::CPU>>(
                internal::cooToCsr(matrix.rows(), matrix.cols(), rows, cols, values));
        }

        Vector solveVector(const Vector& rhs, const Vector& guess) const
        {
            const auto settings = internal::currentExecutionSettings();
            const bool large_enough = _csr->nnz() >= 4096 || _dimension >= 1024;
            const bool wants_gpu = settings.policy == internal::ExecutionPolicy::GpuRequired ||
                                   settings.policy == internal::ExecutionPolicy::GpuPreferred ||
                                   (settings.policy == internal::ExecutionPolicy::Auto && large_enough);
            if (wants_gpu)
            {
                if constexpr (supportsResidentGpu)
                {
                    try
                    {
                        return solveVectorGpu(rhs, guess, settings);
                    }
                    catch (const internal::Error&)
                    {
                        if (settings.policy == internal::ExecutionPolicy::GpuRequired)
                        {
                            throw;
                        }
                    }
                }
                else if (settings.policy == internal::ExecutionPolicy::GpuRequired)
                {
                    throw internal::Error(
                        internal::ErrorCode::UnsupportedOperation,
                        "selected ConjugateGradient preconditioner has no resident GPU implementation",
                        settings.preferredGpu);
                }
            }
            return solveVectorCpu(rhs, guess);
        }

        Vector solveVectorGpu(const Vector& rhs, const Vector& guess, const internal::ExecutionSettings& settings) const
        {
            const internal::Backend candidates[] = {
                settings.preferredGpu,
                settings.preferredGpu == internal::Backend::Cuda ? internal::Backend::OpenCl : internal::Backend::Cuda,
                settings.preferredGpu == internal::Backend::Vulkan ? internal::Backend::OpenCl
                                                                   : internal::Backend::Vulkan};
            std::exception_ptr last_error;
            for (const internal::Backend candidate : candidates)
            {
                if (candidate == internal::Backend::Cpu)
                {
                    continue;
                }
                try
                {
                    auto context = internal::ExecutionContext::create({candidate, 0});
                    auto matrix = internal::ResidentCsrMatrix<Scalar>::copyFrom(*_csr, context);
                    auto device_rhs = internal::ResidentVector<Scalar>::copyFrom(rhs, context);
                    auto device_solution = internal::ResidentVector<Scalar>::copyFrom(guess, context);
                    internal::SolveOptions options;
                    options.maxIterations = static_cast<int>(maxIterations());
                    options.relativeTolerance = static_cast<double>(_tolerance);
                    options.useJacobiPreconditioner = std::is_same_v<Preconditioner, DiagonalPreconditioner<Scalar>>;
                    const internal::IterativeSolverReport report =
                        internal::pcg(matrix, device_rhs, device_solution, options, context);
                    Vector result = device_solution.toHostVector();
                    const double rhs_norm = static_cast<double>(rhs.norm());
                    _iterations = report.iterations;
                    _error = rhs_norm == 0.0
                                 ? (report.finalResidual == 0.0 ? Scalar{} : std::numeric_limits<Scalar>::infinity())
                                 : static_cast<Scalar>(report.finalResidual / rhs_norm);
                    _info = report.converged ? Success : NoConvergence;
                    _backend = candidate;
                    return result;
                }
                catch (...)
                {
                    last_error = std::current_exception();
                    if (settings.policy == internal::ExecutionPolicy::GpuRequired)
                    {
                        break;
                    }
                }
            }
            if (last_error)
            {
                std::rethrow_exception(last_error);
            }
            throw internal::Error(internal::ErrorCode::BackendUnavailable,
                                  "no resident GPU backend is available for ConjugateGradient",
                                  settings.preferredGpu);
        }

        Vector solveVectorCpu(const Vector& rhs, const Vector& guess) const
        {
            Vector result(guess);
            const double rhs_norm = static_cast<double>(rhs.norm());
            try
            {
                if constexpr (std::is_same_v<Preconditioner, IdentityPreconditioner> ||
                              std::is_same_v<Preconditioner, DiagonalPreconditioner<Scalar>>)
                {
                    internal::IterativeSolverOptions options;
                    options.maxIterations = static_cast<int>(maxIterations());
                    options.relativeTolerance = 0.0;
                    options.absoluteTolerance = static_cast<double>(_tolerance) * rhs_norm;
                    const internal::IterativeSolverReport report =
                        std::is_same_v<Preconditioner, IdentityPreconditioner>
                            ? internal::cg(*_csr, rhs, result, options)
                            : internal::pcg(*_csr, rhs, result, options);
                    updateReport(report.iterations, report.finalResidual, rhs_norm, report.converged);
                }
                else
                {
                    genericPcg(rhs, result, rhs_norm);
                }
                _backend = internal::Backend::Cpu;
            }
            catch (...)
            {
                _info = NumericalIssue;
                throw;
            }
            return result;
        }

        void genericPcg(const Vector& rhs, Vector& result, double rhs_norm) const
        {
            Vector residual = rhs - multiply(result);
            double residual_norm = static_cast<double>(residual.norm());
            const double target = static_cast<double>(_tolerance) * rhs_norm;
            if (residual_norm <= target)
            {
                updateReport(0, residual_norm, rhs_norm, true);
                return;
            }
            Vector transformed = _preconditioner.solve(residual);
            Vector direction(transformed);
            Scalar rho = residual.dot(transformed);
            for (Index iteration = 0; iteration < maxIterations(); ++iteration)
            {
                const Vector product = multiply(direction);
                const Scalar denominator = direction.dot(product);
                if (!std::isfinite(denominator) || denominator <= Scalar{})
                {
                    _info = NumericalIssue;
                    return;
                }
                const Scalar alpha = rho / denominator;
                result += direction * alpha;
                residual -= product * alpha;
                residual_norm = static_cast<double>(residual.norm());
                if (residual_norm <= target)
                {
                    updateReport(iteration + 1, residual_norm, rhs_norm, true);
                    return;
                }
                transformed = _preconditioner.solve(residual);
                const Scalar next_rho = residual.dot(transformed);
                direction = transformed + direction * (next_rho / rho);
                rho = next_rho;
            }
            updateReport(maxIterations(), residual_norm, rhs_norm, false);
        }

        Vector multiply(const Vector& input) const
        {
            Vector output = Vector::Zero(_dimension);
            for (Index row = 0; row < _dimension; ++row)
            {
                for (Index position = _csr->rowOffsets()[row]; position < _csr->rowOffsets()[row + 1]; ++position)
                {
                    output(row) += _csr->values()[position] * input(_csr->colIndices()[position]);
                }
            }
            return output;
        }

        void updateReport(Index iterations, double final_residual, double rhs_norm, bool converged) const
        {
            _iterations = iterations;
            _error = rhs_norm == 0.0 ? (final_residual == 0.0 ? Scalar{} : std::numeric_limits<Scalar>::infinity())
                                     : static_cast<Scalar>(final_residual / rhs_norm);
            _info = converged ? Success : NoConvergence;
        }

        std::unique_ptr<internal::CsrStorage<Scalar, internal::Device::CPU>> _csr;
        Preconditioner _preconditioner;
        bool _analysisIsOk = false;
        Index _dimension = 0;
        std::vector<sparse_detail::Coordinate> _pattern;
        Index _maxIterations = -1;
        Scalar _tolerance = std::numeric_limits<Scalar>::epsilon();
        mutable Index _iterations = 0;
        mutable Scalar _error = Scalar{};
        mutable ComputationInfo _info = InvalidInput;
        mutable internal::Backend _backend = internal::Backend::Cpu;
    };
} // namespace plamatrix::v1
