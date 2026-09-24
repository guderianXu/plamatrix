#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "plamatrix/sparse/detail/sparse_direct_common.h"
#include "plamatrix/sparse/preconditioners.h"

namespace plamatrix::v1
{
    /// Eigen-style preconditioned BiCGSTAB for general square sparse systems.
    template <typename MatrixType, typename Preconditioner>
    class BiCGSTAB : public SparseSolverBase<BiCGSTAB<MatrixType, Preconditioner>>
    {
        using Base = SparseSolverBase<BiCGSTAB<MatrixType, Preconditioner>>;
        using Scalar = typename MatrixType::Scalar;
        static_assert(std::is_floating_point_v<Scalar>, "BiCGSTAB requires floating-point coefficients");

    public:
        using Vector = Matrix<Scalar, Dynamic, 1>;

        BiCGSTAB() = default;
        explicit BiCGSTAB(const MatrixType& matrix)
        {
            compute(matrix);
        }

        BiCGSTAB& analyzePattern(const MatrixType& matrix)
        {
            _csr.reset();
            Base::_isInitialized = false;
            _analysisIsOk = false;
            _dimension = 0;
            _info = InvalidInput;
            if (matrix.rows() != matrix.cols())
            {
                throw std::invalid_argument("BiCGSTAB requires a square matrix");
            }
            _dimension = matrix.rows();
            _pattern = sparse_detail::patternOf(matrix);
            _preconditioner.analyzePattern(matrix);
            if (_preconditioner.info() != Success)
            {
                _info = _preconditioner.info();
                return *this;
            }
            _analysisIsOk = true;
            _info = Success;
            return *this;
        }

        BiCGSTAB& factorize(const MatrixType& matrix)
        {
            _csr.reset();
            Base::_isInitialized = false;
            if (!_analysisIsOk || matrix.rows() != _dimension || matrix.cols() != _dimension ||
                sparse_detail::patternOf(matrix) != _pattern)
            {
                _info = InvalidInput;
                return *this;
            }
            _csr = std::make_unique<internal::CsrStorage<Scalar, internal::Device::CPU>>(
                internal::SparseAccess::csrSnapshot(matrix));
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
            return *this;
        }

        BiCGSTAB& compute(const MatrixType& matrix)
        {
            analyzePattern(matrix);
            if (_analysisIsOk)
            {
                factorize(matrix);
            }
            return *this;
        }

        BiCGSTAB& setMaxIterations(Index count)
        {
            if (count < 0 || count > static_cast<Index>(std::numeric_limits<int>::max()))
            {
                throw std::invalid_argument("BiCGSTAB max iterations are out of range");
            }
            _maxIterations = count;
            return *this;
        }

        BiCGSTAB& setTolerance(Scalar tolerance)
        {
            if (!std::isfinite(tolerance) || tolerance < Scalar{})
            {
                throw std::invalid_argument("BiCGSTAB tolerance must be finite and non-negative");
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
            const auto settings = internal::currentExecutionSettings();
            if (settings.policy == internal::ExecutionPolicy::GpuRequired)
            {
                throw internal::Error(internal::ErrorCode::UnsupportedOperation,
                                      "BiCGSTAB has no resident GPU implementation",
                                      settings.preferredGpu);
            }
            const auto& rhs = rhs_base.derived();
            const auto& guess = guess_base.derived();
            using RhsScalar = typename detail::DenseTraits<Rhs>::ScalarType;
            using GuessScalar = typename detail::DenseTraits<Guess>::ScalarType;
            static_assert(std::is_same_v<RhsScalar, Scalar> && std::is_same_v<GuessScalar, Scalar>,
                          "BiCGSTAB rhs and guess scalars must match the matrix");
            constexpr int RightCols = detail::DenseTraits<Rhs>::ColsAtCompileTime;
            if (!_csr)
            {
                _info = InvalidInput;
                throw std::logic_error("BiCGSTAB::compute must be called before solve");
            }
            if (rhs.rows() != _dimension || guess.rows() != _dimension || rhs.cols() != guess.cols())
            {
                _info = InvalidInput;
                throw std::invalid_argument("BiCGSTAB rhs or guess has the wrong size");
            }
            Matrix<Scalar, Dynamic, RightCols> result(_dimension, rhs.cols());
            _iterations = 0;
            _error = Scalar{};
            for (Index column = 0; column < rhs.cols(); ++column)
            {
                Vector rhs_column(_dimension);
                Vector solution(_dimension);
                for (Index row = 0; row < _dimension; ++row)
                {
                    rhs_column(row) = rhs(row, column);
                    solution(row) = guess(row, column);
                }
                solveVector(rhs_column, solution);
                for (Index row = 0; row < _dimension; ++row)
                {
                    result(row, column) = solution(row);
                }
            }
            return result;
        }

    private:
        void solveVector(const Vector& rhs, Vector& solution) const
        {
            const double rhs_norm = static_cast<double>(rhs.norm());
            const double target = static_cast<double>(_tolerance) * rhs_norm;
            Vector residual = rhs - multiply(solution);
            double residual_norm = static_cast<double>(residual.norm());
            if (residual_norm <= target)
            {
                updateReport(0, residual_norm, rhs_norm, true);
                return;
            }
            Vector shadow(residual);
            Vector direction = Vector::Zero(_dimension);
            Vector product = Vector::Zero(_dimension);
            Scalar rho_previous = Scalar{1};
            Scalar alpha = Scalar{1};
            Scalar omega = Scalar{1};
            const Scalar breakdown = std::numeric_limits<Scalar>::epsilon();
            for (Index iteration = 0; iteration < maxIterations(); ++iteration)
            {
                const Scalar rho = shadow.dot(residual);
                if (!std::isfinite(rho) || std::abs(rho) <= breakdown)
                {
                    updateReport(iteration, residual_norm, rhs_norm, false, NumericalIssue);
                    return;
                }
                const Scalar beta = (rho / rho_previous) * (alpha / omega);
                direction = residual + (direction - product * omega) * beta;
                const Vector transformed_direction = _preconditioner.solve(direction);
                product = multiply(transformed_direction);
                const Scalar denominator = shadow.dot(product);
                if (!std::isfinite(denominator) || std::abs(denominator) <= breakdown)
                {
                    updateReport(iteration, residual_norm, rhs_norm, false, NumericalIssue);
                    return;
                }
                alpha = rho / denominator;
                Vector intermediate = residual - product * alpha;
                const double intermediate_norm = static_cast<double>(intermediate.norm());
                if (intermediate_norm <= target)
                {
                    solution += transformed_direction * alpha;
                    updateReport(iteration + 1, intermediate_norm, rhs_norm, true);
                    return;
                }
                const Vector transformed_intermediate = _preconditioner.solve(intermediate);
                const Vector second_product = multiply(transformed_intermediate);
                const Scalar second_norm = second_product.dot(second_product);
                if (!std::isfinite(second_norm) || second_norm <= breakdown)
                {
                    updateReport(iteration, residual_norm, rhs_norm, false, NumericalIssue);
                    return;
                }
                omega = second_product.dot(intermediate) / second_norm;
                if (!std::isfinite(omega) || std::abs(omega) <= breakdown)
                {
                    updateReport(iteration, residual_norm, rhs_norm, false, NumericalIssue);
                    return;
                }
                solution += transformed_direction * alpha + transformed_intermediate * omega;
                residual = intermediate - second_product * omega;
                residual_norm = static_cast<double>(residual.norm());
                if (residual_norm <= target)
                {
                    updateReport(iteration + 1, residual_norm, rhs_norm, true);
                    return;
                }
                rho_previous = rho;
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

        void updateReport(Index iterations,
                          double final_residual,
                          double rhs_norm,
                          bool converged,
                          ComputationInfo failure = NoConvergence) const
        {
            _iterations = iterations;
            _error = rhs_norm == 0.0 ? (final_residual == 0.0 ? Scalar{} : std::numeric_limits<Scalar>::infinity())
                                     : static_cast<Scalar>(final_residual / rhs_norm);
            _info = converged ? Success : failure;
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
    };
} // namespace plamatrix::v1
