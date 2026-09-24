#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include "plamatrix/dense/permutation_matrix.h"
#include "plamatrix/dense/detail/static_or_dynamic_vector.h"
#include "plamatrix/dense/solver_base.h"
#include "plamatrix/internal/ops/decomposition.h"

namespace plamatrix::v1
{
    template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols, int UpLo>
    class LLT<Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>, UpLo>
        : public SolverBase<LLT<Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>, UpLo>>
    {
        static_assert(UpLo == Lower || UpLo == Upper, "LLT requires Lower or Upper");

    public:
        using MatrixType = Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>;
        using RealScalar = typename NumTraits<Scalar>::Real;
        LLT() = default;
        explicit LLT(Index size) : _factor(size, size)
        {
        }
        explicit LLT(const MatrixType& input)
        {
            compute(input);
        }

        LLT& compute(const MatrixType& input)
        {
            _initialized = true;
            _info = NumericalIssue;
            if (input.rows() != input.cols() || !input.allFinite())
            {
                _info = InvalidInput;
                return *this;
            }
#ifdef PLAMATRIX_WITH_CUDA
            const auto settings = internal::currentExecutionSettings();
            const long double work = static_cast<long double>(input.rows()) * input.rows() * input.rows();
            const bool use_cuda =
                internal::detail::GpuOps<Scalar>::available(internal::Backend::Cuda) &&
                settings.policy != internal::ExecutionPolicy::CpuOnly &&
                ((settings.policy == internal::ExecutionPolicy::GpuRequired &&
                  settings.preferredGpu == internal::Backend::Cuda) ||
                 (settings.policy == internal::ExecutionPolicy::GpuPreferred &&
                  settings.preferredGpu == internal::Backend::Cuda) ||
                 (settings.policy == internal::ExecutionPolicy::Auto && work >= 4.0L * 1024.0L * 1024.0L));
            if (use_cuda)
            {
                try
                {
                    internal::DenseStorage<Scalar, internal::Device::CPU> host(input.rows(), input.cols());
                    for (Index column = 0; column < input.cols(); ++column)
                        for (Index row = 0; row < input.rows(); ++row)
                            host(row, column) = row >= column
                                                    ? (UpLo == Lower ? input(row, column) : input(column, row))
                                                    : (UpLo == Lower ? input(column, row) : input(row, column));
                    auto gpu_factor = internal::cholesky(host.toGpu());
                    auto factor = gpu_factor.toCpu();
                    _factor = MatrixType::Zero(input.rows(), input.cols());
                    for (Index column = 0; column < input.cols(); ++column)
                        for (Index row = column; row < input.rows(); ++row)
                            _factor(row, column) = factor(row, column);
                    _info = Success;
                    _backend = internal::Backend::Cuda;
                    return *this;
                }
                catch (const std::runtime_error&)
                {
                    _info = NumericalIssue;
                    return *this;
                }
            }
            if (settings.policy == internal::ExecutionPolicy::GpuRequired)
            {
                throw internal::Error(internal::ErrorCode::UnsupportedOperation,
                                      "LLT is unavailable on the selected GPU",
                                      settings.preferredGpu);
            }
#else
            if (internal::currentExecutionSettings().policy == internal::ExecutionPolicy::GpuRequired)
                throw internal::Error(internal::ErrorCode::UnsupportedOperation, "LLT requires CUDA for GPU execution");
#endif
            _factor = MatrixType::Zero(input.rows(), input.cols());
            for (Index column = 0; column < input.cols(); ++column)
            {
                for (Index row = column; row < input.rows(); ++row)
                {
                    Scalar value = UpLo == Lower ? input(row, column) : input(column, row);
                    for (Index inner = 0; inner < column; ++inner)
                    {
                        value -= _factor(row, inner) * _factor(column, inner);
                    }
                    if (row == column)
                    {
                        if (!(value > Scalar{}) || !std::isfinite(value))
                        {
                            return *this;
                        }
                        _factor(column, column) = std::sqrt(value);
                    }
                    else
                    {
                        _factor(row, column) = value / _factor(column, column);
                    }
                }
            }
            _info = Success;
            _backend = internal::Backend::Cpu;
            return *this;
        }

        MatrixType matrixL() const
        {
            MatrixType result = MatrixType::Zero(_factor.rows(), _factor.cols());
            for (Index column = 0; column < result.cols(); ++column)
            {
                for (Index row = column; row < result.rows(); ++row)
                {
                    result(row, column) = _factor(row, column);
                }
            }
            return result;
        }

        MatrixType matrixU() const
        {
            return matrixL().transpose().eval();
        }
        ComputationInfo info() const noexcept
        {
            return _info;
        }
        internal::Backend backend() const noexcept
        {
            return _backend;
        }
        bool isInitialized() const noexcept
        {
            return _initialized;
        }

        template <typename Rhs> auto _solve(const Rhs& right) const
        {
            if (_info != Success || right.rows() != _factor.rows())
            {
                throw std::runtime_error("LLT solve requires a successful factorization and matching right-hand side");
            }
            Matrix<Scalar, Rows, Rhs::ColsAtCompileTime> result(right);
            for (Index column = 0; column < result.cols(); ++column)
            {
                for (Index row = 0; row < result.rows(); ++row)
                {
                    for (Index inner = 0; inner < row; ++inner)
                    {
                        result(row, column) -= _factor(row, inner) * result(inner, column);
                    }
                    result(row, column) /= _factor(row, row);
                }
                for (Index row = result.rows(); row-- > 0;)
                {
                    for (Index inner = row + 1; inner < result.rows(); ++inner)
                    {
                        result(row, column) -= _factor(inner, row) * result(inner, column);
                    }
                    result(row, column) /= _factor(row, row);
                }
            }
            return result;
        }

        template <typename Node> auto solve(const DenseExpression<Node>& right) const
        {
            return _solve(right.eval());
        }
        template <typename Rhs> auto solve(const MatrixBase<Rhs>& right) const
        {
            return _solve(right.derived());
        }

    private:
        MatrixType _factor;
        ComputationInfo _info = InvalidInput;
        internal::Backend _backend = internal::Backend::Cpu;
        bool _initialized = false;
    };

    template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols, typename PermutationIndex>
    class FullPivLU<Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>, PermutationIndex>
        : public SolverBase<FullPivLU<Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>, PermutationIndex>>
    {
    public:
        using MatrixType = Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>;
        using RealScalar = typename NumTraits<Scalar>::Real;
        using PermutationPType = PermutationMatrix<Rows, MaxRows, PermutationIndex>;
        using PermutationQType = PermutationMatrix<Cols, MaxCols, PermutationIndex>;
        FullPivLU() = default;
        explicit FullPivLU(Index rows, Index columns) : _lu(rows, columns)
        {
        }
        explicit FullPivLU(const MatrixType& input)
        {
            compute(input);
        }

        FullPivLU& compute(const MatrixType& input)
        {
            if (!input.allFinite())
            {
                throw std::invalid_argument("FullPivLU requires finite values");
            }
            _lu = input;
            _matrixOneNorm = RealScalar{};
            for (Index column = 0; column < input.cols(); ++column)
            {
                RealScalar column_sum{};
                for (Index row = 0; row < input.rows(); ++row)
                {
                    column_sum += std::abs(input(row, column));
                }
                _matrixOneNorm = std::max(_matrixOneNorm, column_sum);
            }
            decomposition_detail::StaticOrDynamicVector<PermutationIndex, Rows> rows(
                static_cast<std::size_t>(input.rows()));
            decomposition_detail::StaticOrDynamicVector<PermutationIndex, Cols> columns(
                static_cast<std::size_t>(input.cols()));
            std::iota(rows.begin(), rows.end(), PermutationIndex{});
            std::iota(columns.begin(), columns.end(), PermutationIndex{});
            const Index steps = std::min(input.rows(), input.cols());
            RealScalar largest{};
            for (Index index = 0; index < input.size(); ++index)
            {
                largest = std::max(largest, std::abs(input.data()[index]));
            }
            _threshold = _prescribedThreshold >= RealScalar{}
                             ? _prescribedThreshold
                             : std::numeric_limits<RealScalar>::epsilon() *
                                   static_cast<RealScalar>(std::max(input.rows(), input.cols()));
            _rank = 0;
            _detSign = Scalar{1};
            for (Index step = 0; step < steps; ++step)
            {
                Index pivot_row = step;
                Index pivot_column = step;
                RealScalar pivot{};
                for (Index column = step; column < input.cols(); ++column)
                {
                    for (Index row = step; row < input.rows(); ++row)
                    {
                        if (std::abs(_lu(row, column)) > pivot)
                        {
                            pivot = std::abs(_lu(row, column));
                            pivot_row = row;
                            pivot_column = column;
                        }
                    }
                }
                if (!(pivot > _threshold * largest))
                {
                    break;
                }
                ++_rank;
                if (pivot_row != step)
                {
                    for (Index column = 0; column < input.cols(); ++column)
                    {
                        std::swap(_lu(step, column), _lu(pivot_row, column));
                    }
                    std::swap(rows[static_cast<std::size_t>(step)], rows[static_cast<std::size_t>(pivot_row)]);
                    _detSign = -_detSign;
                }
                if (pivot_column != step)
                {
                    for (Index row = 0; row < input.rows(); ++row)
                    {
                        std::swap(_lu(row, step), _lu(row, pivot_column));
                    }
                    std::swap(columns[static_cast<std::size_t>(step)], columns[static_cast<std::size_t>(pivot_column)]);
                    _detSign = -_detSign;
                }
                for (Index row = step + 1; row < input.rows(); ++row)
                {
                    _lu(row, step) /= _lu(step, step);
                    for (Index column = step + 1; column < input.cols(); ++column)
                    {
                        _lu(row, column) -= _lu(row, step) * _lu(step, column);
                    }
                }
            }
            _p = PermutationPType(rows);
            _q = PermutationQType(columns);
            _initialized = true;
            return *this;
        }

        FullPivLU& setThreshold(RealScalar threshold)
        {
            _prescribedThreshold = threshold;
            return *this;
        }
        RealScalar threshold() const noexcept
        {
            return _threshold;
        }
        Index rank() const noexcept
        {
            return _rank;
        }
        Index dimensionOfKernel() const noexcept
        {
            return _lu.cols() - _rank;
        }
        bool isInjective() const noexcept
        {
            return _rank == _lu.cols();
        }
        bool isSurjective() const noexcept
        {
            return _rank == _lu.rows();
        }
        bool isInvertible() const noexcept
        {
            return _lu.rows() == _lu.cols() && isInjective();
        }
        Scalar determinant() const
        {
            if (_lu.rows() != _lu.cols())
                throw std::logic_error("determinant requires a square matrix");
            if (!isInvertible())
                return Scalar{};
            Scalar result = _detSign;
            for (Index index = 0; index < _lu.rows(); ++index)
                result *= _lu(index, index);
            return result;
        }
        const MatrixType& matrixLU() const noexcept
        {
            return _lu;
        }
        const PermutationPType& permutationP() const noexcept
        {
            return _p;
        }
        const PermutationQType& permutationQ() const noexcept
        {
            return _q;
        }
        bool isInitialized() const noexcept
        {
            return _initialized;
        }

        RealScalar rcond() const
        {
            if (!isInvertible() || _matrixOneNorm == RealScalar{})
            {
                return RealScalar{};
            }
            using IdentityType = Matrix<Scalar, Rows, Rows, Options, MaxRows, MaxRows>;
            const auto inverse = _solve(IdentityType::Identity(_lu.rows(), _lu.rows()));
            RealScalar inverse_norm{};
            for (Index column = 0; column < inverse.cols(); ++column)
            {
                RealScalar column_sum{};
                for (Index row = 0; row < inverse.rows(); ++row)
                {
                    column_sum += std::abs(inverse(row, column));
                }
                inverse_norm = std::max(inverse_norm, column_sum);
            }
            return inverse_norm == RealScalar{} ? RealScalar{} : RealScalar{1} / (_matrixOneNorm * inverse_norm);
        }

        Matrix<Scalar, Cols, Dynamic, ColMajor, MaxCols, Dynamic> kernel() const
        {
            const Index nullity = dimensionOfKernel();
            Matrix<Scalar, Cols, Dynamic, ColMajor, MaxCols, Dynamic> result(_lu.cols(), nullity);
            for (Index basis = 0; basis < nullity; ++basis)
            {
                decomposition_detail::StaticOrDynamicVector<Scalar, Cols> permuted(static_cast<std::size_t>(_lu.cols()),
                                                                                   Scalar{});
                permuted[static_cast<std::size_t>(_rank + basis)] = Scalar{1};
                for (Index row = _rank; row-- > 0;)
                {
                    Scalar value = -_lu(row, _rank + basis);
                    for (Index column = row + 1; column < _rank; ++column)
                        value -= _lu(row, column) * permuted[static_cast<std::size_t>(column)];
                    permuted[static_cast<std::size_t>(row)] = value / _lu(row, row);
                }
                for (Index row = 0; row < _lu.cols(); ++row)
                    result(_q.indices()(row), basis) = permuted[static_cast<std::size_t>(row)];
            }
            return result;
        }

        template <typename Original>
        Matrix<Scalar, Rows, Dynamic, ColMajor, MaxRows, Dynamic> image(const MatrixBase<Original>& original) const
        {
            if (original.rows() != _lu.rows() || original.cols() != _lu.cols())
                throw std::invalid_argument("image() requires the factorized matrix shape");
            Matrix<Scalar, Rows, Dynamic, ColMajor, MaxRows, Dynamic> result(_lu.rows(), _rank);
            for (Index column = 0; column < _rank; ++column)
                for (Index row = 0; row < _lu.rows(); ++row)
                    result(row, column) = original.derived()(row, _q.indices()(column));
            return result;
        }

        template <typename Rhs> auto _solve(const Rhs& right) const
        {
            if (!isInvertible() || right.rows() != _lu.rows())
            {
                throw std::runtime_error("FullPivLU solve requires an invertible square matrix");
            }
            Matrix<Scalar, Rows, Rhs::ColsAtCompileTime> work(_lu.rows(), right.cols());
            Matrix<Scalar, Cols, Rhs::ColsAtCompileTime> result(_lu.cols(), right.cols());
            for (Index column = 0; column < right.cols(); ++column)
            {
                for (Index row = 0; row < _lu.rows(); ++row)
                {
                    work(row, column) = right(_p.indices()(row), column);
                    for (Index previous = 0; previous < row; ++previous)
                    {
                        work(row, column) -= _lu(row, previous) * work(previous, column);
                    }
                }
                for (Index row = _lu.rows(); row-- > 0;)
                {
                    for (Index next = row + 1; next < _lu.rows(); ++next)
                    {
                        work(row, column) -= _lu(row, next) * work(next, column);
                    }
                    work(row, column) /= _lu(row, row);
                }
                for (Index row = 0; row < _lu.cols(); ++row)
                {
                    result(_q.indices()(row), column) = work(row, column);
                }
            }
            return result;
        }

        template <typename Node> auto solve(const DenseExpression<Node>& right) const
        {
            return _solve(right.eval());
        }
        template <typename Rhs> auto solve(const MatrixBase<Rhs>& right) const
        {
            return _solve(right.derived());
        }

    private:
        MatrixType _lu;
        PermutationPType _p;
        PermutationQType _q;
        RealScalar _threshold{};
        RealScalar _prescribedThreshold = RealScalar{-1};
        Scalar _detSign = Scalar{1};
        RealScalar _matrixOneNorm{};
        Index _rank = 0;
        bool _initialized = false;
    };
} // namespace plamatrix::v1
