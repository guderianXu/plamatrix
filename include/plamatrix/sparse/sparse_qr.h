#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "plamatrix/sparse/detail/sparse_direct_common.h"
#include "plamatrix/sparse/ordering_methods.h"

namespace plamatrix::v1
{
    template <typename MatrixType_, typename OrderingType_>
    class SparseQR : public SparseSolverBase<SparseQR<MatrixType_, OrderingType_>>
    {
        using Base = SparseSolverBase<SparseQR<MatrixType_, OrderingType_>>;

    public:
        using MatrixType = MatrixType_;
        using OrderingType = OrderingType_;
        using Scalar = typename MatrixType::Scalar;
        using RealScalar = Scalar;
        using StorageIndex = typename MatrixType::StorageIndex;
        using PermutationType = PermutationMatrix<Dynamic, Dynamic, StorageIndex>;

        static_assert(std::is_floating_point_v<Scalar>, "SparseQR requires floating-point coefficients");

        SparseQR() = default;
        explicit SparseQR(const MatrixType& matrix)
        {
            compute(matrix);
        }

        void analyzePattern(const MatrixType& matrix)
        {
            sparse_detail::requireCpu("SparseQR::analyzePattern");
            reset();
            _rows = matrix.rows();
            _cols = matrix.cols();
            _pattern = sparse_detail::patternOf(matrix);
            OrderingType ordering;
            ordering(matrix, _orderingPermutation);
            _analysisIsOk = true;
            _info = Success;
            _lastError.clear();
        }

        void factorize(const MatrixType& matrix)
        {
            sparse_detail::requireCpu("SparseQR::factorize");
            Base::_isInitialized = false;
            if (!_analysisIsOk)
            {
                fail(InvalidInput, "SparseQR::analyzePattern must be called before factorize");
                return;
            }
            if (matrix.rows() != _rows || matrix.cols() != _cols || sparse_detail::patternOf(matrix) != _pattern)
            {
                fail(InvalidInput, "SparseQR::factorize requires the analyzed sparsity pattern");
                return;
            }

            std::vector<Index> inverse_columns(static_cast<std::size_t>(_cols));
            for (Index column = 0; column < _cols; ++column)
            {
                inverse_columns[static_cast<std::size_t>(_orderingPermutation.indices()(column))] = column;
            }
            std::vector<std::map<Index, Scalar>> work(static_cast<std::size_t>(_cols));
            for (Index outer = 0; outer < matrix.outerSize(); ++outer)
            {
                for (typename MatrixType::InnerIterator entry(matrix, outer); entry; ++entry)
                {
                    const Scalar value = static_cast<Scalar>(entry.value());
                    if (!std::isfinite(value))
                    {
                        fail(NumericalIssue, "SparseQR input contains non-finite coefficients");
                        return;
                    }
                    const Index column = inverse_columns[static_cast<std::size_t>(entry.col())];
                    work[static_cast<std::size_t>(column)][entry.row()] = value;
                }
            }
            Base::_isInitialized = true;
            const Index diagonal_size = std::min(_rows, _cols);
            _q = Matrix<Scalar, Dynamic, Dynamic>::Zero(_rows, diagonal_size);
            _r = Matrix<Scalar, Dynamic, Dynamic>::Zero(diagonal_size, _cols);
            _permutation = _orderingPermutation;
            for (Index col = 0; col < _cols; ++col)
            {
                _permutation.indices()(col) = _orderingPermutation.indices()(col);
            }

            Scalar largest_norm = Scalar{};
            for (Index col = 0; col < _cols; ++col)
            {
                largest_norm = std::max(largest_norm, columnNorm(work[static_cast<std::size_t>(col)]));
            }
            const Scalar threshold = _useDefaultThreshold
                                         ? std::numeric_limits<Scalar>::epsilon() *
                                               static_cast<Scalar>(std::max<Index>(Index{1}, std::max(_rows, _cols))) *
                                               largest_norm
                                         : _threshold;
            _rank = 0;
            for (Index step = 0; step < diagonal_size; ++step)
            {
                Index pivot = step;
                Scalar pivot_norm = columnNorm(work[static_cast<std::size_t>(step)]);
                for (Index col = step + 1; col < _cols; ++col)
                {
                    const Scalar norm = columnNorm(work[static_cast<std::size_t>(col)]);
                    if (norm > pivot_norm)
                    {
                        pivot = col;
                        pivot_norm = norm;
                    }
                }
                if (pivot_norm <= threshold)
                {
                    break;
                }
                std::swap(work[static_cast<std::size_t>(step)], work[static_cast<std::size_t>(pivot)]);
                std::swap(_permutation.indices()(step), _permutation.indices()(pivot));
                for (Index row = 0; row < step; ++row)
                {
                    std::swap(_r(row, step), _r(row, pivot));
                }

                _r(step, step) = pivot_norm;
                for (const auto& [row, value] : work[static_cast<std::size_t>(step)])
                {
                    _q(row, step) = value / pivot_norm;
                }
                for (Index col = step + 1; col < _cols; ++col)
                {
                    Scalar projection = Scalar{};
                    auto& target = work[static_cast<std::size_t>(col)];
                    for (const auto& [row, value] : target)
                    {
                        projection += _q(row, step) * value;
                    }
                    _r(step, col) = projection;
                    for (Index row = 0; row < _rows; ++row)
                    {
                        if (_q(row, step) == Scalar{})
                        {
                            continue;
                        }
                        const Scalar updated = target[row] - _q(row, step) * projection;
                        if (updated == Scalar{})
                        {
                            target.erase(row);
                        }
                        else
                        {
                            target[row] = updated;
                        }
                    }
                }
                ++_rank;
            }
            _info = Success;
            _lastError.clear();
        }

        void compute(const MatrixType& matrix)
        {
            analyzePattern(matrix);
            if (_analysisIsOk)
            {
                factorize(matrix);
            }
        }

        Index rows() const noexcept
        {
            return _rows;
        }
        Index cols() const noexcept
        {
            return _cols;
        }
        Index rank() const
        {
            requireSuccessfulFactorization();
            return _rank;
        }
        ComputationInfo info() const noexcept
        {
            return _info;
        }
        std::string lastErrorMessage() const
        {
            return _lastError;
        }
        const Matrix<Scalar, Dynamic, Dynamic>& matrixR() const
        {
            requireSuccessfulFactorization();
            return _r;
        }
        const PermutationType& colsPermutation() const
        {
            requireSuccessfulFactorization();
            return _permutation;
        }
        void setPivotThreshold(const RealScalar& threshold)
        {
            if (!std::isfinite(threshold) || threshold < RealScalar{})
            {
                throw std::invalid_argument("SparseQR pivot threshold must be finite and non-negative");
            }
            _useDefaultThreshold = false;
            _threshold = threshold;
        }

        template <typename Rhs> auto _solve(const Rhs& right) const
        {
            sparse_detail::requireCpu("SparseQR::solve");
            requireSuccessfulFactorization();
            using RhsScalar = typename detail::DenseTraits<Rhs>::ScalarType;
            static_assert(std::is_same_v<RhsScalar, Scalar>, "SparseQR right-hand side scalar must match the matrix");
            constexpr int RightCols = detail::DenseTraits<Rhs>::ColsAtCompileTime;
            Matrix<Scalar, Dynamic, RightCols> result = Matrix<Scalar, Dynamic, RightCols>::Zero(_cols, right.cols());
            if (!right.allFinite())
            {
                _info = InvalidInput;
                throw std::invalid_argument("SparseQR solve requires a finite right-hand side");
            }
            Matrix<Scalar, Dynamic, 1> coefficients = Matrix<Scalar, Dynamic, 1>::Zero(_cols);
            for (Index rhs_col = 0; rhs_col < right.cols(); ++rhs_col)
            {
                coefficients.setZero();
                for (Index row = 0; row < _rank; ++row)
                {
                    for (Index source_row = 0; source_row < _rows; ++source_row)
                    {
                        coefficients(row) += _q(source_row, row) * right(source_row, rhs_col);
                    }
                }
                for (Index row = _rank; row-- > 0;)
                {
                    for (Index next = row + 1; next < _rank; ++next)
                    {
                        coefficients(row) -= _r(row, next) * coefficients(next);
                    }
                    coefficients(row) /= _r(row, row);
                }
                for (Index index = 0; index < _cols; ++index)
                {
                    result(_permutation.indices()(index), rhs_col) = coefficients(index);
                }
            }
            _info = result.allFinite() ? Success : NumericalIssue;
            return result;
        }

    private:
        static Scalar columnNorm(const std::map<Index, Scalar>& column)
        {
            Scalar squared_norm = Scalar{};
            for (const auto& [row, value] : column)
            {
                static_cast<void>(row);
                squared_norm += value * value;
            }
            return std::sqrt(squared_norm);
        }

        void reset()
        {
            Base::_isInitialized = false;
            _analysisIsOk = false;
            _rows = 0;
            _cols = 0;
            _rank = 0;
            _pattern.clear();
            _permutation = PermutationType{};
            _orderingPermutation = PermutationType{};
            _q.resize(0, 0);
            _r.resize(0, 0);
            _info = InvalidInput;
        }

        void fail(ComputationInfo info, std::string message)
        {
            _info = info;
            _lastError = std::move(message);
        }

        void requireSuccessfulFactorization() const
        {
            if (!Base::_isInitialized || _info != Success)
            {
                throw std::logic_error("SparseQR has no successful factorization");
            }
        }

        Index _rows = 0;
        Index _cols = 0;
        Index _rank = 0;
        bool _analysisIsOk = false;
        bool _useDefaultThreshold = true;
        RealScalar _threshold = RealScalar{};
        std::vector<sparse_detail::Coordinate> _pattern;
        PermutationType _permutation;
        PermutationType _orderingPermutation;
        Matrix<Scalar, Dynamic, Dynamic> _q;
        Matrix<Scalar, Dynamic, Dynamic> _r;
        mutable ComputationInfo _info = InvalidInput;
        std::string _lastError;
    };
} // namespace plamatrix::v1
