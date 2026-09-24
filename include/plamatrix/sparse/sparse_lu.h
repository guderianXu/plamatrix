#pragma once

#include <cmath>
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
    class SparseLU : public SparseSolverBase<SparseLU<MatrixType_, OrderingType_>>
    {
        using Base = SparseSolverBase<SparseLU<MatrixType_, OrderingType_>>;

    public:
        using MatrixType = MatrixType_;
        using OrderingType = OrderingType_;
        using Scalar = typename MatrixType::Scalar;
        using RealScalar = Scalar;
        using StorageIndex = typename MatrixType::StorageIndex;
        using PermutationType = PermutationMatrix<Dynamic, Dynamic, StorageIndex>;
        using FactorMatrix = SparseMatrix<Scalar, RowMajor, StorageIndex>;

        static_assert(std::is_floating_point_v<Scalar>, "SparseLU requires floating-point coefficients");

        SparseLU() = default;
        explicit SparseLU(const MatrixType& matrix)
        {
            compute(matrix);
        }

        void analyzePattern(const MatrixType& matrix)
        {
            sparse_detail::requireCpu("SparseLU::analyzePattern");
            resetFactorization();
            if (matrix.rows() != matrix.cols())
            {
                fail(InvalidInput, "SparseLU requires a square matrix");
                return;
            }
            _rows = matrix.rows();
            _pattern = sparse_detail::patternOf(matrix);
            OrderingType ordering;
            ordering(matrix, _columnPermutation);
            _analysisIsOk = true;
            _info = Success;
            _lastError.clear();
        }

        void factorize(const MatrixType& matrix)
        {
            sparse_detail::requireCpu("SparseLU::factorize");
            Base::_isInitialized = false;
            if (!_analysisIsOk)
            {
                fail(InvalidInput, "SparseLU::analyzePattern must be called before factorize");
                return;
            }
            if (matrix.rows() != _rows || matrix.cols() != _rows || sparse_detail::patternOf(matrix) != _pattern)
            {
                fail(InvalidInput, "SparseLU::factorize requires the analyzed sparsity pattern");
                return;
            }

            std::vector<Index> inverse_columns(static_cast<std::size_t>(_rows));
            for (Index column = 0; column < _rows; ++column)
            {
                inverse_columns[static_cast<std::size_t>(_columnPermutation.indices()(column))] = column;
            }
            _factors.assign(static_cast<std::size_t>(_rows), {});
            Scalar scale = Scalar{};
            for (Index outer = 0; outer < matrix.outerSize(); ++outer)
            {
                for (typename MatrixType::InnerIterator entry(matrix, outer); entry; ++entry)
                {
                    const Scalar value = static_cast<Scalar>(entry.value());
                    if (!std::isfinite(value))
                    {
                        fail(NumericalIssue, "SparseLU input contains non-finite coefficients");
                        return;
                    }
                    const Index column = inverse_columns[static_cast<std::size_t>(entry.col())];
                    _factors[static_cast<std::size_t>(entry.row())][column] = value;
                    scale = std::max(scale, std::abs(value));
                }
            }
            _rowPermutation = PermutationType(_rows);
            Base::_isInitialized = true;
            const Scalar tolerance =
                std::numeric_limits<Scalar>::epsilon() * scale * static_cast<Scalar>(std::max<Index>(Index{1}, _rows));
            for (Index column = 0; column < _rows; ++column)
            {
                Index pivot = column;
                Scalar largest = coefficientMagnitude(_factors[static_cast<std::size_t>(column)], column);
                for (Index row = column + 1; row < _rows; ++row)
                {
                    const Scalar magnitude = coefficientMagnitude(_factors[static_cast<std::size_t>(row)], column);
                    if (magnitude > largest)
                    {
                        largest = magnitude;
                        pivot = row;
                    }
                }
                const Scalar diagonal = coefficientMagnitude(_factors[static_cast<std::size_t>(column)], column);
                if (diagonal > tolerance && diagonal >= _pivotThreshold * largest)
                {
                    pivot = column;
                }
                if (largest <= tolerance)
                {
                    fail(NumericalIssue, "SparseLU matrix is singular to working precision");
                    return;
                }
                if (pivot != column)
                {
                    std::swap(_factors[static_cast<std::size_t>(pivot)], _factors[static_cast<std::size_t>(column)]);
                    std::swap(_rowPermutation.indices()(pivot), _rowPermutation.indices()(column));
                }
                const Scalar pivot_value = _factors[static_cast<std::size_t>(column)].at(column);
                for (Index row = column + 1; row < _rows; ++row)
                {
                    auto& target = _factors[static_cast<std::size_t>(row)];
                    const auto entry = target.find(column);
                    if (entry == target.end())
                    {
                        continue;
                    }
                    const Scalar factor = entry->second / pivot_value;
                    entry->second = factor;
                    for (const auto& [next, value] : _factors[static_cast<std::size_t>(column)])
                    {
                        if (next > column)
                        {
                            const Scalar updated = target[next] - factor * value;
                            if (updated == Scalar{})
                            {
                                target.erase(next);
                            }
                            else
                            {
                                target[next] = updated;
                            }
                        }
                    }
                }
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
            return _rows;
        }
        ComputationInfo info() const noexcept
        {
            return _info;
        }
        std::string lastErrorMessage() const
        {
            return _lastError;
        }
        void isSymmetric(bool symmetric) noexcept
        {
            _symmetricMode = symmetric;
        }
        void setPivotThreshold(const RealScalar& threshold)
        {
            if (!std::isfinite(threshold) || threshold < RealScalar{})
            {
                throw std::invalid_argument("SparseLU pivot threshold must be finite and non-negative");
            }
            _pivotThreshold = threshold;
        }

        const PermutationType& rowsPermutation() const noexcept
        {
            return _rowPermutation;
        }

        const PermutationType& colsPermutation() const noexcept
        {
            return _columnPermutation;
        }

        FactorMatrix matrixL() const
        {
            requireSuccessfulFactorization();
            FactorMatrix result(_rows, _rows);
            std::vector<Triplet<Scalar, StorageIndex>> triplets;
            for (Index row = 0; row < _rows; ++row)
            {
                triplets.emplace_back(static_cast<StorageIndex>(row), static_cast<StorageIndex>(row), Scalar{1});
                for (const auto& [column, value] : _factors[static_cast<std::size_t>(row)])
                {
                    if (column < row)
                    {
                        triplets.emplace_back(static_cast<StorageIndex>(row), static_cast<StorageIndex>(column), value);
                    }
                }
            }
            result.setFromTriplets(triplets.begin(), triplets.end());
            return result;
        }

        FactorMatrix matrixU() const
        {
            requireSuccessfulFactorization();
            FactorMatrix result(_rows, _rows);
            std::vector<Triplet<Scalar, StorageIndex>> triplets;
            for (Index row = 0; row < _rows; ++row)
            {
                for (const auto& [column, value] : _factors[static_cast<std::size_t>(row)])
                {
                    if (column >= row)
                    {
                        triplets.emplace_back(static_cast<StorageIndex>(row), static_cast<StorageIndex>(column), value);
                    }
                }
            }
            result.setFromTriplets(triplets.begin(), triplets.end());
            return result;
        }

        template <typename Rhs> auto _solve(const Rhs& right) const
        {
            sparse_detail::requireCpu("SparseLU::solve");
            requireSuccessfulFactorization();
            using RhsScalar = typename detail::DenseTraits<Rhs>::ScalarType;
            static_assert(std::is_same_v<RhsScalar, Scalar>, "SparseLU right-hand side scalar must match the matrix");
            constexpr int RightCols = detail::DenseTraits<Rhs>::ColsAtCompileTime;
            Matrix<Scalar, Dynamic, RightCols> result(_rows, right.cols());
            Matrix<Scalar, Dynamic, 1> work(_rows);
            if (!right.allFinite())
            {
                _info = InvalidInput;
                throw std::invalid_argument("SparseLU solve requires a finite right-hand side");
            }
            for (Index rhs_column = 0; rhs_column < right.cols(); ++rhs_column)
            {
                for (Index row = 0; row < _rows; ++row)
                {
                    work(row) = right(_rowPermutation.indices()(row), rhs_column);
                    for (const auto& [column, value] : _factors[static_cast<std::size_t>(row)])
                    {
                        if (column < row)
                        {
                            work(row) -= value * work(column);
                        }
                    }
                }
                for (Index row = _rows; row-- > 0;)
                {
                    const auto& factor_row = _factors[static_cast<std::size_t>(row)];
                    for (const auto& [column, value] : factor_row)
                    {
                        if (column > row)
                        {
                            work(row) -= value * work(column);
                        }
                    }
                    work(row) /= factor_row.at(row);
                }
                for (Index column = 0; column < _rows; ++column)
                {
                    result(_columnPermutation.indices()(column), rhs_column) = work(column);
                }
            }
            _info = result.allFinite() ? Success : NumericalIssue;
            return result;
        }

    private:
        static Scalar coefficientMagnitude(const std::map<Index, Scalar>& row, Index column)
        {
            const auto entry = row.find(column);
            return entry == row.end() ? Scalar{} : std::abs(entry->second);
        }

        void requireSuccessfulFactorization() const
        {
            if (_info != Success || !Base::_isInitialized)
            {
                throw std::logic_error("SparseLU has no successful factorization");
            }
        }

        void resetFactorization()
        {
            Base::_isInitialized = false;
            _analysisIsOk = false;
            _rows = 0;
            _pattern.clear();
            _factors.clear();
            _rowPermutation = PermutationType{};
            _columnPermutation = PermutationType{};
            _info = InvalidInput;
        }

        void fail(ComputationInfo info, std::string message)
        {
            _info = info;
            _lastError = std::move(message);
        }

        Index _rows = 0;
        bool _analysisIsOk = false;
        bool _symmetricMode = false;
        RealScalar _pivotThreshold = RealScalar{1};
        std::vector<sparse_detail::Coordinate> _pattern;
        std::vector<std::map<Index, Scalar>> _factors;
        PermutationType _rowPermutation;
        PermutationType _columnPermutation;
        mutable ComputationInfo _info = InvalidInput;
        std::string _lastError;
    };
} // namespace plamatrix::v1
