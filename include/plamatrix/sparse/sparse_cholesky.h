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
    namespace sparse_detail
    {
        template <typename Derived> struct CholeskyTraits;

        template <typename MatrixType, int UpLo, typename Ordering>
        struct CholeskyTraits<SimplicialLLT<MatrixType, UpLo, Ordering>>
        {
            using Matrix = MatrixType;
            using OrderingType = Ordering;
            static constexpr int Triangle = UpLo;
        };

        template <typename MatrixType, int UpLo, typename Ordering>
        struct CholeskyTraits<SimplicialLDLT<MatrixType, UpLo, Ordering>>
        {
            using Matrix = MatrixType;
            using OrderingType = Ordering;
            static constexpr int Triangle = UpLo;
        };
    } // namespace sparse_detail

    template <typename Derived> class SimplicialCholeskyBase : public SparseSolverBase<Derived>
    {
        using SolverBase = SparseSolverBase<Derived>;
        using Traits = sparse_detail::CholeskyTraits<Derived>;

    public:
        using MatrixType = typename Traits::Matrix;
        using OrderingType = typename Traits::OrderingType;
        using Scalar = typename MatrixType::Scalar;
        using RealScalar = Scalar;
        using StorageIndex = typename MatrixType::StorageIndex;
        using PermutationType = PermutationMatrix<Dynamic, Dynamic, StorageIndex>;

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

        Derived& setShift(const RealScalar& offset, const RealScalar& scale = RealScalar{1})
        {
            if (!std::isfinite(offset) || !std::isfinite(scale))
            {
                throw std::invalid_argument("sparse Cholesky shift must be finite");
            }
            _shiftOffset = offset;
            _shiftScale = scale;
            return derived();
        }

        const Matrix<Scalar, Dynamic, Dynamic>& matrixL() const
        {
            requireSuccessfulFactorization();
            if (!_denseFactorValid)
            {
                _denseFactor = Matrix<Scalar, Dynamic, Dynamic>::Zero(_rows, _rows);
                for (Index row = 0; row < _rows; ++row)
                {
                    for (const auto& [column, value] : _lowerRows[static_cast<std::size_t>(row)])
                    {
                        _denseFactor(row, column) = value;
                    }
                }
                _denseFactorValid = true;
            }
            return _denseFactor;
        }

        Matrix<Scalar, Dynamic, Dynamic> matrixU() const
        {
            return matrixL().transpose();
        }

        const PermutationType& permutationP() const noexcept
        {
            return _permutation;
        }

        const PermutationType& permutationPinv() const
        {
            if (!_inversePermutationValid)
            {
                _inversePermutation = _permutation.inverse();
                _inversePermutationValid = true;
            }
            return _inversePermutation;
        }

        Index factorNonZeros() const
        {
            requireSuccessfulFactorization();
            Index count = 0;
            for (const auto& row : _lowerRows)
            {
                count += static_cast<Index>(row.size());
            }
            return count;
        }

        template <typename Rhs> auto _solve(const Rhs& right) const
        {
            sparse_detail::requireCpu("sparse Cholesky solve");
            requireSuccessfulFactorization();
            using RhsScalar = typename detail::DenseTraits<Rhs>::ScalarType;
            static_assert(std::is_same_v<RhsScalar, Scalar>,
                          "sparse Cholesky right-hand side scalar must match the matrix");
            constexpr int RightCols = detail::DenseTraits<Rhs>::ColsAtCompileTime;
            Matrix<Scalar, Dynamic, RightCols> result(_rows, right.cols());
            Matrix<Scalar, Dynamic, 1> work(_rows);
            for (Index rhs_column = 0; rhs_column < right.cols(); ++rhs_column)
            {
                for (Index row = 0; row < _rows; ++row)
                {
                    work(row) = right(_permutation.indices()(row), rhs_column);
                    for (const auto& [column, value] : _lowerRows[static_cast<std::size_t>(row)])
                    {
                        if (column < row)
                        {
                            work(row) -= value * work(column);
                        }
                    }
                    if (!_isLdlt)
                    {
                        work(row) /= _lowerRows[static_cast<std::size_t>(row)].at(row);
                    }
                }
                if (_isLdlt)
                {
                    for (Index row = 0; row < _rows; ++row)
                    {
                        work(row) /= _diagonal(row);
                    }
                }
                for (Index row = _rows; row-- > 0;)
                {
                    for (const auto& [next, value] : _lowerColumns[static_cast<std::size_t>(row)])
                    {
                        if (next > row)
                        {
                            work(row) -= value * work(next);
                        }
                    }
                    if (!_isLdlt)
                    {
                        work(row) /= _lowerRows[static_cast<std::size_t>(row)].at(row);
                    }
                }
                for (Index row = 0; row < _rows; ++row)
                {
                    result(_permutation.indices()(row), rhs_column) = work(row);
                }
            }
            _info = result.allFinite() ? Success : NumericalIssue;
            return result;
        }

    protected:
        Derived& derived() noexcept
        {
            return *static_cast<Derived*>(this);
        }

        template <bool Ldlt> void analyzePatternImpl(const MatrixType& matrix)
        {
            sparse_detail::requireCpu("sparse Cholesky analyzePattern");
            reset();
            _isLdlt = Ldlt;
            if (matrix.rows() != matrix.cols())
            {
                fail(InvalidInput, "sparse Cholesky requires a square matrix");
                return;
            }
            _rows = matrix.rows();
            _pattern = sparse_detail::patternOf(matrix,
                                                [](Index row, Index col)
                                                {
                                                    if constexpr (Traits::Triangle == Lower)
                                                    {
                                                        return row >= col;
                                                    }
                                                    return row <= col;
                                                });
            OrderingType ordering;
            ordering(matrix, _permutation);
            _analysisIsOk = true;
            _info = Success;
            _lastError.clear();
        }

        template <bool Ldlt> void factorizeImpl(const MatrixType& matrix)
        {
            sparse_detail::requireCpu("sparse Cholesky factorize");
            SolverBase::_isInitialized = false;
            if (!_analysisIsOk || _isLdlt != Ldlt)
            {
                fail(InvalidInput, "analyzePattern must be called before sparse Cholesky factorize");
                return;
            }
            const auto pattern = sparse_detail::patternOf(matrix,
                                                          [](Index row, Index col)
                                                          {
                                                              if constexpr (Traits::Triangle == Lower)
                                                              {
                                                                  return row >= col;
                                                              }
                                                              return row <= col;
                                                          });
            if (matrix.rows() != _rows || matrix.cols() != _rows || pattern != _pattern)
            {
                fail(InvalidInput, "sparse Cholesky factorize requires the analyzed sparsity pattern");
                return;
            }

            std::vector<Index> inverse(static_cast<std::size_t>(_rows));
            for (Index index = 0; index < _rows; ++index)
            {
                inverse[static_cast<std::size_t>(_permutation.indices()(index))] = index;
            }
            std::vector<std::map<Index, Scalar>> input_columns(static_cast<std::size_t>(_rows));
            Scalar scale = Scalar{};
            for (Index outer = 0; outer < matrix.outerSize(); ++outer)
            {
                for (typename MatrixType::InnerIterator entry(matrix, outer); entry; ++entry)
                {
                    if ((Traits::Triangle == Lower && entry.row() < entry.col()) ||
                        (Traits::Triangle == Upper && entry.row() > entry.col()))
                    {
                        continue;
                    }
                    Index row = inverse[static_cast<std::size_t>(entry.row())];
                    Index column = inverse[static_cast<std::size_t>(entry.col())];
                    if (row < column)
                    {
                        std::swap(row, column);
                    }
                    const Scalar value = static_cast<Scalar>(entry.value());
                    input_columns[static_cast<std::size_t>(column)][row] = value;
                    scale = std::max(scale, std::abs(value));
                }
            }

            _lowerRows.assign(static_cast<std::size_t>(_rows), {});
            _lowerColumns.assign(static_cast<std::size_t>(_rows), {});
            _diagonal = Matrix<Scalar, Dynamic, 1>::Zero(_rows);
            _denseFactorValid = false;
            _inversePermutationValid = false;
            SolverBase::_isInitialized = true;
            const Scalar tolerance =
                std::numeric_limits<Scalar>::epsilon() * scale * static_cast<Scalar>(std::max<Index>(Index{1}, _rows));
            for (Index column = 0; column < _rows; ++column)
            {
                std::map<Index, Scalar> work = input_columns[static_cast<std::size_t>(column)];
                work[column] = _shiftOffset + _shiftScale * work[column];
                for (const auto& [previous, column_value] : _lowerRows[static_cast<std::size_t>(column)])
                {
                    if (previous >= column)
                    {
                        continue;
                    }
                    const Scalar weight = Ldlt ? _diagonal(previous) : Scalar{1};
                    for (const auto& [row, row_value] : _lowerColumns[static_cast<std::size_t>(previous)])
                    {
                        if (row >= column)
                        {
                            work[row] -= row_value * column_value * weight;
                        }
                    }
                }
                const Scalar diagonal = work[column];
                if constexpr (Ldlt)
                {
                    if (!std::isfinite(diagonal) || std::abs(diagonal) <= tolerance)
                    {
                        fail(NumericalIssue, "sparse LDLT encountered a zero numerical pivot");
                        return;
                    }
                    _diagonal(column) = diagonal;
                    _lowerRows[static_cast<std::size_t>(column)][column] = Scalar{1};
                    _lowerColumns[static_cast<std::size_t>(column)][column] = Scalar{1};
                    for (auto it = work.upper_bound(column); it != work.end(); ++it)
                    {
                        const Scalar value = it->second / diagonal;
                        if (value != Scalar{})
                        {
                            _lowerRows[static_cast<std::size_t>(it->first)][column] = value;
                            _lowerColumns[static_cast<std::size_t>(column)][it->first] = value;
                        }
                    }
                }
                else
                {
                    if (!std::isfinite(diagonal) || !(diagonal > tolerance))
                    {
                        fail(NumericalIssue, "sparse LLT matrix is not positive definite");
                        return;
                    }
                    const Scalar root = std::sqrt(diagonal);
                    _lowerRows[static_cast<std::size_t>(column)][column] = root;
                    _lowerColumns[static_cast<std::size_t>(column)][column] = root;
                    for (auto it = work.upper_bound(column); it != work.end(); ++it)
                    {
                        const Scalar value = it->second / root;
                        if (value != Scalar{})
                        {
                            _lowerRows[static_cast<std::size_t>(it->first)][column] = value;
                            _lowerColumns[static_cast<std::size_t>(column)][it->first] = value;
                        }
                    }
                }
            }
            _info = Success;
            _lastError.clear();
        }

        template <bool Ldlt> Derived& computeImpl(const MatrixType& matrix)
        {
            analyzePatternImpl<Ldlt>(matrix);
            if (_analysisIsOk)
            {
                factorizeImpl<Ldlt>(matrix);
            }
            return derived();
        }

        Scalar determinantImpl() const
        {
            requireSuccessfulFactorization();
            Scalar determinant = Scalar{1};
            for (Index index = 0; index < _rows; ++index)
            {
                if (_isLdlt)
                {
                    determinant *= _diagonal(index);
                }
                else
                {
                    const Scalar diagonal = _lowerRows[static_cast<std::size_t>(index)].at(index);
                    determinant *= diagonal * diagonal;
                }
            }
            return determinant;
        }

        const Matrix<Scalar, Dynamic, 1>& diagonal() const
        {
            requireSuccessfulFactorization();
            return _diagonal;
        }

    private:
        void requireSuccessfulFactorization() const
        {
            if (_info != Success || !SolverBase::_isInitialized)
            {
                throw std::logic_error("sparse Cholesky has no successful factorization");
            }
        }

        void fail(ComputationInfo info, std::string message)
        {
            _info = info;
            _lastError = std::move(message);
        }

        void reset()
        {
            SolverBase::_isInitialized = false;
            _analysisIsOk = false;
            _rows = 0;
            _pattern.clear();
            _lowerRows.clear();
            _lowerColumns.clear();
            _diagonal.resize(0, 1);
            _permutation = PermutationType{};
            _denseFactor.resize(0, 0);
            _denseFactorValid = false;
            _inversePermutationValid = false;
            _info = InvalidInput;
            _lastError.clear();
        }

        bool _analysisIsOk = false;
        bool _isLdlt = false;
        Index _rows = 0;
        RealScalar _shiftOffset = RealScalar{};
        RealScalar _shiftScale = RealScalar{1};
        std::vector<sparse_detail::Coordinate> _pattern;
        std::vector<std::map<Index, Scalar>> _lowerRows;
        std::vector<std::map<Index, Scalar>> _lowerColumns;
        Matrix<Scalar, Dynamic, 1> _diagonal;
        PermutationType _permutation;
        mutable PermutationType _inversePermutation;
        mutable bool _inversePermutationValid = false;
        mutable Matrix<Scalar, Dynamic, Dynamic> _denseFactor;
        mutable bool _denseFactorValid = false;
        mutable ComputationInfo _info = InvalidInput;
        std::string _lastError;
    };

    template <typename MatrixType_, int UpLo_, typename Ordering_>
    class SimplicialLLT : public SimplicialCholeskyBase<SimplicialLLT<MatrixType_, UpLo_, Ordering_>>
    {
        using Base = SimplicialCholeskyBase<SimplicialLLT<MatrixType_, UpLo_, Ordering_>>;

    public:
        using MatrixType = MatrixType_;
        using OrderingType = Ordering_;
        using Scalar = typename MatrixType::Scalar;
        using RealScalar = Scalar;
        using StorageIndex = typename MatrixType::StorageIndex;
        static constexpr int UpLo = UpLo_;
        static_assert(UpLo == Lower || UpLo == Upper, "SimplicialLLT requires Lower or Upper");

        SimplicialLLT() = default;
        explicit SimplicialLLT(const MatrixType& matrix)
        {
            compute(matrix);
        }
        SimplicialLLT& compute(const MatrixType& matrix)
        {
            return Base::template computeImpl<false>(matrix);
        }
        void analyzePattern(const MatrixType& matrix)
        {
            Base::template analyzePatternImpl<false>(matrix);
        }
        void factorize(const MatrixType& matrix)
        {
            Base::template factorizeImpl<false>(matrix);
        }
        Scalar determinant() const
        {
            return Base::determinantImpl();
        }
    };

    template <typename MatrixType_, int UpLo_, typename Ordering_>
    class SimplicialLDLT : public SimplicialCholeskyBase<SimplicialLDLT<MatrixType_, UpLo_, Ordering_>>
    {
        using Base = SimplicialCholeskyBase<SimplicialLDLT<MatrixType_, UpLo_, Ordering_>>;

    public:
        using MatrixType = MatrixType_;
        using OrderingType = Ordering_;
        using Scalar = typename MatrixType::Scalar;
        using RealScalar = Scalar;
        using StorageIndex = typename MatrixType::StorageIndex;
        static constexpr int UpLo = UpLo_;
        static_assert(UpLo == Lower || UpLo == Upper, "SimplicialLDLT requires Lower or Upper");

        SimplicialLDLT() = default;
        explicit SimplicialLDLT(const MatrixType& matrix)
        {
            compute(matrix);
        }
        SimplicialLDLT& compute(const MatrixType& matrix)
        {
            return Base::template computeImpl<true>(matrix);
        }
        void analyzePattern(const MatrixType& matrix)
        {
            Base::template analyzePatternImpl<true>(matrix);
        }
        void factorize(const MatrixType& matrix)
        {
            Base::template factorizeImpl<true>(matrix);
        }
        const Matrix<Scalar, Dynamic, 1>& vectorD() const
        {
            return Base::diagonal();
        }
        Scalar determinant() const
        {
            return Base::determinantImpl();
        }
    };
} // namespace plamatrix::v1
