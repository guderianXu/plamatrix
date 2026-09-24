#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "plamatrix/dense/permutation_matrix.h"
#include "plamatrix/sparse/ordering_methods.h"
#include "plamatrix/sparse/sparse_matrix.h"

namespace plamatrix::v1
{
    namespace sparse_preconditioner_detail
    {
        template <typename Scalar, typename Rhs, typename ApplyColumn>
        auto applyColumns(const Rhs& right, Index rows, ApplyColumn apply_column)
        {
            constexpr int RightCols = detail::DenseTraits<Rhs>::ColsAtCompileTime;
            Matrix<Scalar, Dynamic, RightCols> result(rows, right.cols());
            Matrix<Scalar, Dynamic, 1> input(rows);
            Matrix<Scalar, Dynamic, 1> output(rows);
            for (Index column = 0; column < right.cols(); ++column)
            {
                for (Index row = 0; row < rows; ++row)
                {
                    input(row) = right(row, column);
                }
                apply_column(input, output);
                for (Index row = 0; row < rows; ++row)
                {
                    result(row, column) = output(row);
                }
            }
            return result;
        }
    } // namespace sparse_preconditioner_detail

    class IdentityPreconditioner
    {
    public:
        template <typename MatrixType> IdentityPreconditioner& analyzePattern(const MatrixType& matrix)
        {
            _rows = matrix.rows();
            _info = matrix.rows() == matrix.cols() ? Success : InvalidInput;
            return *this;
        }

        template <typename MatrixType> IdentityPreconditioner& factorize(const MatrixType& matrix)
        {
            return analyzePattern(matrix);
        }

        template <typename MatrixType> IdentityPreconditioner& compute(const MatrixType& matrix)
        {
            return factorize(matrix);
        }

        template <typename Rhs> auto solve(const MatrixBase<Rhs>& right) const
        {
            if (_info != Success || right.rows() != _rows)
            {
                throw std::logic_error("IdentityPreconditioner is not initialized for this right-hand side");
            }
            return right.derived().eval();
        }

        ComputationInfo info() const noexcept
        {
            return _info;
        }

    private:
        Index _rows = 0;
        ComputationInfo _info = InvalidInput;
    };

    template <typename Scalar> class DiagonalPreconditioner
    {
        static_assert(std::is_floating_point_v<Scalar>, "DiagonalPreconditioner requires floating-point coefficients");

    public:
        DiagonalPreconditioner() = default;

        template <typename MatrixType> explicit DiagonalPreconditioner(const MatrixType& matrix)
        {
            compute(matrix);
        }

        template <typename MatrixType> DiagonalPreconditioner& analyzePattern(const MatrixType& matrix)
        {
            _inverse.resize(matrix.rows(), 1);
            _info = matrix.rows() == matrix.cols() ? Success : InvalidInput;
            return *this;
        }

        template <typename MatrixType> DiagonalPreconditioner& factorize(const MatrixType& matrix)
        {
            if (matrix.rows() != matrix.cols())
            {
                _info = InvalidInput;
                return *this;
            }
            _inverse.resize(matrix.rows(), 1);
            for (Index index = 0; index < matrix.rows(); ++index)
            {
                const Scalar diagonal = static_cast<Scalar>(matrix.coeff(index, index));
                if (!std::isfinite(diagonal) || diagonal == Scalar{})
                {
                    _info = NumericalIssue;
                    return *this;
                }
                _inverse(index) = Scalar{1} / diagonal;
            }
            _info = Success;
            return *this;
        }

        template <typename MatrixType> DiagonalPreconditioner& compute(const MatrixType& matrix)
        {
            analyzePattern(matrix);
            return factorize(matrix);
        }

        template <typename Rhs> auto solve(const MatrixBase<Rhs>& right) const
        {
            if (_info != Success || right.rows() != _inverse.rows())
            {
                throw std::logic_error("DiagonalPreconditioner is not initialized for this right-hand side");
            }
            return sparse_preconditioner_detail::applyColumns<Scalar>(right.derived(),
                                                                      _inverse.rows(),
                                                                      [&](const auto& input, auto& output)
                                                                      {
                                                                          for (Index row = 0; row < _inverse.rows();
                                                                               ++row)
                                                                          {
                                                                              output(row) = _inverse(row) * input(row);
                                                                          }
                                                                      });
        }

        ComputationInfo info() const noexcept
        {
            return _info;
        }

    private:
        Matrix<Scalar, Dynamic, 1> _inverse;
        ComputationInfo _info = InvalidInput;
    };

    template <typename Scalar, int UpLo, typename OrderingType> class IncompleteCholesky
    {
        static_assert(std::is_floating_point_v<Scalar>, "IncompleteCholesky requires floating-point coefficients");
        static_assert(UpLo == Lower || UpLo == Upper, "IncompleteCholesky UpLo must be Lower or Upper");

    public:
        using StorageIndex = DefaultPermutationIndex;
        using PermutationType = PermutationMatrix<Dynamic, Dynamic, StorageIndex>;

        IncompleteCholesky() = default;

        template <typename MatrixType> explicit IncompleteCholesky(const MatrixType& matrix)
        {
            compute(matrix);
        }

        IncompleteCholesky& setInitialShift(const Scalar& shift)
        {
            if (!std::isfinite(shift) || shift < Scalar{})
            {
                throw std::invalid_argument("IncompleteCholesky shift must be finite and non-negative");
            }
            _initialShift = shift;
            return *this;
        }

        template <typename MatrixType> IncompleteCholesky& analyzePattern(const MatrixType& matrix)
        {
            reset();
            if (matrix.rows() != matrix.cols())
            {
                _info = InvalidInput;
                return *this;
            }
            OrderingType ordering;
            ordering(matrix, _permutation);
            _rows = matrix.rows();
            _info = Success;
            return *this;
        }

        template <typename MatrixType> IncompleteCholesky& factorize(const MatrixType& matrix)
        {
            if (_info != Success || matrix.rows() != _rows || matrix.cols() != _rows)
            {
                _info = InvalidInput;
                return *this;
            }
            std::vector<Index> inverse(static_cast<std::size_t>(_rows));
            for (Index index = 0; index < _rows; ++index)
            {
                inverse[static_cast<std::size_t>(_permutation.indices()(index))] = index;
            }
            std::vector<std::map<Index, Scalar>> values(static_cast<std::size_t>(_rows));
            for (Index outer = 0; outer < matrix.outerSize(); ++outer)
            {
                for (typename MatrixType::InnerIterator entry(matrix, outer); entry; ++entry)
                {
                    if ((UpLo == Lower && entry.row() < entry.col()) || (UpLo == Upper && entry.row() > entry.col()))
                    {
                        continue;
                    }
                    Index row = inverse[static_cast<std::size_t>(entry.row())];
                    Index column = inverse[static_cast<std::size_t>(entry.col())];
                    if (row < column)
                    {
                        std::swap(row, column);
                    }
                    values[static_cast<std::size_t>(row)][column] = static_cast<Scalar>(entry.value());
                }
            }

            _lower.assign(static_cast<std::size_t>(_rows), {});
            for (Index row = 0; row < _rows; ++row)
            {
                auto& lower_row = _lower[static_cast<std::size_t>(row)];
                const auto& source_row = values[static_cast<std::size_t>(row)];
                for (const auto& [column, source] : source_row)
                {
                    if (column >= row)
                    {
                        continue;
                    }
                    Scalar value = source;
                    for (const auto& [previous, left] : lower_row)
                    {
                        if (previous >= column)
                        {
                            break;
                        }
                        const auto right = _lower[static_cast<std::size_t>(column)].find(previous);
                        if (right != _lower[static_cast<std::size_t>(column)].end())
                        {
                            value -= left * right->second;
                        }
                    }
                    const auto diagonal = _lower[static_cast<std::size_t>(column)].find(column);
                    if (diagonal == _lower[static_cast<std::size_t>(column)].end() || diagonal->second == Scalar{})
                    {
                        _info = NumericalIssue;
                        return *this;
                    }
                    lower_row[column] = value / diagonal->second;
                }
                Scalar diagonal = _initialShift;
                const auto source_diagonal = source_row.find(row);
                if (source_diagonal != source_row.end())
                {
                    diagonal += source_diagonal->second;
                }
                for (const auto& [column, value] : lower_row)
                {
                    static_cast<void>(column);
                    diagonal -= value * value;
                }
                if (!std::isfinite(diagonal) || diagonal <= Scalar{})
                {
                    _info = NumericalIssue;
                    return *this;
                }
                lower_row[row] = std::sqrt(diagonal);
            }
            _info = Success;
            return *this;
        }

        template <typename MatrixType> IncompleteCholesky& compute(const MatrixType& matrix)
        {
            analyzePattern(matrix);
            if (_info == Success)
            {
                factorize(matrix);
            }
            return *this;
        }

        template <typename Rhs> auto solve(const MatrixBase<Rhs>& right) const
        {
            if (_info != Success || right.rows() != _rows)
            {
                throw std::logic_error("IncompleteCholesky is not initialized for this right-hand side");
            }
            return sparse_preconditioner_detail::applyColumns<Scalar>(
                right.derived(),
                _rows,
                [&](const auto& input, auto& output)
                {
                    Matrix<Scalar, Dynamic, 1> work(_rows);
                    for (Index row = 0; row < _rows; ++row)
                    {
                        work(row) = input(_permutation.indices()(row));
                        const auto& lower_row = _lower[static_cast<std::size_t>(row)];
                        for (const auto& [column, value] : lower_row)
                        {
                            if (column < row)
                            {
                                work(row) -= value * work(column);
                            }
                        }
                        work(row) /= lower_row.at(row);
                    }
                    for (Index row = _rows; row-- > 0;)
                    {
                        work(row) /= _lower[static_cast<std::size_t>(row)].at(row);
                        for (Index target = 0; target < row; ++target)
                        {
                            const auto found = _lower[static_cast<std::size_t>(row)].find(target);
                            if (found != _lower[static_cast<std::size_t>(row)].end())
                            {
                                work(target) -= found->second * work(row);
                            }
                        }
                    }
                    for (Index row = 0; row < _rows; ++row)
                    {
                        output(_permutation.indices()(row)) = work(row);
                    }
                });
        }

        ComputationInfo info() const noexcept
        {
            return _info;
        }

        const PermutationType& permutationP() const noexcept
        {
            return _permutation;
        }

    private:
        void reset()
        {
            _rows = 0;
            _lower.clear();
            _permutation = PermutationType{};
            _info = InvalidInput;
        }

        Index _rows = 0;
        Scalar _initialShift = Scalar{};
        std::vector<std::map<Index, Scalar>> _lower;
        PermutationType _permutation;
        ComputationInfo _info = InvalidInput;
    };

    template <typename Scalar, typename StorageIndex> class IncompleteLUT
    {
        static_assert(std::is_floating_point_v<Scalar>, "IncompleteLUT requires floating-point coefficients");
        static_assert(std::is_integral_v<StorageIndex>, "IncompleteLUT requires an integral storage index");

    public:
        IncompleteLUT() = default;

        template <typename MatrixType> explicit IncompleteLUT(const MatrixType& matrix)
        {
            compute(matrix);
        }

        IncompleteLUT& setDroptol(const Scalar& tolerance)
        {
            if (!std::isfinite(tolerance) || tolerance < Scalar{})
            {
                throw std::invalid_argument("IncompleteLUT drop tolerance must be finite and non-negative");
            }
            _dropTolerance = tolerance;
            return *this;
        }

        IncompleteLUT& setFillfactor(int factor)
        {
            if (factor < 1)
            {
                throw std::invalid_argument("IncompleteLUT fill factor must be positive");
            }
            _fillFactor = factor;
            return *this;
        }

        template <typename MatrixType> IncompleteLUT& analyzePattern(const MatrixType& matrix)
        {
            reset();
            _rows = matrix.rows();
            _info = matrix.rows() == matrix.cols() ? Success : InvalidInput;
            return *this;
        }

        template <typename MatrixType> IncompleteLUT& factorize(const MatrixType& matrix)
        {
            if (_info != Success || matrix.rows() != _rows || matrix.cols() != _rows)
            {
                _info = InvalidInput;
                return *this;
            }
            std::vector<std::map<Index, Scalar>> source(static_cast<std::size_t>(_rows));
            for (Index outer = 0; outer < matrix.outerSize(); ++outer)
            {
                for (typename MatrixType::InnerIterator entry(matrix, outer); entry; ++entry)
                {
                    source[static_cast<std::size_t>(entry.row())][entry.col()] = static_cast<Scalar>(entry.value());
                }
            }
            _lower.assign(static_cast<std::size_t>(_rows), {});
            _upper.assign(static_cast<std::size_t>(_rows), {});
            for (Index row = 0; row < _rows; ++row)
            {
                auto work = source[static_cast<std::size_t>(row)];
                Scalar norm = Scalar{};
                for (const auto& [column, value] : work)
                {
                    static_cast<void>(column);
                    norm = std::max(norm, std::abs(value));
                }
                auto current = work.begin();
                while (current != work.end() && current->first < row)
                {
                    const Index pivot = current->first;
                    const auto diagonal = _upper[static_cast<std::size_t>(pivot)].find(pivot);
                    if (diagonal == _upper[static_cast<std::size_t>(pivot)].end() || diagonal->second == Scalar{})
                    {
                        _info = NumericalIssue;
                        return *this;
                    }
                    const Scalar factor = current->second / diagonal->second;
                    current = work.erase(current);
                    if (std::abs(factor) > _dropTolerance * std::max(norm, Scalar{1}))
                    {
                        _lower[static_cast<std::size_t>(row)][pivot] = factor;
                        for (const auto& [column, value] : _upper[static_cast<std::size_t>(pivot)])
                        {
                            if (column > pivot)
                            {
                                work[column] -= factor * value;
                            }
                        }
                    }
                    current = work.upper_bound(pivot);
                }
                const auto diagonal = work.find(row);
                if (diagonal == work.end() || !std::isfinite(diagonal->second) || diagonal->second == Scalar{})
                {
                    _info = NumericalIssue;
                    return *this;
                }
                for (auto it = work.lower_bound(row); it != work.end(); ++it)
                {
                    if (it->first == row || std::abs(it->second) > _dropTolerance * std::max(norm, Scalar{1}))
                    {
                        _upper[static_cast<std::size_t>(row)][it->first] = it->second;
                    }
                }
                const std::size_t original_count = source[static_cast<std::size_t>(row)].size();
                const std::size_t limit =
                    std::max<std::size_t>(1U, original_count * static_cast<std::size_t>(_fillFactor));
                trimLargest(_lower[static_cast<std::size_t>(row)], limit, -1);
                trimLargest(_upper[static_cast<std::size_t>(row)], limit, row);
            }
            _info = Success;
            return *this;
        }

        template <typename MatrixType> IncompleteLUT& compute(const MatrixType& matrix)
        {
            analyzePattern(matrix);
            if (_info == Success)
            {
                factorize(matrix);
            }
            return *this;
        }

        template <typename Rhs> auto solve(const MatrixBase<Rhs>& right) const
        {
            if (_info != Success || right.rows() != _rows)
            {
                throw std::logic_error("IncompleteLUT is not initialized for this right-hand side");
            }
            return sparse_preconditioner_detail::applyColumns<Scalar>(
                right.derived(),
                _rows,
                [&](const auto& input, auto& output)
                {
                    for (Index row = 0; row < _rows; ++row)
                    {
                        output(row) = input(row);
                        for (const auto& [column, value] : _lower[static_cast<std::size_t>(row)])
                        {
                            output(row) -= value * output(column);
                        }
                    }
                    for (Index row = _rows; row-- > 0;)
                    {
                        const auto& upper_row = _upper[static_cast<std::size_t>(row)];
                        for (const auto& [column, value] : upper_row)
                        {
                            if (column > row)
                            {
                                output(row) -= value * output(column);
                            }
                        }
                        output(row) /= upper_row.at(row);
                    }
                });
        }

        ComputationInfo info() const noexcept
        {
            return _info;
        }

    private:
        static void trimLargest(std::map<Index, Scalar>& values, std::size_t limit, Index protected_index)
        {
            if (values.size() <= limit)
            {
                return;
            }
            std::vector<std::pair<Scalar, Index>> ranked;
            ranked.reserve(values.size());
            for (const auto& [index, value] : values)
            {
                if (index != protected_index)
                {
                    ranked.emplace_back(std::abs(value), index);
                }
            }
            std::sort(ranked.begin(),
                      ranked.end(),
                      [](const auto& left, const auto& right) {
                          return left.first > right.first || (left.first == right.first && left.second < right.second);
                      });
            const std::size_t allowed = protected_index >= 0 ? limit - 1U : limit;
            for (std::size_t position = allowed; position < ranked.size(); ++position)
            {
                values.erase(ranked[position].second);
            }
        }

        void reset()
        {
            _rows = 0;
            _lower.clear();
            _upper.clear();
            _info = InvalidInput;
        }

        Index _rows = 0;
        Scalar _dropTolerance = Scalar{1e-4};
        int _fillFactor = 10;
        std::vector<std::map<Index, Scalar>> _lower;
        std::vector<std::map<Index, Scalar>> _upper;
        ComputationInfo _info = InvalidInput;
    };
} // namespace plamatrix::v1
