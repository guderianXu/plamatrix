#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <type_traits>
#include <vector>

#include "plamatrix/dense/detail/static_or_dynamic_vector.h"
#include "plamatrix/dense/matrix.h"

namespace plamatrix::v1
{
    /// CPU pivoted LDLT for finite symmetric positive- or negative-semidefinite systems.
    template <typename Scalar, int Rows, int Cols, int StorageOptions, int MaxRows, int MaxCols, int UpLo>
    class LDLT<Matrix<Scalar, Rows, Cols, StorageOptions, MaxRows, MaxCols>, UpLo>
    {
        static_assert(std::is_floating_point_v<Scalar>, "LDLT requires a floating-point scalar");
        static_assert(UpLo == Lower || UpLo == Upper, "LDLT requires Lower or Upper");

    public:
        template <typename Node> auto solve(const DenseExpression<Node>& right) const
        {
            return solve(right.eval());
        }
        explicit LDLT(const Matrix<Scalar, Rows, Cols, StorageOptions, MaxRows, MaxCols>& input)
            : _lower(input.rows(), input.cols())
        {
            if (internal::currentExecutionSettings().policy == internal::ExecutionPolicy::GpuRequired)
            {
                throw internal::Error(internal::ErrorCode::UnsupportedOperation,
                                      "LDLT is not available on the selected GPU");
            }
            if (input.rows() != input.cols())
            {
                throw internal::Error(internal::ErrorCode::InvalidArgument, "LDLT requires a square matrix");
            }
            const Index n = input.rows();
            Matrix<Scalar, Rows, Cols, StorageOptions, MaxRows, MaxCols> trailing(n, n);
            _permutation.resize(static_cast<std::size_t>(n));
            std::iota(_permutation.begin(), _permutation.end(), Index{});
            _diagonal.assign(static_cast<std::size_t>(n), Scalar{});
            Scalar largest_diagonal{};
            for (Index col = 0; col < n; ++col)
            {
                for (Index row = 0; row < n; ++row)
                {
                    const Scalar value = UpLo == Lower ? (row >= col ? input(row, col) : input(col, row))
                                                       : (row <= col ? input(row, col) : input(col, row));
                    if (!std::isfinite(value))
                    {
                        throw internal::Error(internal::ErrorCode::InvalidArgument,
                                              "LDLT requires finite values in the selected triangle");
                    }
                    trailing(row, col) = value;
                }
                largest_diagonal = std::max(largest_diagonal, std::abs(trailing(col, col)));
            }
            _zeroTolerance = std::numeric_limits<Scalar>::epsilon() * static_cast<Scalar>(n) * largest_diagonal;
            for (Index column = 0; column < n; ++column)
            {
                Index pivot_row = column;
                Scalar largest = std::abs(trailing(column, column));
                for (Index row = column + 1; row < n; ++row)
                {
                    const Scalar magnitude = std::abs(trailing(row, row));
                    if (magnitude > largest)
                    {
                        largest = magnitude;
                        pivot_row = row;
                    }
                }
                if (pivot_row != column)
                {
                    for (Index index = 0; index < n; ++index)
                    {
                        std::swap(trailing(column, index), trailing(pivot_row, index));
                    }
                    for (Index index = 0; index < n; ++index)
                    {
                        std::swap(trailing(index, column), trailing(index, pivot_row));
                    }
                    for (Index previous = 0; previous < column; ++previous)
                    {
                        std::swap(_lower(column, previous), _lower(pivot_row, previous));
                    }
                    std::swap(_permutation[static_cast<std::size_t>(column)],
                              _permutation[static_cast<std::size_t>(pivot_row)]);
                }

                const Scalar pivot = trailing(column, column);
                if (!std::isfinite(pivot))
                {
                    throw internal::Error(internal::ErrorCode::InvalidArgument,
                                          "LDLT factorization produced non-finite values");
                }
                if (std::abs(pivot) <= _zeroTolerance)
                {
                    for (Index col = column; col < n; ++col)
                    {
                        for (Index row = col; row < n; ++row)
                        {
                            if (std::abs(trailing(row, col)) > _zeroTolerance)
                            {
                                throw internal::Error(internal::ErrorCode::InvalidArgument,
                                                      "LDLT requires a semidefinite matrix");
                            }
                        }
                    }
                    _lower(column, column) = Scalar{1};
                    continue;
                }
                _isPositive = _isPositive && pivot > Scalar{};
                _isNegative = _isNegative && pivot < Scalar{};
                if (!_isPositive && !_isNegative)
                {
                    throw internal::Error(internal::ErrorCode::InvalidArgument, "LDLT requires a semidefinite matrix");
                }
                _diagonal[static_cast<std::size_t>(column)] = pivot;
                _lower(column, column) = Scalar{1};
                for (Index row = column + 1; row < n; ++row)
                {
                    _lower(row, column) = trailing(row, column) / pivot;
                }
                for (Index col = column + 1; col < n; ++col)
                {
                    for (Index row = col; row < n; ++row)
                    {
                        const Scalar value = trailing(row, col) - _lower(row, column) * pivot * _lower(col, column);
                        trailing(row, col) = value;
                        trailing(col, row) = value;
                    }
                }
            }
            if (!_lower.allFinite())
            {
                throw internal::Error(internal::ErrorCode::InvalidArgument,
                                      "LDLT factorization produced non-finite values");
            }
        }

        template <int RightRows, int RightCols>
        Matrix<Scalar, Rows, RightCols> solve(const Matrix<Scalar, RightRows, RightCols>& right) const
        {
            if (internal::currentExecutionSettings().policy == internal::ExecutionPolicy::GpuRequired)
            {
                throw internal::Error(internal::ErrorCode::UnsupportedOperation,
                                      "LDLT solve is not available on the selected GPU");
            }
            if (right.rows() != _lower.rows() || !right.allFinite())
            {
                throw internal::Error(internal::ErrorCode::InvalidArgument,
                                      "LDLT right-hand side has invalid shape or values");
            }
            Matrix<Scalar, Rows, RightCols> result(right.rows(), right.cols());
            const Index n = _lower.rows();
            decomposition_detail::StaticOrDynamicVector<Scalar, Rows> unpermuted(static_cast<std::size_t>(n));
            for (Index col = 0; col < result.cols(); ++col)
            {
                Scalar rhs_scale{};
                for (Index row = 0; row < n; ++row)
                {
                    result(row, col) = right(_permutation[static_cast<std::size_t>(row)], col);
                    rhs_scale = std::max(rhs_scale, std::abs(result(row, col)));
                }
                for (Index row = 0; row < n; ++row)
                {
                    for (Index previous = 0; previous < row; ++previous)
                    {
                        result(row, col) -= _lower(row, previous) * result(previous, col);
                    }
                }
                for (Index row = 0; row < n; ++row)
                {
                    const Scalar diagonal = _diagonal[static_cast<std::size_t>(row)];
                    if (diagonal == Scalar{})
                    {
                        const Scalar tolerance =
                            std::numeric_limits<Scalar>::epsilon() * static_cast<Scalar>(n) * rhs_scale;
                        if (std::abs(result(row, col)) > tolerance)
                        {
                            throw internal::Error(internal::ErrorCode::InvalidArgument,
                                                  "LDLT right-hand side is outside the matrix range");
                        }
                        result(row, col) = Scalar{};
                    }
                    else
                    {
                        result(row, col) /= diagonal;
                    }
                }
                for (Index row = n; row-- > 0;)
                {
                    for (Index next = row + 1; next < n; ++next)
                    {
                        result(row, col) -= _lower(next, row) * result(next, col);
                    }
                }
                for (Index row = 0; row < n; ++row)
                {
                    unpermuted[static_cast<std::size_t>(_permutation[static_cast<std::size_t>(row)])] =
                        result(row, col);
                }
                for (Index row = 0; row < n; ++row)
                {
                    result(row, col) = unpermuted[static_cast<std::size_t>(row)];
                }
            }
            if (!result.allFinite())
            {
                throw internal::Error(internal::ErrorCode::InvalidArgument, "LDLT solve produced non-finite values");
            }
            result._info.reason = "CPU LDLT solve";
            return result;
        }

        bool isPositive() const noexcept
        {
            return _isPositive;
        }
        bool isNegative() const noexcept
        {
            return _isNegative;
        }

        Matrix<Scalar, Rows, 1, ColMajor, MaxRows, 1> vectorD() const
        {
            Matrix<Scalar, Rows, 1, ColMajor, MaxRows, 1> result(static_cast<Index>(_diagonal.size()), 1);
            for (Index row = 0; row < result.rows(); ++row)
            {
                result(row) = _diagonal[static_cast<std::size_t>(row)];
            }
            return result;
        }

        const Matrix<Scalar, Rows, Cols, StorageOptions, MaxRows, MaxCols>& matrixL() const noexcept
        {
            return _lower;
        }

    private:
        Matrix<Scalar, Rows, Cols, StorageOptions, MaxRows, MaxCols> _lower;
        decomposition_detail::StaticOrDynamicVector<Scalar, Rows> _diagonal;
        decomposition_detail::StaticOrDynamicVector<Index, Rows> _permutation;
        Scalar _zeroTolerance{};
        bool _isPositive = true;
        bool _isNegative = true;
    };
} // namespace plamatrix::v1

#include "plamatrix/dense/cholesky_lu.h"
#include "plamatrix/dense/qr.h"
#include "plamatrix/dense/svd.h"
#include "plamatrix/dense/eigen_solvers.h"
#include "plamatrix/dense/matrix_structural_decompositions.h"
#include "plamatrix/dense/schur.h"
#include "plamatrix/dense/complete_orthogonal_decomposition.h"
#include "plamatrix/dense/complex_eigen_solver.h"

namespace plamatrix::v1
{
    template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    LDLT<Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>>
    Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>::ldlt() const
    {
        return LDLT<Matrix>(*this);
    }

    template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    template <int UpLo>
    LLT<Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>, UpLo>
    Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>::llt() const
    {
        return LLT<Matrix, UpLo>(*this);
    }

    template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    template <typename PermutationIndex>
    FullPivLU<Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>, PermutationIndex>
    Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>::fullPivLu() const
    {
        return FullPivLU<Matrix, PermutationIndex>(*this);
    }

    template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    HouseholderQR<Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>>
    Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>::householderQr() const
    {
        return HouseholderQR<Matrix>(*this);
    }

    template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    template <typename PermutationIndex>
    ColPivHouseholderQR<Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>, PermutationIndex>
    Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>::colPivHouseholderQr() const
    {
        return ColPivHouseholderQR<Matrix, PermutationIndex>(*this);
    }

    template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    template <int SvdOptions>
    JacobiSVD<Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>, SvdOptions>
    Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>::jacobiSvd() const
    {
        return JacobiSVD<Matrix, SvdOptions>(*this);
    }

    template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    template <int SvdOptions>
    BDCSVD<Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>, SvdOptions>
    Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>::bdcSvd() const
    {
        return BDCSVD<Matrix, SvdOptions>(*this);
    }
} // namespace plamatrix::v1
