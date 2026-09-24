#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

#include "plamatrix/dense/matrix.h"
#include "plamatrix/dense/detail/static_or_dynamic_vector.h"

namespace plamatrix::v1
{

    /// Reusable CPU LU factorization with partial row pivoting for a square matrix.
    template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols, typename PermutationIndex>
    class PartialPivLU<Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>, PermutationIndex>
    {
        static_assert(std::is_floating_point_v<Scalar> || NumTraits<Scalar>::IsComplex,
                      "PartialPivLU requires a real or complex floating-point scalar");
        static_assert(std::is_integral_v<PermutationIndex>, "PartialPivLU requires an integral permutation index");

    public:
        using RealScalar = typename NumTraits<Scalar>::Real;
        template <typename Node> auto solve(const DenseExpression<Node>& right) const
        {
            return solve(right.eval());
        }
        explicit PartialPivLU(const Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>& input) : _factors(input)
        {
            if (internal::currentExecutionSettings().policy == internal::ExecutionPolicy::GpuRequired)
            {
                throw internal::Error(internal::ErrorCode::UnsupportedOperation,
                                      "Partial-pivot LU is not available on the selected GPU");
            }
            if (input.rows() != input.cols() || !input.allFinite())
            {
                throw internal::Error(internal::ErrorCode::InvalidArgument,
                                      "Partial-pivot LU requires a finite square matrix");
            }

            const Index n = _factors.rows();
            const Scalar* input_values = input.data();
            const auto offset = [n](Index row, Index column)
            {
                if constexpr ((Options & RowMajor) == RowMajor)
                    return column + row * n;
                else
                    return row + column * n;
            };
            for (Index column = 0; column < n; ++column)
            {
                RealScalar column_sum{};
                for (Index row = 0; row < n; ++row)
                {
                    column_sum += std::abs(input_values[offset(row, column)]);
                }
                _matrixOneNorm = std::max(_matrixOneNorm, column_sum);
            }
            if constexpr (std::numeric_limits<PermutationIndex>::digits < std::numeric_limits<Index>::digits)
            {
                if (n > static_cast<Index>(std::numeric_limits<PermutationIndex>::max()))
                {
                    throw internal::Error(internal::ErrorCode::InvalidArgument,
                                          "LU permutation index cannot represent matrix size");
                }
            }
            _pivots.resize(static_cast<std::size_t>(n));
            Scalar* factors = _factors.data();
            for (Index column = 0; column < n; ++column)
            {
                Index pivot = column;
                RealScalar largest = std::abs(factors[offset(column, column)]);
                for (Index row = column + 1; row < n; ++row)
                {
                    const RealScalar magnitude = std::abs(factors[offset(row, column)]);
                    if (magnitude > largest)
                    {
                        largest = magnitude;
                        pivot = row;
                    }
                }
                if (largest == RealScalar{})
                {
                    throw internal::Error(internal::ErrorCode::InvalidArgument, "Partial-pivot LU matrix is singular");
                }
                _pivots[static_cast<std::size_t>(column)] = static_cast<PermutationIndex>(pivot);
                if (pivot != column)
                {
                    for (Index other_column = 0; other_column < n; ++other_column)
                    {
                        std::swap(factors[offset(column, other_column)], factors[offset(pivot, other_column)]);
                    }
                }
                const Scalar diagonal = factors[offset(column, column)];
                if constexpr ((Options & RowMajor) == RowMajor)
                {
                    for (Index row = column + 1; row < n; ++row)
                    {
                        const Scalar factor = factors[offset(row, column)] / diagonal;
                        factors[offset(row, column)] = factor;
#pragma omp simd
                        for (Index other_column = column + 1; other_column < n; ++other_column)
                            factors[offset(row, other_column)] -= factor * factors[offset(column, other_column)];
                    }
                }
                else
                {
                    for (Index row = column + 1; row < n; ++row)
                        factors[offset(row, column)] /= diagonal;
                    const Scalar* lower_column = factors + column * n;
                    Index other_column = column + 1;
                    for (; other_column + 3 < n; other_column += 4)
                    {
                        Scalar* first = factors + other_column * n;
                        Scalar* second = first + n;
                        Scalar* third = second + n;
                        Scalar* fourth = third + n;
                        const Scalar first_pivot = first[column];
                        const Scalar second_pivot = second[column];
                        const Scalar third_pivot = third[column];
                        const Scalar fourth_pivot = fourth[column];
#pragma omp simd
                        for (Index row = column + 1; row < n; ++row)
                        {
                            const Scalar factor = lower_column[row];
                            first[row] -= factor * first_pivot;
                            second[row] -= factor * second_pivot;
                            third[row] -= factor * third_pivot;
                            fourth[row] -= factor * fourth_pivot;
                        }
                    }
                    for (; other_column < n; ++other_column)
                    {
                        Scalar* trailing_column = factors + other_column * n;
                        const Scalar pivot_value = trailing_column[column];
#pragma omp simd
                        for (Index row = column + 1; row < n; ++row)
                            trailing_column[row] -= lower_column[row] * pivot_value;
                    }
                }
            }
            if (!_factors.allFinite())
            {
                throw internal::Error(internal::ErrorCode::InvalidArgument,
                                      "Partial-pivot LU factorization produced non-finite values");
            }
        }

        /// Solve A * X = right for a vector or multiple columns without refactorizing A.
        template <int RightRows, int RightCols>
        Matrix<Scalar, Rows, RightCols> solve(const Matrix<Scalar, RightRows, RightCols>& right) const
        {
            if (internal::currentExecutionSettings().policy == internal::ExecutionPolicy::GpuRequired)
            {
                throw internal::Error(internal::ErrorCode::UnsupportedOperation,
                                      "Partial-pivot LU solve is not available on the selected GPU");
            }
            if (right.rows() != _factors.rows())
            {
                throw internal::Error(internal::ErrorCode::InvalidArgument,
                                      "LU solve requires a right-hand side with matching rows");
            }
            const std::size_t downloaded_bytes = right.pendingDownloadBytes();
            if (!right.allFinite())
            {
                throw internal::Error(internal::ErrorCode::InvalidArgument,
                                      "LU solve requires a finite right-hand side");
            }
            Matrix<Scalar, Rows, RightCols> result(right);
            const Index n = _factors.rows();
            const Scalar* factors = _factors.data();
            Scalar* values = result.data();
            const auto factor_offset = [n](Index row, Index column)
            {
                if constexpr ((Options & RowMajor) == RowMajor)
                    return column + row * n;
                else
                    return row + column * n;
            };
            for (Index column = 0; column < result.cols(); ++column)
            {
                Scalar* result_column = values + column * n;
                for (Index row = 0; row < n; ++row)
                {
                    const Index pivot = _pivots[static_cast<std::size_t>(row)];
                    if (pivot != row)
                    {
                        std::swap(result_column[row], result_column[pivot]);
                    }
                }
                for (Index previous = 0; previous < n; ++previous)
                {
                    const Scalar solved = result_column[previous];
#pragma omp simd
                    for (Index row = previous + 1; row < n; ++row)
                        result_column[row] -= factors[factor_offset(row, previous)] * solved;
                }
                for (Index row = n; row-- > 0;)
                {
                    result_column[row] /= factors[factor_offset(row, row)];
                    const Scalar solved = result_column[row];
#pragma omp simd
                    for (Index previous = 0; previous < row; ++previous)
                        result_column[previous] -= factors[factor_offset(previous, row)] * solved;
                }
            }
            if (!result.allFinite())
            {
                throw internal::Error(internal::ErrorCode::InvalidArgument,
                                      "Partial-pivot LU solve produced non-finite values");
            }
            result._info.backend = internal::Backend::Cpu;
            result._info.bytesDownloaded = downloaded_bytes;
            result._info.reason = "CPU partial-pivot LU solve";
            return result;
        }

        RealScalar rcond() const
        {
            if (_matrixOneNorm == RealScalar{})
            {
                return RealScalar{};
            }
            using IdentityType = Matrix<Scalar, Rows, Rows, Options, MaxRows, MaxRows>;
            const auto inverse = solve(IdentityType::Identity(_factors.rows(), _factors.rows()));
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

    private:
        Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols> _factors;
        decomposition_detail::StaticOrDynamicVector<PermutationIndex, Rows> _pivots;
        RealScalar _matrixOneNorm{};
    };

    template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    template <typename PermutationIndex>
    PartialPivLU<Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>, PermutationIndex>
    Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>::partialPivLu() const
    {
        return PartialPivLU<Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>, PermutationIndex>(*this);
    }

    template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    template <typename PermutationIndex>
    PartialPivLU<Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>, PermutationIndex>
    Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>::lu() const
    {
        return partialPivLu<PermutationIndex>();
    }

} // namespace plamatrix::v1
