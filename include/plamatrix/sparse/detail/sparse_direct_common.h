#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "plamatrix/sparse/sparse_matrix.h"
#include "plamatrix/sparse/sparse_solver_base.h"

namespace plamatrix::v1::sparse_detail
{
    using Coordinate = std::pair<Index, Index>;

    inline void requireCpu(const char* operation)
    {
        if (internal::currentExecutionSettings().policy == internal::ExecutionPolicy::GpuRequired)
        {
            throw internal::Error(internal::ErrorCode::UnsupportedOperation,
                                  std::string(operation) + " is not available on the selected GPU");
        }
    }

    template <typename MatrixType, typename Include>
    std::vector<Coordinate> patternOf(const MatrixType& matrix, Include include)
    {
        std::vector<Coordinate> pattern;
        pattern.reserve(static_cast<std::size_t>(matrix.nonZeros()));
        for (Index outer = 0; outer < matrix.outerSize(); ++outer)
        {
            for (typename MatrixType::InnerIterator entry(matrix, outer); entry; ++entry)
            {
                if (include(entry.row(), entry.col()))
                {
                    pattern.emplace_back(entry.row(), entry.col());
                }
            }
        }
        std::sort(pattern.begin(), pattern.end());
        return pattern;
    }

    template <typename MatrixType> std::vector<Coordinate> patternOf(const MatrixType& matrix)
    {
        return patternOf(matrix, [](Index, Index) { return true; });
    }

    template <typename MatrixType>
    Matrix<typename MatrixType::Scalar, Dynamic, Dynamic> denseCopy(const MatrixType& matrix)
    {
        using Scalar = typename MatrixType::Scalar;
        Matrix<Scalar, Dynamic, Dynamic> dense = Matrix<Scalar, Dynamic, Dynamic>::Zero(matrix.rows(), matrix.cols());
        for (Index outer = 0; outer < matrix.outerSize(); ++outer)
        {
            for (typename MatrixType::InnerIterator entry(matrix, outer); entry; ++entry)
            {
                dense(entry.row(), entry.col()) = entry.value();
            }
        }
        return dense;
    }

    template <int UpLo, typename MatrixType>
    Matrix<typename MatrixType::Scalar, Dynamic, Dynamic> selfAdjointDenseCopy(const MatrixType& matrix)
    {
        using Scalar = typename MatrixType::Scalar;
        Matrix<Scalar, Dynamic, Dynamic> dense = Matrix<Scalar, Dynamic, Dynamic>::Zero(matrix.rows(), matrix.cols());
        for (Index outer = 0; outer < matrix.outerSize(); ++outer)
        {
            for (typename MatrixType::InnerIterator entry(matrix, outer); entry; ++entry)
            {
                const Index row = entry.row();
                const Index col = entry.col();
                if ((UpLo == Lower && row < col) || (UpLo == Upper && row > col))
                {
                    continue;
                }
                dense(row, col) = entry.value();
                dense(col, row) = entry.value();
            }
        }
        return dense;
    }

    template <typename Scalar> Scalar numericalTolerance(const Matrix<Scalar, Dynamic, Dynamic>& matrix)
    {
        Scalar scale = Scalar{};
        for (Index col = 0; col < matrix.cols(); ++col)
        {
            for (Index row = 0; row < matrix.rows(); ++row)
            {
                scale = std::max(scale, std::abs(matrix(row, col)));
            }
        }
        return std::numeric_limits<Scalar>::epsilon() * scale *
               static_cast<Scalar>(std::max<Index>(Index{1}, std::max(matrix.rows(), matrix.cols())));
    }

    template <typename Rhs> auto dynamicCopy(const Rhs& source)
    {
        using Scalar = typename detail::DenseTraits<Rhs>::ScalarType;
        constexpr int Cols = detail::DenseTraits<Rhs>::ColsAtCompileTime;
        Matrix<Scalar, Dynamic, Cols> result(source.rows(), source.cols());
        for (Index col = 0; col < source.cols(); ++col)
        {
            for (Index row = 0; row < source.rows(); ++row)
            {
                result(row, col) = source(row, col);
            }
        }
        return result;
    }
} // namespace plamatrix::v1::sparse_detail
