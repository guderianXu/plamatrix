#pragma once

#include <cmath>
#include <stdexcept>
#include <type_traits>

#include "plamatrix/dense/matrix.h"

namespace plamatrix::v1
{
    template <typename SourceDerived, typename DestinationDerived>
    auto umeyama(const MatrixBase<SourceDerived>& source,
                 const MatrixBase<DestinationDerived>& destination,
                 bool with_scaling = true)
    {
        using Scalar = typename SourceDerived::Scalar;
        static_assert(std::is_same_v<Scalar, typename DestinationDerived::Scalar>,
                      "umeyama() source and destination scalar types must match");
        static_assert(std::is_floating_point_v<Scalar>, "umeyama() requires floating-point coefficients");
        constexpr int SourceRows = detail::DenseTraits<SourceDerived>::RowsAtCompileTime;
        constexpr int ResultRows = SourceRows == Dynamic ? Dynamic : SourceRows + 1;
        using Result = Matrix<Scalar, ResultRows, ResultRows>;

        if (source.rows() != destination.rows() || source.cols() != destination.cols() || source.cols() == 0)
        {
            throw std::invalid_argument("umeyama() requires equally shaped nonempty point sets");
        }
        const Index dimension = source.rows();
        const Index count = source.cols();
        if (dimension <= 0)
        {
            throw std::invalid_argument("umeyama() requires a positive point dimension");
        }

        Matrix<Scalar, Dynamic, 1> source_mean(dimension);
        Matrix<Scalar, Dynamic, 1> destination_mean(dimension);
        source_mean.setZero();
        destination_mean.setZero();
        for (Index column = 0; column < count; ++column)
        {
            for (Index row = 0; row < dimension; ++row)
            {
                const Scalar source_value = source.derived()(row, column);
                const Scalar destination_value = destination.derived()(row, column);
                if (!std::isfinite(source_value) || !std::isfinite(destination_value))
                {
                    throw std::invalid_argument("umeyama() requires finite point coordinates");
                }
                source_mean(row) += source_value;
                destination_mean(row) += destination_value;
            }
        }
        const Scalar inverse_count = Scalar{1} / static_cast<Scalar>(count);
        for (Index row = 0; row < dimension; ++row)
        {
            source_mean(row) *= inverse_count;
            destination_mean(row) *= inverse_count;
        }

        Matrix<Scalar, Dynamic, Dynamic> covariance(dimension, dimension);
        covariance.setZero();
        Scalar source_variance{};
        for (Index column = 0; column < count; ++column)
        {
            for (Index row = 0; row < dimension; ++row)
            {
                const Scalar source_centered = source.derived()(row, column) - source_mean(row);
                source_variance += source_centered * source_centered;
                for (Index other = 0; other < dimension; ++other)
                {
                    covariance(row, other) += (destination.derived()(row, column) - destination_mean(row)) *
                                              (source.derived()(other, column) - source_mean(other));
                }
            }
        }
        source_variance *= inverse_count;
        for (Index row = 0; row < dimension; ++row)
            for (Index column = 0; column < dimension; ++column)
                covariance(row, column) *= inverse_count;

        JacobiSVD<Matrix<Scalar, Dynamic, Dynamic>> decomposition(covariance, ComputeFullU | ComputeFullV);
        if (decomposition.info() != Success)
        {
            throw std::runtime_error("umeyama() covariance SVD failed");
        }
        const auto& u = decomposition.matrixU();
        const auto& v = decomposition.matrixV();
        Matrix<Scalar, Dynamic, 1> signs(dimension);
        signs.setOnes();

        Matrix<Scalar, Dynamic, Dynamic> uv_transpose(dimension, dimension);
        for (Index row = 0; row < dimension; ++row)
            for (Index column = 0; column < dimension; ++column)
            {
                Scalar value{};
                for (Index inner = 0; inner < dimension; ++inner)
                    value += u(row, inner) * v(column, inner);
                uv_transpose(row, column) = value;
            }
        if (uv_transpose.fullPivLu().determinant() < Scalar{})
            signs(dimension - 1) = Scalar{-1};

        Matrix<Scalar, Dynamic, Dynamic> rotation(dimension, dimension);
        for (Index row = 0; row < dimension; ++row)
            for (Index column = 0; column < dimension; ++column)
            {
                Scalar value{};
                for (Index inner = 0; inner < dimension; ++inner)
                    value += u(row, inner) * signs(inner) * v(column, inner);
                rotation(row, column) = value;
            }

        Scalar scale = Scalar{1};
        if (with_scaling)
        {
            if (source_variance <= std::numeric_limits<Scalar>::epsilon())
                throw std::runtime_error("umeyama() cannot estimate scale from a degenerate source point set");
            scale = Scalar{};
            for (Index index = 0; index < dimension; ++index)
                scale += decomposition.singularValues()(index) * signs(index);
            scale /= source_variance;
        }

        Result result(dimension + 1, dimension + 1);
        result.setIdentity();
        for (Index row = 0; row < dimension; ++row)
        {
            Scalar rotated_mean{};
            for (Index column = 0; column < dimension; ++column)
            {
                result(row, column) = scale * rotation(row, column);
                rotated_mean += rotation(row, column) * source_mean(column);
            }
            result(row, dimension) = destination_mean(row) - scale * rotated_mean;
        }
        return result;
    }
} // namespace plamatrix::v1
