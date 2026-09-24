#pragma once

#include <cmath>
#include <stdexcept>

#include "plamatrix/dense/matrix.h"

namespace plamatrix::v1
{
    namespace geometry_detail
    {
        inline void validateEulerAxes(Index a0, Index a1, Index a2)
        {
            if (a0 < 0 || a0 > 2 || a1 < 0 || a1 > 2 || a2 < 0 || a2 > 2 || a0 == a1 || a1 == a2)
            {
                throw std::invalid_argument("Euler axes must be in [0,2] and adjacent axes must differ");
            }
        }
    } // namespace geometry_detail

    template <typename Derived>
    Matrix<typename MatrixBase<Derived>::Scalar, 3, 1>
    MatrixBase<Derived>::canonicalEulerAngles(Index a0, Index a1, Index a2) const
    {
        geometry_detail::validateEulerAxes(a0, a1, a2);
        if (this->rows() != 3 || this->cols() != 3)
        {
            throw std::invalid_argument("canonicalEulerAngles() requires a 3x3 matrix");
        }
        Matrix<Scalar, 3, 1> result;
        const auto& matrix = this->derived();
        const Index odd = ((a0 + 1) % 3 == a1) ? 0 : 1;
        const Index i = a0;
        const Index j = (a0 + 1 + odd) % 3;
        const Index k = (a0 + 2 - odd) % 3;

        if (a0 == a2)
        {
            const Scalar s2 = std::hypot(matrix(j, i), matrix(k, i));
            if (odd)
            {
                result[0] = std::atan2(matrix(j, i), matrix(k, i));
                result[1] = std::atan2(s2, matrix(i, i));
            }
            else
            {
                result[0] = std::atan2(-matrix(j, i), -matrix(k, i));
                result[1] = -std::atan2(s2, matrix(i, i));
            }
            const Scalar s1 = std::sin(result[0]);
            const Scalar c1 = std::cos(result[0]);
            result[2] = std::atan2(c1 * matrix(j, k) - s1 * matrix(k, k), c1 * matrix(j, j) - s1 * matrix(k, j));
        }
        else
        {
            result[0] = std::atan2(matrix(j, k), matrix(k, k));
            const Scalar c2 = std::hypot(matrix(i, i), matrix(i, j));
            result[1] = std::atan2(-matrix(i, k), c2);
            const Scalar s1 = std::sin(result[0]);
            const Scalar c1 = std::cos(result[0]);
            result[2] = std::atan2(s1 * matrix(k, i) - c1 * matrix(j, i), c1 * matrix(j, j) - s1 * matrix(k, j));
        }
        if (!odd)
        {
            for (Index index = 0; index < 3; ++index)
                result[index] = -result[index];
        }
        return result;
    }

    template <typename Derived>
    Matrix<typename MatrixBase<Derived>::Scalar, 3, 1>
    MatrixBase<Derived>::eulerAngles(Index a0, Index a1, Index a2) const
    {
        geometry_detail::validateEulerAxes(a0, a1, a2);
        if (this->rows() != 3 || this->cols() != 3)
        {
            throw std::invalid_argument("eulerAngles() requires a 3x3 matrix");
        }
        Matrix<Scalar, 3, 1> result;
        const auto& matrix = this->derived();
        const Index odd = ((a0 + 1) % 3 == a1) ? 0 : 1;
        const Index i = a0;
        const Index j = (a0 + 1 + odd) % 3;
        const Index k = (a0 + 2 - odd) % 3;

        if (a0 == a2)
        {
            result[0] = std::atan2(matrix(j, i), matrix(k, i));
            if ((odd && result[0] < Scalar{}) || (!odd && result[0] > Scalar{}))
            {
                result[0] += result[0] > Scalar{} ? -Scalar{3.141592653589793238462643383279502884L}
                                                  : Scalar{3.141592653589793238462643383279502884L};
                const Scalar s2 = std::hypot(matrix(j, i), matrix(k, i));
                result[1] = -std::atan2(s2, matrix(i, i));
            }
            else
            {
                const Scalar s2 = std::hypot(matrix(j, i), matrix(k, i));
                result[1] = std::atan2(s2, matrix(i, i));
            }
        }
        else
        {
            result[0] = std::atan2(matrix(j, k), matrix(k, k));
            const Scalar c2 = std::hypot(matrix(i, i), matrix(i, j));
            if ((odd && result[0] < Scalar{}) || (!odd && result[0] > Scalar{}))
            {
                result[0] += result[0] > Scalar{} ? -Scalar{3.141592653589793238462643383279502884L}
                                                  : Scalar{3.141592653589793238462643383279502884L};
                result[1] = std::atan2(-matrix(i, k), -c2);
            }
            else
            {
                result[1] = std::atan2(-matrix(i, k), c2);
            }
        }
        const Scalar s1 = std::sin(result[0]);
        const Scalar c1 = std::cos(result[0]);
        if (a0 == a2)
        {
            result[2] = std::atan2(c1 * matrix(j, k) - s1 * matrix(k, k), c1 * matrix(j, j) - s1 * matrix(k, j));
        }
        else
        {
            result[2] = std::atan2(s1 * matrix(k, i) - c1 * matrix(j, i), c1 * matrix(j, j) - s1 * matrix(k, j));
        }
        if (!odd)
        {
            for (Index index = 0; index < 3; ++index)
                result[index] = -result[index];
        }
        return result;
    }
} // namespace plamatrix::v1
