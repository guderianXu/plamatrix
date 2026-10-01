#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <type_traits>

namespace plamatrix::v1
{
    /// Invert a row-major 3x3 matrix without a decomposition or heap allocation.
    /// abs_determinant_threshold is a finite, nonnegative absolute cutoff; equality is accepted.
    /// Returns false for invalid arguments, non-finite input/output, or a determinant below the cutoff.
    /// On failure, inverse is unchanged and determinant is zero unless a finite determinant was computed.
    /// matrix and inverse may refer to the same array. This CPU function does not inspect execution policy.
    template <typename Scalar>
    bool tryInverse3x3RowMajor(const std::array<Scalar, 9>& matrix,
                               std::array<Scalar, 9>* inverse,
                               Scalar abs_determinant_threshold,
                               Scalar* determinant = nullptr) noexcept
    {
        static_assert(std::is_floating_point_v<Scalar>, "3x3 inversion requires a floating-point scalar");
        if (determinant)
        {
            *determinant = Scalar(0);
        }
        if (!inverse || !std::isfinite(abs_determinant_threshold) || abs_determinant_threshold < Scalar(0))
        {
            return false;
        }
        const Scalar cofactor_00 = matrix[4] * matrix[8] - matrix[5] * matrix[7];
        const Scalar cofactor_01 = matrix[5] * matrix[6] - matrix[3] * matrix[8];
        const Scalar cofactor_02 = matrix[3] * matrix[7] - matrix[4] * matrix[6];
        const Scalar value = matrix[0] * cofactor_00 + matrix[1] * cofactor_01 + matrix[2] * cofactor_02;
        if (!std::isfinite(value))
        {
            return false;
        }
        if (determinant)
        {
            *determinant = value;
        }
        if (value == Scalar(0) || std::abs(value) < abs_determinant_threshold)
        {
            return false;
        }

        const Scalar reciprocal = Scalar(1) / value;
        const std::array<Scalar, 9> candidate{{
            cofactor_00 * reciprocal,
            (matrix[2] * matrix[7] - matrix[1] * matrix[8]) * reciprocal,
            (matrix[1] * matrix[5] - matrix[2] * matrix[4]) * reciprocal,
            cofactor_01 * reciprocal,
            (matrix[0] * matrix[8] - matrix[2] * matrix[6]) * reciprocal,
            (matrix[2] * matrix[3] - matrix[0] * matrix[5]) * reciprocal,
            cofactor_02 * reciprocal,
            (matrix[1] * matrix[6] - matrix[0] * matrix[7]) * reciprocal,
            (matrix[0] * matrix[4] - matrix[1] * matrix[3]) * reciprocal,
        }};
        for (const Scalar entry : candidate)
        {
            if (!std::isfinite(entry))
            {
                return false;
            }
        }
        *inverse = candidate;
        return true;
    }
} // namespace plamatrix::v1
