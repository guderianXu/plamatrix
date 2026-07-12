#pragma once

#include <array>
#include <cmath>
#include <limits>

namespace plamatrix
{

/// Lightweight three-dimensional vector for scalar geometry operations.
/// @tparam Scalar Component type.
template <typename Scalar>
struct Vec3
{
    Scalar x{};
    Scalar y{};
    Scalar z{};

    /// Construct a zero-initialized vector.
    constexpr Vec3() = default;

    /// Construct from three components.
    /// @param x_value X component.
    /// @param y_value Y component.
    /// @param z_value Z component.
    constexpr Vec3(Scalar x_value, Scalar y_value, Scalar z_value)
        : x(x_value), y(y_value), z(z_value)
    {
    }

    /// Construct from an array in x, y, z order.
    /// @param values Source components.
    explicit constexpr Vec3(const std::array<Scalar, 3>& values)
        : x(values[0]), y(values[1]), z(values[2])
    {
    }

    /// Convert to an array in x, y, z order.
    /// @return Array containing the vector components.
    constexpr std::array<Scalar, 3> toArray() const
    {
        return {x, y, z};
    }
};

/// Add two vectors component-wise.
/// @param lhs Left operand.
/// @param rhs Right operand.
/// @return Component-wise sum.
template <typename Scalar>
constexpr Vec3<Scalar> operator+(const Vec3<Scalar>& lhs, const Vec3<Scalar>& rhs)
{
    return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
}

/// Subtract two vectors component-wise.
/// @param lhs Left operand.
/// @param rhs Right operand.
/// @return Component-wise difference.
template <typename Scalar>
constexpr Vec3<Scalar> operator-(const Vec3<Scalar>& lhs, const Vec3<Scalar>& rhs)
{
    return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
}

/// Multiply a vector by a scalar.
/// @param value Input vector.
/// @param scale Scalar multiplier.
/// @return Scaled vector.
template <typename Scalar>
constexpr Vec3<Scalar> operator*(const Vec3<Scalar>& value, Scalar scale)
{
    return {value.x * scale, value.y * scale, value.z * scale};
}

/// Multiply a vector by a scalar with the scalar on the left.
/// @param scale Scalar multiplier.
/// @param value Input vector.
/// @return Scaled vector.
template <typename Scalar>
constexpr Vec3<Scalar> operator*(Scalar scale, const Vec3<Scalar>& value)
{
    return value * scale;
}

/// Divide a vector by a scalar using normal scalar arithmetic.
/// @param value Input vector.
/// @param divisor Scalar divisor.
/// @return Component-wise quotient.
template <typename Scalar>
constexpr Vec3<Scalar> operator/(const Vec3<Scalar>& value, Scalar divisor)
{
    return {value.x / divisor, value.y / divisor, value.z / divisor};
}

/// Add a vector to another vector in place.
/// @param lhs Vector to update.
/// @param rhs Vector to add.
/// @return Reference to lhs.
template <typename Scalar>
constexpr Vec3<Scalar>& operator+=(Vec3<Scalar>& lhs, const Vec3<Scalar>& rhs)
{
    lhs.x += rhs.x;
    lhs.y += rhs.y;
    lhs.z += rhs.z;
    return lhs;
}

/// Subtract a vector from another vector in place.
/// @param lhs Vector to update.
/// @param rhs Vector to subtract.
/// @return Reference to lhs.
template <typename Scalar>
constexpr Vec3<Scalar>& operator-=(Vec3<Scalar>& lhs, const Vec3<Scalar>& rhs)
{
    lhs.x -= rhs.x;
    lhs.y -= rhs.y;
    lhs.z -= rhs.z;
    return lhs;
}

/// Compute the dot product of two vectors.
/// @param lhs Left operand.
/// @param rhs Right operand.
/// @return Scalar dot product.
template <typename Scalar>
constexpr Scalar dot(const Vec3<Scalar>& lhs, const Vec3<Scalar>& rhs)
{
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

/// Compute the right-handed cross product of two vectors.
/// @param lhs Left operand.
/// @param rhs Right operand.
/// @return Cross-product vector.
template <typename Scalar>
constexpr Vec3<Scalar> cross(const Vec3<Scalar>& lhs, const Vec3<Scalar>& rhs)
{
    return {
        lhs.y * rhs.z - lhs.z * rhs.y,
        lhs.z * rhs.x - lhs.x * rhs.z,
        lhs.x * rhs.y - lhs.y * rhs.x
    };
}

/// Compute the squared Euclidean norm.
/// @param value Input vector.
/// @return Sum of squared components.
template <typename Scalar>
constexpr Scalar squaredNorm(const Vec3<Scalar>& value)
{
    return dot(value, value);
}

/// Compute the Euclidean norm.
/// @param value Input vector.
/// @return Vector length.
template <typename Scalar>
Scalar norm(const Vec3<Scalar>& value)
{
    using std::sqrt;
    return sqrt(squaredNorm(value));
}

/// Normalize a vector unless its length is at or below the supplied threshold.
/// @param value Input vector.
/// @param epsilon Near-zero length threshold.
/// @return Unit vector, or the unchanged input when its length is at or below epsilon.
template <typename Scalar>
Vec3<Scalar> normalized(
    const Vec3<Scalar>& value,
    Scalar epsilon = std::numeric_limits<Scalar>::epsilon())
{
    const Scalar length = norm(value);
    if (length <= epsilon)
    {
        return value;
    }
    return value / length;
}

/// Test whether every component is finite.
/// @param value Input vector.
/// @return True when no component is NaN or infinity.
template <typename Scalar>
bool isFinite(const Vec3<Scalar>& value)
{
    using std::isfinite;
    return isfinite(value.x) && isfinite(value.y) && isfinite(value.z);
}

} // namespace plamatrix
