#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <type_traits>

#include "plamatrix/dense/matrix.h"

namespace plamatrix::v1
{
    template <typename Derived, int Dim> class RotationBase
    {
    public:
        Derived& derived() noexcept
        {
            return *static_cast<Derived*>(this);
        }
        const Derived& derived() const noexcept
        {
            return *static_cast<const Derived*>(this);
        }

        auto matrix() const
        {
            return derived().toRotationMatrix();
        }
        auto inverse() const
        {
            return derived().inverse();
        }

        template <typename OtherDerived> auto operator*(const MatrixBase<OtherDerived>& vector) const
        {
            using Scalar = typename OtherDerived::Scalar;
            if (vector.rows() != Dim || vector.cols() != 1)
            {
                throw std::invalid_argument("rotation requires a compatible column vector");
            }
            Matrix<Scalar, Dim, 1> result(Dim, 1);
            const auto rotation = derived().toRotationMatrix();
            for (Index row = 0; row < Dim; ++row)
            {
                Scalar value{};
                for (Index inner = 0; inner < Dim; ++inner)
                {
                    value += rotation(row, inner) * vector.derived()(inner, 0);
                }
                result(row) = value;
            }
            return result;
        }
    };

    namespace geometry_detail
    {
        template <typename Type> struct QuaternionTraits;
        template <typename Scalar, int Options> struct QuaternionTraits<Quaternion<Scalar, Options>>
        {
            using ScalarType = Scalar;
        };

        template <typename T> inline T clamp(T value, T lower, T upper)
        {
            return std::max(lower, std::min(upper, value));
        }

        template <typename Scalar> Matrix<Scalar, 3, 1> unitOrthogonal(const Matrix<Scalar, 3, 1>& vector)
        {
            Matrix<Scalar, 3, 1> result;
            if (std::abs(vector.x()) > std::abs(vector.z()))
            {
                result << -vector.y(), vector.x(), Scalar{};
            }
            else
            {
                result << Scalar{}, -vector.z(), vector.y();
            }
            return result.normalized();
        }
    } // namespace geometry_detail

    template <typename Derived> class QuaternionBase : public RotationBase<Derived, 3>
    {
    public:
        using Scalar = typename geometry_detail::QuaternionTraits<Derived>::ScalarType;
        using RealScalar = Scalar;
        using Vector3 = Matrix<Scalar, 3, 1>;
        using Matrix3 = Matrix<Scalar, 3, 3>;
        using AngleAxisType = AngleAxis<Scalar>;

        Derived& derived() noexcept
        {
            return *static_cast<Derived*>(this);
        }
        const Derived& derived() const noexcept
        {
            return *static_cast<const Derived*>(this);
        }

        Scalar x() const
        {
            return derived().coeffs()(0);
        }
        Scalar y() const
        {
            return derived().coeffs()(1);
        }
        Scalar z() const
        {
            return derived().coeffs()(2);
        }
        Scalar w() const
        {
            return derived().coeffs()(3);
        }
        Scalar& x()
        {
            return derived().coeffs()(0);
        }
        Scalar& y()
        {
            return derived().coeffs()(1);
        }
        Scalar& z()
        {
            return derived().coeffs()(2);
        }
        Scalar& w()
        {
            return derived().coeffs()(3);
        }

        const auto& coeffs() const
        {
            return derived().coeffs();
        }
        auto& coeffs()
        {
            return derived().coeffs();
        }
        auto vec() const
        {
            return coeffs().template head<3>();
        }
        auto vec()
        {
            return coeffs().template head<3>();
        }

        Matrix<Scalar, 4, 1> coeffsScalarFirst() const
        {
            return Matrix<Scalar, 4, 1>(w(), x(), y(), z());
        }
        Matrix<Scalar, 4, 1> coeffsScalarLast() const
        {
            return coeffs();
        }

        static Quaternion<Scalar> Identity()
        {
            return Quaternion<Scalar>(Scalar{1}, Scalar{}, Scalar{}, Scalar{});
        }
        QuaternionBase& setIdentity()
        {
            coeffs() << Scalar{}, Scalar{}, Scalar{}, Scalar{1};
            return *this;
        }

        Scalar squaredNorm() const
        {
            return coeffs().squaredNorm();
        }
        Scalar norm() const
        {
            return coeffs().norm();
        }
        void normalize()
        {
            coeffs().normalize();
        }
        Quaternion<Scalar> normalized() const;

        template <typename OtherDerived> Scalar dot(const QuaternionBase<OtherDerived>& other) const
        {
            return x() * other.x() + y() * other.y() + z() * other.z() + w() * other.w();
        }

        template <typename OtherDerived> Scalar angularDistance(const QuaternionBase<OtherDerived>& other) const;
        Matrix3 toRotationMatrix() const;

        template <typename VectorDerivedA, typename VectorDerivedB>
        Derived& setFromTwoVectors(const MatrixBase<VectorDerivedA>& a, const MatrixBase<VectorDerivedB>& b);

        template <typename OtherDerived> Quaternion<Scalar> operator*(const QuaternionBase<OtherDerived>& other) const;
        template <typename OtherDerived> Derived& operator*=(const QuaternionBase<OtherDerived>& other)
        {
            derived() = derived() * other;
            return derived();
        }

        Quaternion<Scalar> inverse() const;
        Quaternion<Scalar> conjugate() const
        {
            return Quaternion<Scalar>(w(), -x(), -y(), -z());
        }

        template <typename OtherDerived>
        Quaternion<Scalar> slerp(const Scalar& t, const QuaternionBase<OtherDerived>& other) const;

        template <typename OtherDerived>
        bool isApprox(const QuaternionBase<OtherDerived>& other,
                      Scalar precision = Scalar{100} * std::numeric_limits<Scalar>::epsilon()) const
        {
            return coeffs().isApprox(other.coeffs(), precision);
        }
        template <typename OtherDerived> bool operator==(const QuaternionBase<OtherDerived>& other) const
        {
            return x() == other.x() && y() == other.y() && z() == other.z() && w() == other.w();
        }
        template <typename OtherDerived> bool operator!=(const QuaternionBase<OtherDerived>& other) const
        {
            return !(*this == other);
        }

        template <typename VectorDerived> Vector3 operator*(const MatrixBase<VectorDerived>& vector) const
        {
            return RotationBase<Derived, 3>::operator*(vector);
        }
    };

    template <typename Scalar_, int Options> class Quaternion : public QuaternionBase<Quaternion<Scalar_, Options>>
    {
    public:
        using Scalar = Scalar_;
        using ScalarType = Scalar_;
        using Coefficients = Matrix<Scalar, 4, 1>;
        using Matrix3 = Matrix<Scalar, 3, 3>;

        Quaternion() = default;
        Quaternion(const Scalar& w, const Scalar& x, const Scalar& y, const Scalar& z) : _coeffs(x, y, z, w)
        {
        }
        explicit Quaternion(const Coefficients& coefficients) : _coeffs(coefficients)
        {
        }

        template <typename OtherScalar, int OtherOptions>
        explicit Quaternion(const Quaternion<OtherScalar, OtherOptions>& other)
            : _coeffs(other.coeffs().template cast<Scalar>())
        {
        }

        explicit Quaternion(const AngleAxis<Scalar>& angle_axis)
        {
            *this = angle_axis;
        }

        template <typename MatrixDerived> explicit Quaternion(const MatrixBase<MatrixDerived>& matrix)
        {
            *this = matrix;
        }

        static Quaternion Identity()
        {
            return Quaternion(Scalar{1}, Scalar{}, Scalar{}, Scalar{});
        }
        static Quaternion UnitRandom()
        {
            thread_local std::mt19937 engine(std::random_device{}());
            std::uniform_real_distribution<Scalar> unit(Scalar{}, Scalar{1});
            const Scalar pi = std::acos(Scalar{-1});
            const Scalar u1 = unit(engine);
            const Scalar u2 = Scalar{2} * pi * unit(engine);
            const Scalar u3 = Scalar{2} * pi * unit(engine);
            const Scalar a = std::sqrt(Scalar{1} - u1);
            const Scalar b = std::sqrt(u1);
            return Quaternion(a * std::sin(u2), a * std::cos(u2), b * std::sin(u3), b * std::cos(u3));
        }
        static Quaternion FromCoeffsScalarLast(const Scalar& x, const Scalar& y, const Scalar& z, const Scalar& w)
        {
            return Quaternion(w, x, y, z);
        }
        static Quaternion FromCoeffsScalarFirst(const Scalar& w, const Scalar& x, const Scalar& y, const Scalar& z)
        {
            return Quaternion(w, x, y, z);
        }
        template <typename VectorDerivedA, typename VectorDerivedB>
        static Quaternion FromTwoVectors(const MatrixBase<VectorDerivedA>& a, const MatrixBase<VectorDerivedB>& b)
        {
            Quaternion result;
            result.setFromTwoVectors(a, b);
            return result;
        }

        const Coefficients& coeffs() const noexcept
        {
            return _coeffs;
        }
        Coefficients& coeffs() noexcept
        {
            return _coeffs;
        }
        Coefficients coeffsScalarFirst() const
        {
            return Coefficients(this->w(), this->x(), this->y(), this->z());
        }
        Coefficients coeffsScalarLast() const
        {
            return _coeffs;
        }

        Quaternion& operator=(const AngleAxis<Scalar>& angle_axis);

        template <typename MatrixDerived> Quaternion& operator=(const MatrixBase<MatrixDerived>& source)
        {
            if (source.size() == 4 && (source.rows() == 1 || source.cols() == 1))
            {
                this->x() = source.derived()(0);
                this->y() = source.derived()(1);
                this->z() = source.derived()(2);
                this->w() = source.derived()(3);
                return *this;
            }
            if (source.rows() != 3 || source.cols() != 3)
            {
                throw std::invalid_argument("Quaternion input must be a 3x3 rotation matrix or 4-vector");
            }
            const auto& matrix = source.derived();
            const Scalar trace = matrix(0, 0) + matrix(1, 1) + matrix(2, 2);
            if (trace > Scalar{})
            {
                const Scalar scale = Scalar{2} * std::sqrt(trace + Scalar{1});
                this->w() = scale / Scalar{4};
                this->x() = (matrix(2, 1) - matrix(1, 2)) / scale;
                this->y() = (matrix(0, 2) - matrix(2, 0)) / scale;
                this->z() = (matrix(1, 0) - matrix(0, 1)) / scale;
            }
            else
            {
                Index axis = 0;
                if (matrix(1, 1) > matrix(0, 0))
                    axis = 1;
                if (matrix(2, 2) > matrix(axis, axis))
                    axis = 2;
                const Index next = (axis + 1) % 3;
                const Index last = (axis + 2) % 3;
                const Scalar scale =
                    Scalar{2} *
                    std::sqrt(
                        std::max(Scalar{}, Scalar{1} + matrix(axis, axis) - matrix(next, next) - matrix(last, last)));
                if (scale <= std::numeric_limits<Scalar>::epsilon())
                {
                    this->setIdentity();
                    return *this;
                }
                Scalar values[4]{};
                values[axis] = scale / Scalar{4};
                values[3] = (matrix(last, next) - matrix(next, last)) / scale;
                values[next] = (matrix(next, axis) + matrix(axis, next)) / scale;
                values[last] = (matrix(last, axis) + matrix(axis, last)) / scale;
                this->x() = values[0];
                this->y() = values[1];
                this->z() = values[2];
                this->w() = values[3];
            }
            this->normalize();
            return *this;
        }

        template <typename NewScalar> Quaternion<NewScalar, Options> cast() const
        {
            return Quaternion<NewScalar, Options>(static_cast<NewScalar>(this->w()),
                                                  static_cast<NewScalar>(this->x()),
                                                  static_cast<NewScalar>(this->y()),
                                                  static_cast<NewScalar>(this->z()));
        }

    private:
        Coefficients _coeffs;
    };

    template <typename Scalar_> class AngleAxis : public RotationBase<AngleAxis<Scalar_>, 3>
    {
    public:
        using Scalar = Scalar_;
        using ScalarType = Scalar_;
        using Matrix3 = Matrix<Scalar, 3, 3>;
        using Vector3 = Matrix<Scalar, 3, 1>;
        using QuaternionType = Quaternion<Scalar>;

        AngleAxis() = default;
        template <typename AxisDerived>
        AngleAxis(const Scalar& angle, const MatrixBase<AxisDerived>& axis) : _axis(axis.derived()), _angle(angle)
        {
            if (axis.rows() * axis.cols() != 3)
            {
                throw std::invalid_argument("AngleAxis axis must have three coefficients");
            }
        }
        template <typename QuaternionDerived> explicit AngleAxis(const QuaternionBase<QuaternionDerived>& quaternion)
        {
            *this = quaternion;
        }
        template <typename MatrixDerived> explicit AngleAxis(const MatrixBase<MatrixDerived>& matrix)
        {
            *this = matrix;
        }
        template <typename OtherScalar>
        explicit AngleAxis(const AngleAxis<OtherScalar>& other)
            : _axis(other.axis().template cast<Scalar>()), _angle(static_cast<Scalar>(other.angle()))
        {
        }

        Scalar angle() const
        {
            return _angle;
        }
        Scalar& angle()
        {
            return _angle;
        }
        const Vector3& axis() const
        {
            return _axis;
        }
        Vector3& axis()
        {
            return _axis;
        }

        static AngleAxis Identity()
        {
            return AngleAxis(Scalar{}, Vector3::UnitX());
        }
        AngleAxis inverse() const
        {
            return AngleAxis(-_angle, _axis);
        }

        QuaternionType operator*(const AngleAxis& other) const
        {
            return QuaternionType(*this) * QuaternionType(other);
        }
        QuaternionType operator*(const QuaternionType& other) const
        {
            return QuaternionType(*this) * other;
        }
        friend QuaternionType operator*(const QuaternionType& left, const AngleAxis& right)
        {
            return left * QuaternionType(right);
        }

        template <typename QuaternionDerived> AngleAxis& operator=(const QuaternionBase<QuaternionDerived>& input)
        {
            QuaternionType quaternion(input.w(), input.x(), input.y(), input.z());
            quaternion.normalize();
            const Scalar vector_norm = std::sqrt(quaternion.x() * quaternion.x() + quaternion.y() * quaternion.y() +
                                                 quaternion.z() * quaternion.z());
            if (vector_norm <= std::numeric_limits<Scalar>::epsilon())
            {
                _angle = Scalar{};
                _axis = Vector3::UnitX();
            }
            else
            {
                const Scalar sign = quaternion.w() < Scalar{} ? Scalar{-1} : Scalar{1};
                _angle = Scalar{2} * std::atan2(vector_norm, std::abs(quaternion.w()));
                _axis << sign * quaternion.x() / vector_norm, sign * quaternion.y() / vector_norm,
                    sign * quaternion.z() / vector_norm;
            }
            return *this;
        }

        template <typename MatrixDerived> AngleAxis& operator=(const MatrixBase<MatrixDerived>& matrix)
        {
            return *this = QuaternionType(matrix);
        }
        template <typename MatrixDerived> AngleAxis& fromRotationMatrix(const MatrixBase<MatrixDerived>& matrix)
        {
            return *this = matrix;
        }

        Matrix3 toRotationMatrix() const
        {
            const Scalar x = _axis.x();
            const Scalar y = _axis.y();
            const Scalar z = _axis.z();
            const Scalar c = std::cos(_angle);
            const Scalar s = std::sin(_angle);
            const Scalar one_minus_c = Scalar{1} - c;
            Matrix3 result;
            result << c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s,
                y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s,
                z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c;
            return result;
        }

        template <typename NewScalar> AngleAxis<NewScalar> cast() const
        {
            return AngleAxis<NewScalar>(static_cast<NewScalar>(_angle), _axis.template cast<NewScalar>());
        }
        bool isApprox(const AngleAxis& other,
                      Scalar precision = Scalar{100} * std::numeric_limits<Scalar>::epsilon()) const
        {
            return _axis.isApprox(other._axis, precision) && std::abs(_angle - other._angle) <= precision;
        }

        template <typename VectorDerived> Vector3 operator*(const MatrixBase<VectorDerived>& vector) const
        {
            return RotationBase<AngleAxis, 3>::operator*(vector);
        }

    private:
        Vector3 _axis{};
        Scalar _angle{};
    };

    template <typename Derived>
    Quaternion<typename QuaternionBase<Derived>::Scalar> QuaternionBase<Derived>::normalized() const
    {
        Quaternion<Scalar> result(w(), x(), y(), z());
        result.normalize();
        return result;
    }

    template <typename Derived>
    template <typename OtherDerived>
    typename QuaternionBase<Derived>::Scalar
    QuaternionBase<Derived>::angularDistance(const QuaternionBase<OtherDerived>& other) const
    {
        const Quaternion<Scalar> difference = this->inverse() * other;
        const Scalar vector_norm = std::sqrt(difference.x() * difference.x() + difference.y() * difference.y() +
                                             difference.z() * difference.z());
        return Scalar{2} * std::atan2(vector_norm, std::abs(difference.w()));
    }

    template <typename Derived>
    typename QuaternionBase<Derived>::Matrix3 QuaternionBase<Derived>::toRotationMatrix() const
    {
        const Scalar tx = Scalar{2} * x();
        const Scalar ty = Scalar{2} * y();
        const Scalar tz = Scalar{2} * z();
        const Scalar twx = tx * w();
        const Scalar twy = ty * w();
        const Scalar twz = tz * w();
        const Scalar txx = tx * x();
        const Scalar txy = ty * x();
        const Scalar txz = tz * x();
        const Scalar tyy = ty * y();
        const Scalar tyz = tz * y();
        const Scalar tzz = tz * z();
        Matrix3 result;
        result << Scalar{1} - (tyy + tzz), txy - twz, txz + twy, txy + twz, Scalar{1} - (txx + tzz), tyz - twx,
            txz - twy, tyz + twx, Scalar{1} - (txx + tyy);
        return result;
    }

    template <typename Derived>
    template <typename VectorDerivedA, typename VectorDerivedB>
    Derived& QuaternionBase<Derived>::setFromTwoVectors(const MatrixBase<VectorDerivedA>& a,
                                                        const MatrixBase<VectorDerivedB>& b)
    {
        Vector3 first(a.derived());
        Vector3 second(b.derived());
        if (first.size() != 3 || second.size() != 3)
        {
            throw std::invalid_argument("setFromTwoVectors requires 3D vectors");
        }
        first.normalize();
        second.normalize();
        const Scalar cosine = geometry_detail::clamp(first.dot(second), Scalar{-1}, Scalar{1});
        if (cosine < Scalar{-1} + Scalar{10} * std::numeric_limits<Scalar>::epsilon())
        {
            const Vector3 axis = geometry_detail::unitOrthogonal(first);
            derived() = Quaternion<Scalar>(Scalar{}, axis.x(), axis.y(), axis.z());
        }
        else
        {
            const Vector3 axis = first.cross(second);
            derived() = Quaternion<Scalar>(Scalar{1} + cosine, axis.x(), axis.y(), axis.z()).normalized();
        }
        return derived();
    }

    template <typename Derived>
    template <typename OtherDerived>
    Quaternion<typename QuaternionBase<Derived>::Scalar>
    QuaternionBase<Derived>::operator*(const QuaternionBase<OtherDerived>& other) const
    {
        return Quaternion<Scalar>(w() * other.w() - x() * other.x() - y() * other.y() - z() * other.z(),
                                  w() * other.x() + x() * other.w() + y() * other.z() - z() * other.y(),
                                  w() * other.y() - x() * other.z() + y() * other.w() + z() * other.x(),
                                  w() * other.z() + x() * other.y() - y() * other.x() + z() * other.w());
    }

    template <typename Derived>
    Quaternion<typename QuaternionBase<Derived>::Scalar> QuaternionBase<Derived>::inverse() const
    {
        const Scalar squared_norm = squaredNorm();
        if (squared_norm <= std::numeric_limits<Scalar>::epsilon())
        {
            return Quaternion<Scalar>(Scalar{}, Scalar{}, Scalar{}, Scalar{});
        }
        return Quaternion<Scalar>(w() / squared_norm, -x() / squared_norm, -y() / squared_norm, -z() / squared_norm);
    }

    template <typename Derived>
    template <typename OtherDerived>
    Quaternion<typename QuaternionBase<Derived>::Scalar>
    QuaternionBase<Derived>::slerp(const Scalar& t, const QuaternionBase<OtherDerived>& other) const
    {
        Scalar cosine = dot(other);
        Scalar sign = Scalar{1};
        if (cosine < Scalar{})
        {
            cosine = -cosine;
            sign = Scalar{-1};
        }
        cosine = geometry_detail::clamp(cosine, Scalar{-1}, Scalar{1});
        Scalar left_weight = Scalar{1} - t;
        Scalar right_weight = t;
        if (Scalar{1} - cosine > Scalar{10} * std::numeric_limits<Scalar>::epsilon())
        {
            const Scalar angle = std::acos(cosine);
            const Scalar sine = std::sin(angle);
            left_weight = std::sin((Scalar{1} - t) * angle) / sine;
            right_weight = std::sin(t * angle) / sine;
        }
        return Quaternion<Scalar>(left_weight * w() + right_weight * sign * other.w(),
                                  left_weight * x() + right_weight * sign * other.x(),
                                  left_weight * y() + right_weight * sign * other.y(),
                                  left_weight * z() + right_weight * sign * other.z())
            .normalized();
    }

    template <typename Scalar, int Options>
    Quaternion<Scalar, Options>& Quaternion<Scalar, Options>::operator=(const AngleAxis<Scalar>& angle_axis)
    {
        const Scalar half = angle_axis.angle() / Scalar{2};
        const Scalar sine = std::sin(half);
        this->w() = std::cos(half);
        this->x() = angle_axis.axis().x() * sine;
        this->y() = angle_axis.axis().y() * sine;
        this->z() = angle_axis.axis().z() * sine;
        return *this;
    }

    using Quaternionf = Quaternion<float>;
    using Quaterniond = Quaternion<double>;
    using AngleAxisf = AngleAxis<float>;
    using AngleAxisd = AngleAxis<double>;
} // namespace plamatrix::v1
