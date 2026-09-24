#pragma once

#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "plamatrix/geometry/rotation.h"

namespace plamatrix::v1
{
    template <typename Scalar_> class UniformScaling
    {
    public:
        using Scalar = Scalar_;

        UniformScaling() = default;
        explicit UniformScaling(const Scalar& factor) : _factor(factor)
        {
        }

        const Scalar& factor() const noexcept
        {
            return _factor;
        }
        Scalar& factor() noexcept
        {
            return _factor;
        }
        UniformScaling inverse() const
        {
            if (_factor == Scalar{})
                throw std::invalid_argument("UniformScaling inverse requires a nonzero factor");
            return UniformScaling(Scalar{1} / _factor);
        }
        UniformScaling operator*(const UniformScaling& other) const
        {
            return UniformScaling(_factor * other._factor);
        }
        template <typename Derived> auto operator*(const MatrixBase<Derived>& matrix) const
        {
            return (_factor * matrix.derived()).eval();
        }
        template <typename NewScalar> UniformScaling<NewScalar> cast() const
        {
            return UniformScaling<NewScalar>(static_cast<NewScalar>(_factor));
        }
        bool isApprox(const UniformScaling& other,
                      Scalar precision = Scalar{100} * std::numeric_limits<Scalar>::epsilon()) const
        {
            return std::abs(_factor - other._factor) <=
                   precision * std::max({Scalar{1}, std::abs(_factor), std::abs(other._factor)});
        }

    private:
        Scalar _factor{};
    };

    template <typename Scalar> UniformScaling<Scalar> Scaling(const Scalar& factor)
    {
        return UniformScaling<Scalar>(factor);
    }

    template <typename Scalar_, int Dim_> class Translation
    {
    public:
        using Scalar = Scalar_;
        static constexpr int Dim = Dim_;
        using VectorType = Matrix<Scalar, Dim, 1>;
        using LinearMatrixType = Matrix<Scalar, Dim, Dim>;
        using AffineTransformType = Transform<Scalar, Dim, Affine>;
        using IsometryTransformType = Transform<Scalar, Dim, Isometry>;

        Translation() = default;
        Translation(const Scalar& x, const Scalar& y) : _coeffs(x, y)
        {
            static_assert(Dim == 2, "two-coefficient Translation requires Dim=2");
        }
        Translation(const Scalar& x, const Scalar& y, const Scalar& z) : _coeffs(x, y, z)
        {
            static_assert(Dim == 3, "three-coefficient Translation requires Dim=3");
        }
        explicit Translation(const VectorType& vector) : _coeffs(vector)
        {
        }
        template <typename OtherScalar>
        explicit Translation(const Translation<OtherScalar, Dim>& other)
            : _coeffs(other.vector().template cast<Scalar>())
        {
        }

        Scalar x() const
        {
            return _coeffs.x();
        }
        Scalar y() const
        {
            return _coeffs.y();
        }
        Scalar z() const
        {
            return _coeffs.z();
        }
        Scalar& x()
        {
            return _coeffs.x();
        }
        Scalar& y()
        {
            return _coeffs.y();
        }
        Scalar& z()
        {
            return _coeffs.z();
        }
        const VectorType& vector() const noexcept
        {
            return _coeffs;
        }
        VectorType& vector() noexcept
        {
            return _coeffs;
        }
        const VectorType& translation() const noexcept
        {
            return _coeffs;
        }
        VectorType& translation() noexcept
        {
            return _coeffs;
        }

        Translation operator*(const Translation& other) const
        {
            VectorType result;
            for (Index index = 0; index < Dim; ++index)
                result(index) = _coeffs(index) + other._coeffs(index);
            return Translation(result);
        }

        template <typename Derived> VectorType operator*(const MatrixBase<Derived>& vector) const
        {
            if (vector.rows() != Dim || vector.cols() != 1)
                throw std::invalid_argument("Translation requires a compatible column vector");
            VectorType result;
            for (Index index = 0; index < Dim; ++index)
                result(index) = _coeffs(index) + vector.derived()(index, 0);
            return result;
        }

        Translation inverse() const
        {
            VectorType result;
            for (Index index = 0; index < Dim; ++index)
                result(index) = -_coeffs(index);
            return Translation(result);
        }
        static Translation Identity()
        {
            return Translation(VectorType::Zero());
        }
        template <typename NewScalar> Translation<NewScalar, Dim> cast() const
        {
            return Translation<NewScalar, Dim>(_coeffs.template cast<NewScalar>());
        }
        bool isApprox(const Translation& other,
                      Scalar precision = Scalar{100} * std::numeric_limits<Scalar>::epsilon()) const
        {
            return _coeffs.isApprox(other._coeffs, precision);
        }

        AffineTransformType operator*(const UniformScaling<Scalar>& scaling) const;
        template <typename RotationDerived>
        IsometryTransformType operator*(const RotationBase<RotationDerived, Dim>& rotation) const;
        template <int Mode, int Options>
        Transform<Scalar, Dim, Mode> operator*(const Transform<Scalar, Dim, Mode, Options>& transform) const
        {
            Transform<Scalar, Dim, Mode> result(transform);
            result.pretranslate(_coeffs);
            return result;
        }

    private:
        VectorType _coeffs{};
    };

    template <typename Scalar_, int Dim_, int Mode_, int Options_> class Transform
    {
        static_assert(Dim_ > 0, "PlaMatrix Transform currently requires a fixed positive dimension");
        static_assert(Mode_ == Isometry || Mode_ == Affine || Mode_ == AffineCompact || Mode_ == Projective,
                      "unsupported Transform mode");

    public:
        using Scalar = Scalar_;
        static constexpr int Dim = Dim_;
        static constexpr int Mode = Mode_;
        static constexpr int Options = Options_;
        static constexpr int Rows = Mode == AffineCompact ? Dim : Dim + 1;
        using MatrixType = Matrix<Scalar, Rows, Dim + 1, Options>;
        using LinearMatrixType = Matrix<Scalar, Dim, Dim>;
        using VectorType = Matrix<Scalar, Dim, 1>;
        using TranslationType = Translation<Scalar, Dim>;
        using ConstLinearPart = Block<const MatrixType, Dim, Dim>;
        using RotationReturnType = std::conditional_t<Mode == Isometry, ConstLinearPart, LinearMatrixType>;

        Transform() = default;
        explicit Transform(const MatrixType& matrix) : _matrix(matrix)
        {
        }

        template <typename Derived> explicit Transform(const MatrixBase<Derived>& source)
        {
            setIdentity();
            if (source.rows() == Rows && source.cols() == Dim + 1)
            {
                for (Index row = 0; row < Rows; ++row)
                    for (Index col = 0; col < Dim + 1; ++col)
                        _matrix(row, col) = source.derived()(row, col);
            }
            else if (source.rows() == Dim && source.cols() == Dim)
            {
                for (Index row = 0; row < Dim; ++row)
                    for (Index col = 0; col < Dim; ++col)
                        _matrix(row, col) = source.derived()(row, col);
            }
            else
            {
                throw std::invalid_argument("Transform matrix has an incompatible shape");
            }
        }

        template <typename RotationDerived> Transform(const RotationBase<RotationDerived, Dim>& rotation)
        {
            setIdentity();
            assignLinear(rotation.derived().toRotationMatrix());
        }
        Transform(const TranslationType& translation)
        {
            setIdentity();
            assignTranslation(translation.vector());
        }
        Transform(const UniformScaling<Scalar>& scaling)
        {
            setIdentity();
            for (Index index = 0; index < Dim; ++index)
                _matrix(index, index) = scaling.factor();
        }
        template <int OtherMode, int OtherOptions>
        Transform(const Transform<Scalar, Dim, OtherMode, OtherOptions>& other)
        {
            assignFromHomogeneous(other.homogeneousMatrix());
        }

        static Transform Identity()
        {
            Transform result;
            result.setIdentity();
            return result;
        }
        void setIdentity()
        {
            _matrix.setZero();
            for (Index index = 0; index < Dim; ++index)
                _matrix(index, index) = Scalar{1};
            if constexpr (Mode != AffineCompact)
                _matrix(Dim, Dim) = Scalar{1};
        }

        Scalar operator()(Index row, Index col) const
        {
            return _matrix(row, col);
        }
        Scalar& operator()(Index row, Index col)
        {
            return _matrix(row, col);
        }
        const MatrixType& matrix() const noexcept
        {
            return _matrix;
        }
        MatrixType& matrix() noexcept
        {
            return _matrix;
        }
        auto linear()
        {
            return _matrix.template block<Dim, Dim>(0, 0);
        }
        auto linear() const
        {
            return _matrix.template block<Dim, Dim>(0, 0);
        }
        auto translation()
        {
            return _matrix.template block<Dim, 1>(0, Dim);
        }
        auto translation() const
        {
            return _matrix.template block<Dim, 1>(0, Dim);
        }
        auto affine()
        {
            return _matrix.template block<Dim, Dim + 1>(0, 0);
        }
        auto affine() const
        {
            return _matrix.template block<Dim, Dim + 1>(0, 0);
        }
        RotationReturnType rotation() const
        {
            if constexpr (Mode == Isometry)
            {
                return linear();
            }
            else
            {
                LinearMatrixType result;
                computeRotationScaling(&result, static_cast<LinearMatrixType*>(nullptr));
                return result;
            }
        }

        template <typename RotationMatrixType, typename ScalingMatrixType>
        void computeRotationScaling(RotationMatrixType* rotation_matrix, ScalingMatrixType* scaling_matrix) const
        {
            JacobiSVD<LinearMatrixType> decomposition(LinearMatrixType(linear()), ComputeFullU | ComputeFullV);
            if (decomposition.info() != Success)
                throw std::runtime_error("Transform rotation-scaling SVD failed");
            const auto& u = decomposition.matrixU();
            const auto& v = decomposition.matrixV();
            VectorType singular_values(decomposition.singularValues());
            LinearMatrixType uv_transpose;
            for (Index row = 0; row < Dim; ++row)
                for (Index col = 0; col < Dim; ++col)
                {
                    Scalar value{};
                    for (Index inner = 0; inner < Dim; ++inner)
                        value += u(row, inner) * v(col, inner);
                    uv_transpose(row, col) = value;
                }
            const Scalar sign = uv_transpose.fullPivLu().determinant() < Scalar{} ? Scalar{-1} : Scalar{1};
            singular_values(Dim - 1) *= sign;
            if (rotation_matrix != nullptr)
            {
                for (Index row = 0; row < Dim; ++row)
                    for (Index col = 0; col < Dim; ++col)
                    {
                        Scalar value{};
                        for (Index inner = 0; inner < Dim; ++inner)
                        {
                            const Scalar adjusted_u = inner == Dim - 1 ? sign * u(row, inner) : u(row, inner);
                            value += adjusted_u * v(col, inner);
                        }
                        (*rotation_matrix)(row, col) = value;
                    }
            }
            if (scaling_matrix != nullptr)
            {
                for (Index row = 0; row < Dim; ++row)
                    for (Index col = 0; col < Dim; ++col)
                    {
                        Scalar value{};
                        for (Index inner = 0; inner < Dim; ++inner)
                            value += v(row, inner) * singular_values(inner) * v(col, inner);
                        (*scaling_matrix)(row, col) = value;
                    }
            }
        }

        template <typename ScalingMatrixType, typename RotationMatrixType>
        void computeScalingRotation(ScalingMatrixType* scaling_matrix, RotationMatrixType* rotation_matrix) const
        {
            JacobiSVD<LinearMatrixType> decomposition(LinearMatrixType(linear()), ComputeFullU | ComputeFullV);
            if (decomposition.info() != Success)
                throw std::runtime_error("Transform scaling-rotation SVD failed");
            const auto& u = decomposition.matrixU();
            const auto& v = decomposition.matrixV();
            VectorType singular_values(decomposition.singularValues());
            LinearMatrixType uv_transpose;
            for (Index row = 0; row < Dim; ++row)
                for (Index col = 0; col < Dim; ++col)
                {
                    Scalar value{};
                    for (Index inner = 0; inner < Dim; ++inner)
                        value += u(row, inner) * v(col, inner);
                    uv_transpose(row, col) = value;
                }
            const Scalar sign = uv_transpose.fullPivLu().determinant() < Scalar{} ? Scalar{-1} : Scalar{1};
            singular_values(Dim - 1) *= sign;
            if (rotation_matrix != nullptr)
            {
                for (Index row = 0; row < Dim; ++row)
                    for (Index col = 0; col < Dim; ++col)
                    {
                        Scalar value{};
                        for (Index inner = 0; inner < Dim; ++inner)
                        {
                            const Scalar adjusted_u = inner == Dim - 1 ? sign * u(row, inner) : u(row, inner);
                            value += adjusted_u * v(col, inner);
                        }
                        (*rotation_matrix)(row, col) = value;
                    }
            }
            if (scaling_matrix != nullptr)
            {
                for (Index row = 0; row < Dim; ++row)
                    for (Index col = 0; col < Dim; ++col)
                    {
                        Scalar value{};
                        for (Index inner = 0; inner < Dim; ++inner)
                            value += u(row, inner) * singular_values(inner) * u(col, inner);
                        (*scaling_matrix)(row, col) = value;
                    }
            }
        }

        const Scalar* data() const noexcept
        {
            return _matrix.data();
        }
        Scalar* data() noexcept
        {
            return _matrix.data();
        }

        void makeAffine()
        {
            if constexpr (Mode != AffineCompact)
            {
                for (Index col = 0; col < Dim; ++col)
                    _matrix(Dim, col) = Scalar{};
                _matrix(Dim, Dim) = Scalar{1};
            }
        }

        Matrix<Scalar, Dim + 1, Dim + 1> homogeneousMatrix() const
        {
            Matrix<Scalar, Dim + 1, Dim + 1> result = Matrix<Scalar, Dim + 1, Dim + 1>::Identity();
            for (Index row = 0; row < Rows; ++row)
                for (Index col = 0; col < Dim + 1; ++col)
                    result(row, col) = _matrix(row, col);
            return result;
        }

        template <typename Derived> auto operator*(const MatrixBase<Derived>& points) const
        {
            if (points.rows() != Dim)
                throw std::invalid_argument("Transform requires points with Dim rows");
            constexpr int ResultCols = detail::DenseTraits<Derived>::ColsAtCompileTime;
            Matrix<Scalar, Dim, ResultCols> result(Dim, points.cols());
            for (Index col = 0; col < points.cols(); ++col)
            {
                Scalar weight = Scalar{1};
                if constexpr (Mode == Projective)
                {
                    weight = _matrix(Dim, Dim);
                    for (Index inner = 0; inner < Dim; ++inner)
                        weight += _matrix(Dim, inner) * points.derived()(inner, col);
                    if (weight == Scalar{})
                        throw std::runtime_error("projective Transform produced zero homogeneous weight");
                }
                for (Index row = 0; row < Dim; ++row)
                {
                    Scalar value = _matrix(row, Dim);
                    for (Index inner = 0; inner < Dim; ++inner)
                        value += _matrix(row, inner) * points.derived()(inner, col);
                    result(row, col) = value / weight;
                }
            }
            return result;
        }

        template <int OtherMode, int OtherOptions>
        auto operator*(const Transform<Scalar, Dim, OtherMode, OtherOptions>& other) const
        {
            constexpr int ResultMode = Mode == Projective || OtherMode == Projective
                                           ? Projective
                                           : (Mode == Isometry && OtherMode == Isometry ? Isometry : Affine);
            Transform<Scalar, Dim, ResultMode> result;
            const auto left = homogeneousMatrix();
            const auto right = other.homogeneousMatrix();
            Matrix<Scalar, Dim + 1, Dim + 1> product;
            for (Index row = 0; row < Dim + 1; ++row)
                for (Index col = 0; col < Dim + 1; ++col)
                {
                    Scalar value{};
                    for (Index inner = 0; inner < Dim + 1; ++inner)
                        value += left(row, inner) * right(inner, col);
                    product(row, col) = value;
                }
            result.assignFromHomogeneous(product);
            return result;
        }

        Transform operator*(const TranslationType& other) const
        {
            Transform result(*this);
            result.translate(other.vector());
            return result;
        }
        Transform operator*(const UniformScaling<Scalar>& other) const
        {
            Transform result(*this);
            result.scale(other.factor());
            return result;
        }
        template <typename RotationDerived>
        Transform operator*(const RotationBase<RotationDerived, Dim>& rotation) const
        {
            Transform result(*this);
            result.rotate(rotation.derived());
            return result;
        }

        template <typename Derived> Transform& translate(const MatrixBase<Derived>& value)
        {
            if (value.rows() != Dim || value.cols() != 1)
                throw std::invalid_argument("translate() requires a Dim-vector");
            VectorType delta;
            for (Index row = 0; row < Dim; ++row)
            {
                Scalar transformed{};
                for (Index inner = 0; inner < Dim; ++inner)
                    transformed += _matrix(row, inner) * value.derived()(inner, 0);
                delta(row) = transformed;
            }
            for (Index row = 0; row < Dim; ++row)
                _matrix(row, Dim) += delta(row);
            return *this;
        }
        template <typename Derived> Transform& pretranslate(const MatrixBase<Derived>& value)
        {
            if (value.rows() != Dim || value.cols() != 1)
                throw std::invalid_argument("pretranslate() requires a Dim-vector");
            for (Index row = 0; row < Dim; ++row)
                _matrix(row, Dim) += value.derived()(row, 0);
            return *this;
        }
        Transform& scale(const Scalar& factor)
        {
            for (Index row = 0; row < Dim; ++row)
                for (Index col = 0; col < Dim; ++col)
                    _matrix(row, col) *= factor;
            return *this;
        }
        Transform& prescale(const Scalar& factor)
        {
            for (Index row = 0; row < Dim; ++row)
                for (Index col = 0; col < Dim + 1; ++col)
                    _matrix(row, col) *= factor;
            return *this;
        }
        template <typename Derived> Transform& scale(const MatrixBase<Derived>& factors)
        {
            if (factors.rows() * factors.cols() != Dim)
                throw std::invalid_argument("scale() requires Dim coefficients");
            for (Index col = 0; col < Dim; ++col)
                for (Index row = 0; row < Dim; ++row)
                    _matrix(row, col) *= factors.derived()(col);
            return *this;
        }
        template <typename Derived> Transform& prescale(const MatrixBase<Derived>& factors)
        {
            if (factors.rows() * factors.cols() != Dim)
                throw std::invalid_argument("prescale() requires Dim coefficients");
            for (Index row = 0; row < Dim; ++row)
                for (Index col = 0; col < Dim + 1; ++col)
                    _matrix(row, col) *= factors.derived()(row);
            return *this;
        }
        template <typename RotationType> Transform& rotate(const RotationType& rotation_value)
        {
            const LinearMatrixType rotation_matrix = rotation_value.toRotationMatrix();
            LinearMatrixType product;
            for (Index row = 0; row < Dim; ++row)
                for (Index col = 0; col < Dim; ++col)
                {
                    Scalar value{};
                    for (Index inner = 0; inner < Dim; ++inner)
                        value += _matrix(row, inner) * rotation_matrix(inner, col);
                    product(row, col) = value;
                }
            assignLinear(product);
            return *this;
        }
        template <typename RotationType> Transform& prerotate(const RotationType& rotation_value)
        {
            const LinearMatrixType rotation_matrix = rotation_value.toRotationMatrix();
            Matrix<Scalar, Dim, Dim + 1> product;
            for (Index row = 0; row < Dim; ++row)
                for (Index col = 0; col < Dim + 1; ++col)
                {
                    Scalar value{};
                    for (Index inner = 0; inner < Dim; ++inner)
                        value += rotation_matrix(row, inner) * _matrix(inner, col);
                    product(row, col) = value;
                }
            for (Index row = 0; row < Dim; ++row)
                for (Index col = 0; col < Dim + 1; ++col)
                    _matrix(row, col) = product(row, col);
            return *this;
        }
        Transform& shear(const Scalar& x, const Scalar& y)
        {
            static_assert(Dim == 2, "shear() is defined for 2D transforms");
            LinearMatrixType shear_matrix;
            shear_matrix << Scalar{1}, x, y, Scalar{1};
            return rotateMatrixRight(shear_matrix);
        }
        Transform& preshear(const Scalar& x, const Scalar& y)
        {
            static_assert(Dim == 2, "preshear() is defined for 2D transforms");
            LinearMatrixType shear_matrix;
            shear_matrix << Scalar{1}, x, y, Scalar{1};
            return premultiplyLinear(shear_matrix);
        }

        Transform inverse(TransformTraits traits = static_cast<TransformTraits>(Mode)) const
        {
            Transform result;
            const auto source = homogeneousMatrix();
            Matrix<Scalar, Dim + 1, Dim + 1> inverse_matrix;
            if (traits == Isometry)
            {
                inverse_matrix = Matrix<Scalar, Dim + 1, Dim + 1>::Identity();
                for (Index row = 0; row < Dim; ++row)
                    for (Index col = 0; col < Dim; ++col)
                        inverse_matrix(row, col) = source(col, row);
                for (Index row = 0; row < Dim; ++row)
                {
                    Scalar value{};
                    for (Index inner = 0; inner < Dim; ++inner)
                        value -= source(inner, row) * source(inner, Dim);
                    inverse_matrix(row, Dim) = value;
                }
            }
            else
            {
                inverse_matrix = source.fullPivLu().solve(Matrix<Scalar, Dim + 1, Dim + 1>::Identity());
            }
            result.assignFromHomogeneous(inverse_matrix);
            return result;
        }

        template <typename NewScalar> Transform<NewScalar, Dim, Mode, Options> cast() const
        {
            return Transform<NewScalar, Dim, Mode, Options>(_matrix.template cast<NewScalar>());
        }
        bool isApprox(const Transform& other,
                      Scalar precision = Scalar{100} * std::numeric_limits<Scalar>::epsilon()) const
        {
            return _matrix.isApprox(other._matrix, precision);
        }

        Transform& operator=(const TranslationType& value)
        {
            setIdentity();
            assignTranslation(value.vector());
            return *this;
        }
        Transform& operator*=(const TranslationType& value)
        {
            return translate(value.vector());
        }
        Transform& operator=(const UniformScaling<Scalar>& value)
        {
            setIdentity();
            return scale(value.factor());
        }
        Transform& operator*=(const UniformScaling<Scalar>& value)
        {
            return scale(value.factor());
        }
        template <typename RotationDerived> Transform& operator=(const RotationBase<RotationDerived, Dim>& rotation)
        {
            setIdentity();
            assignLinear(rotation.derived().toRotationMatrix());
            return *this;
        }
        template <typename RotationDerived> Transform& operator*=(const RotationBase<RotationDerived, Dim>& rotation)
        {
            return rotate(rotation.derived());
        }

        template <typename, int, int, int> friend class Transform;

    private:
        template <typename MatrixDerived> void assignLinear(const MatrixDerived& value)
        {
            for (Index row = 0; row < Dim; ++row)
                for (Index col = 0; col < Dim; ++col)
                    _matrix(row, col) = value(row, col);
        }
        template <typename VectorDerived> void assignTranslation(const VectorDerived& value)
        {
            for (Index row = 0; row < Dim; ++row)
                _matrix(row, Dim) = value(row);
        }
        template <typename MatrixDerived> void assignFromHomogeneous(const MatrixDerived& value)
        {
            for (Index row = 0; row < Rows; ++row)
                for (Index col = 0; col < Dim + 1; ++col)
                    _matrix(row, col) = value(row, col);
        }
        Transform& rotateMatrixRight(const LinearMatrixType& value)
        {
            LinearMatrixType product;
            for (Index row = 0; row < Dim; ++row)
                for (Index col = 0; col < Dim; ++col)
                {
                    Scalar sum{};
                    for (Index inner = 0; inner < Dim; ++inner)
                        sum += _matrix(row, inner) * value(inner, col);
                    product(row, col) = sum;
                }
            assignLinear(product);
            return *this;
        }
        Transform& premultiplyLinear(const LinearMatrixType& value)
        {
            Matrix<Scalar, Dim, Dim + 1> product;
            for (Index row = 0; row < Dim; ++row)
                for (Index col = 0; col < Dim + 1; ++col)
                {
                    Scalar sum{};
                    for (Index inner = 0; inner < Dim; ++inner)
                        sum += value(row, inner) * _matrix(inner, col);
                    product(row, col) = sum;
                }
            for (Index row = 0; row < Dim; ++row)
                for (Index col = 0; col < Dim + 1; ++col)
                    _matrix(row, col) = product(row, col);
            return *this;
        }

        MatrixType _matrix{};
    };

    template <typename Scalar, int Dim>
    typename Translation<Scalar, Dim>::AffineTransformType
    Translation<Scalar, Dim>::operator*(const UniformScaling<Scalar>& scaling) const
    {
        AffineTransformType result(scaling);
        result.pretranslate(_coeffs);
        return result;
    }

    template <typename Scalar, int Dim>
    template <typename RotationDerived>
    typename Translation<Scalar, Dim>::IsometryTransformType
    Translation<Scalar, Dim>::operator*(const RotationBase<RotationDerived, Dim>& rotation) const
    {
        IsometryTransformType result(rotation);
        result.pretranslate(_coeffs);
        return result;
    }

    template <typename RotationDerived, typename Scalar, int Dim>
    Transform<Scalar, Dim, Isometry> operator*(const RotationBase<RotationDerived, Dim>& rotation,
                                               const Translation<Scalar, Dim>& translation)
    {
        Transform<Scalar, Dim, Isometry> result(rotation);
        result.translate(translation.vector());
        return result;
    }

    using Translation2f = Translation<float, 2>;
    using Translation2d = Translation<double, 2>;
    using Translation3f = Translation<float, 3>;
    using Translation3d = Translation<double, 3>;
    using Isometry2f = Transform<float, 2, Isometry>;
    using Isometry2d = Transform<double, 2, Isometry>;
    using Isometry3f = Transform<float, 3, Isometry>;
    using Isometry3d = Transform<double, 3, Isometry>;
    using Affine2f = Transform<float, 2, Affine>;
    using Affine2d = Transform<double, 2, Affine>;
    using Affine3f = Transform<float, 3, Affine>;
    using Affine3d = Transform<double, 3, Affine>;
    using AffineCompact2f = Transform<float, 2, AffineCompact>;
    using AffineCompact2d = Transform<double, 2, AffineCompact>;
    using AffineCompact3f = Transform<float, 3, AffineCompact>;
    using AffineCompact3d = Transform<double, 3, AffineCompact>;
    using Projective2f = Transform<float, 2, Projective>;
    using Projective2d = Transform<double, 2, Projective>;
    using Projective3f = Transform<float, 3, Projective>;
    using Projective3d = Transform<double, 3, Projective>;
} // namespace plamatrix::v1
