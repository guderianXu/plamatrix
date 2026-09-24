#pragma once

namespace plamatrix::v1
{
    template <typename Derived>
    template <typename OtherDerived>
    bool DenseBase<Derived>::isApprox(const DenseBase<OtherDerived>& other, const RealScalar& precision) const
    {
        using OtherScalar = typename detail::DenseTraits<OtherDerived>::ScalarType;
        static_assert(std::is_same_v<Scalar, OtherScalar>, "isApprox() scalar types must match");
        if (this->rows() != other.rows() || this->cols() != other.cols() || precision < RealScalar{})
        {
            return false;
        }
        if constexpr (std::is_integral_v<Scalar>)
        {
            for (Index col = 0; col < this->cols(); ++col)
            {
                for (Index row = 0; row < this->rows(); ++row)
                {
                    if (this->derived()(row, col) != other.derived()(row, col))
                    {
                        return false;
                    }
                }
            }
            return true;
        }
        long double difference_squared = 0.0L;
        long double left_squared = 0.0L;
        long double right_squared = 0.0L;
        for (Index col = 0; col < this->cols(); ++col)
        {
            for (Index row = 0; row < this->rows(); ++row)
            {
                const Scalar left = this->derived()(row, col);
                const Scalar right = other.derived()(row, col);
                const long double difference = static_cast<long double>(reduction_detail::absoluteValue(left - right));
                const long double left_absolute = static_cast<long double>(reduction_detail::absoluteValue(left));
                const long double right_absolute = static_cast<long double>(reduction_detail::absoluteValue(right));
                difference_squared += difference * difference;
                left_squared += left_absolute * left_absolute;
                right_squared += right_absolute * right_absolute;
            }
        }
        const long double tolerance = static_cast<long double>(precision);
        return difference_squared <= tolerance * tolerance * std::min(left_squared, right_squared);
    }

    template <typename Derived>
    bool DenseBase<Derived>::isMuchSmallerThan(const RealScalar& other, const RealScalar& precision) const
    {
        if constexpr (std::is_integral_v<Scalar>)
        {
            return this->derived().squaredNorm() == Scalar{};
        }
        long double squared_norm = 0.0L;
        reduction_detail::forEachCoefficient(this->derived(),
                                             [&](const Scalar& value)
                                             {
                                                 const long double coefficient =
                                                     static_cast<long double>(reduction_detail::absoluteValue(value));
                                                 squared_norm += coefficient * coefficient;
                                             });
        const long double threshold = static_cast<long double>(precision) * static_cast<long double>(other);
        return squared_norm <= threshold * threshold;
    }

    template <typename Derived>
    template <typename OtherDerived>
    bool DenseBase<Derived>::isMuchSmallerThan(const DenseBase<OtherDerived>& other, const RealScalar& precision) const
    {
        using OtherScalar = typename detail::DenseTraits<OtherDerived>::ScalarType;
        static_assert(std::is_same_v<Scalar, OtherScalar>, "isMuchSmallerThan() scalar types must match");
        if (this->rows() != other.rows() || this->cols() != other.cols() || precision < RealScalar{})
        {
            return false;
        }
        if constexpr (std::is_integral_v<Scalar>)
        {
            return this->derived().squaredNorm() == Scalar{};
        }
        long double left_squared = 0.0L;
        long double right_squared = 0.0L;
        for (Index col = 0; col < this->cols(); ++col)
        {
            for (Index row = 0; row < this->rows(); ++row)
            {
                const long double left =
                    static_cast<long double>(reduction_detail::absoluteValue(this->derived()(row, col)));
                const long double right =
                    static_cast<long double>(reduction_detail::absoluteValue(other.derived()(row, col)));
                left_squared += left * left;
                right_squared += right * right;
            }
        }
        const long double tolerance = static_cast<long double>(precision);
        return left_squared <= tolerance * tolerance * right_squared;
    }

    template <typename Derived>
    template <int p>
    typename MatrixBase<Derived>::RealScalar MatrixBase<Derived>::lpNorm() const
    {
        static_assert(p > 0 || p == Infinity, "lpNorm<p>() requires p > 0 or Infinity");
        if (this->size() == 0)
        {
            return RealScalar{};
        }
        if constexpr (p == Infinity)
        {
            RealScalar result{};
            reduction_detail::forEachCoefficient(
                this->derived(),
                [&](const Scalar& value)
                { result = std::max(result, static_cast<RealScalar>(reduction_detail::absoluteValue(value))); });
            return result;
        }
        else if constexpr (p == 2)
        {
            return this->derived().norm();
        }
        else
        {
            long double sum_of_powers = 0.0L;
            reduction_detail::forEachCoefficient(
                this->derived(),
                [&](const Scalar& value)
                { sum_of_powers += std::pow(static_cast<long double>(reduction_detail::absoluteValue(value)), p); });
            return static_cast<RealScalar>(std::pow(sum_of_powers, 1.0L / p));
        }
    }

    template <typename Derived> typename MatrixBase<Derived>::RealScalar MatrixBase<Derived>::squaredNorm() const
    {
        long double result = 0.0L;
        reduction_detail::forEachCoefficient(this->derived(),
                                             [&](const Scalar& value)
                                             {
                                                 const long double coefficient =
                                                     static_cast<long double>(reduction_detail::absoluteValue(value));
                                                 result += coefficient * coefficient;
                                             });
        return static_cast<RealScalar>(result);
    }

    template <typename Derived> typename MatrixBase<Derived>::RealScalar MatrixBase<Derived>::norm() const
    {
        return static_cast<RealScalar>(std::sqrt(static_cast<long double>(squaredNorm())));
    }

    template <typename Derived> typename MatrixBase<Derived>::Scalar MatrixBase<Derived>::trace() const
    {
        Scalar result{};
        for (Index index = 0; index < std::min(this->rows(), this->cols()); ++index)
        {
            result += this->derived()(index, index);
        }
        return result;
    }

    template <typename Derived>
    template <typename OtherDerived>
    typename ScalarBinaryOpTraits<typename MatrixBase<Derived>::Scalar,
                                  typename detail::DenseTraits<OtherDerived>::ScalarType>::ReturnType
    MatrixBase<Derived>::dot(const MatrixBase<OtherDerived>& other) const
    {
        using OtherScalar = typename detail::DenseTraits<OtherDerived>::ScalarType;
        using ReturnScalar = typename ScalarBinaryOpTraits<Scalar, OtherScalar>::ReturnType;
        if ((this->rows() != 1 && this->cols() != 1) || (other.rows() != 1 && other.cols() != 1) ||
            this->size() != other.size())
        {
            throw std::invalid_argument("dot() requires vectors of equal length");
        }
        ReturnScalar result{};
        for (Index index = 0; index < this->size(); ++index)
        {
            const ReturnScalar left =
                static_cast<ReturnScalar>(this->cols() == 1 ? this->derived()(index, 0) : this->derived()(0, index));
            const ReturnScalar right =
                static_cast<ReturnScalar>(other.cols() == 1 ? other.derived()(index, 0) : other.derived()(0, index));
            if constexpr (NumTraits<Scalar>::IsComplex)
            {
                result += std::conj(left) * right;
            }
            else
            {
                result += left * right;
            }
        }
        return result;
    }

    template <typename Derived> typename MatrixBase<Derived>::RealScalar MatrixBase<Derived>::stableNorm() const
    {
        RealScalar scale{};
        reduction_detail::forEachCoefficient(
            this->derived(),
            [&](const Scalar& value)
            { scale = std::max(scale, static_cast<RealScalar>(reduction_detail::absoluteValue(value))); });
        if (scale == RealScalar{} || !std::isfinite(static_cast<long double>(scale)))
        {
            return scale;
        }
        long double sum = 0.0L;
        reduction_detail::forEachCoefficient(this->derived(),
                                             [&](const Scalar& value)
                                             {
                                                 const long double scaled = static_cast<long double>(value) / scale;
                                                 sum += scaled * scaled;
                                             });
        return static_cast<RealScalar>(static_cast<long double>(scale) * std::sqrt(sum));
    }

    template <typename Derived> typename MatrixBase<Derived>::RealScalar MatrixBase<Derived>::blueNorm() const
    {
        RealScalar scale{};
        long double sum_of_squares = 1.0L;
        bool has_nonzero = false;
        bool has_infinity = false;
        bool has_nan = false;
        reduction_detail::forEachCoefficient(
            this->derived(),
            [&](const Scalar& value)
            {
                const RealScalar absolute = static_cast<RealScalar>(reduction_detail::absoluteValue(value));
                if (std::isnan(static_cast<long double>(absolute)))
                {
                    has_nan = true;
                    return;
                }
                if (std::isinf(static_cast<long double>(absolute)))
                {
                    has_infinity = true;
                    return;
                }
                if (absolute == RealScalar{})
                {
                    return;
                }
                has_nonzero = true;
                if (scale < absolute)
                {
                    const long double ratio = static_cast<long double>(scale) / absolute;
                    sum_of_squares = 1.0L + sum_of_squares * ratio * ratio;
                    scale = absolute;
                }
                else
                {
                    const long double ratio = static_cast<long double>(absolute) / scale;
                    sum_of_squares += ratio * ratio;
                }
            });
        if (has_nan)
        {
            return std::numeric_limits<RealScalar>::quiet_NaN();
        }
        if (has_infinity)
        {
            return std::numeric_limits<RealScalar>::infinity();
        }
        if (!has_nonzero)
        {
            return scale;
        }
        return static_cast<RealScalar>(static_cast<long double>(scale) * std::sqrt(sum_of_squares));
    }

    template <typename Derived> typename MatrixBase<Derived>::RealScalar MatrixBase<Derived>::hypotNorm() const
    {
        RealScalar result{};
        reduction_detail::forEachCoefficient(
            this->derived(),
            [&](const Scalar& value)
            { result = std::hypot(result, static_cast<RealScalar>(reduction_detail::absoluteValue(value))); });
        return result;
    }
} // namespace plamatrix::v1
