#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace plamatrix::v1
{
    namespace reduction_detail
    {
        template <typename Expression, typename Function>
        void forEachCoefficient(const Expression& expression, Function&& function)
        {
            for (Index col = 0; col < expression.cols(); ++col)
            {
                for (Index row = 0; row < expression.rows(); ++row)
                {
                    function(expression(row, col));
                }
            }
        }

        template <bool Minimum, typename Derived, typename IndexType>
        typename DenseBase<Derived>::Scalar
        extremumWithIndex(const Derived& expression, IndexType* row_result, IndexType* col_result)
        {
            using Base = DenseBase<Derived>;
            using Scalar = typename Base::Scalar;
            if (expression.size() == 0)
            {
                throw std::invalid_argument(Minimum ? "minCoeff() requires a nonempty expression"
                                                    : "maxCoeff() requires a nonempty expression");
            }
            Index best_row = 0;
            Index best_col = 0;
            Scalar best = expression(0, 0);
            for (Index linear = 1; linear < expression.size(); ++linear)
            {
                const Index row = Base::IsRowMajor ? linear / expression.cols() : linear % expression.rows();
                const Index col = Base::IsRowMajor ? linear % expression.cols() : linear / expression.rows();
                const Scalar value = expression(row, col);
                if ((Minimum && value < best) || (!Minimum && value > best))
                {
                    best = value;
                    best_row = row;
                    best_col = col;
                }
            }
            *row_result = static_cast<IndexType>(best_row);
            *col_result = static_cast<IndexType>(best_col);
            return best;
        }

        template <typename Scalar> auto absoluteValue(const Scalar& value)
        {
            using std::abs;
            return abs(value);
        }
    } // namespace reduction_detail

    template <typename Derived> typename DenseBase<Derived>::Scalar DenseBase<Derived>::sum() const
    {
        Scalar result{};
        reduction_detail::forEachCoefficient(this->derived(), [&](const Scalar& value) { result += value; });
        return result;
    }

    template <typename Derived> typename DenseBase<Derived>::Scalar DenseBase<Derived>::mean() const
    {
        return sum() / static_cast<Scalar>(this->size());
    }

    template <typename Derived> typename DenseBase<Derived>::Scalar DenseBase<Derived>::prod() const
    {
        Scalar result{1};
        reduction_detail::forEachCoefficient(this->derived(), [&](const Scalar& value) { result *= value; });
        return result;
    }

    template <typename Derived> typename DenseBase<Derived>::Scalar DenseBase<Derived>::minCoeff() const
    {
        Index row = 0;
        Index col = 0;
        return reduction_detail::extremumWithIndex<true>(this->derived(), &row, &col);
    }

    template <typename Derived> typename DenseBase<Derived>::Scalar DenseBase<Derived>::maxCoeff() const
    {
        Index row = 0;
        Index col = 0;
        return reduction_detail::extremumWithIndex<false>(this->derived(), &row, &col);
    }

    template <typename Derived>
    template <typename IndexType>
    typename DenseBase<Derived>::Scalar DenseBase<Derived>::minCoeff(IndexType* row, IndexType* col) const
    {
        return reduction_detail::extremumWithIndex<true>(this->derived(), row, col);
    }

    template <typename Derived>
    template <typename IndexType>
    typename DenseBase<Derived>::Scalar DenseBase<Derived>::maxCoeff(IndexType* row, IndexType* col) const
    {
        return reduction_detail::extremumWithIndex<false>(this->derived(), row, col);
    }

    template <typename Derived>
    template <typename IndexType>
    typename DenseBase<Derived>::Scalar DenseBase<Derived>::minCoeff(IndexType* index) const
    {
        static_assert(IsVectorAtCompileTime, "Single-index minCoeff() requires a vector expression");
        IndexType row = 0;
        IndexType col = 0;
        const Scalar result = minCoeff(&row, &col);
        *index = RowsAtCompileTime == 1 ? col : row;
        return result;
    }

    template <typename Derived>
    template <typename IndexType>
    typename DenseBase<Derived>::Scalar DenseBase<Derived>::maxCoeff(IndexType* index) const
    {
        static_assert(IsVectorAtCompileTime, "Single-index maxCoeff() requires a vector expression");
        IndexType row = 0;
        IndexType col = 0;
        const Scalar result = maxCoeff(&row, &col);
        *index = RowsAtCompileTime == 1 ? col : row;
        return result;
    }

    template <typename Derived> bool DenseBase<Derived>::all() const
    {
        bool result = true;
        reduction_detail::forEachCoefficient(this->derived(),
                                             [&](const Scalar& value) { result = result && static_cast<bool>(value); });
        return result;
    }

    template <typename Derived> bool DenseBase<Derived>::any() const
    {
        bool result = false;
        reduction_detail::forEachCoefficient(this->derived(),
                                             [&](const Scalar& value) { result = result || static_cast<bool>(value); });
        return result;
    }

    template <typename Derived> typename DenseBase<Derived>::Index DenseBase<Derived>::count() const
    {
        Index result = 0;
        reduction_detail::forEachCoefficient(this->derived(),
                                             [&](const Scalar& value) { result += static_cast<bool>(value) ? 1 : 0; });
        return result;
    }

    template <typename Derived> bool DenseBase<Derived>::allFinite() const
    {
        if constexpr (NumTraits<Scalar>::IsComplex)
        {
            bool result = true;
            reduction_detail::forEachCoefficient(this->derived(),
                                                 [&](const Scalar& value) {
                                                     result = result && std::isfinite(value.real()) &&
                                                              std::isfinite(value.imag());
                                                 });
            return result;
        }
        else if constexpr (!std::is_floating_point_v<Scalar>)
        {
            return true;
        }
        else
        {
            bool result = true;
            reduction_detail::forEachCoefficient(this->derived(),
                                                 [&](const Scalar& value) { result = result && std::isfinite(value); });
            return result;
        }
    }

} // namespace plamatrix::v1

#include "plamatrix/dense/detail/matrix_norm_fuzzy_methods.h"
