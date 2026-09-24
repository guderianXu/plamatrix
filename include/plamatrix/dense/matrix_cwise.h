#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "plamatrix/dense/matrix_base.h"

namespace plamatrix::internal
{
    namespace cwise_detail
    {
        template <typename Scalar> bool isNan(const Scalar& value)
        {
            if constexpr (std::is_floating_point_v<Scalar>)
            {
                return std::isnan(value);
            }
            return false;
        }

        template <int NaNPropagation, bool Minimum, typename LeftScalar, typename RightScalar>
        auto extremum(const LeftScalar& left, const RightScalar& right)
        {
            using Result = std::common_type_t<LeftScalar, RightScalar>;
            const Result converted_left = static_cast<Result>(left);
            const Result converted_right = static_cast<Result>(right);
            if constexpr (NaNPropagation == ::plamatrix::v1::PropagateNaN)
            {
                if (isNan(converted_left))
                {
                    return converted_left;
                }
                if (isNan(converted_right))
                {
                    return converted_right;
                }
            }
            else if constexpr (NaNPropagation == ::plamatrix::v1::PropagateNumbers)
            {
                if (isNan(converted_left))
                {
                    return converted_right;
                }
                if (isNan(converted_right))
                {
                    return converted_left;
                }
            }
            if constexpr (Minimum)
            {
                return std::min(converted_left, converted_right);
            }
            return std::max(converted_left, converted_right);
        }
    } // namespace cwise_detail

    template <typename Scalar> struct scalar_abs_op
    {
        auto operator()(const Scalar& value) const
        {
            using std::abs;
            return abs(value);
        }
    };

    template <typename Scalar> struct scalar_conjugate_op
    {
        Scalar operator()(const Scalar& value) const
        {
            if constexpr (::plamatrix::v1::NumTraits<Scalar>::IsComplex)
            {
                return std::conj(value);
            }
            else
            {
                return value;
            }
        }
    };

    template <typename LeftScalar, typename RightScalar, int NaNPropagation> struct scalar_min_op
    {
        auto operator()(const LeftScalar& left, const RightScalar& right) const
        {
            return cwise_detail::extremum<NaNPropagation, true>(left, right);
        }
    };

    template <typename LeftScalar, typename RightScalar, int NaNPropagation> struct scalar_max_op
    {
        auto operator()(const LeftScalar& left, const RightScalar& right) const
        {
            return cwise_detail::extremum<NaNPropagation, false>(left, right);
        }
    };

    template <typename Scalar, int NaNPropagation> struct scalar_min_constant_op
    {
        explicit scalar_min_constant_op(const Scalar& value) : reference(value)
        {
        }

        Scalar operator()(const Scalar& value) const
        {
            return cwise_detail::extremum<NaNPropagation, true>(value, reference);
        }

        Scalar reference;
    };

    template <typename Scalar, int NaNPropagation> struct scalar_max_constant_op
    {
        explicit scalar_max_constant_op(const Scalar& value) : reference(value)
        {
        }

        Scalar operator()(const Scalar& value) const
        {
            return cwise_detail::extremum<NaNPropagation, false>(value, reference);
        }

        Scalar reference;
    };
} // namespace plamatrix::internal

namespace plamatrix::v1
{
    namespace cwise_detail
    {
        template <typename Type> struct IsPlainMatrix : std::false_type
        {
        };

        template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
        struct IsPlainMatrix<Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>> : std::true_type
        {
        };

        template <typename XprType> class NestedExpressionStorage
        {
            using RawExpression = std::remove_const_t<XprType>;
            static constexpr bool stores_reference = IsPlainMatrix<RawExpression>::value;
            using Storage = std::conditional_t<stores_reference, const RawExpression*, RawExpression>;

        public:
            explicit NestedExpressionStorage(const RawExpression& expression) : _expression(makeStorage(expression))
            {
            }

            const RawExpression& get() const noexcept
            {
                if constexpr (stores_reference)
                {
                    return *_expression;
                }
                else
                {
                    return _expression;
                }
            }

        private:
            static Storage makeStorage(const RawExpression& expression)
            {
                if constexpr (stores_reference)
                {
                    return &expression;
                }
                else
                {
                    return expression;
                }
            }

            Storage _expression;
        };
    } // namespace cwise_detail

    template <typename UnaryOp, typename XprType> class CwiseUnaryOp : public MatrixBase<CwiseUnaryOp<UnaryOp, XprType>>
    {
        using Traits = detail::DenseTraits<CwiseUnaryOp>;
        using RawExpression = std::remove_const_t<XprType>;

    public:
        using Scalar = typename Traits::ScalarType;
        using RealScalar = typename NumTraits<Scalar>::Real;
        using PlainObject = typename Traits::PlainObject;
        using NestedExpression = RawExpression;
        using Index = plamatrix::Index;
        static constexpr int RowsAtCompileTime = Traits::RowsAtCompileTime;
        static constexpr int ColsAtCompileTime = Traits::ColsAtCompileTime;
        static constexpr int MaxRowsAtCompileTime = Traits::MaxRowsAtCompileTime;
        static constexpr int MaxColsAtCompileTime = Traits::MaxColsAtCompileTime;
        static constexpr int IsRowMajor = (Traits::StorageOptions & RowMajor) == RowMajor;
        static constexpr int IsVectorAtCompileTime = RowsAtCompileTime == 1 || ColsAtCompileTime == 1;
        static constexpr int SizeAtCompileTime = RowsAtCompileTime == Dynamic || ColsAtCompileTime == Dynamic
                                                     ? Dynamic
                                                     : RowsAtCompileTime * ColsAtCompileTime;

        explicit CwiseUnaryOp(const RawExpression& expression, const UnaryOp& function = UnaryOp())
            : _expression(expression), _function(function)
        {
        }

        Index rows() const noexcept
        {
            return nestedExpression().rows();
        }

        Index cols() const noexcept
        {
            return nestedExpression().cols();
        }

        Index size() const noexcept
        {
            return rows() * cols();
        }

        Scalar operator()(Index row, Index col) const
        {
            return _function(nestedExpression()(row, col));
        }

        Scalar operator()(Index index) const
        {
            if (cols() == 1)
            {
                return (*this)(index, 0);
            }
            if (rows() == 1)
            {
                return (*this)(0, index);
            }
            throw std::invalid_argument("Single-index coefficient access requires a vector expression");
        }

        Scalar coeff(Index row, Index col) const
        {
            return (*this)(row, col);
        }

        const UnaryOp& functor() const noexcept
        {
            return _function;
        }

        const RawExpression& nestedExpression() const noexcept
        {
            return _expression.get();
        }

    private:
        cwise_detail::NestedExpressionStorage<XprType> _expression;
        UnaryOp _function;
    };

    template <typename BinaryOp, typename LhsType, typename RhsType>
    class CwiseBinaryOp : public MatrixBase<CwiseBinaryOp<BinaryOp, LhsType, RhsType>>
    {
        using Traits = detail::DenseTraits<CwiseBinaryOp>;
        using RawLeft = std::remove_const_t<LhsType>;
        using RawRight = std::remove_const_t<RhsType>;
        using LeftTraits = detail::DenseTraits<RawLeft>;
        using RightTraits = detail::DenseTraits<RawRight>;

        static_assert(LeftTraits::RowsAtCompileTime == Dynamic || RightTraits::RowsAtCompileTime == Dynamic ||
                          LeftTraits::RowsAtCompileTime == RightTraits::RowsAtCompileTime,
                      "binaryExpr() requires matching row dimensions");
        static_assert(LeftTraits::ColsAtCompileTime == Dynamic || RightTraits::ColsAtCompileTime == Dynamic ||
                          LeftTraits::ColsAtCompileTime == RightTraits::ColsAtCompileTime,
                      "binaryExpr() requires matching column dimensions");

    public:
        using Scalar = typename Traits::ScalarType;
        using RealScalar = typename NumTraits<Scalar>::Real;
        using PlainObject = typename Traits::PlainObject;
        using Lhs = RawLeft;
        using Rhs = RawRight;
        using Index = plamatrix::Index;
        static constexpr int RowsAtCompileTime = Traits::RowsAtCompileTime;
        static constexpr int ColsAtCompileTime = Traits::ColsAtCompileTime;
        static constexpr int MaxRowsAtCompileTime = Traits::MaxRowsAtCompileTime;
        static constexpr int MaxColsAtCompileTime = Traits::MaxColsAtCompileTime;
        static constexpr int IsRowMajor = (Traits::StorageOptions & RowMajor) == RowMajor;
        static constexpr int IsVectorAtCompileTime = RowsAtCompileTime == 1 || ColsAtCompileTime == 1;
        static constexpr int SizeAtCompileTime = RowsAtCompileTime == Dynamic || ColsAtCompileTime == Dynamic
                                                     ? Dynamic
                                                     : RowsAtCompileTime * ColsAtCompileTime;

        CwiseBinaryOp(const RawLeft& left, const RawRight& right, const BinaryOp& function = BinaryOp())
            : _left(left), _right(right), _function(function)
        {
            if (left.rows() != right.rows() || left.cols() != right.cols())
            {
                throw std::invalid_argument("binaryExpr() requires expressions with equal shapes");
            }
        }

        Index rows() const noexcept
        {
            return lhs().rows();
        }

        Index cols() const noexcept
        {
            return lhs().cols();
        }

        Index size() const noexcept
        {
            return rows() * cols();
        }

        Scalar operator()(Index row, Index col) const
        {
            return _function(lhs()(row, col), rhs()(row, col));
        }

        Scalar operator()(Index index) const
        {
            if (cols() == 1)
            {
                return (*this)(index, 0);
            }
            if (rows() == 1)
            {
                return (*this)(0, index);
            }
            throw std::invalid_argument("Single-index coefficient access requires a vector expression");
        }

        Scalar coeff(Index row, Index col) const
        {
            return (*this)(row, col);
        }

        const RawLeft& lhs() const noexcept
        {
            return _left.get();
        }

        const RawRight& rhs() const noexcept
        {
            return _right.get();
        }

        const BinaryOp& functor() const noexcept
        {
            return _function;
        }

    private:
        cwise_detail::NestedExpressionStorage<LhsType> _left;
        cwise_detail::NestedExpressionStorage<RhsType> _right;
        BinaryOp _function;
    };

} // namespace plamatrix::v1

#include "plamatrix/dense/detail/matrix_cwise_methods.h"
