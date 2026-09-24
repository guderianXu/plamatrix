#pragma once

#include "plamatrix/dense/detail/matrix_expression_array_math.h"

namespace plamatrix::v1
{
    namespace expression_detail
    {
        template <typename Storage> class ArrayIdentityNode
        {
            using Source = std::decay_t<decltype(unwrap(std::declval<const Storage&>()))>;
            using SourceTraits = Traits<Source>;

        public:
            using Value = typename SourceTraits::Value;
            static constexpr int RowsAtCompileTime = SourceTraits::rows;
            static constexpr int ColsAtCompileTime = SourceTraits::cols;
            static constexpr bool ArraySemantics = true;
            static constexpr bool LinearAccess = hasLinearAccess<Source>;
            static constexpr bool SimdSafe = LinearAccess && isSimdSafe<Source>;
            static constexpr bool DirectDataAccess = hasDirectDataAccess<Source>;
            static constexpr std::size_t PacketTermCount = packetReductionTermCount<Source>;
            using Result = Matrix<Value, RowsAtCompileTime, ColsAtCompileTime>;

            explicit ArrayIdentityNode(Storage source) : _source(std::move(source))
            {
            }
            Index rows() const
            {
                return value().rows();
            }
            Index cols() const
            {
                return value().cols();
            }
            bool hasResidentInput() const
            {
                return resident(value());
            }
            std::size_t pendingDownloadBytes() const
            {
                return pendingDownload(value());
            }
            bool hasProduct() const
            {
                return expression_detail::hasProduct(value());
            }
            bool supportsGpu() const
            {
                return false;
            }
            long double work() const
            {
                return expression_detail::work(value());
            }
            Value coeff(Index row, Index col) const
            {
                return coefficient(value(), row, col);
            }
            Value linearCoeff(Index index) const
            {
                static_assert(LinearAccess, "linearCoeff requires a contiguous column-major operand");
                return linearCoefficient(value(), index);
            }
            auto bindLinear() const
            {
                static_assert(LinearAccess, "bindLinear requires a contiguous column-major operand");
                return linearBind(value());
            }
            const Value* directData() const
            {
                static_assert(DirectDataAccess, "directData requires a contiguous matrix operand");
                return expression_detail::directData(value());
            }
            auto packetReductionTerms() const
            {
                static_assert(PacketTermCount > 0, "packetReductionTerms requires a packet-reducible array operand");
                return expression_detail::packetReductionTerms(value());
            }
            auto bind() const
            {
                return expression_detail::bind(value());
            }
            Result eager() const
            {
                Result result(rows(), cols());
                assignNoAlias(result);
                return result;
            }
            void assignNoAlias(Result& output) const
            {
                Value* destination = output.data();
                if constexpr (LinearAccess)
                {
                    const auto evaluator = bindLinear();
                    if constexpr (SimdSafe && std::is_floating_point_v<Value>)
                    {
#pragma omp simd
                        for (Index index = 0; index < rows() * cols(); ++index)
                            destination[index] = evaluator(index);
                    }
                    else
                    {
                        for (Index index = 0; index < rows() * cols(); ++index)
                            destination[index] = evaluator(index);
                    }
                }
                else
                {
                    const auto evaluator = bind();
                    for (Index col = 0; col < cols(); ++col)
                        for (Index row = 0; row < rows(); ++row)
                            destination[row + col * rows()] = evaluator(row, col);
                }
            }

        private:
            const Source& value() const
            {
                return unwrap(_source);
            }
            Storage _source;
        };

        template <typename Storage, bool Square> class ArrayScalarNode
        {
            using Source = std::decay_t<decltype(unwrap(std::declval<const Storage&>()))>;
            using SourceTraits = Traits<Source>;

        public:
            using Value = typename SourceTraits::Value;
            static constexpr int RowsAtCompileTime = SourceTraits::rows;
            static constexpr int ColsAtCompileTime = SourceTraits::cols;
            static constexpr bool ArraySemantics = true;
            static constexpr bool LinearAccess = hasLinearAccess<Source>;
            static constexpr bool SimdSafe = LinearAccess && isSimdSafe<Source>;
            static constexpr bool PacketScalarCompatible =
                std::is_same_v<Value, float> || std::is_same_v<Value, double>;
            static constexpr std::size_t PacketTermCount =
                Square ? (PacketScalarCompatible && hasDirectDataAccess<Source> ? 1 : 0)
                       : (PacketScalarCompatible && packetReductionTermCount<Source> > 0
                              ? packetReductionTermCount<Source> + 1
                              : 0);
            using Result = Matrix<Value, RowsAtCompileTime, ColsAtCompileTime>;

            ArrayScalarNode(Storage source, Value shift) : _source(std::move(source)), _shift(shift)
            {
            }
            Index rows() const
            {
                return value().rows();
            }
            Index cols() const
            {
                return value().cols();
            }
            bool hasResidentInput() const
            {
                return resident(value());
            }
            std::size_t pendingDownloadBytes() const
            {
                return pendingDownload(value());
            }
            bool hasProduct() const
            {
                return expression_detail::hasProduct(value());
            }
            bool supportsGpu() const
            {
                return false;
            }
            long double work() const
            {
                return expression_detail::work(value()) + static_cast<long double>(rows()) * cols();
            }
            Value coeff(Index row, Index col) const
            {
                const Value item = coefficient(value(), row, col);
                if constexpr (Square)
                    return item * item;
                return item + _shift;
            }
            Value linearCoeff(Index index) const
            {
                static_assert(LinearAccess, "linearCoeff requires a contiguous column-major operand");
                const Value item = linearCoefficient(value(), index);
                if constexpr (Square)
                    return item * item;
                return item + _shift;
            }
            auto bindLinear() const
            {
                static_assert(LinearAccess, "bindLinear requires a contiguous column-major operand");
                const auto evaluator = linearBind(value());
                return [evaluator, shift = _shift](Index index)
                {
                    const Value item = evaluator(index);
                    if constexpr (Square)
                        return item * item;
                    return item + shift;
                };
            }
            auto packetReductionTerms() const
            {
                static_assert(PacketTermCount > 0,
                              "packetReductionTerms requires a packet-reducible array scalar expression");
                using Term = internal::detail::PacketReductionTerm<Value>;
                if constexpr (Square)
                {
                    const Value* source = expression_detail::directData(value());
                    return std::array<Term, 1>{{Term{source, source, Value{1}}}};
                }
                else
                {
                    auto source_terms = expression_detail::packetReductionTerms(value());
                    std::array<Term, 1> shift_term{{Term{nullptr, nullptr, _shift}}};
                    return concatenatePacketReductionTerms(source_terms, shift_term, Value{1});
                }
            }
            auto bind() const
            {
                const auto evaluator = expression_detail::bind(value());
                return [evaluator, shift = _shift](Index row, Index col)
                {
                    const Value item = evaluator(row, col);
                    if constexpr (Square)
                        return item * item;
                    return item + shift;
                };
            }
            Result eager() const
            {
                Result result(rows(), cols());
                assignNoAlias(result);
                return result;
            }
            void assignNoAlias(Result& output) const
            {
                Value* destination = output.data();
                if constexpr (LinearAccess)
                {
                    const auto evaluator = bindLinear();
                    if constexpr (SimdSafe && std::is_floating_point_v<Value>)
                    {
#pragma omp simd
                        for (Index index = 0; index < rows() * cols(); ++index)
                            destination[index] = evaluator(index);
                    }
                    else
                    {
                        for (Index index = 0; index < rows() * cols(); ++index)
                            destination[index] = evaluator(index);
                    }
                }
                else
                {
                    const auto evaluator = bind();
                    for (Index col = 0; col < cols(); ++col)
                        for (Index row = 0; row < rows(); ++row)
                            destination[row + col * rows()] = evaluator(row, col);
                }
            }

        private:
            const Source& value() const
            {
                return unwrap(_source);
            }
            Storage _source;
            Value _shift;
        };
    } // namespace expression_detail

    /// Coefficientwise operations that compose with dense expressions.
    template <typename Storage> class ArrayExpressionProxy : public ArrayBase<ArrayExpressionProxy<Storage>>
    {
        using Source = std::decay_t<decltype(expression_detail::unwrap(std::declval<const Storage&>()))>;

    public:
        using Value = typename expression_detail::Traits<Source>::Value;
        explicit ArrayExpressionProxy(Storage source) : _source(std::move(source))
        {
        }

        Index rows() const
        {
            return source().rows();
        }

        Index cols() const
        {
            return source().cols();
        }

        Index size() const
        {
            return rows() * cols();
        }

        Value operator()(Index row, Index col) const
        {
            return expression_detail::coefficient(source(), row, col);
        }

        Value operator()(Index index) const
        {
            if (cols() == 1)
            {
                return (*this)(index, 0);
            }
            if (rows() == 1)
            {
                return (*this)(0, index);
            }
            throw internal::Error(internal::ErrorCode::InvalidArgument, "Single-index array access requires a vector");
        }

        Value coeff(Index row, Index col) const
        {
            return (*this)(row, col);
        }

        auto matrix() const&
        {
            return wrap(_source);
        }
        auto matrix() &&
        {
            return wrap(std::move(_source));
        }

        template <typename Other> auto operator*(ArrayExpressionProxy<Other> other) const&
        {
            return binary<expression_detail::BinaryKind::CwiseProduct>(std::move(other));
        }
        template <typename Other> auto operator*(ArrayExpressionProxy<Other> other) &&
        {
            return std::move(*this).template binary<expression_detail::BinaryKind::CwiseProduct>(std::move(other));
        }
        template <typename Other> auto operator/(ArrayExpressionProxy<Other> other) const&
        {
            return binary<expression_detail::BinaryKind::CwiseDivide>(std::move(other));
        }
        template <typename Other> auto operator/(ArrayExpressionProxy<Other> other) &&
        {
            return std::move(*this).template binary<expression_detail::BinaryKind::CwiseDivide>(std::move(other));
        }
        template <typename Other> auto operator+(ArrayExpressionProxy<Other> other) const&
        {
            return binary<expression_detail::BinaryKind::Add>(std::move(other));
        }
        template <typename Other> auto operator+(ArrayExpressionProxy<Other> other) &&
        {
            return std::move(*this).template binary<expression_detail::BinaryKind::Add>(std::move(other));
        }
        template <typename Other> auto operator-(ArrayExpressionProxy<Other> other) const&
        {
            return binary<expression_detail::BinaryKind::Subtract>(std::move(other));
        }
        template <typename Other> auto operator-(ArrayExpressionProxy<Other> other) &&
        {
            return std::move(*this).template binary<expression_detail::BinaryKind::Subtract>(std::move(other));
        }
        auto operator*(Value scale) const&
        {
            return scaleExpression(matrix(), scale);
        }
        auto operator*(Value scale) &&
        {
            return scaleExpression(std::move(*this).matrix(), scale);
        }
        auto operator/(Value divisor) const&
        {
            return matrix() / divisor;
        }
        auto operator/(Value divisor) &&
        {
            return std::move(*this).matrix() / divisor;
        }
        auto operator+(Value shift) const&
        {
            return shifted(_source, shift);
        }
        auto operator+(Value shift) &&
        {
            return shifted(std::move(_source), shift);
        }
        auto operator-(Value shift) const&
        {
            return shifted(_source, -shift);
        }
        auto operator-(Value shift) &&
        {
            return shifted(std::move(_source), -shift);
        }
        auto square() const&
        {
            return squared(_source);
        }
        auto square() &&
        {
            return squared(std::move(_source));
        }
        auto abs() const&
        {
            return math<expression_detail::ArrayUnaryKind::Abs>(_source);
        }
        auto abs() &&
        {
            return math<expression_detail::ArrayUnaryKind::Abs>(std::move(_source));
        }
        auto sqrt() const&
        {
            return math<expression_detail::ArrayUnaryKind::Sqrt>(_source);
        }
        auto sqrt() &&
        {
            return math<expression_detail::ArrayUnaryKind::Sqrt>(std::move(_source));
        }
        auto exp() const&
        {
            return math<expression_detail::ArrayUnaryKind::Exp>(_source);
        }
        auto exp() &&
        {
            return math<expression_detail::ArrayUnaryKind::Exp>(std::move(_source));
        }
        auto log() const&
        {
            return math<expression_detail::ArrayUnaryKind::Log>(_source);
        }
        auto log() &&
        {
            return math<expression_detail::ArrayUnaryKind::Log>(std::move(_source));
        }

    private:
        const Source& source() const
        {
            return expression_detail::unwrap(_source);
        }

        template <typename SourceStorage> static auto wrap(SourceStorage&& source)
        {
            using Node = expression_detail::ArrayIdentityNode<std::decay_t<SourceStorage>>;
            return DenseExpression<Node>(Node(std::forward<SourceStorage>(source)));
        }
        template <typename SourceStorage> static auto shifted(SourceStorage&& source, Value shift)
        {
            using Node = expression_detail::ArrayScalarNode<std::decay_t<SourceStorage>, false>;
            return DenseExpression<Node>(Node(std::forward<SourceStorage>(source), shift));
        }
        template <typename SourceStorage> static auto squared(SourceStorage&& source)
        {
            using Node = expression_detail::ArrayScalarNode<std::decay_t<SourceStorage>, true>;
            return DenseExpression<Node>(Node(std::forward<SourceStorage>(source), Value{}));
        }
        template <expression_detail::ArrayUnaryKind Kind, typename SourceStorage>
        static auto math(SourceStorage&& source)
        {
            using Node = expression_detail::ArrayMathNode<Kind, std::decay_t<SourceStorage>>;
            return DenseExpression<Node>(Node(std::forward<SourceStorage>(source)));
        }
        template <expression_detail::BinaryKind Kind, typename Other>
        auto binary(ArrayExpressionProxy<Other> other) const&
        {
            return makeBinaryExpression<Kind>(matrix(), std::move(other).matrix());
        }
        template <expression_detail::BinaryKind Kind, typename Other> auto binary(ArrayExpressionProxy<Other> other) &&
        {
            return makeBinaryExpression<Kind>(std::move(*this).matrix(), std::move(other).matrix());
        }

        Storage _source;
    };

    template <typename Value, typename Storage> auto operator+(Value shift, ArrayExpressionProxy<Storage> array)
    {
        return std::move(array) + shift;
    }

    template <typename Value,
              typename Storage,
              std::enable_if_t<!expression_detail::isMatrixLike<Value> &&
                                   std::is_convertible_v<Value, typename ArrayExpressionProxy<Storage>::Value>,
                               int> = 0>
    auto operator*(Value scale, ArrayExpressionProxy<Storage> array)
    {
        using Scalar = typename ArrayExpressionProxy<Storage>::Value;
        return std::move(array) * static_cast<Scalar>(scale);
    }
} // namespace plamatrix::v1
