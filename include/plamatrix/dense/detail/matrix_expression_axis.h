#pragma once

namespace plamatrix::v1
{
    namespace expression_detail
    {
        template <bool ByRows, bool Mean, typename Storage> class AxisReductionNode
        {
            using Source = std::decay_t<decltype(unwrap(std::declval<const Storage&>()))>;
            using SourceTraits = Traits<Source>;

        public:
            using Value = typename SourceTraits::Value;
            static constexpr int RowsAtCompileTime = ByRows ? SourceTraits::rows : 1;
            static constexpr int ColsAtCompileTime = ByRows ? 1 : SourceTraits::cols;
            using Result = Matrix<Value, RowsAtCompileTime, ColsAtCompileTime>;

            explicit AxisReductionNode(Storage source) : _source(std::move(source))
            {
                if constexpr (Mean)
                    if ((ByRows ? value().cols() : value().rows()) == 0)
                        throw internal::Error(internal::ErrorCode::InvalidArgument, "Axiswise mean requires a nonempty reduction axis");
            }
            Index rows() const
            {
                return ByRows ? value().rows() : 1;
            }
            Index cols() const
            {
                return ByRows ? 1 : value().cols();
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
                return expression_detail::work(value()) + static_cast<long double>(value().rows()) * value().cols();
            }
            Value coeff(Index row, Index col) const
            {
                return bind()(row, col);
            }
            auto bind() const
            {
                const auto source = expression_detail::bind(value());
                const Index count = ByRows ? value().cols() : value().rows();
                return [source, count](Index row, Index col)
                {
                    Value result{};
                    for (Index index = 0; index < count; ++index)
                    {
                        if constexpr (ByRows)
                            result += source(row, index);
                        else
                            result += source(index, col);
                    }
                    if constexpr (Mean)
                        result /= static_cast<Value>(count);
                    return result;
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
                const auto evaluator = bind();
                Value* destination = output.data();
                for (Index col = 0; col < cols(); ++col)
                    for (Index row = 0; row < rows(); ++row)
                        destination[row + col * rows()] = evaluator(row, col);
            }

        private:
            const Source& value() const
            {
                return unwrap(_source);
            }
            Storage _source;
        };

        template <bool ByRows, bool Subtract, typename SourceStorage, typename VectorStorage> class BroadcastNode
        {
            using Source = std::decay_t<decltype(unwrap(std::declval<const SourceStorage&>()))>;
            using Vector = std::decay_t<decltype(unwrap(std::declval<const VectorStorage&>()))>;
            using SourceTraits = Traits<Source>;
            using VectorTraits = Traits<Vector>;

        public:
            using Value = typename SourceTraits::Value;
            static constexpr int RowsAtCompileTime = SourceTraits::rows;
            static constexpr int ColsAtCompileTime = SourceTraits::cols;
            using Result = Matrix<Value, RowsAtCompileTime, ColsAtCompileTime>;

            BroadcastNode(SourceStorage source, VectorStorage vector)
                : _source(std::move(source)), _vector(std::move(vector))
            {
                static_assert(std::is_same_v<Value, typename VectorTraits::Value>, "Broadcast scalar types must match");
                static_assert(ByRows ? VectorTraits::rows == Dynamic || VectorTraits::rows == 1
                                     : VectorTraits::cols == Dynamic || VectorTraits::cols == 1,
                              "Broadcast operand must be a row or column vector");
                if (ByRows ? (vectorValue().rows() != 1 || vectorValue().cols() != value().cols())
                           : (vectorValue().rows() != value().rows() || vectorValue().cols() != 1))
                    throw internal::Error(internal::ErrorCode::InvalidArgument, "Axiswise broadcast vector shape does not match");
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
                return resident(value()) || resident(vectorValue());
            }
            std::size_t pendingDownloadBytes() const
            {
                return pendingDownload(value()) + pendingDownload(vectorValue());
            }
            bool hasProduct() const
            {
                return expression_detail::hasProduct(value()) || expression_detail::hasProduct(vectorValue());
            }
            bool supportsGpu() const
            {
                return false;
            }
            long double work() const
            {
                return expression_detail::work(value()) + expression_detail::work(vectorValue()) +
                       static_cast<long double>(rows()) * cols();
            }
            Value coeff(Index row, Index col) const
            {
                const Value vector_item = coefficient(vectorValue(), ByRows ? 0 : row, ByRows ? col : 0);
                if constexpr (Subtract)
                    return coefficient(value(), row, col) - vector_item;
                return coefficient(value(), row, col) + vector_item;
            }
            auto bind() const
            {
                const auto source = expression_detail::bind(value());
                const auto vector = expression_detail::bind(vectorValue());
                return [source, vector](Index row, Index col)
                {
                    const Value vector_item = vector(ByRows ? 0 : row, ByRows ? col : 0);
                    if constexpr (Subtract)
                        return source(row, col) - vector_item;
                    return source(row, col) + vector_item;
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
                const auto evaluator = bind();
                Value* destination = output.data();
                for (Index col = 0; col < cols(); ++col)
                    for (Index row = 0; row < rows(); ++row)
                        destination[row + col * rows()] = evaluator(row, col);
            }

        private:
            const Source& value() const
            {
                return unwrap(_source);
            }
            const Vector& vectorValue() const
            {
                return unwrap(_vector);
            }
            SourceStorage _source;
            VectorStorage _vector;
        };
    } // namespace expression_detail

    /// Lazy row or column reduction and broadcast facade.
    template <bool ByRows, typename Storage> class AxiswiseExpressionProxy
    {
    public:
        explicit AxiswiseExpressionProxy(Storage source) : _source(std::move(source))
        {
        }

        auto sum() const&
        {
            return reduce<false>(_source);
        }
        auto sum() &&
        {
            return reduce<false>(std::move(_source));
        }
        auto mean() const&
        {
            return reduce<true>(_source);
        }
        auto mean() &&
        {
            return reduce<true>(std::move(_source));
        }

        template <typename Vector, std::enable_if_t<expression_detail::isMatrixLike<Vector>, int> = 0>
        auto operator+(Vector&& vector) const&
        {
            return broadcast<false>(_source, std::forward<Vector>(vector));
        }
        template <typename Vector, std::enable_if_t<expression_detail::isMatrixLike<Vector>, int> = 0>
        auto operator+(Vector&& vector) &&
        {
            return broadcast<false>(std::move(_source), std::forward<Vector>(vector));
        }
        template <typename Vector, std::enable_if_t<expression_detail::isMatrixLike<Vector>, int> = 0>
        auto operator-(Vector&& vector) const&
        {
            return broadcast<true>(_source, std::forward<Vector>(vector));
        }
        template <typename Vector, std::enable_if_t<expression_detail::isMatrixLike<Vector>, int> = 0>
        auto operator-(Vector&& vector) &&
        {
            return broadcast<true>(std::move(_source), std::forward<Vector>(vector));
        }

    private:
        template <bool Mean, typename SourceStorage> static auto reduce(SourceStorage&& source)
        {
            using Node = expression_detail::AxisReductionNode<ByRows, Mean, std::decay_t<SourceStorage>>;
            return DenseExpression<Node>(Node(std::forward<SourceStorage>(source)));
        }
        template <bool Subtract, typename SourceStorage, typename Vector>
        static auto broadcast(SourceStorage&& source, Vector&& vector)
        {
            using VectorStorage = decltype(expression_detail::capture(std::forward<Vector>(vector)));
            using Node = expression_detail::BroadcastNode<ByRows, Subtract, std::decay_t<SourceStorage>, VectorStorage>;
            return DenseExpression<Node>(
                Node(std::forward<SourceStorage>(source), expression_detail::capture(std::forward<Vector>(vector))));
        }
        Storage _source;
    };
} // namespace plamatrix::v1
