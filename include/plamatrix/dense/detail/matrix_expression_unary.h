#pragma once

#include "plamatrix/dense/detail/matrix_expression_binary.h"

namespace plamatrix::v1
{
    namespace expression_detail
    {
        template <typename Operand> class ScaleNode
        {
            using OperandType = std::decay_t<decltype(unwrap(std::declval<const Operand&>()))>;
            using OperandTraits = Traits<OperandType>;

        public:
            using Value = typename OperandTraits::Value;
            static constexpr int RowsAtCompileTime = OperandTraits::rows;
            static constexpr int ColsAtCompileTime = OperandTraits::cols;
            static constexpr bool ArraySemantics = ArraySyntax<OperandType>::value;
            static constexpr bool LinearAccess = hasLinearAccess<OperandType>;
            static constexpr bool SimdSafe = LinearAccess && isSimdSafe<OperandType>;
            static constexpr std::size_t PacketTermCount =
                (std::is_same_v<Value, float> || std::is_same_v<Value, double>) ? packetReductionTermCount<OperandType>
                                                                                : 0;
            using Result = Matrix<Value, RowsAtCompileTime, ColsAtCompileTime>;

            ScaleNode(Operand operand, Value scale) : _operand(std::move(operand)), _scale(scale)
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
                return expression_detail::supportsGpu(value());
            }
            long double work() const
            {
                return expression_detail::work(value()) + static_cast<long double>(rows()) * cols();
            }
            Value coeff(Index row, Index col) const
            {
                return coefficient(value(), row, col) * _scale;
            }
            Value linearCoeff(Index index) const
            {
                static_assert(LinearAccess, "linearCoeff requires a contiguous column-major operand");
                return linearCoefficient(value(), index) * _scale;
            }
            auto bindLinear() const
            {
                static_assert(LinearAccess, "bindLinear requires a contiguous column-major operand");
                const auto source = linearBind(value());
                return [source, scale = _scale](Index index) { return source(index) * scale; };
            }
            auto packetReductionTerms() const
            {
                static_assert(PacketTermCount > 0,
                              "packetReductionTerms requires a packet-reducible scaled expression");
                return scalePacketReductionTerms(expression_detail::packetReductionTerms(value()), _scale);
            }
            auto bind() const
            {
                auto source = expression_detail::bind(value());
                return [source, scale = _scale](Index row, Index col) { return source(row, col) * scale; };
            }
            Result eager() const
            {
                const auto& operand = materialize(value());
                if constexpr (std::is_same_v<std::decay_t<decltype(operand)>, Result>)
                    return internal::MatrixAccess::scaled(operand, _scale);
                else
                    return internal::MatrixAccess::scaled(Result(operand), _scale);
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
            void assignCoefficients(Result& output) const
            {
                assignNoAlias(output);
            }

        private:
            const OperandType& value() const
            {
                return unwrap(_operand);
            }
            Operand _operand;
            Value _scale;
        };

        template <typename Operand> class DivideNode
        {
            using OperandType = std::decay_t<decltype(unwrap(std::declval<const Operand&>()))>;
            using OperandTraits = Traits<OperandType>;

        public:
            using Value = typename OperandTraits::Value;
            static constexpr int RowsAtCompileTime = OperandTraits::rows;
            static constexpr int ColsAtCompileTime = OperandTraits::cols;
            static constexpr bool ArraySemantics = ArraySyntax<OperandType>::value;
            static constexpr bool LinearAccess = hasLinearAccess<OperandType>;
            static constexpr bool SimdSafe = LinearAccess && isSimdSafe<OperandType>;
            using Result = Matrix<Value, RowsAtCompileTime, ColsAtCompileTime>;

            DivideNode(Operand operand, Value divisor) : _operand(std::move(operand)), _divisor(divisor)
            {
                if (_divisor == Value{})
                {
                    throw internal::Error(internal::ErrorCode::InvalidArgument, "Matrix scalar division by zero");
                }
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
                return expression_detail::supportsGpu(value());
            }
            long double work() const
            {
                return expression_detail::work(value()) + static_cast<long double>(rows()) * cols();
            }
            Value coeff(Index row, Index col) const
            {
                return coefficient(value(), row, col) / _divisor;
            }
            Value linearCoeff(Index index) const
            {
                static_assert(LinearAccess, "linearCoeff requires a contiguous column-major operand");
                return linearCoefficient(value(), index) / _divisor;
            }
            auto bindLinear() const
            {
                static_assert(LinearAccess, "bindLinear requires a contiguous column-major operand");
                const auto source = linearBind(value());
                return [source, divisor = _divisor](Index index) { return source(index) / divisor; };
            }
            auto bind() const
            {
                auto source = expression_detail::bind(value());
                return [source, divisor = _divisor](Index row, Index col) { return source(row, col) / divisor; };
            }
            Result eager() const
            {
                if constexpr (std::is_floating_point_v<Value>)
                {
                    const auto& operand = materialize(value());
                    if constexpr (std::is_same_v<std::decay_t<decltype(operand)>, Result>)
                        return internal::MatrixAccess::scaled(operand, Value{1} / _divisor);
                    else
                        return internal::MatrixAccess::scaled(Result(operand), Value{1} / _divisor);
                }
                else
                {
                    Result result(rows(), cols());
                    assignNoAlias(result);
                    return result;
                }
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
            void assignCoefficients(Result& output) const
            {
                assignNoAlias(output);
            }

        private:
            const OperandType& value() const
            {
                return unwrap(_operand);
            }
            Operand _operand;
            Value _divisor;
        };
    } // namespace expression_detail
} // namespace plamatrix::v1
