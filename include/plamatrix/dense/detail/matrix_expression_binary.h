#pragma once

#include <memory>

#include "plamatrix/dense/detail/matrix_expression_detail.h"
#include "plamatrix/internal/dense/packet_reduction.h"

namespace plamatrix::v1
{
    namespace expression_detail
    {
        enum class BinaryKind
        {
            Add,
            Subtract,
            Product,
            CwiseProduct,
            CwiseDivide
        };

        template <BinaryKind Kind, typename Left, typename Right> class BinaryNode
        {
            using LeftType = std::decay_t<decltype(unwrap(std::declval<const Left&>()))>;
            using RightType = std::decay_t<decltype(unwrap(std::declval<const Right&>()))>;
            using LeftTraits = Traits<LeftType>;
            using RightTraits = Traits<RightType>;

        public:
            using LeftValue = typename LeftTraits::Value;
            using RightValue = typename RightTraits::Value;
            using Value = typename ScalarBinaryOpTraits<LeftValue, RightValue>::ReturnType;
            static constexpr int RowsAtCompileTime = LeftTraits::rows;
            static constexpr int ColsAtCompileTime = Kind == BinaryKind::Product ? RightTraits::cols : LeftTraits::cols;
            static constexpr bool ArraySemantics =
                Kind != BinaryKind::Product && ArraySyntax<LeftType>::value && ArraySyntax<RightType>::value;
            static constexpr bool LinearAccess =
                Kind != BinaryKind::Product && hasLinearAccess<LeftType> && hasLinearAccess<RightType>;
            static constexpr bool SimdSafe = LinearAccess && isSimdSafe<LeftType> && isSimdSafe<RightType> &&
                                             !(Kind == BinaryKind::CwiseDivide && std::is_integral_v<Value>);
            static constexpr bool PacketScalarCompatible =
                std::is_same_v<LeftValue, Value> && std::is_same_v<RightValue, Value> &&
                (std::is_same_v<Value, float> || std::is_same_v<Value, double>);
            static constexpr bool PacketProduct = Kind == BinaryKind::CwiseProduct && PacketScalarCompatible &&
                                                  hasDirectDataAccess<LeftType> && hasDirectDataAccess<RightType>;
            static constexpr bool PacketCombination =
                (Kind == BinaryKind::Add || Kind == BinaryKind::Subtract) && PacketScalarCompatible &&
                packetReductionTermCount<LeftType> > 0 && packetReductionTermCount<RightType> > 0;
            static constexpr std::size_t PacketTermCount =
                PacketProduct
                    ? 1
                    : (PacketCombination ? packetReductionTermCount<LeftType> + packetReductionTermCount<RightType>
                                         : 0);
            using Result = Matrix<Value, RowsAtCompileTime, ColsAtCompileTime>;

            BinaryNode(Left left, Right right) : _left(std::move(left)), _right(std::move(right))
            {
                if constexpr (Kind == BinaryKind::Product)
                {
                    static_assert(LeftTraits::cols == Dynamic || RightTraits::rows == Dynamic ||
                                      LeftTraits::cols == RightTraits::rows,
                                  "Fixed product dimensions do not match");
                    if (leftValue().cols() != rightValue().rows())
                    {
                        throw internal::Error(internal::ErrorCode::InvalidArgument,
                                              "Matrix multiplication dimensions do not match");
                    }
                }
                else
                {
                    static_assert(LeftTraits::rows == Dynamic || RightTraits::rows == Dynamic ||
                                      LeftTraits::rows == RightTraits::rows,
                                  "Fixed matrix row dimensions do not match");
                    static_assert(LeftTraits::cols == Dynamic || RightTraits::cols == Dynamic ||
                                      LeftTraits::cols == RightTraits::cols,
                                  "Fixed matrix column dimensions do not match");
                    if (leftValue().rows() != rightValue().rows() || leftValue().cols() != rightValue().cols())
                    {
                        throw internal::Error(internal::ErrorCode::InvalidArgument,
                                              "Elementwise matrix operation requires equal shapes");
                    }
                }
            }

            Index rows() const
            {
                return leftValue().rows();
            }
            Index cols() const
            {
                return Kind == BinaryKind::Product ? rightValue().cols() : leftValue().cols();
            }
            bool hasResidentInput() const
            {
                return resident(leftValue()) || resident(rightValue());
            }
            std::size_t pendingDownloadBytes() const
            {
                return pendingDownload(leftValue()) + pendingDownload(rightValue());
            }
            bool hasProduct() const
            {
                return Kind == BinaryKind::Product || expression_detail::hasProduct(leftValue()) ||
                       expression_detail::hasProduct(rightValue());
            }
            bool supportsGpu() const
            {
                return Kind != BinaryKind::CwiseDivide && expression_detail::supportsGpu(leftValue()) &&
                       expression_detail::supportsGpu(rightValue());
            }
            long double work() const
            {
                const long double own = Kind == BinaryKind::Product
                                            ? static_cast<long double>(rows()) * cols() * leftValue().cols()
                                            : static_cast<long double>(rows()) * cols();
                return own + expression_detail::work(leftValue()) + expression_detail::work(rightValue());
            }

            Value coeff(Index row, Index col) const
            {
                if constexpr (Kind == BinaryKind::Product)
                {
                    Value sum{};
                    for (Index inner = 0; inner < leftValue().cols(); ++inner)
                    {
                        sum += static_cast<Value>(coefficient(leftValue(), row, inner)) *
                               static_cast<Value>(coefficient(rightValue(), inner, col));
                    }
                    return sum;
                }
                else if constexpr (Kind == BinaryKind::Add)
                {
                    return static_cast<Value>(coefficient(leftValue(), row, col)) +
                           static_cast<Value>(coefficient(rightValue(), row, col));
                }
                else if constexpr (Kind == BinaryKind::Subtract)
                {
                    return static_cast<Value>(coefficient(leftValue(), row, col)) -
                           static_cast<Value>(coefficient(rightValue(), row, col));
                }
                else if constexpr (Kind == BinaryKind::CwiseDivide)
                {
                    const Value divisor = static_cast<Value>(coefficient(rightValue(), row, col));
                    if constexpr (std::is_integral_v<Value>)
                        if (divisor == Value{})
                            throw internal::Error(internal::ErrorCode::InvalidArgument,
                                                  "Integer array division by zero");
                    return static_cast<Value>(coefficient(leftValue(), row, col)) / divisor;
                }
                else
                {
                    return static_cast<Value>(coefficient(leftValue(), row, col)) *
                           static_cast<Value>(coefficient(rightValue(), row, col));
                }
            }

            Value linearCoeff(Index index) const
            {
                static_assert(LinearAccess, "linearCoeff requires contiguous column-major operands");
                if constexpr (Kind == BinaryKind::Add)
                    return static_cast<Value>(linearCoefficient(leftValue(), index)) +
                           static_cast<Value>(linearCoefficient(rightValue(), index));
                else if constexpr (Kind == BinaryKind::Subtract)
                    return static_cast<Value>(linearCoefficient(leftValue(), index)) -
                           static_cast<Value>(linearCoefficient(rightValue(), index));
                else if constexpr (Kind == BinaryKind::CwiseDivide)
                {
                    const Value divisor = static_cast<Value>(linearCoefficient(rightValue(), index));
                    if constexpr (std::is_integral_v<Value>)
                        if (divisor == Value{})
                            throw internal::Error(internal::ErrorCode::InvalidArgument,
                                                  "Integer array division by zero");
                    return static_cast<Value>(linearCoefficient(leftValue(), index)) / divisor;
                }
                else
                    return static_cast<Value>(linearCoefficient(leftValue(), index)) *
                           static_cast<Value>(linearCoefficient(rightValue(), index));
            }

            auto bindLinear() const
            {
                static_assert(LinearAccess, "bindLinear requires contiguous column-major operands");
                const auto left = linearBind(leftValue());
                const auto right = linearBind(rightValue());
                return [left, right](Index index) -> Value
                {
                    if constexpr (Kind == BinaryKind::Add)
                        return static_cast<Value>(left(index)) + static_cast<Value>(right(index));
                    else if constexpr (Kind == BinaryKind::Subtract)
                        return static_cast<Value>(left(index)) - static_cast<Value>(right(index));
                    else if constexpr (Kind == BinaryKind::CwiseDivide)
                    {
                        const Value divisor = static_cast<Value>(right(index));
                        if constexpr (std::is_integral_v<Value>)
                            if (divisor == Value{})
                                throw internal::Error(internal::ErrorCode::InvalidArgument,
                                                      "Integer array division by zero");
                        return static_cast<Value>(left(index)) / divisor;
                    }
                    else
                        return static_cast<Value>(left(index)) * static_cast<Value>(right(index));
                };
            }

            auto packetReductionTerms() const
            {
                static_assert(PacketTermCount > 0,
                              "packetReductionTerms requires a packet-reducible coefficient expression");
                using Term = internal::detail::PacketReductionTerm<Value>;
                if constexpr (PacketProduct)
                {
                    return std::array<Term, 1>{{Term{expression_detail::directData(leftValue()),
                                                     expression_detail::directData(rightValue()),
                                                     Value{1}}}};
                }
                else
                {
                    auto left_terms = expression_detail::packetReductionTerms(leftValue());
                    auto right_terms = expression_detail::packetReductionTerms(rightValue());
                    constexpr Value right_scale = Kind == BinaryKind::Subtract ? Value{-1} : Value{1};
                    return concatenatePacketReductionTerms(left_terms, right_terms, right_scale);
                }
            }

            auto bind() const
            {
                if constexpr (Kind == BinaryKind::Product)
                {
                    auto product = std::make_shared<Result>(rows(), cols());
                    assignNoAlias(*product);
                    const Value* data = product->data();
                    const Index row_count = rows();
                    return [product, data, row_count](Index row, Index col) -> Value
                    { return data[row + col * row_count]; };
                }
                else
                {
                    auto left = expression_detail::bind(leftValue());
                    auto right = expression_detail::bind(rightValue());
                    return [left, right](Index row, Index col) -> Value
                    {
                        if constexpr (Kind == BinaryKind::Add)
                            return static_cast<Value>(left(row, col)) + static_cast<Value>(right(row, col));
                        else if constexpr (Kind == BinaryKind::Subtract)
                            return static_cast<Value>(left(row, col)) - static_cast<Value>(right(row, col));
                        else if constexpr (Kind == BinaryKind::CwiseDivide)
                        {
                            const Value divisor = static_cast<Value>(right(row, col));
                            if constexpr (std::is_integral_v<Value>)
                                if (divisor == Value{})
                                    throw internal::Error(internal::ErrorCode::InvalidArgument,
                                                          "Integer array division by zero");
                            return static_cast<Value>(left(row, col)) / divisor;
                        }
                        else
                            return static_cast<Value>(left(row, col)) * static_cast<Value>(right(row, col));
                    };
                }
            }

            Result eager() const
            {
                const auto& left = materialize(leftValue());
                const auto& right = materialize(rightValue());
                if constexpr (Kind == BinaryKind::Product)
                {
                    if constexpr (std::is_same_v<std::decay_t<decltype(left)>, Result> &&
                                  std::is_same_v<std::decay_t<decltype(right)>, Result>)
                    {
                        return multiplyEager(left, right);
                    }
                    else
                    {
                        Result result(rows(), cols());
                        assignNoAlias(result);
                        return result;
                    }
                }
                else if constexpr (Kind == BinaryKind::CwiseDivide)
                {
                    Result result(rows(), cols());
                    assignNoAlias(result);
                    return result;
                }
                else
                {
                    if constexpr (std::is_same_v<std::decay_t<decltype(left)>, Result> &&
                                  std::is_same_v<std::decay_t<decltype(right)>, Result>)
                    {
                        if constexpr (Kind == BinaryKind::Add)
                            return left.add(right);
                        if constexpr (Kind == BinaryKind::Subtract)
                            return left.subtract(right);
                        return left.cwiseProductEager(right);
                    }
                    else if constexpr (isCompatibleGpuMatrix<std::decay_t<decltype(left)>> &&
                                       isCompatibleGpuMatrix<std::decay_t<decltype(right)>>)
                    {
                        return eagerElementwise(left, right);
                    }
                    else
                    {
                        Result result(rows(), cols());
                        assignCoefficients(result);
                        return result;
                    }
                }
            }

            void assignNoAlias(Result& output) const
            {
                if constexpr (Kind == BinaryKind::Product)
                {
                    output.resize(rows(), cols());
                    if (output.size() == 0 || leftValue().cols() == 0)
                    {
                        output.setZero();
                    }
                    else if constexpr ((std::is_same_v<Value, float> || std::is_same_v<Value, double>) &&
                                       std::is_same_v<LeftType, Matrix<Value, LeftTraits::rows, LeftTraits::cols>> &&
                                       std::is_same_v<RightType, Matrix<Value, RightTraits::rows, RightTraits::cols>>)
                    {
                        plamatrix::internal::detail::cpuGemm(
                            leftValue().data(), rightValue().data(), output.data(), rows(), cols(), leftValue().cols());
                    }
                    else
                    {
                        const auto left = expression_detail::bind(leftValue());
                        const auto right = expression_detail::bind(rightValue());
                        Value* destination = output.data();
                        for (Index col = 0; col < cols(); ++col)
                        {
                            for (Index row = 0; row < rows(); ++row)
                            {
                                Value sum{};
                                for (Index inner = 0; inner < leftValue().cols(); ++inner)
                                    sum += left(row, inner) * right(inner, col);
                                destination[row + col * rows()] = sum;
                            }
                        }
                    }
                }
                else
                {
                    assignCoefficients(output);
                }
            }

            void assignCoefficients(Result& output) const
            {
                Value* destination = output.data();
                if constexpr (LinearAccess)
                {
                    const Index count = rows() * cols();
                    const auto evaluator = bindLinear();
                    if constexpr (SimdSafe && std::is_floating_point_v<Value>)
                    {
#pragma omp simd
                        for (Index index = 0; index < count; ++index)
                            destination[index] = evaluator(index);
                    }
                    else
                    {
                        for (Index index = 0; index < count; ++index)
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
            template <typename MatrixType>
            static constexpr bool isCompatibleGpuMatrix =
                std::is_same_v<typename Traits<MatrixType>::Value, Value> && MatrixType::IsRowMajor == 0;

            template <typename LeftMatrix, typename RightMatrix>
            Result eagerElementwise(const LeftMatrix& left, const RightMatrix& right) const
            {
                Result result(rows(), cols());
                const bool left_resident = internal::MatrixAccess::isDeviceResident(left);
                const bool right_resident = internal::MatrixAccess::isDeviceResident(right);
                const internal::Backend resident_backend =
                    left_resident
                        ? internal::MatrixAccess::deviceBackend(left)
                        : (right_resident ? internal::MatrixAccess::deviceBackend(right) : internal::Backend::Cpu);
                const auto choice =
                    plamatrix::detail::chooseDenseBackend<Value>(plamatrix::detail::DenseOperation::Elementwise,
                                                                 static_cast<long double>(rows()) * cols(),
                                                                 left_resident || right_resident,
                                                                 resident_backend);
                if (choice.useGpu() && result.size() != 0)
                {
                    std::shared_ptr<internal::detail::GpuStorage<Value>> gpu_result;
                    if constexpr (Kind == BinaryKind::Add)
                    {
                        gpu_result = internal::detail::GpuOps<Value>::add(left.gpuStorage(choice.backend),
                                                                          right.gpuStorage(choice.backend));
                    }
                    else if constexpr (Kind == BinaryKind::Subtract)
                    {
                        gpu_result = internal::detail::GpuOps<Value>::subtract(left.gpuStorage(choice.backend),
                                                                               right.gpuStorage(choice.backend));
                    }
                    else
                    {
                        gpu_result = internal::detail::GpuOps<Value>::cwiseProduct(left.gpuStorage(choice.backend),
                                                                                   right.gpuStorage(choice.backend));
                    }
                    result.adoptGpu(
                        std::move(gpu_result),
                        plamatrix::detail::gpuInfo<Value>(choice.backend,
                                                          choice.reason,
                                                          left_resident,
                                                          right_resident,
                                                          static_cast<std::size_t>(left.size()) * sizeof(Value),
                                                          static_cast<std::size_t>(right.size()) * sizeof(Value)));
                    return result;
                }

                const std::size_t downloaded_bytes = internal::MatrixAccess::pendingHostDownloadBytes(left) +
                                                     internal::MatrixAccess::pendingHostDownloadBytes(right);
                const Value* left_data = left.data();
                const Value* right_data = right.data();
                Value* output = result.data();
#pragma omp simd
                for (Index index = 0; index < result.size(); ++index)
                {
                    if constexpr (Kind == BinaryKind::Add)
                        output[index] = left_data[index] + right_data[index];
                    else if constexpr (Kind == BinaryKind::Subtract)
                        output[index] = left_data[index] - right_data[index];
                    else
                        output[index] = left_data[index] * right_data[index];
                }
                result._info.bytesDownloaded = downloaded_bytes;
                result._info.reason = choice.reason;
                return result;
            }

            const LeftType& leftValue() const
            {
                return unwrap(_left);
            }
            const RightType& rightValue() const
            {
                return unwrap(_right);
            }

            Left _left;
            Right _right;
        };

    } // namespace expression_detail
} // namespace plamatrix::v1
