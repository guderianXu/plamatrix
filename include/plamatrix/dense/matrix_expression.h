#pragma once

#include <ostream>

#include "plamatrix/dense/matrix.h"
#include "plamatrix/dense/detail/matrix_expression_unary.h"

namespace plamatrix::v1
{
    /// A lazily composed dense operation; eval() materializes its result.
    template <typename Node> class DenseExpression : public MatrixBase<DenseExpression<Node>>
    {
    public:
        using Base = MatrixBase<DenseExpression>;
        using ScalarType = typename Node::Value;
        using Scalar = ScalarType;
        using RealScalar = typename NumTraits<Scalar>::Real;
        using Base::isApprox;
        using Base::maxCoeff;
        using Base::minCoeff;
        static constexpr int RowsAtCompileTime = Node::RowsAtCompileTime;
        static constexpr int ColsAtCompileTime = Node::ColsAtCompileTime;
        using Result = Matrix<ScalarType, RowsAtCompileTime, ColsAtCompileTime>;

        explicit DenseExpression(Node node) : _node(std::move(node))
        {
        }
        Index rows() const
        {
            return _node.rows();
        }
        Index cols() const
        {
            return _node.cols();
        }
        Index size() const
        {
            return rows() * cols();
        }
        bool hasResidentInput() const
        {
            return _node.hasResidentInput();
        }
        std::size_t pendingDownloadBytes() const
        {
            return _node.pendingDownloadBytes();
        }
        bool hasProduct() const
        {
            return _node.hasProduct();
        }
        bool supportsGpu() const
        {
            return _node.supportsGpu();
        }
        long double work() const
        {
            return _node.work();
        }
        ScalarType operator()(Index row, Index col) const
        {
            if (row < 0 || col < 0 || row >= rows() || col >= cols())
                throw std::out_of_range("Expression coefficient index is out of range");
            return _node.coeff(row, col);
        }
        ScalarType operator()(Index index) const
        {
            if (index < 0 || index >= size())
                throw std::out_of_range("Expression coefficient index is out of range");
            if (cols() == 1)
                return _node.coeff(index, 0);
            if (rows() == 1)
                return _node.coeff(0, index);
            throw internal::Error(internal::ErrorCode::InvalidArgument,
                                  "Single-index expression access requires a vector");
        }
        ScalarType coeff(Index row, Index col) const
        {
            return _node.coeff(row, col);
        }
        ScalarType linearCoeff(Index index) const
        {
            static_assert(expression_detail::NodeLinearAccess<Node>::value,
                          "linearCoeff requires a contiguous expression");
            return _node.linearCoeff(index);
        }
        auto bind() const
        {
            return _node.bind();
        }
        auto bindLinear() const
        {
            static_assert(expression_detail::NodeLinearAccess<Node>::value,
                          "bindLinear requires a contiguous expression");
            return _node.bindLinear();
        }
        const ScalarType* directData() const
        {
            static_assert(expression_detail::NodeDirectDataAccess<Node>::value,
                          "directData requires a contiguous matrix expression");
            return _node.directData();
        }
        auto packetReductionTerms() const
        {
            static_assert(expression_detail::NodePacketTermCount<Node>::value > 0,
                          "packetReductionTerms requires a packet-reducible expression");
            return _node.packetReductionTerms();
        }
        ScalarType sum() const
        {
            if constexpr (expression_detail::NodePacketSum<Node>::value)
            {
                const auto terms = _node.packetReductionTerms();
                return internal::detail::packetExpressionSum(terms.data(), static_cast<Index>(terms.size()), size());
            }
            ScalarType result{};
            if constexpr (expression_detail::NodeLinearAccess<Node>::value)
            {
                const auto evaluator = _node.bindLinear();
                if constexpr (expression_detail::NodeSimdSafe<Node>::value && std::is_floating_point_v<ScalarType>)
                {
#pragma omp simd reduction(+ : result)
                    for (Index index = 0; index < size(); ++index)
                        result += evaluator(index);
                }
                else
                {
                    for (Index index = 0; index < size(); ++index)
                        result += evaluator(index);
                }
            }
            else
            {
                const auto evaluator = _node.bind();
                for (Index col = 0; col < cols(); ++col)
                    for (Index row = 0; row < rows(); ++row)
                        result += evaluator(row, col);
            }
            return result;
        }
        ScalarType minCoeff() const
        {
            if (size() == 0)
                throw internal::Error(internal::ErrorCode::InvalidArgument, "Minimum requires a nonempty expression");
            const auto evaluator = _node.bind();
            ScalarType result = evaluator(0, 0);
            for (Index col = 0; col < cols(); ++col)
                for (Index row = 0; row < rows(); ++row)
                    result = std::min(result, evaluator(row, col));
            return result;
        }
        ScalarType maxCoeff() const
        {
            if (size() == 0)
                throw internal::Error(internal::ErrorCode::InvalidArgument, "Maximum requires a nonempty expression");
            const auto evaluator = _node.bind();
            ScalarType result = evaluator(0, 0);
            for (Index col = 0; col < cols(); ++col)
                for (Index row = 0; row < rows(); ++row)
                    result = std::max(result, evaluator(row, col));
            return result;
        }
        /// Return the vector dot product without materializing either input.
        template <typename Other> auto dot(const Other& other) const
        {
            static_assert(expression_detail::isMatrixLike<Other>, "dot() requires a matrix expression");
            using OtherScalar = typename expression_detail::TypeTraits<Other>::Value;
            using ReturnScalar = typename ScalarBinaryOpTraits<ScalarType, OtherScalar>::ReturnType;
            if ((rows() != 1 && cols() != 1) || (other.rows() != 1 && other.cols() != 1) || size() != other.size())
                throw internal::Error(internal::ErrorCode::InvalidArgument, "dot() requires vectors of equal length");
            ReturnScalar result{};
            if constexpr (expression_detail::NodeLinearAccess<Node>::value &&
                          expression_detail::hasLinearAccess<Other> && expression_detail::NodeSimdSafe<Node>::value &&
                          expression_detail::isSimdSafe<Other> && std::is_floating_point_v<ReturnScalar>)
            {
                const auto left = _node.bindLinear();
                const auto right = expression_detail::linearBind(other);
#pragma omp simd reduction(+ : result)
                for (Index index = 0; index < size(); ++index)
                    result += static_cast<ReturnScalar>(left(index)) * static_cast<ReturnScalar>(right(index));
            }
            else
            {
                const auto left = _node.bind();
                const auto right = expression_detail::bind(other);
                for (Index index = 0; index < size(); ++index)
                {
                    const ReturnScalar left_value =
                        static_cast<ReturnScalar>(cols() == 1 ? left(index, 0) : left(0, index));
                    const ReturnScalar right_value =
                        static_cast<ReturnScalar>(other.cols() == 1 ? right(index, 0) : right(0, index));
                    if constexpr (NumTraits<ReturnScalar>::IsComplex)
                        result += std::conj(left_value) * right_value;
                    else
                        result += left_value * right_value;
                }
            }
            return result;
        }

        RealScalar squaredNorm() const
        {
            RealScalar result{};
            if constexpr (expression_detail::NodeLinearAccess<Node>::value &&
                          expression_detail::NodeSimdSafe<Node>::value && std::is_floating_point_v<ScalarType>)
            {
                const auto evaluator = _node.bindLinear();
#pragma omp simd reduction(+ : result)
                for (Index index = 0; index < size(); ++index)
                {
                    const ScalarType value = evaluator(index);
                    result += value * value;
                }
            }
            else
            {
                const auto evaluator = _node.bind();
                for (Index col = 0; col < cols(); ++col)
                    for (Index row = 0; row < rows(); ++row)
                    {
                        const ScalarType value = evaluator(row, col);
                        if constexpr (NumTraits<ScalarType>::IsComplex)
                            result += std::norm(value);
                        else
                            result += value * value;
                    }
            }
            return result;
        }
        RealScalar norm() const
        {
            return static_cast<RealScalar>(std::sqrt(squaredNorm()));
        }

        ScalarType trace() const
        {
            const auto evaluator = _node.bind();
            ScalarType result{};
            for (Index index = 0; index < std::min(rows(), cols()); ++index)
                result += evaluator(index, index);
            return result;
        }

        bool allFinite() const
        {
            if constexpr (NumTraits<ScalarType>::IsComplex)
            {
                const auto evaluator = _node.bind();
                for (Index col = 0; col < cols(); ++col)
                    for (Index row = 0; row < rows(); ++row)
                    {
                        const ScalarType value = evaluator(row, col);
                        if (!std::isfinite(value.real()) || !std::isfinite(value.imag()))
                            return false;
                    }
                return true;
            }
            else if constexpr (!std::is_floating_point_v<ScalarType>)
                return true;
            const auto evaluator = _node.bind();
            for (Index col = 0; col < cols(); ++col)
                for (Index row = 0; row < rows(); ++row)
                    if (!std::isfinite(evaluator(row, col)))
                        return false;
            return true;
        }

        const ScalarType* data() const
        {
            return cached().data();
        }
        Index innerStride() const noexcept
        {
            return 1;
        }
        Index outerStride() const noexcept
        {
            return rows();
        }

    private:
        friend struct internal::MatrixAccess;
        const internal::ExecutionInfo& executionInfo() const
        {
            return cached().executionInfo();
        }
        bool isDeviceResident() const
        {
            return cached().isDeviceResident();
        }

    public:
        Transpose<DenseExpression> transpose();
        const Transpose<const DenseExpression> transpose() const;
        template <typename Other> auto cwiseProduct(Other&& other) const&;
        template <typename Other> auto cwiseProduct(Other&& other) &&;
        template <typename Other> auto cwiseProduct(Other&& other) const&&;
        auto array() const&;
        auto array() &&;
        auto array() const&&;
        template <typename N = Node, std::enable_if_t<N::ArraySemantics, int> = 0> auto square() const&
        {
            return array().square();
        }
        template <typename N = Node, std::enable_if_t<N::ArraySemantics, int> = 0> auto square() &&
        {
            return std::move(*this).array().square();
        }
        template <typename N = Node, std::enable_if_t<N::ArraySemantics, int> = 0> auto abs() const&
        {
            return array().abs();
        }
        template <typename N = Node, std::enable_if_t<N::ArraySemantics, int> = 0> auto abs() &&
        {
            return std::move(*this).array().abs();
        }
        template <typename N = Node, std::enable_if_t<N::ArraySemantics, int> = 0> auto sqrt() const&
        {
            return array().sqrt();
        }
        template <typename N = Node, std::enable_if_t<N::ArraySemantics, int> = 0> auto sqrt() &&
        {
            return std::move(*this).array().sqrt();
        }
        template <typename N = Node, std::enable_if_t<N::ArraySemantics, int> = 0> auto exp() const&
        {
            return array().exp();
        }
        template <typename N = Node, std::enable_if_t<N::ArraySemantics, int> = 0> auto exp() &&
        {
            return std::move(*this).array().exp();
        }
        template <typename N = Node, std::enable_if_t<N::ArraySemantics, int> = 0> auto log() const&
        {
            return array().log();
        }
        template <typename N = Node, std::enable_if_t<N::ArraySemantics, int> = 0> auto log() &&
        {
            return std::move(*this).array().log();
        }
        auto rowwise() const&;
        auto rowwise() &&;
        auto rowwise() const&&;
        auto colwise() const&;
        auto colwise() &&;
        auto colwise() const&&;

        Result eval() const
        {
            Result result(rows(), cols());
            evaluateInto(result);
            return result;
        }

        template <int Rows, int Cols, int Options, int MaxRows, int MaxCols>
        void evaluateInto(Matrix<ScalarType, Rows, Cols, Options, MaxRows, MaxCols>& output,
                          bool no_alias = false) const
        {
            if (!supportsGpu() && internal::currentExecutionSettings().policy == internal::ExecutionPolicy::GpuRequired)
                throw internal::Error(internal::ErrorCode::UnsupportedOperation,
                                      "Expression operation has no CUDA implementation");
            if (supportsGpu())
            {
                const auto choice = plamatrix::detail::chooseDenseBackend<ScalarType>(
                    hasProduct() ? plamatrix::detail::DenseOperation::Product
                                 : plamatrix::detail::DenseOperation::Elementwise,
                    work(),
                    hasResidentInput());
                if (choice.useGpu())
                {
                    Result gpu_result = _node.eager();
                    if constexpr (Rows == RowsAtCompileTime && Cols == ColsAtCompileTime &&
                                  Options == Result::Options && MaxRows == Result::MaxRowsAtCompileTime &&
                                  MaxCols == Result::MaxColsAtCompileTime)
                    {
                        output = std::move(gpu_result);
                    }
                    else if constexpr (Options == ColMajor)
                    {
                        if (gpu_result.isDeviceResident())
                        {
                            output.resize(rows(), cols());
                            output.adoptGpu(std::move(gpu_result._gpu), std::move(gpu_result._info));
                        }
                        else
                        {
                            output = std::move(gpu_result);
                        }
                    }
                    else
                    {
                        output = std::move(gpu_result);
                    }
                    return;
                }
            }
            plamatrix::internal::ScopedExecutionPolicy cpu_only(internal::ExecutionPolicy::CpuOnly);
            const std::size_t downloaded_bytes = pendingDownloadBytes();
            output.resize(rows(), cols());
            if constexpr (Rows == RowsAtCompileTime && Cols == ColsAtCompileTime && Options == Result::Options &&
                          MaxRows == Result::MaxRowsAtCompileTime && MaxCols == Result::MaxColsAtCompileTime)
            {
                _node.assignNoAlias(output);
            }
            else
            {
                const auto evaluator = _node.bind();
                for (Index col = 0; col < cols(); ++col)
                {
                    for (Index row = 0; row < rows(); ++row)
                    {
                        output(row, col) = evaluator(row, col);
                    }
                }
            }
            output._info.backend = internal::Backend::Cpu;
            output._info.bytesDownloaded = downloaded_bytes;
            output._info.reason = no_alias        ? "CPU noalias expression assignment"
                                  : supportsGpu() ? "Fused CPU expression evaluation"
                                                  : "CPU expression operation has no CUDA implementation";
        }

    private:
        const Result& cached() const
        {
            if (!_cache)
                _cache.emplace(eval());
            return *_cache;
        }
        Node _node;
        mutable std::optional<Result> _cache;
    };

    template <
        expression_detail::BinaryKind Kind,
        typename Left,
        typename Right,
        std::enable_if_t<expression_detail::isMatrixLike<Left> && expression_detail::isMatrixLike<Right>, int> = 0>
    auto makeBinaryExpression(Left&& left, Right&& right)
    {
        using LeftStorage = decltype(expression_detail::capture(std::forward<Left>(left)));
        using RightStorage = decltype(expression_detail::capture(std::forward<Right>(right)));
        using Node = expression_detail::BinaryNode<Kind, LeftStorage, RightStorage>;
        return DenseExpression<Node>(Node(expression_detail::capture(std::forward<Left>(left)),
                                          expression_detail::capture(std::forward<Right>(right))));
    }

    template <
        typename Left,
        typename Right,
        std::enable_if_t<expression_detail::isMatrixLike<Left> && expression_detail::isMatrixLike<Right>, int> = 0>
    auto operator+(Left&& left, Right&& right)
    {
        return makeBinaryExpression<expression_detail::BinaryKind::Add>(std::forward<Left>(left),
                                                                        std::forward<Right>(right));
    }

    template <
        typename Left,
        typename Right,
        std::enable_if_t<expression_detail::isMatrixLike<Left> && expression_detail::isMatrixLike<Right>, int> = 0>
    auto operator-(Left&& left, Right&& right)
    {
        return makeBinaryExpression<expression_detail::BinaryKind::Subtract>(std::forward<Left>(left),
                                                                             std::forward<Right>(right));
    }

    template <
        typename Left,
        typename Right,
        std::enable_if_t<expression_detail::isMatrixLike<Left> && expression_detail::isMatrixLike<Right>, int> = 0>
    auto operator*(Left&& left, Right&& right)
    {
        return makeBinaryExpression<expression_detail::BinaryKind::Product>(std::forward<Left>(left),
                                                                            std::forward<Right>(right));
    }

    template <typename Operand, std::enable_if_t<expression_detail::isMatrixLike<Operand>, int> = 0>
    auto scaleExpression(Operand&& operand, typename expression_detail::TypeTraits<Operand>::Value scale)
    {
        using Storage = decltype(expression_detail::capture(std::forward<Operand>(operand)));
        using Node = expression_detail::ScaleNode<Storage>;
        return DenseExpression<Node>(Node(expression_detail::capture(std::forward<Operand>(operand)), scale));
    }

    template <typename Operand, std::enable_if_t<expression_detail::isMatrixLike<Operand>, int> = 0>
    auto operator-(Operand&& operand)
    {
        using Scalar = typename expression_detail::TypeTraits<Operand>::Value;
        return scaleExpression(std::forward<Operand>(operand), Scalar{-1});
    }

    template <
        typename Operand,
        typename Scalar,
        std::enable_if_t<expression_detail::isMatrixLike<Operand> &&
                             std::is_convertible_v<Scalar, typename expression_detail::TypeTraits<Operand>::Value>,
                         int> = 0>
    auto operator*(Operand&& operand, Scalar scale)
    {
        using Value = typename expression_detail::TypeTraits<Operand>::Value;
        return scaleExpression(std::forward<Operand>(operand), static_cast<Value>(scale));
    }

    template <
        typename Scalar,
        typename Operand,
        std::enable_if_t<expression_detail::isMatrixLike<Operand> &&
                             std::is_convertible_v<Scalar, typename expression_detail::TypeTraits<Operand>::Value>,
                         int> = 0>
    auto operator*(Scalar scale, Operand&& operand)
    {
        using Value = typename expression_detail::TypeTraits<Operand>::Value;
        return scaleExpression(std::forward<Operand>(operand), static_cast<Value>(scale));
    }

    template <
        typename Operand,
        typename Scalar,
        std::enable_if_t<expression_detail::isMatrixLike<Operand> &&
                             std::is_convertible_v<Scalar, typename expression_detail::TypeTraits<Operand>::Value>,
                         int> = 0>
    auto operator/(Operand&& operand, Scalar divisor)
    {
        using Storage = decltype(expression_detail::capture(std::forward<Operand>(operand)));
        using Node = expression_detail::DivideNode<Storage>;
        using Value = typename expression_detail::TypeTraits<Operand>::Value;
        return DenseExpression<Node>(
            Node(expression_detail::capture(std::forward<Operand>(operand)), static_cast<Value>(divisor)));
    }

    template <typename Operand, typename Other> auto cwiseExpression(Operand&& operand, Other&& other)
    {
        return makeBinaryExpression<expression_detail::BinaryKind::CwiseProduct>(std::forward<Operand>(operand),
                                                                                 std::forward<Other>(other));
    }

} // namespace plamatrix::v1

#include "plamatrix/dense/detail/matrix_expression_array.h"
#include "plamatrix/dense/detail/matrix_expression_axis.h"

namespace plamatrix::v1
{
    template <typename Node,
              typename Scalar,
              std::enable_if_t<Node::ArraySemantics && std::is_convertible_v<Scalar, typename Node::Value>, int> = 0>
    auto operator+(DenseExpression<Node> expression, Scalar shift)
    {
        using Value = typename Node::Value;
        using ShiftNode = expression_detail::ArrayScalarNode<DenseExpression<Node>, false>;
        return DenseExpression<ShiftNode>(ShiftNode(std::move(expression), static_cast<Value>(shift)));
    }

    template <typename Scalar,
              typename Node,
              std::enable_if_t<Node::ArraySemantics && std::is_convertible_v<Scalar, typename Node::Value>, int> = 0>
    auto operator+(Scalar shift, DenseExpression<Node> expression)
    {
        return std::move(expression) + shift;
    }

    template <typename Node,
              typename Scalar,
              std::enable_if_t<Node::ArraySemantics && std::is_convertible_v<Scalar, typename Node::Value>, int> = 0>
    auto operator-(DenseExpression<Node> expression, Scalar shift)
    {
        using Value = typename Node::Value;
        using ShiftNode = expression_detail::ArrayScalarNode<DenseExpression<Node>, false>;
        return DenseExpression<ShiftNode>(ShiftNode(std::move(expression), -static_cast<Value>(shift)));
    }
    template <typename Operand> auto arrayExpression(Operand&& operand)
    {
        using Storage = decltype(expression_detail::capture(std::forward<Operand>(operand)));
        return ArrayExpressionProxy<Storage>(expression_detail::capture(std::forward<Operand>(operand)));
    }

    template <bool ByRows, typename Operand> auto axiswiseExpression(Operand&& operand)
    {
        using Storage = decltype(expression_detail::capture(std::forward<Operand>(operand)));
        return AxiswiseExpressionProxy<ByRows, Storage>(expression_detail::capture(std::forward<Operand>(operand)));
    }

} // namespace plamatrix::v1

#include "plamatrix/dense/detail/matrix_expression_facades.h"

namespace plamatrix::v1
{
    template <typename Node> std::ostream& operator<<(std::ostream& output, const DenseExpression<Node>& expression)
    {
        for (Index row = 0; row < expression.rows(); ++row)
        {
            if (row != 0)
                output << '\n';
            for (Index col = 0; col < expression.cols(); ++col)
            {
                if (col != 0)
                    output << ' ';
                output << expression(row, col);
            }
        }
        return output;
    }
} // namespace plamatrix::v1

namespace plamatrix::internal
{
    // Block views live in the implementation namespace; expose the expression operators to ADL.
    using plamatrix::v1::operator+;
    using plamatrix::v1::operator-;
    using plamatrix::v1::operator*;
    using plamatrix::v1::operator/;
} // namespace plamatrix::internal
