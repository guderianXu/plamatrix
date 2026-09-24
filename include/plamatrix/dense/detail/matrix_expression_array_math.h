#pragma once

#include <cmath>
#include <limits>

namespace plamatrix::v1::expression_detail
{
    enum class ArrayUnaryKind
    {
        Abs,
        Sqrt,
        Exp,
        Log
    };

    template <ArrayUnaryKind Kind, typename Storage> class ArrayMathNode
    {
        using Source = std::decay_t<decltype(unwrap(std::declval<const Storage&>()))>;
        using SourceTraits = Traits<Source>;

    public:
        using Value = typename SourceTraits::Value;
        static constexpr int RowsAtCompileTime = SourceTraits::rows;
        static constexpr int ColsAtCompileTime = SourceTraits::cols;
        static constexpr bool ArraySemantics = true;
        using Result = Matrix<Value, RowsAtCompileTime, ColsAtCompileTime>;

        explicit ArrayMathNode(Storage source) : _source(std::move(source))
        {
            static_assert(std::is_floating_point_v<Value> || (Kind == ArrayUnaryKind::Abs && std::is_signed_v<Value>),
                          "Array math requires floating-point coefficients, except signed integer abs()");
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
            return compute(coefficient(value(), row, col));
        }
        auto bind() const
        {
            const auto source = expression_detail::bind(value());
            return [source](Index row, Index col) { return compute(source(row, col)); };
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
        static Value compute(Value source)
        {
            if constexpr (Kind == ArrayUnaryKind::Abs)
            {
                if constexpr (std::is_integral_v<Value>)
                    if (source == std::numeric_limits<Value>::min())
                        throw internal::Error(internal::ErrorCode::InvalidArgument, "abs() cannot represent the minimum signed integer");
                return static_cast<Value>(std::abs(source));
            }
            else if constexpr (Kind == ArrayUnaryKind::Sqrt)
                return static_cast<Value>(std::sqrt(source));
            else if constexpr (Kind == ArrayUnaryKind::Exp)
                return static_cast<Value>(std::exp(source));
            else
                return static_cast<Value>(std::log(source));
        }

        const Source& value() const
        {
            return unwrap(_source);
        }
        Storage _source;
    };
} // namespace plamatrix::v1::expression_detail
