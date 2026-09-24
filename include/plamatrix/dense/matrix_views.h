#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace plamatrix::detail
{
    template <int Value> struct DiagonalSize
    {
        template <int Rows, int Cols> static constexpr int forShape()
        {
            if constexpr (Rows == Dynamic || Cols == Dynamic || Value == DynamicIndex)
            {
                return Dynamic;
            }
            else if constexpr (Value >= 0)
            {
                return Cols > Value ? std::min(Rows, Cols - Value) : 0;
            }
            else
            {
                return Rows > -Value ? std::min(Rows + Value, Cols) : 0;
            }
        }
    };

    template <typename MatrixType, int DiagIndex> struct DenseTraits<Diagonal<MatrixType, DiagIndex>>
    {
        using NestedTraits = DenseTraits<std::remove_const_t<MatrixType>>;
        using ScalarType = typename NestedTraits::ScalarType;
        static constexpr int RowsAtCompileTime =
            DiagonalSize<DiagIndex>::template forShape<NestedTraits::RowsAtCompileTime,
                                                       NestedTraits::ColsAtCompileTime>();
        static constexpr int ColsAtCompileTime = 1;
        static constexpr int MaxRowsAtCompileTime =
            DiagonalSize<DiagIndex>::template forShape<NestedTraits::MaxRowsAtCompileTime,
                                                       NestedTraits::MaxColsAtCompileTime>();
        static constexpr int MaxColsAtCompileTime = 1;
        static constexpr int StorageOptions = ColMajor;
        using PlainObject = Matrix<ScalarType, RowsAtCompileTime, 1, ColMajor, MaxRowsAtCompileTime, 1>;
    };

    template <typename DiagonalVectorType> struct DenseTraits<DiagonalWrapper<DiagonalVectorType>>
    {
        using NestedTraits = DenseTraits<std::remove_const_t<DiagonalVectorType>>;
        using ScalarType = typename NestedTraits::ScalarType;
        static constexpr int VectorSize =
            NestedTraits::RowsAtCompileTime == 1 ? NestedTraits::ColsAtCompileTime : NestedTraits::RowsAtCompileTime;
        static constexpr int MaxVectorSize = NestedTraits::MaxRowsAtCompileTime == 1
                                                 ? NestedTraits::MaxColsAtCompileTime
                                                 : NestedTraits::MaxRowsAtCompileTime;
        static constexpr int RowsAtCompileTime = VectorSize;
        static constexpr int ColsAtCompileTime = VectorSize;
        static constexpr int MaxRowsAtCompileTime = MaxVectorSize;
        static constexpr int MaxColsAtCompileTime = MaxVectorSize;
        static constexpr int StorageOptions = ColMajor;
        using PlainObject = Matrix<ScalarType,
                                   RowsAtCompileTime,
                                   ColsAtCompileTime,
                                   ColMajor,
                                   MaxRowsAtCompileTime,
                                   MaxColsAtCompileTime>;
    };

    template <typename MatrixType, unsigned int Mode>
    struct DenseTraits<TriangularView<MatrixType, Mode>> : DenseTraits<std::remove_const_t<MatrixType>>
    {
    };

    template <typename MatrixType, unsigned int UpLo>
    struct DenseTraits<SelfAdjointView<MatrixType, UpLo>> : DenseTraits<std::remove_const_t<MatrixType>>
    {
    };

    template <typename XprType, int Rows, int Cols, int Order> struct DenseTraits<Reshaped<XprType, Rows, Cols, Order>>
    {
        using NestedTraits = DenseTraits<std::remove_const_t<XprType>>;
        using ScalarType = typename NestedTraits::ScalarType;
        static constexpr int RowsAtCompileTime = Rows;
        static constexpr int ColsAtCompileTime = Cols;
        static constexpr int MaxRowsAtCompileTime = Rows;
        static constexpr int MaxColsAtCompileTime = Cols;
        static constexpr int StorageOptions = Order == RowMajor ? RowMajor : ColMajor;
        using PlainObject = Matrix<ScalarType, Rows, Cols, StorageOptions>;
    };

    template <int Size, int Factor> struct ReplicatedSize
    {
        static constexpr int value = Size == Dynamic || Factor == Dynamic ? Dynamic : Size * Factor;
    };

    template <typename MatrixType, int RowFactor, int ColFactor>
    struct DenseTraits<Replicate<MatrixType, RowFactor, ColFactor>>
    {
        using NestedTraits = DenseTraits<std::remove_const_t<MatrixType>>;
        using ScalarType = typename NestedTraits::ScalarType;
        static constexpr int RowsAtCompileTime = ReplicatedSize<NestedTraits::RowsAtCompileTime, RowFactor>::value;
        static constexpr int ColsAtCompileTime = ReplicatedSize<NestedTraits::ColsAtCompileTime, ColFactor>::value;
        static constexpr int MaxRowsAtCompileTime =
            ReplicatedSize<NestedTraits::MaxRowsAtCompileTime, RowFactor>::value;
        static constexpr int MaxColsAtCompileTime =
            ReplicatedSize<NestedTraits::MaxColsAtCompileTime, ColFactor>::value;
        static constexpr int StorageOptions = NestedTraits::StorageOptions;
        using PlainObject = Matrix<ScalarType,
                                   RowsAtCompileTime,
                                   ColsAtCompileTime,
                                   StorageOptions,
                                   MaxRowsAtCompileTime,
                                   MaxColsAtCompileTime>;
    };

    template <typename MatrixType, int Direction>
    struct DenseTraits<Reverse<MatrixType, Direction>> : DenseTraits<std::remove_const_t<MatrixType>>
    {
    };
} // namespace plamatrix::detail

namespace plamatrix::v1
{
    namespace view_detail
    {
        template <typename Expression>
        using CoefficientReference =
            decltype(std::declval<Expression&>()(std::declval<plamatrix::Index>(), std::declval<plamatrix::Index>()));

        template <typename Expression>
        inline constexpr bool IsReadOnly = !std::is_lvalue_reference_v<CoefficientReference<Expression>> ||
                                           std::is_const_v<std::remove_reference_t<CoefficientReference<Expression>>>;

        template <typename Derived> class CommonDenseView : public MatrixBase<Derived>
        {
        public:
            using Base = MatrixBase<Derived>;
            using Traits = detail::DenseTraits<Derived>;
            using Scalar = typename Traits::ScalarType;
            using Base::isApprox;
            using Base::maxCoeff;
            using Base::minCoeff;
            using RealScalar = typename NumTraits<Scalar>::Real;
            using PlainObject = typename Traits::PlainObject;
            using Index = plamatrix::Index;

            PlainObject eval() const
            {
                const Derived& self = derivedView();
                PlainObject result(self.rows(), self.cols());
                for (Index col = 0; col < self.cols(); ++col)
                {
                    for (Index row = 0; row < self.rows(); ++row)
                    {
                        result(row, col) = self(row, col);
                    }
                }
                return result;
            }

            Scalar sum() const
            {
                Scalar result{};
                forEach([&](const Scalar& value) { result += value; });
                return result;
            }

            Scalar minCoeff() const
            {
                requireNonempty("minCoeff");
                Scalar result = derivedView()(0, 0);
                forEach([&](const Scalar& value) { result = std::min(result, value); });
                return result;
            }

            Scalar maxCoeff() const
            {
                requireNonempty("maxCoeff");
                Scalar result = derivedView()(0, 0);
                forEach([&](const Scalar& value) { result = std::max(result, value); });
                return result;
            }

            Scalar squaredNorm() const
            {
                Scalar result{};
                forEach([&](const Scalar& value) { result += value * value; });
                return result;
            }

            Scalar norm() const
            {
                return static_cast<Scalar>(std::sqrt(squaredNorm()));
            }

            Scalar trace() const
            {
                Scalar result{};
                for (Index index = 0; index < std::min(derivedView().rows(), derivedView().cols()); ++index)
                {
                    result += derivedView()(index, index);
                }
                return result;
            }

            bool allFinite() const
            {
                bool result = true;
                forEach([&](const Scalar& value) { result = result && std::isfinite(value); });
                return result;
            }

            template <typename Other> Scalar dot(const Other& other) const
            {
                const Derived& self = derivedView();
                if ((self.rows() != 1 && self.cols() != 1) || (other.rows() != 1 && other.cols() != 1) ||
                    self.size() != other.size())
                {
                    throw internal::Error(internal::ErrorCode::InvalidArgument,
                                          "dot() requires vectors of equal length");
                }
                Scalar result{};
                for (Index index = 0; index < self.size(); ++index)
                {
                    result += self(index) * other(index);
                }
                return result;
            }

        private:
            const Derived& derivedView() const noexcept
            {
                return *static_cast<const Derived*>(this);
            }

            template <typename Function> void forEach(Function&& function) const
            {
                const Derived& self = derivedView();
                for (Index col = 0; col < self.cols(); ++col)
                {
                    for (Index row = 0; row < self.rows(); ++row)
                    {
                        function(static_cast<Scalar>(self(row, col)));
                    }
                }
            }

            void requireNonempty(const char* operation) const
            {
                if (derivedView().size() == 0)
                {
                    throw internal::Error(internal::ErrorCode::InvalidArgument,
                                          std::string(operation) + "() requires a nonempty expression");
                }
            }
        };

        template <typename Type, typename = void> struct HasResize : std::false_type
        {
        };

        template <typename Type>
        struct HasResize<Type,
                         std::void_t<decltype(std::declval<Type&>().resize(
                             std::declval<plamatrix::Index>(), std::declval<plamatrix::Index>()))>> : std::true_type
        {
        };
    } // namespace view_detail

} // namespace plamatrix::v1

#include "plamatrix/dense/detail/matrix_vector_diagonal.h"
#include "plamatrix/dense/detail/matrix_structured_views.h"
#include "plamatrix/dense/detail/matrix_reorder_views.h"

#include "plamatrix/dense/detail/matrix_view_methods.h"
