#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "plamatrix/internal/dense/packet_reduction.h"

namespace plamatrix::v1
{
    namespace expression_detail
    {
        template <typename Type, typename = void> struct Traits
        {
            static constexpr bool valid = false;
        };

        template <typename Type>
        struct Traits<Type, std::void_t<typename ::plamatrix::detail::DenseTraits<std::remove_cv_t<Type>>::ScalarType>>
        {
            using Dense = ::plamatrix::detail::DenseTraits<std::remove_cv_t<Type>>;
            using Value = typename Dense::ScalarType;
            static constexpr int rows = Dense::RowsAtCompileTime;
            static constexpr int cols = Dense::ColsAtCompileTime;
            static constexpr bool valid = true;
            static constexpr bool view = true;
        };

        template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
        struct Traits<Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>>
        {
            using Value = Scalar;
            static constexpr int rows = Rows;
            static constexpr int cols = Cols;
            static constexpr bool valid = true;
            static constexpr bool view = false;
        };

        template <typename Scalar, bool IsConst>
        struct Traits<internal::BasicMatrixView<Scalar, internal::Device::CPU, IsConst>>
        {
            using Value = Scalar;
            static constexpr int rows = Dynamic;
            static constexpr int cols = Dynamic;
            static constexpr bool valid = true;
            static constexpr bool view = true;
        };

        template <typename Scalar, int Rows, int Cols, bool IsConst>
        struct Traits<internal::FixedMatrixView<Scalar, Rows, Cols, IsConst>>
        {
            using Value = Scalar;
            static constexpr int rows = Rows;
            static constexpr int cols = Cols;
            static constexpr bool valid = true;
            static constexpr bool view = true;
        };

        template <typename Scalar,
                  int Rows,
                  int Cols,
                  int Options,
                  int MaxRows,
                  int MaxCols,
                  int MapOptions,
                  typename StrideType>
        struct Traits<Map<Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>, MapOptions, StrideType>>
        {
            using Value = Scalar;
            static constexpr int rows = Rows;
            static constexpr int cols = Cols;
            static constexpr bool valid = true;
            static constexpr bool view = true;
        };

        template <typename Scalar,
                  int Rows,
                  int Cols,
                  int Options,
                  int MaxRows,
                  int MaxCols,
                  int MapOptions,
                  typename StrideType>
        struct Traits<Map<const Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>, MapOptions, StrideType>>
        {
            using Value = Scalar;
            static constexpr int rows = Rows;
            static constexpr int cols = Cols;
            static constexpr bool valid = true;
            static constexpr bool view = true;
        };

        template <typename PlainObjectType, int Options, typename StrideType>
        struct Traits<Ref<PlainObjectType, Options, StrideType>>
        {
            using PlainTraits = ::plamatrix::detail::DenseTraits<std::remove_const_t<PlainObjectType>>;
            using Value = typename PlainTraits::ScalarType;
            static constexpr int rows = PlainTraits::RowsAtCompileTime;
            static constexpr int cols = PlainTraits::ColsAtCompileTime;
            static constexpr bool valid = true;
            static constexpr bool view = true;
        };

        template <typename XprType, int BlockRows, int BlockCols, bool InnerPanel>
        struct Traits<Block<XprType, BlockRows, BlockCols, InnerPanel>>
        {
            using BlockTraits = ::plamatrix::detail::DenseTraits<Block<XprType, BlockRows, BlockCols, InnerPanel>>;
            using Value = typename BlockTraits::ScalarType;
            static constexpr int rows = BlockTraits::RowsAtCompileTime;
            static constexpr int cols = BlockTraits::ColsAtCompileTime;
            static constexpr bool valid = true;
            static constexpr bool view = true;
        };

        template <typename MatrixType> struct Traits<Transpose<MatrixType>>
        {
            using TransposeTraits = ::plamatrix::detail::DenseTraits<Transpose<MatrixType>>;
            using Value = typename TransposeTraits::ScalarType;
            static constexpr int rows = TransposeTraits::RowsAtCompileTime;
            static constexpr int cols = TransposeTraits::ColsAtCompileTime;
            static constexpr bool valid = true;
            static constexpr bool view = true;
        };

        template <typename Node> struct Traits<DenseExpression<Node>>
        {
            using Value = typename Node::Value;
            static constexpr int rows = Node::RowsAtCompileTime;
            static constexpr int cols = Node::ColsAtCompileTime;
            static constexpr bool valid = true;
            static constexpr bool view = false;
        };

        template <typename Type> using TypeTraits = Traits<std::decay_t<Type>>;
        template <typename Type> constexpr bool isMatrixLike = TypeTraits<Type>::valid;

        template <typename Type> struct LinearAccess : std::false_type
        {
        };
        template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
        struct LinearAccess<Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>>
            : std::bool_constant<(Options & RowMajor) == 0>
        {
        };
        template <typename Node, typename = void> struct NodeLinearAccess : std::false_type
        {
        };
        template <typename Node>
        struct NodeLinearAccess<Node, std::void_t<decltype(Node::LinearAccess)>>
            : std::bool_constant<Node::LinearAccess>
        {
        };
        template <typename Node> struct LinearAccess<DenseExpression<Node>> : NodeLinearAccess<Node>
        {
        };
        template <typename Type> constexpr bool hasLinearAccess = LinearAccess<std::decay_t<Type>>::value;

        template <typename Node, typename = void> struct NodeSimdSafe : std::false_type
        {
        };
        template <typename Node>
        struct NodeSimdSafe<Node, std::void_t<decltype(Node::SimdSafe)>> : std::bool_constant<Node::SimdSafe>
        {
        };
        template <typename Type> struct SimdSafe : std::false_type
        {
        };
        template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
        struct SimdSafe<Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>> : std::true_type
        {
        };
        template <typename Node> struct SimdSafe<DenseExpression<Node>> : NodeSimdSafe<Node>
        {
        };
        template <typename Type> constexpr bool isSimdSafe = SimdSafe<std::decay_t<Type>>::value;

        template <typename Type> struct DirectDataAccess : std::false_type
        {
        };
        template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
        struct DirectDataAccess<Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>>
            : std::bool_constant<(Options & RowMajor) == 0>
        {
        };
        template <typename Node, typename = void> struct NodeDirectDataAccess : std::false_type
        {
        };
        template <typename Node>
        struct NodeDirectDataAccess<Node, std::void_t<decltype(Node::DirectDataAccess)>>
            : std::bool_constant<Node::DirectDataAccess>
        {
        };
        template <typename Node> struct DirectDataAccess<DenseExpression<Node>> : NodeDirectDataAccess<Node>
        {
        };
        template <typename Type> constexpr bool hasDirectDataAccess = DirectDataAccess<std::decay_t<Type>>::value;

        template <typename Type, typename = void>
        struct PacketReductionTermCount : std::integral_constant<std::size_t, 0>
        {
        };
        template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
        struct PacketReductionTermCount<Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>>
            : std::integral_constant<
                  std::size_t,
                  (Options & RowMajor) == 0 && (std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double>) ? 1
                                                                                                                 : 0>
        {
        };
        template <typename Node, typename = void> struct NodePacketTermCount : std::integral_constant<std::size_t, 0>
        {
        };
        template <typename Node>
        struct NodePacketTermCount<Node, std::void_t<decltype(Node::PacketTermCount)>>
            : std::integral_constant<std::size_t, Node::PacketTermCount>
        {
        };
        template <typename Node> struct PacketReductionTermCount<DenseExpression<Node>> : NodePacketTermCount<Node>
        {
        };
        template <typename Type>
        constexpr std::size_t packetReductionTermCount = PacketReductionTermCount<std::decay_t<Type>>::value;
        template <typename Node> struct NodePacketSum : std::bool_constant<(NodePacketTermCount<Node>::value > 0)>
        {
        };

        template <typename Type, typename = void> struct ArraySyntax : std::false_type
        {
        };
        template <typename Node>
        struct ArraySyntax<DenseExpression<Node>, std::void_t<decltype(Node::ArraySemantics)>>
            : std::bool_constant<Node::ArraySemantics>
        {
        };

        template <typename Type> auto capture(Type&& operand)
        {
            if constexpr (TypeTraits<Type>::view || !std::is_lvalue_reference_v<Type>)
            {
                return std::forward<Type>(operand);
            }
            else
            {
                return std::cref(operand);
            }
        }

        template <typename Type> const Type& unwrap(const Type& operand)
        {
            return operand;
        }
        template <typename Type> const Type& unwrap(const std::reference_wrapper<const Type>& operand)
        {
            return operand.get();
        }

        template <typename Type> bool resident(const Type&)
        {
            return false;
        }
        template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
        bool resident(const Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>& value)
        {
            return internal::MatrixAccess::isDeviceResident(value);
        }
        template <typename Node> bool resident(const DenseExpression<Node>& value)
        {
            return value.hasResidentInput();
        }

        template <typename Type> std::size_t pendingDownload(const Type&)
        {
            return 0;
        }
        template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
        std::size_t pendingDownload(const Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>& value)
        {
            return internal::MatrixAccess::pendingHostDownloadBytes(value);
        }
        template <typename Node> std::size_t pendingDownload(const DenseExpression<Node>& value)
        {
            return value.pendingDownloadBytes();
        }

        template <typename Type> bool hasProduct(const Type&)
        {
            return false;
        }
        template <typename Node> bool hasProduct(const DenseExpression<Node>& value)
        {
            return value.hasProduct();
        }

        template <typename Type> bool supportsGpu(const Type&)
        {
            return true;
        }
        template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
        bool supportsGpu(const Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>&)
        {
            return Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>::IsRowMajor == 0;
        }
        template <typename Node> bool supportsGpu(const DenseExpression<Node>& value)
        {
            return value.supportsGpu();
        }

        template <typename Type> long double work(const Type& value)
        {
            return static_cast<long double>(value.rows()) * value.cols();
        }
        template <typename Node> long double work(const DenseExpression<Node>& value)
        {
            return value.work();
        }

        template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
        const Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>&
        materialize(const Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>& value)
        {
            return value;
        }

        template <typename Scalar, bool IsConst>
        Matrix<Scalar, Dynamic, Dynamic>
        materialize(const internal::BasicMatrixView<Scalar, internal::Device::CPU, IsConst>& value)
        {
            return Matrix<Scalar, Dynamic, Dynamic>(value);
        }

        template <typename Scalar, int Rows, int Cols, bool IsConst>
        Matrix<Scalar, Rows, Cols> materialize(const internal::FixedMatrixView<Scalar, Rows, Cols, IsConst>& value)
        {
            return Matrix<Scalar, Rows, Cols>(value);
        }

        template <typename MatrixType, int MapOptions, typename StrideType>
        auto materialize(const Map<MatrixType, MapOptions, StrideType>& value)
        {
            using MapTraits = Traits<Map<MatrixType, MapOptions, StrideType>>;
            Matrix<typename MapTraits::Value, MapTraits::rows, MapTraits::cols> result(value.rows(), value.cols());
            for (Index col = 0; col < value.cols(); ++col)
            {
                for (Index row = 0; row < value.rows(); ++row)
                {
                    result(row, col) = value(row, col);
                }
            }
            return result;
        }

        template <typename PlainObjectType, int Options, typename StrideType>
        auto materialize(const Ref<PlainObjectType, Options, StrideType>& value)
        {
            using RefTraits = Traits<Ref<PlainObjectType, Options, StrideType>>;
            Matrix<typename RefTraits::Value, RefTraits::rows, RefTraits::cols> result(value.rows(), value.cols());
            for (Index col = 0; col < value.cols(); ++col)
            {
                for (Index row = 0; row < value.rows(); ++row)
                {
                    result(row, col) = value(row, col);
                }
            }
            return result;
        }

        template <typename XprType, int BlockRows, int BlockCols, bool InnerPanel>
        auto materialize(const Block<XprType, BlockRows, BlockCols, InnerPanel>& value)
        {
            return value.eval();
        }

        template <typename MatrixType> auto materialize(const Transpose<MatrixType>& value)
        {
            return value.eval();
        }

        template <typename Node> auto materialize(const DenseExpression<Node>& value)
        {
            return value.eval();
        }

        template <typename Derived> auto materialize(const MatrixBase<Derived>& value)
        {
            return value.derived().eval();
        }

        template <typename Type> auto coefficient(const Type& value, Index row, Index col)
        {
            return value(row, col);
        }
        template <typename Node> auto coefficient(const DenseExpression<Node>& value, Index row, Index col)
        {
            return value.coeff(row, col);
        }

        template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
        Scalar linearCoefficient(const Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>& value, Index index)
        {
            static_assert((Options & RowMajor) == 0, "linearCoefficient requires column-major storage");
            return value.data()[index];
        }
        template <typename Node> auto linearCoefficient(const DenseExpression<Node>& value, Index index)
        {
            return value.linearCoeff(index);
        }

        template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
        auto linearBind(const Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>& value)
        {
            static_assert((Options & RowMajor) == 0, "linearBind requires column-major storage");
            const Scalar* data = value.data();
            return [data](Index index) { return data[index]; };
        }
        template <typename Node> auto linearBind(const DenseExpression<Node>& value)
        {
            return value.bindLinear();
        }

        template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
        const Scalar* directData(const Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>& value)
        {
            static_assert((Options & RowMajor) == 0, "directData requires column-major storage");
            return value.data();
        }
        template <typename Node> auto directData(const DenseExpression<Node>& value)
        {
            return value.directData();
        }

        template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
        auto packetReductionTerms(const Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>& value)
        {
            static_assert(packetReductionTermCount<Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>> == 1,
                          "packetReductionTerms requires a contiguous float or double matrix");
            using Term = internal::detail::PacketReductionTerm<Scalar>;
            return std::array<Term, 1>{{Term{value.data(), nullptr, Scalar{1}}}};
        }

        template <typename Node> auto packetReductionTerms(const DenseExpression<Node>& value)
        {
            static_assert(NodePacketTermCount<Node>::value > 0,
                          "packetReductionTerms requires a packet-reducible expression");
            return value.packetReductionTerms();
        }

        template <typename Scalar, std::size_t Count>
        auto scalePacketReductionTerms(std::array<internal::detail::PacketReductionTerm<Scalar>, Count> terms,
                                       Scalar scale)
        {
            for (auto& term : terms)
                term.scale *= scale;
            return terms;
        }

        template <typename Scalar, std::size_t LeftCount, std::size_t RightCount>
        auto concatenatePacketReductionTerms(
            const std::array<internal::detail::PacketReductionTerm<Scalar>, LeftCount>& left,
            std::array<internal::detail::PacketReductionTerm<Scalar>, RightCount> right,
            Scalar right_scale)
        {
            std::array<internal::detail::PacketReductionTerm<Scalar>, LeftCount + RightCount> result{};
            std::copy(left.begin(), left.end(), result.begin());
            for (std::size_t index = 0; index < RightCount; ++index)
            {
                right[index].scale *= right_scale;
                result[LeftCount + index] = right[index];
            }
            return result;
        }

        template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
        auto bind(const Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>& value)
        {
            const Scalar* data = value.data();
            const Index rows = value.rows();
            const Index cols = value.cols();
            return [data, rows, cols](Index row, Index col)
            {
                if constexpr (Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>::IsRowMajor)
                {
                    return data[col + row * cols];
                }
                else
                {
                    return data[row + col * rows];
                }
            };
        }

        template <typename Scalar, bool IsConst>
        auto bind(const internal::BasicMatrixView<Scalar, internal::Device::CPU, IsConst>& value)
        {
            const Scalar* data = value.data();
            const Index row_stride = value.rowStride();
            const Index col_stride = value.colStride();
            return [data, row_stride, col_stride](Index row, Index col)
            { return data[row * row_stride + col * col_stride]; };
        }

        template <typename Scalar, int Rows, int Cols, bool IsConst>
        auto bind(const internal::FixedMatrixView<Scalar, Rows, Cols, IsConst>& value)
        {
            return bind(static_cast<const internal::BasicMatrixView<Scalar, internal::Device::CPU, IsConst>&>(value));
        }

        template <typename MatrixType, int MapOptions, typename StrideType>
        auto bind(const Map<MatrixType, MapOptions, StrideType>& value)
        {
            return [value](Index row, Index col) { return value(row, col); };
        }

        template <typename PlainObjectType, int Options, typename StrideType>
        auto bind(const Ref<PlainObjectType, Options, StrideType>& value)
        {
            return [value](Index row, Index col) { return value(row, col); };
        }

        template <typename XprType, int BlockRows, int BlockCols, bool InnerPanel>
        auto bind(const Block<XprType, BlockRows, BlockCols, InnerPanel>& value)
        {
            return [value](Index row, Index col) { return value(row, col); };
        }

        template <typename MatrixType> auto bind(const Transpose<MatrixType>& value)
        {
            return [value](Index row, Index col) { return value(row, col); };
        }

        template <typename Node> auto bind(const DenseExpression<Node>& value)
        {
            return value.bind();
        }

        template <typename Derived> auto bind(const MatrixBase<Derived>& value)
        {
            return [expression = value.derived()](Index row, Index col) { return expression(row, col); };
        }

    } // namespace expression_detail
} // namespace plamatrix::v1
