#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace plamatrix::v1
{
    template <int OuterStrideAtCompileTime_, int InnerStrideAtCompileTime_> class Stride
    {
    public:
        using Index = plamatrix::Index;
        static constexpr int OuterStrideAtCompileTime = OuterStrideAtCompileTime_;
        static constexpr int InnerStrideAtCompileTime = InnerStrideAtCompileTime_;

        Stride() : _outer(OuterStrideAtCompileTime), _inner(InnerStrideAtCompileTime)
        {
            if (OuterStrideAtCompileTime == Dynamic || InnerStrideAtCompileTime == Dynamic)
            {
                throw internal::Error(internal::ErrorCode::InvalidArgument,
                                      "Dynamic Map strides require runtime values");
            }
        }

        Stride(Index outer, Index inner) : _outer(outer), _inner(inner)
        {
            if ((OuterStrideAtCompileTime != Dynamic && outer != OuterStrideAtCompileTime) ||
                (InnerStrideAtCompileTime != Dynamic && inner != InnerStrideAtCompileTime))
            {
                throw internal::Error(internal::ErrorCode::InvalidArgument,
                                      "Map stride conflicts with its fixed value");
            }
        }

        Index outer() const noexcept
        {
            return _outer;
        }

        Index inner() const noexcept
        {
            return _inner;
        }

    protected:
        Index _outer;
        Index _inner;
    };

    template <int Value> class InnerStride : public Stride<0, Value>
    {
        using Base = Stride<0, Value>;

    public:
        InnerStride() : Base()
        {
        }

        InnerStride(Index value) : Base(0, value)
        {
        }
    };

    template <int Value> class OuterStride : public Stride<Value, 0>
    {
        using Base = Stride<Value, 0>;

    public:
        OuterStride() : Base()
        {
        }

        OuterStride(Index value) : Base(value, 0)
        {
        }
    };

    namespace map_detail
    {
        template <typename MatrixType> struct MapMatrixTraits;

        template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
        struct MapMatrixTraits<Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>>
        {
            using ScalarType = Scalar;
            static constexpr int RowsAtCompileTime = Rows;
            static constexpr int ColsAtCompileTime = Cols;
            static constexpr bool IsRowMajor = (Options & RowMajor) == RowMajor;
        };

        template <typename MatrixType> struct MapMatrixTraits<const MatrixType> : MapMatrixTraits<MatrixType>
        {
        };

        template <typename Type> struct IsOwningMatrix : std::false_type
        {
        };

        template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
        struct IsOwningMatrix<Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>> : std::true_type
        {
        };

        template <typename StrideType> struct RefStrideFactory;

        template <int Outer, int Inner> struct RefStrideFactory<Stride<Outer, Inner>>
        {
            static Stride<Outer, Inner> make(Index outer, Index inner)
            {
                return Stride<Outer, Inner>(outer, inner);
            }
        };

        template <int Value> struct RefStrideFactory<InnerStride<Value>>
        {
            static InnerStride<Value> make(Index, Index inner)
            {
                return InnerStride<Value>(inner);
            }
        };

        template <int Value> struct RefStrideFactory<OuterStride<Value>>
        {
            static OuterStride<Value> make(Index outer, Index)
            {
                return OuterStride<Value>(outer);
            }
        };
    } // namespace map_detail

    /// Eigen-style non-owning dense view over caller-owned CPU storage.
    template <typename MatrixType, int MapOptions, typename StrideType>
    class Map : public MatrixBase<Map<MatrixType, MapOptions, StrideType>>
    {
        using Traits = map_detail::MapMatrixTraits<MatrixType>;
        using ScalarValue = typename Traits::ScalarType;
        static constexpr bool read_only = std::is_const_v<MatrixType>;
        using View =
            internal::FixedMatrixView<ScalarValue, Traits::RowsAtCompileTime, Traits::ColsAtCompileTime, read_only>;

    public:
        using Base = MatrixBase<Map>;
        using Scalar = ScalarValue;
        using Base::isApprox;
        using Base::maxCoeff;
        using Base::minCoeff;
        using Pointer = std::conditional_t<read_only, const Scalar*, Scalar*>;
        using Reference = std::conditional_t<read_only, const Scalar&, Scalar&>;
        using PlainObject = std::remove_const_t<MatrixType>;
        static constexpr int RowsAtCompileTime = Traits::RowsAtCompileTime;
        static constexpr int ColsAtCompileTime = Traits::ColsAtCompileTime;

        template <int R = RowsAtCompileTime,
                  int C = ColsAtCompileTime,
                  std::enable_if_t<R != Dynamic && C != Dynamic, int> = 0>
        explicit Map(Pointer data) : _view(makeContiguousView(data, RowsAtCompileTime, ColsAtCompileTime))
        {
        }

        template <int R = RowsAtCompileTime,
                  int C = ColsAtCompileTime,
                  std::enable_if_t<(R == 1 || C == 1) && (R == Dynamic || C == Dynamic), int> = 0>
        Map(Pointer data, Index length) : _view(makeContiguousView(data, R == 1 ? 1 : length, C == 1 ? 1 : length))
        {
        }

        template <int R = RowsAtCompileTime,
                  int C = ColsAtCompileTime,
                  std::enable_if_t<(R == 1 || C == 1) && (R == Dynamic || C == Dynamic), int> = 0>
        Map(Pointer data, Index length, const StrideType& stride)
            : _view(makeStridedView(data, R == 1 ? 1 : length, C == 1 ? 1 : length, stride))
        {
        }

        Map(Pointer data, Index rows, Index cols) : _view(makeContiguousView(data, rows, cols))
        {
        }

        Map(Pointer data, Index rows, Index cols, const StrideType& stride)
            : _view(makeStridedView(data, rows, cols, stride))
        {
        }

        Map(const Map&) = default;

        Map& operator=(const Map& other)
        {
            if constexpr (read_only)
            {
                _view = other._view;
            }
            else
            {
                _view.operator=(other._view);
            }
            return *this;
        }

        template <typename Other, bool Enabled = !read_only, std::enable_if_t<Enabled, int> = 0>
        Map& operator=(const Other& other)
        {
            _view.operator=(other);
            return *this;
        }

        Index rows() const noexcept
        {
            return _view.rows();
        }

        Index cols() const noexcept
        {
            return _view.cols();
        }

        Index size() const noexcept
        {
            return _view.size();
        }

        Index innerStride() const noexcept
        {
            return Traits::IsRowMajor ? _view.colStride() : _view.rowStride();
        }

        Index outerStride() const noexcept
        {
            return Traits::IsRowMajor ? _view.rowStride() : _view.colStride();
        }

        Pointer data() const noexcept
        {
            return _view.data();
        }

        Reference operator()(Index row, Index col) const
        {
            return _view(row, col);
        }

        Reference operator()(Index index) const
        {
            return _view(index);
        }

        const Scalar& coeff(Index row, Index col) const
        {
            return _view(row, col);
        }

        template <bool Enabled = !read_only, std::enable_if_t<Enabled, int> = 0> Scalar& coeffRef(Index row, Index col)
        {
            return _view(row, col);
        }

        Block<Map, Dynamic, Dynamic> block(Index row, Index col, Index rows, Index cols)
        {
            return Block<Map, Dynamic, Dynamic>(*this, row, col, rows, cols);
        }

        const Block<const Map, Dynamic, Dynamic> block(Index row, Index col, Index rows, Index cols) const
        {
            return Block<const Map, Dynamic, Dynamic>(*this, row, col, rows, cols);
        }

        template <int Rows, int Cols> Block<Map, Rows, Cols> block(Index row, Index col)
        {
            return Block<Map, Rows, Cols>(*this, row, col);
        }

        template <int Rows, int Cols> const Block<const Map, Rows, Cols> block(Index row, Index col) const
        {
            return Block<const Map, Rows, Cols>(*this, row, col);
        }

        Block<Map, 1, ColsAtCompileTime, Traits::IsRowMajor> row(Index index)
        {
            return Block<Map, 1, ColsAtCompileTime, Traits::IsRowMajor>(*this, index);
        }

        const Block<const Map, 1, ColsAtCompileTime, Traits::IsRowMajor> row(Index index) const
        {
            return Block<const Map, 1, ColsAtCompileTime, Traits::IsRowMajor>(*this, index);
        }

        Block<Map, RowsAtCompileTime, 1, !Traits::IsRowMajor> col(Index index)
        {
            return Block<Map, RowsAtCompileTime, 1, !Traits::IsRowMajor>(*this, index);
        }

        const Block<const Map, RowsAtCompileTime, 1, !Traits::IsRowMajor> col(Index index) const
        {
            return Block<const Map, RowsAtCompileTime, 1, !Traits::IsRowMajor>(*this, index);
        }

        Transpose<Map> transpose()
        {
            return Transpose<Map>(*this);
        }

        const Transpose<const Map> transpose() const
        {
            return Transpose<const Map>(*this);
        }

        template <typename Other> Scalar dot(const Other& other) const
        {
            if ((rows() != 1 && cols() != 1) || (other.rows() != 1 && other.cols() != 1) || size() != other.size())
            {
                throw internal::Error(internal::ErrorCode::InvalidArgument,
                                      "Map dot() requires vectors of equal length");
            }
            Scalar result{};
            for (Index index = 0; index < size(); ++index)
            {
                result += (*this)(index)*other(index);
            }
            return result;
        }

        Scalar squaredNorm() const
        {
            Scalar result{};
            for (Index col_index = 0; col_index < cols(); ++col_index)
            {
                for (Index row_index = 0; row_index < rows(); ++row_index)
                {
                    const Scalar value = (*this)(row_index, col_index);
                    result += value * value;
                }
            }
            return result;
        }

        Scalar norm() const
        {
            return static_cast<Scalar>(std::sqrt(squaredNorm()));
        }

        Scalar sum() const
        {
            Scalar result{};
            for (Index col_index = 0; col_index < cols(); ++col_index)
            {
                for (Index row_index = 0; row_index < rows(); ++row_index)
                {
                    result += (*this)(row_index, col_index);
                }
            }
            return result;
        }

        Scalar minCoeff() const
        {
            if (size() == 0)
            {
                throw internal::Error(internal::ErrorCode::InvalidArgument, "minCoeff() requires a nonempty map");
            }
            Scalar result = (*this)(0, 0);
            for (Index col_index = 0; col_index < cols(); ++col_index)
            {
                for (Index row_index = 0; row_index < rows(); ++row_index)
                {
                    result = std::min(result, (*this)(row_index, col_index));
                }
            }
            return result;
        }

        Scalar maxCoeff() const
        {
            if (size() == 0)
            {
                throw internal::Error(internal::ErrorCode::InvalidArgument, "maxCoeff() requires a nonempty map");
            }
            Scalar result = (*this)(0, 0);
            for (Index col_index = 0; col_index < cols(); ++col_index)
            {
                for (Index row_index = 0; row_index < rows(); ++row_index)
                {
                    result = std::max(result, (*this)(row_index, col_index));
                }
            }
            return result;
        }

        Scalar trace() const
        {
            Scalar result{};
            const Index diagonal_size = std::min(rows(), cols());
            for (Index index = 0; index < diagonal_size; ++index)
            {
                result += (*this)(index, index);
            }
            return result;
        }

        bool allFinite() const
        {
            for (Index col_index = 0; col_index < cols(); ++col_index)
            {
                for (Index row_index = 0; row_index < rows(); ++row_index)
                {
                    if (!std::isfinite((*this)(row_index, col_index)))
                    {
                        return false;
                    }
                }
            }
            return true;
        }

        auto array() const;
        auto rowwise() const;
        auto colwise() const;

    private:
        static View makeContiguousView(Pointer data, Index rows, Index cols)
        {
            return View(internal::BasicMatrixView<Scalar, internal::Device::CPU, read_only>(
                data, rows, cols, Traits::IsRowMajor ? cols : 1, Traits::IsRowMajor ? 1 : rows));
        }

        static View makeStridedView(Pointer data, Index rows, Index cols, const StrideType& stride)
        {
            const Index inner = stride.inner() == 0 ? 1 : stride.inner();
            const Index inner_size = Traits::IsRowMajor ? cols : rows;
            const Index outer = stride.outer() == 0 ? inner * inner_size : stride.outer();
            return View(internal::BasicMatrixView<Scalar, internal::Device::CPU, read_only>(
                data, rows, cols, Traits::IsRowMajor ? outer : inner, Traits::IsRowMajor ? inner : outer));
        }

        View _view;
    };

    /// Eigen-style reference accepted by non-owning numerical APIs.
    template <typename PlainObjectType, int Options, typename StrideType>
    class Ref : public MatrixBase<Ref<PlainObjectType, Options, StrideType>>
    {
        static constexpr bool read_only = std::is_const_v<PlainObjectType>;
        using Plain = std::remove_const_t<PlainObjectType>;
        using MapType = Map<PlainObjectType, Options, StrideType>;

    public:
        using Base = MatrixBase<Ref>;
        using Scalar = typename Plain::Scalar;
        using Base::isApprox;
        using Base::maxCoeff;
        using Base::minCoeff;
        using Pointer = std::conditional_t<read_only, const Scalar*, Scalar*>;

        template <typename Derived, bool Enabled = !read_only, std::enable_if_t<Enabled, int> = 0>
        Ref(MatrixBase<Derived>& value)
            : _map(value.derived().data(),
                   value.rows(),
                   value.cols(),
                   map_detail::RefStrideFactory<StrideType>::make(value.derived().outerStride(),
                                                                  value.derived().innerStride()))
        {
            validateSource<Derived>();
        }

        template <typename Derived,
                  bool Enabled = !read_only && !map_detail::IsOwningMatrix<std::remove_const_t<Derived>>::value,
                  std::enable_if_t<Enabled, int> = 0>
        Ref(MatrixBase<Derived>&& value)
            : _map(value.derived().data(),
                   value.rows(),
                   value.cols(),
                   map_detail::RefStrideFactory<StrideType>::make(value.derived().outerStride(),
                                                                  value.derived().innerStride()))
        {
            validateSource<Derived>();
        }

        template <typename Derived, bool Enabled = read_only, std::enable_if_t<Enabled, int> = 0>
        Ref(const MatrixBase<Derived>& value)
            : _map(value.derived().data(),
                   value.rows(),
                   value.cols(),
                   map_detail::RefStrideFactory<StrideType>::make(value.derived().outerStride(),
                                                                  value.derived().innerStride()))
        {
            validateSource<Derived>();
        }

        Index rows() const noexcept
        {
            return _map.rows();
        }

        Index cols() const noexcept
        {
            return _map.cols();
        }

        Index size() const noexcept
        {
            return _map.size();
        }

        Index innerStride() const noexcept
        {
            return _map.innerStride();
        }

        Index outerStride() const noexcept
        {
            return _map.outerStride();
        }

        Pointer data() const noexcept
        {
            return _map.data();
        }

        decltype(auto) operator()(Index row, Index col) const
        {
            return _map(row, col);
        }

        decltype(auto) operator()(Index index) const
        {
            return _map(index);
        }

        const Scalar& coeff(Index row, Index col) const
        {
            return _map.coeff(row, col);
        }

        template <bool Enabled = !read_only, std::enable_if_t<Enabled, int> = 0> Scalar& coeffRef(Index row, Index col)
        {
            return _map.coeffRef(row, col);
        }

        template <typename Other, bool Enabled = !read_only, std::enable_if_t<Enabled, int> = 0>
        Ref& operator=(const Other& other)
        {
            _map = other;
            return *this;
        }

        Block<Ref, Dynamic, Dynamic> block(Index row, Index col, Index rows, Index cols)
        {
            return Block<Ref, Dynamic, Dynamic>(*this, row, col, rows, cols);
        }

        const Block<const Ref, Dynamic, Dynamic> block(Index row, Index col, Index rows, Index cols) const
        {
            return Block<const Ref, Dynamic, Dynamic>(*this, row, col, rows, cols);
        }

        template <int Rows, int Cols> Block<Ref, Rows, Cols> block(Index row, Index col)
        {
            return Block<Ref, Rows, Cols>(*this, row, col);
        }

        template <int Rows, int Cols> const Block<const Ref, Rows, Cols> block(Index row, Index col) const
        {
            return Block<const Ref, Rows, Cols>(*this, row, col);
        }

        Block<Ref, 1, Plain::ColsAtCompileTime, Plain::IsRowMajor != 0> row(Index index)
        {
            return Block<Ref, 1, Plain::ColsAtCompileTime, Plain::IsRowMajor != 0>(*this, index);
        }

        const Block<const Ref, 1, Plain::ColsAtCompileTime, Plain::IsRowMajor != 0> row(Index index) const
        {
            return Block<const Ref, 1, Plain::ColsAtCompileTime, Plain::IsRowMajor != 0>(*this, index);
        }

        Block<Ref, Plain::RowsAtCompileTime, 1, Plain::IsRowMajor == 0> col(Index index)
        {
            return Block<Ref, Plain::RowsAtCompileTime, 1, Plain::IsRowMajor == 0>(*this, index);
        }

        const Block<const Ref, Plain::RowsAtCompileTime, 1, Plain::IsRowMajor == 0> col(Index index) const
        {
            return Block<const Ref, Plain::RowsAtCompileTime, 1, Plain::IsRowMajor == 0>(*this, index);
        }

        template <typename Other> Scalar dot(const Other& other) const
        {
            return _map.dot(other);
        }

        Scalar squaredNorm() const
        {
            return _map.squaredNorm();
        }

        Scalar norm() const
        {
            return _map.norm();
        }

        Scalar sum() const
        {
            return _map.sum();
        }

        Scalar minCoeff() const
        {
            return _map.minCoeff();
        }

        Scalar maxCoeff() const
        {
            return _map.maxCoeff();
        }

        Scalar trace() const
        {
            return _map.trace();
        }

        bool allFinite() const
        {
            return _map.allFinite();
        }

        Transpose<Ref> transpose()
        {
            return Transpose<Ref>(*this);
        }

        const Transpose<const Ref> transpose() const
        {
            return Transpose<const Ref>(*this);
        }

        auto array() const
        {
            return _map.array();
        }

        auto rowwise() const
        {
            return _map.rowwise();
        }

        auto colwise() const
        {
            return _map.colwise();
        }

    private:
        template <typename Derived> static void validateSource()
        {
            using SourceTraits = ::plamatrix::detail::DenseTraits<Derived>;
            static_assert(std::is_same_v<Scalar, typename SourceTraits::ScalarType>,
                          "Ref source scalar type does not match its PlainObjectType");
            constexpr bool source_row_major = (SourceTraits::StorageOptions & RowMajor) == RowMajor;
            static_assert(Plain::IsVectorAtCompileTime || source_row_major == (Plain::IsRowMajor != 0),
                          "Ref source storage order does not match its PlainObjectType");
        }

        MapType _map;
    };
} // namespace plamatrix::v1
