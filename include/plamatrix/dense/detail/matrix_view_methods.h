#pragma once

namespace plamatrix::v1
{
    template <typename Derived>
    typename DenseBase<Derived>::SegmentReturnType DenseBase<Derived>::segment(Index start, Index length)
    {
        return SegmentReturnType(this->derived(), start, length);
    }

    template <typename Derived>
    typename DenseBase<Derived>::ConstSegmentReturnType DenseBase<Derived>::segment(Index start, Index length) const
    {
        return ConstSegmentReturnType(this->derived(), start, length);
    }

    template <typename Derived>
    template <int Length>
    VectorBlock<Derived, Length> DenseBase<Derived>::segment(Index start)
    {
        static_assert(Length >= 0, "Segment length must be non-negative");
        static_assert(IsVectorAtCompileTime, "Fixed-size segment requires a vector expression");
        return VectorBlock<Derived, Length>(this->derived(), start);
    }

    template <typename Derived>
    template <int Length>
    const VectorBlock<const Derived, Length> DenseBase<Derived>::segment(Index start) const
    {
        static_assert(Length >= 0, "Segment length must be non-negative");
        static_assert(IsVectorAtCompileTime, "Fixed-size segment requires a vector expression");
        return VectorBlock<const Derived, Length>(this->derived(), start);
    }

    template <typename Derived> typename DenseBase<Derived>::SegmentReturnType DenseBase<Derived>::head(Index length)
    {
        return segment(0, length);
    }

    template <typename Derived>
    typename DenseBase<Derived>::ConstSegmentReturnType DenseBase<Derived>::head(Index length) const
    {
        return segment(0, length);
    }

    template <typename Derived> typename DenseBase<Derived>::SegmentReturnType DenseBase<Derived>::tail(Index length)
    {
        return segment(this->size() - length, length);
    }

    template <typename Derived>
    typename DenseBase<Derived>::ConstSegmentReturnType DenseBase<Derived>::tail(Index length) const
    {
        return segment(this->size() - length, length);
    }

    template <typename Derived> template <int Length> VectorBlock<Derived, Length> DenseBase<Derived>::head()
    {
        return segment<Length>(0);
    }

    template <typename Derived>
    template <int Length>
    const VectorBlock<const Derived, Length> DenseBase<Derived>::head() const
    {
        return segment<Length>(0);
    }

    template <typename Derived> template <int Length> VectorBlock<Derived, Length> DenseBase<Derived>::tail()
    {
        return segment<Length>(this->size() - Length);
    }

    template <typename Derived>
    template <int Length>
    const VectorBlock<const Derived, Length> DenseBase<Derived>::tail() const
    {
        return segment<Length>(this->size() - Length);
    }

    template <typename Derived>
    template <int RowFactor, int ColFactor>
    const Replicate<Derived, RowFactor, ColFactor> DenseBase<Derived>::replicate() const
    {
        static_assert(RowFactor >= 0 && ColFactor >= 0, "Replicate factors must be non-negative");
        return Replicate<Derived, RowFactor, ColFactor>(this->derived());
    }

    template <typename Derived>
    const Replicate<Derived, Dynamic, Dynamic> DenseBase<Derived>::replicate(Index row_factor, Index col_factor) const
    {
        return Replicate<Derived, Dynamic, Dynamic>(this->derived(), row_factor, col_factor);
    }

    template <typename Derived> typename DenseBase<Derived>::ReverseReturnType DenseBase<Derived>::reverse()
    {
        return ReverseReturnType(this->derived());
    }

    template <typename Derived> typename DenseBase<Derived>::ConstReverseReturnType DenseBase<Derived>::reverse() const
    {
        return ConstReverseReturnType(this->derived());
    }

    template <typename Derived> void DenseBase<Derived>::reverseInPlace()
    {
        PlainObject snapshot(this->derived());
        for (Index col = 0; col < this->cols(); ++col)
        {
            for (Index row = 0; row < this->rows(); ++row)
            {
                this->derived()(row, col) = snapshot(this->rows() - row - 1, this->cols() - col - 1);
            }
        }
    }

    template <typename Derived>
    template <int Order>
    Reshaped<Derived, Dynamic, Dynamic, Order> DenseBase<Derived>::reshaped(Index rows, Index cols)
    {
        return Reshaped<Derived, Dynamic, Dynamic, Order>(this->derived(), rows, cols);
    }

    template <typename Derived>
    template <int Order>
    const Reshaped<const Derived, Dynamic, Dynamic, Order> DenseBase<Derived>::reshaped(Index rows, Index cols) const
    {
        return Reshaped<const Derived, Dynamic, Dynamic, Order>(this->derived(), rows, cols);
    }

    template <typename Derived>
    template <int Order>
    Reshaped<Derived, DenseBase<Derived>::SizeAtCompileTime, 1, Order> DenseBase<Derived>::reshaped()
    {
        return Reshaped<Derived, SizeAtCompileTime, 1, Order>(this->derived(), this->size(), 1);
    }

    template <typename Derived>
    template <int Order>
    const Reshaped<const Derived, DenseBase<Derived>::SizeAtCompileTime, 1, Order> DenseBase<Derived>::reshaped() const
    {
        return Reshaped<const Derived, SizeAtCompileTime, 1, Order>(this->derived(), this->size(), 1);
    }

    template <typename Derived> void DenseBase<Derived>::transposeInPlace()
    {
        using Transposed = Matrix<Scalar,
                                  ColsAtCompileTime,
                                  RowsAtCompileTime,
                                  IsRowMajor ? ColMajor : RowMajor,
                                  MaxColsAtCompileTime,
                                  MaxRowsAtCompileTime>;
        Transposed snapshot(this->cols(), this->rows());
        for (Index col = 0; col < this->cols(); ++col)
        {
            for (Index row = 0; row < this->rows(); ++row)
            {
                snapshot(col, row) = this->derived()(row, col);
            }
        }
        if (this->rows() != this->cols())
        {
            if constexpr (view_detail::HasResize<Derived>::value)
            {
                this->derived().resize(snapshot.rows(), snapshot.cols());
            }
            else
            {
                throw internal::Error(internal::ErrorCode::InvalidArgument,
                                      "A non-square fixed view cannot be transposed in place");
            }
        }
        this->derived() = snapshot;
    }

    template <typename Derived> typename MatrixBase<Derived>::DiagonalReturnType MatrixBase<Derived>::diagonal()
    {
        return DiagonalReturnType(this->derived());
    }

    template <typename Derived>
    typename MatrixBase<Derived>::ConstDiagonalReturnType MatrixBase<Derived>::diagonal() const
    {
        return ConstDiagonalReturnType(this->derived());
    }

    template <typename Derived> template <int DiagIndex> Diagonal<Derived, DiagIndex> MatrixBase<Derived>::diagonal()
    {
        return Diagonal<Derived, DiagIndex>(this->derived(), DiagIndex);
    }

    template <typename Derived>
    template <int DiagIndex>
    const Diagonal<const Derived, DiagIndex> MatrixBase<Derived>::diagonal() const
    {
        return Diagonal<const Derived, DiagIndex>(this->derived(), DiagIndex);
    }

    template <typename Derived> Diagonal<Derived, DynamicIndex> MatrixBase<Derived>::diagonal(Index index)
    {
        return Diagonal<Derived, DynamicIndex>(this->derived(), index);
    }

    template <typename Derived>
    const Diagonal<const Derived, DynamicIndex> MatrixBase<Derived>::diagonal(Index index) const
    {
        return Diagonal<const Derived, DynamicIndex>(this->derived(), index);
    }

    template <typename Derived> const DiagonalWrapper<const Derived> MatrixBase<Derived>::asDiagonal() const
    {
        return DiagonalWrapper<const Derived>(this->derived());
    }

    template <typename Derived>
    template <unsigned int Mode>
    TriangularView<Derived, Mode> MatrixBase<Derived>::triangularView()
    {
        return TriangularView<Derived, Mode>(this->derived());
    }

    template <typename Derived>
    template <unsigned int Mode>
    const TriangularView<const Derived, Mode> MatrixBase<Derived>::triangularView() const
    {
        return TriangularView<const Derived, Mode>(this->derived());
    }

    template <typename Derived>
    template <unsigned int UpLo>
    SelfAdjointView<Derived, UpLo> MatrixBase<Derived>::selfadjointView()
    {
        return SelfAdjointView<Derived, UpLo>(this->derived());
    }

    template <typename Derived>
    template <unsigned int UpLo>
    const SelfAdjointView<const Derived, UpLo> MatrixBase<Derived>::selfadjointView() const
    {
        return SelfAdjointView<const Derived, UpLo>(this->derived());
    }
} // namespace plamatrix::v1
