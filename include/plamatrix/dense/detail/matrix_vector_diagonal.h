#pragma once

namespace plamatrix::v1
{
    /// Eigen-compatible public vector segment expression.
    template <typename VectorType, int Size>
    class VectorBlock
        : public Block<
              VectorType,
              (detail::DenseTraits<std::remove_const_t<VectorType>>::StorageOptions & RowMajor) == RowMajor ? 1 : Size,
              (detail::DenseTraits<std::remove_const_t<VectorType>>::StorageOptions & RowMajor) == RowMajor ? Size : 1>
    {
        using Traits = detail::DenseTraits<std::remove_const_t<VectorType>>;
        static constexpr bool row_vector = (Traits::StorageOptions & RowMajor) == RowMajor;
        using Base = Block<VectorType, row_vector ? 1 : Size, row_vector ? Size : 1>;

    public:
        using Base::operator=;

        VectorBlock(VectorType& vector, Index start, Index size)
            : Base(vector, row_vector ? 0 : start, row_vector ? start : 0, row_vector ? 1 : size, row_vector ? size : 1)
        {
            validateVector(vector);
        }

        VectorBlock(VectorType& vector, Index start) : Base(vector, row_vector ? 0 : start, row_vector ? start : 0)
        {
            static_assert(Size != Dynamic, "Fixed-size VectorBlock construction requires a fixed Size");
            validateVector(vector);
        }

    private:
        static void validateVector(const VectorType& vector)
        {
            if (vector.rows() != 1 && vector.cols() != 1)
            {
                throw internal::Error(internal::ErrorCode::InvalidArgument, "VectorBlock requires a vector expression");
            }
        }
    };

    /// Non-owning view of one matrix diagonal.
    template <typename MatrixType, int DiagIndex>
    class Diagonal : public view_detail::CommonDenseView<Diagonal<MatrixType, DiagIndex>>
    {
        using Traits = detail::DenseTraits<Diagonal>;
        static constexpr bool read_only = view_detail::IsReadOnly<MatrixType>;

    public:
        using Scalar = typename Traits::ScalarType;
        using PlainObject = typename Traits::PlainObject;
        using Index = plamatrix::Index;
        using Pointer = std::conditional_t<read_only, const Scalar*, Scalar*>;
        using Reference = std::conditional_t<read_only, const Scalar&, Scalar&>;
        static constexpr int RowsAtCompileTime = Traits::RowsAtCompileTime;
        static constexpr int ColsAtCompileTime = 1;
        static constexpr int MaxRowsAtCompileTime = Traits::MaxRowsAtCompileTime;
        static constexpr int MaxColsAtCompileTime = 1;
        static constexpr int IsRowMajor = 0;
        static constexpr int Options = ColMajor;
        static constexpr int IsVectorAtCompileTime = 1;
        static constexpr int SizeAtCompileTime = RowsAtCompileTime;

        explicit Diagonal(MatrixType& matrix, Index index = DiagIndex == DynamicIndex ? 0 : DiagIndex)
            : _matrix(&matrix), _index(index)
        {
            if constexpr (DiagIndex != DynamicIndex)
            {
                if (index != DiagIndex)
                {
                    throw internal::Error(internal::ErrorCode::InvalidArgument,
                                          "Diagonal runtime index conflicts with its compile-time index");
                }
            }
            const Index row_offset = _index < 0 ? -_index : 0;
            const Index col_offset = _index > 0 ? _index : 0;
            _size = row_offset > matrix.rows() || col_offset > matrix.cols()
                        ? 0
                        : std::min(matrix.rows() - row_offset, matrix.cols() - col_offset);
        }

        Diagonal(const Diagonal&) = default;

        Diagonal& operator=(const Diagonal& other)
        {
            assign(other);
            return *this;
        }

        template <typename Other, bool Enabled = !read_only, std::enable_if_t<Enabled, int> = 0>
        Diagonal& operator=(const Other& other)
        {
            assign(other);
            return *this;
        }

        template <typename Other, bool Enabled = !read_only, std::enable_if_t<Enabled, int> = 0>
        Diagonal& operator+=(const Other& other)
        {
            update(other, 1);
            return *this;
        }

        template <typename Other, bool Enabled = !read_only, std::enable_if_t<Enabled, int> = 0>
        Diagonal& operator-=(const Other& other)
        {
            update(other, -1);
            return *this;
        }

        Index rows() const noexcept
        {
            return _size;
        }
        Index cols() const noexcept
        {
            return 1;
        }
        Index size() const noexcept
        {
            return _size;
        }
        Index innerStride() const noexcept
        {
            return rowStride() + colStride();
        }
        Index outerStride() const noexcept
        {
            return innerStride() * std::max<Index>(_size, 1);
        }

        Pointer data() const noexcept
        {
            Pointer result = _matrix->data();
            if (_size != 0)
            {
                result += (_index < 0 ? -_index : 0) * rowStride() + (_index > 0 ? _index : 0) * colStride();
            }
            return result;
        }

        Reference operator()(Index row, Index col) const
        {
            if (col != 0)
            {
                throw std::out_of_range("Diagonal column index is out of range");
            }
            return (*this)(row);
        }

        Reference operator()(Index index) const
        {
            if (index < 0 || index >= _size)
            {
                throw std::out_of_range("Diagonal coefficient index is out of range");
            }
            return data()[index * innerStride()];
        }

        const Scalar& coeff(Index row, Index col) const
        {
            return (*this)(row, col);
        }
        const Scalar& coeff(Index index) const
        {
            return (*this)(index);
        }

        template <bool Enabled = !read_only, std::enable_if_t<Enabled, int> = 0> Scalar& coeffRef(Index row, Index col)
        {
            return (*this)(row, col);
        }

        MatrixType& nestedExpression() const noexcept
        {
            return *_matrix;
        }
        Index index() const noexcept
        {
            return _index;
        }

    private:
        Index rowStride() const noexcept
        {
            constexpr bool row_major =
                (detail::DenseTraits<std::remove_const_t<MatrixType>>::StorageOptions & RowMajor) == RowMajor;
            return row_major ? _matrix->outerStride() : _matrix->innerStride();
        }

        Index colStride() const noexcept
        {
            constexpr bool row_major =
                (detail::DenseTraits<std::remove_const_t<MatrixType>>::StorageOptions & RowMajor) == RowMajor;
            return row_major ? _matrix->innerStride() : _matrix->outerStride();
        }

        template <typename Other> void assign(const Other& other)
        {
            static_assert(!read_only, "Cannot assign through a const Diagonal expression");
            update(other, 0);
        }

        template <typename Other> void update(const Other& other, int operation)
        {
            if (other.rows() != rows() || other.cols() != 1)
            {
                throw std::invalid_argument("Diagonal assignment requires equal vector shapes");
            }
            std::vector<Scalar> snapshot(static_cast<std::size_t>(_size));
            for (Index index = 0; index < _size; ++index)
            {
                snapshot[static_cast<std::size_t>(index)] = other(index);
            }
            for (Index index = 0; index < _size; ++index)
            {
                Scalar& target = (*this)(index);
                const Scalar value = snapshot[static_cast<std::size_t>(index)];
                target = operation == 0 ? value : (operation > 0 ? target + value : target - value);
            }
        }

        MatrixType* _matrix;
        Index _index;
        Index _size = 0;
    };

    /// Lazy square diagonal-matrix expression wrapping a vector.
    template <typename DiagonalVectorType>
    class DiagonalWrapper : public view_detail::CommonDenseView<DiagonalWrapper<DiagonalVectorType>>
    {
        using Traits = detail::DenseTraits<DiagonalWrapper>;

    public:
        using Scalar = typename Traits::ScalarType;
        using PlainObject = typename Traits::PlainObject;
        using Index = plamatrix::Index;
        static constexpr int RowsAtCompileTime = Traits::RowsAtCompileTime;
        static constexpr int ColsAtCompileTime = Traits::ColsAtCompileTime;
        static constexpr int MaxRowsAtCompileTime = Traits::MaxRowsAtCompileTime;
        static constexpr int MaxColsAtCompileTime = Traits::MaxColsAtCompileTime;
        static constexpr int IsRowMajor = 0;
        static constexpr int Options = ColMajor;
        static constexpr int IsVectorAtCompileTime = 0;
        static constexpr int SizeAtCompileTime =
            RowsAtCompileTime == Dynamic ? Dynamic : RowsAtCompileTime * ColsAtCompileTime;

        explicit DiagonalWrapper(DiagonalVectorType& diagonal) : _diagonal(&diagonal)
        {
            if (diagonal.rows() != 1 && diagonal.cols() != 1)
            {
                throw internal::Error(internal::ErrorCode::InvalidArgument, "asDiagonal() requires a vector");
            }
        }

        Index rows() const noexcept
        {
            return _diagonal->size();
        }
        Index cols() const noexcept
        {
            return _diagonal->size();
        }
        Index size() const noexcept
        {
            return rows() * cols();
        }

        Scalar operator()(Index row, Index col) const
        {
            validateIndex(row, col);
            return row == col ? (*_diagonal)(row) : Scalar{};
        }

        Scalar coeff(Index row, Index col) const
        {
            return (*this)(row, col);
        }
        const DiagonalVectorType& diagonal() const noexcept
        {
            return *_diagonal;
        }

    private:
        void validateIndex(Index row, Index col) const
        {
            if (row < 0 || col < 0 || row >= rows() || col >= cols())
            {
                throw std::out_of_range("DiagonalWrapper coefficient index is out of range");
            }
        }

        DiagonalVectorType* _diagonal;
    };

} // namespace plamatrix::v1
