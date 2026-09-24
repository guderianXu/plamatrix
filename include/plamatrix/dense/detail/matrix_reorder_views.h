#pragma once

namespace plamatrix::v1
{
    /// Lazy reshaping expression with Eigen-compatible traversal-order selection.
    template <typename XprType, int Rows, int Cols, int Order>
    class Reshaped : public view_detail::CommonDenseView<Reshaped<XprType, Rows, Cols, Order>>
    {
        using Traits = detail::DenseTraits<Reshaped>;
        static constexpr bool read_only = view_detail::IsReadOnly<XprType>;
        static constexpr int resolved_order =
            Order == AutoOrder ? detail::DenseTraits<std::remove_const_t<XprType>>::StorageOptions & RowMajor : Order;
        static_assert(Order == ColMajor || Order == RowMajor || Order == AutoOrder,
                      "Reshaped order must be ColMajor, RowMajor, or AutoOrder");

    public:
        using Scalar = typename Traits::ScalarType;
        using PlainObject = typename Traits::PlainObject;
        using Index = plamatrix::Index;
        using Reference = view_detail::CoefficientReference<XprType>;
        static constexpr int RowsAtCompileTime = Rows;
        static constexpr int ColsAtCompileTime = Cols;
        static constexpr int MaxRowsAtCompileTime = Rows;
        static constexpr int MaxColsAtCompileTime = Cols;
        static constexpr int IsRowMajor = resolved_order == RowMajor;
        static constexpr int Options = IsRowMajor ? RowMajor : ColMajor;
        static constexpr int IsVectorAtCompileTime = Rows == 1 || Cols == 1;
        static constexpr int SizeAtCompileTime = Rows == Dynamic || Cols == Dynamic ? Dynamic : Rows * Cols;

        Reshaped(XprType& expression, Index rows, Index cols) : _expression(&expression), _rows(rows), _cols(cols)
        {
            if (rows < 0 || cols < 0 || (rows != 0 && cols > std::numeric_limits<Index>::max() / rows) ||
                rows * cols != expression.size())
            {
                throw internal::Error(internal::ErrorCode::InvalidArgument,
                                      "reshaped() dimensions must preserve the coefficient count");
            }
            if ((Rows != Dynamic && rows != Rows) || (Cols != Dynamic && cols != Cols))
            {
                throw internal::Error(internal::ErrorCode::InvalidArgument,
                                      "Reshaped runtime shape conflicts with its compile-time shape");
            }
        }

        Reshaped(const Reshaped&) = default;

        Reshaped& operator=(const Reshaped& other)
        {
            assign(other);
            return *this;
        }

        template <typename Other, bool Enabled = !read_only, std::enable_if_t<Enabled, int> = 0>
        Reshaped& operator=(const Other& other)
        {
            assign(other);
            return *this;
        }

        Index rows() const noexcept
        {
            return _rows;
        }
        Index cols() const noexcept
        {
            return _cols;
        }
        Index size() const noexcept
        {
            return _rows * _cols;
        }

        Reference operator()(Index row, Index col) const
        {
            if (row < 0 || col < 0 || row >= _rows || col >= _cols)
            {
                throw std::out_of_range("Reshaped coefficient index is out of range");
            }
            const Index linear = resolved_order == RowMajor ? col + row * _cols : row + col * _rows;
            const Index source_row =
                resolved_order == RowMajor ? linear / _expression->cols() : linear % _expression->rows();
            const Index source_col =
                resolved_order == RowMajor ? linear % _expression->cols() : linear / _expression->rows();
            return (*_expression)(source_row, source_col);
        }

        Reference operator()(Index index) const
        {
            if (_cols == 1)
            {
                return (*this)(index, 0);
            }
            if (_rows == 1)
            {
                return (*this)(0, index);
            }
            throw internal::Error(internal::ErrorCode::InvalidArgument,
                                  "Single-index Reshaped access requires a vector");
        }

        decltype(auto) coeff(Index row, Index col) const
        {
            return (*this)(row, col);
        }

        template <bool Enabled = !read_only, std::enable_if_t<Enabled, int> = 0> Scalar& coeffRef(Index row, Index col)
        {
            return (*this)(row, col);
        }

        XprType& nestedExpression() const noexcept
        {
            return *_expression;
        }

    private:
        template <typename Other> void assign(const Other& other)
        {
            static_assert(!read_only, "Cannot assign through a const Reshaped expression");
            if (rows() != other.rows() || cols() != other.cols())
            {
                throw std::invalid_argument("Reshaped assignment requires equal shapes");
            }
            PlainObject snapshot(other);
            for (Index col = 0; col < cols(); ++col)
            {
                for (Index row = 0; row < rows(); ++row)
                {
                    (*this)(row, col) = snapshot(row, col);
                }
            }
        }

        XprType* _expression;
        Index _rows;
        Index _cols;
    };

    /// Lazy coefficient replication expression.
    template <typename MatrixType, int RowFactor, int ColFactor>
    class Replicate : public view_detail::CommonDenseView<Replicate<MatrixType, RowFactor, ColFactor>>
    {
        using Traits = detail::DenseTraits<Replicate>;

    public:
        using Scalar = typename Traits::ScalarType;
        using PlainObject = typename Traits::PlainObject;
        using Index = plamatrix::Index;
        static constexpr int RowsAtCompileTime = Traits::RowsAtCompileTime;
        static constexpr int ColsAtCompileTime = Traits::ColsAtCompileTime;
        static constexpr int MaxRowsAtCompileTime = Traits::MaxRowsAtCompileTime;
        static constexpr int MaxColsAtCompileTime = Traits::MaxColsAtCompileTime;
        static constexpr int IsRowMajor = (Traits::StorageOptions & RowMajor) == RowMajor;
        static constexpr int Options = Traits::StorageOptions;
        static constexpr int IsVectorAtCompileTime = RowsAtCompileTime == 1 || ColsAtCompileTime == 1;
        static constexpr int SizeAtCompileTime = RowsAtCompileTime == Dynamic || ColsAtCompileTime == Dynamic
                                                     ? Dynamic
                                                     : RowsAtCompileTime * ColsAtCompileTime;

        template <int R = RowFactor, int C = ColFactor, std::enable_if_t<R != Dynamic && C != Dynamic, int> = 0>
        explicit Replicate(const MatrixType& matrix) : Replicate(matrix, RowFactor, ColFactor)
        {
        }

        Replicate(const MatrixType& matrix, Index row_factor, Index col_factor)
            : _matrix(&matrix), _rowFactor(row_factor), _colFactor(col_factor)
        {
            if (row_factor < 0 || col_factor < 0 || (RowFactor != Dynamic && row_factor != RowFactor) ||
                (ColFactor != Dynamic && col_factor != ColFactor) ||
                (matrix.rows() != 0 && row_factor > std::numeric_limits<Index>::max() / matrix.rows()) ||
                (matrix.cols() != 0 && col_factor > std::numeric_limits<Index>::max() / matrix.cols()))
            {
                throw internal::Error(internal::ErrorCode::InvalidArgument, "Invalid replicate factors");
            }
            const Index result_rows = matrix.rows() * row_factor;
            const Index result_cols = matrix.cols() * col_factor;
            if (result_rows != 0 && result_cols > std::numeric_limits<Index>::max() / result_rows)
            {
                throw internal::Error(internal::ErrorCode::InvalidArgument,
                                      "Replicate dimensions overflow the index range");
            }
        }

        Index rows() const noexcept
        {
            return _matrix->rows() * _rowFactor;
        }
        Index cols() const noexcept
        {
            return _matrix->cols() * _colFactor;
        }
        Index size() const noexcept
        {
            return rows() * cols();
        }

        Scalar operator()(Index row, Index col) const
        {
            if (row < 0 || col < 0 || row >= rows() || col >= cols())
            {
                throw std::out_of_range("Replicate coefficient index is out of range");
            }
            return (*_matrix)(row % _matrix->rows(), col % _matrix->cols());
        }

        Scalar operator()(Index index) const
        {
            if (cols() == 1)
            {
                return (*this)(index, 0);
            }
            if (rows() == 1)
            {
                return (*this)(0, index);
            }
            throw internal::Error(internal::ErrorCode::InvalidArgument,
                                  "Single-index Replicate access requires a vector");
        }

        Scalar coeff(Index row, Index col) const
        {
            return (*this)(row, col);
        }
        const MatrixType& nestedExpression() const noexcept
        {
            return *_matrix;
        }

    private:
        const MatrixType* _matrix;
        Index _rowFactor;
        Index _colFactor;
    };

    /// Lazy reversible lvalue expression.
    template <typename MatrixType, int Direction>
    class Reverse : public view_detail::CommonDenseView<Reverse<MatrixType, Direction>>
    {
        using Traits = detail::DenseTraits<Reverse>;
        static constexpr bool read_only = view_detail::IsReadOnly<MatrixType>;
        static constexpr bool reverse_rows = Direction == Vertical || Direction == BothDirections;
        static constexpr bool reverse_cols = Direction == Horizontal || Direction == BothDirections;
        static_assert(Direction == Vertical || Direction == Horizontal || Direction == BothDirections,
                      "Reverse direction is invalid");

    public:
        using Scalar = typename Traits::ScalarType;
        using PlainObject = typename Traits::PlainObject;
        using Index = plamatrix::Index;
        using Reference = view_detail::CoefficientReference<MatrixType>;
        static constexpr int RowsAtCompileTime = Traits::RowsAtCompileTime;
        static constexpr int ColsAtCompileTime = Traits::ColsAtCompileTime;
        static constexpr int MaxRowsAtCompileTime = Traits::MaxRowsAtCompileTime;
        static constexpr int MaxColsAtCompileTime = Traits::MaxColsAtCompileTime;
        static constexpr int IsRowMajor = (Traits::StorageOptions & RowMajor) == RowMajor;
        static constexpr int Options = Traits::StorageOptions;
        static constexpr int IsVectorAtCompileTime = RowsAtCompileTime == 1 || ColsAtCompileTime == 1;
        static constexpr int SizeAtCompileTime = RowsAtCompileTime == Dynamic || ColsAtCompileTime == Dynamic
                                                     ? Dynamic
                                                     : RowsAtCompileTime * ColsAtCompileTime;

        explicit Reverse(MatrixType& matrix) : _matrix(&matrix)
        {
        }
        Reverse(const Reverse&) = default;

        Reverse& operator=(const Reverse& other)
        {
            assign(other);
            return *this;
        }

        template <typename Other, bool Enabled = !read_only, std::enable_if_t<Enabled, int> = 0>
        Reverse& operator=(const Other& other)
        {
            assign(other);
            return *this;
        }

        Index rows() const noexcept
        {
            return _matrix->rows();
        }
        Index cols() const noexcept
        {
            return _matrix->cols();
        }
        Index size() const noexcept
        {
            return rows() * cols();
        }

        Reference operator()(Index row, Index col) const
        {
            if (row < 0 || col < 0 || row >= rows() || col >= cols())
            {
                throw std::out_of_range("Reverse coefficient index is out of range");
            }
            return (*_matrix)(reverse_rows ? rows() - row - 1 : row, reverse_cols ? cols() - col - 1 : col);
        }

        Reference operator()(Index index) const
        {
            if (cols() == 1)
            {
                return (*this)(index, 0);
            }
            if (rows() == 1)
            {
                return (*this)(0, index);
            }
            throw internal::Error(internal::ErrorCode::InvalidArgument,
                                  "Single-index Reverse access requires a vector");
        }

        decltype(auto) coeff(Index row, Index col) const
        {
            return (*this)(row, col);
        }

        template <bool Enabled = !read_only, std::enable_if_t<Enabled, int> = 0> Scalar& coeffRef(Index row, Index col)
        {
            return (*this)(row, col);
        }

        MatrixType& nestedExpression() const noexcept
        {
            return *_matrix;
        }

    private:
        template <typename Other> void assign(const Other& other)
        {
            static_assert(!read_only, "Cannot assign through a const Reverse expression");
            if (rows() != other.rows() || cols() != other.cols())
            {
                throw std::invalid_argument("Reverse assignment requires equal shapes");
            }
            PlainObject snapshot(other);
            for (Index col = 0; col < cols(); ++col)
            {
                for (Index row = 0; row < rows(); ++row)
                {
                    (*this)(row, col) = snapshot(row, col);
                }
            }
        }

        MatrixType* _matrix;
    };

} // namespace plamatrix::v1
