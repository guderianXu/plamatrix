#pragma once

namespace plamatrix::v1
{
    /// Lazy view interpreting one triangle of a dense matrix.
    template <typename MatrixType, unsigned int Mode>
    class TriangularView : public view_detail::CommonDenseView<TriangularView<MatrixType, Mode>>
    {
        using Traits = detail::DenseTraits<TriangularView>;
        static constexpr bool read_only = view_detail::IsReadOnly<MatrixType>;
        static constexpr bool lower = (Mode & Lower) != 0;
        static constexpr bool upper = (Mode & Upper) != 0;
        static constexpr bool unit_diagonal = (Mode & UnitDiag) != 0;
        static constexpr bool zero_diagonal = (Mode & ZeroDiag) != 0;
        static_assert(lower != upper, "TriangularView mode must select exactly one of Lower or Upper");
        static_assert(!(unit_diagonal && zero_diagonal), "TriangularView cannot have both unit and zero diagonal");

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

        explicit TriangularView(MatrixType& matrix) : _matrix(&matrix)
        {
        }
        TriangularView(const TriangularView&) = default;

        TriangularView& operator=(const TriangularView& other)
        {
            assign(other);
            return *this;
        }

        template <typename Other, bool Enabled = !read_only, std::enable_if_t<Enabled, int> = 0>
        TriangularView& operator=(const Other& other)
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

        Scalar operator()(Index row, Index col) const
        {
            validateIndex(row, col);
            if (row == col && unit_diagonal)
            {
                return Scalar{1};
            }
            return selected(row, col) ? static_cast<Scalar>((*_matrix)(row, col)) : Scalar{};
        }

        template <bool Enabled = !read_only, std::enable_if_t<Enabled, int> = 0>
        Scalar& operator()(Index row, Index col)
        {
            validateIndex(row, col);
            if (!selected(row, col) || (row == col && (unit_diagonal || zero_diagonal)))
            {
                throw internal::Error(internal::ErrorCode::InvalidArgument,
                                      "Cannot write outside the stored triangular coefficients");
            }
            return (*_matrix)(row, col);
        }

        Scalar coeff(Index row, Index col) const
        {
            return (*this)(row, col);
        }

        template <bool Enabled = !read_only, std::enable_if_t<Enabled, int> = 0> Scalar& coeffRef(Index row, Index col)
        {
            return (*this)(row, col);
        }

        void setZero()
        {
            static_assert(!read_only, "Cannot modify a const TriangularView");
            forStored([&](Index row, Index col) { (*_matrix)(row, col) = Scalar{}; });
        }

        MatrixType& nestedExpression() const noexcept
        {
            return *_matrix;
        }

    private:
        bool selected(Index row, Index col) const noexcept
        {
            if (row == col && zero_diagonal)
            {
                return false;
            }
            return lower ? row >= col : row <= col;
        }

        void validateIndex(Index row, Index col) const
        {
            if (row < 0 || col < 0 || row >= rows() || col >= cols())
            {
                throw std::out_of_range("TriangularView coefficient index is out of range");
            }
        }

        template <typename Function> void forStored(Function&& function)
        {
            for (Index col = 0; col < cols(); ++col)
            {
                for (Index row = 0; row < rows(); ++row)
                {
                    if (selected(row, col) && !(row == col && unit_diagonal))
                    {
                        function(row, col);
                    }
                }
            }
        }

        template <typename Other> void assign(const Other& other)
        {
            static_assert(!read_only, "Cannot assign through a const TriangularView");
            if (rows() != other.rows() || cols() != other.cols())
            {
                throw std::invalid_argument("TriangularView assignment requires equal shapes");
            }
            PlainObject snapshot(other);
            forStored([&](Index row, Index col) { (*_matrix)(row, col) = snapshot(row, col); });
        }

        MatrixType* _matrix;
    };

    /// Lazy symmetric/self-adjoint view backed by one stored triangle.
    template <typename MatrixType, unsigned int UpLo>
    class SelfAdjointView : public view_detail::CommonDenseView<SelfAdjointView<MatrixType, UpLo>>
    {
        using Traits = detail::DenseTraits<SelfAdjointView>;
        static constexpr bool read_only = view_detail::IsReadOnly<MatrixType>;
        static constexpr bool lower = (UpLo & Lower) != 0;
        static constexpr bool upper = (UpLo & Upper) != 0;
        static_assert(lower != upper, "SelfAdjointView must select exactly one stored triangle");

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
        static constexpr int IsVectorAtCompileTime = 0;
        static constexpr int SizeAtCompileTime =
            RowsAtCompileTime == Dynamic ? Dynamic : RowsAtCompileTime * ColsAtCompileTime;

        explicit SelfAdjointView(MatrixType& matrix) : _matrix(&matrix)
        {
            if (matrix.rows() != matrix.cols())
            {
                throw internal::Error(internal::ErrorCode::InvalidArgument,
                                      "selfadjointView() requires a square matrix");
            }
        }

        SelfAdjointView(const SelfAdjointView&) = default;

        SelfAdjointView& operator=(const SelfAdjointView& other)
        {
            assign(other);
            return *this;
        }

        template <typename Other, bool Enabled = !read_only, std::enable_if_t<Enabled, int> = 0>
        SelfAdjointView& operator=(const Other& other)
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

        Scalar operator()(Index row, Index col) const
        {
            validateIndex(row, col);
            return stored(row, col) ? static_cast<Scalar>((*_matrix)(row, col))
                                    : static_cast<Scalar>((*_matrix)(col, row));
        }

        template <bool Enabled = !read_only, std::enable_if_t<Enabled, int> = 0>
        Scalar& operator()(Index row, Index col)
        {
            validateIndex(row, col);
            return stored(row, col) ? (*_matrix)(row, col) : (*_matrix)(col, row);
        }

        Scalar coeff(Index row, Index col) const
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
        bool stored(Index row, Index col) const noexcept
        {
            return lower ? row >= col : row <= col;
        }

        void validateIndex(Index row, Index col) const
        {
            if (row < 0 || col < 0 || row >= rows() || col >= cols())
            {
                throw std::out_of_range("SelfAdjointView coefficient index is out of range");
            }
        }

        template <typename Other> void assign(const Other& other)
        {
            static_assert(!read_only, "Cannot assign through a const SelfAdjointView");
            if (rows() != other.rows() || cols() != other.cols())
            {
                throw std::invalid_argument("SelfAdjointView assignment requires equal square shapes");
            }
            PlainObject snapshot(other);
            for (Index col = 0; col < cols(); ++col)
            {
                for (Index row = 0; row < rows(); ++row)
                {
                    if (stored(row, col))
                    {
                        (*_matrix)(row, col) = snapshot(row, col);
                    }
                }
            }
        }

        MatrixType* _matrix;
    };

} // namespace plamatrix::v1
