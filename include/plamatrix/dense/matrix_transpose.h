#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace plamatrix::v1
{
    namespace transpose_detail
    {
        template <typename Type> struct IsPlainMatrix : std::false_type
        {
        };

        template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
        struct IsPlainMatrix<Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>> : std::true_type
        {
        };
    } // namespace transpose_detail

    /// Eigen-compatible lazy transpose expression over a dense expression.
    template <typename MatrixType> class Transpose : public MatrixBase<Transpose<MatrixType>>
    {
        using NestedTraits = detail::DenseTraits<std::remove_const_t<MatrixType>>;
        using NestedPointer = decltype(std::declval<MatrixType&>().data());
        static constexpr bool read_only =
            std::is_const_v<MatrixType> || std::is_const_v<std::remove_pointer_t<NestedPointer>>;
        static constexpr bool stores_reference =
            transpose_detail::IsPlainMatrix<std::remove_const_t<MatrixType>>::value;
        using StoredExpression = std::conditional_t<stores_reference, MatrixType*, std::remove_const_t<MatrixType>>;

    public:
        using Base = MatrixBase<Transpose>;
        using Scalar = typename NestedTraits::ScalarType;
        using Base::isApprox;
        using Base::maxCoeff;
        using Base::minCoeff;
        using RealScalar = typename NumTraits<Scalar>::Real;
        using NestedExpression = std::remove_const_t<MatrixType>;
        using Pointer = std::conditional_t<read_only, const Scalar*, Scalar*>;
        using Reference =
            decltype(std::declval<MatrixType&>()(std::declval<plamatrix::Index>(), std::declval<plamatrix::Index>()));
        using Index = plamatrix::Index;
        static constexpr int RowsAtCompileTime = NestedTraits::ColsAtCompileTime;
        static constexpr int ColsAtCompileTime = NestedTraits::RowsAtCompileTime;
        static constexpr int MaxRowsAtCompileTime = NestedTraits::MaxColsAtCompileTime;
        static constexpr int MaxColsAtCompileTime = NestedTraits::MaxRowsAtCompileTime;
        static constexpr int IsRowMajor = (NestedTraits::StorageOptions & RowMajor) == RowMajor ? 0 : 1;
        static constexpr int Options = IsRowMajor ? RowMajor : ColMajor;
        static constexpr int IsVectorAtCompileTime = RowsAtCompileTime == 1 || ColsAtCompileTime == 1;
        static constexpr int SizeAtCompileTime = RowsAtCompileTime == Dynamic || ColsAtCompileTime == Dynamic
                                                     ? Dynamic
                                                     : RowsAtCompileTime * ColsAtCompileTime;
        using PlainObject =
            Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime, Options, MaxRowsAtCompileTime, MaxColsAtCompileTime>;

        explicit Transpose(MatrixType& matrix) : _matrix(makeStorage(matrix))
        {
        }

        Index rows() const noexcept
        {
            return nested().cols();
        }

        Index cols() const noexcept
        {
            return nested().rows();
        }

        Index size() const noexcept
        {
            return rows() * cols();
        }

        Index innerStride() const noexcept
        {
            return nested().innerStride();
        }

        Index outerStride() const noexcept
        {
            return nested().outerStride();
        }

        Pointer data() const noexcept
        {
            return nested().data();
        }

        Reference operator()(Index row, Index col) const
        {
            return nested()(col, row);
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
                                  "Single-index Transpose access requires a vector");
        }

        decltype(auto) coeff(Index row, Index col) const
        {
            return (*this)(row, col);
        }

        template <bool Enabled = !read_only, std::enable_if_t<Enabled, int> = 0> Scalar& coeffRef(Index row, Index col)
        {
            return (*this)(row, col);
        }

        MatrixType& nestedExpression() noexcept
        {
            return nested();
        }

        const std::remove_const_t<MatrixType>& nestedExpression() const noexcept
        {
            return nested();
        }

        Transpose& operator=(const Transpose& other)
        {
            if (this != &other)
            {
                assign(other);
            }
            return *this;
        }

        template <typename Other, bool Enabled = !read_only, std::enable_if_t<Enabled, int> = 0>
        Transpose& operator=(const Other& other)
        {
            assign(other);
            return *this;
        }

        Block<Transpose, Dynamic, Dynamic> block(Index row, Index col, Index rows, Index cols)
        {
            return Block<Transpose, Dynamic, Dynamic>(*this, row, col, rows, cols);
        }

        const Block<const Transpose, Dynamic, Dynamic> block(Index row, Index col, Index rows, Index cols) const
        {
            return Block<const Transpose, Dynamic, Dynamic>(*this, row, col, rows, cols);
        }

        template <int Rows, int Cols> Block<Transpose, Rows, Cols> block(Index row, Index col)
        {
            return Block<Transpose, Rows, Cols>(*this, row, col);
        }

        template <int Rows, int Cols> const Block<const Transpose, Rows, Cols> block(Index row, Index col) const
        {
            return Block<const Transpose, Rows, Cols>(*this, row, col);
        }

        Block<Transpose, 1, ColsAtCompileTime, IsRowMajor != 0> row(Index index)
        {
            return Block<Transpose, 1, ColsAtCompileTime, IsRowMajor != 0>(*this, index);
        }

        const Block<const Transpose, 1, ColsAtCompileTime, IsRowMajor != 0> row(Index index) const
        {
            return Block<const Transpose, 1, ColsAtCompileTime, IsRowMajor != 0>(*this, index);
        }

        Block<Transpose, RowsAtCompileTime, 1, IsRowMajor == 0> col(Index index)
        {
            return Block<Transpose, RowsAtCompileTime, 1, IsRowMajor == 0>(*this, index);
        }

        const Block<const Transpose, RowsAtCompileTime, 1, IsRowMajor == 0> col(Index index) const
        {
            return Block<const Transpose, RowsAtCompileTime, 1, IsRowMajor == 0>(*this, index);
        }

        MatrixType& transpose() noexcept
        {
            return nested();
        }

        const std::remove_const_t<MatrixType>& transpose() const noexcept
        {
            return nested();
        }

        auto array() const;
        auto rowwise() const;
        auto colwise() const;
        template <typename Other> auto cwiseProduct(Other&& other) const;

        PlainObject eval() const
        {
            PlainObject result(rows(), cols());
            for (Index col_index = 0; col_index < cols(); ++col_index)
            {
                for (Index row_index = 0; row_index < rows(); ++row_index)
                {
                    result(row_index, col_index) = (*this)(row_index, col_index);
                }
            }
            return result;
        }

        Scalar sum() const
        {
            return nested().sum();
        }

        Scalar minCoeff() const
        {
            return nested().minCoeff();
        }

        Scalar maxCoeff() const
        {
            return nested().maxCoeff();
        }

        Scalar squaredNorm() const
        {
            return nested().squaredNorm();
        }

        Scalar norm() const
        {
            return nested().norm();
        }

        Scalar trace() const
        {
            return nested().trace();
        }

        bool allFinite() const
        {
            return nested().allFinite();
        }

        template <typename Other> Scalar dot(const Other& other) const
        {
            if ((rows() != 1 && cols() != 1) || (other.rows() != 1 && other.cols() != 1) || size() != other.size())
            {
                throw internal::Error(internal::ErrorCode::InvalidArgument,
                                      "Transpose dot() requires vectors of equal length");
            }
            Scalar result{};
            for (Index index = 0; index < size(); ++index)
            {
                result += (*this)(index)*other(index);
            }
            return result;
        }

    private:
        template <typename Other> void assign(const Other& other)
        {
            static_assert(!read_only, "Cannot assign through a const Transpose expression");
            if (rows() != other.rows() || cols() != other.cols())
            {
                throw internal::Error(internal::ErrorCode::InvalidArgument,
                                      "Transpose assignment requires equal shapes");
            }
            std::vector<Scalar> snapshot(static_cast<std::size_t>(size()));
            for (Index col_index = 0; col_index < cols(); ++col_index)
            {
                for (Index row_index = 0; row_index < rows(); ++row_index)
                {
                    snapshot[static_cast<std::size_t>(row_index + col_index * rows())] = other(row_index, col_index);
                }
            }
            for (Index col_index = 0; col_index < cols(); ++col_index)
            {
                for (Index row_index = 0; row_index < rows(); ++row_index)
                {
                    (*this)(row_index, col_index) = snapshot[static_cast<std::size_t>(row_index + col_index * rows())];
                }
            }
        }

        static StoredExpression makeStorage(MatrixType& matrix)
        {
            if constexpr (stores_reference)
            {
                return &matrix;
            }
            else
            {
                return matrix;
            }
        }

        decltype(auto) nested() noexcept
        {
            if constexpr (stores_reference)
            {
                return (*_matrix);
            }
            else
            {
                return (_matrix);
            }
        }

        decltype(auto) nested() const noexcept
        {
            if constexpr (stores_reference)
            {
                return (*_matrix);
            }
            else
            {
                return static_cast<const std::remove_const_t<MatrixType>&>(_matrix);
            }
        }

        StoredExpression _matrix;
    };

    template <typename XprType, int BlockRows, int BlockCols, bool InnerPanel>
    Transpose<Block<XprType, BlockRows, BlockCols, InnerPanel>>
    Block<XprType, BlockRows, BlockCols, InnerPanel>::transpose()
    {
        return Transpose<Block>(*this);
    }

    template <typename XprType, int BlockRows, int BlockCols, bool InnerPanel>
    const Transpose<const Block<XprType, BlockRows, BlockCols, InnerPanel>>
    Block<XprType, BlockRows, BlockCols, InnerPanel>::transpose() const
    {
        return Transpose<const Block>(*this);
    }

    template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    Block<Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>, Dynamic, Dynamic>
    Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>::block(Index row, Index col, Index rows, Index cols)
    {
        return Block<Matrix, Dynamic, Dynamic>(*this, row, col, rows, cols);
    }

    template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    const Block<const Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>, Dynamic, Dynamic>
    Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>::block(Index row, Index col, Index rows, Index cols) const
    {
        return Block<const Matrix, Dynamic, Dynamic>(*this, row, col, rows, cols);
    }

    template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    template <int BlockRows, int BlockCols>
    Block<Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>, BlockRows, BlockCols>
    Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>::block(Index row, Index col)
    {
        return Block<Matrix, BlockRows, BlockCols>(*this, row, col);
    }

    template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    template <int BlockRows, int BlockCols>
    const Block<const Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>, BlockRows, BlockCols>
    Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>::block(Index row, Index col) const
    {
        return Block<const Matrix, BlockRows, BlockCols>(*this, row, col);
    }

    template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    Block<Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>, 1, Cols, (Options & RowMajor) == RowMajor>
    Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>::row(Index index)
    {
        return Block<Matrix, 1, Cols, IsRowMajor != 0>(*this, index);
    }

    template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    const Block<const Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>, 1, Cols, (Options & RowMajor) == RowMajor>
    Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>::row(Index index) const
    {
        return Block<const Matrix, 1, Cols, IsRowMajor != 0>(*this, index);
    }

    template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    Block<Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>, Rows, 1, (Options & RowMajor) != RowMajor>
    Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>::col(Index index)
    {
        return Block<Matrix, Rows, 1, IsRowMajor == 0>(*this, index);
    }

    template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    const Block<const Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>, Rows, 1, (Options & RowMajor) != RowMajor>
    Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>::col(Index index) const
    {
        return Block<const Matrix, Rows, 1, IsRowMajor == 0>(*this, index);
    }

    template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    Transpose<Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>>
    Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>::transpose()
    {
        return Transpose<Matrix>(*this);
    }

    template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    const Transpose<const Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>>
    Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>::transpose() const
    {
        return Transpose<const Matrix>(*this);
    }
} // namespace plamatrix::v1
