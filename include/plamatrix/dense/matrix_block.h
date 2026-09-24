#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace plamatrix::v1
{
    /// Eigen-compatible non-owning rectangular expression over a dense lvalue.
    template <typename XprType, int BlockRows, int BlockCols, bool InnerPanel>
    class Block : public MatrixBase<Block<XprType, BlockRows, BlockCols, InnerPanel>>
    {
        using NestedTraits = detail::DenseTraits<std::remove_const_t<XprType>>;
        using NestedPointer = decltype(std::declval<XprType&>().data());
        static constexpr bool read_only =
            std::is_const_v<XprType> || std::is_const_v<std::remove_pointer_t<NestedPointer>>;
        static constexpr bool nested_row_major = (NestedTraits::StorageOptions & RowMajor) == RowMajor;
        static constexpr int max_rows = BlockRows == Dynamic ? NestedTraits::MaxRowsAtCompileTime : BlockRows;
        static constexpr int max_cols = BlockCols == Dynamic ? NestedTraits::MaxColsAtCompileTime : BlockCols;

    public:
        using Base = MatrixBase<Block>;
        using Scalar = typename NestedTraits::ScalarType;
        using Base::isApprox;
        using Base::maxCoeff;
        using Base::minCoeff;
        using RealScalar = typename NumTraits<Scalar>::Real;
        using NestedExpression = std::remove_const_t<XprType>;
        using Pointer = std::conditional_t<read_only, const Scalar*, Scalar*>;
        using Reference = std::conditional_t<read_only, const Scalar&, Scalar&>;
        using PlainObject = Matrix<Scalar, BlockRows, BlockCols>;
        using Index = plamatrix::Index;

        static constexpr int RowsAtCompileTime = BlockRows;
        static constexpr int ColsAtCompileTime = BlockCols;
        static constexpr int MaxRowsAtCompileTime = max_rows;
        static constexpr int MaxColsAtCompileTime = max_cols;
        static constexpr int IsRowMajor =
            max_rows == 1 && max_cols != 1 ? 1 : (max_cols == 1 && max_rows != 1 ? 0 : nested_row_major);
        static constexpr int Options = IsRowMajor ? RowMajor : ColMajor;
        static constexpr int IsVectorAtCompileTime = BlockRows == 1 || BlockCols == 1;
        static constexpr int SizeAtCompileTime =
            BlockRows == Dynamic || BlockCols == Dynamic ? Dynamic : BlockRows * BlockCols;
        static constexpr bool InnerPanelAtCompileTime = InnerPanel;

        Block(XprType& expression, Index index) : _expression(&expression)
        {
            if constexpr (BlockRows == 1 && BlockCols == NestedTraits::ColsAtCompileTime)
            {
                initialize(index, 0, 1, expression.cols());
            }
            else
            {
                static_assert(BlockCols == 1 && BlockRows == NestedTraits::RowsAtCompileTime,
                              "Single-index Block construction requires a row or column expression");
                initialize(0, index, expression.rows(), 1);
            }
        }

        Block(XprType& expression, Index start_row, Index start_col) : _expression(&expression)
        {
            static_assert(BlockRows != Dynamic && BlockCols != Dynamic,
                          "Two-index Block construction requires fixed block dimensions");
            initialize(start_row, start_col, BlockRows, BlockCols);
        }

        Block(XprType& expression, Index start_row, Index start_col, Index rows, Index cols) : _expression(&expression)
        {
            initialize(start_row, start_col, rows, cols);
        }

        Block(const Block&) = default;

        Block& operator=(const Block& other)
        {
            if (this != &other)
            {
                assign(other);
            }
            return *this;
        }

        template <typename Other, bool Enabled = !read_only, std::enable_if_t<Enabled, int> = 0>
        Block& operator=(const Other& other)
        {
            assign(other);
            return *this;
        }

        template <typename Other, bool Enabled = !read_only, std::enable_if_t<Enabled, int> = 0>
        Block& operator+=(const Other& other)
        {
            update(other, 1);
            return *this;
        }

        template <typename Other, bool Enabled = !read_only, std::enable_if_t<Enabled, int> = 0>
        Block& operator-=(const Other& other)
        {
            update(other, -1);
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

        Index innerStride() const noexcept
        {
            return IsRowMajor ? _colStride : _rowStride;
        }

        Index outerStride() const noexcept
        {
            return IsRowMajor ? _rowStride : _colStride;
        }

        Pointer data() const noexcept
        {
            return _data;
        }

        Reference operator()(Index row, Index col) const
        {
            validateIndex(row, col);
            return _data[row * _rowStride + col * _colStride];
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
            throw internal::Error(internal::ErrorCode::InvalidArgument, "Single-index Block access requires a vector");
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

        XprType& nestedExpression() const noexcept
        {
            return *_expression;
        }

        Block<Block, Dynamic, Dynamic> block(Index row, Index col, Index rows, Index cols)
        {
            return Block<Block, Dynamic, Dynamic>(*this, row, col, rows, cols);
        }

        const Block<const Block, Dynamic, Dynamic> block(Index row, Index col, Index rows, Index cols) const
        {
            return Block<const Block, Dynamic, Dynamic>(*this, row, col, rows, cols);
        }

        template <int Rows, int Cols> Block<Block, Rows, Cols> block(Index row, Index col)
        {
            return Block<Block, Rows, Cols>(*this, row, col);
        }

        template <int Rows, int Cols> const Block<const Block, Rows, Cols> block(Index row, Index col) const
        {
            return Block<const Block, Rows, Cols>(*this, row, col);
        }

        Block<Block, 1, BlockCols, IsRowMajor != 0> row(Index index)
        {
            return Block<Block, 1, BlockCols, IsRowMajor != 0>(*this, index);
        }

        const Block<const Block, 1, BlockCols, IsRowMajor != 0> row(Index index) const
        {
            return Block<const Block, 1, BlockCols, IsRowMajor != 0>(*this, index);
        }

        Block<Block, BlockRows, 1, IsRowMajor == 0> col(Index index)
        {
            return Block<Block, BlockRows, 1, IsRowMajor == 0>(*this, index);
        }

        const Block<const Block, BlockRows, 1, IsRowMajor == 0> col(Index index) const
        {
            return Block<const Block, BlockRows, 1, IsRowMajor == 0>(*this, index);
        }

        Transpose<Block> transpose();
        const Transpose<const Block> transpose() const;

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
            Scalar result{};
            forEach([&result](Scalar value) { result += value; });
            return result;
        }

        Scalar minCoeff() const
        {
            requireNonempty("minCoeff");
            Scalar result = (*this)(0, 0);
            forEach([&result](Scalar value) { result = std::min(result, value); });
            return result;
        }

        Scalar maxCoeff() const
        {
            requireNonempty("maxCoeff");
            Scalar result = (*this)(0, 0);
            forEach([&result](Scalar value) { result = std::max(result, value); });
            return result;
        }

        Scalar squaredNorm() const
        {
            Scalar result{};
            forEach([&result](Scalar value) { result += value * value; });
            return result;
        }

        Scalar norm() const
        {
            return static_cast<Scalar>(std::sqrt(squaredNorm()));
        }

        Scalar trace() const
        {
            Scalar result{};
            for (Index index = 0; index < std::min(rows(), cols()); ++index)
            {
                result += (*this)(index, index);
            }
            return result;
        }

        bool allFinite() const
        {
            bool result = true;
            forEach([&result](Scalar value) { result = result && std::isfinite(value); });
            return result;
        }

        template <typename Other> Scalar dot(const Other& other) const
        {
            if ((rows() != 1 && cols() != 1) || (other.rows() != 1 && other.cols() != 1) || size() != other.size())
            {
                throw internal::Error(internal::ErrorCode::InvalidArgument,
                                      "Block dot() requires vectors of equal length");
            }
            Scalar result{};
            for (Index index = 0; index < size(); ++index)
            {
                result += (*this)(index)*other(index);
            }
            return result;
        }

    private:
        void initialize(Index start_row, Index start_col, Index rows, Index cols)
        {
            if ((BlockRows != Dynamic && rows != BlockRows) || (BlockCols != Dynamic && cols != BlockCols))
            {
                throw internal::Error(internal::ErrorCode::InvalidArgument,
                                      "Block runtime shape conflicts with its compile-time shape");
            }
            if (start_row < 0 || start_col < 0 || rows < 0 || cols < 0 || start_row > _expression->rows() ||
                start_col > _expression->cols() || rows > _expression->rows() - start_row ||
                cols > _expression->cols() - start_col)
            {
                throw std::out_of_range("Block is outside the source expression");
            }
            const Index nested_inner = _expression->innerStride();
            const Index nested_outer = _expression->outerStride();
            _rowStride = nested_row_major ? nested_outer : nested_inner;
            _colStride = nested_row_major ? nested_inner : nested_outer;
            _data = _expression->data();
            if (rows != 0 && cols != 0)
            {
                _data += start_row * _rowStride + start_col * _colStride;
            }
            _rows = rows;
            _cols = cols;
        }

        void validateIndex(Index row, Index col) const
        {
            if (row < 0 || col < 0 || row >= rows() || col >= cols())
            {
                throw std::out_of_range("Block coefficient index is out of range");
            }
        }

        template <typename Other> void assign(const Other& other)
        {
            static_assert(!read_only, "Cannot assign through a const Block expression");
            update(other, 0);
        }

        template <typename Other> void update(const Other& other, int operation)
        {
            static_assert(!read_only, "Cannot update through a const Block expression");
            if (rows() != other.rows() || cols() != other.cols())
            {
                throw std::invalid_argument("Block assignment requires equal shapes");
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
                    Scalar& target = (*this)(row_index, col_index);
                    const Scalar value = snapshot[static_cast<std::size_t>(row_index + col_index * rows())];
                    target = operation == 0 ? value : (operation > 0 ? target + value : target - value);
                }
            }
        }

        template <typename Function> void forEach(Function&& function) const
        {
            for (Index col_index = 0; col_index < cols(); ++col_index)
            {
                for (Index row_index = 0; row_index < rows(); ++row_index)
                {
                    function((*this)(row_index, col_index));
                }
            }
        }

        void requireNonempty(const char* operation) const
        {
            if (size() == 0)
            {
                throw internal::Error(internal::ErrorCode::InvalidArgument,
                                      std::string(operation) + "() requires a nonempty block");
            }
        }

        XprType* _expression;
        Pointer _data = nullptr;
        Index _rows = 0;
        Index _cols = 0;
        Index _rowStride = 0;
        Index _colStride = 0;
    };
} // namespace plamatrix::v1
