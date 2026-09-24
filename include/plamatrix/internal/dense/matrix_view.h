#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "plamatrix/internal/core/device.h"

namespace plamatrix::internal
{

    /// Non-owning two-dimensional matrix view with explicit row and column strides.
    /// The referenced storage must outlive the view and every derived subview.
    template <typename Scalar, Device Dev, bool IsConst> class BasicMatrixView
    {
    public:
        using ValueType = Scalar;
        using Pointer = std::conditional_t<IsConst, const Scalar*, Scalar*>;
        using Reference = std::conditional_t<IsConst, const Scalar&, Scalar&>;

        BasicMatrixView() noexcept = default;

        BasicMatrixView(Pointer data, Index rows, Index cols, Index row_stride, Index col_stride)
            : _data(data), _rows(rows), _cols(cols), _rowStride(row_stride), _colStride(col_stride)
        {
            validateLayout();
        }

        template <bool OtherConst, std::enable_if_t<IsConst && !OtherConst, int> = 0>
        BasicMatrixView(const BasicMatrixView<Scalar, Dev, OtherConst>& other) noexcept
            : _data(other.data()), _rows(other.rows()), _cols(other.cols()), _rowStride(other.rowStride()),
              _colStride(other.colStride())
        {
        }

        /// Copy coefficients into a mutable CPU view; a const or GPU view retains descriptor assignment.
        BasicMatrixView& operator=(const BasicMatrixView& other)
        {
            if constexpr (!IsConst && Dev == Device::CPU)
            {
                update(other, 0);
            }
            else
            {
                _data = other._data;
                _rows = other._rows;
                _cols = other._cols;
                _rowStride = other._rowStride;
                _colStride = other._colStride;
            }
            return *this;
        }

        /// Assign a matrix or const view with the same shape. Overlapping inputs are snapshotted.
        template <typename Other, bool Enabled = !IsConst && Dev == Device::CPU, std::enable_if_t<Enabled, int> = 0>
        BasicMatrixView& operator=(const Other& other)
        {
            update(other, 0);
            return *this;
        }

        /// Add a matrix or view coefficient-wise to this CPU view.
        template <typename Other, bool Enabled = !IsConst && Dev == Device::CPU, std::enable_if_t<Enabled, int> = 0>
        BasicMatrixView& operator+=(const Other& other)
        {
            update(other, 1);
            return *this;
        }

        /// Subtract a matrix or view coefficient-wise from this CPU view.
        template <typename Other, bool Enabled = !IsConst && Dev == Device::CPU, std::enable_if_t<Enabled, int> = 0>
        BasicMatrixView& operator-=(const Other& other)
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
        Index rowStride() const noexcept
        {
            return _rowStride;
        }
        Index colStride() const noexcept
        {
            return _colStride;
        }
        Pointer data() const noexcept
        {
            return _data;
        }

        static constexpr Device device() noexcept
        {
            return Dev;
        }
        static constexpr bool isConst() noexcept
        {
            return IsConst;
        }

        BasicMatrixView<Scalar, Dev, true> asConst() const noexcept
        {
            return BasicMatrixView<Scalar, Dev, true>(_data, _rows, _cols, _rowStride, _colStride);
        }

        bool isContiguousColumnMajor() const noexcept
        {
            return size() == 0 || (_rowStride == 1 && _colStride == _rows);
        }

        bool isContiguousRowMajor() const noexcept
        {
            return size() == 0 || (_rowStride == _cols && _colStride == 1);
        }

        Reference operator()(Index row, Index col) const
        {
            static_assert(Dev == Device::CPU, "MatrixView element access is only available for CPU storage");
            return _data[checkedOffset(row, col)];
        }

        Reference operator()(Index index) const
        {
            static_assert(Dev == Device::CPU, "MatrixView element access is only available for CPU storage");
            if (_cols == 1)
            {
                return (*this)(index, 0);
            }
            if (_rows == 1)
            {
                return (*this)(0, index);
            }
            throw std::invalid_argument("MatrixView single-index access requires a vector");
        }

        BasicMatrixView block(Index row, Index col, Index block_rows, Index block_cols) const
        {
            validateBlock(row, col, block_rows, block_cols);
            Pointer block_data = _data;
            if (block_rows != 0 && block_cols != 0)
            {
                block_data += row * _rowStride + col * _colStride;
            }
            return BasicMatrixView(block_data, block_rows, block_cols, _rowStride, _colStride);
        }

        BasicMatrixView row(Index row_index) const
        {
            if (row_index < 0 || row_index >= _rows)
            {
                throw std::out_of_range("MatrixView row index is out of range");
            }
            return block(row_index, 0, 1, _cols);
        }

        BasicMatrixView col(Index col_index) const
        {
            if (col_index < 0 || col_index >= _cols)
            {
                throw std::out_of_range("MatrixView column index is out of range");
            }
            return block(0, col_index, _rows, 1);
        }

        BasicMatrixView transpose() const noexcept
        {
            return BasicMatrixView(_data, _cols, _rows, _colStride, _rowStride, UncheckedTag{});
        }

    private:
        template <typename Other> void update(const Other& other, int operation)
        {
            static_assert(!IsConst && Dev == Device::CPU, "View coefficient updates require mutable CPU storage");
            if (_rows != other.rows() || _cols != other.cols())
            {
                throw std::invalid_argument("MatrixView coefficient update requires equal shapes");
            }
            std::array<Scalar, 16> small_values{};
            std::vector<Scalar> large_values;
            if (size() > static_cast<Index>(small_values.size()))
            {
                large_values.resize(static_cast<std::size_t>(size()));
            }
            Scalar* values = large_values.empty() ? small_values.data() : large_values.data();
            for (Index col = 0; col < _cols; ++col)
            {
                for (Index row = 0; row < _rows; ++row)
                {
                    values[static_cast<std::size_t>(row + col * _rows)] = other(row, col);
                }
            }
            for (Index col = 0; col < _cols; ++col)
            {
                for (Index row = 0; row < _rows; ++row)
                {
                    Scalar& destination = (*this)(row, col);
                    const Scalar value = values[static_cast<std::size_t>(row + col * _rows)];
                    if (operation == 0)
                    {
                        destination = value;
                    }
                    else if (operation > 0)
                    {
                        destination += value;
                    }
                    else
                    {
                        destination -= value;
                    }
                }
            }
        }

        struct UncheckedTag
        {
        };

        BasicMatrixView(Pointer data, Index rows, Index cols, Index row_stride, Index col_stride, UncheckedTag) noexcept
            : _data(data), _rows(rows), _cols(cols), _rowStride(row_stride), _colStride(col_stride)
        {
        }

        void validateLayout() const
        {
            if (_rows < 0 || _cols < 0)
            {
                throw std::invalid_argument("MatrixView dimensions must be non-negative");
            }
            if (_rows != 0 && _cols > std::numeric_limits<Index>::max() / _rows)
            {
                throw std::overflow_error("MatrixView element count overflows Index");
            }
            if (size() != 0 && _data == nullptr)
            {
                throw std::invalid_argument("MatrixView non-empty storage pointer must not be null");
            }
            if (size() != 0 && (_rowStride == 0 || _colStride == 0))
            {
                throw std::invalid_argument("MatrixView non-empty strides must be positive");
            }
            if (size() != 0)
            {
                const auto magnitude = [](Index value) -> std::uint64_t {
                    return value >= 0 ? static_cast<std::uint64_t>(value)
                                      : static_cast<std::uint64_t>(-(value + 1)) + 1;
                };
                const std::uint64_t maximum = static_cast<std::uint64_t>(std::numeric_limits<Index>::max());
                const std::uint64_t row_stride = magnitude(_rowStride);
                const std::uint64_t col_stride = magnitude(_colStride);
                if (static_cast<std::uint64_t>(_rows - 1) > maximum / row_stride ||
                    static_cast<std::uint64_t>(_cols - 1) > maximum / col_stride)
                {
                    throw std::overflow_error("MatrixView stride span overflows Index");
                }
                const std::uint64_t row_span = static_cast<std::uint64_t>(_rows - 1) * row_stride;
                const std::uint64_t col_span = static_cast<std::uint64_t>(_cols - 1) * col_stride;
                if (col_span > maximum - row_span)
                {
                    throw std::overflow_error("MatrixView stride span overflows Index");
                }
            }
        }

        Index checkedOffset(Index row, Index col) const
        {
            if (row < 0 || row >= _rows || col < 0 || col >= _cols)
            {
                std::ostringstream message;
                message << "MatrixView index out of range: (" << row << ", " << col << ") for view " << _rows << "x"
                        << _cols;
                throw std::out_of_range(message.str());
            }
            return row * _rowStride + col * _colStride;
        }

        void validateBlock(Index row, Index col, Index block_rows, Index block_cols) const
        {
            if (row < 0 || col < 0 || block_rows < 0 || block_cols < 0 || row > _rows || col > _cols ||
                block_rows > _rows - row || block_cols > _cols - col)
            {
                throw std::out_of_range("MatrixView block is outside the source view");
            }
        }

        Pointer _data = nullptr;
        Index _rows = 0;
        Index _cols = 0;
        Index _rowStride = 0;
        Index _colStride = 0;
    };

    template <typename Scalar, Device Dev> using MatrixView = BasicMatrixView<Scalar, Dev, false>;

    template <typename Scalar, Device Dev> using ConstMatrixView = BasicMatrixView<Scalar, Dev, true>;

    /// Non-owning view whose block shape remains visible to compile-time matrix operators.
    template <typename Scalar, int Rows, int Cols, bool IsConst>
    class FixedMatrixView : public BasicMatrixView<Scalar, Device::CPU, IsConst>
    {
        using Base = BasicMatrixView<Scalar, Device::CPU, IsConst>;

    public:
        using Base::operator=;
        static constexpr int RowsAtCompileTime = Rows;
        static constexpr int ColsAtCompileTime = Cols;

        explicit FixedMatrixView(Base view) : Base(view)
        {
            if ((Rows >= 0 && this->rows() != Rows) || (Cols >= 0 && this->cols() != Cols))
            {
                throw std::invalid_argument("FixedMatrixView shape does not match its compile-time dimensions");
            }
        }

        FixedMatrixView& operator=(const FixedMatrixView& other)
        {
            Base::operator=(static_cast<const Base&>(other));
            return *this;
        }

        FixedMatrixView<Scalar, Cols, Rows, IsConst> transpose() const
        {
            return FixedMatrixView<Scalar, Cols, Rows, IsConst>(Base::transpose());
        }
    };

    template <typename Scalar, Device Dev>
    MatrixView<Scalar, Dev> makeMatrixView(Scalar* data, Index rows, Index cols, Index row_stride, Index col_stride)
    {
        return MatrixView<Scalar, Dev>(data, rows, cols, row_stride, col_stride);
    }

    template <typename Scalar, Device Dev>
    ConstMatrixView<Scalar, Dev>
    makeConstMatrixView(const Scalar* data, Index rows, Index cols, Index row_stride, Index col_stride)
    {
        return ConstMatrixView<Scalar, Dev>(data, rows, cols, row_stride, col_stride);
    }

    template <typename Scalar, Device Dev>
    MatrixView<Scalar, Dev> makeColumnMajorView(Scalar* data, Index rows, Index cols)
    {
        return makeMatrixView<Scalar, Dev>(data, rows, cols, 1, rows);
    }

    template <typename Scalar, Device Dev>
    ConstMatrixView<Scalar, Dev> makeColumnMajorView(const Scalar* data, Index rows, Index cols)
    {
        return makeConstMatrixView<Scalar, Dev>(data, rows, cols, 1, rows);
    }

    template <typename Scalar, Device Dev>
    MatrixView<Scalar, Dev> makeRowMajorView(Scalar* data, Index rows, Index cols)
    {
        return makeMatrixView<Scalar, Dev>(data, rows, cols, cols, 1);
    }

    template <typename Scalar, Device Dev>
    ConstMatrixView<Scalar, Dev> makeRowMajorView(const Scalar* data, Index rows, Index cols)
    {
        return makeConstMatrixView<Scalar, Dev>(data, rows, cols, cols, 1);
    }

} // namespace plamatrix::internal
