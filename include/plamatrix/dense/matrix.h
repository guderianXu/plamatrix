#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "plamatrix/internal/core/execution_policy.h"
#include "plamatrix/dense/matrix_base.h"
#include "plamatrix/dense/detail/aligned_allocator.h"
#include "plamatrix/internal/dense/auto_backend.h"
#include "plamatrix/internal/dense/matrix_view.h"
#include "plamatrix/internal/dense/packet_reduction.h"

namespace plamatrix::internal
{
    // Implementation evaluation and diagnostics: deliberately separate from the Eigen-style matrix contract.
    struct MatrixAccess
    {
        template <typename MatrixType, typename Scalar> static auto scaled(const MatrixType& value, Scalar scale)
        {
            return value.scaled(scale);
        }
        template <typename MatrixType> static bool isDeviceResident(const MatrixType& value)
        {
            return value.isDeviceResident();
        }
        template <typename MatrixType> static Backend deviceBackend(const MatrixType& value)
        {
            return value.deviceBackend();
        }
        template <typename MatrixType> static decltype(auto) executionInfo(const MatrixType& value)
        {
            return value.executionInfo();
        }
        template <typename MatrixType> static std::size_t pendingHostDownloadBytes(const MatrixType& value)
        {
            return value.pendingHostDownloadBytes();
        }
    };
} // namespace plamatrix::internal

namespace plamatrix::detail
{
    enum class MatrixBinaryOp;
} // namespace plamatrix::detail

namespace plamatrix
{

    inline namespace v1
    {
        namespace expression_detail
        {
            enum class BinaryKind;
            template <BinaryKind, typename, typename> class BinaryNode;
        } // namespace expression_detail

        template <typename Scalar,
                  int LeftRows,
                  int LeftCols,
                  int LeftOptions,
                  int LeftMaxRows,
                  int LeftMaxCols,
                  int RightRows,
                  int RightCols,
                  int RightOptions,
                  int RightMaxRows,
                  int RightMaxCols>
        Matrix<Scalar, LeftRows, RightCols>
        multiplyEager(const Matrix<Scalar, LeftRows, LeftCols, LeftOptions, LeftMaxRows, LeftMaxCols>& left,
                      const Matrix<Scalar, RightRows, RightCols, RightOptions, RightMaxRows, RightMaxCols>& right);

        template <typename Scalar_, int Rows, int Cols, int Options_, int MaxRows, int MaxCols>
        class Matrix : public MatrixBase<Matrix<Scalar_, Rows, Cols, Options_, MaxRows, MaxCols>>
        {
        public:
            using Base = MatrixBase<Matrix>;
            using Scalar = Scalar_;
            using Base::dot;
            using Base::isApprox;
            using Base::maxCoeff;
            using Base::minCoeff;

        private:
            static_assert(Rows == Dynamic || Rows >= 0, "Matrix rows must be non-negative or Dynamic");
            static_assert(Cols == Dynamic || Cols >= 0, "Matrix columns must be non-negative or Dynamic");
            static_assert(MaxRows == Dynamic || MaxRows >= 0, "Matrix maximum rows must be non-negative or Dynamic");
            static_assert(MaxCols == Dynamic || MaxCols >= 0, "Matrix maximum columns must be non-negative or Dynamic");
            static_assert(Rows == Dynamic || MaxRows == Dynamic || Rows <= MaxRows,
                          "Matrix rows exceed MaxRowsAtCompileTime");
            static_assert(Cols == Dynamic || MaxCols == Dynamic || Cols <= MaxCols,
                          "Matrix columns exceed MaxColsAtCompileTime");
            static_assert((Options_ & ~(RowMajor | DontAlign)) == 0, "Matrix has unsupported storage options");
            static_assert(!(Rows == 1 && Cols != 1) || (Options_ & RowMajor) == RowMajor,
                          "Row vectors must use RowMajor storage");
            static_assert(!(Cols == 1 && Rows != 1) || (Options_ & RowMajor) == 0,
                          "Column vectors must use ColMajor storage");

            static constexpr bool fixed_size = Rows != Dynamic && Cols != Dynamic;
            static constexpr bool fixed_capacity = MaxRows != Dynamic && MaxCols != Dynamic;
            static constexpr bool row_major_storage = (Options_ & RowMajor) == RowMajor;
            static constexpr std::size_t fixed_count =
                fixed_size ? static_cast<std::size_t>(Rows) * static_cast<std::size_t>(Cols) : 0;
            static constexpr std::size_t capacity_count =
                fixed_capacity ? static_cast<std::size_t>(MaxRows) * static_cast<std::size_t>(MaxCols) : 0;
            static constexpr std::size_t packet_alignment = alignof(Scalar) > std::size_t{64} ? alignof(Scalar)
                                                                                              : std::size_t{64};
            using DynamicAllocator = std::conditional_t<(Options_ & DontAlign) == DontAlign,
                                                        storage_detail::DefaultInitAllocator<Scalar>,
                                                        storage_detail::AlignedAllocator<Scalar, packet_alignment>>;
            using HostStorage = std::conditional_t<fixed_capacity,
                                                   std::array<Scalar, capacity_count>,
                                                   std::vector<Scalar, DynamicAllocator>>;

        public:
            using PlainObject = Matrix;
            using ScalarType = Scalar_;
            using ScalarValue = Scalar_;
            using RealScalar = typename NumTraits<Scalar_>::Real;
            using Index = plamatrix::Index;
            static constexpr int RowsAtCompileTime = Rows;
            static constexpr int ColsAtCompileTime = Cols;
            static constexpr int MaxRowsAtCompileTime = MaxRows;
            static constexpr int MaxColsAtCompileTime = MaxCols;
            static constexpr int Options = Options_;
            static constexpr int IsRowMajor = row_major_storage ? 1 : 0;
            static constexpr int IsVectorAtCompileTime = Rows == 1 || Cols == 1;
            static constexpr int SizeAtCompileTime = Rows == Dynamic || Cols == Dynamic ? Dynamic : Rows * Cols;
            static constexpr int MaxSizeAtCompileTime =
                MaxRows == Dynamic || MaxCols == Dynamic ? Dynamic : MaxRows * MaxCols;
            static constexpr int InnerSizeAtCompileTime =
                IsVectorAtCompileTime ? SizeAtCompileTime : (row_major_storage ? Cols : Rows);

            /// Fills scalars or rectangular blocks from left to right and top to bottom.
            class CommaInitializer
            {
            public:
                CommaInitializer(const CommaInitializer&) = delete;
                CommaInitializer& operator=(const CommaInitializer&) = delete;
                CommaInitializer(CommaInitializer&&) = delete;
                CommaInitializer& operator=(CommaInitializer&&) = delete;

                ~CommaInitializer() noexcept(false)
                {
                    if (std::uncaught_exceptions() == 0 && (_row != _rows || _col != 0))
                    {
                        throw internal::Error(internal::ErrorCode::InvalidArgument,
                                              "Matrix comma initialization has too few coefficients");
                    }
                }

                CommaInitializer& operator,(Scalar value)
                {
                    append(value);
                    return *this;
                }

                template <typename Block, std::enable_if_t<!std::is_convertible_v<Block, Scalar>, int> = 0>
                CommaInitializer& operator,(const Block& block)
                {
                    appendBlock(block);
                    return *this;
                }

            private:
                friend class Matrix;

                CommaInitializer(Matrix& matrix, Scalar first)
                    : _data(matrix.data()), _rows(matrix.rows()), _cols(matrix.cols())
                {
                    append(first);
                }

                template <typename Block>
                CommaInitializer(Matrix& matrix, const Block& first)
                    : _data(matrix.data()), _rows(matrix.rows()), _cols(matrix.cols())
                {
                    appendBlock(first);
                }

                void append(Scalar value)
                {
                    prepare(1, 1);
                    _data[storageOffset(_row, _col, _rows, _cols)] = value;
                    advance(1);
                }

                template <typename Block> void appendBlock(const Block& block)
                {
                    prepare(block.rows(), block.cols());
                    std::vector<Scalar> snapshot(static_cast<std::size_t>(block.rows() * block.cols()));
                    for (Index col = 0; col < block.cols(); ++col)
                    {
                        for (Index row = 0; row < block.rows(); ++row)
                        {
                            snapshot[static_cast<std::size_t>(row + col * block.rows())] = block(row, col);
                        }
                    }
                    for (Index col = 0; col < block.cols(); ++col)
                    {
                        for (Index row = 0; row < block.rows(); ++row)
                        {
                            _data[storageOffset(_row + row, _col + col, _rows, _cols)] =
                                snapshot[static_cast<std::size_t>(row + col * block.rows())];
                        }
                    }
                    advance(block.cols());
                }

                void prepare(Index block_rows, Index block_cols)
                {
                    if (block_rows <= 0 || block_cols <= 0 || _row >= _rows || block_rows > _rows - _row ||
                        block_cols > _cols - _col || (_col != 0 && block_rows != _height))
                    {
                        throw internal::Error(internal::ErrorCode::InvalidArgument,
                                              "Matrix comma initialization block does not fit");
                    }
                    if (_col == 0)
                    {
                        _height = block_rows;
                    }
                }

                void advance(Index columns)
                {
                    _col += columns;
                    if (_col == _cols)
                    {
                        _row += _height;
                        _col = 0;
                        _height = 0;
                    }
                }

                Scalar* _data;
                Index _rows;
                Index _cols;
                Index _row = 0;
                Index _col = 0;
                Index _height = 0;
            };

            /// Starts Eigen-style scalar comma initialization; an empty matrix is rejected.
            CommaInitializer operator<<(Scalar first)
            {
                return CommaInitializer(*this, first);
            }

            template <typename Block, std::enable_if_t<!std::is_convertible_v<Block, Scalar>, int> = 0>
            CommaInitializer operator<<(const Block& first)
            {
                return CommaInitializer(*this, first);
            }

            /// Assign a deferred expression without input/output aliasing. CPU writes directly;
            /// CUDA keeps the result resident, using a new device buffer.
            class NoAliasProxy
            {
            public:
                explicit NoAliasProxy(Matrix& target) : _target(target)
                {
                }

                template <typename Node> NoAliasProxy& operator=(const DenseExpression<Node>& expression)
                {
                    expression.evaluateInto(_target, true);
                    return *this;
                }

            private:
                Matrix& _target;
            };

            NoAliasProxy noalias()
            {
                return NoAliasProxy(*this);
            }

            /// Creates a lazy row or column reduction/broadcast proxy.
            auto rowwise() const&;
            auto rowwise() &&;
            auto rowwise() const&&;
            auto colwise() const&;
            auto colwise() &&;
            auto colwise() const&&;

            /// Creates a lazy coefficientwise proxy.
            auto array() const&;
            auto array() &&;
            auto array() const&&;

            Matrix() : _rows(Rows == Dynamic ? 0 : Rows), _cols(Cols == Dynamic ? 0 : Cols)
            {
            }

            Matrix(Index rows, Index cols) : _rows(rows), _cols(cols)
            {
                validateShape(rows, cols);
                if constexpr (!fixed_capacity)
                {
                    _host.resize(checkedCount(rows, cols));
                }
            }

            template <int R = Rows,
                      int C = Cols,
                      std::enable_if_t<(R == Dynamic && C == 1) || (R == 1 && C == Dynamic), int> = 0>
            explicit Matrix(Index length) : Matrix(R == Dynamic ? length : 1, C == Dynamic ? length : 1)
            {
            }

            template <
                typename... Values,
                std::enable_if_t<fixed_size && (fixed_count > 0) && (Rows == 1 || Cols == 1) &&
                                     sizeof...(Values) == fixed_count && (std::is_convertible_v<Values, Scalar> && ...),
                                 int> = 0>
            explicit Matrix(Values... values) : Matrix()
            {
                _host = {static_cast<Scalar>(values)...};
            }

            Matrix(const Matrix& other) : Matrix(other.rows(), other.cols())
            {
                std::copy_n(other.data(), size(), _host.data());
                _info.reason = "Deep copy materialized on the host";
            }

            template <bool IsConst>
            Matrix(const internal::BasicMatrixView<Scalar, internal::Device::CPU, IsConst>& source)
                : Matrix(source.rows(), source.cols())
            {
                for (Index col = 0; col < cols(); ++col)
                {
                    for (Index row = 0; row < rows(); ++row)
                    {
                        (*this)(row, col) = source(row, col);
                    }
                }
            }

            template <
                int OtherRows,
                int OtherCols,
                int OtherOptions,
                int OtherMaxRows,
                int OtherMaxCols,
                std::enable_if_t<
                    !std::is_same_v<Matrix,
                                    Matrix<Scalar, OtherRows, OtherCols, OtherOptions, OtherMaxRows, OtherMaxCols>>,
                    int> = 0>
            Matrix(const Matrix<Scalar, OtherRows, OtherCols, OtherOptions, OtherMaxRows, OtherMaxCols>& other)
                : Matrix(other.rows(), other.cols())
            {
                for (Index col = 0; col < cols(); ++col)
                {
                    for (Index row = 0; row < rows(); ++row)
                    {
                        (*this)(row, col) = other(row, col);
                    }
                }
                _info.reason = "Shape-checked copy materialized on the host";
            }

            template <typename Node>
            Matrix(const DenseExpression<Node>& expression) : Matrix(expression.rows(), expression.cols())
            {
                expression.evaluateInto(*this);
            }

            template <typename Derived,
                      std::enable_if_t<!std::is_same_v<std::remove_const_t<Derived>, Matrix> &&
                                           std::is_same_v<typename Derived::Scalar, Scalar>,
                                       int> = 0>
            Matrix(const MatrixBase<Derived>& expression) : Matrix(expression.rows(), expression.cols())
            {
                for (Index col = 0; col < cols(); ++col)
                {
                    for (Index row = 0; row < rows(); ++row)
                    {
                        (*this)(row, col) = expression(row, col);
                    }
                }
                _info.reason = "Dense expression materialized on the host";
            }

            Matrix& operator=(const Matrix& other)
            {
                if (this != &other)
                {
                    Matrix copy(other);
                    *this = std::move(copy);
                }
                return *this;
            }

            template <int OtherRows, int OtherCols, int OtherOptions, int OtherMaxRows, int OtherMaxCols>
            Matrix&
            operator=(const Matrix<Scalar, OtherRows, OtherCols, OtherOptions, OtherMaxRows, OtherMaxCols>& other)
            {
                Matrix copy(other);
                *this = std::move(copy);
                return *this;
            }

            template <typename Node> Matrix& operator=(const DenseExpression<Node>& expression)
            {
                Matrix result(expression);
                *this = std::move(result);
                return *this;
            }

            template <typename Derived,
                      std::enable_if_t<!std::is_same_v<std::remove_const_t<Derived>, Matrix> &&
                                           std::is_same_v<typename Derived::Scalar, Scalar>,
                                       int> = 0>
            Matrix& operator=(const MatrixBase<Derived>& expression)
            {
                Matrix result(expression);
                *this = std::move(result);
                return *this;
            }

            Matrix(Matrix&&) noexcept = default;
            Matrix& operator=(Matrix&&) noexcept = default;

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
            Index innerSize() const noexcept
            {
                return row_major_storage ? _cols : _rows;
            }
            Index outerSize() const noexcept
            {
                return row_major_storage ? _rows : _cols;
            }
            Index innerStride() const noexcept
            {
                return 1;
            }
            Index outerStride() const noexcept
            {
                return innerSize();
            }
            void resize(Index rows, Index cols)
            {
                validateShape(rows, cols);
                if (rows == _rows && cols == _cols)
                {
                    return;
                }
                if constexpr (!fixed_capacity)
                {
                    HostStorage replacement(checkedCount(rows, cols));
                    _host.swap(replacement);
                }
                _rows = rows;
                _cols = cols;
                _hostValid = true;
                _gpu.reset();
                _info = {};
            }

            /// Resize while preserving coefficients in the overlapping top-left rectangle.
            void conservativeResize(Index rows, Index cols)
            {
                validateShape(rows, cols);
                if (rows == _rows && cols == _cols)
                {
                    return;
                }
                Matrix replacement(rows, cols);
                replacement.setZero();
                const Scalar* source = data();
                Scalar* destination = replacement.data();
                for (Index col = 0; col < std::min(cols, _cols); ++col)
                {
                    for (Index row = 0; row < std::min(rows, _rows); ++row)
                    {
                        destination[storageOffset(row, col, rows, cols)] =
                            source[storageOffset(row, col, _rows, _cols)];
                    }
                }
                *this = std::move(replacement);
            }

            template <int R = Rows,
                      int C = Cols,
                      std::enable_if_t<(R == Dynamic && C == 1) || (R == 1 && C == Dynamic), int> = 0>
            void resize(Index length)
            {
                resize(R == Dynamic ? length : 1, C == Dynamic ? length : 1);
            }

        private:
            friend struct internal::MatrixAccess;
            bool isDeviceResident() const noexcept
            {
                return static_cast<bool>(_gpu);
            }
            internal::Backend deviceBackend() const noexcept
            {
                return _gpu ? internal::detail::GpuOps<Scalar>::backend(*_gpu) : internal::Backend::Cpu;
            }
            std::size_t pendingHostDownloadBytes() const noexcept
            {
                return pendingDownloadBytes();
            }
            const internal::ExecutionInfo& executionInfo() const noexcept
            {
                return _info;
            }

        public:
            const Scalar* data() const
            {
                materializeHost();
                return _host.data();
            }

            Scalar* data()
            {
                materializeHost();
                _gpu.reset();
                return _host.data();
            }

            const Scalar& operator()(Index row, Index col) const
            {
                return data()[checkedOffset(row, col)];
            }

            Scalar& operator()(Index row, Index col)
            {
                return data()[checkedOffset(row, col)];
            }

            const Scalar& coeff(Index row, Index col) const
            {
                return data()[storageOffset(row, col, _rows, _cols)];
            }

            Scalar& coeffRef(Index row, Index col)
            {
                return data()[storageOffset(row, col, _rows, _cols)];
            }

            const Scalar& operator()(Index index) const
            {
                return data()[checkedVectorIndex(index)];
            }

            Scalar& operator()(Index index)
            {
                return data()[checkedVectorIndex(index)];
            }

            const Scalar& coeff(Index index) const
            {
                return data()[static_cast<std::size_t>(index)];
            }
            Scalar& coeffRef(Index index)
            {
                return data()[static_cast<std::size_t>(index)];
            }

            const Scalar& operator[](Index index) const
            {
                return data()[checkedVectorIndex(index)];
            }
            Scalar& operator[](Index index)
            {
                return data()[checkedVectorIndex(index)];
            }

            const Scalar& x() const
            {
                return (*this)[0];
            }
            Scalar& x()
            {
                return (*this)[0];
            }
            const Scalar& y() const
            {
                return (*this)[1];
            }
            Scalar& y()
            {
                return (*this)[1];
            }
            const Scalar& z() const
            {
                return (*this)[2];
            }
            Scalar& z()
            {
                return (*this)[2];
            }
            const Scalar& w() const
            {
                return (*this)[3];
            }
            Scalar& w()
            {
                return (*this)[3];
            }

        public:
            Block<Matrix, Dynamic, Dynamic> block(Index row, Index col, Index rows, Index cols);
            const Block<const Matrix, Dynamic, Dynamic> block(Index row, Index col, Index rows, Index cols) const;

            template <int BlockRows, int BlockCols> Block<Matrix, BlockRows, BlockCols> block(Index row, Index col);

            template <int BlockRows, int BlockCols>
            const Block<const Matrix, BlockRows, BlockCols> block(Index row, Index col) const;

            Block<Matrix, 1, Cols, (Options_ & RowMajor) == RowMajor> row(Index index);
            const Block<const Matrix, 1, Cols, (Options_ & RowMajor) == RowMajor> row(Index index) const;
            Block<Matrix, Rows, 1, (Options_ & RowMajor) != RowMajor> col(Index index);
            const Block<const Matrix, Rows, 1, (Options_ & RowMajor) != RowMajor> col(Index index) const;

            void setConstant(Scalar value)
            {
                if constexpr (!fixed_capacity)
                {
                    _host.resize(checkedCount(_rows, _cols));
                }
                std::fill_n(_host.data(), static_cast<std::size_t>(size()), value);
                _hostValid = true;
                _gpu.reset();
                _info = {};
            }

            void setZero()
            {
                setConstant(Scalar{});
            }
            void setOnes()
            {
                setConstant(Scalar{1});
            }

            void setIdentity()
            {
                setZero();
                for (Index index = 0; index < std::min(_rows, _cols); ++index)
                {
                    _host[storageOffset(index, index, _rows, _cols)] = Scalar{1};
                }
            }

            static Matrix Zero(Index rows, Index cols)
            {
                Matrix result(rows, cols);
                result.setZero();
                return result;
            }

            static Matrix Ones(Index rows, Index cols)
            {
                Matrix result(rows, cols);
                result.setOnes();
                return result;
            }

            static Matrix Constant(Index rows, Index cols, Scalar value)
            {
                Matrix result(rows, cols);
                result.setConstant(value);
                return result;
            }

            static Matrix Identity(Index rows, Index cols)
            {
                Matrix result(rows, cols);
                result.setIdentity();
                return result;
            }

            /// Fill floating-point coefficients uniformly in [-1, 1].
            static Matrix Random(Index rows, Index cols)
            {
                static_assert(std::is_floating_point_v<Scalar>, "Random() requires a floating-point scalar");
                Matrix result(rows, cols);
                thread_local std::mt19937 engine(std::random_device{}());
                std::uniform_real_distribution<Scalar> distribution(Scalar{-1}, Scalar{1});
                for (Index index = 0; index < result.size(); ++index)
                {
                    result.data()[index] = distribution(engine);
                }
                return result;
            }

            template <bool Enabled = fixed_size, std::enable_if_t<Enabled, int> = 0> static Matrix Random()
            {
                return Random(Rows, Cols);
            }

            template <bool Enabled = fixed_size, std::enable_if_t<Enabled, int> = 0> static Matrix Zero()
            {
                return Zero(Rows, Cols);
            }

            template <bool Enabled = fixed_size, std::enable_if_t<Enabled, int> = 0> static Matrix Ones()
            {
                return Ones(Rows, Cols);
            }

            template <bool Enabled = fixed_size, std::enable_if_t<Enabled, int> = 0> static Matrix Identity()
            {
                return Identity(Rows, Cols);
            }

            template <bool Enabled = fixed_size, std::enable_if_t<Enabled, int> = 0>
            static Matrix Constant(Scalar value)
            {
                return Constant(Rows, Cols, value);
            }

            template <int R = Rows,
                      int C = Cols,
                      std::enable_if_t<(R == Dynamic && C == 1) || (R == 1 && C == Dynamic), int> = 0>
            static Matrix Zero(Index length)
            {
                Matrix result(length);
                result.setZero();
                return result;
            }

            template <bool Enabled = fixed_size && (Rows == 1 || Cols == 1), std::enable_if_t<Enabled, int> = 0>
            static Matrix Unit(Index index)
            {
                Matrix result = Zero();
                if (index < 0 || index >= result.size())
                {
                    throw internal::Error(internal::ErrorCode::InvalidArgument, "Unit() index is out of range");
                }
                result[index] = Scalar{1};
                return result;
            }

            template <bool Enabled = fixed_size && (Rows == 1 || Cols == 1) && fixed_count >= 1,
                      std::enable_if_t<Enabled, int> = 0>
            static Matrix UnitX()
            {
                return Unit(0);
            }

            template <bool Enabled = fixed_size && (Rows == 1 || Cols == 1) && fixed_count >= 2,
                      std::enable_if_t<Enabled, int> = 0>
            static Matrix UnitY()
            {
                return Unit(1);
            }

            template <bool Enabled = fixed_size && (Rows == 1 || Cols == 1) && fixed_count >= 3,
                      std::enable_if_t<Enabled, int> = 0>
            static Matrix UnitZ()
            {
                return Unit(2);
            }

            template <int R = Rows,
                      int C = Cols,
                      std::enable_if_t<(R == Dynamic && C == 1) || (R == 1 && C == Dynamic), int> = 0>
            static Matrix Ones(Index length)
            {
                Matrix result(length);
                result.setOnes();
                return result;
            }

            template <int R = Rows,
                      int C = Cols,
                      std::enable_if_t<(R == Dynamic && C == 1) || (R == 1 && C == Dynamic), int> = 0>
            static Matrix Constant(Index length, Scalar value)
            {
                Matrix result(length);
                result.setConstant(value);
                return result;
            }

            Transpose<Matrix> transpose();
            const Transpose<const Matrix> transpose() const;
            template <typename Other> auto cwiseProduct(Other&& other) const&;
            template <typename Other> auto cwiseProduct(Other&& other) &&;
            template <typename Other> auto cwiseProduct(Other&& other) const&&;

        private:
            Matrix scaled(Scalar value) const;
            Matrix add(const Matrix& other) const;
            Matrix subtract(const Matrix& other) const;

        public:
            Matrix& operator+=(const Matrix& other)
            {
                *this = add(other);
                return *this;
            }

            Matrix& operator-=(const Matrix& other)
            {
                *this = subtract(other);
                return *this;
            }

            template <typename Node> Matrix& operator+=(const DenseExpression<Node>& expression)
            {
                *this = *this + expression;
                return *this;
            }

            template <typename Node> Matrix& operator-=(const DenseExpression<Node>& expression)
            {
                *this = *this - expression;
                return *this;
            }

            Matrix& operator*=(Scalar value)
            {
                *this = scaled(value);
                return *this;
            }

            Matrix& operator/=(Scalar value)
            {
                if (value == Scalar{})
                {
                    throw internal::Error(internal::ErrorCode::InvalidArgument, "Matrix scalar division by zero");
                }
                *this = *this / value;
                return *this;
            }

            Scalar dot(const Matrix& other) const
            {
                if (!isVector() || !other.isVector() || size() != other.size())
                {
                    throw internal::Error(internal::ErrorCode::InvalidArgument,
                                          "dot() requires vectors of equal length");
                }
                if constexpr (!row_major_storage && std::is_floating_point_v<Scalar>)
                {
                    const internal::Backend backend = reductionBackend(&other);
                    if (backend != internal::Backend::Cpu && size() != 0)
                        return internal::detail::GpuOps<Scalar>::dot(gpuStorage(backend), other.gpuStorage(backend));
                }
                Scalar result{};
                const Scalar* left = data();
                const Scalar* right = other.data();
                if constexpr (NumTraits<Scalar>::IsComplex)
                {
                    for (Index index = 0; index < size(); ++index)
                    {
                        result += std::conj(left[index]) * right[index];
                    }
                }
                else if constexpr (std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double>)
                {
                    result = internal::detail::packetDot(left, right, size());
                }
                else
                {
                    for (Index index = 0; index < size(); ++index)
                        result += left[index] * right[index];
                }
                return result;
            }

            template <int OtherRows, int OtherCols, int OtherOptions, int OtherMaxRows, int OtherMaxCols>
            Scalar
            dot(const Matrix<Scalar, OtherRows, OtherCols, OtherOptions, OtherMaxRows, OtherMaxCols>& other) const
            {
                if (!isVector() || !other.isVector() || size() != other.size())
                {
                    throw internal::Error(internal::ErrorCode::InvalidArgument,
                                          "dot() requires vectors of equal length");
                }
                constexpr bool other_row_major = (OtherOptions & RowMajor) == RowMajor;
                if constexpr (!row_major_storage && !other_row_major && std::is_floating_point_v<Scalar>)
                {
                    const internal::Backend backend = reductionBackend(&other);
                    if (backend != internal::Backend::Cpu && size() != 0)
                        return internal::detail::GpuOps<Scalar>::dot(gpuStorage(backend), other.gpuStorage(backend));
                }
                Scalar result{};
                const Scalar* left = data();
                const Scalar* right = other.data();
                if constexpr (NumTraits<Scalar>::IsComplex)
                {
                    for (Index index = 0; index < size(); ++index)
                        result += std::conj(left[index]) * right[index];
                }
                else if constexpr (std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double>)
                {
                    result = internal::detail::packetDot(left, right, size());
                }
                else
                {
                    for (Index index = 0; index < size(); ++index)
                        result += left[index] * right[index];
                }
                return result;
            }

            RealScalar norm() const
            {
                return static_cast<RealScalar>(std::sqrt(squaredNorm()));
            }

            RealScalar squaredNorm() const
            {
                if constexpr (!row_major_storage && std::is_floating_point_v<Scalar>)
                {
                    const internal::Backend backend = reductionBackend();
                    if (backend != internal::Backend::Cpu && size() != 0)
                        return internal::detail::GpuOps<Scalar>::dot(gpuStorage(backend), gpuStorage(backend));
                }
                RealScalar result{};
                const Scalar* values = data();
                if constexpr (NumTraits<Scalar>::IsComplex)
                {
                    for (Index index = 0; index < size(); ++index)
                    {
                        result += std::norm(values[index]);
                    }
                }
                else if constexpr (std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double>)
                {
                    result = internal::detail::packetSquaredNorm(values, size());
                }
                else
                {
                    for (Index index = 0; index < size(); ++index)
                        result += values[index] * values[index];
                }
                return result;
            }

            Scalar sum() const
            {
                if constexpr (!row_major_storage && std::is_floating_point_v<Scalar>)
                {
                    const internal::Backend backend = reductionBackend();
                    if (backend != internal::Backend::Cpu && size() != 0)
                        return internal::detail::GpuOps<Scalar>::sum(gpuStorage(backend));
                }
                Scalar result{};
                const Scalar* values = data();
                if constexpr (NumTraits<Scalar>::IsComplex)
                {
                    for (Index index = 0; index < size(); ++index)
                        result += values[index];
                }
                else if constexpr (std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double>)
                {
                    result = internal::detail::packetSum(values, size());
                }
                else
                {
                    for (Index index = 0; index < size(); ++index)
                        result += values[index];
                }
                return result;
            }

            Scalar minCoeff() const
            {
                if (size() == 0)
                {
                    throw internal::Error(internal::ErrorCode::InvalidArgument,
                                          "minCoeff() requires a nonempty matrix");
                }
                return *std::min_element(data(), data() + size());
            }

            Scalar maxCoeff() const
            {
                if (size() == 0)
                {
                    throw internal::Error(internal::ErrorCode::InvalidArgument,
                                          "maxCoeff() requires a nonempty matrix");
                }
                return *std::max_element(data(), data() + size());
            }

            Scalar trace() const
            {
                Scalar result{};
                for (Index index = 0; index < std::min(_rows, _cols); ++index)
                {
                    result += (*this)(index, index);
                }
                return result;
            }

            template <typename OtherScalar> Matrix<OtherScalar, Rows, Cols, Options_, MaxRows, MaxCols> cast() const
            {
                Matrix<OtherScalar, Rows, Cols, Options_, MaxRows, MaxCols> result(rows(), cols());
                const Scalar* source = data();
                OtherScalar* target = result.data();
                for (Index index = 0; index < size(); ++index)
                {
                    target[index] = static_cast<OtherScalar>(source[index]);
                }
                return result;
            }

            Matrix normalized() const
            {
                static_assert(std::is_floating_point_v<Scalar> || NumTraits<Scalar>::IsComplex,
                              "normalized() requires a floating-point scalar");
                const RealScalar length = norm();
                if (length == RealScalar{} || !std::isfinite(static_cast<double>(length)))
                {
                    throw internal::Error(internal::ErrorCode::InvalidArgument,
                                          "normalized() requires a finite nonzero vector");
                }
                return scaled(Scalar{1} / length);
            }

            void normalize()
            {
                *this = normalized();
            }

            /// Factor a square floating-point matrix once for one or more right-hand sides.
            template <typename PermutationIndex = DefaultPermutationIndex>
            PartialPivLU<Matrix, PermutationIndex> partialPivLu() const;
            template <typename PermutationIndex = DefaultPermutationIndex>
            PartialPivLU<Matrix, PermutationIndex> lu() const;
            template <typename PermutationIndex = DefaultPermutationIndex>
            FullPivLU<Matrix, PermutationIndex> fullPivLu() const;
            template <int UpLo = Lower> LLT<Matrix, UpLo> llt() const;
            LDLT<Matrix> ldlt() const;
            HouseholderQR<Matrix> householderQr() const;
            template <typename PermutationIndex = DefaultPermutationIndex>
            ColPivHouseholderQR<Matrix, PermutationIndex> colPivHouseholderQr() const;
            template <int Options = 0> JacobiSVD<Matrix, Options> jacobiSvd() const;
            template <int Options = 0> BDCSVD<Matrix, Options> bdcSvd() const;
            CompleteOrthogonalDecomposition<Matrix> completeOrthogonalDecomposition() const;

            template <int OtherRows, int OtherCols, int OtherOptions, int OtherMaxRows, int OtherMaxCols>
            Matrix<Scalar, 3, 1>
            cross(const Matrix<Scalar, OtherRows, OtherCols, OtherOptions, OtherMaxRows, OtherMaxCols>& other) const
            {
                if (!isVector() || !other.isVector() || size() != 3 || other.size() != 3)
                {
                    throw internal::Error(internal::ErrorCode::InvalidArgument,
                                          "cross() requires three-element vectors");
                }
                const Scalar* left = data();
                const Scalar* right = other.data();
                return Matrix<Scalar, 3, 1>(left[1] * right[2] - left[2] * right[1],
                                            left[2] * right[0] - left[0] * right[2],
                                            left[0] * right[1] - left[1] * right[0]);
            }

            bool allFinite() const
            {
                if constexpr (NumTraits<Scalar>::IsComplex)
                {
                    const Scalar* values = data();
                    for (Index index = 0; index < size(); ++index)
                    {
                        if (!std::isfinite(values[index].real()) || !std::isfinite(values[index].imag()))
                        {
                            return false;
                        }
                    }
                    return true;
                }
                else if constexpr (!std::is_floating_point_v<Scalar>)
                {
                    return true;
                }
                else
                {
                    const Scalar* values = data();
                    for (Index index = 0; index < size(); ++index)
                    {
                        if (!std::isfinite(values[index]))
                        {
                            return false;
                        }
                    }
                    return true;
                }
            }

        private:
            template <detail::MatrixBinaryOp Op> Matrix applyBinary(const Matrix& other) const;
            Matrix<Scalar, Cols, Rows> transposeMaterialized() const;
            Matrix cwiseProductEager(const Matrix& other) const;

            template <typename, int, int, int, int, int> friend class Matrix;
            template <typename, typename> friend class PartialPivLU;
            template <typename, typename> friend class FullPivLU;
            template <typename, int> friend class LLT;
            template <typename, int> friend class LDLT;
            template <typename> friend class HouseholderQR;
            template <typename, typename> friend class ColPivHouseholderQR;
            template <typename, int> friend class JacobiSVD;
            template <typename, int> friend class BDCSVD;
            template <typename> friend class DenseExpression;
            template <expression_detail::BinaryKind, typename, typename> friend class expression_detail::BinaryNode;
            template <typename S, int LR, int LC, int LO, int LMR, int LMC, int RR, int RC, int RO, int RMR, int RMC>
            friend Matrix<S, LR, RC> multiplyEager(const Matrix<S, LR, LC, LO, LMR, LMC>&,
                                                   const Matrix<S, RR, RC, RO, RMR, RMC>&);

            static std::size_t checkedCount(Index rows, Index cols)
            {
                if (rows < 0 || cols < 0 || (cols != 0 && rows > std::numeric_limits<Index>::max() / cols) ||
                    static_cast<std::uint64_t>(rows) * static_cast<std::uint64_t>(cols) >
                        std::numeric_limits<std::size_t>::max() / sizeof(Scalar))
                {
                    throw internal::Error(internal::ErrorCode::InvalidArgument,
                                          "Matrix shape exceeds addressable storage");
                }
                return static_cast<std::size_t>(rows * cols);
            }

            static void validateShape(Index rows, Index cols)
            {
                static_cast<void>(checkedCount(rows, cols));
                if ((Rows != Dynamic && rows != Rows) || (Cols != Dynamic && cols != Cols) ||
                    (MaxRows != Dynamic && rows > MaxRows) || (MaxCols != Dynamic && cols > MaxCols))
                {
                    throw internal::Error(internal::ErrorCode::InvalidArgument,
                                          "Matrix shape conflicts with its fixed dimensions");
                }
            }

            static std::size_t storageOffset(Index row, Index col, Index rows, Index cols) noexcept
            {
                return static_cast<std::size_t>(row_major_storage ? col + row * cols : row + col * rows);
            }

            std::size_t checkedOffset(Index row, Index col) const
            {
                if (row < 0 || col < 0 || row >= _rows || col >= _cols)
                {
                    throw std::out_of_range("Matrix coefficient index is out of range");
                }
                return storageOffset(row, col, _rows, _cols);
            }

            std::size_t checkedVectorIndex(Index index) const
            {
                if (!isVector() || index < 0 || index >= size())
                {
                    throw std::out_of_range("Vector coefficient index is out of range");
                }
                return static_cast<std::size_t>(index);
            }

            bool isVector() const noexcept
            {
                return _rows == 1 || _cols == 1;
            }

            std::size_t pendingDownloadBytes() const noexcept
            {
                return (!_hostValid && _gpu != nullptr) ? static_cast<std::size_t>(size()) * sizeof(Scalar) : 0;
            }

            template <typename OtherMatrix = Matrix>
            internal::Backend reductionBackend(const OtherMatrix* other = nullptr) const
            {
                const auto settings = internal::currentExecutionSettings();
                if (settings.policy == internal::ExecutionPolicy::CpuOnly)
                    return internal::Backend::Cpu;
                if (settings.policy == internal::ExecutionPolicy::Auto)
                {
                    const internal::Backend resident =
                        isDeviceResident() ? deviceBackend()
                                           : (other != nullptr && other->isDeviceResident() ? other->deviceBackend()
                                                                                            : internal::Backend::Cpu);
                    return resident != internal::Backend::Cpu && internal::detail::GpuOps<Scalar>::available(resident)
                               ? resident
                               : internal::Backend::Cpu;
                }
                if (internal::detail::GpuOps<Scalar>::available(settings.preferredGpu))
                    return settings.preferredGpu;
                if (settings.policy == internal::ExecutionPolicy::GpuRequired)
                    throw internal::Error(internal::ErrorCode::BackendUnavailable,
                                          "Selected GPU backend is unavailable for dense reduction",
                                          settings.preferredGpu);
                return internal::Backend::Cpu;
            }

            void materializeHost() const
            {
                if (_hostValid)
                {
                    return;
                }
                if (_gpu == nullptr)
                {
                    throw internal::Error(internal::ErrorCode::InvalidState,
                                          "Matrix has no valid host or device storage");
                }
                if constexpr (!fixed_capacity)
                {
                    _host.resize(checkedCount(_rows, _cols));
                }
                internal::detail::GpuOps<Scalar>::download(*_gpu, _host.data());
                _hostValid = true;
                _info.bytesDownloaded += static_cast<std::size_t>(size()) * sizeof(Scalar);
            }

            const internal::detail::GpuStorage<Scalar>& gpuStorage(internal::Backend backend) const
            {
                static_assert(!row_major_storage, "Dense GPU storage currently requires ColMajor matrices");
                if (_gpu != nullptr && internal::detail::GpuOps<Scalar>::backend(*_gpu) != backend)
                {
                    materializeHost();
                    _gpu.reset();
                }
                if (_gpu == nullptr)
                {
                    _gpu = internal::detail::GpuOps<Scalar>::upload(backend, data(), _rows, _cols);
                    _info.bytesUploaded += static_cast<std::size_t>(size()) * sizeof(Scalar);
                }
                return *_gpu;
            }

            void adoptGpu(std::shared_ptr<internal::detail::GpuStorage<Scalar>> storage, internal::ExecutionInfo info)
            {
                static_assert(!row_major_storage, "Dense GPU storage currently requires ColMajor matrices");
                _gpu = std::move(storage);
                _hostValid = false;
                if constexpr (!fixed_capacity)
                {
                    _host.clear();
                    _host.shrink_to_fit();
                }
                _info = std::move(info);
            }

            Index _rows = 0;
            Index _cols = 0;
            mutable HostStorage _host{};
            mutable bool _hostValid = true;
            mutable std::shared_ptr<internal::detail::GpuStorage<Scalar>> _gpu;
            mutable internal::ExecutionInfo _info;
        };

        using Matrix2d = Matrix<double, 2, 2>;
        using Matrix2f = Matrix<float, 2, 2>;
        using Matrix2i = Matrix<int, 2, 2>;
        using Matrix3d = Matrix<double, 3, 3>;
        using Matrix3f = Matrix<float, 3, 3>;
        using Matrix3i = Matrix<int, 3, 3>;
        using Matrix4d = Matrix<double, 4, 4>;
        using Matrix4f = Matrix<float, 4, 4>;
        using Matrix4i = Matrix<int, 4, 4>;
        using MatrixXd = Matrix<double, Dynamic, Dynamic>;
        using MatrixXf = Matrix<float, Dynamic, Dynamic>;
        using MatrixXcd = Matrix<std::complex<double>, Dynamic, Dynamic>;
        using MatrixXcf = Matrix<std::complex<float>, Dynamic, Dynamic>;
        using MatrixXi = Matrix<int, Dynamic, Dynamic>;
        using Vector2d = Matrix<double, 2, 1>;
        using Vector2f = Matrix<float, 2, 1>;
        using Vector2i = Matrix<int, 2, 1>;
        using Vector3d = Matrix<double, 3, 1>;
        using Vector3f = Matrix<float, 3, 1>;
        using Vector3i = Matrix<int, 3, 1>;
        using Vector4d = Matrix<double, 4, 1>;
        using Vector4f = Matrix<float, 4, 1>;
        using Vector4i = Matrix<int, 4, 1>;
        using VectorXd = Matrix<double, Dynamic, 1>;
        using VectorXf = Matrix<float, Dynamic, 1>;
        using VectorXcd = Matrix<std::complex<double>, Dynamic, 1>;
        using VectorXcf = Matrix<std::complex<float>, Dynamic, 1>;
        using VectorXi = Matrix<int, Dynamic, 1>;
        using RowVector2d = Matrix<double, 1, 2>;
        using RowVector2f = Matrix<float, 1, 2>;
        using RowVector2i = Matrix<int, 1, 2>;
        using RowVector3d = Matrix<double, 1, 3>;
        using RowVector3f = Matrix<float, 1, 3>;
        using RowVector3i = Matrix<int, 1, 3>;
        using RowVector4d = Matrix<double, 1, 4>;
        using RowVector4f = Matrix<float, 1, 4>;
        using RowVector4i = Matrix<int, 1, 4>;
        using RowVectorXd = Matrix<double, 1, Dynamic>;
        using RowVectorXf = Matrix<float, 1, Dynamic>;
        using RowVectorXi = Matrix<int, 1, Dynamic>;

    } // namespace v1

} // namespace plamatrix

#include "plamatrix/dense/matrix_block.h"
#include "plamatrix/dense/matrix_transpose.h"
#include "plamatrix/dense/matrix_views.h"
#include "plamatrix/dense/matrix_ops.h"
#include "plamatrix/dense/matrix_solvers.h"
#include "plamatrix/dense/matrix_factorizations.h"
#include "plamatrix/dense/detail/matrix_self_adjoint_impl.h"
#include "plamatrix/dense/detail/matrix_map_impl.h"
#include "plamatrix/dense/matrix_expression.h"
#include "plamatrix/dense/matrix_cwise.h"
#include "plamatrix/dense/detail/matrix_reduction_methods.h"
