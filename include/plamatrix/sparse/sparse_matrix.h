#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <tuple>
#include <utility>
#include <vector>

#include "plamatrix/core/types.h"
#include "plamatrix/dense/matrix.h"
#include "plamatrix/internal/sparse/sparse_ops.h"

namespace plamatrix::internal
{
    struct SparseAccess;
}

namespace plamatrix::v1
{
    template <typename Derived> class SparseMatrixBase : public EigenBase<Derived>
    {
    public:
        using EigenBase<Derived>::derived;

        Index nonZeros() const
        {
            return derived().nonZeros();
        }

        decltype(auto) coeff(Index row, Index col) const
        {
            return derived().coeff(row, col);
        }
    };

    /// Read-only sparse expression over a dense matrix. The dense object must outlive this view.
    template <typename MatrixType> class SparseView : public SparseMatrixBase<SparseView<MatrixType>>
    {
        using Nested = std::remove_const_t<MatrixType>;
        using Traits = detail::DenseTraits<Nested>;

    public:
        using Scalar = typename Traits::ScalarType;
        using RealScalar = Scalar;
        using StorageIndex = int;
        static constexpr int IsRowMajor = (Traits::StorageOptions & RowMajor) == RowMajor;

        class InnerIterator
        {
        public:
            InnerIterator() = default;
            InnerIterator(const SparseView& view, Index outer) : _view(&view), _outer(outer)
            {
                if (outer < 0 || outer >= view.outerSize())
                {
                    throw std::out_of_range("SparseView outer index is out of bounds");
                }
                advance();
            }

            explicit operator bool() const noexcept
            {
                return _inner < _view->innerSize();
            }

            InnerIterator& operator++()
            {
                ++_inner;
                advance();
                return *this;
            }

            Index row() const noexcept
            {
                return IsRowMajor ? _outer : _inner;
            }

            Index col() const noexcept
            {
                return IsRowMajor ? _inner : _outer;
            }

            Index index() const noexcept
            {
                return _inner;
            }

            Index outer() const noexcept
            {
                return _outer;
            }

            Scalar value() const
            {
                return _view->coeff(row(), col());
            }

        private:
            void advance()
            {
                while (_inner < _view->innerSize() && !_view->keeps(row(), col()))
                {
                    ++_inner;
                }
            }

            const SparseView* _view = nullptr;
            Index _outer = 0;
            Index _inner = 0;
        };

        SparseView(const MatrixType& matrix, const Scalar& reference, const RealScalar& epsilon)
            : _matrix(&matrix), _reference(reference), _epsilon(epsilon)
        {
            if (!std::isfinite(_reference) || !std::isfinite(_epsilon) || _epsilon < RealScalar{})
            {
                throw std::invalid_argument("SparseView reference and epsilon must be finite and epsilon non-negative");
            }
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

        Index outerSize() const noexcept
        {
            return IsRowMajor ? rows() : cols();
        }

        Index innerSize() const noexcept
        {
            return IsRowMajor ? cols() : rows();
        }

        Index nonZeros() const
        {
            Index count = 0;
            for (Index outer = 0; outer < outerSize(); ++outer)
            {
                for (InnerIterator entry(*this, outer); entry; ++entry)
                {
                    ++count;
                }
            }
            return count;
        }

        Scalar coeff(Index row, Index col) const
        {
            return static_cast<Scalar>(_matrix->coeff(row, col));
        }

        const MatrixType& nestedExpression() const noexcept
        {
            return *_matrix;
        }

    private:
        bool keeps(Index row, Index col) const
        {
            return std::abs(coeff(row, col)) > std::abs(_reference) * _epsilon;
        }

        const MatrixType* _matrix;
        Scalar _reference;
        RealScalar _epsilon;
    };

    template <typename Scalar, typename StorageIndex> class Triplet
    {
    public:
        Triplet(StorageIndex row, StorageIndex col, const Scalar& value = Scalar{})
            : _row(row), _col(col), _value(value)
        {
        }

        StorageIndex row() const noexcept
        {
            return _row;
        }
        StorageIndex col() const noexcept
        {
            return _col;
        }
        const Scalar& value() const noexcept
        {
            return _value;
        }

    private:
        StorageIndex _row;
        StorageIndex _col;
        Scalar _value;
    };

    /// CPU sparse matrix with Eigen-style insertion and contiguous compressed primary storage.
    template <typename Scalar_, int Options, typename StorageIndex_>
    class SparseMatrix : public SparseMatrixBase<SparseMatrix<Scalar_, Options, StorageIndex_>>
    {
    public:
        using Scalar = Scalar_;
        using ScalarType = Scalar_;
        using RealScalar = Scalar_;
        using Index = plamatrix::Index;
        using StorageIndex = StorageIndex_;
        using StorageIndexType = StorageIndex_;
        using PlainObject = SparseMatrix;
        static constexpr int IsRowMajor = Options == RowMajor;

    private:
        using Coordinate = std::pair<Index, Index>;
        using Entry = std::pair<Coordinate, Scalar>;

        static_assert(std::is_floating_point_v<Scalar>, "SparseMatrix currently supports float and double");
        static_assert(Options == ColMajor || Options == RowMajor, "unsupported SparseMatrix storage option");
        static_assert(std::is_integral_v<StorageIndex> && std::is_signed_v<StorageIndex>,
                      "SparseMatrix storage index must be a signed integral type");

    public:
        class InnerIterator
        {
        public:
            InnerIterator() = default;
            InnerIterator(const SparseMatrix& matrix, Index outer) : _matrix(&matrix), _outer(outer)
            {
                if (outer < 0 || outer >= matrix.outerSize())
                {
                    throw std::out_of_range("SparseMatrix outer index is out of bounds");
                }
                matrix.prepareIteration();
                const auto& entries = matrix._iterationEntries;
                const auto begin = std::lower_bound(entries.begin(),
                                                    entries.end(),
                                                    outer,
                                                    [](const Entry& entry, Index value)
                                                    { return SparseMatrix::outerIndex(entry) < value; });
                const auto end = std::upper_bound(begin,
                                                  entries.end(),
                                                  outer,
                                                  [](Index value, const Entry& entry)
                                                  { return value < SparseMatrix::outerIndex(entry); });
                _position = static_cast<std::size_t>(begin - entries.begin());
                _end = static_cast<std::size_t>(end - entries.begin());
            }

            explicit operator bool() const noexcept
            {
                return _position < _end;
            }
            InnerIterator& operator++()
            {
                ++_position;
                return *this;
            }
            Index row() const
            {
                return entry().first.first;
            }
            Index col() const
            {
                return entry().first.second;
            }
            Index index() const
            {
                return IsRowMajor ? col() : row();
            }
            Index outer() const noexcept
            {
                return _outer;
            }
            Scalar value() const
            {
                return _matrix->coeff(row(), col());
            }

        private:
            const Entry& entry() const
            {
                return _matrix->_iterationEntries[_position];
            }
            const SparseMatrix* _matrix = nullptr;
            Index _outer = 0;
            std::size_t _position = 0;
            std::size_t _end = 0;
        };

        SparseMatrix() = default;
        SparseMatrix(Index rows, Index cols)
        {
            resize(rows, cols);
        }

        template <typename MatrixType> SparseMatrix(const SparseView<MatrixType>& view)
        {
            assign(view);
        }
        template <int OtherOptions, typename OtherStorageIndex>
        SparseMatrix(const SparseMatrix<Scalar, OtherOptions, OtherStorageIndex>& other)
        {
            assign(other);
        }

        SparseMatrix(const SparseMatrix& other)
        {
            other.validateEscapedCompressedStorage();
            _rows = other._rows;
            _cols = other._cols;
            _assemblyEntries = other._assemblyEntries;
            _values = other._values;
            _innerIndices = other._innerIndices;
            _outerIndices = other._outerIndices;
            _isCompressed = other._isCompressed;
        }
        SparseMatrix& operator=(const SparseMatrix& other)
        {
            if (this != &other)
            {
                SparseMatrix copy(other);
                swap(copy);
            }
            return *this;
        }
        SparseMatrix(SparseMatrix&& other) noexcept
        {
            swap(other);
        }
        SparseMatrix& operator=(SparseMatrix&& other) noexcept
        {
            if (this != &other)
            {
                SparseMatrix moved(std::move(other));
                swap(moved);
            }
            return *this;
        }

        void swap(SparseMatrix& other) noexcept
        {
            std::swap(_rows, other._rows);
            std::swap(_cols, other._cols);
            _assemblyEntries.swap(other._assemblyEntries);
            _values.swap(other._values);
            _innerIndices.swap(other._innerIndices);
            _outerIndices.swap(other._outerIndices);
            std::swap(_isCompressed, other._isCompressed);
            std::swap(_mutableReferencesEscaped, other._mutableReferencesEscaped);
            std::swap(_rawPointersEscaped, other._rawPointersEscaped);
            std::swap(_backInsertionActive, other._backInsertionActive);
            std::swap(_backInsertionOuter, other._backInsertionOuter);
            std::swap(_backInsertionInner, other._backInsertionInner);
            _iterationEntries.swap(other._iterationEntries);
            std::swap(_iterationReady, other._iterationReady);
            _entrySnapshot.swap(other._entrySnapshot);
            std::swap(_entrySnapshotReady, other._entrySnapshotReady);
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
        Index outerSize() const noexcept
        {
            return IsRowMajor ? _rows : _cols;
        }
        Index innerSize() const noexcept
        {
            return IsRowMajor ? _cols : _rows;
        }
        Index nonZeros() const
        {
            validateEscapedCompressedStorage();
            return static_cast<Index>(_isCompressed && !_mutableReferencesEscaped ? _values.size()
                                                                                  : _assemblyEntries.size());
        }

        void resize(Index rows, Index cols)
        {
            if (rows < 0 || cols < 0)
                throw std::invalid_argument("SparseMatrix dimensions must be non-negative");
            _rows = rows;
            _cols = cols;
            setZero();
        }
        void setZero()
        {
            _assemblyEntries.clear();
            _values.clear();
            _innerIndices.clear();
            _outerIndices.clear();
            _isCompressed = false;
            _mutableReferencesEscaped = false;
            _rawPointersEscaped = false;
            resetBackInsertion();
            invalidateSnapshots();
        }
        void setIdentity()
        {
            setZero();
            _assemblyEntries.reserve(static_cast<std::size_t>(std::min(_rows, _cols)));
            for (Index index = 0; index < std::min(_rows, _cols); ++index)
                _assemblyEntries.push_back({{index, index}, Scalar{1}});
            makeCompressed();
        }

        template <typename InputIterator> void setFromTriplets(InputIterator begin, InputIterator end)
        {
            setFromTriplets(begin, end, [](const Scalar& left, const Scalar& right) { return left + right; });
        }
        template <typename InputIterator, typename DupFunctor>
        void setFromTriplets(InputIterator begin, InputIterator end, DupFunctor duplicate)
        {
            SparseMatrix result(_rows, _cols);
            for (auto it = begin; it != end; ++it)
            {
                const Index row = static_cast<Index>(it->row());
                const Index col = static_cast<Index>(it->col());
                checkCoordinate(row, col);
                result._assemblyEntries.push_back({{row, col}, static_cast<Scalar>(it->value())});
            }
            result.sortAndCombineAssembly(std::move(duplicate));
            result.makeCompressed();
            swap(result);
        }

        template <typename MatrixType> SparseMatrix& operator=(const SparseView<MatrixType>& view)
        {
            assign(view);
            return *this;
        }
        template <int OtherOptions, typename OtherStorageIndex>
        SparseMatrix& operator=(const SparseMatrix<Scalar, OtherOptions, OtherStorageIndex>& other)
        {
            assign(other);
            return *this;
        }

        Scalar coeff(Index row, Index col) const
        {
            checkCoordinate(row, col);
            validateEscapedCompressedStorage();
            if (_isCompressed && !_mutableReferencesEscaped)
            {
                const Index outer = IsRowMajor ? row : col;
                const StorageIndex inner =
                    checkedStorageIndex(IsRowMajor ? col : row, "SparseMatrix inner index overflow");
                const auto first =
                    _innerIndices.begin() + static_cast<Index>(_outerIndices[static_cast<std::size_t>(outer)]);
                const auto last =
                    _innerIndices.begin() + static_cast<Index>(_outerIndices[static_cast<std::size_t>(outer + 1)]);
                const auto found = std::lower_bound(first, last, inner);
                return found != last && *found == inner
                           ? _values[static_cast<std::size_t>(found - _innerIndices.begin())]
                           : Scalar{};
            }
            const auto found = findAssembly(row, col);
            return found != _assemblyEntries.end() && found->first == Coordinate{row, col} ? found->second : Scalar{};
        }
        Scalar operator()(Index row, Index col) const
        {
            return coeff(row, col);
        }

        Scalar& coeffRef(Index row, Index col)
        {
            checkCoordinate(row, col);
            uncompress();
            auto found = findAssembly(row, col);
            if (found == _assemblyEntries.end() || found->first != Coordinate{row, col})
                found = _assemblyEntries.insert(found, Entry{{row, col}, Scalar{}});
            _mutableReferencesEscaped = true;
            invalidateNumericalCaches();
            return found->second;
        }
        Scalar& insert(Index row, Index col)
        {
            checkCoordinate(row, col);
            if (coeff(row, col) != Scalar{} || coordinateExists(row, col))
                throw std::invalid_argument("SparseMatrix::insert requires a new coordinate");
            return coeffRef(row, col);
        }

        void startVec(Index outer)
        {
            if (outer < 0 || outer >= outerSize())
                throw std::out_of_range("SparseMatrix::startVec outer index is out of bounds");
            uncompress();
            if (!_backInsertionActive)
            {
                if (!_assemblyEntries.empty() || outer != 0)
                    throw std::logic_error("SparseMatrix::startVec requires an empty matrix and starts at outer 0");
                _backInsertionActive = true;
                _backInsertionOuter = 0;
            }
            else if (outer != _backInsertionOuter + 1)
            {
                throw std::logic_error("SparseMatrix::startVec outer indices must be consecutive");
            }
            else
            {
                _backInsertionOuter = outer;
            }
            _backInsertionInner = -1;
        }
        Scalar& insertBack(Index row, Index col)
        {
            checkCoordinate(row, col);
            const Index outer = IsRowMajor ? row : col;
            const Index inner = IsRowMajor ? col : row;
            if (!_backInsertionActive || outer != _backInsertionOuter)
                throw std::logic_error("SparseMatrix::insertBack requires startVec() for the current outer index");
            if (inner <= _backInsertionInner)
                throw std::logic_error("SparseMatrix::insertBack inner indices must be strictly increasing");
            _backInsertionInner = inner;
            _assemblyEntries.push_back({{row, col}, Scalar{}});
            _mutableReferencesEscaped = true;
            invalidateNumericalCaches();
            return _assemblyEntries.back().second;
        }
        Scalar& insertBackByOuterInner(Index outer, Index inner)
        {
            return IsRowMajor ? insertBack(outer, inner) : insertBack(inner, outer);
        }
        void finalize()
        {
            makeCompressed();
            resetBackInsertion();
        }

        void makeCompressed() const
        {
            if (_isCompressed && !_mutableReferencesEscaped)
            {
                validateEscapedCompressedStorage();
                return;
            }
            auto* self = const_cast<SparseMatrix*>(this);
            self->sortAndCombineAssembly([](const Scalar& left, const Scalar& right) { return left + right; });
            self->_values.resize(self->_assemblyEntries.size());
            self->_innerIndices.resize(self->_assemblyEntries.size());
            self->_outerIndices.assign(static_cast<std::size_t>(outerSize() + 1), StorageIndex{});
            for (const auto& entry : self->_assemblyEntries)
                ++self->_outerIndices[static_cast<std::size_t>(outerIndex(entry) + 1)];
            for (Index outer = 0; outer < outerSize(); ++outer)
            {
                const Index next = static_cast<Index>(self->_outerIndices[static_cast<std::size_t>(outer)]) +
                                   static_cast<Index>(self->_outerIndices[static_cast<std::size_t>(outer + 1)]);
                self->_outerIndices[static_cast<std::size_t>(outer + 1)] =
                    checkedStorageIndex(next, "SparseMatrix nonzero count exceeds StorageIndex range");
            }
            for (std::size_t position = 0; position < self->_assemblyEntries.size(); ++position)
            {
                const auto& entry = self->_assemblyEntries[position];
                self->_values[position] = entry.second;
                self->_innerIndices[position] =
                    checkedStorageIndex(innerIndex(entry), "SparseMatrix inner index overflow");
            }
            if (!self->_mutableReferencesEscaped)
            {
                self->_assemblyEntries.clear();
                self->_assemblyEntries.shrink_to_fit();
            }
            self->_isCompressed = true;
            self->_rawPointersEscaped = false;
            self->resetBackInsertion();
            self->invalidateNumericalCaches();
        }
        bool isCompressed() const noexcept
        {
            return _isCompressed;
        }

        void reserve(Index non_zeros)
        {
            if (non_zeros < 0)
                throw std::invalid_argument("SparseMatrix::reserve count must be non-negative");
            uncompress();
            _assemblyEntries.reserve(static_cast<std::size_t>(non_zeros));
        }
        template <typename SizesType, std::enable_if_t<!std::is_arithmetic_v<SizesType>, int> = 0>
        void reserve(const SizesType& sizes)
        {
            if (sizes.size() != outerSize())
                throw std::invalid_argument("SparseMatrix::reserve per-vector sizes must match outerSize()");
            Index total = 0;
            for (Index outer = 0; outer < outerSize(); ++outer)
            {
                if (sizes(outer) < 0)
                    throw std::invalid_argument("SparseMatrix::reserve sizes must be non-negative");
                if (sizes(outer) > std::numeric_limits<Index>::max() - total)
                    throw std::overflow_error("SparseMatrix::reserve total exceeds Index range");
                total += sizes(outer);
            }
            reserve(total);
        }
        void uncompress()
        {
            validateEscapedCompressedStorage();
            if (!_isCompressed)
                return;
            if (!_mutableReferencesEscaped)
                rebuildAssemblyFromCompressed();
            _values.clear();
            _innerIndices.clear();
            _outerIndices.clear();
            _isCompressed = false;
            _rawPointersEscaped = false;
            _mutableReferencesEscaped = false;
            resetBackInsertion();
            invalidateNumericalCaches();
        }

        const Scalar* valuePtr() const
        {
            makeCompressed();
            return _values.data();
        }
        Scalar* valuePtr()
        {
            makeCompressed();
            _assemblyEntries.clear();
            _assemblyEntries.shrink_to_fit();
            _mutableReferencesEscaped = false;
            _rawPointersEscaped = true;
            invalidateSnapshots();
            return _values.data();
        }
        const StorageIndex* innerIndexPtr() const
        {
            makeCompressed();
            return _innerIndices.data();
        }
        StorageIndex* innerIndexPtr()
        {
            makeCompressed();
            _assemblyEntries.clear();
            _assemblyEntries.shrink_to_fit();
            _mutableReferencesEscaped = false;
            _rawPointersEscaped = true;
            invalidateSnapshots();
            return _innerIndices.data();
        }
        const StorageIndex* outerIndexPtr() const
        {
            makeCompressed();
            return _outerIndices.data();
        }
        StorageIndex* outerIndexPtr()
        {
            makeCompressed();
            _assemblyEntries.clear();
            _assemblyEntries.shrink_to_fit();
            _mutableReferencesEscaped = false;
            _rawPointersEscaped = true;
            invalidateSnapshots();
            return _outerIndices.data();
        }
        void squeeze()
        {
            makeCompressed();
            _values.shrink_to_fit();
            _innerIndices.shrink_to_fit();
            _outerIndices.shrink_to_fit();
        }

        Scalar sum() const
        {
            validateEscapedCompressedStorage();
            Scalar result{};
            if (_isCompressed && !_mutableReferencesEscaped)
                for (const Scalar& value : _values)
                    result += value;
            else
                for (const auto& entry : _assemblyEntries)
                    result += entry.second;
            return result;
        }
        template <typename KeepFunctor> void prune(KeepFunctor keep)
        {
            uncompress();
            _assemblyEntries.erase(std::remove_if(_assemblyEntries.begin(),
                                                  _assemblyEntries.end(),
                                                  [&](const Entry& entry) {
                                                      return !keep(entry.first.first, entry.first.second, entry.second);
                                                  }),
                                   _assemblyEntries.end());
            _mutableReferencesEscaped = false;
            invalidateNumericalCaches();
        }
        void prune(const Scalar& reference, const RealScalar& epsilon = std::numeric_limits<RealScalar>::epsilon())
        {
            if (!std::isfinite(reference) || !std::isfinite(epsilon) || epsilon < RealScalar{})
                throw std::invalid_argument("SparseMatrix prune parameters are invalid");
            prune([&](Index, Index, const Scalar& value) { return std::abs(value) > std::abs(reference) * epsilon; });
        }
        template <typename KeepFunctor> SparseMatrix pruned(KeepFunctor keep) const
        {
            SparseMatrix result(*this);
            result.prune(std::move(keep));
            return result;
        }
        SparseMatrix pruned(const Scalar& reference = Scalar{},
                            const RealScalar& epsilon = std::numeric_limits<RealScalar>::epsilon()) const
        {
            SparseMatrix result(*this);
            result.prune(reference, epsilon);
            return result;
        }

        using TransposeReturnType = SparseMatrix<Scalar, IsRowMajor ? ColMajor : RowMajor, StorageIndex>;
        TransposeReturnType transpose() const
        {
            TransposeReturnType result(_cols, _rows);
            result.reserve(nonZeros());
            for (Index outer = 0; outer < outerSize(); ++outer)
                for (InnerIterator entry(*this, outer); entry; ++entry)
                    result.coeffRef(entry.col(), entry.row()) = entry.value();
            result.makeCompressed();
            return result;
        }
        TransposeReturnType adjoint() const
        {
            return transpose();
        }
        Matrix<Scalar, Dynamic, 1> diagonal() const
        {
            const Index count = std::min(_rows, _cols);
            Matrix<Scalar, Dynamic, 1> result(count);
            for (Index index = 0; index < count; ++index)
                result(index) = coeff(index, index);
            return result;
        }

        template <int OtherOptions, typename OtherStorageIndex>
        SparseMatrix operator+(const SparseMatrix<Scalar, OtherOptions, OtherStorageIndex>& other) const
        {
            return combine(other, Scalar{1});
        }
        template <int OtherOptions, typename OtherStorageIndex>
        SparseMatrix operator-(const SparseMatrix<Scalar, OtherOptions, OtherStorageIndex>& other) const
        {
            return combine(other, Scalar{-1});
        }
        template <int OtherOptions, typename OtherStorageIndex>
        SparseMatrix cwiseProduct(const SparseMatrix<Scalar, OtherOptions, OtherStorageIndex>& other) const
        {
            requireSameShape(other);
            SparseMatrix result(_rows, _cols);
            result.reserve(std::min(nonZeros(), other.nonZeros()));
            for (Index outer = 0; outer < outerSize(); ++outer)
                for (InnerIterator entry(*this, outer); entry; ++entry)
                {
                    const Scalar product = entry.value() * other.coeff(entry.row(), entry.col());
                    if (product != Scalar{})
                        result.coeffRef(entry.row(), entry.col()) = product;
                }
            result.makeCompressed();
            return result;
        }
        template <int OtherOptions, typename OtherStorageIndex>
        SparseMatrix operator*(const SparseMatrix<Scalar, OtherOptions, OtherStorageIndex>& other) const
        {
            if (_cols != other.rows())
                throw std::invalid_argument("sparse matrix product has incompatible dimensions");
            SparseMatrix result(_rows, other.cols());
            std::vector<std::vector<std::pair<Index, Scalar>>> rightRows(static_cast<std::size_t>(other.rows()));
            for (Index outer = 0; outer < other.outerSize(); ++outer)
                for (typename SparseMatrix<Scalar, OtherOptions, OtherStorageIndex>::InnerIterator entry(other, outer);
                     entry;
                     ++entry)
                    rightRows[static_cast<std::size_t>(entry.row())].emplace_back(entry.col(), entry.value());
            for (Index outer = 0; outer < outerSize(); ++outer)
                for (InnerIterator entry(*this, outer); entry; ++entry)
                    for (const auto& [column, value] : rightRows[static_cast<std::size_t>(entry.col())])
                        result.coeffRef(entry.row(), column) += entry.value() * value;
            result.prune(Scalar{});
            result.makeCompressed();
            return result;
        }

    private:
        friend struct internal::SparseAccess;
        internal::CsrStorage<Scalar, internal::Device::CPU> csrSnapshot() const
        {
            validateEscapedCompressedStorage();
            makeCompressed();
            internal::CsrStorage<Scalar, internal::Device::CPU> result(
                _rows, _cols, static_cast<Index>(_values.size()));
            if constexpr (IsRowMajor)
            {
                for (Index row = 0; row <= _rows; ++row)
                    result.rowOffsets()[row] = static_cast<Index>(_outerIndices[static_cast<std::size_t>(row)]);
                for (Index position = 0; position < static_cast<Index>(_values.size()); ++position)
                {
                    result.colIndices()[position] =
                        static_cast<Index>(_innerIndices[static_cast<std::size_t>(position)]);
                    result.values()[position] = _values[static_cast<std::size_t>(position)];
                }
                return result;
            }

            Index* row_offsets = result.rowOffsets();
            for (Index row = 0; row <= _rows; ++row)
                row_offsets[row] = 0;
            for (const StorageIndex row : _innerIndices)
                ++row_offsets[static_cast<Index>(row) + 1];
            for (Index row = 0; row < _rows; ++row)
                row_offsets[row + 1] += row_offsets[row];
            std::vector<Index> next(row_offsets, row_offsets + _rows);
            for (Index col = 0; col < _cols; ++col)
            {
                const Index begin = static_cast<Index>(_outerIndices[static_cast<std::size_t>(col)]);
                const Index end = static_cast<Index>(_outerIndices[static_cast<std::size_t>(col + 1)]);
                for (Index position = begin; position < end; ++position)
                {
                    const Index row = static_cast<Index>(_innerIndices[static_cast<std::size_t>(position)]);
                    const Index target = next[static_cast<std::size_t>(row)]++;
                    result.colIndices()[target] = col;
                    result.values()[target] = _values[static_cast<std::size_t>(position)];
                }
            }
            return result;
        }
        const std::vector<Entry>& entries() const
        {
            validateEscapedCompressedStorage();
            if (!_entrySnapshotReady || _rawPointersEscaped || _mutableReferencesEscaped)
            {
                if (_isCompressed && !_mutableReferencesEscaped)
                    rebuildEntriesFromCompressed(_entrySnapshot);
                else
                    _entrySnapshot = _assemblyEntries;
                std::sort(_entrySnapshot.begin(),
                          _entrySnapshot.end(),
                          [](const Entry& left, const Entry& right) { return left.first < right.first; });
                _entrySnapshotReady = true;
            }
            return _entrySnapshot;
        }

    public:
        template <int RightRows, int RightCols>
        Matrix<Scalar, Dynamic, RightCols> operator*(const Matrix<Scalar, RightRows, RightCols>& right) const
        {
            if (right.rows() != _cols)
                throw std::invalid_argument("SparseMatrix product has incompatible dimensions");
            if (internal::currentExecutionSettings().policy == internal::ExecutionPolicy::GpuRequired)
                throw internal::Error(internal::ErrorCode::UnsupportedOperation,
                                      "SparseMatrix product is not available on the selected GPU");
            Matrix<Scalar, Dynamic, RightCols> result(_rows, right.cols());
            result.setZero();
            makeCompressed();
            for (Index outer = 0; outer < outerSize(); ++outer)
                for (Index position = static_cast<Index>(_outerIndices[static_cast<std::size_t>(outer)]);
                     position < static_cast<Index>(_outerIndices[static_cast<std::size_t>(outer + 1)]);
                     ++position)
                {
                    const Index row =
                        IsRowMajor ? outer : static_cast<Index>(_innerIndices[static_cast<std::size_t>(position)]);
                    const Index inner =
                        IsRowMajor ? static_cast<Index>(_innerIndices[static_cast<std::size_t>(position)]) : outer;
                    for (Index col = 0; col < right.cols(); ++col)
                        result(row, col) += _values[static_cast<std::size_t>(position)] * right(inner, col);
                }
            return result;
        }

    private:
        template <typename MatrixType> void assign(const SparseView<MatrixType>& view)
        {
            SparseMatrix result(view.rows(), view.cols());
            result.reserve(view.nonZeros());
            for (Index outer = 0; outer < view.outerSize(); ++outer)
                for (typename SparseView<MatrixType>::InnerIterator entry(view, outer); entry; ++entry)
                    result.coeffRef(entry.row(), entry.col()) = entry.value();
            result.makeCompressed();
            swap(result);
        }
        template <int OtherOptions, typename OtherStorageIndex>
        void assign(const SparseMatrix<Scalar, OtherOptions, OtherStorageIndex>& other)
        {
            SparseMatrix result(other.rows(), other.cols());
            result.reserve(other.nonZeros());
            for (Index outer = 0; outer < other.outerSize(); ++outer)
                for (typename SparseMatrix<Scalar, OtherOptions, OtherStorageIndex>::InnerIterator entry(other, outer);
                     entry;
                     ++entry)
                    result.coeffRef(entry.row(), entry.col()) = entry.value();
            result.makeCompressed();
            swap(result);
        }
        template <int OtherOptions, typename OtherStorageIndex>
        void requireSameShape(const SparseMatrix<Scalar, OtherOptions, OtherStorageIndex>& other) const
        {
            if (_rows != other.rows() || _cols != other.cols())
                throw std::invalid_argument("sparse matrices have incompatible dimensions");
        }
        template <int OtherOptions, typename OtherStorageIndex>
        SparseMatrix combine(const SparseMatrix<Scalar, OtherOptions, OtherStorageIndex>& other, Scalar scale) const
        {
            requireSameShape(other);
            SparseMatrix result(*this);
            result.reserve(nonZeros() + other.nonZeros());
            for (Index outer = 0; outer < other.outerSize(); ++outer)
                for (typename SparseMatrix<Scalar, OtherOptions, OtherStorageIndex>::InnerIterator entry(other, outer);
                     entry;
                     ++entry)
                    result.coeffRef(entry.row(), entry.col()) += scale * entry.value();
            result.prune(Scalar{});
            result.makeCompressed();
            return result;
        }

        static Index outerIndex(const Entry& entry) noexcept
        {
            return IsRowMajor ? entry.first.first : entry.first.second;
        }
        static Index innerIndex(const Entry& entry) noexcept
        {
            return IsRowMajor ? entry.first.second : entry.first.first;
        }
        static bool storageLess(const Entry& left, const Entry& right) noexcept
        {
            return std::pair{outerIndex(left), innerIndex(left)} < std::pair{outerIndex(right), innerIndex(right)};
        }
        static bool storageLessCoordinate(const Entry& left, const Coordinate& right) noexcept
        {
            const Index rightOuter = IsRowMajor ? right.first : right.second;
            const Index rightInner = IsRowMajor ? right.second : right.first;
            return std::pair{outerIndex(left), innerIndex(left)} < std::pair{rightOuter, rightInner};
        }
        auto findAssembly(Index row, Index col)
        {
            return std::lower_bound(
                _assemblyEntries.begin(), _assemblyEntries.end(), Coordinate{row, col}, storageLessCoordinate);
        }
        auto findAssembly(Index row, Index col) const
        {
            return std::lower_bound(
                _assemblyEntries.begin(), _assemblyEntries.end(), Coordinate{row, col}, storageLessCoordinate);
        }
        bool coordinateExists(Index row, Index col) const
        {
            if (_isCompressed && !_mutableReferencesEscaped)
            {
                const Index outer = IsRowMajor ? row : col;
                const StorageIndex inner =
                    checkedStorageIndex(IsRowMajor ? col : row, "SparseMatrix inner index overflow");
                const auto first =
                    _innerIndices.begin() + static_cast<Index>(_outerIndices[static_cast<std::size_t>(outer)]);
                const auto last =
                    _innerIndices.begin() + static_cast<Index>(_outerIndices[static_cast<std::size_t>(outer + 1)]);
                return std::binary_search(first, last, inner);
            }
            const auto found = findAssembly(row, col);
            return found != _assemblyEntries.end() && found->first == Coordinate{row, col};
        }
        template <typename DupFunctor> void sortAndCombineAssembly(DupFunctor duplicate)
        {
            std::stable_sort(_assemblyEntries.begin(), _assemblyEntries.end(), storageLess);
            std::size_t output = 0;
            for (auto& entry : _assemblyEntries)
            {
                if (output > 0 && _assemblyEntries[output - 1].first == entry.first)
                    _assemblyEntries[output - 1].second = duplicate(_assemblyEntries[output - 1].second, entry.second);
                else
                    _assemblyEntries[output++] = std::move(entry);
            }
            _assemblyEntries.resize(output);
        }
        void rebuildAssemblyFromCompressed()
        {
            rebuildEntriesFromCompressed(_assemblyEntries);
        }
        void rebuildEntriesFromCompressed(std::vector<Entry>& destination) const
        {
            destination.clear();
            destination.reserve(_values.size());
            for (Index outer = 0; outer < outerSize(); ++outer)
                for (Index position = static_cast<Index>(_outerIndices[static_cast<std::size_t>(outer)]);
                     position < static_cast<Index>(_outerIndices[static_cast<std::size_t>(outer + 1)]);
                     ++position)
                {
                    const Index inner = static_cast<Index>(_innerIndices[static_cast<std::size_t>(position)]);
                    const Coordinate coordinate = IsRowMajor ? Coordinate{outer, inner} : Coordinate{inner, outer};
                    destination.push_back({coordinate, _values[static_cast<std::size_t>(position)]});
                }
        }
        void prepareIteration() const
        {
            validateEscapedCompressedStorage();
            if (_iterationReady && !_rawPointersEscaped && !_mutableReferencesEscaped)
                return;
            if (_isCompressed && !_mutableReferencesEscaped)
                rebuildEntriesFromCompressed(_iterationEntries);
            else
                _iterationEntries = _assemblyEntries;
            std::sort(_iterationEntries.begin(), _iterationEntries.end(), storageLess);
            _iterationReady = true;
        }
        static StorageIndex checkedStorageIndex(Index value, const char* what)
        {
            if (value < static_cast<Index>(std::numeric_limits<StorageIndex>::lowest()) ||
                value > static_cast<Index>(std::numeric_limits<StorageIndex>::max()))
                throw std::overflow_error(what);
            return static_cast<StorageIndex>(value);
        }
        void validateEscapedCompressedStorage() const
        {
            if (!_rawPointersEscaped)
                return;
            if (!_isCompressed || _outerIndices.size() != static_cast<std::size_t>(outerSize() + 1) ||
                _innerIndices.size() != _values.size() || _outerIndices.empty() ||
                _outerIndices.front() != StorageIndex{} ||
                _outerIndices.back() != checkedStorageIndex(static_cast<Index>(_values.size()),
                                                            "SparseMatrix nonzero count exceeds StorageIndex range"))
                throw std::invalid_argument("SparseMatrix compressed pointer data is structurally invalid");
            for (Index outer = 0; outer < outerSize(); ++outer)
            {
                const Index begin = static_cast<Index>(_outerIndices[static_cast<std::size_t>(outer)]);
                const Index end = static_cast<Index>(_outerIndices[static_cast<std::size_t>(outer + 1)]);
                if (begin < 0 || begin > end || end > static_cast<Index>(_values.size()))
                    throw std::invalid_argument("SparseMatrix compressed outer indices are invalid");
                StorageIndex previous = StorageIndex{-1};
                for (Index position = begin; position < end; ++position)
                {
                    const StorageIndex inner = _innerIndices[static_cast<std::size_t>(position)];
                    if (inner < 0 || static_cast<Index>(inner) >= innerSize() || inner <= previous)
                        throw std::invalid_argument("SparseMatrix compressed inner indices must be sorted and unique");
                    previous = inner;
                }
            }
        }
        void invalidateSnapshots() const noexcept
        {
            _iterationEntries.clear();
            _iterationReady = false;
            _entrySnapshot.clear();
            _entrySnapshotReady = false;
        }
        void invalidateNumericalCaches() const noexcept
        {
            invalidateSnapshots();
        }
        void resetBackInsertion() const noexcept
        {
            _backInsertionActive = false;
            _backInsertionOuter = 0;
            _backInsertionInner = -1;
        }
        static std::unique_ptr<internal::CsrStorage<Scalar, internal::Device::CPU>>
        materialize(Index rowsCount, Index colsCount, const std::vector<Entry>& entries)
        {
            std::vector<Index> rows;
            std::vector<Index> cols;
            std::vector<Scalar> values;
            rows.reserve(entries.size());
            cols.reserve(entries.size());
            values.reserve(entries.size());
            for (const auto& [coordinate, value] : entries)
            {
                rows.push_back(coordinate.first);
                cols.push_back(coordinate.second);
                values.push_back(value);
            }
            return std::make_unique<internal::CsrStorage<Scalar, internal::Device::CPU>>(
                internal::cooToCsr(rowsCount, colsCount, rows, cols, values));
        }
        void checkCoordinate(Index row, Index col) const
        {
            if (row < 0 || row >= _rows || col < 0 || col >= _cols)
                throw std::out_of_range("SparseMatrix coordinate is out of bounds");
        }

        Index _rows = 0;
        Index _cols = 0;
        mutable std::vector<Entry> _assemblyEntries;
        mutable std::vector<Scalar> _values;
        mutable std::vector<StorageIndex> _innerIndices;
        mutable std::vector<StorageIndex> _outerIndices;
        mutable bool _isCompressed = false;
        mutable bool _mutableReferencesEscaped = false;
        mutable bool _rawPointersEscaped = false;
        mutable bool _backInsertionActive = false;
        mutable Index _backInsertionOuter = 0;
        mutable Index _backInsertionInner = -1;
        mutable std::vector<Entry> _iterationEntries;
        mutable bool _iterationReady = false;
        mutable std::vector<Entry> _entrySnapshot;
        mutable bool _entrySnapshotReady = false;
    };

    namespace sparse_map_detail
    {
        template <typename Derived, typename SparseType, bool IsConst>
        class SparseMapBase : public SparseMatrixBase<Derived>
        {
        public:
            using Scalar = typename SparseType::Scalar;
            using ScalarType = Scalar;
            using RealScalar = typename SparseType::RealScalar;
            using StorageIndex = typename SparseType::StorageIndex;
            using Index = plamatrix::Index;
            using PlainObject = SparseType;
            static constexpr int IsRowMajor = SparseType::IsRowMajor;
            using ScalarPointer = std::conditional_t<IsConst, const Scalar*, Scalar*>;
            using IndexPointer = std::conditional_t<IsConst, const StorageIndex*, StorageIndex*>;

            class InnerIterator
            {
            public:
                InnerIterator(const SparseMapBase& matrix, Index outer) : _matrix(&matrix), _outer(outer)
                {
                    if (outer < 0 || outer >= matrix.outerSize())
                    {
                        throw std::out_of_range("sparse Map outer index is out of bounds");
                    }
                    _position = static_cast<Index>(matrix._outerIndices[outer]);
                    _end = static_cast<Index>(matrix._outerIndices[outer + 1]);
                }

                explicit operator bool() const noexcept
                {
                    return _position < _end;
                }
                InnerIterator& operator++()
                {
                    ++_position;
                    return *this;
                }
                Index row() const noexcept
                {
                    return IsRowMajor ? _outer : index();
                }
                Index col() const noexcept
                {
                    return IsRowMajor ? index() : _outer;
                }
                Index index() const noexcept
                {
                    return static_cast<Index>(_matrix->_innerIndices[_position]);
                }
                Index outer() const noexcept
                {
                    return _outer;
                }
                const Scalar& value() const noexcept
                {
                    return _matrix->_values[_position];
                }

                template <bool Enabled = !IsConst, std::enable_if_t<Enabled, int> = 0> Scalar& valueRef() const noexcept
                {
                    return _matrix->_values[_position];
                }

            private:
                const SparseMapBase* _matrix;
                Index _outer;
                Index _position;
                Index _end;
            };

            SparseMapBase(Index rows,
                          Index cols,
                          Index non_zeros,
                          IndexPointer outer_indices,
                          IndexPointer inner_indices,
                          ScalarPointer values,
                          IndexPointer = nullptr)
                : _rows(rows), _cols(cols), _nonZeros(non_zeros), _outerIndices(outer_indices),
                  _innerIndices(inner_indices), _values(values)
            {
                if (rows < 0 || cols < 0 || non_zeros < 0 || outer_indices == nullptr ||
                    (non_zeros != 0 && (inner_indices == nullptr || values == nullptr)))
                {
                    throw std::invalid_argument("sparse Map received invalid dimensions or pointers");
                }
                if (_outerIndices[0] != StorageIndex{} || static_cast<Index>(_outerIndices[outerSize()]) != non_zeros)
                {
                    throw std::invalid_argument("sparse Map outer indices do not delimit the nonzeros");
                }
                for (Index outer = 0; outer < outerSize(); ++outer)
                {
                    const Index begin = static_cast<Index>(_outerIndices[outer]);
                    const Index end = static_cast<Index>(_outerIndices[outer + 1]);
                    if (begin < 0 || end < begin || end > non_zeros)
                    {
                        throw std::invalid_argument("sparse Map outer indices must be monotonic and in range");
                    }
                    StorageIndex previous{};
                    bool has_previous = false;
                    for (Index position = begin; position < end; ++position)
                    {
                        const StorageIndex inner = _innerIndices[position];
                        const bool negative = [](StorageIndex value)
                        {
                            if constexpr (std::is_signed_v<StorageIndex>)
                                return value < StorageIndex{};
                            return false;
                        }(inner);
                        if (negative || static_cast<Index>(inner) >= innerSize() || (has_previous && inner <= previous))
                        {
                            throw std::invalid_argument(
                                "sparse Map inner indices must be strictly increasing and in range");
                        }
                        previous = inner;
                        has_previous = true;
                    }
                }
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
            Index outerSize() const noexcept
            {
                return IsRowMajor ? _rows : _cols;
            }
            Index innerSize() const noexcept
            {
                return IsRowMajor ? _cols : _rows;
            }
            Index nonZeros() const noexcept
            {
                return _nonZeros;
            }
            bool isCompressed() const noexcept
            {
                return true;
            }

            Scalar coeff(Index row, Index col) const
            {
                checkCoordinate(row, col);
                const Index outer = IsRowMajor ? row : col;
                const StorageIndex inner = static_cast<StorageIndex>(IsRowMajor ? col : row);
                const StorageIndex* begin = _innerIndices + _outerIndices[outer];
                const StorageIndex* end = _innerIndices + _outerIndices[outer + 1];
                const StorageIndex* position = std::lower_bound(begin, end, inner);
                return position != end && *position == inner ? _values[static_cast<Index>(position - _innerIndices)]
                                                             : Scalar{};
            }
            Scalar operator()(Index row, Index col) const
            {
                return coeff(row, col);
            }

            template <bool Enabled = !IsConst, std::enable_if_t<Enabled, int> = 0>
            Scalar& coeffRef(Index row, Index col)
            {
                checkCoordinate(row, col);
                const Index outer = IsRowMajor ? row : col;
                const StorageIndex inner = static_cast<StorageIndex>(IsRowMajor ? col : row);
                StorageIndex* begin = _innerIndices + _outerIndices[outer];
                StorageIndex* end = _innerIndices + _outerIndices[outer + 1];
                StorageIndex* position = std::lower_bound(begin, end, inner);
                if (position == end || *position != inner)
                {
                    throw std::out_of_range("sparse Map cannot insert a new coefficient");
                }
                return _values[static_cast<Index>(position - _innerIndices)];
            }

            const Scalar* valuePtr() const noexcept
            {
                return _values;
            }
            const StorageIndex* innerIndexPtr() const noexcept
            {
                return _innerIndices;
            }
            const StorageIndex* outerIndexPtr() const noexcept
            {
                return _outerIndices;
            }

            template <bool Enabled = !IsConst, std::enable_if_t<Enabled, int> = 0> Scalar* valuePtr() noexcept
            {
                return _values;
            }
            template <bool Enabled = !IsConst, std::enable_if_t<Enabled, int> = 0>
            StorageIndex* innerIndexPtr() noexcept
            {
                return _innerIndices;
            }
            template <bool Enabled = !IsConst, std::enable_if_t<Enabled, int> = 0>
            StorageIndex* outerIndexPtr() noexcept
            {
                return _outerIndices;
            }

        private:
            void checkCoordinate(Index row, Index col) const
            {
                if (row < 0 || row >= _rows || col < 0 || col >= _cols)
                {
                    throw std::out_of_range("sparse Map coordinate is out of bounds");
                }
            }

            Index _rows;
            Index _cols;
            Index _nonZeros;
            IndexPointer _outerIndices;
            IndexPointer _innerIndices;
            ScalarPointer _values;
        };
    } // namespace sparse_map_detail

    template <typename Scalar, int Options, typename StorageIndex, int MapOptions, typename StrideType>
    class Map<SparseMatrix<Scalar, Options, StorageIndex>, MapOptions, StrideType>
        : public sparse_map_detail::SparseMapBase<
              Map<SparseMatrix<Scalar, Options, StorageIndex>, MapOptions, StrideType>,
              SparseMatrix<Scalar, Options, StorageIndex>,
              false>
    {
        using Base =
            sparse_map_detail::SparseMapBase<Map<SparseMatrix<Scalar, Options, StorageIndex>, MapOptions, StrideType>,
                                             SparseMatrix<Scalar, Options, StorageIndex>,
                                             false>;

    public:
        using Base::Base;
    };

    template <typename Scalar, int Options, typename StorageIndex, int MapOptions, typename StrideType>
    class Map<const SparseMatrix<Scalar, Options, StorageIndex>, MapOptions, StrideType>
        : public sparse_map_detail::SparseMapBase<
              Map<const SparseMatrix<Scalar, Options, StorageIndex>, MapOptions, StrideType>,
              SparseMatrix<Scalar, Options, StorageIndex>,
              true>
    {
        using Base = sparse_map_detail::SparseMapBase<
            Map<const SparseMatrix<Scalar, Options, StorageIndex>, MapOptions, StrideType>,
            SparseMatrix<Scalar, Options, StorageIndex>,
            true>;

    public:
        using Base::Base;
    };

    template <typename Derived>
    SparseView<const Derived> MatrixBase<Derived>::sparseView(const Scalar& reference, const RealScalar& epsilon) const
    {
        return SparseView<const Derived>(this->derived(), reference, epsilon);
    }

    template <int Size, int MaxSize, typename PermutationIndex, typename Scalar, int Options, typename StorageIndex>
    SparseMatrix<Scalar, Options, StorageIndex>
    operator*(const PermutationMatrix<Size, MaxSize, PermutationIndex>& permutation,
              const SparseMatrix<Scalar, Options, StorageIndex>& matrix)
    {
        permutation.validate();
        if (permutation.size() != matrix.rows())
        {
            throw std::invalid_argument("permutation and sparse matrix row counts differ");
        }
        SparseMatrix<Scalar, Options, StorageIndex> result(matrix.rows(), matrix.cols());
        std::vector<Index> inverse(static_cast<std::size_t>(matrix.rows()));
        for (Index row = 0; row < matrix.rows(); ++row)
        {
            inverse[static_cast<std::size_t>(permutation.indices()(row))] = row;
        }
        std::vector<Triplet<Scalar, StorageIndex>> triplets;
        triplets.reserve(static_cast<std::size_t>(matrix.nonZeros()));
        for (Index outer = 0; outer < matrix.outerSize(); ++outer)
        {
            for (typename SparseMatrix<Scalar, Options, StorageIndex>::InnerIterator entry(matrix, outer); entry;
                 ++entry)
            {
                triplets.emplace_back(static_cast<StorageIndex>(inverse[static_cast<std::size_t>(entry.row())]),
                                      static_cast<StorageIndex>(entry.col()),
                                      entry.value());
            }
        }
        result.setFromTriplets(triplets.begin(), triplets.end());
        return result;
    }

    template <typename Scalar, int Options, typename StorageIndex, int Size, int MaxSize, typename PermutationIndex>
    SparseMatrix<Scalar, Options, StorageIndex>
    operator*(const SparseMatrix<Scalar, Options, StorageIndex>& matrix,
              const PermutationMatrix<Size, MaxSize, PermutationIndex>& permutation)
    {
        permutation.validate();
        if (permutation.size() != matrix.cols())
        {
            throw std::invalid_argument("sparse matrix and permutation column counts differ");
        }
        SparseMatrix<Scalar, Options, StorageIndex> result(matrix.rows(), matrix.cols());
        std::vector<Index> inverse(static_cast<std::size_t>(matrix.cols()));
        for (Index column = 0; column < matrix.cols(); ++column)
        {
            inverse[static_cast<std::size_t>(permutation.indices()(column))] = column;
        }
        std::vector<Triplet<Scalar, StorageIndex>> triplets;
        triplets.reserve(static_cast<std::size_t>(matrix.nonZeros()));
        for (Index outer = 0; outer < matrix.outerSize(); ++outer)
        {
            for (typename SparseMatrix<Scalar, Options, StorageIndex>::InnerIterator entry(matrix, outer); entry;
                 ++entry)
            {
                triplets.emplace_back(static_cast<StorageIndex>(entry.row()),
                                      static_cast<StorageIndex>(inverse[static_cast<std::size_t>(entry.col())]),
                                      entry.value());
            }
        }
        result.setFromTriplets(triplets.begin(), triplets.end());
        return result;
    }
} // namespace plamatrix::v1

namespace plamatrix::internal
{
    struct SparseAccess
    {
        template <typename MatrixType> static auto csrSnapshot(const MatrixType& value)
        {
            return value.csrSnapshot();
        }
        template <typename MatrixType> static decltype(auto) entries(const MatrixType& value)
        {
            return value.entries();
        }
    };
} // namespace plamatrix::internal
