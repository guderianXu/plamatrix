#pragma once

#include <functional>
#include <limits>
#include <type_traits>

#include "plamatrix/dense/matrix_fwd.h"

namespace plamatrix::detail
{
    template <typename Type> struct DenseTraits;

    template <typename Scalar> constexpr Scalar dummyPrecision() noexcept
    {
        if constexpr (std::is_same_v<Scalar, float>)
        {
            return 1.0e-5f;
        }
        else if constexpr (std::is_same_v<Scalar, double>)
        {
            return 1.0e-12;
        }
        else if constexpr (std::is_same_v<Scalar, long double>)
        {
            return 1.0e-15L;
        }
        else
        {
            return Scalar{};
        }
    }

    template <typename Type> struct DenseTraits<const Type> : DenseTraits<Type>
    {
    };

    template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    struct DenseTraits<Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>>
    {
        using ScalarType = Scalar;
        using PlainObject = Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>;
        static constexpr int RowsAtCompileTime = Rows;
        static constexpr int ColsAtCompileTime = Cols;
        static constexpr int MaxRowsAtCompileTime = MaxRows;
        static constexpr int MaxColsAtCompileTime = MaxCols;
        static constexpr int StorageOptions = Options;
    };

    template <typename MatrixType, int MapOptions, typename StrideType>
    struct DenseTraits<Map<MatrixType, MapOptions, StrideType>> : DenseTraits<std::remove_const_t<MatrixType>>
    {
    };

    template <typename PlainObjectType, int Options, typename StrideType>
    struct DenseTraits<Ref<PlainObjectType, Options, StrideType>> : DenseTraits<std::remove_const_t<PlainObjectType>>
    {
    };

    template <typename XprType, int BlockRows, int BlockCols, bool InnerPanel>
    struct DenseTraits<Block<XprType, BlockRows, BlockCols, InnerPanel>>
    {
        using NestedTraits = DenseTraits<std::remove_const_t<XprType>>;
        using ScalarType = typename NestedTraits::ScalarType;
        static constexpr int RowsAtCompileTime = BlockRows;
        static constexpr int ColsAtCompileTime = BlockCols;
        static constexpr int MaxRowsAtCompileTime =
            BlockRows == Dynamic ? NestedTraits::MaxRowsAtCompileTime : BlockRows;
        static constexpr int MaxColsAtCompileTime =
            BlockCols == Dynamic ? NestedTraits::MaxColsAtCompileTime : BlockCols;
        static constexpr bool NestedIsRowMajor = (NestedTraits::StorageOptions & RowMajor) == RowMajor;
        static constexpr bool IsRowMajor = MaxRowsAtCompileTime == 1 && MaxColsAtCompileTime != 1   ? true
                                           : MaxColsAtCompileTime == 1 && MaxRowsAtCompileTime != 1 ? false
                                                                                                    : NestedIsRowMajor;
        static constexpr int StorageOptions = IsRowMajor ? RowMajor : ColMajor;
        using PlainObject =
            Matrix<ScalarType, BlockRows, BlockCols, StorageOptions, MaxRowsAtCompileTime, MaxColsAtCompileTime>;
    };

    template <typename MatrixType> struct DenseTraits<Transpose<MatrixType>>
    {
        using NestedTraits = DenseTraits<std::remove_const_t<MatrixType>>;
        using ScalarType = typename NestedTraits::ScalarType;
        static constexpr int StorageOptions = (NestedTraits::StorageOptions & RowMajor) == RowMajor
                                                  ? (NestedTraits::StorageOptions & ~RowMajor)
                                                  : (NestedTraits::StorageOptions | RowMajor);
        using PlainObject = Matrix<ScalarType,
                                   NestedTraits::ColsAtCompileTime,
                                   NestedTraits::RowsAtCompileTime,
                                   StorageOptions,
                                   NestedTraits::MaxColsAtCompileTime,
                                   NestedTraits::MaxRowsAtCompileTime>;
        static constexpr int RowsAtCompileTime = NestedTraits::ColsAtCompileTime;
        static constexpr int ColsAtCompileTime = NestedTraits::RowsAtCompileTime;
        static constexpr int MaxRowsAtCompileTime = NestedTraits::MaxColsAtCompileTime;
        static constexpr int MaxColsAtCompileTime = NestedTraits::MaxRowsAtCompileTime;
    };

    template <typename VectorType, int Size>
    struct DenseTraits<VectorBlock<VectorType, Size>>
        : DenseTraits<
              Block<VectorType,
                    (DenseTraits<std::remove_const_t<VectorType>>::StorageOptions & RowMajor) == RowMajor ? 1 : Size,
                    (DenseTraits<std::remove_const_t<VectorType>>::StorageOptions & RowMajor) == RowMajor ? Size : 1>>
    {
    };

    template <typename MatrixType, int DiagIndex> struct DenseTraits<Diagonal<MatrixType, DiagIndex>>;
    template <typename DiagonalVectorType> struct DenseTraits<DiagonalWrapper<DiagonalVectorType>>;
    template <typename MatrixType, unsigned int Mode> struct DenseTraits<TriangularView<MatrixType, Mode>>;
    template <typename MatrixType, unsigned int UpLo> struct DenseTraits<SelfAdjointView<MatrixType, UpLo>>;
    template <typename XprType, int Rows, int Cols, int Order> struct DenseTraits<Reshaped<XprType, Rows, Cols, Order>>;
    template <typename MatrixType, int RowFactor, int ColFactor>
    struct DenseTraits<Replicate<MatrixType, RowFactor, ColFactor>>;
    template <typename MatrixType, int Direction> struct DenseTraits<Reverse<MatrixType, Direction>>;

    template <typename UnaryOp, typename XprType> struct DenseTraits<CwiseUnaryOp<UnaryOp, XprType>>
    {
        using NestedTraits = DenseTraits<std::remove_const_t<XprType>>;
        using ScalarType = std::decay_t<std::invoke_result_t<UnaryOp, const typename NestedTraits::ScalarType&>>;
        static constexpr int RowsAtCompileTime = NestedTraits::RowsAtCompileTime;
        static constexpr int ColsAtCompileTime = NestedTraits::ColsAtCompileTime;
        static constexpr int MaxRowsAtCompileTime = NestedTraits::MaxRowsAtCompileTime;
        static constexpr int MaxColsAtCompileTime = NestedTraits::MaxColsAtCompileTime;
        static constexpr int StorageOptions = NestedTraits::StorageOptions;
        using PlainObject = Matrix<ScalarType,
                                   RowsAtCompileTime,
                                   ColsAtCompileTime,
                                   StorageOptions,
                                   MaxRowsAtCompileTime,
                                   MaxColsAtCompileTime>;
    };

    template <typename BinaryOp, typename LhsType, typename RhsType>
    struct DenseTraits<CwiseBinaryOp<BinaryOp, LhsType, RhsType>>
    {
        using LeftTraits = DenseTraits<std::remove_const_t<LhsType>>;
        using RightTraits = DenseTraits<std::remove_const_t<RhsType>>;
        using ScalarType = std::decay_t<std::invoke_result_t<BinaryOp,
                                                             const typename LeftTraits::ScalarType&,
                                                             const typename RightTraits::ScalarType&>>;
        static constexpr int RowsAtCompileTime = LeftTraits::RowsAtCompileTime;
        static constexpr int ColsAtCompileTime = LeftTraits::ColsAtCompileTime;
        static constexpr int MaxRowsAtCompileTime = LeftTraits::MaxRowsAtCompileTime;
        static constexpr int MaxColsAtCompileTime = LeftTraits::MaxColsAtCompileTime;
        static constexpr int StorageOptions = LeftTraits::StorageOptions;
        using PlainObject = Matrix<ScalarType,
                                   RowsAtCompileTime,
                                   ColsAtCompileTime,
                                   StorageOptions,
                                   MaxRowsAtCompileTime,
                                   MaxColsAtCompileTime>;
    };

    template <typename Node> struct DenseTraits<DenseExpression<Node>>
    {
        using ScalarType = typename Node::Value;
        using PlainObject = Matrix<ScalarType, Node::RowsAtCompileTime, Node::ColsAtCompileTime>;
        static constexpr int RowsAtCompileTime = Node::RowsAtCompileTime;
        static constexpr int ColsAtCompileTime = Node::ColsAtCompileTime;
        static constexpr int MaxRowsAtCompileTime = RowsAtCompileTime;
        static constexpr int MaxColsAtCompileTime = ColsAtCompileTime;
        static constexpr int StorageOptions = PlainObject::Options;
    };

    template <typename Storage> struct UnwrapDenseStorage
    {
        using Unwrapped = std::decay_t<Storage>;
    };

    template <typename Type> struct UnwrapDenseStorage<std::reference_wrapper<const Type>>
    {
        using Unwrapped = std::remove_const_t<Type>;
    };

    template <typename Storage>
    struct DenseTraits<ArrayExpressionProxy<Storage>>
        : DenseTraits<typename UnwrapDenseStorage<std::decay_t<Storage>>::Unwrapped>
    {
    };
} // namespace plamatrix::detail

namespace plamatrix::v1
{
    /// Common CRTP root shared by dense PlaMatrix expressions, matching Eigen's public hierarchy.
    template <typename Derived> struct EigenBase
    {
        Derived& derived() noexcept
        {
            return *static_cast<Derived*>(this);
        }

        const Derived& derived() const noexcept
        {
            return *static_cast<const Derived*>(this);
        }

        Index rows() const noexcept(noexcept(derived().rows()))
        {
            return derived().rows();
        }

        Index cols() const noexcept(noexcept(derived().cols()))
        {
            return derived().cols();
        }

        Index size() const noexcept(noexcept(derived().size()))
        {
            return derived().size();
        }
    };

    /// Base class for dense expressions accepted by generic Eigen-style functions.
    template <typename Derived> class DenseBase : public EigenBase<Derived>
    {
        using Traits = detail::DenseTraits<Derived>;

    public:
        using Scalar = typename Traits::ScalarType;
        using RealScalar = typename NumTraits<Scalar>::Real;
        using Index = plamatrix::Index;
        using PlainObject = typename Traits::PlainObject;

        static constexpr int RowsAtCompileTime = Traits::RowsAtCompileTime;
        static constexpr int ColsAtCompileTime = Traits::ColsAtCompileTime;
        static constexpr int MaxRowsAtCompileTime = Traits::MaxRowsAtCompileTime;
        static constexpr int MaxColsAtCompileTime = Traits::MaxColsAtCompileTime;
        static constexpr bool IsVectorAtCompileTime = RowsAtCompileTime == 1 || ColsAtCompileTime == 1;
        static constexpr int SizeAtCompileTime = RowsAtCompileTime == Dynamic || ColsAtCompileTime == Dynamic
                                                     ? Dynamic
                                                     : RowsAtCompileTime * ColsAtCompileTime;
        static constexpr bool IsRowMajor = (Traits::StorageOptions & RowMajor) == RowMajor;

        using BlockXpr = Block<Derived>;
        using ConstBlockXpr = const Block<const Derived>;
        using RowXpr = Block<Derived, 1, ColsAtCompileTime, IsRowMajor>;
        using ConstRowXpr = const Block<const Derived, 1, ColsAtCompileTime, IsRowMajor>;
        using ColXpr = Block<Derived, RowsAtCompileTime, 1, !IsRowMajor>;
        using ConstColXpr = const Block<const Derived, RowsAtCompileTime, 1, !IsRowMajor>;
        using TransposeReturnType = Transpose<Derived>;
        using ConstTransposeReturnType = Transpose<const Derived>;
        using SegmentReturnType = VectorBlock<Derived>;
        using ConstSegmentReturnType = const VectorBlock<const Derived>;
        using ReverseReturnType = Reverse<Derived, BothDirections>;
        using ConstReverseReturnType = const Reverse<const Derived, BothDirections>;

        template <int Rows, int Cols> struct FixedBlockXpr
        {
            using Type = Block<Derived, Rows, Cols>;
        };

        template <int Rows, int Cols> struct ConstFixedBlockXpr
        {
            using Type = const Block<const Derived, Rows, Cols>;
        };

        using EigenBase<Derived>::derived;

        decltype(auto) coeff(Index row, Index col) const
        {
            return this->derived().coeff(row, col);
        }

        decltype(auto) operator()(Index row, Index col) const
        {
            return this->derived()(row, col);
        }

        decltype(auto) operator()(Index index) const
        {
            return this->derived()(index);
        }

        const Derived& eval() const noexcept
        {
            return this->derived();
        }

        Scalar sum() const;
        Scalar mean() const;
        Scalar prod() const;
        Scalar minCoeff() const;
        Scalar maxCoeff() const;

        template <typename IndexType> Scalar minCoeff(IndexType* row, IndexType* col) const;
        template <typename IndexType> Scalar maxCoeff(IndexType* row, IndexType* col) const;
        template <typename IndexType> Scalar minCoeff(IndexType* index) const;
        template <typename IndexType> Scalar maxCoeff(IndexType* index) const;

        bool all() const;
        bool any() const;
        Index count() const;
        bool allFinite() const;

        template <typename OtherDerived>
        bool isApprox(const DenseBase<OtherDerived>& other,
                      const RealScalar& precision = detail::dummyPrecision<RealScalar>()) const;
        bool isMuchSmallerThan(const RealScalar& other,
                               const RealScalar& precision = detail::dummyPrecision<RealScalar>()) const;
        template <typename OtherDerived>
        bool isMuchSmallerThan(const DenseBase<OtherDerived>& other,
                               const RealScalar& precision = detail::dummyPrecision<RealScalar>()) const;

        template <typename CustomUnaryOp>
        const CwiseUnaryOp<CustomUnaryOp, const Derived>
        unaryExpr(const CustomUnaryOp& function = CustomUnaryOp()) const;

        template <typename CustomBinaryOp, typename OtherDerived>
        const CwiseBinaryOp<CustomBinaryOp, const Derived, const OtherDerived>
        binaryExpr(const DenseBase<OtherDerived>& other, const CustomBinaryOp& function = CustomBinaryOp()) const;

        BlockXpr block(Index row, Index col, Index rows, Index cols)
        {
            return BlockXpr(this->derived(), row, col, rows, cols);
        }

        ConstBlockXpr block(Index row, Index col, Index rows, Index cols) const
        {
            return ConstBlockXpr(this->derived(), row, col, rows, cols);
        }

        template <int Rows, int Cols> typename FixedBlockXpr<Rows, Cols>::Type block(Index row, Index col)
        {
            return typename FixedBlockXpr<Rows, Cols>::Type(this->derived(), row, col);
        }

        template <int Rows, int Cols> typename ConstFixedBlockXpr<Rows, Cols>::Type block(Index row, Index col) const
        {
            return typename ConstFixedBlockXpr<Rows, Cols>::Type(this->derived(), row, col);
        }

        RowXpr row(Index index)
        {
            return RowXpr(this->derived(), index);
        }

        ConstRowXpr row(Index index) const
        {
            return ConstRowXpr(this->derived(), index);
        }

        ColXpr col(Index index)
        {
            return ColXpr(this->derived(), index);
        }

        ConstColXpr col(Index index) const
        {
            return ConstColXpr(this->derived(), index);
        }

        TransposeReturnType transpose()
        {
            return TransposeReturnType(this->derived());
        }

        const ConstTransposeReturnType transpose() const
        {
            return ConstTransposeReturnType(this->derived());
        }

        SegmentReturnType segment(Index start, Index length);
        ConstSegmentReturnType segment(Index start, Index length) const;

        template <int Length> VectorBlock<Derived, Length> segment(Index start);
        template <int Length> const VectorBlock<const Derived, Length> segment(Index start) const;

        SegmentReturnType head(Index length);
        ConstSegmentReturnType head(Index length) const;
        SegmentReturnType tail(Index length);
        ConstSegmentReturnType tail(Index length) const;

        template <int Length> VectorBlock<Derived, Length> head();
        template <int Length> const VectorBlock<const Derived, Length> head() const;
        template <int Length> VectorBlock<Derived, Length> tail();
        template <int Length> const VectorBlock<const Derived, Length> tail() const;

        template <int RowFactor, int ColFactor> const Replicate<Derived, RowFactor, ColFactor> replicate() const;
        const Replicate<Derived, Dynamic, Dynamic> replicate(Index row_factor, Index col_factor) const;

        ReverseReturnType reverse();
        ConstReverseReturnType reverse() const;
        void reverseInPlace();

        template <int Order = ColMajor> Reshaped<Derived, Dynamic, Dynamic, Order> reshaped(Index rows, Index cols);
        template <int Order = ColMajor>
        const Reshaped<const Derived, Dynamic, Dynamic, Order> reshaped(Index rows, Index cols) const;
        template <int Order = ColMajor> Reshaped<Derived, SizeAtCompileTime, 1, Order> reshaped();
        template <int Order = ColMajor> const Reshaped<const Derived, SizeAtCompileTime, 1, Order> reshaped() const;

        void transposeInPlace();
    };

    template <typename Derived> class MatrixBase : public DenseBase<Derived>
    {
    public:
        using Base = DenseBase<Derived>;
        using Base::derived;
        using Base::maxCoeff;
        using Base::minCoeff;
        using typename Base::Index;
        using typename Base::PlainObject;
        using typename Base::RealScalar;
        using typename Base::Scalar;

        using DiagonalReturnType = Diagonal<Derived>;
        using ConstDiagonalReturnType = const Diagonal<const Derived>;

        DiagonalReturnType diagonal();
        ConstDiagonalReturnType diagonal() const;
        template <int DiagIndex> Diagonal<Derived, DiagIndex> diagonal();
        template <int DiagIndex> const Diagonal<const Derived, DiagIndex> diagonal() const;
        Diagonal<Derived, DynamicIndex> diagonal(Index index);
        const Diagonal<const Derived, DynamicIndex> diagonal(Index index) const;

        const DiagonalWrapper<const Derived> asDiagonal() const;

        template <unsigned int Mode> TriangularView<Derived, Mode> triangularView();
        template <unsigned int Mode> const TriangularView<const Derived, Mode> triangularView() const;

        template <unsigned int UpLo> SelfAdjointView<Derived, UpLo> selfadjointView();
        template <unsigned int UpLo> const SelfAdjointView<const Derived, UpLo> selfadjointView() const;

        RealScalar squaredNorm() const;
        RealScalar norm() const;

        template <int p> RealScalar lpNorm() const;
        RealScalar stableNorm() const;
        RealScalar blueNorm() const;
        RealScalar hypotNorm() const;

        SparseView<const Derived>
        sparseView(const Scalar& reference = Scalar{},
                   const RealScalar& epsilon = std::numeric_limits<RealScalar>::epsilon()) const;

        Scalar trace() const;

        Matrix<Scalar, 3, 1> eulerAngles(Index a0, Index a1, Index a2) const;
        Matrix<Scalar, 3, 1> canonicalEulerAngles(Index a0, Index a1, Index a2) const;

        template <typename OtherDerived>
        typename ScalarBinaryOpTraits<Scalar, typename detail::DenseTraits<OtherDerived>::ScalarType>::ReturnType
        dot(const MatrixBase<OtherDerived>& other) const;

        using CwiseAbsReturnType = CwiseUnaryOp<internal::scalar_abs_op<Scalar>, const Derived>;
        const CwiseAbsReturnType cwiseAbs() const;
        using ConjugateReturnType = CwiseUnaryOp<internal::scalar_conjugate_op<Scalar>, const Derived>;
        const ConjugateReturnType conjugate() const;
        auto adjoint() const;

        template <int NaNPropagation = PropagateFast, typename OtherDerived>
        const CwiseBinaryOp<
            internal::scalar_min_op<Scalar, typename detail::DenseTraits<OtherDerived>::ScalarType, NaNPropagation>,
            const Derived,
            const OtherDerived>
        cwiseMin(const MatrixBase<OtherDerived>& other) const;

        template <int NaNPropagation = PropagateFast>
        const CwiseUnaryOp<internal::scalar_min_constant_op<Scalar, NaNPropagation>, const Derived>
        cwiseMin(const Scalar& other) const;

        template <int NaNPropagation = PropagateFast, typename OtherDerived>
        const CwiseBinaryOp<
            internal::scalar_max_op<Scalar, typename detail::DenseTraits<OtherDerived>::ScalarType, NaNPropagation>,
            const Derived,
            const OtherDerived>
        cwiseMax(const MatrixBase<OtherDerived>& other) const;

        template <int NaNPropagation = PropagateFast>
        const CwiseUnaryOp<internal::scalar_max_constant_op<Scalar, NaNPropagation>, const Derived>
        cwiseMax(const Scalar& other) const;

        auto array() const
        {
            return this->derived().array();
        }

        auto rowwise() const
        {
            return this->derived().rowwise();
        }

        auto colwise() const
        {
            return this->derived().colwise();
        }
    };

    template <typename Derived> class ArrayBase : public DenseBase<Derived>
    {
    public:
        using Base = DenseBase<Derived>;
        using Base::derived;
        using typename Base::Index;
        using typename Base::PlainObject;
        using typename Base::RealScalar;
        using typename Base::Scalar;
    };
} // namespace plamatrix::v1
