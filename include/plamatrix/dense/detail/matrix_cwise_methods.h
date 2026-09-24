#pragma once

namespace plamatrix::v1
{
    template <typename Derived>
    template <typename CustomUnaryOp>
    const CwiseUnaryOp<CustomUnaryOp, const Derived> DenseBase<Derived>::unaryExpr(const CustomUnaryOp& function) const
    {
        return CwiseUnaryOp<CustomUnaryOp, const Derived>(this->derived(), function);
    }

    template <typename Derived>
    template <typename CustomBinaryOp, typename OtherDerived>
    const CwiseBinaryOp<CustomBinaryOp, const Derived, const OtherDerived>
    DenseBase<Derived>::binaryExpr(const DenseBase<OtherDerived>& other, const CustomBinaryOp& function) const
    {
        return CwiseBinaryOp<CustomBinaryOp, const Derived, const OtherDerived>(
            this->derived(), other.derived(), function);
    }

    template <typename Derived>
    const typename MatrixBase<Derived>::CwiseAbsReturnType MatrixBase<Derived>::cwiseAbs() const
    {
        return CwiseAbsReturnType(this->derived());
    }

    template <typename Derived>
    const typename MatrixBase<Derived>::ConjugateReturnType MatrixBase<Derived>::conjugate() const
    {
        return ConjugateReturnType(this->derived());
    }

    template <typename Derived> auto MatrixBase<Derived>::adjoint() const
    {
        using Traits = detail::DenseTraits<Derived>;
        using Result = Matrix<Scalar,
                              Traits::ColsAtCompileTime,
                              Traits::RowsAtCompileTime,
                              (Traits::StorageOptions & RowMajor) == RowMajor ? ColMajor : RowMajor,
                              Traits::MaxColsAtCompileTime,
                              Traits::MaxRowsAtCompileTime>;
        Result result(this->cols(), this->rows());
        for (Index col = 0; col < this->cols(); ++col)
        {
            for (Index row = 0; row < this->rows(); ++row)
            {
                result(col, row) = internal::scalar_conjugate_op<Scalar>{}(this->derived()(row, col));
            }
        }
        return result;
    }

    template <typename Derived>
    template <int NaNPropagation, typename OtherDerived>
    const CwiseBinaryOp<internal::scalar_min_op<typename MatrixBase<Derived>::Scalar,
                                                typename detail::DenseTraits<OtherDerived>::ScalarType,
                                                NaNPropagation>,
                        const Derived,
                        const OtherDerived>
    MatrixBase<Derived>::cwiseMin(const MatrixBase<OtherDerived>& other) const
    {
        using Operation =
            internal::scalar_min_op<Scalar, typename detail::DenseTraits<OtherDerived>::ScalarType, NaNPropagation>;
        return CwiseBinaryOp<Operation, const Derived, const OtherDerived>(this->derived(), other.derived());
    }

    template <typename Derived>
    template <int NaNPropagation>
    const CwiseUnaryOp<internal::scalar_min_constant_op<typename MatrixBase<Derived>::Scalar, NaNPropagation>,
                       const Derived>
    MatrixBase<Derived>::cwiseMin(const Scalar& other) const
    {
        using Operation = internal::scalar_min_constant_op<Scalar, NaNPropagation>;
        return CwiseUnaryOp<Operation, const Derived>(this->derived(), Operation(other));
    }

    template <typename Derived>
    template <int NaNPropagation, typename OtherDerived>
    const CwiseBinaryOp<internal::scalar_max_op<typename MatrixBase<Derived>::Scalar,
                                                typename detail::DenseTraits<OtherDerived>::ScalarType,
                                                NaNPropagation>,
                        const Derived,
                        const OtherDerived>
    MatrixBase<Derived>::cwiseMax(const MatrixBase<OtherDerived>& other) const
    {
        using Operation =
            internal::scalar_max_op<Scalar, typename detail::DenseTraits<OtherDerived>::ScalarType, NaNPropagation>;
        return CwiseBinaryOp<Operation, const Derived, const OtherDerived>(this->derived(), other.derived());
    }

    template <typename Derived>
    template <int NaNPropagation>
    const CwiseUnaryOp<internal::scalar_max_constant_op<typename MatrixBase<Derived>::Scalar, NaNPropagation>,
                       const Derived>
    MatrixBase<Derived>::cwiseMax(const Scalar& other) const
    {
        using Operation = internal::scalar_max_constant_op<Scalar, NaNPropagation>;
        return CwiseUnaryOp<Operation, const Derived>(this->derived(), Operation(other));
    }
} // namespace plamatrix::v1
