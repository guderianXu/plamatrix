#pragma once

namespace plamatrix::v1
{
    template <typename MatrixType, int MapOptions, typename StrideType>
    auto Map<MatrixType, MapOptions, StrideType>::array() const
    {
        return arrayExpression(Map(*this));
    }
    template <typename MatrixType, int MapOptions, typename StrideType>
    auto Map<MatrixType, MapOptions, StrideType>::rowwise() const
    {
        return axiswiseExpression<true>(Map(*this));
    }
    template <typename MatrixType, int MapOptions, typename StrideType>
    auto Map<MatrixType, MapOptions, StrideType>::colwise() const
    {
        return axiswiseExpression<false>(Map(*this));
    }
    template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    auto Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>::array() const&
    {
        return arrayExpression(*this);
    }
    template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    auto Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>::array() &&
    {
        return arrayExpression(std::move(*this));
    }
    template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    auto Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>::array() const&&
    {
        return arrayExpression(Matrix(*this));
    }
    template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    auto Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>::rowwise() const&
    {
        return axiswiseExpression<true>(*this);
    }
    template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    auto Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>::rowwise() &&
    {
        return axiswiseExpression<true>(std::move(*this));
    }
    template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    auto Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>::rowwise() const&&
    {
        return axiswiseExpression<true>(Matrix(*this));
    }
    template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    auto Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>::colwise() const&
    {
        return axiswiseExpression<false>(*this);
    }
    template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    auto Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>::colwise() &&
    {
        return axiswiseExpression<false>(std::move(*this));
    }
    template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    auto Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>::colwise() const&&
    {
        return axiswiseExpression<false>(Matrix(*this));
    }
    template <typename Node> auto DenseExpression<Node>::array() const&
    {
        return arrayExpression(*this);
    }
    template <typename Node> auto DenseExpression<Node>::array() &&
    {
        return arrayExpression(std::move(*this));
    }
    template <typename Node> auto DenseExpression<Node>::array() const&&
    {
        return arrayExpression(DenseExpression(*this));
    }
    template <typename Node> auto DenseExpression<Node>::rowwise() const&
    {
        return axiswiseExpression<true>(*this);
    }
    template <typename Node> auto DenseExpression<Node>::rowwise() &&
    {
        return axiswiseExpression<true>(std::move(*this));
    }
    template <typename Node> auto DenseExpression<Node>::rowwise() const&&
    {
        return axiswiseExpression<true>(DenseExpression(*this));
    }
    template <typename Node> auto DenseExpression<Node>::colwise() const&
    {
        return axiswiseExpression<false>(*this);
    }
    template <typename Node> auto DenseExpression<Node>::colwise() &&
    {
        return axiswiseExpression<false>(std::move(*this));
    }
    template <typename Node> auto DenseExpression<Node>::colwise() const&&
    {
        return axiswiseExpression<false>(DenseExpression(*this));
    }

    template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    template <typename Other>
    auto Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>::cwiseProduct(Other&& other) const&
    {
        return cwiseExpression(*this, std::forward<Other>(other));
    }

    template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    template <typename Other>
    auto Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>::cwiseProduct(Other&& other) &&
    {
        return cwiseExpression(std::move(*this), std::forward<Other>(other));
    }

    template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    template <typename Other>
    auto Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>::cwiseProduct(Other&& other) const&&
    {
        return cwiseExpression(Matrix(*this), std::forward<Other>(other));
    }

    template <typename Node> Transpose<DenseExpression<Node>> DenseExpression<Node>::transpose()
    {
        return Transpose<DenseExpression>(*this);
    }

    template <typename Node> const Transpose<const DenseExpression<Node>> DenseExpression<Node>::transpose() const
    {
        return Transpose<const DenseExpression>(*this);
    }

    template <typename Node> template <typename Other> auto DenseExpression<Node>::cwiseProduct(Other&& other) const&
    {
        return cwiseExpression(*this, std::forward<Other>(other));
    }

    template <typename Node> template <typename Other> auto DenseExpression<Node>::cwiseProduct(Other&& other) &&
    {
        return cwiseExpression(std::move(*this), std::forward<Other>(other));
    }

    template <typename Node> template <typename Other> auto DenseExpression<Node>::cwiseProduct(Other&& other) const&&
    {
        return cwiseExpression(DenseExpression(*this), std::forward<Other>(other));
    }

    template <typename XprType, int BlockRows, int BlockCols, bool InnerPanel>
    auto Block<XprType, BlockRows, BlockCols, InnerPanel>::array() const
    {
        return arrayExpression(Block(*this));
    }

    template <typename XprType, int BlockRows, int BlockCols, bool InnerPanel>
    auto Block<XprType, BlockRows, BlockCols, InnerPanel>::rowwise() const
    {
        return axiswiseExpression<true>(Block(*this));
    }

    template <typename XprType, int BlockRows, int BlockCols, bool InnerPanel>
    auto Block<XprType, BlockRows, BlockCols, InnerPanel>::colwise() const
    {
        return axiswiseExpression<false>(Block(*this));
    }

    template <typename XprType, int BlockRows, int BlockCols, bool InnerPanel>
    template <typename Other>
    auto Block<XprType, BlockRows, BlockCols, InnerPanel>::cwiseProduct(Other&& other) const
    {
        return cwiseExpression(Block(*this), std::forward<Other>(other));
    }

    template <typename MatrixType> auto Transpose<MatrixType>::array() const
    {
        return arrayExpression(Transpose(*this));
    }

    template <typename MatrixType> auto Transpose<MatrixType>::rowwise() const
    {
        return axiswiseExpression<true>(Transpose(*this));
    }

    template <typename MatrixType> auto Transpose<MatrixType>::colwise() const
    {
        return axiswiseExpression<false>(Transpose(*this));
    }

    template <typename MatrixType>
    template <typename Other>
    auto Transpose<MatrixType>::cwiseProduct(Other&& other) const
    {
        return cwiseExpression(Transpose(*this), std::forward<Other>(other));
    }

} // namespace plamatrix::v1
