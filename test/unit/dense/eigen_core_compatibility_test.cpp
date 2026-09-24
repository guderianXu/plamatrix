#include <gtest/gtest.h>

#include <cstdint>

#include <algorithm>
#include <array>
#include <cmath>
#include <type_traits>
#include <utility>

#include "plamatrix/plamatrix.h"

namespace
{
    template <typename Derived>
    typename Derived::Scalar normThroughMatrixBase(const plamatrix::MatrixBase<Derived>& value)
    {
        return value.norm();
    }

    template <typename Derived>
    typename Derived::Scalar sumThroughMatrixBase(const plamatrix::MatrixBase<Derived>& value)
    {
        return value.sum();
    }

    template <typename Derived> plamatrix::Index coefficientCount(const plamatrix::DenseBase<Derived>& value)
    {
        return value.size();
    }

    template <typename Derived>
    auto fixedBlockThroughMatrixBase(plamatrix::MatrixBase<Derived>& value) ->
        typename Derived::template FixedBlockXpr<2, 2>::Type
    {
        return value.template block<2, 2>(0, 0);
    }
} // namespace

TEST(EigenCoreCompatibility, MatrixUsesEigenTemplateContractAndBaseHierarchy)
{
    using BoundedRowMajor =
        plamatrix::Matrix<double, plamatrix::Dynamic, plamatrix::Dynamic, plamatrix::RowMajor, 4, 5>;
    static_assert(BoundedRowMajor::Options == plamatrix::RowMajor);
    static_assert(BoundedRowMajor::MaxRowsAtCompileTime == 4);
    static_assert(BoundedRowMajor::MaxColsAtCompileTime == 5);
    static_assert(BoundedRowMajor::IsRowMajor == 1);
    static_assert(std::is_base_of_v<plamatrix::EigenBase<plamatrix::MatrixXd>, plamatrix::MatrixXd>);
    static_assert(std::is_base_of_v<plamatrix::DenseBase<plamatrix::MatrixXd>, plamatrix::MatrixXd>);
    static_assert(std::is_base_of_v<plamatrix::MatrixBase<plamatrix::MatrixXd>, plamatrix::MatrixXd>);

    BoundedRowMajor matrix(2, 3);
    matrix << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
    EXPECT_EQ(coefficientCount(matrix), 6);
    EXPECT_DOUBLE_EQ(normThroughMatrixBase(matrix), std::sqrt(91.0));
    EXPECT_DOUBLE_EQ(sumThroughMatrixBase(matrix), 21.0);
    EXPECT_THROW(matrix.resize(5, 3), std::runtime_error);
}

TEST(EigenCoreCompatibility, DynamicMatrixStorageSupportsWidePackets)
{
    plamatrix::MatrixXd matrix(17, 19);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(matrix.data()) % 64, 0U);

    plamatrix::Matrix<double, plamatrix::Dynamic, plamatrix::Dynamic, plamatrix::DontAlign> unaligned(17, 19);
    EXPECT_NE(unaligned.data(), nullptr);
}

TEST(EigenCoreCompatibility, RowMajorStorageAndExpressionsPreserveLogicalCoefficients)
{
    using RowMajorMatrix = plamatrix::Matrix<double, plamatrix::Dynamic, plamatrix::Dynamic, plamatrix::RowMajor>;
    RowMajorMatrix left(2, 3);
    left << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
    const std::array<double, 6> expected_storage{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    EXPECT_TRUE(std::equal(expected_storage.begin(), expected_storage.end(), left.data()));

    RowMajorMatrix right = RowMajorMatrix::Ones(2, 3);
    RowMajorMatrix sum = left + right;
    EXPECT_DOUBLE_EQ(sum(0, 2), 4.0);
    EXPECT_DOUBLE_EQ(sum(1, 0), 5.0);

    plamatrix::Matrix<double, 3, 2, plamatrix::RowMajor> multiplier;
    multiplier << 1.0, 0.0, 0.0, 1.0, 1.0, 1.0;
    const plamatrix::MatrixXd product = left * multiplier;
    EXPECT_DOUBLE_EQ(product(0, 0), 4.0);
    EXPECT_DOUBLE_EQ(product(0, 1), 5.0);
    EXPECT_DOUBLE_EQ(product(1, 0), 10.0);
    EXPECT_DOUBLE_EQ(product(1, 1), 11.0);
}

TEST(EigenCoreCompatibility, MapAndRefUseEigenTemplateSignatures)
{
    using RowMajorMatrix = plamatrix::Matrix<double, plamatrix::Dynamic, plamatrix::Dynamic, plamatrix::RowMajor>;
    using StridedMap =
        plamatrix::Map<RowMajorMatrix, plamatrix::Unaligned, plamatrix::Stride<plamatrix::Dynamic, plamatrix::Dynamic>>;

    std::array<double, 8> padded{1.0, 2.0, 3.0, -1.0, 4.0, 5.0, 6.0, -1.0};
    StridedMap mapped(padded.data(), 2, 3, plamatrix::Stride<plamatrix::Dynamic, plamatrix::Dynamic>(4, 1));
    static_assert(std::is_base_of_v<plamatrix::MatrixBase<StridedMap>, StridedMap>);
    EXPECT_DOUBLE_EQ(mapped(1, 2), 6.0);
    EXPECT_DOUBLE_EQ(sumThroughMatrixBase(mapped), 21.0);

    RowMajorMatrix matrix = RowMajorMatrix::Zero(2, 3);
    plamatrix::Ref<RowMajorMatrix> mutable_ref(matrix);
    plamatrix::Ref<const RowMajorMatrix> const_ref(matrix);
    static_assert(std::is_base_of_v<plamatrix::MatrixBase<decltype(mutable_ref)>, decltype(mutable_ref)>);
    mutable_ref(1, 2) = 9.0;
    EXPECT_DOUBLE_EQ(const_ref(1, 2), 9.0);

    plamatrix::Matrix3d fixed = plamatrix::Matrix3d::Identity();
    plamatrix::Ref<plamatrix::MatrixXd> dynamic_ref(fixed);
    plamatrix::Ref<const plamatrix::MatrixXd> dynamic_const_ref(fixed);
    dynamic_ref(0, 1) = 4.0;
    EXPECT_DOUBLE_EQ(dynamic_const_ref(0, 1), 4.0);
}

TEST(EigenCoreCompatibility, ArrayExpressionsUseArrayBase)
{
    plamatrix::Matrix2d matrix;
    matrix << 1.0, 2.0, 3.0, 4.0;
    auto array = matrix.array();
    static_assert(std::is_base_of_v<plamatrix::ArrayBase<decltype(array)>, decltype(array)>);
    EXPECT_EQ(array.rows(), 2);
    EXPECT_DOUBLE_EQ(array(1, 0), 3.0);
}

TEST(EigenCoreCompatibility, MapSupportsEigenNegativeInnerStride)
{
    std::array<double, 3> storage{1.0, 2.0, 3.0};
    using ReverseMap =
        plamatrix::Map<plamatrix::VectorXd, plamatrix::Unaligned, plamatrix::InnerStride<plamatrix::Dynamic>>;
    ReverseMap reversed(storage.data() + 2, 3, plamatrix::InnerStride<plamatrix::Dynamic>(-1));
    EXPECT_DOUBLE_EQ(reversed(0), 3.0);
    EXPECT_DOUBLE_EQ(reversed(2), 1.0);
}

TEST(EigenCoreCompatibility, BlockAndTransposeExposeEigenReturnTypes)
{
    using Matrix = plamatrix::Matrix<double, plamatrix::Dynamic, plamatrix::Dynamic>;
    using RowMajorMatrix = plamatrix::Matrix<double, plamatrix::Dynamic, plamatrix::Dynamic, plamatrix::RowMajor>;
    using DynamicBlock = plamatrix::Block<Matrix>;
    using ConstDynamicBlock = const plamatrix::Block<const Matrix>;
    using FixedBlock = plamatrix::Block<Matrix, 2, 3>;
    using Row = plamatrix::Block<Matrix, 1, plamatrix::Dynamic, false>;
    using Column = plamatrix::Block<Matrix, plamatrix::Dynamic, 1, true>;
    using Mapped = plamatrix::Map<Matrix>;
    using Referenced = plamatrix::Ref<Matrix>;
    using SumExpression = decltype(std::declval<Matrix>() + std::declval<Matrix>());

    static_assert(std::is_same_v<decltype(std::declval<Matrix&>().block(0, 0, 2, 3)), DynamicBlock>);
    static_assert(std::is_same_v<decltype(std::declval<const Matrix&>().block(0, 0, 2, 3)), ConstDynamicBlock>);
    static_assert(std::is_same_v<decltype(std::declval<Matrix&>().template block<2, 3>(0, 0)), FixedBlock>);
    static_assert(std::is_same_v<decltype(std::declval<Matrix&>().row(0)), Row>);
    static_assert(std::is_same_v<decltype(std::declval<Matrix&>().col(0)), Column>);
    static_assert(std::is_same_v<decltype(std::declval<Matrix&>().transpose()), plamatrix::Transpose<Matrix>>);
    static_assert(std::is_same_v<decltype(std::declval<Matrix&&>().transpose()), plamatrix::Transpose<Matrix>>);
    static_assert(
        std::is_same_v<decltype(std::declval<const Matrix&>().transpose()), const plamatrix::Transpose<const Matrix>>);
    static_assert(std::is_same_v<decltype(std::declval<FixedBlock&&>().transpose()), plamatrix::Transpose<FixedBlock>>);
    static_assert(
        std::is_same_v<decltype(std::declval<SumExpression&&>().transpose()), plamatrix::Transpose<SumExpression>>);
    static_assert(std::is_same_v<decltype(std::declval<RowMajorMatrix&>().row(0)),
                                 plamatrix::Block<RowMajorMatrix, 1, plamatrix::Dynamic, true>>);
    static_assert(std::is_same_v<typename Matrix::BlockXpr, DynamicBlock>);
    static_assert(std::is_same_v<typename Matrix::ConstBlockXpr, ConstDynamicBlock>);
    static_assert(std::is_same_v<typename Matrix::RowXpr, Row>);
    static_assert(std::is_same_v<typename Matrix::ColXpr, Column>);
    static_assert(std::is_same_v<typename Matrix::TransposeReturnType, plamatrix::Transpose<Matrix>>);
    static_assert(std::is_same_v<typename Matrix::ConstTransposeReturnType, plamatrix::Transpose<const Matrix>>);
    static_assert(std::is_same_v<decltype(std::declval<Mapped&>().block(0, 0, 2, 3)), plamatrix::Block<Mapped>>);
    static_assert(std::is_same_v<decltype(std::declval<Mapped&>().transpose()), plamatrix::Transpose<Mapped>>);
    static_assert(
        std::is_same_v<decltype(std::declval<Referenced&>().block(0, 0, 2, 3)), plamatrix::Block<Referenced>>);
    static_assert(std::is_same_v<decltype(std::declval<Referenced&>().transpose()), plamatrix::Transpose<Referenced>>);

    Matrix matrix(4, 5);
    for (plamatrix::Index col = 0; col < matrix.cols(); ++col)
    {
        for (plamatrix::Index row = 0; row < matrix.rows(); ++row)
        {
            matrix(row, col) = static_cast<double>(row + 10 * col);
        }
    }
    auto block = matrix.block<2, 3>(1, 1);
    auto transpose = block.transpose();
    static_assert(std::is_same_v<decltype(transpose), plamatrix::Transpose<FixedBlock>>);
    EXPECT_EQ(transpose.rows(), 3);
    EXPECT_EQ(transpose.cols(), 2);
    EXPECT_DOUBLE_EQ(transpose(2, 1), matrix(2, 3));

    transpose(0, 0) = 99.0;
    EXPECT_DOUBLE_EQ(matrix(1, 1), 99.0);

    auto generic_block = fixedBlockThroughMatrixBase(matrix);
    generic_block(0, 0) = 77.0;
    EXPECT_DOUBLE_EQ(matrix(0, 0), 77.0);
}

TEST(EigenCoreCompatibility, RefBindsNonContiguousBlocksWithTheirRuntimeStrides)
{
    plamatrix::MatrixXd matrix(6, 5);
    for (plamatrix::Index col = 0; col < matrix.cols(); ++col)
    {
        for (plamatrix::Index row = 0; row < matrix.rows(); ++row)
        {
            matrix(row, col) = static_cast<double>(row + 10 * col);
        }
    }

    auto block = matrix.block(1, 1, 3, 2);
    EXPECT_EQ(block.innerStride(), 1);
    EXPECT_EQ(block.outerStride(), 6);
    plamatrix::Ref<plamatrix::MatrixXd> ref(block);
    EXPECT_EQ(ref.innerStride(), 1);
    EXPECT_EQ(ref.outerStride(), 6);
    ref(2, 1) = 123.0;
    EXPECT_DOUBLE_EQ(matrix(3, 2), 123.0);

    const auto const_block = static_cast<const plamatrix::MatrixXd&>(matrix).block(2, 2, 2, 2);
    plamatrix::Ref<const plamatrix::MatrixXd> const_ref(const_block);
    EXPECT_EQ(const_ref.outerStride(), 6);
    EXPECT_DOUBLE_EQ(const_ref(1, 0), matrix(3, 2));

    auto row = matrix.row(4);
    plamatrix::Ref<plamatrix::RowVectorXd, 0, plamatrix::InnerStride<plamatrix::Dynamic>> row_ref(row);
    EXPECT_EQ(row_ref.innerStride(), 6);
    row_ref(3) = 456.0;
    EXPECT_DOUBLE_EQ(matrix(4, 3), 456.0);

    auto nested = block.block(1, 0, 2, 2);
    plamatrix::Ref<plamatrix::MatrixXd> nested_ref(nested);
    EXPECT_EQ(nested_ref.outerStride(), 6);
    nested_ref(0, 0) = 789.0;
    EXPECT_DOUBLE_EQ(matrix(2, 1), 789.0);

    using RowMajorMatrix = plamatrix::Matrix<double, plamatrix::Dynamic, plamatrix::Dynamic, plamatrix::RowMajor>;
    RowMajorMatrix row_major = RowMajorMatrix::Zero(5, 6);
    auto row_major_block = row_major.block(1, 2, 3, 2);
    plamatrix::Ref<RowMajorMatrix> row_major_ref(row_major_block);
    EXPECT_EQ(row_major_ref.innerStride(), 1);
    EXPECT_EQ(row_major_ref.outerStride(), 6);
    row_major_ref(2, 1) = 321.0;
    EXPECT_DOUBLE_EQ(row_major(3, 3), 321.0);

    const auto write_temporary_block = [](plamatrix::Ref<plamatrix::MatrixXd> values) { values(0, 0) = 654.0; };
    write_temporary_block(matrix.block(0, 3, 2, 2));
    EXPECT_DOUBLE_EQ(matrix(0, 3), 654.0);

    const auto read_temporary_block = [](plamatrix::Ref<const plamatrix::MatrixXd> values) { return values.sum(); };
    EXPECT_DOUBLE_EQ(read_temporary_block(matrix.block(0, 3, 2, 2)),
                     matrix(0, 3) + matrix(1, 3) + matrix(0, 4) + matrix(1, 4));
}
