#include <type_traits>

#include <gtest/gtest.h>

#include "plamatrix/plamatrix.h"

TEST(MatrixStructuredViews, VectorBlocksExposeEigenTypesAndWriteThrough)
{
    plamatrix::VectorXd values(5);
    values << 1.0, 2.0, 3.0, 4.0, 5.0;

    static_assert(std::is_same_v<decltype(values.segment(1, 3)),
                                 plamatrix::VectorBlock<plamatrix::VectorXd, plamatrix::Dynamic>>);
    static_assert(std::is_same_v<decltype(values.head<2>()), plamatrix::VectorBlock<plamatrix::VectorXd, 2>>);

    values.segment(1, 3) = values.head(3);
    values.tail<2>() += plamatrix::Vector2d(10.0, 20.0);
    EXPECT_DOUBLE_EQ(values(1), 1.0);
    EXPECT_DOUBLE_EQ(values(3), 13.0);
    EXPECT_DOUBLE_EQ(values(4), 25.0);
}

TEST(MatrixStructuredViews, DiagonalViewsSupportOffsetsAndDiagonalWrappers)
{
    plamatrix::MatrixXd matrix = plamatrix::MatrixXd::Zero(3, 4);
    matrix.diagonal() = plamatrix::Vector3d(1.0, 2.0, 3.0);
    matrix.diagonal<1>() = plamatrix::Vector3d(4.0, 5.0, 6.0);

    static_assert(decltype(std::declval<plamatrix::Matrix3d&>().diagonal())::RowsAtCompileTime == 3);
    static_assert(decltype(std::declval<plamatrix::Matrix3d&>().diagonal<1>())::RowsAtCompileTime == 2);
    EXPECT_DOUBLE_EQ(matrix(2, 2), 3.0);
    EXPECT_DOUBLE_EQ(matrix(2, 3), 6.0);
    EXPECT_EQ(matrix.diagonal(-1).size(), 2);

    const plamatrix::Matrix3d diagonal = plamatrix::Vector3d(7.0, 8.0, 9.0).asDiagonal();
    EXPECT_DOUBLE_EQ(diagonal(1, 1), 8.0);
    EXPECT_DOUBLE_EQ(diagonal(0, 2), 0.0);
}

TEST(MatrixStructuredViews, TriangularAndSelfAdjointViewsApplyStructure)
{
    plamatrix::Matrix3d storage;
    storage << 1.0, 20.0, 30.0, 2.0, 3.0, 40.0, 4.0, 5.0, 6.0;

    const plamatrix::Matrix3d lower = storage.triangularView<plamatrix::Lower>();
    EXPECT_DOUBLE_EQ(lower(0, 2), 0.0);
    EXPECT_DOUBLE_EQ(lower(2, 0), 4.0);

    const plamatrix::Matrix3d unit = storage.triangularView<plamatrix::UnitUpper>();
    EXPECT_DOUBLE_EQ(unit(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(unit(0, 2), 30.0);
    EXPECT_DOUBLE_EQ(unit(2, 0), 0.0);

    const plamatrix::Matrix3d symmetric = storage.selfadjointView<plamatrix::Lower>();
    EXPECT_DOUBLE_EQ(symmetric(0, 2), 4.0);
    EXPECT_DOUBLE_EQ(symmetric(2, 0), 4.0);
    storage.selfadjointView<plamatrix::Lower>()(0, 2) = 11.0;
    EXPECT_DOUBLE_EQ(storage(2, 0), 11.0);
}

TEST(MatrixStructuredViews, ReshapeReplicateReverseAndInPlaceTransposeMatchEigenSemantics)
{
    plamatrix::MatrixXd matrix(2, 3);
    matrix << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;

    const auto column_order = matrix.reshaped(3, 2);
    EXPECT_DOUBLE_EQ(column_order(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(column_order(1, 0), 4.0);
    EXPECT_DOUBLE_EQ(column_order(2, 0), 2.0);

    const auto row_order = matrix.reshaped<plamatrix::RowMajor>(3, 2);
    EXPECT_DOUBLE_EQ(row_order(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(row_order(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(row_order(1, 0), 3.0);

    const auto tiled = matrix.replicate<2, 2>();
    static_assert(decltype(tiled)::RowsAtCompileTime == plamatrix::Dynamic);
    EXPECT_EQ(tiled.rows(), 4);
    EXPECT_EQ(tiled.cols(), 6);
    EXPECT_DOUBLE_EQ(tiled(3, 5), 6.0);

    matrix.reverse()(0, 0) = 99.0;
    EXPECT_DOUBLE_EQ(matrix(1, 2), 99.0);
    matrix.reverseInPlace();
    EXPECT_DOUBLE_EQ(matrix(0, 0), 99.0);
    matrix.reverseInPlace();
    matrix.transposeInPlace();
    EXPECT_EQ(matrix.rows(), 3);
    EXPECT_EQ(matrix.cols(), 2);
    EXPECT_DOUBLE_EQ(matrix(2, 1), 99.0);

    plamatrix::Matrix3d fixed;
    fixed << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0;
    fixed.transposeInPlace();
    EXPECT_DOUBLE_EQ(fixed(0, 2), 7.0);
    EXPECT_DOUBLE_EQ(fixed(2, 0), 3.0);
}

TEST(MatrixStructuredViews, StructuredExpressionsComposeWithDenseOperators)
{
    plamatrix::Matrix3d matrix;
    matrix << 2.0, 9.0, 8.0, 1.0, 3.0, 7.0, 4.0, 5.0, 6.0;
    const plamatrix::Vector3d vector(1.0, 2.0, 3.0);

    const plamatrix::Vector3d diagonal_product = vector.asDiagonal() * vector;
    EXPECT_DOUBLE_EQ(diagonal_product(2), 9.0);

    const plamatrix::Vector3d triangular_product = matrix.triangularView<plamatrix::Lower>() * vector;
    EXPECT_DOUBLE_EQ(triangular_product(0), 2.0);
    EXPECT_DOUBLE_EQ(triangular_product(1), 7.0);
    EXPECT_DOUBLE_EQ(triangular_product(2), 32.0);

    const plamatrix::Vector3d symmetric_product = matrix.selfadjointView<plamatrix::Lower>() * vector;
    EXPECT_DOUBLE_EQ(symmetric_product(0), 16.0);
    EXPECT_DOUBLE_EQ(symmetric_product(1), 22.0);
    EXPECT_DOUBLE_EQ(symmetric_product(2), 32.0);
}
