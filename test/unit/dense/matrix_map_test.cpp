#include <gtest/gtest.h>

#include <array>
#include <type_traits>

#include "plamatrix/dense/matrix_map.h"

TEST(MatrixMap, FixedAndDynamicMapsShareCallerStorage)
{
    std::array<double, 6> column_major{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    plamatrix::Map<plamatrix::Matrix<double, 2, 3>> fixed(column_major.data());
    static_assert(decltype(fixed)::RowsAtCompileTime == 2);
    EXPECT_DOUBLE_EQ(fixed(1, 2), 6.0);
    fixed(0, 1) = 9.0;
    EXPECT_DOUBLE_EQ(column_major[2], 9.0);

    plamatrix::Map<const plamatrix::MatrixXd> read_only(column_major.data(), 2, 3);
    static_assert(std::is_same_v<decltype(read_only(0, 0)), const double&>);
    EXPECT_DOUBLE_EQ(read_only(0, 1), 9.0);
    plamatrix::MatrixXd copied = fixed + read_only;
    EXPECT_DOUBLE_EQ(copied(0, 1), 18.0);
}

TEST(MatrixMap, StridedMapParticipatesInExpressionsAndAssignment)
{
    std::array<double, 8> row_major{1.0, 2.0, 3.0, -1.0, 4.0, 5.0, 6.0, -1.0};
    using RowMajorMap = plamatrix::
        Map<plamatrix::MatrixXd, plamatrix::Unaligned, plamatrix::Stride<plamatrix::Dynamic, plamatrix::Dynamic>>;
    RowMajorMap mapped(row_major.data(), 2, 3, plamatrix::Stride<plamatrix::Dynamic, plamatrix::Dynamic>(1, 4));
    EXPECT_DOUBLE_EQ(mapped(1, 2), 6.0);
    const plamatrix::MatrixXd gram = mapped.transpose() * mapped;
    EXPECT_DOUBLE_EQ(gram(0, 0), 17.0);
    EXPECT_DOUBLE_EQ(gram(2, 2), 45.0);

    plamatrix::MatrixXd replacement = plamatrix::MatrixXd::Constant(2, 3, 7.0);
    mapped = replacement;
    EXPECT_DOUBLE_EQ(row_major[0], 7.0);
    EXPECT_DOUBLE_EQ(row_major[6], 7.0);
    EXPECT_DOUBLE_EQ(row_major[3], -1.0);
    EXPECT_DOUBLE_EQ(row_major[7], -1.0);
}

TEST(MatrixMap, CopyAssignmentWritesCoefficientsAndFacadesRemainLazy)
{
    std::array<double, 4> source{1.0, 2.0, 3.0, 4.0};
    std::array<double, 4> destination{};
    plamatrix::Map<plamatrix::Matrix2d> input(source.data());
    plamatrix::Map<plamatrix::Matrix2d> output(destination.data());
    output = input;
    EXPECT_EQ(destination, source);
    output(0, 0) = 9.0;
    EXPECT_DOUBLE_EQ(source[0], 1.0);

    auto squared = input.array().square();
    source[0] = 5.0;
    EXPECT_DOUBLE_EQ(squared(0, 0), 25.0);
    EXPECT_DOUBLE_EQ(input.rowwise().sum()(0), 8.0);
    const plamatrix::Map<const plamatrix::Matrix2d> read_only(source.data());
    EXPECT_DOUBLE_EQ(read_only.colwise().sum()(0, 1), 7.0);
}

TEST(MatrixMap, VectorAndInvalidLayoutFollowMatrixShape)
{
    std::array<double, 3> values{1.0, 2.0, 3.0};
    plamatrix::Map<plamatrix::Vector3d> fixed(values.data());
    plamatrix::Map<const plamatrix::VectorXd> dynamic(values.data(), 3);
    EXPECT_DOUBLE_EQ(fixed.dot(plamatrix::Vector3d(1.0, 1.0, 1.0)), 6.0);
    EXPECT_DOUBLE_EQ(dynamic(2), 3.0);
    EXPECT_THROW((plamatrix::Map<plamatrix::Matrix3d>(values.data(), 2, 3)), std::invalid_argument);
    EXPECT_THROW((plamatrix::Map<plamatrix::MatrixXd>(nullptr, 2, 3)), std::invalid_argument);
}
