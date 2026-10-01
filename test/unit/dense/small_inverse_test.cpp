#include <array>
#include <cmath>
#include <limits>

#include <gtest/gtest.h>

#include "plamatrix/dense/matrix.h"
#include "plamatrix/dense/small_inverse.h"

namespace
{
    template <typename Scalar>
    void expectInverse(const std::array<Scalar, 9>& matrix, const std::array<Scalar, 9>& inverse, double tolerance)
    {
        for (int row = 0; row < 3; ++row)
        {
            for (int column = 0; column < 3; ++column)
            {
                Scalar product = Scalar(0);
                for (int inner = 0; inner < 3; ++inner)
                {
                    product += matrix[static_cast<std::size_t>(row * 3 + inner)] *
                               inverse[static_cast<std::size_t>(inner * 3 + column)];
                }
                EXPECT_NEAR(product, row == column ? 1.0 : 0.0, tolerance);
            }
        }
    }
} // namespace

TEST(SmallInverse, RowMajorKernelInvertsGeneralMatricesWithoutChangingInput)
{
    const std::array<double, 9> matrix{{4.0, 1.0, 2.0, 1.0, 3.0, 0.0, 2.0, 0.0, 5.0}};
    const auto original = matrix;
    std::array<double, 9> inverse{};
    double determinant = 0.0;
    ASSERT_TRUE(plamatrix::tryInverse3x3RowMajor(matrix, &inverse, 1.0e-15, &determinant));
    EXPECT_DOUBLE_EQ(determinant, 43.0);
    EXPECT_EQ(matrix, original);
    expectInverse(matrix, inverse, 1.0e-14);

    std::array<double, 9> indefinite{{1.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 3.0}};
    ASSERT_TRUE(plamatrix::tryInverse3x3RowMajor(indefinite, &indefinite, 1.0e-15, &determinant));
    EXPECT_DOUBLE_EQ(determinant, -6.0);
    EXPECT_DOUBLE_EQ(indefinite[4], -0.5);
}

TEST(SmallInverse, RejectsSingularNonFiniteAndInvalidThresholdWithoutPublishingInverse)
{
    const std::array<double, 9> near_singular{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0e-16}};
    const std::array<double, 9> unchanged{{7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0}};
    auto inverse = unchanged;
    double determinant = 0.0;
    EXPECT_FALSE(plamatrix::tryInverse3x3RowMajor(near_singular, &inverse, 1.0e-15, &determinant));
    EXPECT_DOUBLE_EQ(determinant, 1.0e-16);
    EXPECT_EQ(inverse, unchanged);

    EXPECT_TRUE(plamatrix::tryInverse3x3RowMajor(near_singular, &inverse, 1.0e-16, &determinant));
    EXPECT_DOUBLE_EQ(inverse[8], 1.0e16);
    inverse = unchanged;

    for (std::size_t index = 0; index < near_singular.size(); ++index)
    {
        for (const double bad_value :
             {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()})
        {
            auto non_finite = near_singular;
            non_finite[index] = bad_value;
            EXPECT_FALSE(plamatrix::tryInverse3x3RowMajor(non_finite, &inverse, 0.0, &determinant));
            EXPECT_DOUBLE_EQ(determinant, 0.0);
            EXPECT_EQ(inverse, unchanged);
        }
    }

    EXPECT_FALSE(plamatrix::tryInverse3x3RowMajor(near_singular, &inverse, -1.0, &determinant));
    EXPECT_EQ(inverse, unchanged);
    EXPECT_FALSE(plamatrix::tryInverse3x3RowMajor(
        near_singular, &inverse, std::numeric_limits<double>::quiet_NaN(), &determinant));
    EXPECT_EQ(inverse, unchanged);
    EXPECT_FALSE(plamatrix::tryInverse3x3RowMajor<double>(near_singular, nullptr, 0.0, &determinant));
}

TEST(SmallInverse, MatrixMemberSupportsColumnMajorRowMajorAndInPlaceOutput)
{
    const std::array<double, 9> values{{4.0, 1.0, 2.0, 1.0, 3.0, 0.0, 2.0, 0.0, 5.0}};
    plamatrix::Matrix3d column_major;
    plamatrix::Matrix<double, 3, 3, plamatrix::RowMajor> row_major;
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            const double value = values[static_cast<std::size_t>(row * 3 + column)];
            column_major(row, column) = value;
            row_major(row, column) = value;
        }
    }

    double determinant = 0.0;
    bool invertible = false;
    column_major.computeInverseAndDetWithCheck(column_major, determinant, invertible, 1.0e-15);
    ASSERT_TRUE(invertible);
    EXPECT_DOUBLE_EQ(determinant, 43.0);

    decltype(row_major) row_major_inverse;
    row_major.computeInverseAndDetWithCheck(row_major_inverse, determinant, invertible, 1.0e-15);
    ASSERT_TRUE(invertible);
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            EXPECT_NEAR(column_major(row, column), row_major_inverse(row, column), 1.0e-14);
        }
    }
}

TEST(SmallInverse, FloatKernelInvertsThreeByThreeMatrix)
{
    const std::array<float, 9> matrix{{4.0f, 1.0f, 2.0f, 1.0f, 3.0f, 0.0f, 2.0f, 0.0f, 5.0f}};
    std::array<float, 9> inverse{};
    float determinant = 0.0f;
    ASSERT_TRUE(plamatrix::tryInverse3x3RowMajor(matrix, &inverse, 1.0e-6f, &determinant));
    EXPECT_FLOAT_EQ(determinant, 43.0f);
    expectInverse(matrix, inverse, 1.0e-6);
}

TEST(SmallInverse, MatrixMemberPreservesOutputOnFailureAndRejectsRequiredDevicePolicy)
{
    plamatrix::Matrix3d singular = plamatrix::Matrix3d::Identity();
    singular(2, 2) = 0.0;
    plamatrix::Matrix3d inverse = plamatrix::Matrix3d::Identity();
    double determinant = -1.0;
    bool invertible = true;
    singular.computeInverseAndDetWithCheck(inverse, determinant, invertible, 1.0e-15);
    EXPECT_FALSE(invertible);
    EXPECT_DOUBLE_EQ(determinant, 0.0);
    EXPECT_DOUBLE_EQ(inverse(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(inverse(2, 2), 1.0);

    const plamatrix::internal::ScopedExecutionPolicy gpu_required(plamatrix::internal::ExecutionPolicy::GpuRequired);
    EXPECT_THROW(singular.computeInverseAndDetWithCheck(inverse, determinant, invertible, 1.0e-15),
                 plamatrix::internal::Error);
}
