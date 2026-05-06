#include <cmath>

#include <gtest/gtest.h>

#include <plamatrix/ops/point_cloud.h>

using namespace plamatrix;

// PointCloud: rotationMatrix_ZAxis_90deg_Cpu — rotate (1,0,0) by 90 degrees around Z → (0,1,0)
TEST(PointCloud, rotationMatrix_ZAxis_90deg_Cpu)
{
    Vec3<double> axis = {0.0, 0.0, 1.0};
    double angle = M_PI / 2.0;

    auto R = rotationMatrix<double, Device::CPU>(axis, angle);

    // Check dimensions
    EXPECT_EQ(R.rows(), 3);
    EXPECT_EQ(R.cols(), 3);

    // Expected rotation matrix around Z by 90 degrees:
    // [[0, -1, 0],
    //  [1,  0, 0],
    //  [0,  0, 1]]
    EXPECT_NEAR(R(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(R(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(R(2, 0), 0.0, 1e-9);

    EXPECT_NEAR(R(0, 1), -1.0, 1e-9);
    EXPECT_NEAR(R(1, 1), 0.0, 1e-9);
    EXPECT_NEAR(R(2, 1), 0.0, 1e-9);

    EXPECT_NEAR(R(0, 2), 0.0, 1e-9);
    EXPECT_NEAR(R(1, 2), 0.0, 1e-9);
    EXPECT_NEAR(R(2, 2), 1.0, 1e-9);

    // Verify transformation: R * (1, 0, 0) should give (0, 1, 0)
    DenseMatrix<double, Device::CPU> point(1, 3);
    point.setValue(0, 0, 1.0);
    point.setValue(0, 1, 0.0);
    point.setValue(0, 2, 0.0);

    double rx = R(0, 0) * point(0, 0) + R(0, 1) * point(0, 1) + R(0, 2) * point(0, 2);
    double ry = R(1, 0) * point(0, 0) + R(1, 1) * point(0, 1) + R(1, 2) * point(0, 2);
    double rz = R(2, 0) * point(0, 0) + R(2, 1) * point(0, 1) + R(2, 2) * point(0, 2);

    EXPECT_NEAR(rx, 0.0, 1e-9);
    EXPECT_NEAR(ry, 1.0, 1e-9);
    EXPECT_NEAR(rz, 0.0, 1e-9);
}

// PointCloud: rigidTransform_4x4 — identity R + (10,20,30) translation → verify 4x4 bottom row
TEST(PointCloud, rigidTransform_4x4)
{
    // Identity rotation
    DenseMatrix<double, Device::CPU> R(3, 3);
    for (Index i = 0; i < 3; ++i)
    {
        R.setValue(i, i, 1.0);
    }

    Vec3<double> t = {10.0, 20.0, 30.0};

    auto T = rigidTransform<double, Device::CPU>(R, t);

    // Check dimensions
    EXPECT_EQ(T.rows(), 4);
    EXPECT_EQ(T.cols(), 4);

    // Check bottom row is [0, 0, 0, 1]
    EXPECT_NEAR(T(3, 0), 0.0, 1e-9);
    EXPECT_NEAR(T(3, 1), 0.0, 1e-9);
    EXPECT_NEAR(T(3, 2), 0.0, 1e-9);
    EXPECT_NEAR(T(3, 3), 1.0, 1e-9);

    // Check translation is in last column
    EXPECT_NEAR(T(0, 3), 10.0, 1e-9);
    EXPECT_NEAR(T(1, 3), 20.0, 1e-9);
    EXPECT_NEAR(T(2, 3), 30.0, 1e-9);

    // Check R is in top-left 3x3
    EXPECT_NEAR(T(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(T(1, 1), 1.0, 1e-9);
    EXPECT_NEAR(T(2, 2), 1.0, 1e-9);
    EXPECT_NEAR(T(1, 0), 0.0, 1e-9);
    EXPECT_NEAR(T(2, 0), 0.0, 1e-9);
}

// PointCloud: covarianceMatrix_4Points_Cpu — 4 points, verify 3x3 PSD matrix
TEST(PointCloud, covarianceMatrix_4Points_Cpu)
{
    // 4 points: (1,0,0), (0,1,0), (0,0,1), (0,0,0)
    DenseMatrix<double, Device::CPU> points(4, 3);
    points.setValue(0, 0, 1.0);
    points.setValue(0, 1, 0.0);
    points.setValue(0, 2, 0.0);

    points.setValue(1, 0, 0.0);
    points.setValue(1, 1, 1.0);
    points.setValue(1, 2, 0.0);

    points.setValue(2, 0, 0.0);
    points.setValue(2, 1, 0.0);
    points.setValue(2, 2, 1.0);

    points.setValue(3, 0, 0.0);
    points.setValue(3, 1, 0.0);
    points.setValue(3, 2, 0.0);

    auto C = covarianceMatrix<double, Device::CPU>(points);

    // Check dimensions
    EXPECT_EQ(C.rows(), 3);
    EXPECT_EQ(C.cols(), 3);

    // Expected covariance: centroid = (0.25, 0.25, 0.25)
    // centered points: (0.75,-0.25,-0.25), (-0.25,0.75,-0.25), (-0.25,-0.25,0.75), (-0.25,-0.25,-0.25)
    // C(i,j) = (1/4) * sum_k centered(k,i) * centered(k,j)
    // Diagonal: (1/4) * (0.5625 + 0.0625 + 0.0625 + 0.0625) = 0.1875
    // Off-diagonal: (1/4) * (-0.1875 - 0.1875 + 0.0625 + 0.0625) = -0.0625
    double expected_diag = 0.1875;
    double expected_off = -0.0625;

    EXPECT_NEAR(C(0, 0), expected_diag, 1e-9);
    EXPECT_NEAR(C(1, 1), expected_diag, 1e-9);
    EXPECT_NEAR(C(2, 2), expected_diag, 1e-9);
    EXPECT_NEAR(C(0, 1), expected_off, 1e-9);
    EXPECT_NEAR(C(1, 0), expected_off, 1e-9);
    EXPECT_NEAR(C(0, 2), expected_off, 1e-9);
    EXPECT_NEAR(C(2, 0), expected_off, 1e-9);
    EXPECT_NEAR(C(1, 2), expected_off, 1e-9);
    EXPECT_NEAR(C(2, 1), expected_off, 1e-9);
}

// PointCloud: transformPoints_Cpu — 2 points, translate by (10,0,0), verify shifted
TEST(PointCloud, transformPoints_Cpu)
{
    // Identity rotation + translation (10, 0, 0)
    DenseMatrix<double, Device::CPU> R(3, 3);
    for (Index i = 0; i < 3; ++i)
    {
        R.setValue(i, i, 1.0);
    }

    Vec3<double> t = {10.0, 0.0, 0.0};
    auto T = rigidTransform<double, Device::CPU>(R, t);

    // 2 points: (1, 2, 3) and (4, 5, 6)
    DenseMatrix<double, Device::CPU> points(2, 3);
    points.setValue(0, 0, 1.0);
    points.setValue(0, 1, 2.0);
    points.setValue(0, 2, 3.0);
    points.setValue(1, 0, 4.0);
    points.setValue(1, 1, 5.0);
    points.setValue(1, 2, 6.0);

    auto transformed = transformPoints<double, Device::CPU>(T, points);

    // Check dimensions
    EXPECT_EQ(transformed.rows(), 2);
    EXPECT_EQ(transformed.cols(), 3);

    // Point 0: (1, 2, 3) + (10, 0, 0) = (11, 2, 3)
    EXPECT_NEAR(transformed(0, 0), 11.0, 1e-9);
    EXPECT_NEAR(transformed(0, 1), 2.0, 1e-9);
    EXPECT_NEAR(transformed(0, 2), 3.0, 1e-9);

    // Point 1: (4, 5, 6) + (10, 0, 0) = (14, 5, 6)
    EXPECT_NEAR(transformed(1, 0), 14.0, 1e-9);
    EXPECT_NEAR(transformed(1, 1), 5.0, 1e-9);
    EXPECT_NEAR(transformed(1, 2), 6.0, 1e-9);
}
