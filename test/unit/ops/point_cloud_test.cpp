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

TEST(PointCloud, rigidTransform_RejectsNon3x3Rotation)
{
    DenseMatrix<double, Device::CPU> bad_R(3, 4);
    Vec3<double> t = {1.0, 2.0, 3.0};

    EXPECT_THROW((rigidTransform<double, Device::CPU>(bad_R, t)), std::runtime_error);
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

#ifdef PLAMATRIX_WITH_CUDA
TEST(PointCloud, covarianceMatrix_GpuMatchesCpu_MediumCloud)
{
    constexpr Index N = 4096;
    DenseMatrix<float, Device::CPU> points(N, 3);
    for (Index i = 0; i < N; ++i)
    {
        float x = static_cast<float>((i % 97) - 48) * 0.25f;
        float y = static_cast<float>((i % 53) - 26) * -0.5f;
        float z = static_cast<float>((i % 29) - 14) * 0.75f;
        points.setValue(i, 0, x);
        points.setValue(i, 1, y);
        points.setValue(i, 2, z);
    }

    auto cpu = covarianceMatrix<float, Device::CPU>(points);
    auto gpu = covarianceMatrix<float, Device::GPU>(points.toGpu()).toCpu();

    for (Index j = 0; j < 3; ++j)
    {
        for (Index i = 0; i < 3; ++i)
        {
            EXPECT_NEAR(cpu(i, j), gpu(i, j), 1e-3f)
                << "covariance mismatch at (" << i << ", " << j << ")";
        }
    }
}

TEST(PointCloud, covarianceMatrix_GpuMatchesCpu_WithLargeOffset)
{
    constexpr Index N = 4096;
    DenseMatrix<float, Device::CPU> points(N, 3);
    for (Index i = 0; i < N; ++i)
    {
        float dx = static_cast<float>((i % 17) - 8) * 0.125f;
        float dy = static_cast<float>((i % 19) - 9) * -0.25f;
        float dz = static_cast<float>((i % 23) - 11) * 0.5f;
        points.setValue(i, 0, 1000000.0f + dx);
        points.setValue(i, 1, -2000000.0f + dy);
        points.setValue(i, 2, 3000000.0f + dz);
    }

    double mean[3] = {};
    for (Index i = 0; i < N; ++i)
    {
        mean[0] += points.getValue(i, 0);
        mean[1] += points.getValue(i, 1);
        mean[2] += points.getValue(i, 2);
    }
    for (double& value : mean)
    {
        value /= static_cast<double>(N);
    }

    double reference[9] = {};
    for (Index k = 0; k < N; ++k)
    {
        double dx = static_cast<double>(points.getValue(k, 0)) - mean[0];
        double dy = static_cast<double>(points.getValue(k, 1)) - mean[1];
        double dz = static_cast<double>(points.getValue(k, 2)) - mean[2];
        double centered[3] = {dx, dy, dz};
        for (Index col = 0; col < 3; ++col)
        {
            for (Index row = 0; row < 3; ++row)
            {
                reference[row + col * 3] += centered[row] * centered[col] / static_cast<double>(N);
            }
        }
    }

    auto cpu = covarianceMatrix<float, Device::CPU>(points);
    auto gpu = covarianceMatrix<float, Device::GPU>(points.toGpu()).toCpu();

    for (Index j = 0; j < 3; ++j)
    {
        for (Index i = 0; i < 3; ++i)
        {
            float expected = static_cast<float>(reference[i + j * 3]);
            EXPECT_NEAR(cpu(i, j), expected, 1e-3f)
                << "large-offset CPU covariance mismatch at (" << i << ", " << j << ")";
            EXPECT_NEAR(gpu(i, j), expected, 1e-3f)
                << "large-offset GPU covariance mismatch at (" << i << ", " << j << ")";
        }
    }
}

TEST(PointCloud, covarianceMatrix_GpuOutputReuseAndAsyncWorkspace)
{
    DenseMatrix<float, Device::CPU> points(4, 3);
    points.setValue(0, 0, 1.0f);
    points.setValue(0, 1, 0.0f);
    points.setValue(0, 2, 0.0f);
    points.setValue(1, 0, 0.0f);
    points.setValue(1, 1, 1.0f);
    points.setValue(1, 2, 0.0f);
    points.setValue(2, 0, 0.0f);
    points.setValue(2, 1, 0.0f);
    points.setValue(2, 2, 1.0f);
    points.setValue(3, 0, 0.0f);
    points.setValue(3, 1, 0.0f);
    points.setValue(3, 2, 0.0f);

    auto expected = covarianceMatrix<float, Device::CPU>(points);
    auto points_gpu = points.toGpu();
    DenseMatrix<float, Device::GPU> sync_output(3, 3);
    DenseMatrix<float, Device::GPU> async_output(3, 3);
    GpuCovarianceWorkspace<float> workspace;

    covarianceMatrix(points_gpu, sync_output);

    cudaStream_t stream = nullptr;
    PLAMATRIX_CHECK_CUDA(cudaStreamCreate(&stream));
    covarianceMatrixAsync(points_gpu, async_output, workspace, stream);
    PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
    PLAMATRIX_CHECK_CUDA(cudaStreamDestroy(stream));

    auto sync_cpu = sync_output.toCpu();
    auto async_cpu = async_output.toCpu();
    for (Index j = 0; j < 3; ++j)
    {
        for (Index i = 0; i < 3; ++i)
        {
            EXPECT_NEAR(sync_cpu(i, j), expected(i, j), 1e-6f);
            EXPECT_NEAR(async_cpu(i, j), expected(i, j), 1e-6f);
        }
    }
}

TEST(PointCloud, covarianceMatrix_GpuWorkspaceHandlesAsyncGrowth)
{
    DenseMatrix<float, Device::CPU> small_points(4, 3);
    DenseMatrix<float, Device::CPU> large_points(300, 3);

    for (Index i = 0; i < small_points.rows(); ++i)
    {
        small_points.setValue(i, 0, static_cast<float>(i));
        small_points.setValue(i, 1, static_cast<float>(i * 2));
        small_points.setValue(i, 2, static_cast<float>(i % 2));
    }
    for (Index i = 0; i < large_points.rows(); ++i)
    {
        large_points.setValue(i, 0, static_cast<float>(i % 17) * 0.25f);
        large_points.setValue(i, 1, static_cast<float>(i % 11) * 0.5f);
        large_points.setValue(i, 2, static_cast<float>(i % 7) * 0.75f);
    }

    auto small_expected = covarianceMatrix<float, Device::CPU>(small_points);
    auto large_expected = covarianceMatrix<float, Device::CPU>(large_points);
    auto small_gpu = small_points.toGpu();
    auto large_gpu = large_points.toGpu();

    DenseMatrix<float, Device::GPU> small_output(3, 3);
    DenseMatrix<float, Device::GPU> large_output(3, 3);
    GpuCovarianceWorkspace<float> workspace;

    cudaStream_t stream = nullptr;
    PLAMATRIX_CHECK_CUDA(cudaStreamCreate(&stream));
    covarianceMatrixAsync(small_gpu, small_output, workspace, stream);
    EXPECT_EQ(workspace.blockCapacity(), 1);
    covarianceMatrixAsync(large_gpu, large_output, workspace, stream);
    EXPECT_EQ(workspace.blockCapacity(), 2);
    PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
    PLAMATRIX_CHECK_CUDA(cudaStreamDestroy(stream));

    auto small_cpu = small_output.toCpu();
    auto large_cpu = large_output.toCpu();
    for (Index j = 0; j < 3; ++j)
    {
        for (Index i = 0; i < 3; ++i)
        {
            EXPECT_NEAR(small_cpu(i, j), small_expected(i, j), 1e-6f);
            EXPECT_NEAR(large_cpu(i, j), large_expected(i, j), 1e-5f);
        }
    }
}

TEST(PointCloud, covarianceMatrix_GpuRejectsOutputDimensionMismatch)
{
    DenseMatrix<float, Device::CPU> points(4, 3);
    auto points_gpu = points.toGpu();
    DenseMatrix<float, Device::GPU> bad_output(3, 2);

    EXPECT_THROW(covarianceMatrix(points_gpu, bad_output), std::runtime_error);
}
#endif

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

#ifdef PLAMATRIX_WITH_CUDA
TEST(PointCloud, transformPoints_GpuOutputReuseAndAsync)
{
    DenseMatrix<float, Device::CPU> R(3, 3);
    for (Index i = 0; i < 3; ++i)
    {
        R.setValue(i, i, 1.0f);
    }

    Vec3<float> t = {10.0f, 0.0f, 0.0f};
    auto T_gpu = rigidTransform<float, Device::CPU>(R, t).toGpu();

    DenseMatrix<float, Device::CPU> points(2, 3);
    points.setValue(0, 0, 1.0f);
    points.setValue(0, 1, 2.0f);
    points.setValue(0, 2, 3.0f);
    points.setValue(1, 0, 4.0f);
    points.setValue(1, 1, 5.0f);
    points.setValue(1, 2, 6.0f);
    auto points_gpu = points.toGpu();

    DenseMatrix<float, Device::GPU> sync_output(2, 3);
    DenseMatrix<float, Device::GPU> async_output(2, 3);

    transformPoints(T_gpu, points_gpu, sync_output);

    cudaStream_t stream = nullptr;
    PLAMATRIX_CHECK_CUDA(cudaStreamCreate(&stream));
    transformPointsAsync(T_gpu, points_gpu, async_output, stream);
    PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
    PLAMATRIX_CHECK_CUDA(cudaStreamDestroy(stream));

    auto sync_cpu = sync_output.toCpu();
    auto async_cpu = async_output.toCpu();

    EXPECT_FLOAT_EQ(sync_cpu(0, 0), 11.0f);
    EXPECT_FLOAT_EQ(sync_cpu(0, 1), 2.0f);
    EXPECT_FLOAT_EQ(sync_cpu(0, 2), 3.0f);
    EXPECT_FLOAT_EQ(sync_cpu(1, 0), 14.0f);
    EXPECT_FLOAT_EQ(sync_cpu(1, 1), 5.0f);
    EXPECT_FLOAT_EQ(sync_cpu(1, 2), 6.0f);
    EXPECT_FLOAT_EQ(async_cpu(0, 0), 11.0f);
    EXPECT_FLOAT_EQ(async_cpu(1, 0), 14.0f);
}

TEST(PointCloud, transformPoints_GpuRejectsOutputDimensionMismatch)
{
    DenseMatrix<float, Device::CPU> R(3, 3);
    for (Index i = 0; i < 3; ++i)
    {
        R.setValue(i, i, 1.0f);
    }

    Vec3<float> t = {0.0f, 0.0f, 0.0f};
    auto T_gpu = rigidTransform<float, Device::CPU>(R, t).toGpu();
    DenseMatrix<float, Device::CPU> points(2, 3);
    auto points_gpu = points.toGpu();
    DenseMatrix<float, Device::GPU> bad_output(3, 3);

    EXPECT_THROW(transformPoints(T_gpu, points_gpu, bad_output), std::runtime_error);
}
#endif
