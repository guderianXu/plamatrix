#include <gtest/gtest.h>

#include <plamatrix/dense/dense_matrix.h>

using namespace plamatrix;

TEST(DenseMatrix, construction_Cpu)
{
    DenseMatrix<float, Device::CPU> mat(3, 4);
    EXPECT_EQ(mat.rows(), 3);
    EXPECT_EQ(mat.cols(), 4);
    EXPECT_EQ(mat.size(), 12);
    EXPECT_EQ(mat.device(), Device::CPU);
    EXPECT_NE(mat.data(), nullptr);
}

TEST(DenseMatrix, construction_Gpu)
{
    DenseMatrix<float, Device::GPU> mat(3, 4);
    EXPECT_EQ(mat.rows(), 3);
    EXPECT_EQ(mat.cols(), 4);
    EXPECT_EQ(mat.size(), 12);
    EXPECT_EQ(mat.device(), Device::GPU);
    EXPECT_NE(mat.data(), nullptr);
}

TEST(DenseMatrix, construction_RejectsNegativeDimensions)
{
    EXPECT_THROW((DenseMatrix<float, Device::CPU>(-1, 3)), std::invalid_argument);
    EXPECT_THROW((DenseMatrix<float, Device::GPU>(3, -1)), std::invalid_argument);
}

TEST(DenseMatrix, construction_ZeroSizedCpuMatrixIsEmpty)
{
    DenseMatrix<float, Device::CPU> mat(0, 3);
    EXPECT_EQ(mat.rows(), 0);
    EXPECT_EQ(mat.cols(), 3);
    EXPECT_EQ(mat.size(), 0);
    EXPECT_EQ(mat.data(), nullptr);

    mat.fill(1.0f);
    auto transposed = mat.transpose();
    EXPECT_EQ(transposed.rows(), 3);
    EXPECT_EQ(transposed.cols(), 0);
    EXPECT_EQ(transposed.size(), 0);
    EXPECT_EQ(transposed.data(), nullptr);
}

#ifdef PLAMATRIX_WITH_CUDA
TEST(DenseMatrix, transfer_ZeroSizedGpuRoundTrip)
{
    DenseMatrix<float, Device::CPU> cpu(0, 3);
    auto gpu = cpu.toGpu();
    EXPECT_EQ(gpu.rows(), 0);
    EXPECT_EQ(gpu.cols(), 3);
    EXPECT_EQ(gpu.size(), 0);
    EXPECT_EQ(gpu.data(), nullptr);

    auto back = gpu.toCpu();
    EXPECT_EQ(back.rows(), 0);
    EXPECT_EQ(back.cols(), 3);
    EXPECT_EQ(back.size(), 0);
    EXPECT_EQ(back.data(), nullptr);
}

TEST(DenseMatrix, transpose_ZeroSizedGpu)
{
    DenseMatrix<float, Device::CPU> cpu(0, 3);
    auto gpu = cpu.toGpu();
    auto transposed = gpu.transpose();
    EXPECT_EQ(transposed.rows(), 3);
    EXPECT_EQ(transposed.cols(), 0);
    EXPECT_EQ(transposed.size(), 0);
    EXPECT_EQ(transposed.data(), nullptr);
}
#endif

TEST(DenseMatrix, fill_Cpu)
{
    DenseMatrix<float, Device::CPU> mat(3, 4);
    mat.fill(3.14f);

    for (Index col = 0; col < mat.cols(); ++col)
    {
        for (Index row = 0; row < mat.rows(); ++row)
        {
            EXPECT_FLOAT_EQ(mat(row, col), 3.14f);
        }
    }
}

TEST(DenseMatrix, setValue_cpu_ColumnMajorIndexing)
{
    DenseMatrix<float, Device::CPU> mat(3, 4);
    mat.setValue(0, 1, 42.0f);

    EXPECT_FLOAT_EQ(mat(0, 1), 42.0f);
    EXPECT_FLOAT_EQ(mat(0, 0), 0.0f);
}

TEST(DenseMatrix, moveConstructor)
{
    DenseMatrix<float, Device::CPU> mat(3, 4);
    float* original_ptr = mat.data();

    DenseMatrix<float, Device::CPU> moved(std::move(mat));

    EXPECT_EQ(moved.rows(), 3);
    EXPECT_EQ(moved.cols(), 4);
    EXPECT_EQ(moved.data(), original_ptr);

    EXPECT_EQ(mat.data(), nullptr);
    EXPECT_EQ(mat.rows(), 0);
    EXPECT_EQ(mat.cols(), 0);
}
