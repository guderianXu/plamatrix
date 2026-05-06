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
