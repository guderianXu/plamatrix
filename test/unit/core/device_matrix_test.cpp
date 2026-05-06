#include <gtest/gtest.h>

#include <plamatrix/core/device_matrix.h>

using namespace plamatrix;

TEST(DeviceMatrix, construction_DefaultDimensions)
{
    DeviceMatrix<float, Device::CPU> mat(3, 4);
    EXPECT_EQ(mat.rows(), 3);
    EXPECT_EQ(mat.cols(), 4);
    EXPECT_EQ(mat.size(), 12);
    EXPECT_EQ(mat.device(), Device::CPU);
    EXPECT_NE(mat.data(), nullptr);
}

TEST(DeviceMatrix, construction_Gpu)
{
    DeviceMatrix<float, Device::GPU> mat(5, 6);
    EXPECT_EQ(mat.rows(), 5);
    EXPECT_EQ(mat.cols(), 6);
    EXPECT_EQ(mat.size(), 30);
    EXPECT_EQ(mat.device(), Device::GPU);
    EXPECT_NE(mat.data(), nullptr);
}

TEST(DeviceMatrix, moveConstructor_TransfersOwnership)
{
    DeviceMatrix<float, Device::CPU> mat(2, 3);
    float* original_ptr = mat.data();

    DeviceMatrix<float, Device::CPU> moved(std::move(mat));

    EXPECT_EQ(moved.rows(), 2);
    EXPECT_EQ(moved.cols(), 3);
    EXPECT_EQ(moved.size(), 6);
    EXPECT_EQ(moved.data(), original_ptr);

    EXPECT_EQ(mat.data(), nullptr);
    EXPECT_EQ(mat.rows(), 0);
    EXPECT_EQ(mat.cols(), 0);
}

TEST(DeviceMatrix, moveAssignment_TransfersOwnership)
{
    DeviceMatrix<float, Device::CPU> mat1(2, 3);
    float* ptr1 = mat1.data();

    DeviceMatrix<float, Device::CPU> mat2(4, 5);

    mat2 = std::move(mat1);

    EXPECT_EQ(mat2.rows(), 2);
    EXPECT_EQ(mat2.cols(), 3);
    EXPECT_EQ(mat2.data(), ptr1);

    EXPECT_EQ(mat1.data(), nullptr);
    EXPECT_EQ(mat1.rows(), 0);
    EXPECT_EQ(mat1.cols(), 0);
}
