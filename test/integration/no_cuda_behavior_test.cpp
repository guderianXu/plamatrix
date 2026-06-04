#include <gtest/gtest.h>

#include <plamatrix/dense/dense_matrix.h>

using namespace plamatrix;

#ifdef PLAMATRIX_NO_CUDA
TEST(NoCudaStubs, transfer_RoundTripUsesStubbedDeviceMemory)
{
    DenseMatrix<double, Device::CPU> cpu(2, 3);
    cpu(0, 0) = 1.0;
    cpu(1, 0) = 2.0;
    cpu(0, 1) = 3.0;
    cpu(1, 1) = 4.0;
    cpu(0, 2) = 5.0;
    cpu(1, 2) = 6.0;

    auto device = cpu.toGpu();
    ASSERT_NE(device.data(), nullptr);
    EXPECT_EQ(device.rows(), 2);
    EXPECT_EQ(device.cols(), 3);
    EXPECT_DOUBLE_EQ(device.getValue(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(device.getValue(1, 2), 6.0);

    device.setValue(1, 2, 42.5);
    auto back = device.toCpu();

    EXPECT_EQ(back.rows(), 2);
    EXPECT_EQ(back.cols(), 3);
    EXPECT_DOUBLE_EQ(back(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(back(1, 2), 42.5);
}

TEST(NoCudaStubs, transfer_ZeroSizedDeviceMatrixRoundTrips)
{
    DenseMatrix<float, Device::CPU> cpu(0, 7);

    auto device = cpu.toGpu();
    EXPECT_EQ(device.rows(), 0);
    EXPECT_EQ(device.cols(), 7);
    EXPECT_EQ(device.size(), 0);
    EXPECT_EQ(device.data(), nullptr);

    auto back = device.toCpu();
    EXPECT_EQ(back.rows(), 0);
    EXPECT_EQ(back.cols(), 7);
    EXPECT_EQ(back.size(), 0);
    EXPECT_EQ(back.data(), nullptr);
}
#endif
