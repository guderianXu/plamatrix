#include <gtest/gtest.h>

#include <plamatrix/dense/dense_matrix.h>
#include <plamatrix/dense/dense_ops.h>
#include <plamatrix/ops/gemm.h>

using namespace plamatrix;

#ifdef PLAMATRIX_NO_CUDA
TEST(NoCudaStubs, checkMacros_ThrowOnStubbedErrors)
{
    EXPECT_THROW(PLAMATRIX_CHECK_CUDA(static_cast<cudaError_t>(1)), std::runtime_error);
    EXPECT_THROW(PLAMATRIX_CHECK_CUBLAS(static_cast<cublasStatus_t>(1)), std::runtime_error);
    EXPECT_THROW(PLAMATRIX_CHECK_CUSOLVER(static_cast<cusolverStatus_t>(1)), std::runtime_error);
}

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

TEST(NoCudaStubs, gpuDenseAlgorithms_ThrowClearErrorsInsteadOfLinkingMissingCudaObjects)
{
    DenseMatrix<float, Device::GPU> A(2, 2);
    DenseMatrix<float, Device::GPU> B(2, 2);

    EXPECT_THROW(A.fill(1.0f), std::runtime_error);
    EXPECT_THROW(A.transpose(), std::runtime_error);
    EXPECT_THROW(add(A, B), std::runtime_error);
    EXPECT_THROW(addAsync(A, B), std::runtime_error);
    EXPECT_THROW(sub(A, B), std::runtime_error);
    EXPECT_THROW(subAsync(A, B), std::runtime_error);
    EXPECT_THROW(gemm(A, B), std::runtime_error);
    EXPECT_THROW(gemmAsync(A, B), std::runtime_error);
}
#endif
