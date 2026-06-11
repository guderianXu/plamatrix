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

TEST(DenseMatrix, construction_PinnedCpu)
{
    auto mat = DenseMatrix<float, Device::CPU>::pinned(3, 4);
    EXPECT_EQ(mat.rows(), 3);
    EXPECT_EQ(mat.cols(), 4);
    EXPECT_EQ(mat.size(), 12);
    EXPECT_EQ(mat.device(), Device::CPU);
    EXPECT_TRUE(mat.isPinnedHost());
    EXPECT_NE(mat.data(), nullptr);

    mat(0, 0) = 42.0f;
    EXPECT_FLOAT_EQ(mat(0, 0), 42.0f);
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

TEST(DenseMatrix, transferAsync_RoundTripsOnExplicitStream)
{
    auto cpu = DenseMatrix<float, Device::CPU>::pinned(2, 2);
    cpu(0, 0) = 1.0f;
    cpu(1, 0) = 2.0f;
    cpu(0, 1) = 3.0f;
    cpu(1, 1) = 4.0f;

    cudaStream_t stream = nullptr;
    PLAMATRIX_CHECK_CUDA(cudaStreamCreate(&stream));

    auto gpu = cpu.toGpuAsync(stream);
    auto back = DenseMatrix<float, Device::CPU>::pinned(2, 2);
    gpu.copyToCpuAsync(back, stream);
    PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
    PLAMATRIX_CHECK_CUDA(cudaStreamDestroy(stream));

    EXPECT_TRUE(back.isPinnedHost());
    EXPECT_FLOAT_EQ(back(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(back(1, 0), 2.0f);
    EXPECT_FLOAT_EQ(back(0, 1), 3.0f);
    EXPECT_FLOAT_EQ(back(1, 1), 4.0f);
}

TEST(DenseMatrix, transferAsync_RejectsOutputDimensionMismatch)
{
    DenseMatrix<float, Device::CPU> cpu(2, 2);
    DenseMatrix<float, Device::GPU> gpu(2, 2);
    DenseMatrix<float, Device::GPU> wrong_gpu(2, 3);
    DenseMatrix<float, Device::CPU> wrong_cpu(3, 2);

    EXPECT_THROW(cpu.copyToGpuAsync(wrong_gpu), std::runtime_error);
    EXPECT_THROW(gpu.copyToCpuAsync(wrong_cpu), std::runtime_error);
}

TEST(DenseMatrix, fill_NonZeroGpuIntMatrix)
{
    DenseMatrix<int, Device::GPU> gpu(2, 3);
    gpu.fill(7);

    const auto cpu = gpu.toCpu();
    for (Index col = 0; col < cpu.cols(); ++col)
    {
        for (Index row = 0; row < cpu.rows(); ++row)
        {
            EXPECT_EQ(cpu(row, col), 7);
        }
    }
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

TEST(DenseMatrix, accessors_RejectOutOfRangeIndices)
{
    DenseMatrix<float, Device::CPU> mat(3, 4);

    EXPECT_THROW(mat(-1, 0), std::out_of_range);
    EXPECT_THROW(mat(0, -1), std::out_of_range);
    EXPECT_THROW(mat(3, 0), std::out_of_range);
    EXPECT_THROW(mat(0, 4), std::out_of_range);
    EXPECT_THROW(mat.setValue(3, 0, 1.0f), std::out_of_range);
    EXPECT_THROW(mat.getValue(0, 4), std::out_of_range);
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
