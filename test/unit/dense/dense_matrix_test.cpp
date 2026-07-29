#include <gtest/gtest.h>

#include <limits>
#include <string>
#include <utility>

#include <plamatrix/dense/dense_matrix.h>

#include "support/cuda_test_utils.h"

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

TEST(DenseMatrix, uninitializedAsync_RejectsInvalidDimensionsBeforeAllocation)
{
    EXPECT_THROW(
        (DenseMatrix<float, Device::GPU>::uninitializedAsync(-1, 3)),
        std::invalid_argument);
    EXPECT_THROW(
        (DenseMatrix<float, Device::GPU>::uninitializedAsync(
            std::numeric_limits<Index>::max(), 2)),
        std::overflow_error);
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
TEST(DenseMatrix, uninitialized_UsesOrdinaryLifetimeAcrossStreamDestruction)
{
    DenseMatrix<float, Device::GPU> matrix;
    {
        cudaStream_t stream = nullptr;
        PLAMATRIX_CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        matrix = DenseMatrix<float, Device::GPU>::uninitialized(3, 4);
        PLAMATRIX_CHECK_CUDA(cudaMemsetAsync(
            matrix.data(), 0, 12 * sizeof(float), stream));
        PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
        PLAMATRIX_CHECK_CUDA(cudaStreamDestroy(stream));
    }

    EXPECT_EQ(matrix.rows(), 3);
    EXPECT_EQ(matrix.cols(), 4);
    EXPECT_NE(matrix.data(), nullptr);
    EXPECT_NO_THROW(matrix.toCpu());
}

TEST(DenseMatrix, uninitializedAsync_UsesExplicitStreamAndPreservesDimensions)
{
    test::CudaStreamGuard stream;

    auto matrix = DenseMatrix<float, Device::GPU>::uninitializedAsync(3, 4, stream.get());
    EXPECT_EQ(matrix.rows(), 3);
    EXPECT_EQ(matrix.cols(), 4);
    EXPECT_EQ(matrix.size(), 12);
    EXPECT_NE(matrix.data(), nullptr);

    PLAMATRIX_CHECK_CUDA(cudaMemsetAsync(matrix.data(), 0, 12 * sizeof(float), stream.get()));
    EXPECT_NO_THROW(matrix.closeAsyncAllocation());
    EXPECT_EQ(matrix.rows(), 0);
    EXPECT_EQ(matrix.cols(), 0);
    EXPECT_EQ(matrix.data(), nullptr);

    EXPECT_NO_THROW(matrix.closeAsyncAllocation());
    stream.synchronize();
}

TEST(DenseMatrix, uninitializedAsync_MoveConstructorRetainsCloseProvenance)
{
    test::CudaStreamGuard stream;

    auto matrix = DenseMatrix<float, Device::GPU>::uninitializedAsync(2, 3, stream.get());
    float* original_ptr = matrix.data();
    DenseMatrix<float, Device::GPU> moved(std::move(matrix));

    EXPECT_EQ(moved.data(), original_ptr);
    EXPECT_EQ(moved.rows(), 2);
    EXPECT_EQ(moved.cols(), 3);
    EXPECT_EQ(matrix.data(), nullptr);
    EXPECT_EQ(matrix.rows(), 0);
    EXPECT_EQ(matrix.cols(), 0);

    EXPECT_NO_THROW(moved.closeAsyncAllocation());
    EXPECT_EQ(moved.data(), nullptr);
    EXPECT_EQ(moved.rows(), 0);
    EXPECT_EQ(moved.cols(), 0);

    stream.synchronize();
}

TEST(DenseMatrix, uninitializedAsync_MoveAssignmentRetainsCloseProvenance)
{
    test::CudaStreamGuard stream;

    auto source = DenseMatrix<float, Device::GPU>::uninitializedAsync(2, 3, stream.get());
    DenseMatrix<float, Device::GPU> moved;
    moved = std::move(source);

    EXPECT_EQ(source.data(), nullptr);
    EXPECT_NO_THROW(moved.closeAsyncAllocation());
    EXPECT_EQ(moved.data(), nullptr);

    stream.synchronize();
}

TEST(DenseMatrix, closeAsyncAllocationRejectsOrdinaryAllocationWithClearLogicError)
{
    DenseMatrix<float, Device::GPU> matrix(2, 3);

    try
    {
        matrix.closeAsyncAllocation();
        FAIL() << "closeAsyncAllocation should reject ordinary GPU allocation";
    }
    catch (const std::logic_error& error)
    {
        EXPECT_NE(std::string(error.what()).find("stream-ordered GPU allocation"),
                  std::string::npos);
    }

    EXPECT_NE(matrix.data(), nullptr);
}

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

    test::CudaStreamGuard stream;

    auto gpu = cpu.toGpuAsync(stream.get());
    auto back = DenseMatrix<float, Device::CPU>::pinned(2, 2);
    gpu.copyToCpuAsync(back, stream.get());
    stream.synchronize();

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
#else
TEST(DenseMatrix, uninitializedAsync_ThrowsClearErrorWithoutCuda)
{
    try
    {
        static_cast<void>(DenseMatrix<float, Device::GPU>::uninitializedAsync(2, 3));
        FAIL() << "uninitializedAsync should reject CPU-only builds";
    }
    catch (const std::runtime_error& error)
    {
        const std::string message = error.what();
        EXPECT_NE(message.find("uninitializedAsync"), std::string::npos);
        EXPECT_NE(message.find("PLAMATRIX_WITH_CUDA=ON"), std::string::npos);
    }
}

TEST(DenseMatrixNoCuda, closeAsyncAllocationContractIsExplicit)
{
    DenseMatrix<float, Device::GPU> ordinary(2, 3);
    EXPECT_THROW(ordinary.closeAsyncAllocation(), std::logic_error);
    EXPECT_NE(ordinary.data(), nullptr);

    DenseMatrix<float, Device::GPU> empty(0, 3);
    EXPECT_NO_THROW(empty.closeAsyncAllocation());
    EXPECT_EQ(empty.rows(), 0);
    EXPECT_EQ(empty.cols(), 0);
    EXPECT_EQ(empty.data(), nullptr);
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
