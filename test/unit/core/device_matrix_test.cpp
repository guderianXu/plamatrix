#include <gtest/gtest.h>

#include <limits>
#include <string>
#include <utility>

#include <plamatrix/core/device_matrix.h>

#include "support/cuda_test_utils.h"

using namespace plamatrix;

namespace
{

class AsyncGpuDeviceMatrix : public DeviceMatrix<float, Device::GPU>
{
public:
    AsyncGpuDeviceMatrix()
        : DeviceMatrix(0, 0)
    {
    }

    AsyncGpuDeviceMatrix(Index rows, Index cols, cudaStream_t stream)
        : DeviceMatrix(rows, cols, detail::AsyncGpuAllocationTag{}, stream)
    {
    }

    AsyncGpuDeviceMatrix(AsyncGpuDeviceMatrix&&) noexcept = default;
    AsyncGpuDeviceMatrix& operator=(AsyncGpuDeviceMatrix&&) noexcept = default;
};

} // anonymous namespace

TEST(DeviceMatrix, cusparseErrorCheckReportsBackendExpressionAndSource)
{
    try
    {
        PLAMATRIX_CHECK_CUSPARSE(CUSPARSE_STATUS_INVALID_VALUE);
        FAIL() << "invalid cuSPARSE status should throw";
    }
    catch (const std::runtime_error& error)
    {
        const std::string message = error.what();
        EXPECT_NE(message.find("cuSPARSE"), std::string::npos);
        EXPECT_NE(message.find("CUSPARSE_STATUS_INVALID_VALUE"), std::string::npos);
        EXPECT_NE(message.find("device_matrix_test.cpp"), std::string::npos);
    }
}

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

TEST(DeviceMatrix, construction_RejectsNegativeDimensions)
{
    EXPECT_THROW((DeviceMatrix<float, Device::CPU>(-1, 3)), std::invalid_argument);
    EXPECT_THROW((DeviceMatrix<float, Device::GPU>(3, -1)), std::invalid_argument);
}

TEST(DeviceMatrix, asyncGpuConstruction_RejectsInvalidDimensionsBeforeAllocation)
{
    EXPECT_THROW(AsyncGpuDeviceMatrix(-1, 3, nullptr), std::invalid_argument);
    EXPECT_THROW(
        AsyncGpuDeviceMatrix(std::numeric_limits<Index>::max(), 2, nullptr),
        std::overflow_error);
}

#ifdef PLAMATRIX_WITH_CUDA
TEST(DeviceMatrix, asyncGpuConstruction_PreservesDimensionsOnExplicitStream)
{
    test::CudaStreamGuard stream;

    AsyncGpuDeviceMatrix matrix(3, 4, stream.get());
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

TEST(DeviceMatrix, asyncGpuMoveConstructor_PreservesStreamOrderedProvenance)
{
    test::GpuMemoryPoolGuard<float> memory_pool(true);
    test::CudaStreamGuard stream;

    AsyncGpuDeviceMatrix source(2, 3, stream.get());
    float* original_ptr = source.data();
    AsyncGpuDeviceMatrix moved(std::move(source));

    EXPECT_EQ(moved.data(), original_ptr);
    EXPECT_EQ(source.data(), nullptr);
    PLAMATRIX_CHECK_CUDA(cudaMemsetAsync(moved.data(), 0, 6 * sizeof(float), stream.get()));
    EXPECT_NO_THROW(moved.closeAsyncAllocation());
    EXPECT_EQ(moved.data(), nullptr);

    EXPECT_EQ(GpuAllocator<float>::cachedBlockCount(), 0);
    stream.synchronize();
}

TEST(DeviceMatrix, asyncGpuMoveAssignment_PreservesStreamOrderedProvenance)
{
    test::GpuMemoryPoolGuard<float> memory_pool(true);
    test::CudaStreamGuard stream;

    AsyncGpuDeviceMatrix source(2, 3, stream.get());
    float* original_ptr = source.data();
    AsyncGpuDeviceMatrix moved;
    moved = std::move(source);

    EXPECT_EQ(moved.data(), original_ptr);
    EXPECT_EQ(source.data(), nullptr);
    PLAMATRIX_CHECK_CUDA(cudaMemsetAsync(moved.data(), 0, 6 * sizeof(float), stream.get()));
    EXPECT_NO_THROW(moved.closeAsyncAllocation());
    EXPECT_EQ(moved.data(), nullptr);

    EXPECT_EQ(GpuAllocator<float>::cachedBlockCount(), 0);
    stream.synchronize();
}

TEST(DeviceMatrix, normalGpuAllocation_CloseAsyncAllocationThrowsClearLogicError)
{
    DeviceMatrix<float, Device::GPU> matrix(2, 3);

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
    EXPECT_EQ(matrix.rows(), 2);
    EXPECT_EQ(matrix.cols(), 3);
}

TEST(DeviceMatrix, normalGpuAllocation_StillUsesMemoryPool)
{
    test::GpuMemoryPoolGuard<float> memory_pool(true);

    {
        DeviceMatrix<float, Device::GPU> matrix(2, 3);
        EXPECT_NE(matrix.data(), nullptr);
    }

    EXPECT_EQ(GpuAllocator<float>::cachedBlockCount(), 1);
    EXPECT_EQ(GpuAllocator<float>::cachedBytes(), 6 * sizeof(float));
}
#else
TEST(DeviceMatrixNoCuda, asyncConstruction_ThrowsClearErrorWithoutCuda)
{
    try
    {
        static_cast<void>(AsyncGpuDeviceMatrix(3, 4, nullptr));
        FAIL() << "async construction should reject CPU-only builds";
    }
    catch (const std::runtime_error& error)
    {
        const std::string message = error.what();
        EXPECT_NE(message.find("GpuAllocator::allocateAsync"), std::string::npos);
        EXPECT_NE(message.find("PLAMATRIX_WITH_CUDA=ON"), std::string::npos);
    }
}

TEST(DeviceMatrixNoCuda, closeAsyncAllocationRejectsOrdinaryNonEmptyGpuMatrix)
{
    DeviceMatrix<float, Device::GPU> matrix(2, 3);

    EXPECT_THROW(matrix.closeAsyncAllocation(), std::logic_error);
    EXPECT_NE(matrix.data(), nullptr);
}

TEST(DeviceMatrixNoCuda, closeAsyncAllocationAcceptsEmptyGpuMatrix)
{
    DeviceMatrix<float, Device::GPU> matrix(0, 3);

    EXPECT_NO_THROW(matrix.closeAsyncAllocation());
    EXPECT_EQ(matrix.rows(), 0);
    EXPECT_EQ(matrix.cols(), 0);
    EXPECT_EQ(matrix.data(), nullptr);
}
#endif

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
