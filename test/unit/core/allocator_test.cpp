#include <gtest/gtest.h>

#include <limits>

#include <plamatrix/core/allocator.h>

using namespace plamatrix;

TEST(CpuAllocator, allocate_ReturnsNonNull)
{
    float* ptr = CpuAllocator<float>::allocate(100);
    ASSERT_NE(ptr, nullptr);
    CpuAllocator<float>::deallocate(ptr);
}

TEST(CpuAllocator, allocate_CanReadWrite)
{
    float* ptr = CpuAllocator<float>::allocate(3);
    ptr[0] = 1.0f;
    ptr[1] = 2.0f;
    ptr[2] = 3.0f;
    EXPECT_FLOAT_EQ(ptr[0], 1.0f);
    EXPECT_FLOAT_EQ(ptr[1], 2.0f);
    EXPECT_FLOAT_EQ(ptr[2], 3.0f);
    CpuAllocator<float>::deallocate(ptr);
}

TEST(CpuAllocator, allocate_RejectsByteSizeOverflow)
{
    const std::size_t count = std::numeric_limits<std::size_t>::max() / sizeof(float) + 1;
    EXPECT_THROW(CpuAllocator<float>::allocate(count), std::overflow_error);
}

TEST(PinnedCpuAllocator, allocate_ReturnsNonNullAndCanReadWrite)
{
    float* ptr = PinnedCpuAllocator<float>::allocate(3);
    ASSERT_NE(ptr, nullptr);
    ptr[0] = 1.0f;
    ptr[1] = 2.0f;
    ptr[2] = 3.0f;
    EXPECT_FLOAT_EQ(ptr[0], 1.0f);
    EXPECT_FLOAT_EQ(ptr[1], 2.0f);
    EXPECT_FLOAT_EQ(ptr[2], 3.0f);
    PinnedCpuAllocator<float>::deallocate(ptr);
}

TEST(PinnedCpuAllocator, allocate_RejectsByteSizeOverflow)
{
    const std::size_t count = std::numeric_limits<std::size_t>::max() / sizeof(float) + 1;
    EXPECT_THROW(PinnedCpuAllocator<float>::allocate(count), std::overflow_error);
}

TEST(GpuAllocator, allocate_ReturnsNonNull)
{
    float* ptr = GpuAllocator<float>::allocate(100);
    ASSERT_NE(ptr, nullptr);
    GpuAllocator<float>::deallocate(ptr);
}

TEST(GpuAllocator, allocate_CudaMemcpyRoundTrip)
{
    float host_data[3] = { 1.0f, 2.0f, 3.0f };
    float* gpu_ptr = GpuAllocator<float>::allocate(3);
    PLAMATRIX_CHECK_CUDA(cudaMemcpy(gpu_ptr, host_data, 3 * sizeof(float), cudaMemcpyHostToDevice));

    float result[3] = {};
    PLAMATRIX_CHECK_CUDA(cudaMemcpy(result, gpu_ptr, 3 * sizeof(float), cudaMemcpyDeviceToHost));

    EXPECT_FLOAT_EQ(result[0], 1.0f);
    EXPECT_FLOAT_EQ(result[1], 2.0f);
    EXPECT_FLOAT_EQ(result[2], 3.0f);

    GpuAllocator<float>::deallocate(gpu_ptr);
}

TEST(GpuAllocator, allocate_RejectsByteSizeOverflow)
{
    const std::size_t count = std::numeric_limits<std::size_t>::max() / sizeof(float) + 1;
    EXPECT_THROW(GpuAllocator<float>::allocate(count), std::overflow_error);
}

#ifdef PLAMATRIX_WITH_CUDA
TEST(GpuAllocator, memoryPool_ReusesReturnedBlockWhenEnabled)
{
    GpuAllocator<float>::setMemoryPoolEnabled(true);
    GpuAllocator<float>::releaseMemoryPool();

    float* first = GpuAllocator<float>::allocate(128);
    GpuAllocator<float>::deallocate(first, 128);

    EXPECT_EQ(GpuAllocator<float>::cachedBlockCount(), 1);
    EXPECT_EQ(GpuAllocator<float>::cachedBytes(), 128 * sizeof(float));

    float* second = GpuAllocator<float>::allocate(128);
    EXPECT_EQ(second, first);
    EXPECT_EQ(GpuAllocator<float>::cachedBlockCount(), 0);

    GpuAllocator<float>::deallocate(second, 128);
    GpuAllocator<float>::releaseMemoryPool();
    GpuAllocator<float>::setMemoryPoolEnabled(false);
}
#endif
