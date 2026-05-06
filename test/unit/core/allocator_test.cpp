#include <gtest/gtest.h>
#include <plamatrix/core/allocator.h>

using namespace plamatrix;

TEST(Allocator, cpuAllocate_ReturnsNonNull)
{
    float* ptr = CpuAllocator<float>::allocate(100);
    ASSERT_NE(ptr, nullptr);
    CpuAllocator<float>::deallocate(ptr);
}

TEST(Allocator, cpuAllocate_CanReadWrite)
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

TEST(Allocator, gpuAllocate_ReturnsNonNull)
{
    float* ptr = GpuAllocator<float>::allocate(100);
    ASSERT_NE(ptr, nullptr);
    GpuAllocator<float>::deallocate(ptr);
}

TEST(Allocator, gpuToCpu_RoundTrip)
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
