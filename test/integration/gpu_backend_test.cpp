#include <gtest/gtest.h>

#include <cstring>

#include <plamatrix/core/gpu_backend.h>

using namespace plamatrix;

TEST(BackendQuery, query_ReturnsConfiguredBackendName)
{
    const char* name = gpuBackendName();
    ASSERT_NE(name, nullptr);

#ifdef PLAMATRIX_WITH_CUDA
    EXPECT_EQ(std::strcmp(name, "cuda"), 0);
    EXPECT_EQ(gpuBackend(), GpuBackend::Cuda);
#elif defined(PLAMATRIX_WITH_METAL)
    EXPECT_EQ(std::strcmp(name, "metal"), 0);
    EXPECT_EQ(gpuBackend(), GpuBackend::Metal);
#else
    EXPECT_EQ(std::strcmp(name, "none"), 0);
    EXPECT_EQ(gpuBackend(), GpuBackend::None);
#endif
}

TEST(BackendQuery, compileDefinitions_AreMutuallyExclusive)
{
#if defined(PLAMATRIX_WITH_CUDA) && defined(PLAMATRIX_WITH_METAL)
    FAIL() << "CUDA and Metal backends must not be enabled together";
#elif defined(PLAMATRIX_WITH_CUDA)
    SUCCEED() << "CUDA backend selected";
#elif defined(PLAMATRIX_WITH_METAL)
    SUCCEED() << "Metal backend selected";
#else
    SUCCEED() << "No GPU backend selected";
#endif
}
