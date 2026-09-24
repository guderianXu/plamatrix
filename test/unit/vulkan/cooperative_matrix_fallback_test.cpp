#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include "plamatrix/internal/vulkan/cooperative_matrix.h"

namespace plamatrix::internal::vulkan
{
    namespace
    {

        constexpr std::size_t kMatrixElements = 16 * 16;

        std::vector<float> makeIdentityBatch(std::size_t batch_count)
        {
            std::vector<float> result(batch_count * kMatrixElements, 0.0f);
            for (std::size_t batch = 0; batch < batch_count; ++batch)
            {
                for (std::size_t diagonal = 0; diagonal < 16; ++diagonal)
                {
                    result[batch * kMatrixElements + diagonal * 16 + diagonal] = 1.0f;
                }
            }
            return result;
        }

    } // namespace

    TEST(VulkanCooperativeMatrixFallback, CpuFp32IsTheDefault)
    {
        std::vector<float> left(2 * kMatrixElements);
        for (std::size_t index = 0; index < left.size(); ++index)
        {
            left[index] = static_cast<float>(static_cast<int>(index % 29) - 14) * 0.037f;
        }
        const auto right = makeIdentityBatch(2);
        BatchedMatrixMultiplyReport report;
        const auto output = batchedMultiply16x16(left, right, 2, {}, &report);
        EXPECT_EQ(output, left);
        EXPECT_EQ(report.backend, BatchedMatrixMultiplyBackend::CpuFp32);
        EXPECT_EQ(report.batchCount, 2u);
    }

    TEST(VulkanCooperativeMatrixFallback, RejectsIncorrectInputSize)
    {
        EXPECT_THROW(batchedMultiply16x16(std::vector<float>(255), std::vector<float>(256), 1), std::invalid_argument);
    }

    TEST(VulkanCooperativeMatrixFallback, UnsafeHalfInputFallsBackToFp32)
    {
        auto left = makeIdentityBatch(1);
        const auto right = makeIdentityBatch(1);
        left[0] = 70000.0f;
        BatchedMatrixMultiplyOptions options;
        options.allowMixedPrecision = true;
        BatchedMatrixMultiplyReport report;
        const auto output = batchedMultiply16x16(left, right, 1, options, &report);
        EXPECT_EQ(output, left);
        EXPECT_EQ(report.backend, BatchedMatrixMultiplyBackend::CpuFp32);
        EXPECT_TRUE(report.numericalSafetyFallback);
    }

} // namespace plamatrix::internal::vulkan
