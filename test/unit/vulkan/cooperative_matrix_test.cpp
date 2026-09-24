#include <vector>

#include <gtest/gtest.h>

#include "plamatrix/internal/vulkan/cooperative_matrix.h"
#include "plamatrix/internal/vulkan/runtime.h"

namespace plamatrix::internal::vulkan
{
    namespace
    {

        constexpr std::size_t kMatrixElements = 16 * 16;

        std::vector<float> makeInputBatch(std::size_t batch_count)
        {
            std::vector<float> result(batch_count * kMatrixElements);
            for (std::size_t index = 0; index < result.size(); ++index)
            {
                result[index] = static_cast<float>(static_cast<int>(index % 29) - 14) * 0.037f;
            }
            return result;
        }

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

    TEST(VulkanCooperativeMatrix, ReportsAndExecutesMixedPrecisionCapability)
    {
        if (!hasUsableVulkanDevice())
        {
            GTEST_SKIP() << "No usable Vulkan compute device";
        }
        const auto capabilities = selectedVulkanCooperativeMatrixCapabilities();
        if (capabilities.enabled)
        {
            EXPECT_TRUE(capabilities.hardwareSupported);
            EXPECT_TRUE(capabilities.fp16InputsFp32Accumulation);
            EXPECT_EQ(capabilities.mSize, 16u);
            EXPECT_EQ(capabilities.nSize, 16u);
            EXPECT_EQ(capabilities.kSize, 16u);
            EXPECT_FALSE(capabilities.extensionName.empty());
        }

        const auto left = makeInputBatch(4);
        const auto right = makeInputBatch(4);
        BatchedMatrixMultiplyOptions options;
        options.allowMixedPrecision = true;
        BatchedMatrixMultiplyReport mixed_report;
        const auto mixed = batchedMultiply16x16(left, right, 4, options, &mixed_report);
        const auto reference = batchedMultiply16x16(left, right, 4);
        ASSERT_EQ(mixed.size(), reference.size());
        for (std::size_t index = 0; index < mixed.size(); ++index)
        {
            EXPECT_NEAR(mixed[index], reference[index], 1.5e-3f);
        }
        if (capabilities.enabled)
        {
            EXPECT_EQ(mixed_report.backend, BatchedMatrixMultiplyBackend::VulkanCooperativeMatrix);
            EXPECT_FALSE(mixed_report.deviceName.empty());
            EXPECT_GE(mixed_report.gpuMilliseconds, 0.0);
        }
        else
        {
            EXPECT_EQ(mixed_report.backend, BatchedMatrixMultiplyBackend::CpuFp32);
        }
    }

    TEST(VulkanCooperativeMatrix, ReusesRecordedCommandsWithFreshInput)
    {
        if (!hasUsableVulkanDevice() || !selectedVulkanCooperativeMatrixCapabilities().enabled)
        {
            GTEST_SKIP() << "Required cooperative-matrix path is unavailable";
        }

        BatchedMatrixMultiplyOptions options;
        options.allowMixedPrecision = true;
        options.requireCooperativeMatrix = true;
        const auto identity = makeIdentityBatch(2);
        const auto first_input = makeInputBatch(2);
        static_cast<void>(batchedMultiply16x16(first_input, identity, 2, options));

        auto second_input = makeInputBatch(2);
        for (float& value : second_input)
        {
            value *= -0.5f;
        }
        const auto second_output = batchedMultiply16x16(second_input, identity, 2, options);
        ASSERT_EQ(second_output.size(), second_input.size());
        for (std::size_t index = 0; index < second_output.size(); ++index)
        {
            EXPECT_NEAR(second_output[index], second_input[index], 3.0e-4f);
        }
    }

} // namespace plamatrix::internal::vulkan
