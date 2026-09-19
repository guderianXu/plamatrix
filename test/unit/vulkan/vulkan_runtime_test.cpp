#include <algorithm>
#include <cmath>

#include <gtest/gtest.h>

#include "plamatrix/sparse/iterative_solver.h"
#include "plamatrix/vulkan/execution.h"
#include "plamatrix/vulkan/iterative_solver.h"
#include "plamatrix/vulkan/runtime.h"

namespace plamatrix::vulkan
{

    TEST(VulkanRuntime, EnumeratesComputeDevice)
    {
        if (!hasUsableVulkanDevice())
            GTEST_SKIP() << "No usable Vulkan compute device";
        const auto devices = enumerateVulkanDevices();
        ASSERT_FALSE(devices.empty());
        EXPECT_FALSE(selectedVulkanDeviceName().empty());
        EXPECT_TRUE(devices.front().hasComputeQueue);
    }

    TEST(VulkanSolver, MatchesCpuPcg)
    {
        if (!hasUsableVulkanDevice())
            GTEST_SKIP() << "No usable Vulkan compute device";
        CSRMatrix<float, Device::CPU> matrix(3, 3, 7);
        const Index row_offsets[] = {0, 2, 5, 7};
        const Index columns[] = {0, 1, 0, 1, 2, 1, 2};
        const float values[] = {4.0f, 1.0f, 1.0f, 3.0f, 1.0f, 1.0f, 2.0f};
        std::copy(row_offsets, row_offsets + 4, matrix.rowOffsets());
        std::copy(columns, columns + 7, matrix.colIndices());
        std::copy(values, values + 7, matrix.values());
        DenseMatrix<float, Device::CPU> rhs(3, 1);
        rhs(0, 0) = 1.0f;
        rhs(1, 0) = 2.0f;
        rhs(2, 0) = 3.0f;
        DenseMatrix<float, Device::CPU> expected(3, 1);
        DenseMatrix<float, Device::CPU> actual(3, 1);
        expected.fill(0.0f);
        actual.fill(0.0f);
        IterativeSolverOptions options;
        options.maxIterations = 100;
        options.relativeTolerance = 1.0e-5;
        options.requireConvergence = true;
        const auto cpu_report = plamatrix::pcg(matrix, rhs, expected, options);
        const auto vulkan_report = pcg(matrix, rhs, actual, options);
        EXPECT_TRUE(cpu_report.converged);
        EXPECT_TRUE(vulkan_report.converged);
        EXPECT_EQ(cpu_report.iterations, vulkan_report.iterations);
        EXPECT_GT(vulkan_report.commandSubmissions, 0u);
        for (Index row = 0; row < 3; ++row)
            EXPECT_NEAR(expected(row, 0), actual(row, 0), 2.0e-4f);
    }

    TEST(VulkanSolver, SupportsBatchedConvergenceChecks)
    {
        if (!hasUsableVulkanDevice())
            GTEST_SKIP() << "No usable Vulkan compute device";
        CSRMatrix<float, Device::CPU> matrix(2, 2, 4);
        const Index row_offsets[] = {0, 2, 4};
        const Index columns[] = {0, 1, 0, 1};
        const float values[] = {4.0f, 1.0f, 1.0f, 3.0f};
        std::copy(row_offsets, row_offsets + 3, matrix.rowOffsets());
        std::copy(columns, columns + 4, matrix.colIndices());
        std::copy(values, values + 4, matrix.values());
        DenseMatrix<float, Device::CPU> rhs(2, 1);
        rhs(0, 0) = 1.0f;
        rhs(1, 0) = 2.0f;
        DenseMatrix<float, Device::CPU> expected(2, 1);
        DenseMatrix<float, Device::CPU> actual(2, 1);
        expected.fill(0.0f);
        actual.fill(0.0f);
        IterativeSolverOptions options;
        options.maxIterations = 100;
        options.relativeTolerance = 1.0e-5;
        options.requireConvergence = true;
        options.convergenceCheckInterval = 2;
        const auto cpu_report = plamatrix::pcg(matrix, rhs, expected, options);
        const auto vulkan_report = pcg(matrix, rhs, actual, options);
        EXPECT_TRUE(cpu_report.converged);
        EXPECT_TRUE(vulkan_report.converged);
        EXPECT_GT(vulkan_report.commandSubmissions, 0u);
        for (Index row = 0; row < 2; ++row)
            EXPECT_NEAR(expected(row, 0), actual(row, 0), 2.0e-4f);
    }

    TEST(VulkanExecution, ReusesCommandContextAcrossBatches)
    {
        if (!hasUsableVulkanDevice())
            GTEST_SKIP() << "No usable Vulkan compute device";
        Runtime& runtime = Runtime::instance();
        CommandContext context(runtime);
        context.begin();
        EXPECT_EQ(context.pendingDispatchCount(), 0u);
        context.submitAndWait();
        context.begin();
        EXPECT_EQ(context.pendingDispatchCount(), 0u);
        context.submitAndWait();
    }

    TEST(VulkanExecution, CopiesBetweenHostAndDeviceLocalBuffers)
    {
        if (!hasUsableVulkanDevice())
            GTEST_SKIP() << "No usable Vulkan compute device";
        Runtime& runtime = Runtime::instance();
        Buffer source(runtime, sizeof(float), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, BufferMemory::HostVisible);
        Buffer device(runtime, sizeof(float), VK_BUFFER_USAGE_TRANSFER_DST_BIT, BufferMemory::DeviceLocal);
        Buffer result(runtime, sizeof(float), VK_BUFFER_USAGE_TRANSFER_DST_BIT, BufferMemory::HostVisible);
        *static_cast<float*>(source.map()) = 42.5f;
        source.unmap();
        CommandContext context(runtime);
        context.begin();
        context.copy(source, device, sizeof(float));
        context.submitAndWait();
        context.begin();
        context.copy(device, result, sizeof(float));
        context.submitAndWait();
        EXPECT_FLOAT_EQ(*static_cast<const float*>(result.map()), 42.5f);
        result.unmap();
    }

} // namespace plamatrix::vulkan
