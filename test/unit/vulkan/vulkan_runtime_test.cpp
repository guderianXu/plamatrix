#include <algorithm>
#include <cmath>

#include <gtest/gtest.h>

#include "plamatrix/sparse/iterative_solver.h"
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
        for (Index row = 0; row < 3; ++row)
            EXPECT_NEAR(expected(row, 0), actual(row, 0), 2.0e-4f);
    }

} // namespace plamatrix::vulkan
