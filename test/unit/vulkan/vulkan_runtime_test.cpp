#include <algorithm>
#include <cmath>
#include <cstdint>

#include <gtest/gtest.h>

#include "plamatrix/sparse/iterative_solver.h"
#include "plamatrix/vulkan/execution.h"
#include "plamatrix/vulkan/iterative_solver.h"
#include "plamatrix/vulkan/runtime.h"

namespace plamatrix::vulkan
{

    TEST(VulkanRuntime, LoadsPcgStatePipelines)
    {
        if (!hasUsableVulkanDevice())
            GTEST_SKIP() << "No usable Vulkan compute device";
        Runtime& runtime = Runtime::instance();
        EXPECT_NO_THROW(ComputePipeline(runtime, "pcg_dot_state", 3, sizeof(std::uint32_t) * 4));
    }

    TEST(VulkanRuntime, EnumeratesComputeDevice)
    {
        if (!hasUsableVulkanDevice())
            GTEST_SKIP() << "No usable Vulkan compute device";
        const auto devices = enumerateVulkanDevices();
        ASSERT_FALSE(devices.empty());
        EXPECT_FALSE(selectedVulkanDeviceName().empty());
        EXPECT_TRUE(devices.front().hasComputeQueue);
    }

    TEST(VulkanRuntime, ReportsSubgroupArithmeticCapabilities)
    {
        if (!hasUsableVulkanDevice())
            GTEST_SKIP() << "No usable Vulkan compute device";
        const Runtime& runtime = Runtime::instance();
        if (runtime.supportsSubgroupArithmetic())
        {
            EXPECT_GT(runtime.subgroupSize(), 0u);
            EXPECT_EQ(128u % runtime.subgroupSize(), 0u);
        }
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
        EXPECT_LE(vulkan_report.commandSubmissions, 1u);
        for (Index row = 0; row < 2; ++row)
            EXPECT_NEAR(expected(row, 0), actual(row, 0), 2.0e-4f);
    }

    TEST(VulkanSolver, ReusesRecordedIterationBatchAcrossSubmissionsAndSolves)
    {
        if (!hasUsableVulkanDevice())
            GTEST_SKIP() << "No usable Vulkan compute device";
        constexpr Index size = 257;
        CSRMatrix<float, Device::CPU> matrix(size, size, 3 * size - 2);
        Index offset = 0;
        for (Index row = 0; row < size; ++row)
        {
            matrix.rowOffsets()[row] = offset;
            if (row > 0)
            {
                matrix.colIndices()[offset] = row - 1;
                matrix.values()[offset++] = -1.0f;
            }
            matrix.colIndices()[offset] = row;
            matrix.values()[offset++] = 4.0f;
            if (row + 1 < size)
            {
                matrix.colIndices()[offset] = row + 1;
                matrix.values()[offset++] = -1.0f;
            }
        }
        matrix.rowOffsets()[size] = offset;
        DenseMatrix<float, Device::CPU> rhs(size, 1);
        DenseMatrix<float, Device::CPU> first(size, 1);
        DenseMatrix<float, Device::CPU> second(size, 1);
        for (Index row = 0; row < size; ++row)
            rhs(row, 0) = 1.0f + static_cast<float>(row % 11);
        first.fill(0.0f);
        second.fill(0.0f);
        IterativeSolverOptions options;
        options.maxIterations = 100;
        options.relativeTolerance = 1.0e-5;
        options.requireConvergence = true;
        options.convergenceCheckInterval = 1;

        const auto firstReport = pcg(matrix, rhs, first, options);
        const auto secondReport = pcg(matrix, rhs, second, options);
        EXPECT_TRUE(firstReport.converged);
        EXPECT_TRUE(secondReport.converged);
        EXPECT_GT(firstReport.commandSubmissions, firstReport.commandBufferRecordings);
        EXPECT_EQ(secondReport.commandBufferRecordings, 1u);
        EXPECT_EQ(firstReport.iterations, secondReport.iterations);
        for (Index row = 0; row < size; ++row)
            EXPECT_NEAR(first(row, 0), second(row, 0), 1.0e-6f);
    }

    TEST(VulkanSolver, ResetsBatchedStateBetweenWarmSolves)
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
        DenseMatrix<float, Device::CPU> rhsFirst(2, 1);
        DenseMatrix<float, Device::CPU> rhsSecond(2, 1);
        rhsFirst(0, 0) = 1.0f;
        rhsFirst(1, 0) = 2.0f;
        rhsSecond(0, 0) = 2.0f;
        rhsSecond(1, 0) = 1.0f;
        DenseMatrix<float, Device::CPU> expectedFirst(2, 1);
        DenseMatrix<float, Device::CPU> expectedSecond(2, 1);
        DenseMatrix<float, Device::CPU> actualFirst(2, 1);
        DenseMatrix<float, Device::CPU> actualSecond(2, 1);
        expectedFirst.fill(0.0f);
        expectedSecond.fill(0.0f);
        actualFirst.fill(0.0f);
        actualSecond.fill(0.0f);
        IterativeSolverOptions options;
        options.maxIterations = 16;
        options.relativeTolerance = 1.0e-5;
        options.requireConvergence = true;
        options.convergenceCheckInterval = 4;
        const auto expectedFirstReport = plamatrix::pcg(matrix, rhsFirst, expectedFirst, options);
        const auto expectedSecondReport = plamatrix::pcg(matrix, rhsSecond, expectedSecond, options);
        const auto actualFirstReport = pcg(matrix, rhsFirst, actualFirst, options);
        const auto actualSecondReport = pcg(matrix, rhsSecond, actualSecond, options);
        EXPECT_TRUE(expectedFirstReport.converged);
        EXPECT_TRUE(expectedSecondReport.converged);
        EXPECT_TRUE(actualFirstReport.converged);
        EXPECT_TRUE(actualSecondReport.converged);
        EXPECT_LE(actualFirstReport.commandSubmissions, 8u);
        EXPECT_LE(actualSecondReport.commandSubmissions, 8u);
        for (Index row = 0; row < 2; ++row)
        {
            EXPECT_NEAR(expectedFirst(row, 0), actualFirst(row, 0), 2.0e-4f);
            EXPECT_NEAR(expectedSecond(row, 0), actualSecond(row, 0), 2.0e-4f);
        }
    }

    TEST(VulkanSolver, StopsAtMaxIterationsInsideBatch)
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
        DenseMatrix<float, Device::CPU> solution(2, 1);
        rhs(0, 0) = 1.0f;
        rhs(1, 0) = 2.0f;
        solution.fill(0.0f);
        IterativeSolverOptions options;
        options.maxIterations = 3;
        options.relativeTolerance = 0.0;
        options.absoluteTolerance = 0.0;
        options.convergenceCheckInterval = 8;
        const auto report = pcg(matrix, rhs, solution, options);
        EXPECT_FALSE(report.converged);
        EXPECT_EQ(report.iterations, 3);
        EXPECT_LE(report.commandSubmissions, 5u);
        EXPECT_TRUE(std::isfinite(solution(0, 0)));
        EXPECT_TRUE(std::isfinite(solution(1, 0)));
    }

    TEST(VulkanSolver, HandlesExactConvergenceAtBatchBoundary)
    {
        if (!hasUsableVulkanDevice())
            GTEST_SKIP() << "No usable Vulkan compute device";
        CSRMatrix<float, Device::CPU> matrix(1, 1, 1);
        matrix.rowOffsets()[0] = 0;
        matrix.rowOffsets()[1] = 1;
        matrix.colIndices()[0] = 0;
        matrix.values()[0] = 4.0f;
        DenseMatrix<float, Device::CPU> rhs(1, 1);
        DenseMatrix<float, Device::CPU> solution(1, 1);
        rhs(0, 0) = 4.0f;
        solution(0, 0) = 0.0f;
        IterativeSolverOptions options;
        options.maxIterations = 8;
        options.relativeTolerance = 1.0e-6;
        options.requireConvergence = true;
        options.convergenceCheckInterval = 4;
        const auto report = pcg(matrix, rhs, solution, options);
        EXPECT_TRUE(report.converged);
        EXPECT_EQ(report.iterations, 1);
        EXPECT_NEAR(solution(0, 0), 1.0f, 1.0e-5f);
    }

    TEST(VulkanSolver, ReportsInitiallyConvergedStateWithoutIteration)
    {
        if (!hasUsableVulkanDevice())
            GTEST_SKIP() << "No usable Vulkan compute device";
        CSRMatrix<float, Device::CPU> matrix(1, 1, 1);
        matrix.rowOffsets()[0] = 0;
        matrix.rowOffsets()[1] = 1;
        matrix.colIndices()[0] = 0;
        matrix.values()[0] = 4.0f;
        DenseMatrix<float, Device::CPU> rhs(1, 1);
        DenseMatrix<float, Device::CPU> solution(1, 1);
        rhs(0, 0) = 4.0f;
        solution(0, 0) = 1.0f;
        IterativeSolverOptions options;
        options.maxIterations = 8;
        options.relativeTolerance = 1.0e-6;
        options.requireConvergence = true;
        options.convergenceCheckInterval = 4;
        const auto report = pcg(matrix, rhs, solution, options);
        EXPECT_TRUE(report.converged);
        EXPECT_EQ(report.iterations, 0);
        EXPECT_EQ(report.initialResidual, 0.0);
        EXPECT_EQ(report.finalResidual, 0.0);
        EXPECT_LE(report.commandSubmissions, 1u);
        EXPECT_NEAR(solution(0, 0), 1.0f, 1.0e-6f);
    }

    TEST(VulkanSolver, ReportsInitialResidualWhenIterationsAreDisabled)
    {
        if (!hasUsableVulkanDevice())
            GTEST_SKIP() << "No usable Vulkan compute device";
        CSRMatrix<float, Device::CPU> matrix(1, 1, 1);
        matrix.rowOffsets()[0] = 0;
        matrix.rowOffsets()[1] = 1;
        matrix.colIndices()[0] = 0;
        matrix.values()[0] = 4.0f;
        DenseMatrix<float, Device::CPU> rhs(1, 1);
        DenseMatrix<float, Device::CPU> solution(1, 1);
        rhs(0, 0) = 4.0f;
        solution(0, 0) = 0.0f;
        IterativeSolverOptions options;
        options.maxIterations = 0;
        options.relativeTolerance = 1.0e-6;
        const auto report = pcg(matrix, rhs, solution, options);
        EXPECT_FALSE(report.converged);
        EXPECT_EQ(report.iterations, 0);
        EXPECT_NEAR(report.initialResidual, 4.0, 1.0e-6);
        EXPECT_EQ(report.finalResidual, report.initialResidual);
        EXPECT_LE(report.commandSubmissions, 1u);
        EXPECT_EQ(solution(0, 0), 0.0f);
    }

    TEST(VulkanSolver, KeepsSeparateSolutionReadbackForLargeVectors)
    {
        if (!hasUsableVulkanDevice())
            GTEST_SKIP() << "No usable Vulkan compute device";
        constexpr Index size = 16385;
        CSRMatrix<float, Device::CPU> matrix(size, size, size);
        DenseMatrix<float, Device::CPU> rhs(size, 1);
        DenseMatrix<float, Device::CPU> solution(size, 1);
        for (Index row = 0; row < size; ++row)
        {
            matrix.rowOffsets()[row] = row;
            matrix.colIndices()[row] = row;
            matrix.values()[row] = 2.0f;
            rhs(row, 0) = 1.0f;
            solution(row, 0) = 0.0f;
        }
        matrix.rowOffsets()[size] = size;
        IterativeSolverOptions options;
        options.maxIterations = 4;
        options.relativeTolerance = 1.0e-6;
        options.requireConvergence = true;
        options.convergenceCheckInterval = 4;
        const auto report = pcg(matrix, rhs, solution, options);
        EXPECT_TRUE(report.converged);
        EXPECT_EQ(report.iterations, 1);
        EXPECT_EQ(report.commandSubmissions, 2u);
        EXPECT_NEAR(solution(0, 0), 0.5f, 1.0e-5f);
        EXPECT_NEAR(solution(size - 1, 0), 0.5f, 1.0e-5f);
    }

    TEST(VulkanSolver, ReportsDeviceScalarBreakdown)
    {
        if (!hasUsableVulkanDevice())
            GTEST_SKIP() << "No usable Vulkan compute device";
        CSRMatrix<float, Device::CPU> matrix(1, 1, 1);
        matrix.rowOffsets()[0] = 0;
        matrix.rowOffsets()[1] = 1;
        matrix.colIndices()[0] = 0;
        matrix.values()[0] = -1.0f;
        DenseMatrix<float, Device::CPU> rhs(1, 1);
        DenseMatrix<float, Device::CPU> solution(1, 1);
        rhs(0, 0) = 1.0f;
        solution(0, 0) = 0.0f;
        IterativeSolverOptions options;
        options.maxIterations = 4;
        options.relativeTolerance = 0.0;
        options.absoluteTolerance = 0.0;
        options.useJacobiPreconditioner = false;
        EXPECT_THROW(pcg(matrix, rhs, solution, options), std::runtime_error);
    }

    TEST(VulkanSolver, HandlesReductionTileBoundary)
    {
        if (!hasUsableVulkanDevice())
            GTEST_SKIP() << "No usable Vulkan compute device";
        constexpr Index size = 257;
        CSRMatrix<float, Device::CPU> matrix(size, size, 3 * size - 2);
        Index offset = 0;
        for (Index row = 0; row < size; ++row)
        {
            matrix.rowOffsets()[row] = offset;
            if (row > 0)
            {
                matrix.colIndices()[offset] = row - 1;
                matrix.values()[offset++] = 1.0f;
            }
            matrix.colIndices()[offset] = row;
            matrix.values()[offset++] = 4.0f;
            if (row + 1 < size)
            {
                matrix.colIndices()[offset] = row + 1;
                matrix.values()[offset++] = 1.0f;
            }
        }
        matrix.rowOffsets()[size] = offset;
        DenseMatrix<float, Device::CPU> rhs(size, 1);
        DenseMatrix<float, Device::CPU> expected(size, 1);
        DenseMatrix<float, Device::CPU> actual(size, 1);
        rhs.fill(1.0f);
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
        for (Index row = 0; row < size; ++row)
            EXPECT_NEAR(expected(row, 0), actual(row, 0), 2.0e-4f);
    }

    TEST(VulkanSolver, UsesSubgroupSpmvForLongRows)
    {
        if (!hasUsableVulkanDevice())
            GTEST_SKIP() << "No usable Vulkan compute device";
        if (!Runtime::instance().supportsSubgroupArithmetic())
            GTEST_SKIP() << "Selected Vulkan device has no compute subgroup arithmetic";
        constexpr Index size = 65;
        CSRMatrix<float, Device::CPU> matrix(size, size, size * size);
        for (Index row = 0; row < size; ++row)
        {
            matrix.rowOffsets()[row] = row * size;
            for (Index column = 0; column < size; ++column)
            {
                const Index entry = row * size + column;
                matrix.colIndices()[entry] = column;
                matrix.values()[entry] = row == column ? 66.0f : 1.0f;
            }
        }
        matrix.rowOffsets()[size] = size * size;
        DenseMatrix<float, Device::CPU> rhs(size, 1);
        DenseMatrix<float, Device::CPU> expected(size, 1);
        DenseMatrix<float, Device::CPU> actual(size, 1);
        for (Index row = 0; row < size; ++row)
            rhs(row, 0) = 1.0f + static_cast<float>(row % 7);
        expected.fill(0.0f);
        actual.fill(0.0f);
        IterativeSolverOptions options;
        options.maxIterations = 32;
        options.relativeTolerance = 1.0e-5;
        options.requireConvergence = true;
        const auto cpuReport = plamatrix::pcg(matrix, rhs, expected, options);
        const auto vulkanReport = pcg(matrix, rhs, actual, options);
        EXPECT_TRUE(cpuReport.converged);
        EXPECT_TRUE(vulkanReport.converged);
        EXPECT_TRUE(vulkanReport.subgroupSpmv);
        for (Index row = 0; row < size; ++row)
            EXPECT_NEAR(expected(row, 0), actual(row, 0), 2.0e-4f);
    }

    TEST(VulkanExecution, ReportsGpuTimestampForSubmission)
    {
        if (!hasUsableVulkanDevice())
            GTEST_SKIP() << "No usable Vulkan compute device";
        Runtime& runtime = Runtime::instance();
        ComputePipeline pipeline(runtime, "jacobi", 3, sizeof(std::uint32_t));
        Buffer residual(runtime, sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        Buffer inverse(runtime, sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        Buffer transformed(runtime, sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        const std::uint32_t count = 1;
        CommandContext context(runtime);
        context.begin();
        context.dispatch(pipeline, {&residual, &inverse, &transformed}, 1, &count, sizeof(count));
        context.submitAndWait();
        EXPECT_GE(context.gpuMilliseconds(), 0.0);
        EXPECT_GT(context.submissionCount(), 0u);
    }

    TEST(VulkanExecution, ResubmitsReusableCommandBufferWithoutRecordingAgain)
    {
        if (!hasUsableVulkanDevice())
            GTEST_SKIP() << "No usable Vulkan compute device";
        Runtime& runtime = Runtime::instance();
        ComputePipeline pipeline(runtime, "jacobi", 3, sizeof(std::uint32_t));
        Buffer residual(runtime, sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        Buffer inverse(runtime, sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        Buffer transformed(runtime, sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        *static_cast<float*>(residual.map()) = 2.0f;
        residual.unmap();
        *static_cast<float*>(inverse.map()) = 3.0f;
        inverse.unmap();
        const std::uint32_t count = 1;
        CommandContext context(runtime);
        context.beginReusable();
        context.dispatch(pipeline, {&residual, &inverse, &transformed}, 1, &count, sizeof(count));
        context.submitAndWait();
        context.submitAndWait();
        EXPECT_EQ(context.commandBufferRecordings(), 1u);
        EXPECT_EQ(context.submissionCount(), 2u);
        EXPECT_FLOAT_EQ(*static_cast<const float*>(transformed.map()), 6.0f);
        transformed.unmap();
    }

    TEST(VulkanExecution, ReusesDescriptorSetsAcrossBatches)
    {
        if (!hasUsableVulkanDevice())
            GTEST_SKIP() << "No usable Vulkan compute device";
        Runtime& runtime = Runtime::instance();
        ComputePipeline pipeline(runtime, "jacobi", 3, sizeof(std::uint32_t));
        Buffer residual(runtime, sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        Buffer inverse(runtime, sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        Buffer transformed(runtime, sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        const std::uint32_t count = 1;
        CommandContext context(runtime);
        context.begin();
        context.dispatch(pipeline, {&residual, &inverse, &transformed}, 1, &count, sizeof(count));
        context.submitAndWait();
        EXPECT_EQ(context.descriptorSetAllocations(), 1u);
        context.begin();
        context.dispatch(pipeline, {&residual, &inverse, &transformed}, 1, &count, sizeof(count));
        context.submitAndWait();
        EXPECT_EQ(context.descriptorSetAllocations(), 1u);
    }

    TEST(VulkanExecution, CopiesBetweenHostAndDeviceLocalBuffers)
    {
        if (!hasUsableVulkanDevice())
            GTEST_SKIP() << "No usable Vulkan compute device";
        Runtime& runtime = Runtime::instance();
        Buffer source(runtime, sizeof(float), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, BufferMemory::HostVisible);
        Buffer device(runtime,
                      sizeof(float),
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      BufferMemory::DeviceLocal);
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
