#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "plamatrix/internal/optimization/block_schur.h"
#include "plamatrix/internal/vulkan/runtime.h"

namespace plamatrix::internal
{
    namespace
    {

        constexpr Index kPrimarySize = 9;
        constexpr Index kEliminatedSize = 3;

        BlockNormalEquations<float> makeBaEquations()
        {
            constexpr Index primary_count = 3;
            constexpr Index eliminated_count = 4;
            BlockNormalEquations<float> equations(primary_count, eliminated_count, kPrimarySize, kEliminatedSize);
            const std::array<std::array<Index, 3>, eliminated_count> cameras{{
                {{0, 1, -1}},
                {{1, 2, -1}},
                {{0, 2, -1}},
                {{0, 1, 2}},
            }};
            for (Index point = 0; point < eliminated_count; ++point)
            {
                for (Index observation = 0; observation < 3; ++observation)
                {
                    const Index camera =
                        cameras[static_cast<std::size_t>(point)][static_cast<std::size_t>(observation)];
                    if (camera < 0)
                        continue;
                    std::array<float, 2 * kPrimarySize> primary_jacobian{};
                    std::array<float, 2 * kEliminatedSize> eliminated_jacobian{};
                    for (Index row = 0; row < 2; ++row)
                    {
                        for (Index column = 0; column < kPrimarySize; ++column)
                        {
                            const int pattern = static_cast<int>((camera * 11 + point * 7 + row * 5 + column) % 13) - 6;
                            primary_jacobian[static_cast<std::size_t>(row * kPrimarySize + column)] =
                                static_cast<float>(pattern) * 0.025f;
                        }
                        primary_jacobian[static_cast<std::size_t>(row * kPrimarySize + row)] += 0.75f;
                        for (Index column = 0; column < kEliminatedSize; ++column)
                        {
                            const int pattern = static_cast<int>((point * 5 + camera * 3 + row + column) % 7) - 3;
                            eliminated_jacobian[static_cast<std::size_t>(row * kEliminatedSize + column)] =
                                static_cast<float>(pattern) * 0.04f;
                        }
                        eliminated_jacobian[static_cast<std::size_t>(row * kEliminatedSize + row)] += 0.6f;
                    }
                    const std::array<float, 2> residual{{
                        0.1f * static_cast<float>(1 + point),
                        -0.07f * static_cast<float>(1 + camera),
                    }};
                    equations.addResidualBlock(
                        camera, point, primary_jacobian.data(), eliminated_jacobian.data(), residual.data(), 2);
                }
            }

            std::array<float, kPrimarySize * kPrimarySize> primary_prior{};
            for (Index diagonal = 0; diagonal < kPrimarySize; ++diagonal)
                primary_prior[static_cast<std::size_t>(diagonal * kPrimarySize + diagonal)] = 0.2f;
            std::array<float, kPrimarySize> primary_residual{};
            for (Index camera = 0; camera < primary_count; ++camera)
            {
                primary_residual[static_cast<std::size_t>(camera)] = 0.03f * static_cast<float>(camera + 1);
                equations.addPrimaryResidualBlock(camera, primary_prior.data(), primary_residual.data(), kPrimarySize);
            }

            std::array<float, kEliminatedSize * kEliminatedSize> eliminated_prior{};
            for (Index diagonal = 0; diagonal < kEliminatedSize; ++diagonal)
                eliminated_prior[static_cast<std::size_t>(diagonal * kEliminatedSize + diagonal)] = 0.15f;
            const std::array<float, kEliminatedSize> eliminated_residual{{0.02f, -0.01f, 0.03f}};
            for (Index point = 0; point < eliminated_count; ++point)
            {
                equations.addEliminatedResidualBlock(
                    point, eliminated_prior.data(), eliminated_residual.data(), kEliminatedSize);
            }
            return equations;
        }

        TEST(VulkanBlockSchur, CooperativeAssemblyMatchesDenseCpuAndReusesRecording)
        {
            if (!vulkan::hasUsableVulkanDevice())
                GTEST_SKIP() << "No usable Vulkan compute device";
            if (!vulkan::selectedVulkanCooperativeMatrixCapabilities().enabled)
                GTEST_SKIP() << "Required cooperative-matrix configuration is unavailable";
            const auto equations = makeBaEquations();
            SchurComplementSolverOptions<float> reference_options;
            reference_options.linearBackend = SchurComplementLinearBackend::DenseCpu;
            std::vector<float> expected_primary;
            std::vector<float> expected_eliminated;
            const auto reference =
                solveDampedSchurComplement(equations, 0.1f, reference_options, &expected_primary, &expected_eliminated);
            ASSERT_TRUE(reference.converged) << reference.message;

            SchurComplementSolverOptions<float> options;
            options.linearBackend = SchurComplementLinearBackend::Vulkan;
            options.useMixedPrecision = true;
            options.maxIterations = 100;
            options.relativeTolerance = 2.0e-5f;
            options.absoluteTolerance = 1.0e-7f;
            SchurComplementSolverWorkspace<float> workspace;
            std::vector<float> actual_primary;
            std::vector<float> actual_eliminated;
            const auto first =
                solveDampedSchurComplement(equations, 0.1f, options, workspace, &actual_primary, &actual_eliminated);
            ASSERT_TRUE(first.converged) << first.message;
            EXPECT_TRUE(first.schurAssemblyOnDevice);
            EXPECT_TRUE(first.cooperativeMatrixUsed);
            EXPECT_TRUE(first.blockSpmvUsed);
            EXPECT_TRUE(first.deviceBlockJacobiUsed);
            EXPECT_TRUE(first.mixedPrecisionUsed);
            EXPECT_FALSE(first.deviceName.empty());
            EXPECT_GT(first.commandSubmissions, 0u);
            EXPECT_EQ(first.commandSubmissions, static_cast<std::uint32_t>((std::max(first.iterations, 1) + 7) / 8));
            EXPECT_GT(first.commandBufferRecordings, 0u);
            ASSERT_EQ(actual_primary.size(), expected_primary.size());
            ASSERT_EQ(actual_eliminated.size(), expected_eliminated.size());
            for (std::size_t index = 0; index < actual_primary.size(); ++index)
                EXPECT_NEAR(actual_primary[index], expected_primary[index], 3.0e-3f);
            for (std::size_t index = 0; index < actual_eliminated.size(); ++index)
                EXPECT_NEAR(actual_eliminated[index], expected_eliminated[index], 3.0e-3f);

            const auto second =
                solveDampedSchurComplement(equations, 0.1f, options, workspace, &actual_primary, &actual_eliminated);
            ASSERT_TRUE(second.converged) << second.message;
            EXPECT_TRUE(second.schurPatternReused);
            EXPECT_EQ(second.commandSubmissions, static_cast<std::uint32_t>((std::max(second.iterations, 1) + 7) / 8));
            EXPECT_EQ(second.commandBufferRecordings, 1u);

            const auto third =
                solveDampedSchurComplement(equations, 0.1f, options, workspace, &actual_primary, &actual_eliminated);
            ASSERT_TRUE(third.converged) << third.message;
            EXPECT_TRUE(third.schurPatternReused);
            EXPECT_EQ(third.commandSubmissions, static_cast<std::uint32_t>((std::max(third.iterations, 1) + 7) / 8));
            EXPECT_EQ(third.commandBufferRecordings, 0u);
        }

        TEST(VulkanBlockSchur, RequiresExplicitFloatMixedPrecisionOptIn)
        {
            const auto equations = makeBaEquations();
            SchurComplementSolverOptions<float> options;
            options.linearBackend = SchurComplementLinearBackend::Vulkan;
            std::vector<float> primary;
            std::vector<float> eliminated;
            EXPECT_THROW(solveDampedSchurComplement(equations, 0.1f, options, &primary, &eliminated),
                         std::invalid_argument);
        }

        TEST(VulkanBlockSchur, DoubleEquationsUseGuardedFloatSolve)
        {
            if (!vulkan::hasUsableVulkanDevice())
                GTEST_SKIP() << "No usable Vulkan compute device";

            BlockNormalEquations<double> equations(1, 1, 1, 1);
            const double primary_jacobian = 2.0;
            const double eliminated_jacobian = 0.5;
            const double residual = 1.0;
            equations.addResidualBlock(0, 0, &primary_jacobian, &eliminated_jacobian, &residual, 1);
            const double prior_jacobian = 1.0;
            const double prior_residual = 0.25;
            equations.addPrimaryResidualBlock(0, &prior_jacobian, &prior_residual, 1);
            equations.addEliminatedResidualBlock(0, &prior_jacobian, &prior_residual, 1);

            SchurComplementSolverOptions<double> reference_options;
            reference_options.linearBackend = SchurComplementLinearBackend::DenseCpu;
            std::vector<double> expected_primary;
            std::vector<double> expected_eliminated;
            const auto reference =
                solveDampedSchurComplement(equations, 0.1, reference_options, &expected_primary, &expected_eliminated);
            ASSERT_TRUE(reference.converged) << reference.message;

            SchurComplementSolverOptions<double> options;
            options.linearBackend = SchurComplementLinearBackend::Vulkan;
            options.useMixedPrecision = true;
            options.maxIterations = 100;
            std::vector<double> actual_primary;
            std::vector<double> actual_eliminated;
            const auto report =
                solveDampedSchurComplement(equations, 0.1, options, &actual_primary, &actual_eliminated);
            ASSERT_TRUE(report.converged) << report.message;
            EXPECT_TRUE(report.mixedPrecisionUsed);
            EXPECT_FALSE(report.deviceName.empty());
            ASSERT_EQ(actual_primary.size(), expected_primary.size());
            ASSERT_EQ(actual_eliminated.size(), expected_eliminated.size());
            EXPECT_NEAR(actual_primary[0], expected_primary[0], 1.0e-4);
            EXPECT_NEAR(actual_eliminated[0], expected_eliminated[0], 1.0e-4);
        }

    } // namespace
} // namespace plamatrix::internal
