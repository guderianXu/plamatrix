#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include <omp.h>

#include "plamatrix/internal/optimization/block_schur.h"
#include "plamatrix/internal/optimization/levenberg_marquardt.h"
#include "plamatrix/internal/optimization/robust_loss.h"
#include "plamatrix/internal/opencl/runtime.h"

#ifdef PLAMATRIX_WITH_CUDA
#include <cuda_runtime_api.h>
#endif

namespace plamatrix::internal
{
    namespace
    {

        TEST(RobustLossTest, HuberMatchesQuadraticAndLinearBranches)
        {
            const auto quadratic = evaluateHuberLoss(4.0, 3.0);
            EXPECT_DOUBLE_EQ(quadratic.cost, 2.0);
            EXPECT_DOUBLE_EQ(quadratic.weight, 1.0);

            const auto linear = evaluateHuberLoss(25.0, 3.0);
            EXPECT_DOUBLE_EQ(linear.cost, 10.5);
            EXPECT_DOUBLE_EQ(linear.weight, 0.6);
        }

        TEST(RobustLossTest, HuberRejectsInvalidArguments)
        {
            EXPECT_THROW(evaluateHuberLoss(-1.0, 3.0), std::invalid_argument);
            EXPECT_THROW(evaluateHuberLoss(1.0, -1.0), std::invalid_argument);
        }

        void addSyntheticResidual(BlockNormalEquations<double>& equations,
                                  Index primary,
                                  Index eliminated,
                                  const std::array<double, 2>& primary_jacobian,
                                  double eliminated_jacobian,
                                  double residual)
        {
            equations.addResidualBlock(
                primary, eliminated, primary_jacobian.data(), &eliminated_jacobian, &residual, 1, 1.0);
        }

        BlockNormalEquations<double> makeSyntheticEquations()
        {
            BlockNormalEquations<double> equations(2, 2, 2, 1);
            addSyntheticResidual(equations, 0, 0, {{1.0, 2.0}}, 0.5, -1.0);
            addSyntheticResidual(equations, 1, 0, {{-0.5, 1.0}}, 1.5, 0.25);
            addSyntheticResidual(equations, 0, 1, {{2.0, -1.0}}, -0.75, 2.0);
            addSyntheticResidual(equations, 1, 1, {{1.25, 0.5}}, 2.0, -0.5);

            const double primary_prior_jacobian[2] = {0.4, -0.2};
            const double primary_prior_residual = 0.75;
            equations.addPrimaryResidualBlock(0, primary_prior_jacobian, &primary_prior_residual, 1, 2.0);

            const double eliminated_prior_jacobian = 0.8;
            const double eliminated_prior_residual = -0.3;
            equations.addEliminatedResidualBlock(1, &eliminated_prior_jacobian, &eliminated_prior_residual, 1, 1.5);
            return equations;
        }

        void expectMultiPrimaryReference(SchurComplementLinearBackend backend)
        {
            BlockNormalEquations<double> equations(2, 1, 1, 1);
            const double primary_jacobian_0 = 1.0;
            const double primary_jacobian_1 = 2.0;
            const double eliminated_jacobian = 3.0;
            const double residual = 4.0;
            equations.addResidualBlocks(
                {0, 1}, {&primary_jacobian_0, &primary_jacobian_1}, 0, &eliminated_jacobian, &residual, 1);

            const double direct_jacobian_0 = 0.5;
            const double direct_jacobian_1 = -1.0;
            const double direct_residual = 0.2;
            equations.addPrimaryResidualBlocks({0, 1}, {&direct_jacobian_0, &direct_jacobian_1}, &direct_residual, 1);

            SchurComplementSolverOptions<double> options;
            options.linearBackend = backend;
            options.maxIterations = 50;
            options.relativeTolerance = 1e-13;
            options.absoluteTolerance = 1e-14;
            std::vector<double> primary_step;
            std::vector<double> eliminated_step;
            const auto report = solveDampedSchurComplement(equations, 0.1, options, &primary_step, &eliminated_step);

            ASSERT_TRUE(report.converged) << report.message;
            ASSERT_EQ(primary_step.size(), 2u);
            ASSERT_EQ(eliminated_step.size(), 1u);
            EXPECT_NEAR(primary_step[0], -1.345185185185185, 1e-11);
            EXPECT_NEAR(primary_step[1], -0.512592592592593, 1e-11);
            EXPECT_NEAR(eliminated_step[0], -0.493827160493827, 1e-11);
        }

        void expectSyntheticReference(SchurComplementLinearBackend backend)
        {
            const auto equations = makeSyntheticEquations();
            SchurComplementSolverOptions<double> options;
            options.linearBackend = backend;
            options.maxIterations = 50;
            options.relativeTolerance = 1e-13;
            options.absoluteTolerance = 1e-14;

            std::vector<double> primary_step;
            std::vector<double> eliminated_step;
            const auto report = solveDampedSchurComplement(equations, 0.1, options, &primary_step, &eliminated_step);

            ASSERT_TRUE(report.converged) << report.message;
            EXPECT_EQ(report.linearBackend, backend);
            if (backend == SchurComplementLinearBackend::Cuda || backend == SchurComplementLinearBackend::OpenCl)
            {
                EXPECT_FALSE(report.deviceName.empty());
                EXPECT_GT(report.linearSolveSeconds, 0.0);
                EXPECT_TRUE(report.schurAssemblyOnDevice);
            }
            const std::array<double, 6> expected{{
                -0.532196868252671,
                0.717080672008074,
                0.078633780070238,
                -0.210823022930113,
                0.017900956933130,
                0.253200998620430,
            }};
            for (std::size_t index = 0; index < primary_step.size(); ++index)
            {
                EXPECT_NEAR(primary_step[index], expected[index], 1e-10);
            }
            for (std::size_t index = 0; index < eliminated_step.size(); ++index)
            {
                EXPECT_NEAR(eliminated_step[index], expected[primary_step.size() + index], 1e-10);
            }
        }

        void expectAcceleratedPatternReuse(SchurComplementLinearBackend backend)
        {
            const auto equations = makeSyntheticEquations();
            SchurComplementSolverOptions<double> options;
            options.linearBackend = backend;
            options.maxIterations = 50;
            options.relativeTolerance = 1e-13;
            options.absoluteTolerance = 1e-14;
            SchurComplementSolverWorkspace<double> workspace;
            std::vector<double> primary_step;
            std::vector<double> eliminated_step;

            const auto first =
                solveDampedSchurComplement(equations, 0.1, options, workspace, &primary_step, &eliminated_step);
            ASSERT_TRUE(first.converged) << first.message;
            EXPECT_FALSE(first.schurPatternReused);
            EXPECT_EQ(workspace.patternBuildCount(), 1u);

            const auto second =
                solveDampedSchurComplement(equations, 0.2, options, workspace, &primary_step, &eliminated_step);
            ASSERT_TRUE(second.converged) << second.message;
            EXPECT_TRUE(second.schurPatternReused);
            EXPECT_EQ(workspace.patternBuildCount(), 1u);

            BlockNormalEquations<double> changed_equations(1, 1, 1, 1);
            const double jacobian = 1.0;
            const double residual = -0.5;
            changed_equations.addResidualBlock(0, 0, &jacobian, &jacobian, &residual, 1);
            changed_equations.addPrimaryResidualBlock(0, &jacobian, &residual, 1);
            const auto changed =
                solveDampedSchurComplement(changed_equations, 0.2, options, workspace, &primary_step, &eliminated_step);
            ASSERT_TRUE(changed.converged) << changed.message;
            EXPECT_FALSE(changed.schurPatternReused);
            EXPECT_EQ(workspace.patternBuildCount(), 2u);
        }

        template <typename Scalar> void expectSparseCpuEliminatesInactiveCoordinates(Scalar tolerance)
        {
            constexpr Index block_size = 3;
            BlockNormalEquations<Scalar> equations(2, 0, block_size, 1);
            const std::array<Scalar, 2 * block_size> first_jacobian{{
                Scalar(2),
                Scalar(0),
                Scalar(0),
                Scalar(0),
                Scalar(3),
                Scalar(0),
            }};
            const std::array<Scalar, 2> first_residual{{Scalar(1), Scalar(-2)}};
            equations.addPrimaryResidualBlock(0, first_jacobian.data(), first_residual.data(), 2);

            const std::array<Scalar, block_size> left_jacobian{{Scalar(0), Scalar(1), Scalar(0)}};
            const std::array<Scalar, block_size> right_jacobian{{Scalar(2), Scalar(0), Scalar(0)}};
            const Scalar coupled_residual = Scalar(0.5);
            equations.addPrimaryResidualBlocks(
                {0, 1}, {left_jacobian.data(), right_jacobian.data()}, &coupled_residual, 1);

            SchurComplementSolverOptions<Scalar> options;
            options.linearBackend = SchurComplementLinearBackend::SparseCpu;
            options.relativeTolerance = tolerance;
            options.absoluteTolerance = tolerance;
            SchurComplementSolverWorkspace<Scalar> workspace;
            std::vector<Scalar> primary_step;
            std::vector<Scalar> eliminated_step;
            const auto first =
                solveDampedSchurComplement(equations, Scalar(0), options, workspace, &primary_step, &eliminated_step);

            ASSERT_TRUE(first.converged) << first.message;
            ASSERT_EQ(primary_step.size(), 6u);
            EXPECT_TRUE(eliminated_step.empty());
            EXPECT_NEAR(primary_step[0], Scalar(-0.5), tolerance);
            EXPECT_NEAR(primary_step[1], Scalar(2.0 / 3.0), tolerance);
            EXPECT_EQ(primary_step[2], Scalar(0));
            EXPECT_NEAR(primary_step[3], Scalar(-7.0 / 12.0), tolerance);
            EXPECT_EQ(primary_step[4], Scalar(0));
            EXPECT_EQ(primary_step[5], Scalar(0));

            const auto second =
                solveDampedSchurComplement(equations, Scalar(0), options, workspace, &primary_step, &eliminated_step);
            ASSERT_TRUE(second.converged) << second.message;
            EXPECT_TRUE(second.schurPatternReused);
            EXPECT_TRUE(second.symbolicAnalysisReused);
        }

        TEST(BlockSchurTest, MatchesDenseDampedNormalEquation)
        {
            auto equations = makeSyntheticEquations();

            SchurComplementSolverOptions<double> options;
            options.maxIterations = 50;
            options.relativeTolerance = 1e-13;
            options.absoluteTolerance = 1e-14;

            std::vector<double> primary_step;
            std::vector<double> eliminated_step;
            const auto report = solveDampedSchurComplement(equations, 0.1, options, &primary_step, &eliminated_step);

            ASSERT_TRUE(report.converged) << report.message;
            ASSERT_EQ(primary_step.size(), 4u);
            ASSERT_EQ(eliminated_step.size(), 2u);

            // Reference solution from the same complete damped 6x6 normal equation.
            const std::array<double, 6> expected{{
                -0.532196868252671,
                0.717080672008074,
                0.078633780070238,
                -0.210823022930113,
                0.017900956933130,
                0.253200998620430,
            }};
            for (std::size_t index = 0; index < primary_step.size(); ++index)
            {
                EXPECT_NEAR(primary_step[index], expected[index], 1e-11);
            }
            for (std::size_t index = 0; index < eliminated_step.size(); ++index)
            {
                EXPECT_NEAR(eliminated_step[index], expected[primary_step.size() + index], 1e-11);
            }
        }

        TEST(BlockSchurTest, ClusterJacobiMatchesBlockJacobiAndReducesIterations)
        {
            const auto equations = makeSyntheticEquations();
            SchurComplementSolverOptions<double> block_options;
            block_options.maxIterations = 50;
            block_options.relativeTolerance = 1e-13;
            block_options.absoluteTolerance = 1e-14;
            block_options.preconditionerClusterSize = 1;
            std::vector<double> block_primary;
            std::vector<double> block_eliminated;
            const auto block_report =
                solveDampedSchurComplement(equations, 0.1, block_options, &block_primary, &block_eliminated);
            ASSERT_TRUE(block_report.converged) << block_report.message;

            auto cluster_options = block_options;
            cluster_options.preconditionerClusterSize = 2;
            std::vector<double> cluster_primary;
            std::vector<double> cluster_eliminated;
            const auto cluster_report =
                solveDampedSchurComplement(equations, 0.1, cluster_options, &cluster_primary, &cluster_eliminated);
            ASSERT_TRUE(cluster_report.converged) << cluster_report.message;
            EXPECT_EQ(cluster_report.preconditionerName, "cluster_jacobi_2");
            EXPECT_LE(cluster_report.iterations, block_report.iterations);
            ASSERT_EQ(cluster_primary.size(), block_primary.size());
            ASSERT_EQ(cluster_eliminated.size(), block_eliminated.size());
            for (std::size_t index = 0; index < block_primary.size(); ++index)
            {
                EXPECT_NEAR(cluster_primary[index], block_primary[index], 1e-10);
            }
            for (std::size_t index = 0; index < block_eliminated.size(); ++index)
            {
                EXPECT_NEAR(cluster_eliminated[index], block_eliminated[index], 1e-10);
            }
        }

        TEST(BlockNormalEquationsTest, ClearValuesRetainsReusableTopology)
        {
            auto equations = makeSyntheticEquations();
            equations.clearValues();
            addSyntheticResidual(equations, 0, 0, {{1.0, 0.0}}, 1.0, -0.5);

            SchurComplementSolverOptions<double> options;
            options.maxIterations = 20;
            std::vector<double> primary;
            std::vector<double> eliminated;
            const auto report = solveDampedSchurComplement(equations, 0.1, options, &primary, &eliminated);
            EXPECT_TRUE(report.converged) << report.message;
        }

        TEST(BlockNormalEquationsTest, PreReducedPrimaryBlocksMatchResidualAssembly)
        {
            BlockNormalEquations<double> residual_equations(2, 0, 2, 1);
            const std::array<double, 4> left_jacobian{{2.0, -1.0, 0.5, 3.0}};
            const std::array<double, 4> right_jacobian{{-0.25, 1.5, 2.0, 0.75}};
            const std::array<double, 2> residual{{0.4, -1.2}};
            residual_equations.addPrimaryResidualBlocks(
                {0, 1}, {left_jacobian.data(), right_jacobian.data()}, residual.data(), 2);

            std::array<double, 4> left_hessian{};
            std::array<double, 4> right_hessian{};
            std::array<double, 4> cross_hessian{};
            std::array<double, 2> left_gradient{};
            std::array<double, 2> right_gradient{};
            for (std::size_t observation = 0; observation < 2; ++observation)
            {
                for (std::size_t row = 0; row < 2; ++row)
                {
                    left_gradient[row] += left_jacobian[observation * 2 + row] * residual[observation];
                    right_gradient[row] += right_jacobian[observation * 2 + row] * residual[observation];
                    for (std::size_t column = 0; column < 2; ++column)
                    {
                        left_hessian[row * 2 + column] +=
                            left_jacobian[observation * 2 + row] * left_jacobian[observation * 2 + column];
                        right_hessian[row * 2 + column] +=
                            right_jacobian[observation * 2 + row] * right_jacobian[observation * 2 + column];
                        cross_hessian[row * 2 + column] +=
                            left_jacobian[observation * 2 + row] * right_jacobian[observation * 2 + column];
                    }
                }
            }

            BlockNormalEquations<double> reduced_equations(2, 0, 2, 1);
            reduced_equations.addPrimaryHessianBlock(0, 0, left_hessian.data());
            reduced_equations.addPrimaryHessianBlock(1, 1, right_hessian.data());
            reduced_equations.addPrimaryHessianBlock(0, 1, cross_hessian.data());
            reduced_equations.addPrimaryGradientBlock(0, left_gradient.data());
            reduced_equations.addPrimaryGradientBlock(1, right_gradient.data());

            SchurComplementSolverOptions<double> options;
            options.linearBackend = SchurComplementLinearBackend::DenseCpu;
            std::vector<double> residual_step;
            std::vector<double> residual_eliminated;
            std::vector<double> reduced_step;
            std::vector<double> reduced_eliminated;
            const auto residual_report =
                solveDampedSchurComplement(residual_equations, 0.1, options, &residual_step, &residual_eliminated);
            const auto reduced_report =
                solveDampedSchurComplement(reduced_equations, 0.1, options, &reduced_step, &reduced_eliminated);
            ASSERT_TRUE(residual_report.converged) << residual_report.message;
            ASSERT_TRUE(reduced_report.converged) << reduced_report.message;
            EXPECT_EQ(residual_step, reduced_step);
            EXPECT_TRUE(residual_eliminated.empty());
            EXPECT_TRUE(reduced_eliminated.empty());
        }

        TEST(BlockSchurTest, DenseCpuReportsEveryRequestedNumericalPhase)
        {
            const auto equations = makeSyntheticEquations();
            SchurComplementSolverOptions<double> options;
            options.linearBackend = SchurComplementLinearBackend::DenseCpu;
            options.relativeTolerance = 1e-12;
            options.absoluteTolerance = 1e-14;

            std::vector<double> primary_step;
            std::vector<double> eliminated_step;
            const auto report = solveDampedSchurComplement(equations, 0.1, options, &primary_step, &eliminated_step);

            ASSERT_TRUE(report.converged) << report.message;
            EXPECT_GE(report.smallBlockInverseSeconds, 0.0);
            EXPECT_GE(report.schurAccumulationSeconds, 0.0);
            EXPECT_GE(report.csrConversionSeconds, 0.0);
            EXPECT_GE(report.choleskyFactorizationSeconds, 0.0);
            EXPECT_GE(report.triangularSolveSeconds, 0.0);
            EXPECT_GE(report.residualCheckSeconds, 0.0);
            EXPECT_GE(report.backSubstitutionSeconds, 0.0);
            EXPECT_GE(report.schurAssemblySeconds, report.schurAccumulationSeconds + report.csrConversionSeconds);
            EXPECT_GE(report.linearSolveSeconds,
                      report.choleskyFactorizationSeconds + report.triangularSolveSeconds +
                          report.residualCheckSeconds);
        }

        TEST(BlockSchurTest, MultiPrimaryResidualMatchesDenseDampedNormalEquation)
        {
            expectMultiPrimaryReference(SchurComplementLinearBackend::Cpu);
        }

        TEST(BlockSchurTest, DenseCpuMatchesDenseDampedNormalEquation)
        {
            expectSyntheticReference(SchurComplementLinearBackend::DenseCpu);
            expectMultiPrimaryReference(SchurComplementLinearBackend::DenseCpu);
            expectAcceleratedPatternReuse(SchurComplementLinearBackend::DenseCpu);
        }

        TEST(BlockSchurTest, NativeSparseCpuSymbolicReuseIsExplicit)
        {
            EXPECT_TRUE(hasSparseDirectSchurSolver());
            const auto equations = makeSyntheticEquations();
            SchurComplementSolverOptions<double> options;
            options.linearBackend = SchurComplementLinearBackend::SparseCpu;
            options.relativeTolerance = 1e-12;
            options.absoluteTolerance = 1e-14;
            SchurComplementSolverWorkspace<double> workspace;
            std::vector<double> primary_step;
            std::vector<double> eliminated_step;

            const auto first =
                solveDampedSchurComplement(equations, 0.1, options, workspace, &primary_step, &eliminated_step);
            ASSERT_TRUE(first.converged) << first.message;
            EXPECT_EQ(first.linearBackend, SchurComplementLinearBackend::SparseCpu);
            EXPECT_EQ(first.preconditionerName, "none");
            EXPECT_FALSE(first.schurPatternReused);
            EXPECT_FALSE(first.symbolicAnalysisReused);
            EXPECT_GE(first.symbolicAnalysisSeconds, 0.0);
            EXPECT_GE(first.choleskyFactorizationSeconds, 0.0);
            EXPECT_GE(first.triangularSolveSeconds, 0.0);

            const auto second =
                solveDampedSchurComplement(equations, 0.2, options, workspace, &primary_step, &eliminated_step);
            ASSERT_TRUE(second.converged) << second.message;
            EXPECT_TRUE(second.schurPatternReused);
            EXPECT_TRUE(second.symbolicAnalysisReused);
            EXPECT_DOUBLE_EQ(second.symbolicAnalysisSeconds, 0.0);
            EXPECT_EQ(workspace.patternBuildCount(), 1u);

            workspace.clear();
            const auto after_clear =
                solveDampedSchurComplement(equations, 0.2, options, workspace, &primary_step, &eliminated_step);
            ASSERT_TRUE(after_clear.converged) << after_clear.message;
            EXPECT_FALSE(after_clear.symbolicAnalysisReused);
        }

        TEST(BlockSchurTest, CachedSchurTopologyInvalidatesWhenSameEquationsGrow)
        {
            BlockNormalEquations<double> equations(2, 1, 1, 1);
            const double jacobian = 1.0;
            const double residual = -0.5;
            equations.addResidualBlock(0, 0, &jacobian, &jacobian, &residual, 1);
            equations.addPrimaryResidualBlock(1, &jacobian, &residual, 1);

            SchurComplementSolverOptions<double> options;
            options.linearBackend = SchurComplementLinearBackend::SparseCpu;
            SchurComplementSolverWorkspace<double> workspace;
            std::vector<double> primary_step;
            std::vector<double> eliminated_step;
            const auto first =
                solveDampedSchurComplement(equations, 0.1, options, workspace, &primary_step, &eliminated_step);
            ASSERT_TRUE(first.converged) << first.message;
            EXPECT_FALSE(first.schurPatternReused);

            const auto reused =
                solveDampedSchurComplement(equations, 0.2, options, workspace, &primary_step, &eliminated_step);
            ASSERT_TRUE(reused.converged) << reused.message;
            EXPECT_TRUE(reused.schurPatternReused);

            equations.addResidualBlock(1, 0, &jacobian, &jacobian, &residual, 1);
            const auto rebuilt =
                solveDampedSchurComplement(equations, 0.2, options, workspace, &primary_step, &eliminated_step);
            ASSERT_TRUE(rebuilt.converged) << rebuilt.message;
            EXPECT_FALSE(rebuilt.schurPatternReused);
            EXPECT_EQ(workspace.patternBuildCount(), 2U);
        }

        TEST(BlockSchurTest, NativeSparseCpuMatchesDenseDampedNormalEquation)
        {
            expectSyntheticReference(SchurComplementLinearBackend::SparseCpu);
            expectMultiPrimaryReference(SchurComplementLinearBackend::SparseCpu);
        }

        TEST(BlockSchurTest, NativeSparseCpuSupportsFloatEquations)
        {
            constexpr Index block_size = 2;
            BlockNormalEquations<float> equations(2, 0, block_size, 1);
            const std::array<float, block_size * block_size> identity{{1.0F, 0.0F, 0.0F, 1.0F}};
            const std::array<float, block_size> first_residual{{0.2F, -0.1F}};
            const std::array<float, block_size> second_residual{{-0.3F, 0.4F}};
            equations.addPrimaryResidualBlock(0, identity.data(), first_residual.data(), block_size);
            equations.addPrimaryResidualBlock(1, identity.data(), second_residual.data(), block_size);
            const std::array<float, block_size> first_jacobian{{0.5F, -0.25F}};
            const std::array<float, block_size> second_jacobian{{-0.2F, 0.75F}};
            const float coupled_residual = 0.15F;
            equations.addPrimaryResidualBlocks(
                {0, 1}, {first_jacobian.data(), second_jacobian.data()}, &coupled_residual, 1);

            SchurComplementSolverOptions<float> options;
            options.relativeTolerance = 1e-5F;
            options.absoluteTolerance = 1e-7F;
            std::vector<float> dense_primary;
            std::vector<float> dense_eliminated;
            options.linearBackend = SchurComplementLinearBackend::DenseCpu;
            const auto dense = solveDampedSchurComplement(equations, 0.01F, options, &dense_primary, &dense_eliminated);
            ASSERT_TRUE(dense.converged) << dense.message;

            std::vector<float> sparse_primary;
            std::vector<float> sparse_eliminated;
            options.linearBackend = SchurComplementLinearBackend::SparseCpu;
            const auto sparse =
                solveDampedSchurComplement(equations, 0.01F, options, &sparse_primary, &sparse_eliminated);
            ASSERT_TRUE(sparse.converged) << sparse.message;
            ASSERT_EQ(sparse_primary.size(), dense_primary.size());
            EXPECT_TRUE(dense_eliminated.empty());
            EXPECT_TRUE(sparse_eliminated.empty());
            for (std::size_t index = 0; index < dense_primary.size(); ++index)
            {
                EXPECT_NEAR(sparse_primary[index], dense_primary[index], 1e-5F);
            }
        }

        TEST(BlockSchurTest, NativeSparseCpuEliminatesInactiveDoubleCoordinatesAtZeroDamping)
        {
            expectSparseCpuEliminatesInactiveCoordinates<double>(1e-12);
        }

        TEST(BlockSchurTest, NativeSparseCpuEliminatesInactiveFloatCoordinatesAtZeroDamping)
        {
            expectSparseCpuEliminatesInactiveCoordinates<float>(1e-5F);
        }

        TEST(BlockSchurTest, DenseAndNativeSparseCpuSolveBlockChainDeterministically)
        {
            constexpr Index block_count = 24;
            constexpr Index block_size = 6;
            BlockNormalEquations<double> equations(block_count, 0, block_size, 1);

            std::array<double, block_size * block_size> identity{};
            for (Index diagonal = 0; diagonal < block_size; ++diagonal)
            {
                identity[static_cast<std::size_t>(diagonal * block_size + diagonal)] = 1.0;
            }
            for (Index block = 0; block < block_count; ++block)
            {
                std::array<double, block_size> residual{};
                for (Index component = 0; component < block_size; ++component)
                {
                    residual[static_cast<std::size_t>(component)] = 0.01 * static_cast<double>(1 + block + component);
                }
                equations.addPrimaryResidualBlock(block, identity.data(), residual.data(), block_size);
            }

            const std::array<double, block_size> left_jacobian{{1.0, 0.2, -0.1, 0.3, 0.05, -0.2}};
            const std::array<double, block_size> right_jacobian{{-0.8, 0.1, 0.25, -0.2, 0.15, 0.3}};
            for (Index block = 0; block + 1 < block_count; ++block)
            {
                const double residual = 0.02 * static_cast<double>(block + 1);
                equations.addPrimaryResidualBlocks(
                    {block, block + 1}, {left_jacobian.data(), right_jacobian.data()}, &residual, 1);
            }

            SchurComplementSolverOptions<double> options;
            options.linearBackend = SchurComplementLinearBackend::DenseCpu;
            options.relativeTolerance = 1e-12;
            options.absoluteTolerance = 1e-14;
            std::vector<double> first_primary;
            std::vector<double> first_eliminated;
            std::vector<double> second_primary;
            std::vector<double> second_eliminated;
            const auto first = solveDampedSchurComplement(equations, 0.01, options, &first_primary, &first_eliminated);
            const auto second =
                solveDampedSchurComplement(equations, 0.01, options, &second_primary, &second_eliminated);

            ASSERT_TRUE(first.converged) << first.message;
            ASSERT_TRUE(second.converged) << second.message;
            ASSERT_EQ(first_primary.size(), static_cast<std::size_t>(block_count * block_size));
            EXPECT_TRUE(first_eliminated.empty());
            EXPECT_EQ(first_primary, second_primary);
            EXPECT_EQ(first_eliminated, second_eliminated);
            for (double value : first_primary)
            {
                EXPECT_TRUE(std::isfinite(value));
            }

            options.linearBackend = SchurComplementLinearBackend::SparseCpu;
            std::vector<double> sparse_primary;
            std::vector<double> sparse_eliminated;
            const auto sparse =
                solveDampedSchurComplement(equations, 0.01, options, &sparse_primary, &sparse_eliminated);
            ASSERT_TRUE(sparse.converged) << sparse.message;
            ASSERT_EQ(sparse_primary.size(), first_primary.size());
            EXPECT_TRUE(sparse_eliminated.empty());
            for (std::size_t index = 0; index < first_primary.size(); ++index)
            {
                EXPECT_NEAR(sparse_primary[index], first_primary[index], 1e-10);
            }
        }

        TEST(BlockSchurTest, DenseCpuEliminatedAssemblyIsBitwiseStableAcrossThreadCounts)
        {
            constexpr Index primary_count = 36;
            constexpr Index eliminated_count = 240;
            constexpr Index primary_size = 9;
            constexpr Index eliminated_size = 3;
            BlockNormalEquations<double> equations(primary_count, eliminated_count, primary_size, eliminated_size);
            std::array<double, primary_size * primary_size> identity{};
            for (Index diagonal = 0; diagonal < primary_size; ++diagonal)
            {
                identity[static_cast<std::size_t>(diagonal * primary_size + diagonal)] = 1.0;
            }
            std::array<double, primary_size> zero{};
            for (Index primary = 0; primary < primary_count; ++primary)
            {
                equations.addPrimaryResidualBlock(primary, identity.data(), zero.data(), primary_size, 0.25);
            }
            for (Index eliminated = 0; eliminated < eliminated_count; ++eliminated)
            {
                for (Index observation = 0; observation < 4; ++observation)
                {
                    const Index primary = (eliminated * 7 + observation * 5) % primary_count;
                    std::array<double, 2 * primary_size> primary_jacobian{};
                    std::array<double, 2 * eliminated_size> eliminated_jacobian{};
                    for (Index index = 0; index < 2 * primary_size; ++index)
                    {
                        primary_jacobian[static_cast<std::size_t>(index)] =
                            0.01 * static_cast<double>(1 + (index + eliminated + observation) % 13);
                    }
                    for (Index index = 0; index < 2 * eliminated_size; ++index)
                    {
                        eliminated_jacobian[static_cast<std::size_t>(index)] =
                            0.02 * static_cast<double>(1 + (index + eliminated + 2 * observation) % 7);
                    }
                    const std::array<double, 2> residual{
                        {0.001 * static_cast<double>(eliminated + 1), -0.002 * static_cast<double>(observation + 1)}};
                    equations.addResidualBlock(
                        primary, eliminated, primary_jacobian.data(), eliminated_jacobian.data(), residual.data(), 2);
                }
            }

            SchurComplementSolverOptions<double> options;
            options.linearBackend = SchurComplementLinearBackend::DenseCpu;
            options.relativeTolerance = 1e-11;
            options.absoluteTolerance = 1e-13;
            std::vector<double> serial_primary;
            std::vector<double> serial_eliminated;
            std::vector<double> parallel_primary;
            std::vector<double> parallel_eliminated;
            const int original_threads = omp_get_max_threads();
            omp_set_num_threads(1);
            const auto serial =
                solveDampedSchurComplement(equations, 0.1, options, &serial_primary, &serial_eliminated);
            omp_set_num_threads(std::min(8, original_threads));
            const auto parallel =
                solveDampedSchurComplement(equations, 0.1, options, &parallel_primary, &parallel_eliminated);
            omp_set_num_threads(original_threads);

            ASSERT_TRUE(serial.converged) << serial.message;
            ASSERT_TRUE(parallel.converged) << parallel.message;
            EXPECT_EQ(serial_primary, parallel_primary);
            EXPECT_EQ(serial_eliminated, parallel_eliminated);
        }

        TEST(BlockSchurTest, DeterministicMergeMatchesSerialAssembly)
        {
            auto serial = makeSyntheticEquations();
            BlockNormalEquations<double> merged(2, 2, 2, 1);
            merged.mergeFrom(makeSyntheticEquations());

            SchurComplementSolverOptions<double> options;
            options.linearBackend = SchurComplementLinearBackend::DenseCpu;
            options.relativeTolerance = 1e-13;
            options.absoluteTolerance = 1e-14;
            std::vector<double> serial_primary;
            std::vector<double> serial_eliminated;
            std::vector<double> merged_primary;
            std::vector<double> merged_eliminated;
            const auto serial_report =
                solveDampedSchurComplement(serial, 0.1, options, &serial_primary, &serial_eliminated);
            const auto merged_report =
                solveDampedSchurComplement(merged, 0.1, options, &merged_primary, &merged_eliminated);
            ASSERT_TRUE(serial_report.converged) << serial_report.message;
            ASSERT_TRUE(merged_report.converged) << merged_report.message;
            EXPECT_EQ(serial_primary, merged_primary);
            EXPECT_EQ(serial_eliminated, merged_eliminated);
        }

        TEST(BlockSchurTest, EliminatedShardMergeMatchesSerialAssembly)
        {
            auto serial = makeSyntheticEquations();
            BlockNormalEquations<double> first_shard(2, 1, 2, 1);
            addSyntheticResidual(first_shard, 0, 0, {{1.0, 2.0}}, 0.5, -1.0);
            addSyntheticResidual(first_shard, 1, 0, {{-0.5, 1.0}}, 1.5, 0.25);

            BlockNormalEquations<double> second_shard(2, 1, 2, 1);
            addSyntheticResidual(second_shard, 0, 0, {{2.0, -1.0}}, -0.75, 2.0);
            addSyntheticResidual(second_shard, 1, 0, {{1.25, 0.5}}, 2.0, -0.5);
            const double primary_prior_jacobian[2] = {0.4, -0.2};
            const double primary_prior_residual = 0.75;
            second_shard.addPrimaryResidualBlock(0, primary_prior_jacobian, &primary_prior_residual, 1, 2.0);
            const double eliminated_prior_jacobian = 0.8;
            const double eliminated_prior_residual = -0.3;
            second_shard.addEliminatedResidualBlock(0, &eliminated_prior_jacobian, &eliminated_prior_residual, 1, 1.5);

            BlockNormalEquations<double> merged(2, 2, 2, 1);
            merged.mergeEliminatedShardFrom(first_shard, 0);
            merged.mergeEliminatedShardFrom(second_shard, 1);

            SchurComplementSolverOptions<double> options;
            options.linearBackend = SchurComplementLinearBackend::DenseCpu;
            options.relativeTolerance = 1e-13;
            options.absoluteTolerance = 1e-14;
            std::vector<double> serial_primary;
            std::vector<double> serial_eliminated;
            std::vector<double> merged_primary;
            std::vector<double> merged_eliminated;
            const auto serial_report =
                solveDampedSchurComplement(serial, 0.1, options, &serial_primary, &serial_eliminated);
            const auto merged_report =
                solveDampedSchurComplement(merged, 0.1, options, &merged_primary, &merged_eliminated);
            ASSERT_TRUE(serial_report.converged) << serial_report.message;
            ASSERT_TRUE(merged_report.converged) << merged_report.message;
            ASSERT_EQ(serial_primary.size(), merged_primary.size());
            ASSERT_EQ(serial_eliminated.size(), merged_eliminated.size());
            for (std::size_t index = 0; index < serial_primary.size(); ++index)
            {
                EXPECT_NEAR(serial_primary[index], merged_primary[index], 1e-14);
            }
            for (std::size_t index = 0; index < serial_eliminated.size(); ++index)
            {
                EXPECT_NEAR(serial_eliminated[index], merged_eliminated[index], 1e-14);
            }
        }

#ifdef PLAMATRIX_WITH_CUDA
        TEST(BlockSchurAcceleratedTest, CudaMatchesDenseDampedNormalEquation)
        {
            int device_count = 0;
            if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0)
            {
                GTEST_SKIP() << "CUDA device is unavailable";
            }
            expectSyntheticReference(SchurComplementLinearBackend::Cuda);
            expectMultiPrimaryReference(SchurComplementLinearBackend::Cuda);
            expectAcceleratedPatternReuse(SchurComplementLinearBackend::Cuda);
        }

        TEST(BlockSchurAcceleratedTest, CudaMixedPrecisionFallsBackToDoubleForIllConditionedSeed)
        {
            int device_count = 0;
            if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0)
            {
                GTEST_SKIP() << "CUDA device is unavailable";
            }
            const auto equations = makeSyntheticEquations();
            SchurComplementSolverOptions<double> options;
            options.linearBackend = SchurComplementLinearBackend::Cuda;
            options.maxIterations = 50;
            options.relativeTolerance = 1e-12;
            options.absoluteTolerance = 1e-14;
            options.useMixedPrecision = true;
            std::vector<double> primary_step;
            std::vector<double> eliminated_step;
            const auto report = solveDampedSchurComplement(equations, 0.1, options, &primary_step, &eliminated_step);
            ASSERT_TRUE(report.converged) << report.message;
            EXPECT_FALSE(report.mixedPrecisionUsed);
            EXPECT_NEAR(primary_step[0], -0.532196868252671, 1e-10);
            EXPECT_NEAR(eliminated_step[0], 0.017900956933130, 1e-10);
        }
#endif

#ifdef PLAMATRIX_WITH_OPENCL
        TEST(BlockSchurAcceleratedTest, OpenClMatchesDenseDampedNormalEquation)
        {
            if (!opencl::hasUsableOpenClDevice())
            {
                GTEST_SKIP() << "OpenCL GPU is unavailable";
            }
            expectSyntheticReference(SchurComplementLinearBackend::OpenCl);
            expectMultiPrimaryReference(SchurComplementLinearBackend::OpenCl);
            expectAcceleratedPatternReuse(SchurComplementLinearBackend::OpenCl);
        }
#endif

        TEST(BlockSchurTest, AggregatesRepeatedVariablePairsBeforeElimination)
        {
            BlockNormalEquations<double> equations(1, 1, 1, 1);
            addSyntheticResidual(equations, 0, 0, {{1.0, 0.0}}, 1.0, 1.0);

            const double primary_jacobian = 2.0;
            const double eliminated_jacobian = -1.0;
            const double residual = -0.5;
            equations.addResidualBlock(0, 0, &primary_jacobian, &eliminated_jacobian, &residual, 1, 1.0);

            SchurComplementSolverOptions<double> options;
            std::vector<double> primary_step;
            std::vector<double> eliminated_step;
            const auto report = solveDampedSchurComplement(equations, 0.5, options, &primary_step, &eliminated_step);

            ASSERT_TRUE(report.converged) << report.message;
            ASSERT_EQ(primary_step.size(), 1u);
            ASSERT_EQ(eliminated_step.size(), 1u);
            EXPECT_NEAR(primary_step[0], -0.069767441860465, 1e-12);
            EXPECT_NEAR(eliminated_step[0], -0.523255813953488, 1e-12);
        }

        TEST(BlockSchurTest, RejectsDimensionMismatchAndNonFiniteInput)
        {
            BlockNormalEquations<double> equations(1, 1, 2, 1);
            const double primary_jacobian[2] = {1.0, 0.0};
            const double eliminated_jacobian = 1.0;
            const double residual = 0.0;

            EXPECT_THROW(equations.addResidualBlock(1, 0, primary_jacobian, &eliminated_jacobian, &residual, 1, 1.0),
                         std::out_of_range);
            EXPECT_THROW(equations.addResidualBlock(0, 0, primary_jacobian, &eliminated_jacobian, &residual, 0, 1.0),
                         std::invalid_argument);
            EXPECT_THROW(equations.addResidualBlock(0,
                                                    0,
                                                    primary_jacobian,
                                                    &eliminated_jacobian,
                                                    &residual,
                                                    1,
                                                    std::numeric_limits<double>::infinity()),
                         std::invalid_argument);

            SchurComplementSolverOptions<double> options;
            std::vector<double> aliased_step;
            EXPECT_THROW(solveDampedSchurComplement(equations, 1e-3, options, &aliased_step, &aliased_step),
                         std::invalid_argument);
        }

        TEST(BlockSchurTest, FloatPointOnlySystemUsesDirectEliminatedSolve)
        {
            BlockNormalEquations<float> equations(0, 1, 6, 1);
            const float jacobian = 2.0f;
            const float residual = 4.0f;
            equations.addEliminatedResidualBlock(0, &jacobian, &residual, 1);

            SchurComplementSolverOptions<float> options;
            std::vector<float> primary_step;
            std::vector<float> eliminated_step;
            const auto report = solveDampedSchurComplement(equations, 0.5f, options, &primary_step, &eliminated_step);

            ASSERT_TRUE(report.converged) << report.message;
            EXPECT_TRUE(primary_step.empty());
            ASSERT_EQ(eliminated_step.size(), 1u);
            EXPECT_NEAR(eliminated_step[0], -4.0f / 3.0f, 1e-6f);
        }

        TEST(LevenbergMarquardtTest, AcceptedAndRejectedStepsUpdateDampingAndCounters)
        {
            LevenbergMarquardtOptions<double> options;
            options.initialDamping = 1e-3;
            options.minimumDamping = 1e-8;
            options.maximumDamping = 1e4;
            options.decreaseFactor = 0.25;
            options.increaseFactor = 8.0;
            LevenbergMarquardtStrategy<double> strategy(options);

            strategy.acceptStep();
            EXPECT_DOUBLE_EQ(strategy.damping(), 2.5e-4);
            EXPECT_EQ(strategy.acceptedSteps(), 1);
            EXPECT_EQ(strategy.rejectedSteps(), 0);

            strategy.rejectStep();
            EXPECT_DOUBLE_EQ(strategy.damping(), 2e-3);
            EXPECT_EQ(strategy.acceptedSteps(), 1);
            EXPECT_EQ(strategy.rejectedSteps(), 1);
        }

    } // namespace
} // namespace plamatrix::internal
