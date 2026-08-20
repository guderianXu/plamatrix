#include <array>
#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "plamatrix/optimization/block_schur.h"
#include "plamatrix/optimization/levenberg_marquardt.h"
#include "plamatrix/optimization/robust_loss.h"
#include "plamatrix/opencl/runtime.h"

#ifdef PLAMATRIX_WITH_CUDA
#include <cuda_runtime_api.h>
#endif

namespace plamatrix
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
    equations.addResidualBlock(primary,
                               eliminated,
                               primary_jacobian.data(),
                               &eliminated_jacobian,
                               &residual,
                               1,
                               1.0);
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
    equations.addPrimaryResidualBlock(
        0, primary_prior_jacobian, &primary_prior_residual, 1, 2.0);

    const double eliminated_prior_jacobian = 0.8;
    const double eliminated_prior_residual = -0.3;
    equations.addEliminatedResidualBlock(
        1, &eliminated_prior_jacobian, &eliminated_prior_residual, 1, 1.5);
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
        {0, 1},
        {&primary_jacobian_0, &primary_jacobian_1},
        0,
        &eliminated_jacobian,
        &residual,
        1);

    const double direct_jacobian_0 = 0.5;
    const double direct_jacobian_1 = -1.0;
    const double direct_residual = 0.2;
    equations.addPrimaryResidualBlocks(
        {0, 1},
        {&direct_jacobian_0, &direct_jacobian_1},
        &direct_residual,
        1);

    SchurComplementSolverOptions<double> options;
    options.linearBackend = backend;
    options.maxIterations = 50;
    options.relativeTolerance = 1e-13;
    options.absoluteTolerance = 1e-14;
    std::vector<double> primary_step;
    std::vector<double> eliminated_step;
    const auto report = solveDampedSchurComplement(
        equations, 0.1, options, &primary_step, &eliminated_step);

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
    const auto report = solveDampedSchurComplement(
        equations, 0.1, options, &primary_step, &eliminated_step);

    ASSERT_TRUE(report.converged) << report.message;
    EXPECT_EQ(report.linearBackend, backend);
    if (backend == SchurComplementLinearBackend::Cuda ||
        backend == SchurComplementLinearBackend::OpenCl)
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

    const auto first = solveDampedSchurComplement(
        equations, 0.1, options, workspace, &primary_step, &eliminated_step);
    ASSERT_TRUE(first.converged) << first.message;
    EXPECT_FALSE(first.schurPatternReused);
    EXPECT_EQ(workspace.patternBuildCount(), 1u);

    const auto second = solveDampedSchurComplement(
        equations, 0.2, options, workspace, &primary_step, &eliminated_step);
    ASSERT_TRUE(second.converged) << second.message;
    EXPECT_TRUE(second.schurPatternReused);
    EXPECT_EQ(workspace.patternBuildCount(), 1u);

    BlockNormalEquations<double> changed_equations(1, 1, 1, 1);
    const double jacobian = 1.0;
    const double residual = -0.5;
    changed_equations.addResidualBlock(
        0, 0, &jacobian, &jacobian, &residual, 1);
    changed_equations.addPrimaryResidualBlock(
        0, &jacobian, &residual, 1);
    const auto changed = solveDampedSchurComplement(
        changed_equations, 0.2, options, workspace, &primary_step, &eliminated_step);
    ASSERT_TRUE(changed.converged) << changed.message;
    EXPECT_FALSE(changed.schurPatternReused);
    EXPECT_EQ(workspace.patternBuildCount(), 2u);
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
    const auto report = solveDampedSchurComplement(
        equations, 0.1, options, &primary_step, &eliminated_step);

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

TEST(BlockSchurTest, DenseCpuSolvesParallelSizeCoupledSystemDeterministically)
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
            residual[static_cast<std::size_t>(component)] =
                0.01 * static_cast<double>(1 + block + component);
        }
        equations.addPrimaryResidualBlock(
            block, identity.data(), residual.data(), block_size);
    }

    const std::array<double, block_size> left_jacobian{{1.0, 0.2, -0.1, 0.3, 0.05, -0.2}};
    const std::array<double, block_size> right_jacobian{{-0.8, 0.1, 0.25, -0.2, 0.15, 0.3}};
    for (Index block = 0; block + 1 < block_count; ++block)
    {
        const double residual = 0.02 * static_cast<double>(block + 1);
        equations.addPrimaryResidualBlocks(
            {block, block + 1},
            {left_jacobian.data(), right_jacobian.data()},
            &residual,
            1);
    }

    SchurComplementSolverOptions<double> options;
    options.linearBackend = SchurComplementLinearBackend::DenseCpu;
    options.relativeTolerance = 1e-12;
    options.absoluteTolerance = 1e-14;
    std::vector<double> first_primary;
    std::vector<double> first_eliminated;
    std::vector<double> second_primary;
    std::vector<double> second_eliminated;
    const auto first = solveDampedSchurComplement(
        equations, 0.01, options, &first_primary, &first_eliminated);
    const auto second = solveDampedSchurComplement(
        equations, 0.01, options, &second_primary, &second_eliminated);

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
    const auto serial_report = solveDampedSchurComplement(
        serial, 0.1, options, &serial_primary, &serial_eliminated);
    const auto merged_report = solveDampedSchurComplement(
        merged, 0.1, options, &merged_primary, &merged_eliminated);
    ASSERT_TRUE(serial_report.converged) << serial_report.message;
    ASSERT_TRUE(merged_report.converged) << merged_report.message;
    EXPECT_EQ(serial_primary, merged_primary);
    EXPECT_EQ(serial_eliminated, merged_eliminated);
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
    const auto report = solveDampedSchurComplement(
        equations, 0.1, options, &primary_step, &eliminated_step);
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
    equations.addResidualBlock(
        0, 0, &primary_jacobian, &eliminated_jacobian, &residual, 1, 1.0);

    SchurComplementSolverOptions<double> options;
    std::vector<double> primary_step;
    std::vector<double> eliminated_step;
    const auto report = solveDampedSchurComplement(
        equations, 0.5, options, &primary_step, &eliminated_step);

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

    EXPECT_THROW(equations.addResidualBlock(
                     1, 0, primary_jacobian, &eliminated_jacobian, &residual, 1, 1.0),
                 std::out_of_range);
    EXPECT_THROW(equations.addResidualBlock(
                     0, 0, primary_jacobian, &eliminated_jacobian, &residual, 0, 1.0),
                 std::invalid_argument);
    EXPECT_THROW(equations.addResidualBlock(
                     0, 0, primary_jacobian, &eliminated_jacobian, &residual, 1,
                     std::numeric_limits<double>::infinity()),
                 std::invalid_argument);

    SchurComplementSolverOptions<double> options;
    std::vector<double> aliased_step;
    EXPECT_THROW(solveDampedSchurComplement(
                     equations, 1e-3, options, &aliased_step, &aliased_step),
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
    const auto report = solveDampedSchurComplement(
        equations, 0.5f, options, &primary_step, &eliminated_step);

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
} // namespace plamatrix
