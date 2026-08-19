#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

#ifdef PLAMATRIX_WITH_CUDA
#include "iterative_solver_test_hooks.h"
#endif

#include <gtest/gtest.h>

#include <plamatrix/sparse/iterative_solver.h>
#include <plamatrix/sparse/sparse_ops.h>

using namespace plamatrix;

template <typename Scalar>
class IterativeSolverCpuTest : public ::testing::Test
{
};

using IterativeSolverScalars = ::testing::Types<
#ifdef PLAMATRIX_USE_FLOAT
    float
#ifdef PLAMATRIX_USE_DOUBLE
    ,
#endif
#endif
#ifdef PLAMATRIX_USE_DOUBLE
    double
#endif
    >;
TYPED_TEST_SUITE(IterativeSolverCpuTest, IterativeSolverScalars);

static_assert(!std::is_copy_constructible_v<IterativeSolverWorkspace<float>>);
static_assert(!std::is_copy_assignable_v<IterativeSolverWorkspace<float>>);
static_assert(std::is_nothrow_move_constructible_v<IterativeSolverWorkspace<float>>);
static_assert(std::is_nothrow_move_assignable_v<IterativeSolverWorkspace<float>>);
static_assert(std::is_default_constructible_v<AsyncIterativeSolverState>);
static_assert(!std::is_copy_constructible_v<AsyncIterativeSolverState>);
static_assert(std::is_nothrow_move_constructible_v<AsyncIterativeSolverState>);

template <typename Scalar>
CSRMatrix<Scalar, Device::CPU> poisson1d(Index size, Scalar scale = Scalar{1})
{
    std::vector<Index> rows;
    std::vector<Index> cols;
    std::vector<Scalar> values;
    for (Index row = 0; row < size; ++row)
    {
        if (row > 0)
        {
            rows.push_back(row);
            cols.push_back(row - 1);
            values.push_back(-scale);
        }
        rows.push_back(row);
        cols.push_back(row);
        values.push_back(Scalar{2} * scale);
        if (row + 1 < size)
        {
            rows.push_back(row);
            cols.push_back(row + 1);
            values.push_back(-scale);
        }
    }
    return cooToCsr(size, size, rows, cols, values);
}

template <typename Scalar>
double solverTolerance()
{
    return std::is_same_v<Scalar, float> ? 2.0e-4 : 1.0e-10;
}

template <typename Scalar>
CSRMatrix<Scalar, Device::CPU> coupledBlockDiagonalSystem()
{
    return cooToCsr(
        4,
        4,
        std::vector<Index>{0, 0, 1, 1, 2, 2, 3, 3},
        std::vector<Index>{0, 1, 0, 1, 2, 3, 2, 3},
        std::vector<Scalar>{
            Scalar(4), Scalar(1), Scalar(1), Scalar(3),
            Scalar(2), Scalar(0.5), Scalar(0.5), Scalar(1.5)});
}

template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> coupledBlockInverse()
{
    DenseMatrix<Scalar, Device::CPU> inverse(8, 1);
    const std::vector<Scalar> values{
        Scalar(3.0 / 11.0), Scalar(-1.0 / 11.0),
        Scalar(-1.0 / 11.0), Scalar(4.0 / 11.0),
        Scalar(1.5 / 2.75), Scalar(-0.5 / 2.75),
        Scalar(-0.5 / 2.75), Scalar(2.0 / 2.75)};
    std::copy(values.begin(), values.end(), inverse.data());
    return inverse;
}

TYPED_TEST(IterativeSolverCpuTest, pcgSolvesDiagonalSystemInOneIteration)
{
    const auto matrix = cooToCsr(
        3, 3, std::vector<Index>{0, 1, 2}, std::vector<Index>{0, 1, 2},
        std::vector<TypeParam>{TypeParam(2), TypeParam(4), TypeParam(8)});
    DenseMatrix<TypeParam, Device::CPU> rhs(3, 1);
    rhs(0, 0) = TypeParam(2);
    rhs(1, 0) = TypeParam(8);
    rhs(2, 0) = TypeParam(24);
    DenseMatrix<TypeParam, Device::CPU> solution(3, 1);
    solution.fill(TypeParam{0});

    const IterativeSolverReport report = pcg(matrix, rhs, solution);

    EXPECT_TRUE(report.converged);
    EXPECT_EQ(report.iterations, 1);
    EXPECT_NEAR(solution(0, 0), TypeParam(1), solverTolerance<TypeParam>());
    EXPECT_NEAR(solution(1, 0), TypeParam(2), solverTolerance<TypeParam>());
    EXPECT_NEAR(solution(2, 0), TypeParam(3), solverTolerance<TypeParam>());
    EXPECT_LE(report.finalResidual, report.initialResidual);
}

TYPED_TEST(IterativeSolverCpuTest, cgAndPcgSolvePoissonSystemAcrossScales)
{
    for (const TypeParam scale : {TypeParam(1.0e-3), TypeParam(1), TypeParam(1.0e3)})
    {
        const auto matrix = poisson1d<TypeParam>(8, scale);
        DenseMatrix<TypeParam, Device::CPU> expected(8, 1);
        for (Index row = 0; row < 8; ++row)
        {
            expected(row, 0) = TypeParam(row + 1);
        }
        const auto rhs = spmv(matrix, expected);
        DenseMatrix<TypeParam, Device::CPU> cg_solution(8, 1);
        DenseMatrix<TypeParam, Device::CPU> pcg_solution(8, 1);
        cg_solution.fill(TypeParam{0});
        pcg_solution.fill(TypeParam{0});
        IterativeSolverOptions options;
        options.relativeTolerance = std::is_same_v<TypeParam, float> ? 1.0e-5 : 1.0e-12;

        const auto cg_report = cg(matrix, rhs, cg_solution, options);
        const auto pcg_report = pcg(matrix, rhs, pcg_solution, options);

        EXPECT_TRUE(cg_report.converged);
        EXPECT_TRUE(pcg_report.converged);
        for (Index row = 0; row < 8; ++row)
        {
            const double tolerance = solverTolerance<TypeParam>() * (row + 1);
            EXPECT_NEAR(cg_solution(row, 0), expected(row, 0), tolerance);
            EXPECT_NEAR(pcg_solution(row, 0), expected(row, 0), tolerance);
        }
    }
}

TYPED_TEST(IterativeSolverCpuTest, initiallyConvergedSystemsUseZeroIterations)
{
    const auto matrix = poisson1d<TypeParam>(4);
    DenseMatrix<TypeParam, Device::CPU> rhs(4, 1);
    DenseMatrix<TypeParam, Device::CPU> solution(4, 1);
    rhs.fill(TypeParam{0});
    solution.fill(TypeParam{0});

    const auto zero_rhs_report = cg(matrix, rhs, solution);
    EXPECT_TRUE(zero_rhs_report.converged);
    EXPECT_EQ(zero_rhs_report.iterations, 0);
    EXPECT_EQ(zero_rhs_report.initialResidual, 0.0);
    EXPECT_EQ(zero_rhs_report.finalResidual, 0.0);

    solution.fill(TypeParam{2});
    const auto exact_rhs = spmv(matrix, solution);
    const auto exact_guess_report = pcg(matrix, exact_rhs, solution);
    EXPECT_TRUE(exact_guess_report.converged);
    EXPECT_EQ(exact_guess_report.iterations, 0);
}

TYPED_TEST(IterativeSolverCpuTest, rejectsInvalidSystemsOptionsAndJacobiDiagonal)
{
    const auto non_square = cooToCsr(
        2, 3, std::vector<Index>{0}, std::vector<Index>{0},
        std::vector<TypeParam>{TypeParam(1)});
    DenseMatrix<TypeParam, Device::CPU> rhs(2, 1);
    DenseMatrix<TypeParam, Device::CPU> solution(2, 1);
    EXPECT_THROW(cg(non_square, rhs, solution), std::invalid_argument);

    const auto square = poisson1d<TypeParam>(2);
    DenseMatrix<TypeParam, Device::CPU> wrong_rhs(3, 1);
    DenseMatrix<TypeParam, Device::CPU> wrong_columns(2, 2);
    EXPECT_THROW(cg(square, wrong_rhs, solution), std::invalid_argument);
    EXPECT_THROW(cg(square, rhs, wrong_columns), std::invalid_argument);

    IterativeSolverOptions invalid_options;
    invalid_options.maxIterations = -1;
    EXPECT_THROW(cg(square, rhs, solution, invalid_options), std::invalid_argument);
    invalid_options.maxIterations = 10;
    invalid_options.relativeTolerance = -1.0;
    EXPECT_THROW(cg(square, rhs, solution, invalid_options), std::invalid_argument);
    invalid_options.relativeTolerance = 1.0e100;
    EXPECT_THROW(cg(square, rhs, solution, invalid_options), std::invalid_argument);
    invalid_options.relativeTolerance = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(cg(square, rhs, solution, invalid_options), std::invalid_argument);
    invalid_options.relativeTolerance = 1.0e-6;
    invalid_options.absoluteTolerance = std::numeric_limits<double>::infinity();
    EXPECT_THROW(cg(square, rhs, solution, invalid_options), std::invalid_argument);

    rhs.fill(TypeParam{1});
    solution.fill(TypeParam{0});
    const auto missing_diagonal = cooToCsr(
        2, 2, std::vector<Index>{0, 1}, std::vector<Index>{1, 0},
        std::vector<TypeParam>{TypeParam(1), TypeParam(1)});
    EXPECT_THROW(pcg(missing_diagonal, rhs, solution), std::runtime_error);
    const auto zero_diagonal = cooToCsr(
        2, 2, std::vector<Index>{0, 1}, std::vector<Index>{0, 1},
        std::vector<TypeParam>{TypeParam(1), TypeParam(0)});
    EXPECT_THROW(pcg(zero_diagonal, rhs, solution), std::runtime_error);
}

TYPED_TEST(IterativeSolverCpuTest, jacobiSupportsSmallScalesAndCombinedDiagonalEntries)
{
    const TypeParam small = std::is_same_v<TypeParam, float>
        ? TypeParam(1.0e-7F) : TypeParam(1.0e-20);
    const auto small_matrix = cooToCsr(
        2, 2, std::vector<Index>{0, 1}, std::vector<Index>{0, 1},
        std::vector<TypeParam>{small, TypeParam(2) * small});
    DenseMatrix<TypeParam, Device::CPU> rhs(2, 1);
    DenseMatrix<TypeParam, Device::CPU> solution(2, 1);
    rhs(0, 0) = small;
    rhs(1, 0) = TypeParam(2) * small;
    solution.fill(TypeParam{0});

    const auto small_report = pcg(small_matrix, rhs, solution);
    EXPECT_TRUE(small_report.converged);
    EXPECT_EQ(small_report.iterations, 1);
    EXPECT_NEAR(solution(0, 0), TypeParam(1), solverTolerance<TypeParam>());
    EXPECT_NEAR(solution(1, 0), TypeParam(1), solverTolerance<TypeParam>());

    CSRMatrix<TypeParam, Device::CPU> duplicate_diagonal(2, 2, 4);
    duplicate_diagonal.rowOffsets()[0] = 0;
    duplicate_diagonal.rowOffsets()[1] = 2;
    duplicate_diagonal.rowOffsets()[2] = 4;
    duplicate_diagonal.colIndices()[0] = 0;
    duplicate_diagonal.colIndices()[1] = 0;
    duplicate_diagonal.colIndices()[2] = 1;
    duplicate_diagonal.colIndices()[3] = 1;
    duplicate_diagonal.values()[0] = TypeParam(1);
    duplicate_diagonal.values()[1] = TypeParam(1);
    duplicate_diagonal.values()[2] = TypeParam(1);
    duplicate_diagonal.values()[3] = TypeParam(3);
    rhs(0, 0) = TypeParam(2);
    rhs(1, 0) = TypeParam(4);
    solution.fill(TypeParam{0});

    const auto duplicate_report = pcg(duplicate_diagonal, rhs, solution);
    EXPECT_TRUE(duplicate_report.converged);
    EXPECT_EQ(duplicate_report.iterations, 1);
}

TYPED_TEST(IterativeSolverCpuTest, rejectsNonPositivePreconditionerAndMalformedCsr)
{
    DenseMatrix<TypeParam, Device::CPU> rhs(2, 1);
    DenseMatrix<TypeParam, Device::CPU> solution(2, 1);
    rhs.fill(TypeParam{1});
    solution.fill(TypeParam{0});

    const auto negative_diagonal = cooToCsr(
        2, 2, std::vector<Index>{0, 1}, std::vector<Index>{0, 1},
        std::vector<TypeParam>{TypeParam(-1), TypeParam(1)});
    EXPECT_THROW(pcg(negative_diagonal, rhs, solution), std::runtime_error);

    CSRMatrix<TypeParam, Device::CPU> bad_offsets(2, 2, 2);
    bad_offsets.rowOffsets()[0] = 0;
    bad_offsets.rowOffsets()[1] = 2;
    bad_offsets.rowOffsets()[2] = 1;
    EXPECT_THROW(cg(bad_offsets, rhs, solution), std::invalid_argument);

    CSRMatrix<TypeParam, Device::CPU> bad_column(2, 2, 2);
    bad_column.rowOffsets()[0] = 0;
    bad_column.rowOffsets()[1] = 1;
    bad_column.rowOffsets()[2] = 2;
    bad_column.colIndices()[0] = 0;
    bad_column.colIndices()[1] = 2;
    bad_column.values()[0] = TypeParam(1);
    bad_column.values()[1] = TypeParam(1);
    EXPECT_THROW(cg(bad_column, rhs, solution), std::invalid_argument);
}

TYPED_TEST(IterativeSolverCpuTest, supportsAbsoluteToleranceZeroIterationsAndUnpreconditionedPcg)
{
    const auto matrix = poisson1d<TypeParam>(4);
    DenseMatrix<TypeParam, Device::CPU> rhs(4, 1);
    DenseMatrix<TypeParam, Device::CPU> solution(4, 1);
    rhs.fill(TypeParam(1));
    solution.fill(TypeParam{0});

    IterativeSolverOptions options;
    options.maxIterations = 0;
    options.relativeTolerance = 0.0;
    const auto zero_iteration_report = cg(matrix, rhs, solution, options);
    EXPECT_FALSE(zero_iteration_report.converged);
    EXPECT_EQ(zero_iteration_report.iterations, 0);

    options.maxIterations = 10;
    options.absoluteTolerance = 2.0;
    const auto absolute_report = cg(matrix, rhs, solution, options);
    EXPECT_TRUE(absolute_report.converged);
    EXPECT_EQ(absolute_report.iterations, 0);

    options.absoluteTolerance = 0.0;
    options.relativeTolerance = std::is_same_v<TypeParam, float> ? 1.0e-5 : 1.0e-12;
    options.useJacobiPreconditioner = false;
    const auto unpreconditioned_report = pcg(matrix, rhs, solution, options);
    EXPECT_TRUE(unpreconditioned_report.converged);
}

TYPED_TEST(IterativeSolverCpuTest, reportsAndOptionallyThrowsOnNonConvergence)
{
    const auto matrix = poisson1d<TypeParam>(16);
    DenseMatrix<TypeParam, Device::CPU> rhs(16, 1);
    DenseMatrix<TypeParam, Device::CPU> solution(16, 1);
    rhs.fill(TypeParam{1});
    solution.fill(TypeParam{0});
    IterativeSolverOptions options;
    options.maxIterations = 1;
    options.relativeTolerance = 0.0;
    options.absoluteTolerance = 0.0;

    const auto report = cg(matrix, rhs, solution, options);
    EXPECT_FALSE(report.converged);
    EXPECT_EQ(report.iterations, 1);
    EXPECT_TRUE(std::isfinite(report.initialResidual));
    EXPECT_TRUE(std::isfinite(report.finalResidual));

    solution.fill(TypeParam{0});
    options.requireConvergence = true;
    EXPECT_THROW(cg(matrix, rhs, solution, options), std::runtime_error);
}

#ifdef PLAMATRIX_WITH_CUDA

template <typename Scalar>
class IterativeSolverGpuTest : public ::testing::Test
{
};
TYPED_TEST_SUITE(IterativeSolverGpuTest, IterativeSolverScalars);

TYPED_TEST(IterativeSolverGpuTest, adaptiveCgAndPcgMatchCpuAndReuseWorkspace)
{
    const auto matrix_cpu = poisson1d<TypeParam>(32);
    DenseMatrix<TypeParam, Device::CPU> expected(32, 1);
    for (Index row = 0; row < expected.rows(); ++row)
    {
        expected(row, 0) = TypeParam(row + 1);
    }
    const auto rhs_cpu = spmv(matrix_cpu, expected);
    DenseMatrix<TypeParam, Device::CPU> cpu_solution(32, 1);
    cpu_solution.fill(TypeParam{0});
    IterativeSolverOptions options;
    options.relativeTolerance = std::is_same_v<TypeParam, float> ? 1.0e-5 : 1.0e-12;
    const auto cpu_report = pcg(matrix_cpu, rhs_cpu, cpu_solution, options);

    const auto matrix_gpu = matrix_cpu.toGpu();
    const auto rhs_gpu = rhs_cpu.toGpu();
    DenseMatrix<TypeParam, Device::GPU> gpu_solution(32, 1);
    gpu_solution.fill(TypeParam{0});
    IterativeSolverWorkspace<TypeParam> workspace;
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);

    const auto gpu_report = pcg(
        matrix_gpu, rhs_gpu, gpu_solution, workspace, options, stream);
    const auto first_solution = gpu_solution.toCpu();
    EXPECT_TRUE(gpu_report.converged);
    EXPECT_NEAR(gpu_report.finalResidual, cpu_report.finalResidual,
                (std::is_same_v<TypeParam, float> ? 2.0e-4 : 1.0e-9));
    for (Index row = 0; row < expected.rows(); ++row)
    {
        EXPECT_NEAR(first_solution(row, 0), cpu_solution(row, 0),
                    solverTolerance<TypeParam>() * (row + 1));
    }

    gpu_solution.fill(TypeParam{0});
    const auto reused_report = cg(
        matrix_gpu, rhs_gpu, gpu_solution, workspace, options, stream);
    EXPECT_TRUE(reused_report.converged);
    EXPECT_GT(reused_report.iterations, 0);

    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    EXPECT_NO_THROW(workspace.closeAsyncAllocation());
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TYPED_TEST(IterativeSolverGpuTest, blockPcgUsesCallerSuppliedInverseBlocks)
{
    const auto matrix_cpu = coupledBlockDiagonalSystem<TypeParam>();
    DenseMatrix<TypeParam, Device::CPU> expected(4, 1);
    expected(0, 0) = TypeParam(1);
    expected(1, 0) = TypeParam(2);
    expected(2, 0) = TypeParam(-1);
    expected(3, 0) = TypeParam(3);
    const auto rhs_cpu = spmv(matrix_cpu, expected);
    const auto inverse_cpu = coupledBlockInverse<TypeParam>();
    const auto matrix_gpu = matrix_cpu.toGpu();
    const auto rhs_gpu = rhs_cpu.toGpu();
    const auto inverse_gpu = inverse_cpu.toGpu();
    DenseMatrix<TypeParam, Device::GPU> solution_gpu(4, 1);
    solution_gpu.fill(TypeParam(0));
    IterativeSolverWorkspace<TypeParam> workspace;
    IterativeSolverOptions options;
    options.relativeTolerance = std::is_same_v<TypeParam, float> ? 1.0e-5 : 1.0e-12;

    const auto report = blockPcg(
        matrix_gpu, rhs_gpu, solution_gpu, inverse_gpu, 2, workspace, options);

    ASSERT_TRUE(report.converged);
    EXPECT_EQ(report.iterations, 1);
    const auto solution_cpu = solution_gpu.toCpu();
    for (Index row = 0; row < expected.rows(); ++row)
    {
        EXPECT_NEAR(solution_cpu(row, 0), expected(row, 0), solverTolerance<TypeParam>());
    }
    EXPECT_NO_THROW(workspace.closeAsyncAllocation());
}

TYPED_TEST(IterativeSolverGpuTest, adaptiveReportsZeroIterationsAndNonConvergence)
{
    const auto matrix_cpu = poisson1d<TypeParam>(16);
    DenseMatrix<TypeParam, Device::CPU> rhs_cpu(16, 1);
    DenseMatrix<TypeParam, Device::CPU> exact_cpu(16, 1);
    exact_cpu.fill(TypeParam(2));
    const auto exact_rhs_cpu = spmv(matrix_cpu, exact_cpu);
    rhs_cpu.fill(TypeParam(1));
    const auto matrix_gpu = matrix_cpu.toGpu();
    const auto rhs_gpu = rhs_cpu.toGpu();
    const auto exact_rhs_gpu = exact_rhs_cpu.toGpu();
    auto solution_gpu = exact_cpu.toGpu();
    IterativeSolverWorkspace<TypeParam> workspace;

    const auto exact_report = pcg(matrix_gpu, exact_rhs_gpu, solution_gpu, workspace);
    EXPECT_TRUE(exact_report.converged);
    EXPECT_EQ(exact_report.iterations, 0);

    solution_gpu.fill(TypeParam{0});
    IterativeSolverOptions options;
    options.maxIterations = 1;
    options.relativeTolerance = 0.0;
    options.absoluteTolerance = 0.0;
    const auto nonconverged = cg(matrix_gpu, rhs_gpu, solution_gpu, workspace, options);
    EXPECT_FALSE(nonconverged.converged);
    EXPECT_EQ(nonconverged.iterations, 1);

    solution_gpu.fill(TypeParam{0});
    options.requireConvergence = true;
    EXPECT_THROW(cg(matrix_gpu, rhs_gpu, solution_gpu, workspace, options),
                 std::runtime_error);
    ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
    EXPECT_NO_THROW(workspace.closeAsyncAllocation());
}

TYPED_TEST(IterativeSolverGpuTest, pcgHandlesPlannedProblemSizesAndWorkspaceResize)
{
    IterativeSolverWorkspace<TypeParam> workspace;
    for (const Index size : {Index{10}, Index{1000}, Index{100000}})
    {
        std::vector<Index> rows(static_cast<std::size_t>(size));
        std::vector<Index> columns(static_cast<std::size_t>(size));
        std::vector<TypeParam> values(static_cast<std::size_t>(size), TypeParam(2));
        for (Index row = 0; row < size; ++row)
        {
            rows[static_cast<std::size_t>(row)] = row;
            columns[static_cast<std::size_t>(row)] = row;
        }
        const auto matrix_cpu = cooToCsr(size, size, rows, columns, values);
        DenseMatrix<TypeParam, Device::CPU> rhs_cpu(size, 1);
        rhs_cpu.fill(TypeParam(2));
        const auto matrix_gpu = matrix_cpu.toGpu();
        const auto rhs_gpu = rhs_cpu.toGpu();
        DenseMatrix<TypeParam, Device::GPU> solution_gpu(size, 1);
        solution_gpu.fill(TypeParam{0});

        const auto report = pcg(matrix_gpu, rhs_gpu, solution_gpu, workspace);
        EXPECT_TRUE(report.converged);
        EXPECT_EQ(report.iterations, 1);
        const auto solution_cpu = solution_gpu.toCpu();
        EXPECT_NEAR(solution_cpu(0, 0), TypeParam(1), solverTolerance<TypeParam>());
        EXPECT_NEAR(solution_cpu(size - 1, 0), TypeParam(1), solverTolerance<TypeParam>());
        EXPECT_EQ(workspace.capacitySize(), size);
    }
    EXPECT_NO_THROW(workspace.closeAsyncAllocation());
}

TYPED_TEST(IterativeSolverGpuTest, pcgHonorsDisabledJacobiPreconditioner)
{
    const auto matrix_cpu = cooToCsr(
        3, 3, std::vector<Index>{0, 1, 2}, std::vector<Index>{0, 1, 2},
        std::vector<TypeParam>{TypeParam(2), TypeParam(4), TypeParam(8)});
    DenseMatrix<TypeParam, Device::CPU> rhs_cpu(3, 1);
    rhs_cpu(0, 0) = TypeParam(2);
    rhs_cpu(1, 0) = TypeParam(8);
    rhs_cpu(2, 0) = TypeParam(24);
    const auto matrix_gpu = matrix_cpu.toGpu();
    const auto rhs_gpu = rhs_cpu.toGpu();
    DenseMatrix<TypeParam, Device::GPU> solution_gpu(3, 1);
    solution_gpu.fill(TypeParam{0});
    IterativeSolverWorkspace<TypeParam> workspace;
    IterativeSolverOptions options;
    options.maxIterations = 1;
    options.relativeTolerance = 0.0;
    options.absoluteTolerance = 0.0;
    options.useJacobiPreconditioner = false;

    const auto report = pcg(matrix_gpu, rhs_gpu, solution_gpu, workspace, options);

    EXPECT_FALSE(report.converged);
    EXPECT_EQ(report.iterations, 1);
    EXPECT_GT(report.finalResidual, 0.0);
    EXPECT_NO_THROW(workspace.closeAsyncAllocation());
}

TYPED_TEST(IterativeSolverGpuTest, breakdownDoesNotModifySolution)
{
    const auto matrix_cpu = cooToCsr(
        2, 2, std::vector<Index>{0, 1}, std::vector<Index>{0, 1},
        std::vector<TypeParam>{TypeParam(-1), TypeParam(1)});
    DenseMatrix<TypeParam, Device::CPU> rhs_cpu(2, 1);
    rhs_cpu.fill(TypeParam(1));
    const auto matrix_gpu = matrix_cpu.toGpu();
    const auto rhs_gpu = rhs_cpu.toGpu();
    DenseMatrix<TypeParam, Device::GPU> solution_gpu(2, 1);
    solution_gpu.fill(TypeParam{0});
    IterativeSolverWorkspace<TypeParam> workspace;

    EXPECT_THROW(cg(matrix_gpu, rhs_gpu, solution_gpu, workspace), std::runtime_error);

    const auto solution_cpu = solution_gpu.toCpu();
    EXPECT_EQ(solution_cpu(0, 0), TypeParam(0));
    EXPECT_EQ(solution_cpu(1, 0), TypeParam(0));
    EXPECT_NO_THROW(workspace.closeAsyncAllocation());
}

TYPED_TEST(IterativeSolverGpuTest, alphaOverflowDoesNotModifySolution)
{
    const TypeParam tiny = std::numeric_limits<TypeParam>::denorm_min();
    const auto matrix_cpu = cooToCsr(
        1, 1, std::vector<Index>{0}, std::vector<Index>{0},
        std::vector<TypeParam>{tiny});
    DenseMatrix<TypeParam, Device::CPU> rhs_cpu(1, 1);
    rhs_cpu(0, 0) = TypeParam(1);
    const auto matrix_gpu = matrix_cpu.toGpu();
    const auto rhs_gpu = rhs_cpu.toGpu();
    DenseMatrix<TypeParam, Device::GPU> solution_gpu(1, 1);
    solution_gpu.fill(TypeParam{0});
    IterativeSolverWorkspace<TypeParam> workspace;

    EXPECT_THROW(cg(matrix_gpu, rhs_gpu, solution_gpu, workspace), std::runtime_error);

    EXPECT_EQ(solution_gpu.toCpu()(0, 0), TypeParam(0));
    EXPECT_NO_THROW(workspace.closeAsyncAllocation());
}

TYPED_TEST(IterativeSolverGpuTest, deviceCooResultSupportsFixedSolverWithoutRawPointerEscape)
{
    DenseMatrix<Index, Device::CPU> rows_cpu(2, 1);
    DenseMatrix<Index, Device::CPU> columns_cpu(2, 1);
    DenseMatrix<TypeParam, Device::CPU> values_cpu(2, 1);
    rows_cpu(0, 0) = 0;
    rows_cpu(1, 0) = 1;
    columns_cpu(0, 0) = 0;
    columns_cpu(1, 0) = 1;
    values_cpu(0, 0) = TypeParam(2);
    values_cpu(1, 0) = TypeParam(4);
    const auto rows_gpu = rows_cpu.toGpu();
    const auto columns_gpu = columns_cpu.toGpu();
    const auto values_gpu = values_cpu.toGpu();
    SparseOpsWorkspace coo_workspace;
    auto matrix_gpu = cooToCsr(
        2, 2, rows_gpu, columns_gpu, values_gpu, coo_workspace);
    ASSERT_TRUE(matrix_gpu.hasValidatedStructure());
    DenseMatrix<TypeParam, Device::CPU> rhs_cpu(2, 1);
    rhs_cpu(0, 0) = TypeParam(2);
    rhs_cpu(1, 0) = TypeParam(4);
    const auto rhs_gpu = rhs_cpu.toGpu();
    DenseMatrix<TypeParam, Device::GPU> solution_gpu(2, 1);
    solution_gpu.fill(TypeParam{0});
    IterativeSolverWorkspace<TypeParam> solver_workspace;

    auto state = pcgFixedIterationsAsync(
        matrix_gpu, rhs_gpu, solution_gpu, 1, solver_workspace);
    ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
    EXPECT_TRUE(finalizeIterativeSolverReport(state).converged);
    EXPECT_NO_THROW(state.closeAsyncAllocation());
    EXPECT_NO_THROW(solver_workspace.closeAsyncAllocation());
    EXPECT_NO_THROW(coo_workspace.closeAsyncAllocation());
    ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
}

TYPED_TEST(IterativeSolverGpuTest, rejectsMalformedGpuCsrBeforeCusparseLaunch)
{
    auto matrix_gpu = poisson1d<TypeParam>(2).toGpu();
    const Index invalid_offsets[] = {0, 2, 1};
    ASSERT_EQ(cudaMemcpy(
        matrix_gpu.rowOffsets(), invalid_offsets, sizeof(invalid_offsets),
        cudaMemcpyHostToDevice), cudaSuccess);
    DenseMatrix<TypeParam, Device::CPU> rhs_cpu(2, 1);
    rhs_cpu.fill(TypeParam(1));
    const auto rhs_gpu = rhs_cpu.toGpu();
    DenseMatrix<TypeParam, Device::GPU> solution_gpu(2, 1);
    solution_gpu.fill(TypeParam{0});
    IterativeSolverWorkspace<TypeParam> workspace;

    EXPECT_THROW(cg(matrix_gpu, rhs_gpu, solution_gpu, workspace), std::invalid_argument);

    const auto solution_cpu = solution_gpu.toCpu();
    EXPECT_EQ(solution_cpu(0, 0), TypeParam(0));
    EXPECT_EQ(solution_cpu(1, 0), TypeParam(0));
}

TYPED_TEST(IterativeSolverGpuTest, fixedAsyncRejectsEscapedMutableCsrStorage)
{
    auto matrix_gpu = poisson1d<TypeParam>(2).toGpu();
    static_cast<void>(matrix_gpu.rowOffsets());
    ASSERT_NO_THROW(matrix_gpu.validateStructure());
    DenseMatrix<TypeParam, Device::CPU> rhs_cpu(2, 1);
    rhs_cpu.fill(TypeParam(1));
    const auto rhs_gpu = rhs_cpu.toGpu();
    DenseMatrix<TypeParam, Device::GPU> solution_gpu(2, 1);
    solution_gpu.fill(TypeParam{0});
    IterativeSolverWorkspace<TypeParam> workspace;

    EXPECT_THROW(
        (void)cgFixedIterationsAsync(
            matrix_gpu, rhs_gpu, solution_gpu, 1, workspace),
        std::logic_error);
}

TYPED_TEST(IterativeSolverGpuTest, asyncCsrCopyRequiresCompletionOnItsCopyStream)
{
    auto source = CSRMatrix<TypeParam, Device::CPU>::pinned(2, 2, 2);
    source.rowOffsets()[0] = 0;
    source.rowOffsets()[1] = 1;
    source.rowOffsets()[2] = 2;
    source.colIndices()[0] = 0;
    source.colIndices()[1] = 1;
    source.values()[0] = TypeParam(2);
    source.values()[1] = TypeParam(4);
    CSRMatrix<TypeParam, Device::GPU> output(2, 2, 2);
    cudaStream_t copy_stream = nullptr;
    cudaStream_t other_stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaStreamCreateWithFlags(&other_stream, cudaStreamNonBlocking), cudaSuccess);

    source.copyToGpuAsync(output, copy_stream);
    EXPECT_THROW(output.validateStructure(other_stream), std::logic_error);
    ASSERT_EQ(cudaStreamSynchronize(copy_stream), cudaSuccess);
    const auto& const_output = output;
    Index copied_offsets[3] = {};
    Index copied_columns[2] = {};
    TypeParam copied_values[2] = {};
    ASSERT_EQ(cudaMemcpy(
        copied_offsets, const_output.rowOffsets(), sizeof(copied_offsets),
        cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(
        copied_columns, const_output.colIndices(), sizeof(copied_columns),
        cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(
        copied_values, const_output.values(), sizeof(copied_values),
        cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(copied_offsets[0], 0);
    EXPECT_EQ(copied_offsets[1], 1);
    EXPECT_EQ(copied_offsets[2], 2);
    EXPECT_EQ(copied_columns[0], 0);
    EXPECT_EQ(copied_columns[1], 1);
    EXPECT_EQ(copied_values[0], TypeParam(2));
    EXPECT_EQ(copied_values[1], TypeParam(4));
    EXPECT_NO_THROW(output.validateStructure(copy_stream));

    DenseMatrix<TypeParam, Device::CPU> rhs_cpu(2, 1);
    rhs_cpu(0, 0) = TypeParam(2);
    rhs_cpu(1, 0) = TypeParam(4);
    const auto rhs_gpu = rhs_cpu.toGpu();
    DenseMatrix<TypeParam, Device::GPU> solution_gpu(2, 1);
    solution_gpu.fill(TypeParam{0});
    IterativeSolverWorkspace<TypeParam> workspace;
    auto state = pcgFixedIterationsAsync(
        output, rhs_gpu, solution_gpu, 1, workspace, copy_stream);
    ASSERT_EQ(cudaStreamSynchronize(copy_stream), cudaSuccess);
    EXPECT_TRUE(finalizeIterativeSolverReport(state).converged);
    EXPECT_NO_THROW(state.closeAsyncAllocation());
    EXPECT_NO_THROW(workspace.closeAsyncAllocation());
    ASSERT_EQ(cudaStreamSynchronize(copy_stream), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(other_stream), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(copy_stream), cudaSuccess);
}

#ifdef PLAMATRIX_ITERATIVE_SOLVER_TEST_HOOKS
TYPED_TEST(IterativeSolverGpuTest, workspaceRecoversAfterPartialAllocationFailure)
{
    const auto small_matrix_gpu = poisson1d<TypeParam>(4).toGpu();
    const auto matrix_gpu = poisson1d<TypeParam>(8).toGpu();
    DenseMatrix<TypeParam, Device::CPU> small_rhs_cpu(4, 1);
    small_rhs_cpu.fill(TypeParam(1));
    const auto small_rhs_gpu = small_rhs_cpu.toGpu();
    DenseMatrix<TypeParam, Device::CPU> rhs_cpu(8, 1);
    rhs_cpu.fill(TypeParam(1));
    const auto rhs_gpu = rhs_cpu.toGpu();
    DenseMatrix<TypeParam, Device::GPU> solution_gpu(8, 1);
    solution_gpu.fill(TypeParam{0});
    IterativeSolverWorkspace<TypeParam> workspace;

    DenseMatrix<TypeParam, Device::GPU> small_solution_gpu(4, 1);
    small_solution_gpu.fill(TypeParam{0});
    ASSERT_TRUE(pcg(
        small_matrix_gpu, small_rhs_gpu, small_solution_gpu, workspace).converged);
    ASSERT_EQ(workspace.capacitySize(), 4);

    iterative_solver_detail::setForcedWorkspaceAllocationFailureAfter(2);
    EXPECT_THROW(pcg(matrix_gpu, rhs_gpu, solution_gpu, workspace), std::runtime_error);
    iterative_solver_detail::setForcedWorkspaceAllocationFailureAfter(-1);
    EXPECT_EQ(workspace.capacitySize(), 4);

    solution_gpu.fill(TypeParam{0});
    EXPECT_TRUE(pcg(matrix_gpu, rhs_gpu, solution_gpu, workspace).converged);
    EXPECT_EQ(workspace.capacitySize(), 8);
    EXPECT_NO_THROW(workspace.closeAsyncAllocation());
}
#endif

TYPED_TEST(IterativeSolverGpuTest, fixedIterationAsyncRequiresCompletionBeforeFinalization)
{
    const auto matrix_cpu = poisson1d<TypeParam>(16);
    DenseMatrix<TypeParam, Device::CPU> rhs_cpu(16, 1);
    rhs_cpu.fill(TypeParam(1));
    const auto matrix_gpu = matrix_cpu.toGpu();
    const auto rhs_gpu = rhs_cpu.toGpu();
    DenseMatrix<TypeParam, Device::GPU> solution_gpu(16, 1);
    solution_gpu.fill(TypeParam{0});
    IterativeSolverWorkspace<TypeParam> workspace;
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);

    // Initialize CUDA library handles and reusable storage before delaying completion.
    auto warmup_state = cgFixedIterationsAsync(
        matrix_gpu, rhs_gpu, solution_gpu, 0, workspace, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    EXPECT_NO_THROW(finalizeIterativeSolverReport(warmup_state));
    EXPECT_NO_THROW(warmup_state.closeAsyncAllocation());
    solution_gpu.fill(TypeParam{0});

    cudaStream_t gate_stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&gate_stream, cudaStreamNonBlocking), cudaSuccess);
    DenseMatrix<int, Device::GPU> gate_flag(1, 1);
    gate_flag.fill(0);

    iterative_solver_detail::setFixedSolverCompletionGate(gate_flag.data());
    auto state = cgFixedIterationsAsync(
        matrix_gpu, rhs_gpu, solution_gpu, 7, workspace, stream);
    iterative_solver_detail::setFixedSolverCompletionGate(nullptr);
    IterativeSolverOptions options;
    options.maxIterations = 7;
    options.relativeTolerance = 0.0;
    options.absoluteTolerance = 0.0;
    EXPECT_THROW(finalizeIterativeSolverReport(state, options), std::logic_error);
    EXPECT_THROW(workspace.closeAsyncAllocation(), std::logic_error);

    ASSERT_EQ(cudaMemsetAsync(gate_flag.data(), 1, sizeof(int), gate_stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(gate_stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    const auto report = finalizeIterativeSolverReport(state, options);
    EXPECT_EQ(report.iterations, 7);
    EXPECT_GT(report.initialResidual, report.finalResidual);

    options.requireConvergence = true;
    EXPECT_THROW(finalizeIterativeSolverReport(state, options), std::runtime_error);
    EXPECT_NO_THROW(state.closeAsyncAllocation());
    EXPECT_NO_THROW(workspace.closeAsyncAllocation());
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(gate_stream), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TYPED_TEST(IterativeSolverGpuTest, fixedPcgSubmitsExactIterationsAndSupportsZeroIterations)
{
    const auto matrix_cpu = poisson1d<TypeParam>(8);
    DenseMatrix<TypeParam, Device::CPU> rhs_cpu(8, 1);
    rhs_cpu.fill(TypeParam(1));
    const auto matrix_gpu = matrix_cpu.toGpu();
    const auto rhs_gpu = rhs_cpu.toGpu();
    DenseMatrix<TypeParam, Device::GPU> solution_gpu(8, 1);
    solution_gpu.fill(TypeParam{0});
    IterativeSolverWorkspace<TypeParam> workspace;

    auto zero_state = pcgFixedIterationsAsync(
        matrix_gpu, rhs_gpu, solution_gpu, 0, workspace, nullptr);
    ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
    const auto zero_report = finalizeIterativeSolverReport(zero_state, {});
    EXPECT_EQ(zero_report.iterations, 0);
    EXPECT_EQ(zero_report.initialResidual, zero_report.finalResidual);
    EXPECT_NO_THROW(zero_state.closeAsyncAllocation());

    auto state = pcgFixedIterationsAsync(
        matrix_gpu, rhs_gpu, solution_gpu, 4, workspace, nullptr);
    ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
    const auto report = finalizeIterativeSolverReport(state, {});
    EXPECT_EQ(report.iterations, 4);
    EXPECT_TRUE(report.converged);
    EXPECT_TRUE(std::isfinite(report.finalResidual));
    const auto solution_cpu = solution_gpu.toCpu();
    for (Index row = 0; row < 8; ++row)
    {
        const TypeParam expected = TypeParam((row + 1) * (8 - row)) / TypeParam(2);
        EXPECT_NEAR(solution_cpu(row, 0), expected,
                    solverTolerance<TypeParam>() * static_cast<double>(expected));
    }
    EXPECT_NO_THROW(state.closeAsyncAllocation());
    EXPECT_NO_THROW(workspace.closeAsyncAllocation());
    ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
}

#else

TEST(IterativeSolverNoCuda, gpuSurfaceCompilesAndReportsUnavailableBackend)
{
    CSRMatrix<float, Device::GPU> matrix(0, 0, 0);
    DenseMatrix<float, Device::GPU> rhs;
    DenseMatrix<float, Device::GPU> solution;
    IterativeSolverWorkspace<float> workspace;

    EXPECT_THROW(cg(matrix, rhs, solution, workspace), std::runtime_error);
    EXPECT_THROW(pcg(matrix, rhs, solution, workspace), std::runtime_error);
    EXPECT_THROW(blockPcg(matrix, rhs, solution, solution, 1, workspace),
                 std::runtime_error);
    EXPECT_THROW(cgFixedIterationsAsync(matrix, rhs, solution, 0, workspace),
                 std::runtime_error);
    EXPECT_THROW(pcgFixedIterationsAsync(matrix, rhs, solution, 0, workspace),
                 std::runtime_error);

    AsyncIterativeSolverState state;
    EXPECT_THROW(finalizeIterativeSolverReport(state), std::runtime_error);
    EXPECT_NO_THROW(state.closeAsyncAllocation());
    EXPECT_NO_THROW(workspace.closeAsyncAllocation());
}

#endif
