#include <gtest/gtest.h>

#include <plamatrix/opencl/iterative_solver.h>
#include <plamatrix/opencl/runtime.h>
#include <plamatrix/sparse/sparse_ops.h>

#include <cmath>
#include <type_traits>
#include <vector>

namespace
{

template <typename Scalar>
plamatrix::CSRMatrix<Scalar, plamatrix::Device::CPU> poisson1d(plamatrix::Index size)
{
    std::vector<plamatrix::Index> rows;
    std::vector<plamatrix::Index> columns;
    std::vector<Scalar> values;
    for (plamatrix::Index row = 0; row < size; ++row)
    {
        if (row > 0)
        {
            rows.push_back(row);
            columns.push_back(row - 1);
            values.push_back(Scalar{-1});
        }
        rows.push_back(row);
        columns.push_back(row);
        values.push_back(Scalar{2});
        if (row + 1 < size)
        {
            rows.push_back(row);
            columns.push_back(row + 1);
            values.push_back(Scalar{-1});
        }
    }
    return plamatrix::cooToCsr(size, size, rows, columns, values);
}

#ifdef PLAMATRIX_WITH_OPENCL

template <typename Scalar>
class OpenClPcgTest : public ::testing::Test
{
};

using OpenClScalars = ::testing::Types<
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
TYPED_TEST_SUITE(OpenClPcgTest, OpenClScalars);

TYPED_TEST(OpenClPcgTest, SolvesCpuOwnedPoissonSystemOnSelectedGpu)
{
    if (!plamatrix::opencl::hasUsableOpenClDevice())
    {
        GTEST_SKIP() << "No usable OpenCL GPU";
    }
    auto& runtime = plamatrix::opencl::OpenClRuntime::instance();
    if constexpr (std::is_same_v<TypeParam, double>)
    {
        if (!runtime.supportsFp64())
        {
            GTEST_SKIP() << "Selected OpenCL GPU does not support double precision";
        }
    }

    const auto matrix = poisson1d<TypeParam>(64);
    plamatrix::DenseMatrix<TypeParam, plamatrix::Device::CPU> expected(64, 1);
    for (plamatrix::Index row = 0; row < expected.rows(); ++row)
    {
        expected(row, 0) = static_cast<TypeParam>(row + 1);
    }
    const auto rhs = plamatrix::spmv(matrix, expected);
    plamatrix::DenseMatrix<TypeParam, plamatrix::Device::CPU> solution(64, 1);
    solution.fill(TypeParam{0});
    plamatrix::IterativeSolverOptions options;
    options.relativeTolerance = std::is_same_v<TypeParam, float> ? 2.0e-5 : 1.0e-11;

    const auto report = plamatrix::opencl::pcg(matrix, rhs, solution, options);

    EXPECT_TRUE(report.converged);
    EXPECT_GT(report.iterations, 0);
    EXPECT_LE(report.finalResidual,
              options.relativeTolerance * report.initialResidual * 1.01);
    const double tolerance = std::is_same_v<TypeParam, float> ? 2.0e-3 : 1.0e-8;
    for (plamatrix::Index row = 0; row < solution.rows(); ++row)
    {
        EXPECT_NEAR(solution(row, 0), expected(row, 0), tolerance * (row + 1));
    }
}

TEST(OpenClPcgTest, ReportsNonConvergenceAndRejectsInvalidDiagonal)
{
    if (!plamatrix::opencl::hasUsableOpenClDevice())
    {
        GTEST_SKIP() << "No usable OpenCL GPU";
    }
    const auto matrix = poisson1d<float>(32);
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> rhs(32, 1);
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> solution(32, 1);
    rhs.fill(1.0F);
    solution.fill(0.0F);
    plamatrix::IterativeSolverOptions options;
    options.maxIterations = 1;
    options.relativeTolerance = 0.0;
    const auto report = plamatrix::opencl::pcg(matrix, rhs, solution, options);
    EXPECT_FALSE(report.converged);
    EXPECT_EQ(report.iterations, 1);

    const auto invalid = plamatrix::cooToCsr(
        2, 2, std::vector<plamatrix::Index>{0, 1},
        std::vector<plamatrix::Index>{0, 1}, std::vector<float>{1.0F, 0.0F});
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> small_rhs(2, 1);
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> small_solution(2, 1);
    small_rhs.fill(1.0F);
    small_solution.fill(0.0F);
    EXPECT_THROW(plamatrix::opencl::pcg(invalid, small_rhs, small_solution),
                 std::runtime_error);
}

#else

TEST(OpenClPcgTest, DisabledBuildThrowsClearError)
{
    const auto matrix = poisson1d<float>(2);
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> rhs(2, 1);
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> solution(2, 1);
    EXPECT_THROW(plamatrix::opencl::pcg(matrix, rhs, solution), std::runtime_error);
}

#endif

} // namespace
