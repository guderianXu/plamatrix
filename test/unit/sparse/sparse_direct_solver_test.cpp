#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include <plamatrix/plamatrix.h>

namespace
{
    using Sparse = plamatrix::SparseMatrix<double>;
    using Vector = plamatrix::VectorXd;
    using Matrix = plamatrix::MatrixXd;

    Sparse positiveDefiniteMatrix()
    {
        Sparse matrix(3, 3);
        const std::vector<plamatrix::Triplet<double>> triplets{
            {0, 0, 4.0}, {0, 1, 1.0}, {1, 0, 1.0}, {1, 1, 3.0}, {1, 2, 1.0}, {2, 1, 1.0}, {2, 2, 2.0}};
        matrix.setFromTriplets(triplets.begin(), triplets.end());
        return matrix;
    }

    void expectVectorNear(const Vector& actual, const Vector& expected, double tolerance = 1.0e-10)
    {
        ASSERT_EQ(actual.rows(), expected.rows());
        for (plamatrix::Index row = 0; row < actual.rows(); ++row)
        {
            EXPECT_NEAR(actual(row), expected(row), tolerance);
        }
    }
} // namespace

static_assert(std::is_base_of_v<plamatrix::SparseSolverBase<plamatrix::SparseLU<Sparse>>, plamatrix::SparseLU<Sparse>>);
static_assert(
    std::is_base_of_v<plamatrix::SparseSolverBase<
                          plamatrix::SparseQR<Sparse, plamatrix::COLAMDOrdering<typename Sparse::StorageIndex>>>,
                      plamatrix::SparseQR<Sparse, plamatrix::COLAMDOrdering<typename Sparse::StorageIndex>>>);

TEST(SparseDirectSolver, LuSupportsComputeAndMultipleRightHandSides)
{
    Sparse matrix(3, 3);
    const std::vector<plamatrix::Triplet<double>> triplets{{0, 0, 3.0},
                                                           {0, 1, 2.0},
                                                           {0, 2, -1.0},
                                                           {1, 0, 2.0},
                                                           {1, 1, -2.0},
                                                           {1, 2, 4.0},
                                                           {2, 0, -1.0},
                                                           {2, 1, 0.5},
                                                           {2, 2, -1.0}};
    matrix.setFromTriplets(triplets.begin(), triplets.end());
    Matrix expected(3, 2);
    expected << 1.0, 2.0, -2.0, 1.0, -2.0, 3.0;
    const Matrix rhs = matrix * expected;

    plamatrix::SparseLU<Sparse> solver;
    EXPECT_THROW(solver.solve(rhs), std::logic_error);
    solver.compute(matrix);
    EXPECT_EQ(solver.info(), plamatrix::Success);
    EXPECT_EQ(solver.rows(), 3);
    EXPECT_EQ(solver.cols(), 3);
    EXPECT_TRUE(solver.lastErrorMessage().empty());
    const Matrix actual = solver.solve(rhs);
    EXPECT_TRUE(actual.isApprox(expected, 1.0e-10));
    const Matrix expression_actual = solver.solve(rhs + Matrix::Zero(3, 2));
    EXPECT_TRUE(expression_actual.isApprox(expected, 1.0e-10));
}

TEST(SparseDirectSolver, FactorizeReusesOnlyTheAnalyzedPattern)
{
    Sparse matrix = positiveDefiniteMatrix();
    Vector expected(3);
    expected << 1.0, 2.0, 3.0;
    plamatrix::SparseLU<Sparse> solver;
    solver.analyzePattern(matrix);
    EXPECT_EQ(solver.info(), plamatrix::Success);
    EXPECT_THROW(solver.solve(expected), std::logic_error);

    matrix.coeffRef(0, 0) = 8.0;
    const Vector rhs = matrix * expected;
    solver.factorize(matrix);
    EXPECT_EQ(solver.info(), plamatrix::Success);
    expectVectorNear(solver.solve(rhs), expected);

    matrix.coeffRef(0, 2) = 0.25;
    solver.factorize(matrix);
    EXPECT_EQ(solver.info(), plamatrix::InvalidInput);
    EXPECT_FALSE(solver.lastErrorMessage().empty());
    EXPECT_THROW(solver.solve(rhs), std::logic_error);
}

TEST(SparseDirectSolver, LuReportsSingularAndInvalidInput)
{
    Sparse singular(2, 2);
    singular.coeffRef(0, 0) = 1.0;
    singular.coeffRef(0, 1) = 2.0;
    singular.coeffRef(1, 0) = 2.0;
    singular.coeffRef(1, 1) = 4.0;
    plamatrix::SparseLU<Sparse> solver(singular);
    EXPECT_EQ(solver.info(), plamatrix::NumericalIssue);

    Sparse rectangular(2, 3);
    solver.compute(rectangular);
    EXPECT_EQ(solver.info(), plamatrix::InvalidInput);
}

TEST(SparseDirectSolver, SimplicialLltAndLdltSolveAndExposeFactors)
{
    const Sparse matrix = positiveDefiniteMatrix();
    Vector expected(3);
    expected << 1.0, -2.0, 0.5;
    const Vector rhs = matrix * expected;

    plamatrix::SimplicialLLT<Sparse> llt(matrix);
    EXPECT_EQ(llt.info(), plamatrix::Success);
    expectVectorNear(llt.solve(rhs), expected);
    EXPECT_NEAR(llt.determinant(), 18.0, 1.0e-10);
    EXPECT_GT(llt.matrixL()(0, 0), 0.0);
    const Matrix expected_upper(llt.matrixL().transpose());
    EXPECT_TRUE(llt.matrixU().isApprox(expected_upper, 1.0e-12));

    plamatrix::SimplicialLDLT<Sparse> ldlt;
    ldlt.analyzePattern(matrix);
    ldlt.factorize(matrix);
    EXPECT_EQ(ldlt.info(), plamatrix::Success);
    expectVectorNear(ldlt.solve(rhs), expected);
    EXPECT_NEAR(ldlt.determinant(), 18.0, 1.0e-10);
    EXPECT_EQ(ldlt.vectorD().rows(), 3);
    EXPECT_DOUBLE_EQ(ldlt.matrixL()(0, 0), 1.0);
}

TEST(SparseDirectSolver, SimplicialSolversUseSelectedTriangleAndReportIndefiniteInput)
{
    Sparse lower(2, 2);
    lower.coeffRef(0, 0) = 4.0;
    lower.coeffRef(1, 0) = 1.0;
    lower.coeffRef(0, 1) = 99.0;
    lower.coeffRef(1, 1) = 3.0;
    Vector rhs(2);
    rhs << 6.0, 7.0;
    plamatrix::SimplicialLLT<Sparse> solver(lower);
    Vector expected(2);
    expected << 1.0, 2.0;
    expectVectorNear(solver.solve(rhs), expected);

    Sparse indefinite(2, 2);
    indefinite.coeffRef(0, 0) = 1.0;
    indefinite.coeffRef(1, 1) = -1.0;
    solver.compute(indefinite);
    EXPECT_EQ(solver.info(), plamatrix::NumericalIssue);
}

TEST(SparseDirectSolver, UpperTriangleAndTinyScalesRemainNumericallyValid)
{
    Sparse upper(2, 2);
    upper.coeffRef(0, 0) = 4.0e-20;
    upper.coeffRef(0, 1) = 1.0e-20;
    upper.coeffRef(1, 0) = 99.0;
    upper.coeffRef(1, 1) = 3.0e-20;
    Vector expected(2);
    expected << 1.0, 2.0;
    Vector rhs(2);
    rhs << 6.0e-20, 7.0e-20;

    plamatrix::SimplicialLDLT<Sparse, plamatrix::Upper> solver(upper);
    EXPECT_EQ(solver.info(), plamatrix::Success);
    expectVectorNear(solver.solve(rhs), expected, 1.0e-9);

    Sparse tiny_lu(2, 2);
    tiny_lu.coeffRef(0, 0) = 2.0e-20;
    tiny_lu.coeffRef(1, 1) = 3.0e-20;
    plamatrix::SparseLU<Sparse> lu(tiny_lu);
    EXPECT_EQ(lu.info(), plamatrix::Success);
    expectVectorNear(lu.solve(tiny_lu * expected), expected, 1.0e-9);
}

TEST(SparseDirectSolver, SparseQrSolvesTallSystemsAndReportsRank)
{
    Sparse matrix(4, 2);
    const std::vector<plamatrix::Triplet<double>> triplets{
        {0, 0, 1.0}, {1, 1, 1.0}, {2, 0, 1.0}, {2, 1, 1.0}, {3, 0, 2.0}, {3, 1, -1.0}};
    matrix.setFromTriplets(triplets.begin(), triplets.end());
    Vector expected(2);
    expected << 2.0, -3.0;
    const Vector rhs = matrix * expected;

    using Qr = plamatrix::SparseQR<Sparse, plamatrix::COLAMDOrdering<typename Sparse::StorageIndex>>;
    Qr solver(matrix);
    EXPECT_EQ(solver.info(), plamatrix::Success);
    EXPECT_EQ(solver.rank(), 2);
    EXPECT_EQ(solver.matrixR().rows(), 2);
    expectVectorNear(solver.solve(rhs), expected);
}

TEST(SparseDirectSolver, SparseQrHandlesRankDeficiencyAndThreshold)
{
    Sparse matrix(3, 2);
    matrix.coeffRef(0, 0) = 1.0;
    matrix.coeffRef(0, 1) = 2.0;
    matrix.coeffRef(1, 0) = 2.0;
    matrix.coeffRef(1, 1) = 4.0;
    matrix.coeffRef(2, 0) = 3.0;
    matrix.coeffRef(2, 1) = 6.0;
    Vector rhs(3);
    rhs << 5.0, 10.0, 15.0;

    plamatrix::SparseQR<Sparse, plamatrix::NaturalOrdering<int>> solver;
    solver.setPivotThreshold(1.0e-12);
    solver.compute(matrix);
    EXPECT_EQ(solver.info(), plamatrix::Success);
    EXPECT_EQ(solver.rank(), 1);
    EXPECT_TRUE((matrix * solver.solve(rhs)).isApprox(rhs, 1.0e-10));
    EXPECT_THROW(solver.setPivotThreshold(-1.0), std::invalid_argument);
}

TEST(SparseDirectSolver, ConjugateGradientSeparatesAnalysisAndFactorization)
{
    Sparse matrix = positiveDefiniteMatrix();
    Vector expected(3);
    expected << 1.0, 2.0, 3.0;
    plamatrix::ConjugateGradient<Sparse, plamatrix::Lower | plamatrix::Upper> solver;
    solver.analyzePattern(matrix);
    EXPECT_THROW(solver.solve(expected), std::logic_error);

    matrix.coeffRef(0, 0) = 8.0;
    const Vector rhs = matrix * expected;
    solver.factorize(matrix);
    solver.setTolerance(1.0e-12).setMaxIterations(20);
    expectVectorNear(solver.solve(rhs), expected, 1.0e-9);
}

TEST(SparseDirectSolver, EmptyAndFloatSystemsUseTheSameApi)
{
    Sparse empty(0, 0);
    plamatrix::SparseLU<Sparse> empty_lu(empty);
    EXPECT_EQ(empty_lu.info(), plamatrix::Success);
    EXPECT_EQ(empty_lu.solve(Vector::Zero(0)).rows(), 0);

    using SparseFloat = plamatrix::SparseMatrix<float>;
    SparseFloat matrix(2, 2);
    matrix.coeffRef(0, 0) = 2.0F;
    matrix.coeffRef(1, 1) = 4.0F;
    plamatrix::VectorXf rhs(2);
    rhs << 2.0F, 8.0F;
    plamatrix::SimplicialLLT<SparseFloat> solver(matrix);
    const plamatrix::VectorXf solution = solver.solve(rhs);
    EXPECT_EQ(solver.info(), plamatrix::Success);
    EXPECT_NEAR(solution(0), 1.0F, 1.0e-5F);
    EXPECT_NEAR(solution(1), 2.0F, 1.0e-5F);

    plamatrix::SparseMatrix<double, plamatrix::RowMajor> row_major(2, 2);
    row_major.coeffRef(0, 0) = 2.0;
    row_major.coeffRef(1, 1) = 4.0;
    plamatrix::SparseLU<decltype(row_major)> row_major_solver(row_major);
    Vector row_major_rhs(2);
    row_major_rhs << 2.0, 8.0;
    Vector row_major_expected(2);
    row_major_expected << 1.0, 2.0;
    expectVectorNear(row_major_solver.solve(row_major_rhs), row_major_expected);
}

TEST(SparseDirectSolver, RequiredGpuPolicyRejectsCpuOnlyFactorizations)
{
    const Sparse matrix = positiveDefiniteMatrix();
    plamatrix::SparseLU<Sparse> solver;
    plamatrix::internal::ScopedExecutionPolicy gpu_required(plamatrix::internal::ExecutionPolicy::GpuRequired);
    EXPECT_THROW(solver.compute(matrix), plamatrix::internal::Error);
    EXPECT_THROW((plamatrix::SimplicialLLT<Sparse>(matrix)), plamatrix::internal::Error);
    EXPECT_THROW((plamatrix::SparseQR<Sparse, plamatrix::COLAMDOrdering<int>>(matrix)), plamatrix::internal::Error);
}
