#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include <plamatrix/sparse/conjugate_gradient.h>

namespace
{
    using Sparse = plamatrix::SparseMatrix<double>;
    using Vector = plamatrix::VectorXd;
    static_assert(std::is_same_v<typename Sparse::Scalar, double>);
    static_assert(std::is_same_v<typename Sparse::StorageIndex, int>);
    static_assert(Sparse::IsRowMajor == 0);
    static_assert(plamatrix::SparseMatrix<double, plamatrix::RowMajor>::IsRowMajor == 1);

    Sparse poissonMatrix()
    {
        Sparse matrix(3, 3);
        const std::vector<plamatrix::Triplet<double>> triplets{
            {0, 0, 2.0}, {0, 1, -1.0}, {1, 0, -1.0}, {1, 1, 1.0}, {1, 1, 1.0}, {1, 2, -1.0}, {2, 1, -1.0}, {2, 2, 2.0}};
        matrix.setFromTriplets(triplets.begin(), triplets.end());
        return matrix;
    }
} // namespace

TEST(EigenStyleSparse, InnerIteratorFollowsStorageOrderAndTracksMutations)
{
    plamatrix::SparseMatrix<double, plamatrix::RowMajor> rows(3, 4);
    Sparse columns(3, 4);
    const std::vector<plamatrix::Triplet<double>> triplets{{2, 1, 5.0}, {0, 3, 7.0}, {2, 0, 2.0}};
    rows.setFromTriplets(triplets.begin(), triplets.end());
    columns.setFromTriplets(triplets.begin(), triplets.end());
    using Coordinate = std::pair<plamatrix::Index, plamatrix::Index>;
    std::vector<Coordinate> row_order;
    std::vector<Coordinate> column_order;
    for (plamatrix::Index outer = 0; outer < rows.outerSize(); ++outer)
    {
        for (decltype(rows)::InnerIterator entry(rows, outer); entry; ++entry)
        {
            row_order.emplace_back(entry.row(), entry.col());
            EXPECT_EQ(entry.outer(), entry.row());
            EXPECT_EQ(entry.index(), entry.col());
            EXPECT_DOUBLE_EQ(entry.value(), rows.coeff(entry.row(), entry.col()));
        }
    }
    for (plamatrix::Index outer = 0; outer < columns.outerSize(); ++outer)
    {
        for (Sparse::InnerIterator entry(columns, outer); entry; ++entry)
        {
            column_order.emplace_back(entry.row(), entry.col());
            EXPECT_EQ(entry.outer(), entry.col());
            EXPECT_EQ(entry.index(), entry.row());
        }
    }
    EXPECT_EQ(row_order, (std::vector<Coordinate>{{0, 3}, {2, 0}, {2, 1}}));
    EXPECT_EQ(column_order, (std::vector<Coordinate>{{2, 0}, {2, 1}, {0, 3}}));
    EXPECT_FALSE(Sparse::InnerIterator(columns, 2));
    EXPECT_THROW(Sparse::InnerIterator(columns, 4), std::out_of_range);
    EXPECT_THROW(Sparse::InnerIterator(columns, -1), std::out_of_range);

    double& value = columns.coeffRef(2, 0);
    const Sparse::InnerIterator entry(columns, 0);
    value = 42.0;
    EXPECT_DOUBLE_EQ(entry.value(), 42.0);
    const Sparse copy = columns;
    columns.insert(1, 0) = 9.0;
    EXPECT_EQ(Sparse::InnerIterator(columns, 0).row(), 1);
    EXPECT_EQ(Sparse::InnerIterator(copy, 0).row(), 2);
    Sparse assigned;
    assigned = columns;
    EXPECT_DOUBLE_EQ(Sparse::InnerIterator(assigned, 0).value(), 9.0);
    Sparse moved = std::move(assigned);
    EXPECT_DOUBLE_EQ(Sparse::InnerIterator(moved, 0).value(), 9.0);
    EXPECT_EQ(assigned.rows(), 0);
    EXPECT_EQ(assigned.nonZeros(), 0);
    assigned.resize(1, 1);
    assigned.insert(0, 0) = 11.0;
    EXPECT_DOUBLE_EQ(Sparse::InnerIterator(assigned, 0).value(), 11.0);
    assigned = std::move(moved);
    EXPECT_DOUBLE_EQ(Sparse::InnerIterator(assigned, 0).value(), 9.0);
    EXPECT_EQ(moved.rows(), 0);
    EXPECT_EQ(moved.nonZeros(), 0);
    assigned.swap(columns);
    EXPECT_EQ(assigned.rows(), 3);
    EXPECT_DOUBLE_EQ(Sparse::InnerIterator(columns, 0).value(), 9.0);
    columns.setZero();
    EXPECT_FALSE(Sparse::InnerIterator(columns, 0));
    columns.setFromTriplets(triplets.begin(), triplets.end());
    EXPECT_DOUBLE_EQ(Sparse::InnerIterator(columns, 0).value(), 2.0);
}

TEST(EigenStyleSparse, TripletsCombineDuplicatesAndPreserveOriginalOnFailure)
{
    Sparse matrix = poissonMatrix();
    EXPECT_EQ(matrix.rows(), 3);
    EXPECT_EQ(matrix.cols(), 3);
    EXPECT_EQ(matrix.nonZeros(), 7);
    EXPECT_TRUE(matrix.isCompressed());
    EXPECT_DOUBLE_EQ(matrix.coeff(1, 1), 2.0);
    EXPECT_DOUBLE_EQ(matrix.coeff(0, 2), 0.0);
    EXPECT_THROW(matrix.coeff(3, 0), std::out_of_range);

    const std::vector<plamatrix::Triplet<double>> invalid{{0, 0, 1.0}, {3, 0, 1.0}};
    EXPECT_THROW(matrix.setFromTriplets(invalid.begin(), invalid.end()), std::out_of_range);
    EXPECT_EQ(matrix.nonZeros(), 7);

    matrix.makeCompressed();
    EXPECT_TRUE(matrix.isCompressed());
    auto csr = plamatrix::internal::SparseAccess::csrSnapshot(matrix);
    EXPECT_EQ(csr.nnz(), 7);
    matrix.coeffRef(0, 2) = 4.0;
    csr = plamatrix::internal::SparseAccess::csrSnapshot(matrix);
    EXPECT_DOUBLE_EQ(csr.values()[2], 4.0);
    double& escaped = matrix.coeffRef(0, 2);
    matrix.makeCompressed();
    escaped = 6.0;
    csr = plamatrix::internal::SparseAccess::csrSnapshot(matrix);
    EXPECT_DOUBLE_EQ(csr.values()[2], 6.0);
    EXPECT_THROW(matrix.insert(0, 2), std::invalid_argument);

    Sparse copy = matrix;
    copy.coeffRef(0, 2) = 5.0;
    EXPECT_DOUBLE_EQ(matrix.coeff(0, 2), 6.0);
    EXPECT_DOUBLE_EQ(copy.coeff(0, 2), 5.0);
}

TEST(EigenStyleSparse, CompressedPointersAndOrderedInsertionMatchEigenContract)
{
    Sparse matrix(3, 3);
    matrix.reserve(5);
    matrix.startVec(0);
    matrix.insertBack(0, 0) = 2.0;
    matrix.insertBack(2, 0) = 3.0;
    matrix.startVec(1);
    matrix.insertBack(1, 1) = 4.0;
    matrix.startVec(2);
    matrix.insertBackByOuterInner(2, 0) = 5.0;
    matrix.insertBackByOuterInner(2, 2) = 6.0;
    matrix.finalize();

    ASSERT_TRUE(matrix.isCompressed());
    ASSERT_EQ(matrix.nonZeros(), 5);
    EXPECT_EQ(matrix.outerIndexPtr()[0], 0);
    EXPECT_EQ(matrix.outerIndexPtr()[1], 2);
    EXPECT_EQ(matrix.outerIndexPtr()[2], 3);
    EXPECT_EQ(matrix.outerIndexPtr()[3], 5);
    EXPECT_EQ(matrix.innerIndexPtr()[0], 0);
    EXPECT_EQ(matrix.innerIndexPtr()[1], 2);
    EXPECT_DOUBLE_EQ(matrix.valuePtr()[1], 3.0);

    matrix.valuePtr()[1] = 7.0;
    EXPECT_DOUBLE_EQ(matrix.coeff(2, 0), 7.0);
    matrix.uncompress();
    EXPECT_FALSE(matrix.isCompressed());
    matrix.makeCompressed();
    EXPECT_TRUE(matrix.isCompressed());
    EXPECT_DOUBLE_EQ(matrix.valuePtr()[1], 7.0);
}

TEST(EigenStyleSparse, SparseMapIsZeroCopyForCompressedStorage)
{
    using MappedSparse = plamatrix::Map<plamatrix::SparseMatrix<double>>;
    int outer[] = {0, 2, 3};
    int inner[] = {0, 1, 1};
    double values[] = {2.0, 3.0, 4.0};
    MappedSparse mapped(2, 2, 3, outer, inner, values);

    static_assert(std::is_base_of_v<plamatrix::SparseMatrixBase<MappedSparse>, MappedSparse>);
    EXPECT_DOUBLE_EQ(mapped.coeff(1, 0), 3.0);
    mapped.coeffRef(1, 0) = 8.0;
    EXPECT_DOUBLE_EQ(values[1], 8.0);
    mapped.valuePtr()[2] = 9.0;
    EXPECT_DOUBLE_EQ(mapped.coeff(1, 1), 9.0);

    using ConstMappedSparse = plamatrix::Map<const plamatrix::SparseMatrix<double>>;
    ConstMappedSparse read_only(2, 2, 3, outer, inner, values);
    EXPECT_DOUBLE_EQ(read_only.coeff(1, 0), 8.0);

    int invalid_inner[] = {1, 0, 1};
    EXPECT_THROW((MappedSparse(2, 2, 3, outer, invalid_inner, values)), std::invalid_argument);
    int invalid_outer[] = {0, 3, 2};
    EXPECT_THROW((MappedSparse(2, 2, 3, invalid_outer, inner, values)), std::invalid_argument);
}

TEST(EigenStyleSparse, FullMatrixSolveMatchesEigenStyleCall)
{
    const Sparse matrix = poissonMatrix();
    Vector rhs(3);
    rhs << 1.0, 1.0, 1.0;

    plamatrix::ConjugateGradient<Sparse, plamatrix::Lower | plamatrix::Upper> solver;
    solver.setTolerance(1.0e-12).setMaxIterations(20).compute(matrix);
    const Vector solution = solver.solve(rhs);

    EXPECT_EQ(solver.info(), plamatrix::Success);
    EXPECT_LE(solver.error(), solver.tolerance());
    EXPECT_GT(solver.iterations(), 0);
    EXPECT_NEAR(solution(0), 1.5, 1.0e-10);
    EXPECT_NEAR(solution(1), 2.0, 1.0e-10);
    EXPECT_NEAR(solution(2), 1.5, 1.0e-10);

    const Vector exact = solver.solveWithGuess(rhs, solution);
    EXPECT_EQ(solver.info(), plamatrix::Success);
    EXPECT_EQ(solver.iterations(), 0);
    EXPECT_NEAR(exact(1), 2.0, 1.0e-10);
}

TEST(EigenStyleSparse, SparseDenseProductUsesMatrixSyntax)
{
    const Sparse matrix = poissonMatrix();
    Vector input(3);
    input << 1.5, 2.0, 1.5;
    const Vector result = matrix * input;
    for (plamatrix::Index row = 0; row < 3; ++row)
    {
        EXPECT_NEAR(result(row), 1.0, 1.0e-12);
    }

    plamatrix::Matrix<double, plamatrix::Dynamic, 2> right(3, 2);
    right.col(0) = input;
    right.col(1) = input;
    const auto matrix_result = matrix * right;
    EXPECT_NEAR(matrix_result(1, 1), 1.0, 1.0e-12);
    EXPECT_THROW(matrix * Vector::Zero(2), std::invalid_argument);
}

TEST(EigenStyleSparse, DefaultLowerTriangleIgnoresUpperAndMirrorsLower)
{
    Sparse matrix(2, 2);
    matrix.coeffRef(0, 0) = 4.0;
    matrix.coeffRef(1, 0) = 1.0;
    matrix.coeffRef(0, 1) = 2.0;
    matrix.coeffRef(1, 1) = 3.0;
    Vector rhs(2);
    rhs << 6.0, 7.0;

    plamatrix::ConjugateGradient<Sparse> lower(matrix);
    const Vector result = lower.solve(rhs);
    EXPECT_EQ(lower.info(), plamatrix::Success);
    EXPECT_NEAR(result(0), 1.0, 1.0e-10);
    EXPECT_NEAR(result(1), 2.0, 1.0e-10);

    plamatrix::ConjugateGradient<Sparse, plamatrix::Upper> upper;
    upper.compute(matrix);
    EXPECT_NE(upper.solve(rhs)(0), result(0));
}

TEST(EigenStyleSparse, IdentityPreconditionerAndNonConvergence)
{
    const Sparse matrix = poissonMatrix();
    Vector rhs(3);
    rhs << 1.0, 1.0, 1.0;
    plamatrix::ConjugateGradient<Sparse, plamatrix::Lower | plamatrix::Upper, plamatrix::IdentityPreconditioner> solver(
        matrix);
    solver.setMaxIterations(0);
    const Vector unfinished = solver.solve(rhs);
    EXPECT_EQ(solver.info(), plamatrix::NoConvergence);
    EXPECT_EQ(solver.iterations(), 0);
    EXPECT_DOUBLE_EQ(unfinished(0), 0.0);

    solver.setMaxIterations(20);
    const Vector solution = solver.solve(rhs);
    EXPECT_EQ(solver.info(), plamatrix::Success);
    EXPECT_NEAR(solution(1), 2.0, 1.0e-10);
}

TEST(EigenStyleSparse, ComputeOwnsSnapshotUntilRecomputed)
{
    Sparse matrix(2, 2);
    matrix.coeffRef(0, 0) = 2.0;
    matrix.coeffRef(1, 1) = 2.0;
    Vector rhs(2);
    rhs << 2.0, 4.0;

    plamatrix::ConjugateGradient<Sparse> solver(matrix);
    matrix.coeffRef(0, 0) = 4.0;
    EXPECT_NEAR(solver.solve(rhs)(0), 1.0, 1.0e-10);
    solver.compute(matrix);
    EXPECT_NEAR(solver.solve(rhs)(0), 0.5, 1.0e-10);
}

TEST(EigenStyleSparse, RejectsInvalidSizesAndOptions)
{
    Sparse nonsquare(2, 3);
    plamatrix::ConjugateGradient<Sparse> solver;
    EXPECT_THROW(solver.compute(nonsquare), std::invalid_argument);
    EXPECT_EQ(solver.info(), plamatrix::InvalidInput);
    EXPECT_THROW(solver.setMaxIterations(-1), std::invalid_argument);
    EXPECT_THROW(solver.setTolerance(std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);

    solver.compute(poissonMatrix());
    EXPECT_THROW(solver.solve(Vector::Zero(2)), std::invalid_argument);
    EXPECT_EQ(solver.info(), plamatrix::InvalidInput);

    plamatrix::internal::ScopedExecutionPolicy gpu_required(plamatrix::internal::ExecutionPolicy::GpuRequired);
    solver.compute(poissonMatrix());
    try
    {
        const Vector accelerated = solver.solve(Vector::Ones(3));
        EXPECT_NE(solver.backend(), plamatrix::internal::Backend::Cpu);
        EXPECT_EQ(accelerated.rows(), 3);
    }
    catch (const plamatrix::internal::Error& error)
    {
        EXPECT_NE(error.code(), plamatrix::internal::ErrorCode::InvalidArgument);
    }
}

TEST(EigenStyleSparse, FloatSolverUsesSamePublicSyntax)
{
    plamatrix::SparseMatrix<float> matrix(2, 2);
    const std::vector<plamatrix::Triplet<float>> entries{{0, 0, 2.0F}, {1, 1, 4.0F}};
    matrix.setFromTriplets(entries.begin(), entries.end());
    plamatrix::VectorXf rhs(2);
    rhs << 2.0F, 8.0F;
    plamatrix::ConjugateGradient<plamatrix::SparseMatrix<float>> solver(matrix);
    const plamatrix::VectorXf solution = solver.solve(rhs);
    EXPECT_EQ(solver.info(), plamatrix::Success);
    EXPECT_NEAR(solution(0), 1.0F, 1.0e-5F);
    EXPECT_NEAR(solution(1), 2.0F, 1.0e-5F);
}

TEST(EigenStyleSparse, EmptyFullMatrixCanBeComputed)
{
    const Sparse matrix(0, 0);
    plamatrix::ConjugateGradient<Sparse, plamatrix::Lower | plamatrix::Upper> solver(matrix);
    const Vector result = solver.solve(Vector::Zero(0));
    EXPECT_EQ(result.rows(), 0);
    EXPECT_EQ(solver.info(), plamatrix::Success);
    EXPECT_EQ(solver.iterations(), 0);
}
