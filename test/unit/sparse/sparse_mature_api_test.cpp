#include <algorithm>
#include <set>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include <plamatrix/plamatrix.h>

namespace
{
    using Sparse = plamatrix::SparseMatrix<double>;
    using Vector = plamatrix::VectorXd;

    Sparse tridiagonal(plamatrix::Index size, double diagonal = 4.0, double off_diagonal = -1.0)
    {
        Sparse matrix(size, size);
        std::vector<plamatrix::Triplet<double>> triplets;
        for (plamatrix::Index row = 0; row < size; ++row)
        {
            triplets.emplace_back(row, row, diagonal);
            if (row != 0)
            {
                triplets.emplace_back(row, row - 1, off_diagonal);
                triplets.emplace_back(row - 1, row, off_diagonal);
            }
        }
        matrix.setFromTriplets(triplets.begin(), triplets.end());
        return matrix;
    }

    template <typename Permutation> void expectPermutation(const Permutation& permutation, plamatrix::Index size)
    {
        std::set<plamatrix::Index> indices;
        for (plamatrix::Index index = 0; index < size; ++index)
        {
            indices.insert(permutation.indices()(index));
        }
        EXPECT_EQ(indices.size(), static_cast<std::size_t>(size));
        EXPECT_EQ(*indices.begin(), 0);
        EXPECT_EQ(*indices.rbegin(), size - 1);
    }
} // namespace

TEST(SparseMatureApi, DenseSparseViewIsLazyAndMaterializesWithoutZeros)
{
    plamatrix::MatrixXd dense = plamatrix::MatrixXd::Zero(3, 3);
    dense(0, 0) = 2.0;
    dense(2, 1) = -3.0;
    const auto view = dense.sparseView();
    using View = std::remove_cv_t<decltype(view)>;
    static_assert(std::is_base_of_v<plamatrix::SparseMatrixBase<View>, View>);
    EXPECT_EQ(view.nonZeros(), 2);

    dense(1, 2) = 4.0;
    EXPECT_EQ(view.nonZeros(), 3);
    Sparse sparse = view;
    EXPECT_EQ(sparse.nonZeros(), 3);
    EXPECT_DOUBLE_EQ(sparse.coeff(2, 1), -3.0);
    EXPECT_DOUBLE_EQ(sparse.transpose().coeff(1, 2), -3.0);

    sparse.prune(1.0, 2.5);
    EXPECT_EQ(sparse.nonZeros(), 2);
    EXPECT_DOUBLE_EQ(sparse.sum(), 1.0);

    plamatrix::SparseMatrix<double, plamatrix::RowMajor> row_major = sparse;
    const Sparse round_trip = row_major;
    EXPECT_DOUBLE_EQ(round_trip.coeff(1, 2), 4.0);
    EXPECT_EQ((round_trip + round_trip).nonZeros(), 2);
    EXPECT_EQ((round_trip.cwiseProduct(round_trip)).nonZeros(), 2);
    EXPECT_EQ((round_trip * round_trip.transpose()).rows(), 3);
}

TEST(SparseMatureApi, TripletsSupportCustomDuplicateCombination)
{
    Sparse matrix(2, 2);
    const std::vector<plamatrix::Triplet<double>> triplets{{0, 0, 2.0}, {0, 0, 5.0}, {1, 1, 3.0}};
    matrix.setFromTriplets(
        triplets.begin(), triplets.end(), [](double left, double right) { return std::max(left, right); });
    EXPECT_DOUBLE_EQ(matrix.coeff(0, 0), 5.0);
    EXPECT_EQ(matrix.nonZeros(), 2);
}

TEST(SparseMatureApi, NaturalAmdAndColamdProduceUsablePermutations)
{
    Sparse matrix(5, 5);
    std::vector<plamatrix::Triplet<double>> triplets;
    for (plamatrix::Index leaf = 1; leaf < 5; ++leaf)
    {
        triplets.emplace_back(0, leaf, 1.0);
        triplets.emplace_back(leaf, 0, 1.0);
        triplets.emplace_back(leaf, leaf, 2.0);
    }
    triplets.emplace_back(0, 0, 5.0);
    matrix.setFromTriplets(triplets.begin(), triplets.end());

    plamatrix::PermutationMatrix<plamatrix::Dynamic, plamatrix::Dynamic, int> natural;
    plamatrix::PermutationMatrix<plamatrix::Dynamic, plamatrix::Dynamic, int> amd;
    plamatrix::PermutationMatrix<plamatrix::Dynamic, plamatrix::Dynamic, int> colamd;
    plamatrix::NaturalOrdering<int>{}(matrix, natural);
    plamatrix::AMDOrdering<int>{}(matrix, amd);
    plamatrix::COLAMDOrdering<int>{}(matrix, colamd);
    expectPermutation(natural, 5);
    expectPermutation(amd, 5);
    expectPermutation(colamd, 5);
    EXPECT_EQ(natural.indices()(0), 0);
    EXPECT_NE(amd.indices()(0), 0);

    const Sparse reordered = amd * matrix * amd.inverse();
    EXPECT_EQ(reordered.nonZeros(), matrix.nonZeros());
}

TEST(SparseMatureApi, IncompletePreconditionersExposeEigenComputeAndSolveContract)
{
    const Sparse spd = tridiagonal(4);
    Vector expected(4);
    expected << 1.0, 2.0, -1.0, 3.0;
    const Vector rhs = spd * expected;

    plamatrix::DiagonalPreconditioner<double> diagonal(spd);
    EXPECT_EQ(diagonal.info(), plamatrix::Success);
    EXPECT_EQ(diagonal.solve(rhs).rows(), 4);

    plamatrix::IncompleteCholesky<double> ichol(spd);
    EXPECT_EQ(ichol.info(), plamatrix::Success);
    EXPECT_TRUE(ichol.solve(rhs).isApprox(expected, 1.0e-10));

    Sparse general(3, 3);
    const std::vector<plamatrix::Triplet<double>> entries{
        {0, 0, 4.0}, {0, 1, 1.0}, {1, 0, 2.0}, {1, 1, 3.0}, {1, 2, 1.0}, {2, 1, 1.0}, {2, 2, 2.0}};
    general.setFromTriplets(entries.begin(), entries.end());
    Vector general_expected(3);
    general_expected << 2.0, -1.0, 3.0;
    plamatrix::IncompleteLUT<double> ilut;
    ilut.setDroptol(0.0).setFillfactor(4).compute(general);
    EXPECT_EQ(ilut.info(), plamatrix::Success);
    EXPECT_TRUE(ilut.solve(general * general_expected).isApprox(general_expected, 1.0e-10));
}

TEST(SparseMatureApi, CgAcceptsIncompleteCholeskyAndMultipleRightHandSides)
{
    const Sparse matrix = tridiagonal(8);
    plamatrix::Matrix<double, plamatrix::Dynamic, 2> expected(8, 2);
    for (plamatrix::Index row = 0; row < expected.rows(); ++row)
    {
        expected(row, 0) = static_cast<double>(row + 1);
        expected(row, 1) = static_cast<double>(1 - row);
    }
    const auto rhs = matrix * expected;
    using Preconditioner = plamatrix::IncompleteCholesky<double>;
    plamatrix::ConjugateGradient<Sparse, plamatrix::Lower | plamatrix::Upper, Preconditioner> solver;
    solver.setTolerance(1.0e-12).setMaxIterations(50).compute(matrix);
    const auto actual = solver.solve(rhs);
    EXPECT_EQ(solver.info(), plamatrix::Success);
    EXPECT_TRUE(actual.isApprox(expected, 1.0e-9));
    EXPECT_EQ(solver.backend(), plamatrix::internal::Backend::Cpu);
}

TEST(SparseMatureApi, BiCgStabSolvesNonsymmetricSystemsWithIlut)
{
    Sparse matrix(4, 4);
    const std::vector<plamatrix::Triplet<double>> entries{{0, 0, 4.0},
                                                          {0, 1, -1.0},
                                                          {1, 0, 2.0},
                                                          {1, 1, 5.0},
                                                          {1, 2, 1.0},
                                                          {2, 1, -2.0},
                                                          {2, 2, 4.0},
                                                          {2, 3, 1.0},
                                                          {3, 2, 1.0},
                                                          {3, 3, 3.0}};
    matrix.setFromTriplets(entries.begin(), entries.end());
    Vector expected(4);
    expected << 1.0, -2.0, 3.0, 0.5;
    using Solver = plamatrix::BiCGSTAB<Sparse, plamatrix::IncompleteLUT<double>>;
    Solver solver;
    solver.preconditioner().setDroptol(0.0).setFillfactor(4);
    solver.setTolerance(1.0e-12).setMaxIterations(50).compute(matrix);
    const Vector actual = solver.solve(matrix * expected);
    EXPECT_EQ(solver.info(), plamatrix::Success);
    EXPECT_LE(solver.error(), solver.tolerance());
    EXPECT_TRUE(actual.isApprox(expected, 1.0e-9));
}

TEST(SparseMatureApi, DirectFactorsRemainSparseAndExposeAnalyzedPermutations)
{
    constexpr plamatrix::Index size = 64;
    const Sparse matrix = tridiagonal(size);
    plamatrix::SimplicialLLT<Sparse> llt(matrix);
    ASSERT_EQ(llt.info(), plamatrix::Success);
    EXPECT_LT(llt.factorNonZeros(), size * 4);
    expectPermutation(llt.permutationP(), size);

    plamatrix::SparseLU<Sparse> lu(matrix);
    ASSERT_EQ(lu.info(), plamatrix::Success);
    EXPECT_LT(lu.matrixL().nonZeros() + lu.matrixU().nonZeros(), size * 8);
    expectPermutation(lu.rowsPermutation(), size);
    expectPermutation(lu.colsPermutation(), size);

    plamatrix::SparseQR<Sparse, plamatrix::COLAMDOrdering<int>> qr(matrix);
    ASSERT_EQ(qr.info(), plamatrix::Success);
    expectPermutation(qr.colsPermutation(), size);
}
