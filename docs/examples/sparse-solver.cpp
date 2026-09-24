// Eigen-style sparse assembly, iteration and CG. Link with plamatrix::plamatrix.

#include <iostream>
#include <vector>

#include <plamatrix/sparse/conjugate_gradient.h>

int main()
{
    constexpr plamatrix::Index count = 500;
    using SparseMatrix = plamatrix::SparseMatrix<double, plamatrix::RowMajor, plamatrix::Index>;
    std::vector<plamatrix::Triplet<double, plamatrix::Index>> entries;
    entries.reserve(static_cast<std::size_t>(3 * count - 2));
    for (plamatrix::Index row = 0; row < count; ++row)
    {
        entries.emplace_back(row, row, 2.0);
        if (row > 0)
            entries.emplace_back(row, row - 1, -1.0);
        if (row + 1 < count)
            entries.emplace_back(row, row + 1, -1.0);
    }
    SparseMatrix matrix(count, count);
    matrix.setFromTriplets(entries.begin(), entries.end());
    const plamatrix::VectorXd expected = plamatrix::VectorXd::Ones(count);
    const plamatrix::VectorXd rhs = matrix * expected;

    plamatrix::ConjugateGradient<SparseMatrix, plamatrix::Lower | plamatrix::Upper> solver;
    solver.setTolerance(1e-10).setMaxIterations(1000).compute(matrix);
    const plamatrix::VectorXd solution = solver.solve(rhs);
    if (solver.info() != plamatrix::Success)
    {
        std::cerr << "CG failed after " << solver.iterations() << " iterations\n";
        return 1;
    }
    std::cout << "nonzeros: " << matrix.nonZeros() << ", relative residual: " << solver.error() << '\n';
    for (SparseMatrix::InnerIterator entry(matrix, 0); entry; ++entry)
        std::cout << "(" << entry.row() << ", " << entry.col() << ") = " << entry.value() << '\n';
    return solution.isApprox(expected, 1e-8) ? 0 : 1;
}
