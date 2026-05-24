#include <cmath>

#include <gtest/gtest.h>

#include <plamatrix/ops/solver.h>

using namespace plamatrix;

// Solver: solve_2x2_Cpu — solve a simple 2x2 linear system on CPU
TEST(Solver, solve_2x2_Cpu)
{
    // System:
    //   [ 4  1 ] [ x0 ]   [ 5 ]
    //   [ 2  3 ] [ x1 ] = [ 7 ]
    // Expected solution: x0 = 0.8, x1 = 1.8
    DenseMatrix<double, Device::CPU> A(2, 2);
    A.setValue(0, 0, 4.0);
    A.setValue(1, 0, 2.0);
    A.setValue(0, 1, 1.0);
    A.setValue(1, 1, 3.0);

    DenseMatrix<double, Device::CPU> B(2, 1);
    B.setValue(0, 0, 5.0);
    B.setValue(1, 0, 7.0);

    auto X = solve(A, B);

    // Check dimensions
    EXPECT_EQ(X.rows(), 2);
    EXPECT_EQ(X.cols(), 1);

    // Check solution
    EXPECT_NEAR(X(0, 0), 0.8, 1e-9);
    EXPECT_NEAR(X(1, 0), 1.8, 1e-9);

    // Verify: A * X should equal B
    double ax0 = A(0, 0) * X(0, 0) + A(0, 1) * X(1, 0);
    double ax1 = A(1, 0) * X(0, 0) + A(1, 1) * X(1, 0);
    EXPECT_NEAR(ax0, B(0, 0), 1e-9);
    EXPECT_NEAR(ax1, B(1, 0), 1e-9);
}

// Solver: solve_2x2_Gpu — solve a simple 2x2 linear system on GPU
#ifdef PLAMATRIX_WITH_CUDA
TEST(Solver, solve_2x2_Gpu)
{
    DenseMatrix<double, Device::CPU> A_cpu(2, 2);
    A_cpu.setValue(0, 0, 4.0);
    A_cpu.setValue(1, 0, 2.0);
    A_cpu.setValue(0, 1, 1.0);
    A_cpu.setValue(1, 1, 3.0);

    DenseMatrix<double, Device::CPU> B_cpu(2, 1);
    B_cpu.setValue(0, 0, 5.0);
    B_cpu.setValue(1, 0, 7.0);

    auto A_gpu = A_cpu.toGpu();
    auto B_gpu = B_cpu.toGpu();
    auto X_gpu = solve(A_gpu, B_gpu);
    auto X = X_gpu.toCpu();

    // Check dimensions
    EXPECT_EQ(X.rows(), 2);
    EXPECT_EQ(X.cols(), 1);

    // Check solution (GPU tolerance is looser than CPU)
    EXPECT_NEAR(X(0, 0), 0.8, 1e-6);
    EXPECT_NEAR(X(1, 0), 1.8, 1e-6);

    // Verify: A * X should equal B
    double ax0 = A_cpu(0, 0) * X(0, 0) + A_cpu(0, 1) * X(1, 0);
    double ax1 = A_cpu(1, 0) * X(0, 0) + A_cpu(1, 1) * X(1, 0);
    EXPECT_NEAR(ax0, B_cpu(0, 0), 1e-6);
    EXPECT_NEAR(ax1, B_cpu(1, 0), 1e-6);
}
#endif

// Solver: solve_RhsMatrix_Cpu — solve with multiple right-hand sides (identity * X = B)
TEST(Solver, solve_RhsMatrix_Cpu)
{
    // Use identity matrix as A: I * X = B, so X should equal B
    Index n = 3;
    Index nrhs = 2;

    DenseMatrix<double, Device::CPU> A(n, n);
    for (Index i = 0; i < n; ++i)
    {
        A.setValue(i, i, 1.0);
    }

    DenseMatrix<double, Device::CPU> B(n, nrhs);
    B.setValue(0, 0, 1.0);
    B.setValue(1, 0, 2.0);
    B.setValue(2, 0, 3.0);
    B.setValue(0, 1, 4.0);
    B.setValue(1, 1, 5.0);
    B.setValue(2, 1, 6.0);

    auto X = solve(A, B);

    // Check dimensions
    EXPECT_EQ(X.rows(), n);
    EXPECT_EQ(X.cols(), nrhs);

    // X should equal B since A = I
    for (Index j = 0; j < nrhs; ++j)
    {
        for (Index i = 0; i < n; ++i)
        {
            EXPECT_NEAR(X(i, j), B(i, j), 1e-9);
        }
    }
}
