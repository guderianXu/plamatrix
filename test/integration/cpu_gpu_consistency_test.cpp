#include <cmath>
#include <cstdlib>

#include <gtest/gtest.h>
#include <omp.h>

#include <plamatrix/plamatrix.h>

using namespace plamatrix;

// ============================================================================
// Typed test fixture for CPU vs GPU consistency tests.
// Each test runs for both float and double precision.
// ============================================================================

template <typename Scalar>
class CpuGpuConsistencyTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Seed random number generator for reproducibility across runs.
        std::srand(42);
        // Force single-threaded CPU execution for deterministic results.
        omp_set_num_threads(1);
    }

    /// Create an m x n matrix filled with pseudo-random values in [0, 1).
    DenseMatrix<Scalar, Device::CPU> randomMatrix(Index m, Index n)
    {
        DenseMatrix<Scalar, Device::CPU> mat(m, n);
        for (Index j = 0; j < n; ++j)
        {
            for (Index i = 0; i < m; ++i)
            {
                mat(i, j) = static_cast<Scalar>(std::rand()) / static_cast<Scalar>(RAND_MAX);
            }
        }
        return mat;
    }

    /// Create a well-conditioned square matrix via diagonal dominance: M = random + n * I.
    DenseMatrix<Scalar, Device::CPU> wellConditionedMatrix(Index n)
    {
        auto M = randomMatrix(n, n);
        for (Index i = 0; i < n; ++i)
        {
            M(i, i) += static_cast<Scalar>(n);
        }
        return M;
    }

    /// Tolerance for element-wise comparisons, scaled by problem size for accumulating error.
    Scalar tolerance(Index scaling = 1)
    {
        if constexpr (std::is_same_v<Scalar, float>)
        {
            return 1e-3f * static_cast<Scalar>(std::sqrt(static_cast<double>(scaling)));
        }
        else
        {
            return 1e-6 * static_cast<Scalar>(std::sqrt(static_cast<double>(scaling)));
        }
    }

    /// Reconstruct a matrix from its SVD: result = U * diag(S) * Vt.
    DenseMatrix<Scalar, Device::CPU> reconstructFromSvd(const DenseMatrix<Scalar, Device::CPU>& U,
                                                        const DenseMatrix<Scalar, Device::CPU>& S,
                                                        const DenseMatrix<Scalar, Device::CPU>& Vt,
                                                        Index m,
                                                        Index n)
    {
        DenseMatrix<Scalar, Device::CPU> result(m, n);
        Index k = std::min(m, n);
        for (Index i = 0; i < m; ++i)
        {
            for (Index j = 0; j < n; ++j)
            {
                Scalar sum = static_cast<Scalar>(0);
                for (Index t = 0; t < k; ++t)
                {
                    sum += U(i, t) * S(t, 0) * Vt(t, j);
                }
                result(i, j) = sum;
            }
        }
        return result;
    }
};

using ScalarTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(CpuGpuConsistencyTest, ScalarTypes);

#ifdef PLAMATRIX_WITH_CUDA
// ============================================================================
// Test 1: GEMM — matrix multiplication consistency
// ============================================================================

TYPED_TEST(CpuGpuConsistencyTest, gemm)
{
    using Scalar = TypeParam;
    constexpr Index N = 128;

    auto A = this->randomMatrix(N, N);
    auto B = this->randomMatrix(N, N);

    auto C_cpu = gemm(A, B);

    auto A_gpu = A.toGpu();
    auto B_gpu = B.toGpu();
    auto C_gpu = gemm(A_gpu, B_gpu);
    auto C_from_gpu = C_gpu.toCpu();

    const Scalar tol = this->tolerance(N);
    for (Index j = 0; j < N; ++j)
    {
        for (Index i = 0; i < N; ++i)
        {
            EXPECT_NEAR(C_cpu(i, j), C_from_gpu(i, j), tol)
                << "GEMM mismatch at (" << i << ", " << j << ")";
        }
    }
}

// ============================================================================
// Test 2: Element-wise addition consistency
// ============================================================================

TYPED_TEST(CpuGpuConsistencyTest, add)
{
    using Scalar = TypeParam;
    constexpr Index M = 64;
    constexpr Index N = 48;

    auto A = this->randomMatrix(M, N);
    auto B = this->randomMatrix(M, N);

    auto C_cpu = add(A, B);

    auto A_gpu = A.toGpu();
    auto B_gpu = B.toGpu();
    auto C_gpu = add(A_gpu, B_gpu);
    auto C_from_gpu = C_gpu.toCpu();

    const Scalar tol = this->tolerance();
    for (Index j = 0; j < N; ++j)
    {
        for (Index i = 0; i < M; ++i)
        {
            EXPECT_NEAR(C_cpu(i, j), C_from_gpu(i, j), tol)
                << "add mismatch at (" << i << ", " << j << ")";
        }
    }
}

// ============================================================================
// Test 3: Transpose consistency
// ============================================================================

TYPED_TEST(CpuGpuConsistencyTest, transpose)
{
    using Scalar = TypeParam;
    constexpr Index M = 50;
    constexpr Index N = 30;

    auto A = this->randomMatrix(M, N);

    auto At_cpu = A.transpose();

    auto A_gpu = A.toGpu();
    auto At_gpu = A_gpu.transpose();
    auto At_from_gpu = At_gpu.toCpu();

    EXPECT_EQ(At_cpu.rows(), N);
    EXPECT_EQ(At_cpu.cols(), M);
    EXPECT_EQ(At_from_gpu.rows(), N);
    EXPECT_EQ(At_from_gpu.cols(), M);

    const Scalar tol = this->tolerance();
    for (Index j = 0; j < M; ++j)
    {
        for (Index i = 0; i < N; ++i)
        {
            EXPECT_NEAR(At_cpu(i, j), At_from_gpu(i, j), tol)
                << "transpose mismatch at (" << i << ", " << j << ")";
        }
    }
}

// ============================================================================
// Test 4: SVD reconstruction consistency
// ============================================================================

TYPED_TEST(CpuGpuConsistencyTest, svdReconstruction)
{
    using Scalar = TypeParam;
    constexpr Index M = 32;
    constexpr Index N = 24;

    auto A = this->randomMatrix(M, N);

    // CPU SVD
    auto [U_cpu, S_cpu, Vt_cpu] = svd(A);
    auto A_recon_cpu = this->reconstructFromSvd(U_cpu, S_cpu, Vt_cpu, M, N);

    // GPU SVD
    auto A_gpu = A.toGpu();
    auto [U_gpu, S_gpu, Vt_gpu] = svd(A_gpu);
    auto U_cpu_from_gpu = U_gpu.toCpu();
    auto S_cpu_from_gpu = S_gpu.toCpu();
    auto Vt_cpu_from_gpu = Vt_gpu.toCpu();
    auto A_recon_gpu = this->reconstructFromSvd(U_cpu_from_gpu, S_cpu_from_gpu, Vt_cpu_from_gpu, M, N);

    EXPECT_EQ(U_cpu.rows(), M);
    EXPECT_EQ(U_cpu.cols(), M);
    EXPECT_EQ(S_cpu.rows(), std::min(M, N));
    EXPECT_EQ(S_cpu.cols(), 1);
    EXPECT_EQ(Vt_cpu.rows(), N);
    EXPECT_EQ(Vt_cpu.cols(), N);
    EXPECT_EQ(U_cpu_from_gpu.rows(), M);
    EXPECT_EQ(U_cpu_from_gpu.cols(), M);
    EXPECT_EQ(S_cpu_from_gpu.rows(), std::min(M, N));
    EXPECT_EQ(S_cpu_from_gpu.cols(), 1);
    EXPECT_EQ(Vt_cpu_from_gpu.rows(), N);
    EXPECT_EQ(Vt_cpu_from_gpu.cols(), N);

    // Verify both reconstructions match the original A
    const Scalar tol = this->tolerance(std::min(M, N));
    for (Index j = 0; j < N; ++j)
    {
        for (Index i = 0; i < M; ++i)
        {
            EXPECT_NEAR(A(i, j), A_recon_cpu(i, j), tol)
                << "CPU SVD reconstruction mismatch at (" << i << ", " << j << ")";
            EXPECT_NEAR(A(i, j), A_recon_gpu(i, j), tol)
                << "GPU SVD reconstruction mismatch at (" << i << ", " << j << ")";
        }
    }

    // Verify singular values are consistent between CPU and GPU
    EXPECT_EQ(S_cpu.rows(), S_cpu_from_gpu.rows());
    EXPECT_EQ(S_cpu.cols(), S_cpu_from_gpu.cols());

    Index k = std::min(M, N);
    for (Index t = 0; t < k; ++t)
    {
        EXPECT_NEAR(S_cpu(t, 0), S_cpu_from_gpu(t, 0), tol)
            << "Singular value mismatch at index " << t;
    }
}

// ============================================================================
// Test 5: Linear solver consistency
// ============================================================================

TYPED_TEST(CpuGpuConsistencyTest, solver)
{
    using Scalar = TypeParam;
    constexpr Index N = 64;

    // Create well-conditioned system A * X = B
    auto A_cpu = this->wellConditionedMatrix(N);
    auto B_cpu = this->randomMatrix(N, 1);

    // CPU solve
    auto X_cpu = solve(A_cpu, B_cpu);

    // GPU solve
    auto A_gpu = A_cpu.toGpu();
    auto B_gpu = B_cpu.toGpu();
    auto X_gpu = solve(A_gpu, B_gpu);
    auto X_from_gpu = X_gpu.toCpu();

    // Compare solutions
    const Scalar tol = this->tolerance(N);
    for (Index i = 0; i < N; ++i)
    {
        EXPECT_NEAR(X_cpu(i, 0), X_from_gpu(i, 0), tol)
            << "solver mismatch at row " << i;
    }

    // Verify GPU solution actually solves the system: A * X_gpu ≈ B
    for (Index i = 0; i < N; ++i)
    {
        Scalar sum = static_cast<Scalar>(0);
        for (Index k = 0; k < N; ++k)
        {
            sum += A_cpu(i, k) * X_from_gpu(k, 0);
        }
        EXPECT_NEAR(sum, B_cpu(i, 0), tol)
            << "GPU solution residual too large at row " << i;
    }
}

// ============================================================================
// Test 6: Point cloud transform consistency
// ============================================================================

TYPED_TEST(CpuGpuConsistencyTest, pointTransform)
{
    using Scalar = TypeParam;
    constexpr Index NUM_POINTS = 32;

    // Build a rigid transform: rotation around arbitrary axis + translation
    Vec3<Scalar> axis = {static_cast<Scalar>(0.3), static_cast<Scalar>(0.6), static_cast<Scalar>(0.8)};
    Scalar angle = static_cast<Scalar>(1.2); // ~69 degrees

    auto R_cpu = rotationMatrix<Scalar, Device::CPU>(axis, angle);
    Vec3<Scalar> t = {static_cast<Scalar>(5.0), static_cast<Scalar>(-3.0), static_cast<Scalar>(7.0)};
    auto T_cpu = rigidTransform<Scalar, Device::CPU>(R_cpu, t);

    // Verify T dimensions
    EXPECT_EQ(T_cpu.rows(), 4);
    EXPECT_EQ(T_cpu.cols(), 4);

    // Generate random 3D points
    auto points_cpu = this->randomMatrix(NUM_POINTS, 3);

    // CPU transform
    auto result_cpu = transformPoints<Scalar, Device::CPU>(T_cpu, points_cpu);
    EXPECT_EQ(result_cpu.rows(), NUM_POINTS);
    EXPECT_EQ(result_cpu.cols(), 3);

    // GPU transform
    auto T_gpu = T_cpu.toGpu();
    auto points_gpu = points_cpu.toGpu();
    auto result_gpu = transformPoints<Scalar, Device::GPU>(T_gpu, points_gpu);
    auto result_from_gpu = result_gpu.toCpu();

    // Compare transformed points
    const Scalar tol = this->tolerance();
    for (Index j = 0; j < 3; ++j)
    {
        for (Index i = 0; i < NUM_POINTS; ++i)
        {
            EXPECT_NEAR(result_cpu(i, j), result_from_gpu(i, j), tol)
                << "pointTransform mismatch at point " << i << ", coordinate " << j;
        }
    }
}
#endif
