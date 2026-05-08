// ============================================================================
// PlaMatrix 参考验证测试 — 与 NumPy/SciPy ground truth 对比
//
// 用 test/reference/ 目录下的 .bin 文件 (generate_reference.py + NumPy/SciPy 生成)，
// 以 double 精度逐元素比对 PlaMatrix 结果与参考值。
//
// Python 端使用 float64 计算，C++ 端同样使用 double 模板实例化，
// 容差 1e-10 (仅浮点舍入误差)。
// ============================================================================

#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <plamatrix/plamatrix.h>

using namespace plamatrix;

// ============================================================================
// 二进制文件 I/O (与 generate_reference.py 格式一致)
// ============================================================================

DenseMatrix<double, Device::CPU> loadMatrix(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);

    int64_t rows = 0, cols = 0;
    f.read(reinterpret_cast<char*>(&rows), 8);
    f.read(reinterpret_cast<char*>(&cols), 8);

    DenseMatrix<double, Device::CPU> mat(rows, cols);
    f.read(reinterpret_cast<char*>(mat.data()),
           static_cast<std::size_t>(rows * cols) * sizeof(double));
    return mat;
}

std::vector<double> loadVector(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);

    int64_t n = 0;
    f.read(reinterpret_cast<char*>(&n), 8);
    std::vector<double> vec(static_cast<std::size_t>(n));
    f.read(reinterpret_cast<char*>(vec.data()),
           static_cast<std::size_t>(n) * sizeof(double));
    return vec;
}

const std::string REF = std::string(TEST_REFERENCE_DIR) + "/";

// 容差: Python float64 vs C++ double → 只允许浮点舍入差异
constexpr double kTol = 1e-10;

// ============================================================================
// 测试 1: GEMM — 与 NumPy matmul 对比 (double 精度)
// ============================================================================
TEST(NumPyReference, gemm_Large)
{
    auto A = loadMatrix(REF + "gemm_A.bin");
    auto B = loadMatrix(REF + "gemm_B.bin");
    auto C_ref = loadMatrix(REF + "gemm_C_ref.bin");

    auto C = gemm(A, B);

    for (Index j = 0; j < C_ref.cols(); ++j)
        for (Index i = 0; i < C_ref.rows(); ++i)
            EXPECT_NEAR(C(i,j), C_ref(i,j), kTol * C_ref.rows())
                << "gemm mismatch at (" << i << "," << j << ")";
}

TEST(NumPyReference, gemm_Small)
{
    auto A = loadMatrix(REF + "gemm_small_A.bin");
    auto B = loadMatrix(REF + "gemm_small_B.bin");
    auto C_ref = loadMatrix(REF + "gemm_small_C_ref.bin");

    auto C = gemm(A, B);

    for (Index j = 0; j < C_ref.cols(); ++j)
        for (Index i = 0; i < C_ref.rows(); ++i)
            EXPECT_NEAR(C(i,j), C_ref(i,j), kTol)
                << "gemm_small mismatch at (" << i << "," << j << ")";
}

// ============================================================================
// 测试 2: SVD — 与 SciPy linalg.svd 对比 (奇异值 + 重构)
// ============================================================================
TEST(NumPyReference, svd_CompareWithScipy)
{
    auto A = loadMatrix(REF + "svd_small_A.bin");
    auto S_vec = loadVector(REF + "svd_small_S_ref.bin");

    // 使用 GPU (cuSOLVER) 与 SciPy (LAPACK) 对比，两者底层算法一致
    auto A_gpu = A.toGpu();
    auto [U_gpu, S_gpu, Vt_gpu] = svd(A_gpu);
    auto U = U_gpu.toCpu();
    auto S = S_gpu.toCpu();
    auto Vt = Vt_gpu.toCpu();

    // 奇异值对比 (降序)
    for (Index i = 0; i < 3; ++i)
        EXPECT_NEAR(S(i,0), S_vec[static_cast<std::size_t>(i)], 1e-8);

    // 重构: A ≈ U * diag(S) * Vt
    auto D_Vt = DenseMatrix<double, Device::CPU>(3, 3);
    for (Index j = 0; j < 3; ++j)
        for (Index i = 0; i < 3; ++i)
            D_Vt(i,j) = S(i,0) * Vt(i,j);

    auto A_recon = gemm(U, D_Vt);
    for (Index j = 0; j < 3; ++j)
        for (Index i = 0; i < 3; ++i)
            EXPECT_NEAR(A_recon(i,j), A(i,j), 1e-8);

    // 正交性: U^T * U ≈ I
    auto I_u = gemm(U.transpose(), U);
    for (Index i = 0; i < 3; ++i)
    {
        EXPECT_NEAR(I_u(i,i), 1.0, 1e-8) << "U ortho diag " << i;
        for (Index j = 0; j < 3; ++j)
            if (i != j) EXPECT_NEAR(I_u(i,j), 0.0, 1e-8) << "U ortho off-diag " << i << "," << j;
    }
}

// ============================================================================
// 测试 3: QR — 重构 + 上三角验证
// ============================================================================
TEST(NumPyReference, qr_Decomposition)
{
    auto A = loadMatrix(REF + "qr_A.bin");

    auto [Q, R] = qr(A);

    // 重构: A ≈ Q * R
    auto A_recon = gemm(Q, R);
    for (Index j = 0; j < A.cols(); ++j)
        for (Index i = 0; i < A.rows(); ++i)
            EXPECT_NEAR(A_recon(i,j), A(i,j), 1e-8);

    // R 应为上三角
    for (Index j = 0; j < R.cols(); ++j)
        for (Index i = static_cast<Index>(j) + 1; i < R.rows(); ++i)
            EXPECT_NEAR(R(i,j), 0.0, 1e-8)
                << "R not upper-triangular at (" << i << "," << j << ")";

    // Q^T * Q ≈ I
    auto I_q = gemm(Q.transpose(), Q);
    for (Index i = 0; i < Q.cols(); ++i)
        EXPECT_NEAR(I_q(i,i), 1.0, 1e-8) << "Q ortho diag " << i;
}

// ============================================================================
// 测试 4: Eigh — 与 SciPy eigvalsh 对比 (降序特征值)
// ============================================================================
TEST(NumPyReference, eigh_CompareWithScipy)
{
    auto A = loadMatrix(REF + "eigh_A.bin");
    auto eig_ref = loadVector(REF + "eigh_ref.bin");

    // 使用 GPU (cuSOLVER syevd) 与 SciPy (LAPACK) 对比
    auto A_gpu = A.toGpu();
    auto eigvals_gpu = eigh(A_gpu);
    auto eigvals = eigvals_gpu.toCpu();

    ASSERT_EQ(eigvals.rows(), 64);
    double max_diff = 0.0;
    for (Index i = 0; i < 64; ++i)
        max_diff = std::max(max_diff, std::abs(eigvals(i,0) - eig_ref[static_cast<std::size_t>(i)]));

    EXPECT_LT(max_diff, 1e-8) << "Eigh max diff vs SciPy: " << max_diff;

    for (Index i = 1; i < 64; ++i)
        EXPECT_GE(eigvals(i-1,0), eigvals(i,0)) << "not descending at " << i;
}

// ============================================================================
// 测试 5: Solve — 与 NumPy linalg.solve 对比 (已知解验证)
// ============================================================================
TEST(NumPyReference, solve_CompareWithNumPy)
{
    auto A = loadMatrix(REF + "solve_A.bin");
    auto b = loadMatrix(REF + "solve_b.bin");
    auto x_ref = loadMatrix(REF + "solve_x_ref.bin");

    auto x = solve<double, Device::CPU>(A, b);

    for (Index i = 0; i < x.rows(); ++i)
        EXPECT_NEAR(x(i,0), x_ref(i,0), 1e-8)
            << "solve mismatch at [" << i << "]";
}

// ============================================================================
// 测试 6: 旋转矩阵 — 与 NumPy Rodrigues 对比
// ============================================================================
TEST(NumPyReference, rotation_CompareWithNumPy)
{
    auto R_ref = loadMatrix(REF + "rotation_R_ref.bin");

    Vec3<double> axis{0.0, 0.0, 1.0};
    double angle = 0.7853981633974483;  // 45 degrees
    auto R = rotationMatrix<double, Device::CPU>(axis, angle);

    for (Index j = 0; j < 3; ++j)
        for (Index i = 0; i < 3; ++i)
            EXPECT_NEAR(R(i,j), R_ref(i,j), 1e-12)
                << "Rotation mismatch at (" << i << "," << j << ")";
}

// ============================================================================
// 测试 7: 点变换 — 与 NumPy 对比 (5000 点)
// ============================================================================
TEST(NumPyReference, transformPoints_CompareWithNumPy)
{
    auto pts = loadMatrix(REF + "transform_points_in.bin");
    auto pts_ref = loadMatrix(REF + "transform_points_ref.bin");

    Vec3<double> axis{0.0, 0.0, 1.0};
    double angle = 0.7853981633974483;
    auto R = rotationMatrix<double, Device::CPU>(axis, angle);
    Vec3<double> t{10.0, 5.0, 3.0};
    auto T = rigidTransform<double, Device::CPU>(R, t);

    auto result = transformPoints<double, Device::CPU>(T, pts);

    double max_err = 0.0;
    for (Index i = 0; i < 5000; ++i)
        for (Index c = 0; c < 3; ++c)
            max_err = std::max(max_err, std::abs(result(i,c) - pts_ref(i,c)));

    EXPECT_LT(max_err, 1e-12) << "Transform max error: " << max_err;
}

// ============================================================================
// 测试 8: 协方差 — 与 NumPy 对比
// ============================================================================
TEST(NumPyReference, covariance_CompareWithNumPy)
{
    auto pts = loadMatrix(REF + "covariance_points.bin");
    auto cov_ref = loadMatrix(REF + "covariance_ref.bin");

    auto cov = covarianceMatrix<double, Device::CPU>(pts);

    for (Index j = 0; j < 3; ++j)
        for (Index i = 0; i < 3; ++i)
            EXPECT_NEAR(cov(i,j), cov_ref(i,j), 1e-12)
                << "covariance mismatch at (" << i << "," << j << ")";
}

// ============================================================================
// 测试 9: CPU GPU GEMM 一致性 (double 精度)
// ============================================================================
TEST(NumPyReference, gemm_CpuVsGpu)
{
    auto A = loadMatrix(REF + "gemm_A.bin");
    auto B = loadMatrix(REF + "gemm_B.bin");

    auto C_cpu = gemm(A, B);
    auto C_gpu = gemm(A.toGpu(), B.toGpu());
    auto C_from_gpu = C_gpu.toCpu();

    for (Index j = 0; j < C_cpu.cols(); ++j)
        for (Index i = 0; i < C_cpu.rows(); ++i)
            EXPECT_NEAR(C_cpu(i,j), C_from_gpu(i,j), 1e-8)
                << "CPU/GPU GEMM mismatch at (" << i << "," << j << ")";
}
