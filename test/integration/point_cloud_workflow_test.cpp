// ============================================================================
// PlaMatrix 集成测试: 完整点云处理工作流
//
// 模拟真实点云处理场景: 生成点云 → 计算中心 → 构建协方差 →
// PCA 法向量估计 → 刚体变换 → GPU 加速 → 结果验证
// ============================================================================

#include <cmath>
#include <cstdlib>
#include <vector>

#include <gtest/gtest.h>
#include <omp.h>

#include <plamatrix/plamatrix.h>

using namespace plamatrix;

class PointCloudWorkflowTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        std::srand(42);
        omp_set_num_threads(1);
    }

    // 生成球形点云 — 各向同性，协方差矩阵接近单位矩阵的倍数
    DenseMatrix<float, Device::CPU> generateSphere(Index N, float radius)
    {
        DenseMatrix<float, Device::CPU> pts(N, 3);
        for (Index i = 0; i < N; ++i)
        {
            float u = static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f;
            float v = static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f;
            float w = static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f;
            float len = std::sqrt(u*u + v*v + w*w) + 1e-8f;
            pts(i, 0) = radius * u / len;
            pts(i, 1) = radius * v / len;
            pts(i, 2) = radius * w / len;
        }
        return pts;
    }

    // 生成长方体点云 (非各向同性) — 用于验证 PCA 方向
    DenseMatrix<float, Device::CPU> generateBox(Index N, float lx, float ly, float lz)
    {
        DenseMatrix<float, Device::CPU> pts(N, 3);
        for (Index i = 0; i < N; ++i)
        {
            pts(i, 0) = (static_cast<float>(std::rand())/RAND_MAX - 0.5f) * lx;
            pts(i, 1) = (static_cast<float>(std::rand())/RAND_MAX - 0.5f) * ly;
            pts(i, 2) = (static_cast<float>(std::rand())/RAND_MAX - 0.5f) * lz;
        }
        return pts;
    }

    // 计算几何中心 (质心)
    Vec3<float> computeCenter(const DenseMatrix<float, Device::CPU>& pts)
    {
        Vec3<float> c{0, 0, 0};
        Index N = pts.rows();
        for (Index i = 0; i < N; ++i)
        {
            c.x += pts(i, 0);
            c.y += pts(i, 1);
            c.z += pts(i, 2);
        }
        c.x /= N; c.y /= N; c.z /= N;
        return c;
    }
};

// ============================================================================
// 测试 1: 球形点云 PCA — 各向同性 → 三个特征值接近相等
// ============================================================================
TEST_F(PointCloudWorkflowTest, sphere_Isotropic)
{
    Index N = 5000;
    float radius = 5.0f;

    auto pts = generateSphere(N, radius);
    auto center = computeCenter(pts);

    // 质心应接近原点
    EXPECT_NEAR(center.x, 0.0f, 0.5f);
    EXPECT_NEAR(center.y, 0.0f, 0.5f);
    EXPECT_NEAR(center.z, 0.0f, 0.5f);

    // 协方差矩阵
    auto cov = covarianceMatrix<float, Device::CPU>(pts);
    EXPECT_EQ(cov.rows(), 3);
    EXPECT_EQ(cov.cols(), 3);

    // PCA: 特征值应接近相等 (球形 → 各向同性)
    auto eigenvals = eigh(cov);
    float ratio_12 = eigenvals(0,0) / eigenvals(1,0);  // 最大/次大
    float ratio_23 = eigenvals(1,0) / eigenvals(2,0);  // 次大/最小

    EXPECT_NEAR(ratio_12, 1.0f, 0.4f);
    EXPECT_NEAR(ratio_23, 1.0f, 0.4f);

    // 所有特征值应为正
    EXPECT_GT(eigenvals(0,0), 0.0f);
    EXPECT_GT(eigenvals(1,0), 0.0f);
    EXPECT_GT(eigenvals(2,0), 0.0f);
}

// ============================================================================
// 测试 2: 长方体点云 PCA — 特征值比例反映形状
//   10×2×1 的长方体 → 最大特征值 >> 次大 >> 最小
// ============================================================================
TEST_F(PointCloudWorkflowTest, box_Anisotropic)
{
    Index N = 8000;
    float lx = 10.0f, ly = 2.0f, lz = 1.0f;

    auto pts = generateBox(N, lx, ly, lz);
    auto cov = covarianceMatrix<float, Device::CPU>(pts);
    auto eigenvals = eigh(cov);

    // 特征值应降序排列，最大方向方差 ≈ lx²/12
    EXPECT_GE(eigenvals(0,0), eigenvals(1,0));
    EXPECT_GE(eigenvals(1,0), eigenvals(2,0));

    // 最大特征值 / 最小特征值 应远大于球体情况
    float ratio = eigenvals(0,0) / eigenvals(2,0);
    EXPECT_GT(ratio, 3.0f);  // ≠ 球形
}

// ============================================================================
// 测试 3: 刚体变换 — 旋转矩阵正交性、变换正确性
// ============================================================================
TEST_F(PointCloudWorkflowTest, rigidTransform_Orthogonality)
{
    // 绕 X 轴 30°
    Vec3<float> axis{1.0f, 0.0f, 0.0f};
    float angle = 0.523599f;  // 30 degrees
    auto R = rotationMatrix<float, Device::CPU>(axis, angle);

    // R 应为 3x3 旋转矩阵
    EXPECT_EQ(R.rows(), 3);
    EXPECT_EQ(R.cols(), 3);

    // 正交性: R * R^T ≈ I
    auto Rt = R.transpose();
    auto I_check = gemm(R, Rt);

    for (Index i = 0; i < 3; ++i)
    {
        for (Index j = 0; j < 3; ++j)
        {
            float expected = (i == j) ? 1.0f : 0.0f;
            EXPECT_NEAR(I_check(i,j), expected, 1e-4f);
        }
    }

    // det(R) ≈ 1
    float det = R(0,0)*(R(1,1)*R(2,2) - R(1,2)*R(2,1))
              - R(0,1)*(R(1,0)*R(2,2) - R(1,2)*R(2,0))
              + R(0,2)*(R(1,0)*R(2,1) - R(1,1)*R(2,0));
    EXPECT_NEAR(det, 1.0f, 1e-4f);
}

// ============================================================================
// 测试 4: 点变换 — CPU vs GPU 一致性 (中等规模)
// ============================================================================
#ifdef PLAMATRIX_WITH_CUDA
TEST_F(PointCloudWorkflowTest, transformPoints_CpuVsGpu)
{
    Index N = 10000;
    auto pts = generateBox(N, 10.0f, 5.0f, 2.0f);

    // 变换: 绕 Z 轴 45° + 平移
    Vec3<float> axis{0.0f, 0.0f, 1.0f};
    auto R = rotationMatrix<float, Device::CPU>(axis, 0.785398f);
    Vec3<float> t{3.0f, -2.0f, 7.0f};
    auto T = rigidTransform<float, Device::CPU>(R, t);

    auto cpu_result = transformPoints<float, Device::CPU>(T, pts);

    auto gpu_result = transformPoints<float, Device::GPU>(
        T.toGpu(), pts.toGpu()).toCpu();

    for (Index i = 0; i < N; ++i)
    {
        for (Index c = 0; c < 3; ++c)
        {
            EXPECT_NEAR(cpu_result(i,c), gpu_result(i,c), 1e-4f);
        }
    }
}
#endif

// ============================================================================
// 测试 5: SVD 分解 — A = U*S*Vt 重构验证
// ============================================================================
TEST_F(PointCloudWorkflowTest, svd_Reconstruction)
{
    Index N = 32;
    auto A_cpu = DenseMatrix<float, Device::CPU>(N, N);
    for (Index j = 0; j < N; ++j)
    {
        for (Index i = 0; i < N; ++i)
        {
            A_cpu(i,j) = static_cast<float>(std::rand()) / RAND_MAX;
        }
    }

    auto [U, S_vec, Vt] = svd(A_cpu);

    // 重建: A ≈ U * diag(S) * Vt
    // diag(S) 是 N×N 对角矩阵: D(i,j) = (i==j ? S(i) : 0)
    // D * Vt → 每列乘对应奇异值
    auto D_Vt = DenseMatrix<float, Device::CPU>(N, N);
    for (Index j = 0; j < N; ++j)
    {
        for (Index i = 0; i < N; ++i)
        {
            D_Vt(i,j) = S_vec(i,0) * Vt(i,j);
        }
    }
    auto A_recon = gemm(U, D_Vt);

    for (Index j = 0; j < N; ++j)
    {
        for (Index i = 0; i < N; ++i)
        {
            EXPECT_NEAR(A_cpu(i,j), A_recon(i,j), 1e-3f);
        }
    }

    // 奇异值应降序
    EXPECT_GE(S_vec(0,0), S_vec(N-1,0));
}

// ============================================================================
// 测试 6: 线性求解 + 残差验证
// ============================================================================
TEST_F(PointCloudWorkflowTest, solve_ResidualCheck)
{
    Index N = 100;

    // 构造对称正定矩阵 (对角占优)
    auto A = DenseMatrix<float, Device::CPU>(N, N);
    for (Index j = 0; j < N; ++j)
    {
        for (Index i = 0; i < N; ++i)
        {
            A(i,j) = static_cast<float>(std::rand()) / RAND_MAX * 0.1f;
        }
    }
    for (Index i = 0; i < N; ++i) A(i,i) += static_cast<float>(N);

    // 真实解
    auto x_true = DenseMatrix<float, Device::CPU>(N, 1);
    for (Index i = 0; i < N; ++i) x_true(i,0) = static_cast<float>(i % 7);

    // b = A * x_true
    auto b = gemm(A, x_true);

    // 求解 A*x = b
    auto x = solve<float, Device::CPU>(A, b);

    // 验证 x ≈ x_true
    for (Index i = 0; i < N; ++i)
    {
        EXPECT_NEAR(x(i,0), x_true(i,0), 1e-3f);
    }
}

// ============================================================================
// 测试 7: COO/CSR 往返 — 构造 → 转换 → 数据完整性
// ============================================================================
TEST_F(PointCloudWorkflowTest, sparse_CooToCsr_RoundTrip)
{
    Index N = 50;
    COOMatrix<float, Device::CPU> coo(N, N);

    // 五对角矩阵 (BA 中常见)
    for (Index i = 0; i < N; ++i)
    {
        coo.add(i, i, 4.0f);
        if (i > 0)  coo.add(i, i-1, -1.0f);
        if (i < N-1) coo.add(i, i+1, -1.0f);
        if (i > 1)  coo.add(i, i-2, -0.5f);
        if (i < N-2) coo.add(i, i+2, -0.5f);
    }

    EXPECT_EQ(coo.nnz(), 5*N - 6);

    auto csr = coo.toCsr();
    EXPECT_EQ(csr.rows(), N);
    EXPECT_EQ(csr.cols(), N);
    EXPECT_EQ(csr.nnz(), 5*N - 6);

    // 验证 row_offsets 单调非减
    for (Index i = 1; i <= N; ++i)
    {
        EXPECT_GE(csr.rowOffsets()[i], csr.rowOffsets()[i-1]);
    }
    EXPECT_EQ(csr.rowOffsets()[N], 5*N - 6);
}
