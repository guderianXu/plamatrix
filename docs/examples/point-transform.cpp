/**
 * PlaMatrix 示例：点云刚体变换
 *
 * 演示内容：
 * - Rodrigues 旋转矩阵生成
 * - 4x4 刚体变换矩阵构造
 * - 批量点云坐标变换
 * - 协方差矩阵计算
 * - SVD / Eigh 特征值分解
 *
 * 典型应用场景：点云配准、刚体运动估计、PCA 法向量分析
 *
 * 编译：
 *   g++ -std=c++17 -O2 -fopenmp point-transform.cpp \
 *       -I<plamatrix_include_dir> -L<plamatrix_lib_dir> -lplamatrix \
 *       -lcublas -lcusolver -lcudart -o point-transform
 */

#include <plamatrix/plamatrix.h>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

using namespace plamatrix;

/// 生成随机点云（N 个点，散布在单位球内）
DenseMatrix<float, Device::CPU> randomPointCloud(Index N, unsigned int seed = 42)
{
    DenseMatrix<float, Device::CPU> cloud(N, 3);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (Index col = 0; col < 3; ++col)
        for (Index row = 0; row < N; ++row)
            cloud(row, col) = dist(rng);

    return cloud;
}

/// 计算均方根误差 (RMSE) 验证变换后的坐标
float computeRmsError(const DenseMatrix<float, Device::CPU>& expected,
                      const DenseMatrix<float, Device::CPU>& actual)
{
    float sum_sq = 0.0f;
    Index n = expected.size();
    for (Index i = 0; i < n; ++i)
    {
        float diff = expected.data()[i] - actual.data()[i];
        sum_sq += diff * diff;
    }
    return std::sqrt(sum_sq / static_cast<float>(n));
}

/// 手动计算单点变换用于验证
std::vector<float> transformSinglePoint(const DenseMatrix<float, Device::CPU>& R,
                                         const Vec3<float>& t,
                                         float x, float y, float z)
{
    // p' = R * p + t (R 是 3x3, p 是 3x1, t 是 3x1)
    float r00 = R(0, 0), r01 = R(1, 0), r02 = R(2, 0);
    float r10 = R(0, 1), r11 = R(1, 1), r12 = R(2, 1);
    float r20 = R(0, 2), r21 = R(1, 2), r22 = R(2, 2);

    float px = r00 * x + r01 * y + r02 * z + t.x;
    float py = r10 * x + r11 * y + r12 * z + t.y;
    float pz = r20 * x + r21 * y + r22 * z + t.z;

    return {px, py, pz};
}

int main()
{
    try
    {
        // ============================================================
        // 1. 生成测试点云
        // ============================================================
        std::cout << "=== 1. 生成测试点云 ===" << std::endl;
        const Index N = 1000;
        auto cloud = randomPointCloud(N);
        std::cout << "生成 " << N << " 个随机点" << std::endl;
        std::cout << "前 3 个点:" << std::endl;
        for (Index i = 0; i < 3; ++i)
        {
            std::cout << "  (" << cloud(i, 0) << ", "
                      << cloud(i, 1) << ", "
                      << cloud(i, 2) << ")" << std::endl;
        }

        // ============================================================
        // 2. 构造旋转矩阵 (绕 Y 轴旋转 60 度)
        // ============================================================
        std::cout << "\n=== 2. 构造旋转矩阵 ===" << std::endl;
        Vec3<float> axis = {0.0f, 1.0f, 0.0f};  // Y 轴
        float angle = 3.1415926f / 3.0f;          // 60 度
        auto R = rotationMatrix<float, Device::CPU>(axis, angle);

        std::cout << "旋转轴: Y, 角度: 60°" << std::endl;
        std::cout << "旋转矩阵 R (3x3):" << std::endl;
        for (Index i = 0; i < 3; ++i)
        {
            std::cout << "  ";
            for (Index j = 0; j < 3; ++j)
                std::cout << R(i, j) << "\t";
            std::cout << std::endl;
        }

        // ============================================================
        // 3. 构造刚体变换矩阵
        // ============================================================
        std::cout << "\n=== 3. 构造刚体变换矩阵 ===" << std::endl;
        Vec3<float> translation = {5.0f, -2.0f, 10.0f};
        auto T = rigidTransform(R, translation);

        std::cout << "平移向量: (" << translation.x << ", "
                  << translation.y << ", " << translation.z << ")" << std::endl;
        std::cout << "刚体变换矩阵 T (4x4):" << std::endl;
        for (Index i = 0; i < 4; ++i)
        {
            std::cout << "  ";
            for (Index j = 0; j < 4; ++j)
                std::cout << T(i, j) << "\t";
            std::cout << std::endl;
        }

        // ============================================================
        // 4. 变换点云 (CPU)
        // ============================================================
        std::cout << "\n=== 4. 变换点云 ===" << std::endl;
        auto transformed = transformPoints(T, cloud);

        // 手动验证前 3 个点
        std::cout << "CPU 变换结果验证 (前 3 个点):" << std::endl;
        for (Index i = 0; i < 3; ++i)
        {
            float x = cloud(i, 0), y = cloud(i, 1), z = cloud(i, 2);
            auto expected = transformSinglePoint(R, translation, x, y, z);

            std::cout << "  原始: (" << x << ", " << y << ", " << z << ")" << std::endl;
            std::cout << "  期望: (" << expected[0] << ", " << expected[1] << ", " << expected[2] << ")" << std::endl;
            std::cout << "  实际: (" << transformed(i, 0) << ", " << transformed(i, 1) << ", "
                      << transformed(i, 2) << ")" << std::endl;
        }

        // ============================================================
        // 5. GPU 加速变换
        // ============================================================
        std::cout << "\n=== 5. GPU 加速变换 ===" << std::endl;

        auto cloud_gpu = cloud.toGpu();
        auto T_gpu = T.toGpu();
        auto result_gpu = transformPoints(T_gpu, cloud_gpu);
        auto result_cpu = result_gpu.toCpu();

        float rmse = computeRmsError(transformed, result_cpu);
        std::cout << "CPU 与 GPU 变换结果 RMSE: " << rmse << std::endl;
        if (rmse < 1e-4f)
            std::cout << "验证通过: CPU 与 GPU 变换结果一致" << std::endl;

        // ============================================================
        // 6. 计算协方差矩阵
        // ============================================================
        std::cout << "\n=== 6. 计算协方差矩阵 ===" << std::endl;
        auto cov = covarianceMatrix(transformed);

        std::cout << "变换后点云的协方差矩阵 (3x3):" << std::endl;
        for (Index i = 0; i < 3; ++i)
        {
            std::cout << "  ";
            for (Index j = 0; j < 3; ++j)
                std::cout << cov(i, j) << "\t";
            std::cout << std::endl;
        }

        // ============================================================
        // 7. 特征值分解 (PCA)
        // ============================================================
        std::cout << "\n=== 7. 特征值分解 (PCA 主方向分析) ===" << std::endl;

        // 构造对称协方差矩阵用于 eigh（cov 已经是对称的）
        auto eigenvalues = eigh(cov);

        std::cout << "特征值 (降序排列):" << std::endl;
        for (Index i = 0; i < 3; ++i)
            std::cout << "  lambda_" << i << " = " << eigenvalues(i, 0) << std::endl;

        // 解释方差比例
        float total = eigenvalues(0, 0) + eigenvalues(1, 0) + eigenvalues(2, 0);
        if (total > 0)
        {
            std::cout << "方差解释比例:" << std::endl;
            for (Index i = 0; i < 3; ++i)
                std::cout << "  PC" << (i + 1) << ": " << (100.0f * eigenvalues(i, 0) / total) << "%" << std::endl;
        }

        // ============================================================
        // 8. SVD 分解（用于点云配准中的最优旋转估计）
        // ============================================================
        std::cout << "\n=== 8. SVD 分解 ===" << std::endl;

        // 构造一个 4x3 矩阵做 SVD 演示
        DenseMatrix<float, Device::CPU> H(4, 3);
        H(0, 0) = 1.0f; H(0, 1) = 2.0f; H(0, 2) = 3.0f;
        H(1, 0) = 4.0f; H(1, 1) = 5.0f; H(1, 2) = 6.0f;
        H(2, 0) = 7.0f; H(2, 1) = 8.0f; H(2, 2) = 9.0f;
        H(3, 0) = 2.0f; H(3, 1) = 4.0f; H(3, 2) = 6.0f;

        auto [U, S, Vt] = svd(H);

        std::cout << "H (4x3) 的奇异值:" << std::endl;
        Index min_dim = std::min(H.rows(), H.cols());
        for (Index i = 0; i < min_dim; ++i)
            std::cout << "  sigma_" << i << " = " << S(i, 0) << std::endl;

        std::cout << "\n所有点云变换示例执行完成!" << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
