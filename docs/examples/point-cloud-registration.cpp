// ============================================================================
// PlaMatrix 示例 2: 使用公开矩阵 API 做点云刚体变换
//
// 通过 CMake 链接 plamatrix::plamatrix 构建。
//
// 演示: 已知 Z 轴旋转和平移的矩阵计算，以及点云协方差分析。
// ============================================================================

#include <chrono>
#include <cmath>
#include <iostream>
#include <random>

#include <plamatrix/dense/matrix.h>

// 生成球形点云 (N 个随机方向上的点)
plamatrix::MatrixXf generateSphere(plamatrix::Index count, float radius)
{
    plamatrix::MatrixXf pts(count, 3);
    std::mt19937 rng(12345);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    for (plamatrix::Index i = 0; i < count; ++i)
    {
        float x = normal(rng);
        float y = normal(rng);
        float z = normal(rng);
        float len = std::sqrt(x * x + y * y + z * z);
        pts(i, 0) = radius * x / len;
        pts(i, 1) = radius * y / len;
        pts(i, 2) = radius * z / len;
    }
    return pts;
}

int main()
{
    constexpr plamatrix::Index N = 100000;
    std::cout << "=== 点云刚体变换示例 (N=" << N << ") ===\n" << std::endl;

    // ---- 1. 生成球形点云 ----
    std::cout << "1. 生成 " << N << " 个点的球形点云..." << std::endl;
    auto source_pts = generateSphere(N, 5.0f);
    std::cout << "   前三点: (" << source_pts(0, 0) << "," << source_pts(0, 1) << "," << source_pts(0, 2) << ")" << " ("
              << source_pts(1, 0) << "," << source_pts(1, 1) << "," << source_pts(1, 2) << ")" << std::endl;

    // ---- 2. 构建变换: 绕 Z 轴旋转 45°, 平移 (10, 5, 3) ----
    std::cout << "2. 构建刚体变换 (绕Z轴45度, 平移10,5,3)..." << std::endl;
    const float angle = 0.78539816339f; // 45 degrees
    const float cosine = std::cos(angle);
    const float sine = std::sin(angle);
    plamatrix::Matrix3f rotation;
    rotation << cosine, -sine, 0.0f, sine, cosine, 0.0f, 0.0f, 0.0f, 1.0f;
    std::cout << "   旋转矩阵:" << std::endl;
    std::cout << rotation << std::endl;

    plamatrix::RowVectorXf translation(3);
    translation << 10.0f, 5.0f, 3.0f;

    // ---- 3. CPU 变换 ----
    std::cout << "3. CPU 批量点变换..." << std::endl;
    auto t1 = std::chrono::high_resolution_clock::now();
    const plamatrix::MatrixXf target_cpu = (source_pts * rotation.transpose()).rowwise() + translation;
    auto t2 = std::chrono::high_resolution_clock::now();
    auto cpu_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    std::cout << "   CPU: " << cpu_ms << " ms" << std::endl;
    std::cout << "   变换后前三点: (" << target_cpu(0, 0) << "," << target_cpu(0, 1) << "," << target_cpu(0, 2) << ")"
              << " (" << target_cpu(1, 0) << "," << target_cpu(1, 1) << "," << target_cpu(1, 2) << ")" << std::endl;

    // ---- 4. 协方差分析 ----
    std::cout << "4. 目标点云协方差矩阵:" << std::endl;
    const plamatrix::RowVectorXf centroid = target_cpu.colwise().mean();
    const plamatrix::MatrixXf centered = target_cpu.rowwise() - centroid;
    plamatrix::Matrix3f cov;
    cov.noalias() = centered.transpose() * centered;
    cov /= static_cast<float>(N - 1);
    std::cout << "   [" << cov(0, 0) << " " << cov(0, 1) << " " << cov(0, 2) << "]" << std::endl;
    std::cout << "   [" << cov(1, 0) << " " << cov(1, 1) << " " << cov(1, 2) << "]" << std::endl;
    std::cout << "   [" << cov(2, 0) << " " << cov(2, 1) << " " << cov(2, 2) << "]" << std::endl;

    // 特征值分析 (PCA) — 球形点云→各向同性
    const plamatrix::SelfAdjointEigenSolver<plamatrix::Matrix3f> eigensolver(cov);
    const auto& eigenvalues = eigensolver.eigenvalues();
    std::cout << "   特征值: " << eigenvalues(0) << ", " << eigenvalues(1) << ", " << eigenvalues(2) << std::endl;
    std::cout << "   (接近相等→球形点云，各向同性)" << std::endl;

    std::cout << "\n=== 完成 ===" << std::endl;
    return 0;
}
