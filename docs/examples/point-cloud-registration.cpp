// ============================================================================
// PlaMatrix 示例 2: 点云刚体变换与配准
//
// 编译: g++ -std=c++17 -O2 -I../include -fopenmp point-cloud-registration.cpp
//       -L../build -lplamatrix -o point-cloud-registration
//
// 演示: Rodrigues 旋转 → 刚体变换 → 批量点变换 → GPU 加速
//       模拟将源点云通过已知变换配准到目标坐标系
// ============================================================================

#include <chrono>
#include <cmath>
#include <iostream>
#include <random>

#include <plamatrix/plamatrix.h>
using namespace plamatrix;

// 生成球形点云 (N 个随机方向上的点)
DenseMatrix<float, Device::CPU> generateSphere(Index N, float radius)
{
    DenseMatrix<float, Device::CPU> pts(N, 3);
    std::mt19937 rng(12345);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    for (Index i = 0; i < N; ++i)
    {
        float x = normal(rng);
        float y = normal(rng);
        float z = normal(rng);
        float len = std::sqrt(x*x + y*y + z*z);
        pts(i, 0) = radius * x / len;
        pts(i, 1) = radius * y / len;
        pts(i, 2) = radius * z / len;
    }
    return pts;
}

int main()
{
    constexpr Index N = 100000;
    std::cout << "=== 点云刚体变换示例 (N=" << N << ") ===\n" << std::endl;

    // ---- 1. 生成球形点云 ----
    std::cout << "1. 生成 " << N << " 个点的球形点云..." << std::endl;
    auto source_pts = generateSphere(N, 5.0f);
    std::cout << "   前三点: (" << source_pts(0,0) << "," << source_pts(0,1) << "," << source_pts(0,2) << ")"
              << " (" << source_pts(1,0) << "," << source_pts(1,1) << "," << source_pts(1,2) << ")" << std::endl;

    // ---- 2. 构建变换: 绕 Z 轴旋转 45°, 平移 (10, 5, 3) ----
    std::cout << "2. 构建刚体变换 (绕Z轴45度, 平移10,5,3)..." << std::endl;
    Vec3<float> axis{0.0f, 0.0f, 1.0f};
    float angle = 0.785398f;  // 45 degrees
    auto R = rotationMatrix<float, Device::CPU>(axis, angle);
    std::cout << "   旋转矩阵:" << std::endl;
    std::cout << "   [" << R(0,0) << " " << R(0,1) << " " << R(0,2) << "]" << std::endl;
    std::cout << "   [" << R(1,0) << " " << R(1,1) << " " << R(1,2) << "]" << std::endl;
    std::cout << "   [" << R(2,0) << " " << R(2,1) << " " << R(2,2) << "]" << std::endl;

    Vec3<float> translation{10.0f, 5.0f, 3.0f};
    auto T = rigidTransform<float, Device::CPU>(R, translation);

    // ---- 3. CPU 变换 ----
    std::cout << "3. CPU 批量点变换..." << std::endl;
    auto t1 = std::chrono::high_resolution_clock::now();
    auto target_cpu = transformPoints<float, Device::CPU>(T, source_pts);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto cpu_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    std::cout << "   CPU: " << cpu_ms << " ms" << std::endl;
    std::cout << "   变换后前三点: (" << target_cpu(0,0) << "," << target_cpu(0,1) << "," << target_cpu(0,2) << ")"
              << " (" << target_cpu(1,0) << "," << target_cpu(1,1) << "," << target_cpu(1,2) << ")" << std::endl;

    // ---- 4. GPU 变换 ----
    std::cout << "4. GPU 批量点变换..." << std::endl;
    auto pts_gpu = source_pts.toGpu();
    auto T_gpu = T.toGpu();
    auto t3 = std::chrono::high_resolution_clock::now();
    auto target_gpu = transformPoints<float, Device::GPU>(T_gpu, pts_gpu);
    cudaDeviceSynchronize();
    auto t4 = std::chrono::high_resolution_clock::now();
    auto target_result = target_gpu.toCpu();
    auto gpu_ms = std::chrono::duration<double, std::milli>(t4 - t3).count();
    std::cout << "   GPU: " << gpu_ms << " ms (加速 " << cpu_ms / gpu_ms << "x)" << std::endl;

    // ---- 5. 一致性验证 ----
    float max_err = 0.0f;
    for (Index i = 0; i < N; ++i)
    {
        for (Index c = 0; c < 3; ++c)
        {
            float err = std::abs(target_cpu(i,c) - target_result(i,c));
            if (err > max_err) max_err = err;
        }
    }
    std::cout << "5. CPU vs GPU 最大误差: " << max_err << std::endl;

    // ---- 6. 协方差分析 ----
    std::cout << "6. 目标点云协方差矩阵:" << std::endl;
    auto cov = covarianceMatrix<float, Device::CPU>(target_cpu);
    std::cout << "   [" << cov(0,0) << " " << cov(0,1) << " " << cov(0,2) << "]" << std::endl;
    std::cout << "   [" << cov(1,0) << " " << cov(1,1) << " " << cov(1,2) << "]" << std::endl;
    std::cout << "   [" << cov(2,0) << " " << cov(2,1) << " " << cov(2,2) << "]" << std::endl;

    // 特征值分析 (PCA) — 球形点云→各向同性
    auto eigenvals = eigh(cov);
    std::cout << "   特征值: " << eigenvals(0,0) << ", " << eigenvals(1,0) << ", " << eigenvals(2,0) << std::endl;
    std::cout << "   (接近相等→球形点云，各向同性)" << std::endl;

    std::cout << "\n=== 完成 ===" << std::endl;
    return 0;
}
