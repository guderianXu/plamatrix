// ============================================================================
// PlaMatrix 示例 3: 稀疏矩阵 — 三对角系统求解
//
// 编译: g++ -std=c++17 -O2 -I../include -fopenmp sparse-solver.cpp
//       -L../build -lplamatrix -o sparse-solver
//
// 演示: COO 构建 → CSR 转换 → 稀疏求解 → GPU 加速
//       模拟点云 BA (Bundle Adjustment) 中的稀疏线性系统
// ============================================================================

#include <chrono>
#include <iostream>
#include <random>

#include <plamatrix/plamatrix.h>
using namespace plamatrix;

int main()
{
    constexpr Index N = 500;
    std::cout << "=== 稀疏求解示例 (N=" << N << ") ===\n" << std::endl;

    // ---- 1. 构建三对角矩阵 (COO 格式) ----
    // 实际点云 BA 的 Hessian 矩阵具有带状稀疏结构
    std::cout << "1. 构建 " << N << "x" << N << " 三对角稀疏矩阵..." << std::endl;
    COOMatrix<float, Device::CPU> coo(N, N);

    coo.add(0, 0, 2.0f);
    coo.add(0, 1, -1.0f);
    for (Index i = 1; i < N - 1; ++i)
    {
        coo.add(i, i-1, -1.0f);
        coo.add(i, i,   2.0f);
        coo.add(i, i+1, -1.0f);
    }
    coo.add(N-1, N-2, -1.0f);
    coo.add(N-1, N-1,  2.0f);

    std::cout << "   非零元个数: " << coo.nnz() << " (密集矩阵需要 "
              << (N * N) << " → 压缩比 "
              << static_cast<float>(coo.nnz()) / (N * N) * 100 << "%)" << std::endl;

    // ---- 2. COO → CSR 转换 ----
    std::cout << "2. COO → CSR 转换..." << std::endl;
    auto csr = coo.toCsr();

    std::cout << "   CSR row_offsets[0..5]: ";
    for (Index i = 0; i < 5; ++i) std::cout << csr.rowOffsets()[i] << " ";
    std::cout << "..." << std::endl;

    // ---- 3. 构建右端项 ----
    std::cout << "3. 构建右端项..." << std::endl;
    DenseMatrix<float, Device::CPU> b(N, 1);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (Index i = 0; i < N; ++i) b(i, 0) = dist(rng);

    // ---- 4. CPU 稠密求解 (对照) ----
    std::cout << "4. CPU 稠密求解 (对照)..." << std::endl;
    DenseMatrix<float, Device::CPU> A_dense(N, N);
    A_dense(0, 0) = 2; A_dense(0, 1) = -1;
    for (Index i = 1; i < N - 1; ++i)
    {
        A_dense(i, i-1) = -1; A_dense(i, i) = 2; A_dense(i, i+1) = -1;
    }
    A_dense(N-1, N-2) = -1; A_dense(N-1, N-1) = 2;

    auto t1 = std::chrono::high_resolution_clock::now();
    auto x_dense = solve<float, Device::CPU>(A_dense, b);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto dense_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    std::cout << "   稠密求解: " << dense_ms << " ms" << std::endl;

    // ---- 5. 验证: 计算残差 ||A*x - b|| ----
    auto residual = gemm(A_dense, x_dense);
    float max_residual = 0.0f;
    for (Index i = 0; i < N; ++i)
    {
        float r = std::abs(residual(i,0) - b(i,0));
        if (r > max_residual) max_residual = r;
    }
    std::cout << "   最大残差: " << max_residual << std::endl;

    // ---- 6. GPU 稀疏求解 ----
    std::cout << "5. GPU 稀疏求解..." << std::endl;
    auto coo_gpu = COOMatrix<float, Device::GPU>(N, N);
    coo_gpu.add(0, 0, 2.0f); coo_gpu.add(0, 1, -1.0f);
    for (Index i = 1; i < N - 1; ++i)
    {
        coo_gpu.add(i, i-1, -1.0f);
        coo_gpu.add(i, i,   2.0f);
        coo_gpu.add(i, i+1, -1.0f);
    }
    coo_gpu.add(N-1, N-2, -1.0f); coo_gpu.add(N-1, N-1, 2.0f);

    auto csr_gpu = coo_gpu.toCsr();
    auto b_gpu = b.toGpu();
    auto t3 = std::chrono::high_resolution_clock::now();
    auto x_gpu = solve(csr_gpu, b_gpu);
    cudaDeviceSynchronize();
    auto t4 = std::chrono::high_resolution_clock::now();
    auto x_result = x_gpu.toCpu();
    auto gpu_ms = std::chrono::duration<double, std::milli>(t4 - t3).count();
    std::cout << "   GPU 稀疏求解: " << gpu_ms << " ms" << std::endl;
    std::cout << "   解的前5个元素: ";
    for (Index i = 0; i < 5; ++i) std::cout << x_result(i, 0) << " ";
    std::cout << std::endl;

    std::cout << "\n=== 完成 ===" << std::endl;
    return 0;
}
