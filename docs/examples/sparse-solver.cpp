// ============================================================================
// PlaMatrix 示例 3: 稀疏矩阵 — COO 到 CSR 转换
//
// 编译: g++ -std=c++17 -O2 -Iinclude -fopenmp docs/examples/sparse-solver.cpp
//       -Lbuild -lplamatrix -o sparse-solver
//
// 演示: COO 构建 → CSR 转换 → CSR 结构检查 → 稠密求解对照
// ============================================================================

#include <chrono>
#include <cmath>
#include <iostream>
#include <random>

#include <plamatrix/plamatrix.h>
using namespace plamatrix;

int main()
{
    constexpr Index N = 500;
    std::cout << "=== 稀疏矩阵转换示例 (N=" << N << ") ===\n" << std::endl;

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
    for (Index i = 0; i < 5; ++i)
    {
        std::cout << csr.rowOffsets()[i] << " ";
    }
    std::cout << "..." << std::endl;

    // ---- 3. 构建右端项 ----
    std::cout << "3. 构建右端项..." << std::endl;
    DenseMatrix<float, Device::CPU> b(N, 1);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (Index i = 0; i < N; ++i)
    {
        b(i, 0) = dist(rng);
    }

    // ---- 4. CPU 稠密求解 (对照) ----
    std::cout << "4. CPU 稠密求解 (对照)..." << std::endl;
    DenseMatrix<float, Device::CPU> A_dense(N, N);
    A_dense(0, 0) = 2;
    A_dense(0, 1) = -1;
    for (Index i = 1; i < N - 1; ++i)
    {
        A_dense(i, i - 1) = -1;
        A_dense(i, i) = 2;
        A_dense(i, i + 1) = -1;
    }
    A_dense(N - 1, N - 2) = -1;
    A_dense(N - 1, N - 1) = 2;

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
        float r = std::abs(residual(i, 0) - b(i, 0));
        if (r > max_residual)
        {
            max_residual = r;
        }
    }
    std::cout << "   最大残差: " << max_residual << std::endl;

    // ---- 6. CSR 结构检查 ----
    std::cout << "6. CSR 结构检查..." << std::endl;
    std::cout << "   row_offsets[N] = " << csr.rowOffsets()[N]
              << ", nnz = " << csr.nnz() << std::endl;
    std::cout << "   前 5 个 values: ";
    for (Index i = 0; i < 5 && i < csr.nnz(); ++i)
    {
        std::cout << csr.values()[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "\n=== 完成 ===" << std::endl;
    return 0;
}
