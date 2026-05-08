// ============================================================================
// PlaMatrix 示例 1: 基本矩阵运算
//
// 编译: g++ -std=c++17 -O2 -I../include -fopenmp basic-operations.cpp
//       -L../build -lplamatrix -o basic-operations
//
// 演示: 矩阵构造、填充、加减、标量运算、转置、GEMM、CPU/GPU 传输
// ============================================================================

#include <chrono>
#include <iostream>

#include <plamatrix/plamatrix.h>
using namespace plamatrix;

int main()
{
    constexpr Index N = 1024;
    std::cout << "=== PlaMatrix 基本运算示例 (N=" << N << ") ===\n" << std::endl;

    // ---- 1. 构造 + 填充 ----
    std::cout << "1. 创建两个 " << N << "x" << N << " 矩阵..." << std::endl;
    DenseMatrix<float, Device::CPU> A(N, N);
    DenseMatrix<float, Device::CPU> B(N, N);

    for (Index j = 0; j < N; ++j)
    {
        for (Index i = 0; i < N; ++i)
        {
            A(i, j) = static_cast<float>(i + j) / N;
            B(i, j) = static_cast<float>(i * j % 7) / 7.0f;
        }
    }
    std::cout << "   A(0,0)=" << A(0,0) << " A(512,512)=" << A(512,512) << std::endl;

    // ---- 2. 逐元素加法 (自动 OpenMP 并行) ----
    std::cout << "2. 逐元素加法 A + B ..." << std::endl;
    auto t1 = std::chrono::high_resolution_clock::now();
    auto C = add(A, B);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    std::cout << "   CPU add: " << ms << " ms, C(0,0)=" << C(0,0) << std::endl;

    // ---- 3. 标量运算 ----
    std::cout << "3. 标量运算 2.0f * A + B ..." << std::endl;
    auto D = 2.0f * A + B;
    std::cout << "   D(0,0) = 2*" << A(0,0) << "+" << B(0,0)
              << " = " << D(0,0) << std::endl;

    // ---- 4. 转置 ----
    std::cout << "4. 矩阵转置 ..." << std::endl;
    auto At = A.transpose();
    std::cout << "   A(2,5)=" << A(2,5) << "  At(5,2)=" << At(5,2) << std::endl;

    // ---- 5. GPU 矩阵乘法 ----
    std::cout << "5. GPU 矩阵乘法 A * B ..." << std::endl;
    auto A_gpu = A.toGpu();
    auto B_gpu = B.toGpu();
    auto t3 = std::chrono::high_resolution_clock::now();
    auto C_gpu = gemm(A_gpu, B_gpu);
    cudaDeviceSynchronize();
    auto t4 = std::chrono::high_resolution_clock::now();
    auto gpu_ms = std::chrono::duration<double, std::milli>(t4 - t3).count();
    auto C_result = C_gpu.toCpu();
    std::cout << "   GPU gemm: " << gpu_ms << " ms" << std::endl;
    std::cout << "   C(" << N/2 << "," << N/2 << ") = " << C_result(N/2, N/2) << std::endl;

    // ---- 6. CPU 对比 ----
    std::cout << "6. CPU 矩阵乘法 (对比) ..." << std::endl;
    omp_set_num_threads(4);
    auto t5 = std::chrono::high_resolution_clock::now();
    auto C_cpu = gemm(A, B);
    auto t6 = std::chrono::high_resolution_clock::now();
    auto cpu_ms = std::chrono::duration<double, std::milli>(t6 - t5).count();
    std::cout << "   CPU gemm (4 threads): " << cpu_ms << " ms" << std::endl;

    // ---- 7. 结果一致性 ----
    float max_diff = 0.0f;
    for (Index j = 0; j < N; ++j)
    {
        for (Index i = 0; i < N; ++i)
        {
            float diff = std::abs(C_cpu(i,j) - C_result(i,j));
            if (diff > max_diff) max_diff = diff;
        }
    }
    std::cout << "\n7. GPU vs CPU 最大误差: " << max_diff
              << " (加速比: " << cpu_ms / gpu_ms << "x)" << std::endl;

    std::cout << "\n=== 完成 ===" << std::endl;
    return 0;
}
