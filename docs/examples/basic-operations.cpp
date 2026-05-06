/**
 * PlaMatrix 示例：基础矩阵运算
 *
 * 演示内容：
 * - DenseMatrix 的构造与填充
 * - 元素访问 (operator() / setValue / getValue)
 * - CPU ↔ GPU 数据传输
 * - 矩阵加法、减法、标量乘法
 * - 矩阵转置
 * - GEMM 矩阵乘法
 *
 * 编译：
 *   g++ -std=c++17 -O2 -fopenmp basic-operations.cpp \
 *       -I<plamatrix_include_dir> -L<plamatrix_lib_dir> -lplamatrix \
 *       -lcublas -lcusolver -lcudart -o basic-operations
 */

#include <plamatrix/plamatrix.h>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace plamatrix;

int main()
{
    try
    {
        // ============================================================
        // 1. 构造矩阵
        // ============================================================
        std::cout << "=== 1. 构造矩阵 ===" << std::endl;

        // 创建 3x3 矩阵（CPU，全零初始化）
        DenseMatrix<float, Device::CPU> A(3, 3);
        DenseMatrix<float, Device::CPU> B(3, 3);

        std::cout << "A: " << A.rows() << "x" << A.cols() << std::endl;
        std::cout << "B: " << B.rows() << "x" << B.cols() << std::endl;

        // ============================================================
        // 2. 填充矩阵（单位矩阵和全 1 矩阵）
        // ============================================================
        std::cout << "\n=== 2. 填充矩阵 ===" << std::endl;

        // A = 单位矩阵
        for (Index i = 0; i < 3; ++i)
            A(i, i) = 1.0f;

        // B = 全 1 矩阵
        B.fill(2.0f);

        std::cout << "A (单位矩阵):" << std::endl;
        for (Index i = 0; i < 3; ++i)
        {
            for (Index j = 0; j < 3; ++j)
                std::cout << A(i, j) << " ";
            std::cout << std::endl;
        }

        std::cout << "B (全 2):" << std::endl;
        for (Index i = 0; i < 3; ++i)
        {
            for (Index j = 0; j < 3; ++j)
                std::cout << B(i, j) << " ";
            std::cout << std::endl;
        }

        // ============================================================
        // 3. 矩阵加法与减法
        // ============================================================
        std::cout << "\n=== 3. 矩阵加法与减法 ===" << std::endl;

        auto C_add = add(A, B);  // A + B = I + 2*ones
        std::cout << "A + B:" << std::endl;
        for (Index i = 0; i < 3; ++i)
        {
            for (Index j = 0; j < 3; ++j)
                std::cout << C_add(i, j) << " ";
            std::cout << std::endl;
        }

        auto C_sub = sub(A, B);  // A - B = I - 2*ones
        std::cout << "A - B:" << std::endl;
        for (Index i = 0; i < 3; ++i)
        {
            for (Index j = 0; j < 3; ++j)
                std::cout << C_sub(i, j) << " ";
            std::cout << std::endl;
        }

        // ============================================================
        // 4. 标量运算
        // ============================================================
        std::cout << "\n=== 4. 标量运算 ===" << std::endl;

        auto C_scale = 3.0f * A;  // 每个元素乘以 3
        std::cout << "3 * A:" << std::endl;
        for (Index i = 0; i < 3; ++i)
        {
            for (Index j = 0; j < 3; ++j)
                std::cout << C_scale(i, j) << " ";
            std::cout << std::endl;
        }

        auto C_offset = 5.0f + A;  // 每个元素加 5
        std::cout << "5 + A:" << std::endl;
        for (Index i = 0; i < 3; ++i)
        {
            for (Index j = 0; j < 3; ++j)
                std::cout << C_offset(i, j) << " ";
            std::cout << std::endl;
        }

        // ============================================================
        // 5. 矩阵转置
        // ============================================================
        std::cout << "\n=== 5. 矩阵转置 ===" << std::endl;

        DenseMatrix<float, Device::CPU> M(2, 3);
        M(0, 0) = 1.0f; M(0, 1) = 2.0f; M(0, 2) = 3.0f;
        M(1, 0) = 4.0f; M(1, 1) = 5.0f; M(1, 2) = 6.0f;

        std::cout << "M (2x3):" << std::endl;
        for (Index i = 0; i < 2; ++i)
        {
            for (Index j = 0; j < 3; ++j)
                std::cout << M(i, j) << " ";
            std::cout << std::endl;
        }

        auto MT = M.transpose();
        std::cout << "M^T (3x2):" << std::endl;
        for (Index i = 0; i < 3; ++i)
        {
            for (Index j = 0; j < 2; ++j)
                std::cout << MT(i, j) << " ";
            std::cout << std::endl;
        }

        // ============================================================
        // 6. GEMM 矩阵乘法
        // ============================================================
        std::cout << "\n=== 6. GEMM 矩阵乘法 ===" << std::endl;

        DenseMatrix<float, Device::CPU> P(2, 3);  // 2x3
        P(0, 0) = 1.0f; P(0, 1) = 0.0f; P(0, 2) = 2.0f;
        P(1, 0) = 0.0f; P(1, 1) = 3.0f; P(1, 2) = 0.0f;

        DenseMatrix<float, Device::CPU> Q(3, 2);  // 3x2
        Q(0, 0) = 1.0f; Q(0, 1) = 2.0f;
        Q(1, 0) = 3.0f; Q(1, 1) = 4.0f;
        Q(2, 0) = 5.0f; Q(2, 1) = 6.0f;

        auto R = gemm(P, Q);  // P(2x3) * Q(3x2) = R(2x2)
        std::cout << "P:" << std::endl;
        for (Index i = 0; i < 2; ++i)
        {
            for (Index j = 0; j < 3; ++j)
                std::cout << P(i, j) << " ";
            std::cout << std::endl;
        }
        std::cout << "Q:" << std::endl;
        for (Index i = 0; i < 3; ++i)
        {
            for (Index j = 0; j < 2; ++j)
                std::cout << Q(i, j) << " ";
            std::cout << std::endl;
        }
        std::cout << "P * Q = R (2x2):" << std::endl;
        for (Index i = 0; i < 2; ++i)
        {
            for (Index j = 0; j < 2; ++j)
                std::cout << R(i, j) << " ";
            std::cout << std::endl;
        }

        // ============================================================
        // 7. CPU ↔ GPU 数据传输与 GPU GEMM
        // ============================================================
        std::cout << "\n=== 7. CPU ↔ GPU 数据传输与 GPU GEMM ===" << std::endl;

        // 将 CPU 矩阵上传到 GPU
        auto P_gpu = P.toGpu();
        auto Q_gpu = Q.toGpu();

        // 在 GPU 上执行 GEMM（使用 cuBLAS）
        auto R_gpu = gemm(P_gpu, Q_gpu);

        // 将结果下载回 CPU
        auto R_from_gpu = R_gpu.toCpu();

        std::cout << "GPU GEMM 结果:" << std::endl;
        for (Index i = 0; i < 2; ++i)
        {
            for (Index j = 0; j < 2; ++j)
                std::cout << R_from_gpu(i, j) << " ";
            std::cout << std::endl;
        }

        // 验证 CPU 与 GPU 结果一致
        float max_diff = 0.0f;
        for (Index i = 0; i < 2; ++i)
            for (Index j = 0; j < 2; ++j)
                max_diff = std::max(max_diff, std::abs(R(i, j) - R_from_gpu(i, j)));

        std::cout << "CPU vs GPU 最大误差: " << max_diff << std::endl;
        if (max_diff < 1e-5f)
            std::cout << "验证通过: CPU 与 GPU 结果一致" << std::endl;
        else
            std::cout << "警告: 存在较大误差" << std::endl;

        // ============================================================
        // 8. setValue / getValue（GPU 单元素操作）
        // ============================================================
        std::cout << "\n=== 8. GPU 单元素操作 ===" << std::endl;

        DenseMatrix<float, Device::GPU> X_gpu(4, 4);
        X_gpu.setValue(0, 0, 42.0f);   // 在 GPU 上设置单个值
        X_gpu.setValue(3, 3, 99.0f);
        float v0 = X_gpu.getValue(0, 0);  // 从 GPU 读取单个值
        float v1 = X_gpu.getValue(3, 3);
        std::cout << "X(0,0) = " << v0 << ", X(3,3) = " << v1 << std::endl;

        std::cout << "\n所有示例执行完成!" << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
