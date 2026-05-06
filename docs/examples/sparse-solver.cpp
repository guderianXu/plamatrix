/**
 * PlaMatrix 示例：稀疏矩阵构建与线性方程组求解
 *
 * 演示内容：
 * - COO 格式构建稀疏矩阵
 * - COO → CSR 格式转换
 * - 稠密线性方程组求解
 *
 * 场景：类似 Bundle Adjustment 的稀疏 Hessian 矩阵构建，
 * 演示稀疏问题转化为稠密求解的过程。
 *
 * 编译：
 *   g++ -std=c++17 -O2 -fopenmp sparse-solver.cpp \
 *       -I<plamatrix_include_dir> -L<plamatrix_lib_dir> -lplamatrix \
 *       -lcublas -lcusolver -lcudart -o sparse-solver
 */

#include <plamatrix/plamatrix.h>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace plamatrix;

int main()
{
    try
    {
        // ============================================================
        // 1. 构建三对角稀疏矩阵（使用 COO 格式）
        // ============================================================
        std::cout << "=== 1. 构建稀疏三对角矩阵 (COO) ===" << std::endl;

        const Index n = 5;  // 5x5 矩阵

        COOMatrix<double, Device::CPU> coo(n, n);

        // 构建经典三对角矩阵:
        //   [4 -1  0  0  0]
        //   [-1 4 -1  0  0]
        //   [0 -1  4 -1  0]
        //   [0  0 -1  4 -1]
        //   [0  0  0 -1  4]
        for (Index i = 0; i < n; ++i)
        {
            coo.add(i, i, 4.0);  // 对角线

            if (i > 0)
                coo.add(i, i - 1, -1.0);  // 下对角线

            if (i < n - 1)
                coo.add(i, i + 1, -1.0);  // 上对角线
        }

        std::cout << "非零元素数量: " << coo.nnz() << std::endl;
        std::cout << "期望非零元素数量: " << (3 * n - 2) << std::endl;

        // ============================================================
        // 2. 转换为 CSR 格式
        // ============================================================
        std::cout << "\n=== 2. 转换为 CSR 格式 ===" << std::endl;

        auto csr = coo.toCsr();
        std::cout << "CSR nnz = " << csr.nnz() << std::endl;

        // 打印 CSR 内部结构
        const double* vals = csr.values();
        const Index* col_idx = csr.colIndices();
        const Index* row_off = csr.rowOffsets();

        std::cout << "\nCSR 行偏移 (row_offsets): ";
        for (Index i = 0; i <= csr.rows(); ++i)
            std::cout << row_off[i] << " ";
        std::cout << std::endl;

        std::cout << "CSR 列索引 (col_indices): ";
        for (Index k = 0; k < csr.nnz(); ++k)
            std::cout << col_idx[k] << " ";
        std::cout << std::endl;

        std::cout << "CSR 值 (values): ";
        for (Index k = 0; k < csr.nnz(); ++k)
            std::cout << vals[k] << " ";
        std::cout << std::endl;

        // 按行输出非零元素
        std::cout << "\n按行输出:" << std::endl;
        for (Index r = 0; r < csr.rows(); ++r)
        {
            std::cout << "第 " << r << " 行: ";
            for (Index k = row_off[r]; k < row_off[r + 1]; ++k)
            {
                std::cout << "A(" << r << "," << col_idx[k] << ")=" << vals[k] << "  ";
            }
            std::cout << std::endl;
        }

        // ============================================================
        // 3. 构建稠密矩阵并求解 Ax = b
        // ============================================================
        std::cout << "\n=== 3. 稠密线性方程组求解 ===" << std::endl;

        // 将三对角矩阵转为稠密形式用于求解
        DenseMatrix<double, Device::CPU> A(n, n);
        for (Index i = 0; i < n; ++i)
        {
            A(i, i) = 4.0;
            if (i > 0)     A(i, i - 1) = -1.0;
            if (i < n - 1) A(i, i + 1) = -1.0;
        }

        // 右端项 b = [1, 2, 3, 4, 5]^T
        DenseMatrix<double, Device::CPU> b(n, 1);
        for (Index i = 0; i < n; ++i)
            b(i, 0) = static_cast<double>(i + 1);

        std::cout << "系数矩阵 A:" << std::endl;
        for (Index i = 0; i < n; ++i)
        {
            std::cout << "  ";
            for (Index j = 0; j < n; ++j)
                std::cout << A(i, j) << "\t";
            std::cout << std::endl;
        }

        std::cout << "右端项 b: [";
        for (Index i = 0; i < n; ++i)
            std::cout << b(i, 0) << (i < n - 1 ? ", " : "");
        std::cout << "]^T" << std::endl;

        // 求解 Ax = b
        auto x = solve(A, b);

        std::cout << "解向量 x: [";
        for (Index i = 0; i < n; ++i)
            std::cout << x(i, 0) << (i < n - 1 ? ", " : "");
        std::cout << "]^T" << std::endl;

        // 验证残差 ||Ax - b||
        auto Ax = gemm(A, x);  // Ax
        double residual_norm = 0.0;
        for (Index i = 0; i < n; ++i)
        {
            double diff = Ax(i, 0) - b(i, 0);
            residual_norm += diff * diff;
        }
        residual_norm = std::sqrt(residual_norm);

        std::cout << "残差 ||Ax - b||_2 = " << residual_norm << std::endl;
        if (residual_norm < 1e-10)
            std::cout << "求解精度满足要求 (残差 < 1e-10)" << std::endl;

        // ============================================================
        // 4. GPU 求解
        // ============================================================
        std::cout << "\n=== 4. GPU 线性求解 ===" << std::endl;

        auto A_gpu = A.toGpu();
        auto b_gpu = b.toGpu();
        auto x_gpu = solve(A_gpu, b_gpu);
        auto x_from_gpu = x_gpu.toCpu();

        // 比较 CPU 和 GPU 解
        double max_diff = 0.0;
        for (Index i = 0; i < n; ++i)
            max_diff = std::max(max_diff, std::abs(x(i, 0) - x_from_gpu(i, 0)));

        std::cout << "CPU vs GPU 解的差异: " << max_diff << std::endl;
        if (max_diff < 1e-10)
            std::cout << "验证通过: CPU 和 GPU 解一致" << std::endl;

        // ============================================================
        // 5. 多右端项求解
        // ============================================================
        std::cout << "\n=== 5. 多右端项求解 ===" << std::endl;

        DenseMatrix<double, Device::CPU> B_multi(n, 2);
        // 第一个右端项: [1, 2, 3, 4, 5]^T
        // 第二个右端项: [0, 1, 0, 1, 0]^T
        const double rhs1[] = {1.0, 2.0, 3.0, 4.0, 5.0};
        const double rhs2[] = {0.0, 1.0, 0.0, 1.0, 0.0};
        for (Index i = 0; i < n; ++i)
        {
            B_multi(i, 0) = rhs1[i];
            B_multi(i, 1) = rhs2[i];
        }

        auto X_multi = solve(A, B_multi);

        std::cout << "解矩阵 X (n x 2):" << std::endl;
        for (Index i = 0; i < n; ++i)
        {
            std::cout << "  x[" << i << "] = (" << X_multi(i, 0) << ", " << X_multi(i, 1) << ")" << std::endl;
        }

        // 验证两个方程组
        auto AX = gemm(A, X_multi);
        double res_norm_1 = 0.0, res_norm_2 = 0.0;
        for (Index i = 0; i < n; ++i)
        {
            double d1 = AX(i, 0) - B_multi(i, 0);
            double d2 = AX(i, 1) - B_multi(i, 1);
            res_norm_1 += d1 * d1;
            res_norm_2 += d2 * d2;
        }
        std::cout << "方程组 1 残差: " << std::sqrt(res_norm_1) << std::endl;
        std::cout << "方程组 2 残差: " << std::sqrt(res_norm_2) << std::endl;

        // ============================================================
        // 6. QR 分解验证
        // ============================================================
        std::cout << "\n=== 6. QR 分解验证 ===" << std::endl;

        // 对系数矩阵做 QR 分解
        auto [Q, R_mat] = qr(A);

        std::cout << "Q 为正交阵 (Q^T * Q ≈ I)?" << std::endl;
        auto QT = Q.transpose();
        auto QTQ = gemm(QT, Q);  // Q^T * Q 应该接近单位矩阵

        double i_error = 0.0;
        for (Index i = 0; i < n; ++i)
            for (Index j = 0; j < n; ++j)
            {
                double expected = (i == j) ? 1.0 : 0.0;
                i_error = std::max(i_error, std::abs(QTQ(i, j) - expected));
            }
        std::cout << "  ||Q^T Q - I||_max = " << i_error << std::endl;

        // 验证 A ≈ Q * R
        auto QR = gemm(Q, R_mat);
        double A_error = 0.0;
        for (Index i = 0; i < n; ++i)
            for (Index j = 0; j < n; ++j)
                A_error = std::max(A_error, std::abs(A(i, j) - QR(i, j)));
        std::cout << "  ||A - QR||_max = " << A_error << std::endl;

        if (A_error < 1e-10)
            std::cout << "QR 分解验证通过" << std::endl;

        std::cout << "\n所有稀疏求解示例执行完成!" << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
