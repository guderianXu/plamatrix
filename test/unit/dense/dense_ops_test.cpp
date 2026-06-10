#include <gtest/gtest.h>

#include <omp.h>

#include <plamatrix/dense/dense_ops.h>

using namespace plamatrix;

TEST(DenseOps, add_CpuSerial_TwoByTwo)
{
    omp_set_num_threads(1);

    DenseMatrix<float, Device::CPU> A(2, 2);
    DenseMatrix<float, Device::CPU> B(2, 2);

    // A = [1 3; 2 4] in column-major
    A.setValue(0, 0, 1.0f);
    A.setValue(1, 0, 2.0f);
    A.setValue(0, 1, 3.0f);
    A.setValue(1, 1, 4.0f);

    // B = [5 7; 6 8] in column-major
    B.setValue(0, 0, 5.0f);
    B.setValue(1, 0, 6.0f);
    B.setValue(0, 1, 7.0f);
    B.setValue(1, 1, 8.0f);

    auto C = add(A, B);

    EXPECT_FLOAT_EQ(C(0, 0), 6.0f);
    EXPECT_FLOAT_EQ(C(1, 0), 8.0f);
    EXPECT_FLOAT_EQ(C(0, 1), 10.0f);
    EXPECT_FLOAT_EQ(C(1, 1), 12.0f);
}

TEST(DenseOps, sub_CpuSerial)
{
    omp_set_num_threads(1);

    DenseMatrix<float, Device::CPU> A(2, 2);
    DenseMatrix<float, Device::CPU> B(2, 2);

    // A = [10 30; 20 40] in column-major
    A.setValue(0, 0, 10.0f);
    A.setValue(1, 0, 20.0f);
    A.setValue(0, 1, 30.0f);
    A.setValue(1, 1, 40.0f);

    // B = [1 3; 2 4] in column-major
    B.setValue(0, 0, 1.0f);
    B.setValue(1, 0, 2.0f);
    B.setValue(0, 1, 3.0f);
    B.setValue(1, 1, 4.0f);

    auto C = sub(A, B);

    EXPECT_FLOAT_EQ(C(0, 0), 9.0f);
    EXPECT_FLOAT_EQ(C(1, 0), 18.0f);
    EXPECT_FLOAT_EQ(C(0, 1), 27.0f);
    EXPECT_FLOAT_EQ(C(1, 1), 36.0f);
}

TEST(DenseOps, add_RejectsDimensionMismatch)
{
    DenseMatrix<float, Device::CPU> A(2, 3);
    DenseMatrix<float, Device::CPU> B(3, 2);

    EXPECT_THROW(static_cast<void>(add(A, B)), std::runtime_error);
}

TEST(DenseOps, sub_RejectsDimensionMismatch)
{
    DenseMatrix<float, Device::CPU> A(2, 3);
    DenseMatrix<float, Device::CPU> B(3, 2);

    EXPECT_THROW(static_cast<void>(sub(A, B)), std::runtime_error);
}

TEST(DenseOps, add_EmptyCpuMatrices)
{
    DenseMatrix<float, Device::CPU> A(0, 3);
    DenseMatrix<float, Device::CPU> B(0, 3);

    auto C = add(A, B);
    EXPECT_EQ(C.rows(), 0);
    EXPECT_EQ(C.cols(), 3);
    EXPECT_EQ(C.size(), 0);
    EXPECT_EQ(C.data(), nullptr);
}

TEST(DenseOps, sub_EmptyCpuMatrices)
{
    DenseMatrix<float, Device::CPU> A(0, 3);
    DenseMatrix<float, Device::CPU> B(0, 3);

    auto C = sub(A, B);
    EXPECT_EQ(C.rows(), 0);
    EXPECT_EQ(C.cols(), 3);
    EXPECT_EQ(C.size(), 0);
    EXPECT_EQ(C.data(), nullptr);
}

#ifdef PLAMATRIX_WITH_CUDA
TEST(DenseOps, add_Gpu)
{
    DenseMatrix<float, Device::CPU> A_cpu(2, 2);
    DenseMatrix<float, Device::CPU> B_cpu(2, 2);

    // A = [1 3; 2 4] in column-major
    A_cpu.setValue(0, 0, 1.0f);
    A_cpu.setValue(1, 0, 2.0f);
    A_cpu.setValue(0, 1, 3.0f);
    A_cpu.setValue(1, 1, 4.0f);

    // B = [5 7; 6 8] in column-major
    B_cpu.setValue(0, 0, 5.0f);
    B_cpu.setValue(1, 0, 6.0f);
    B_cpu.setValue(0, 1, 7.0f);
    B_cpu.setValue(1, 1, 8.0f);

    auto A_gpu = A_cpu.toGpu();
    auto B_gpu = B_cpu.toGpu();
    auto C_gpu = add(A_gpu, B_gpu);
    auto C_cpu = C_gpu.toCpu();

    EXPECT_FLOAT_EQ(C_cpu(0, 0), 6.0f);
    EXPECT_FLOAT_EQ(C_cpu(1, 0), 8.0f);
    EXPECT_FLOAT_EQ(C_cpu(0, 1), 10.0f);
    EXPECT_FLOAT_EQ(C_cpu(1, 1), 12.0f);
}

TEST(DenseOps, addSub_GpuTwoArgumentOverloads)
{
    DenseMatrix<float, Device::CPU> A_cpu(2, 2);
    DenseMatrix<float, Device::CPU> B_cpu(2, 2);
    A_cpu.setValue(0, 0, 1.0f);
    A_cpu.setValue(1, 0, 2.0f);
    A_cpu.setValue(0, 1, 3.0f);
    A_cpu.setValue(1, 1, 4.0f);
    B_cpu.setValue(0, 0, 10.0f);
    B_cpu.setValue(1, 0, 20.0f);
    B_cpu.setValue(0, 1, 30.0f);
    B_cpu.setValue(1, 1, 40.0f);

    auto A_gpu = A_cpu.toGpu();
    auto B_gpu = B_cpu.toGpu();
    auto sum = add(A_gpu, B_gpu).toCpu();
    auto diff = sub(A_gpu, B_gpu).toCpu();

    EXPECT_FLOAT_EQ(sum(0, 0), 11.0f);
    EXPECT_FLOAT_EQ(sum(1, 0), 22.0f);
    EXPECT_FLOAT_EQ(sum(0, 1), 33.0f);
    EXPECT_FLOAT_EQ(sum(1, 1), 44.0f);
    EXPECT_FLOAT_EQ(diff(0, 0), -9.0f);
    EXPECT_FLOAT_EQ(diff(1, 0), -18.0f);
    EXPECT_FLOAT_EQ(diff(0, 1), -27.0f);
    EXPECT_FLOAT_EQ(diff(1, 1), -36.0f);
}

TEST(DenseOps, addSub_GpuOutputReuse)
{
    DenseMatrix<float, Device::CPU> A_cpu(2, 2);
    DenseMatrix<float, Device::CPU> B_cpu(2, 2);
    A_cpu.setValue(0, 0, 1.0f);
    A_cpu.setValue(1, 0, 2.0f);
    A_cpu.setValue(0, 1, 3.0f);
    A_cpu.setValue(1, 1, 4.0f);
    B_cpu.setValue(0, 0, 10.0f);
    B_cpu.setValue(1, 0, 20.0f);
    B_cpu.setValue(0, 1, 30.0f);
    B_cpu.setValue(1, 1, 40.0f);

    auto A_gpu = A_cpu.toGpu();
    auto B_gpu = B_cpu.toGpu();
    DenseMatrix<float, Device::GPU> sum_gpu(2, 2);
    DenseMatrix<float, Device::GPU> diff_gpu(2, 2);

    add(A_gpu, B_gpu, sum_gpu);
    sub(A_gpu, B_gpu, diff_gpu);

    auto sum = sum_gpu.toCpu();
    auto diff = diff_gpu.toCpu();
    EXPECT_FLOAT_EQ(sum(0, 0), 11.0f);
    EXPECT_FLOAT_EQ(sum(1, 0), 22.0f);
    EXPECT_FLOAT_EQ(sum(0, 1), 33.0f);
    EXPECT_FLOAT_EQ(sum(1, 1), 44.0f);
    EXPECT_FLOAT_EQ(diff(0, 0), -9.0f);
    EXPECT_FLOAT_EQ(diff(1, 0), -18.0f);
    EXPECT_FLOAT_EQ(diff(0, 1), -27.0f);
    EXPECT_FLOAT_EQ(diff(1, 1), -36.0f);
}

TEST(DenseOps, addSubAsync_GpuDefaultStream)
{
    DenseMatrix<float, Device::CPU> A_cpu(2, 2);
    DenseMatrix<float, Device::CPU> B_cpu(2, 2);
    A_cpu.setValue(0, 0, 1.0f);
    A_cpu.setValue(1, 0, 2.0f);
    A_cpu.setValue(0, 1, 3.0f);
    A_cpu.setValue(1, 1, 4.0f);
    B_cpu.setValue(0, 0, 10.0f);
    B_cpu.setValue(1, 0, 20.0f);
    B_cpu.setValue(0, 1, 30.0f);
    B_cpu.setValue(1, 1, 40.0f);

    auto A_gpu = A_cpu.toGpu();
    auto B_gpu = B_cpu.toGpu();
    auto sum = addAsync(A_gpu, B_gpu).toCpu();
    auto diff = subAsync(A_gpu, B_gpu).toCpu();

    EXPECT_FLOAT_EQ(sum(0, 0), 11.0f);
    EXPECT_FLOAT_EQ(sum(1, 0), 22.0f);
    EXPECT_FLOAT_EQ(sum(0, 1), 33.0f);
    EXPECT_FLOAT_EQ(sum(1, 1), 44.0f);
    EXPECT_FLOAT_EQ(diff(0, 0), -9.0f);
    EXPECT_FLOAT_EQ(diff(1, 0), -18.0f);
    EXPECT_FLOAT_EQ(diff(0, 1), -27.0f);
    EXPECT_FLOAT_EQ(diff(1, 1), -36.0f);
}

TEST(DenseOps, add_GpuRejectsOutputDimensionMismatch)
{
    DenseMatrix<float, Device::CPU> A_cpu(2, 2);
    DenseMatrix<float, Device::CPU> B_cpu(2, 2);

    auto A_gpu = A_cpu.toGpu();
    auto B_gpu = B_cpu.toGpu();
    DenseMatrix<float, Device::GPU> C_gpu(2, 3);

    EXPECT_THROW(add(A_gpu, B_gpu, C_gpu), std::runtime_error);
    EXPECT_THROW(addAsync(A_gpu, B_gpu, C_gpu), std::runtime_error);
}

TEST(DenseOps, add_RejectsGpuDimensionMismatch)
{
    DenseMatrix<float, Device::CPU> A_cpu(2, 3);
    DenseMatrix<float, Device::CPU> B_cpu(3, 2);

    auto A_gpu = A_cpu.toGpu();
    auto B_gpu = B_cpu.toGpu();

    EXPECT_THROW(static_cast<void>(add(A_gpu, B_gpu)), std::runtime_error);
}

TEST(DenseOps, sub_RejectsGpuDimensionMismatch)
{
    DenseMatrix<float, Device::CPU> A_cpu(2, 3);
    DenseMatrix<float, Device::CPU> B_cpu(3, 2);

    auto A_gpu = A_cpu.toGpu();
    auto B_gpu = B_cpu.toGpu();

    EXPECT_THROW(static_cast<void>(sub(A_gpu, B_gpu)), std::runtime_error);
}

TEST(DenseOps, add_EmptyGpuMatrices)
{
    DenseMatrix<float, Device::CPU> A_cpu(0, 3);
    DenseMatrix<float, Device::CPU> B_cpu(0, 3);

    auto C_gpu = add(A_cpu.toGpu(), B_cpu.toGpu());
    EXPECT_EQ(C_gpu.rows(), 0);
    EXPECT_EQ(C_gpu.cols(), 3);
    EXPECT_EQ(C_gpu.size(), 0);
    EXPECT_EQ(C_gpu.data(), nullptr);
}

TEST(DenseOps, sub_EmptyGpuMatrices)
{
    DenseMatrix<float, Device::CPU> A_cpu(0, 3);
    DenseMatrix<float, Device::CPU> B_cpu(0, 3);

    auto C_gpu = sub(A_cpu.toGpu(), B_cpu.toGpu());
    EXPECT_EQ(C_gpu.rows(), 0);
    EXPECT_EQ(C_gpu.cols(), 3);
    EXPECT_EQ(C_gpu.size(), 0);
    EXPECT_EQ(C_gpu.data(), nullptr);
}
#endif

// === Task 8: Transpose and Scalar Operations ===

TEST(DenseOps, transpose_Cpu)
{
    omp_set_num_threads(1);

    // 2x3 matrix:
    // [1 3 5]
    // [2 4 6]
    DenseMatrix<float, Device::CPU> A(2, 3);
    A.setValue(0, 0, 1.0f);
    A.setValue(1, 0, 2.0f);
    A.setValue(0, 1, 3.0f);
    A.setValue(1, 1, 4.0f);
    A.setValue(0, 2, 5.0f);
    A.setValue(1, 2, 6.0f);

    auto T = A.transpose();

    // Expected 3x2:
    // [1 2]
    // [3 4]
    // [5 6]
    EXPECT_EQ(T.rows(), 3);
    EXPECT_EQ(T.cols(), 2);
    EXPECT_FLOAT_EQ(T(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(T(0, 1), 2.0f);
    EXPECT_FLOAT_EQ(T(1, 0), 3.0f);
    EXPECT_FLOAT_EQ(T(1, 1), 4.0f);
    EXPECT_FLOAT_EQ(T(2, 0), 5.0f);
    EXPECT_FLOAT_EQ(T(2, 1), 6.0f);
}

TEST(DenseOps, scalarMultiply_Cpu)
{
    omp_set_num_threads(1);

    // 2x2 matrix:
    // [1 3]
    // [2 4]
    DenseMatrix<float, Device::CPU> A(2, 2);
    A.setValue(0, 0, 1.0f);
    A.setValue(1, 0, 2.0f);
    A.setValue(0, 1, 3.0f);
    A.setValue(1, 1, 4.0f);

    auto C = 2.0f * A;

    EXPECT_FLOAT_EQ(C(0, 0), 2.0f);
    EXPECT_FLOAT_EQ(C(1, 0), 4.0f);
    EXPECT_FLOAT_EQ(C(0, 1), 6.0f);
    EXPECT_FLOAT_EQ(C(1, 1), 8.0f);
}

TEST(DenseOps, scalarMultiplyAdd_Cpu)
{
    omp_set_num_threads(1);

    // A = [1 3; 2 4], B = [0.5 0.5; 0.5 0.5]
    DenseMatrix<float, Device::CPU> A(2, 2);
    A.setValue(0, 0, 1.0f);
    A.setValue(1, 0, 2.0f);
    A.setValue(0, 1, 3.0f);
    A.setValue(1, 1, 4.0f);

    DenseMatrix<float, Device::CPU> B(2, 2);
    B.setValue(0, 0, 0.5f);
    B.setValue(1, 0, 0.5f);
    B.setValue(0, 1, 0.5f);
    B.setValue(1, 1, 0.5f);

    auto C = add(2.0f * A, B);

    EXPECT_FLOAT_EQ(C(0, 0), 2.5f);   // 1*2 + 0.5
    EXPECT_FLOAT_EQ(C(1, 0), 4.5f);   // 2*2 + 0.5
    EXPECT_FLOAT_EQ(C(0, 1), 6.5f);   // 3*2 + 0.5
    EXPECT_FLOAT_EQ(C(1, 1), 8.5f);   // 4*2 + 0.5
}

#ifdef PLAMATRIX_WITH_CUDA
TEST(DenseOps, transpose_Gpu)
{
    // 2x3 matrix on CPU, move to GPU, transpose, bring back
    DenseMatrix<float, Device::CPU> A_cpu(2, 3);
    A_cpu.setValue(0, 0, 1.0f);
    A_cpu.setValue(1, 0, 2.0f);
    A_cpu.setValue(0, 1, 3.0f);
    A_cpu.setValue(1, 1, 4.0f);
    A_cpu.setValue(0, 2, 5.0f);
    A_cpu.setValue(1, 2, 6.0f);

    auto A_gpu = A_cpu.toGpu();
    auto T_gpu = A_gpu.transpose();
    auto T_cpu = T_gpu.toCpu();

    // Expected 3x2 transpose
    EXPECT_EQ(T_cpu.rows(), 3);
    EXPECT_EQ(T_cpu.cols(), 2);
    EXPECT_FLOAT_EQ(T_cpu(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(T_cpu(0, 1), 2.0f);
    EXPECT_FLOAT_EQ(T_cpu(1, 0), 3.0f);
    EXPECT_FLOAT_EQ(T_cpu(1, 1), 4.0f);
    EXPECT_FLOAT_EQ(T_cpu(2, 0), 5.0f);
    EXPECT_FLOAT_EQ(T_cpu(2, 1), 6.0f);
}
#endif
