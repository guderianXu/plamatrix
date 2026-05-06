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
