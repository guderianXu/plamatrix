#include <cmath>
#include <cstdlib>

#include <gtest/gtest.h>
#include <omp.h>

#include <plamatrix/ops/gemm.h>

using namespace plamatrix;

// GEMM: multiply_2x3_by_3x2_CpuSerial
// A(2x3) * B(3x2) = C(2x2), single-threaded for deterministic output
TEST(GEMM, multiply_2x3_by_3x2_CpuSerial)
{
    omp_set_num_threads(1);

    DenseMatrix<float, Device::CPU> A(2, 3);
    DenseMatrix<float, Device::CPU> B(3, 2);

    // A = [1  3  5;  2  4  6] in column-major
    A.setValue(0, 0, 1.0f);
    A.setValue(1, 0, 2.0f);
    A.setValue(0, 1, 3.0f);
    A.setValue(1, 1, 4.0f);
    A.setValue(0, 2, 5.0f);
    A.setValue(1, 2, 6.0f);

    // B = [7  10;  8  11;  9  12] in column-major
    B.setValue(0, 0, 7.0f);
    B.setValue(1, 0, 8.0f);
    B.setValue(2, 0, 9.0f);
    B.setValue(0, 1, 10.0f);
    B.setValue(1, 1, 11.0f);
    B.setValue(2, 1, 12.0f);

    auto C = gemm(A, B);

    // C(0,0) = 1*7 + 3*8 + 5*9  = 7 + 24 + 45 = 76
    // C(1,0) = 2*7 + 4*8 + 6*9  = 14 + 32 + 54 = 100
    // C(0,1) = 1*10 + 3*11 + 5*12 = 10 + 33 + 60 = 103
    // C(1,1) = 2*10 + 4*11 + 6*12 = 20 + 44 + 72 = 136
    EXPECT_FLOAT_EQ(C(0, 0), 76.0f);
    EXPECT_FLOAT_EQ(C(1, 0), 100.0f);
    EXPECT_FLOAT_EQ(C(0, 1), 103.0f);
    EXPECT_FLOAT_EQ(C(1, 1), 136.0f);
}

// GEMM: multiply_2x3_by_3x2_Gpu — same calculation on GPU via cuBLAS
#ifdef PLAMATRIX_WITH_CUDA
TEST(GEMM, multiply_2x3_by_3x2_Gpu)
{
    DenseMatrix<float, Device::CPU> A_cpu(2, 3);
    DenseMatrix<float, Device::CPU> B_cpu(3, 2);

    // A = [1  3  5;  2  4  6]
    A_cpu.setValue(0, 0, 1.0f);
    A_cpu.setValue(1, 0, 2.0f);
    A_cpu.setValue(0, 1, 3.0f);
    A_cpu.setValue(1, 1, 4.0f);
    A_cpu.setValue(0, 2, 5.0f);
    A_cpu.setValue(1, 2, 6.0f);

    // B = [7  10;  8  11;  9  12]
    B_cpu.setValue(0, 0, 7.0f);
    B_cpu.setValue(1, 0, 8.0f);
    B_cpu.setValue(2, 0, 9.0f);
    B_cpu.setValue(0, 1, 10.0f);
    B_cpu.setValue(1, 1, 11.0f);
    B_cpu.setValue(2, 1, 12.0f);

    auto A_gpu = A_cpu.toGpu();
    auto B_gpu = B_cpu.toGpu();
    auto C_gpu = gemm(A_gpu, B_gpu);
    auto C_cpu = C_gpu.toCpu();

    EXPECT_FLOAT_EQ(C_cpu(0, 0), 76.0f);
    EXPECT_FLOAT_EQ(C_cpu(1, 0), 100.0f);
    EXPECT_FLOAT_EQ(C_cpu(0, 1), 103.0f);
    EXPECT_FLOAT_EQ(C_cpu(1, 1), 136.0f);
}

TEST(GEMM, multiply_2x3_by_3x2_GpuOutputReuse)
{
    DenseMatrix<float, Device::CPU> A_cpu(2, 3);
    DenseMatrix<float, Device::CPU> B_cpu(3, 2);

    A_cpu.setValue(0, 0, 1.0f);
    A_cpu.setValue(1, 0, 2.0f);
    A_cpu.setValue(0, 1, 3.0f);
    A_cpu.setValue(1, 1, 4.0f);
    A_cpu.setValue(0, 2, 5.0f);
    A_cpu.setValue(1, 2, 6.0f);

    B_cpu.setValue(0, 0, 7.0f);
    B_cpu.setValue(1, 0, 8.0f);
    B_cpu.setValue(2, 0, 9.0f);
    B_cpu.setValue(0, 1, 10.0f);
    B_cpu.setValue(1, 1, 11.0f);
    B_cpu.setValue(2, 1, 12.0f);

    auto A_gpu = A_cpu.toGpu();
    auto B_gpu = B_cpu.toGpu();
    DenseMatrix<float, Device::GPU> C_gpu(2, 2);

    gemm(A_gpu, B_gpu, C_gpu);

    auto C_cpu = C_gpu.toCpu();
    EXPECT_FLOAT_EQ(C_cpu(0, 0), 76.0f);
    EXPECT_FLOAT_EQ(C_cpu(1, 0), 100.0f);
    EXPECT_FLOAT_EQ(C_cpu(0, 1), 103.0f);
    EXPECT_FLOAT_EQ(C_cpu(1, 1), 136.0f);
}

TEST(GEMM, multiply_2x3_by_3x2_GpuAsync)
{
    DenseMatrix<float, Device::CPU> A_cpu(2, 3);
    DenseMatrix<float, Device::CPU> B_cpu(3, 2);

    A_cpu.setValue(0, 0, 1.0f);
    A_cpu.setValue(1, 0, 2.0f);
    A_cpu.setValue(0, 1, 3.0f);
    A_cpu.setValue(1, 1, 4.0f);
    A_cpu.setValue(0, 2, 5.0f);
    A_cpu.setValue(1, 2, 6.0f);

    B_cpu.setValue(0, 0, 7.0f);
    B_cpu.setValue(1, 0, 8.0f);
    B_cpu.setValue(2, 0, 9.0f);
    B_cpu.setValue(0, 1, 10.0f);
    B_cpu.setValue(1, 1, 11.0f);
    B_cpu.setValue(2, 1, 12.0f);

    auto A_gpu = A_cpu.toGpu();
    auto B_gpu = B_cpu.toGpu();
    auto C_cpu = gemmAsync(A_gpu, B_gpu).toCpu();

    EXPECT_FLOAT_EQ(C_cpu(0, 0), 76.0f);
    EXPECT_FLOAT_EQ(C_cpu(1, 0), 100.0f);
    EXPECT_FLOAT_EQ(C_cpu(0, 1), 103.0f);
    EXPECT_FLOAT_EQ(C_cpu(1, 1), 136.0f);
}

TEST(GEMM, rejectsGpuOutputDimensionMismatch)
{
    DenseMatrix<float, Device::CPU> A_cpu(2, 3);
    DenseMatrix<float, Device::CPU> B_cpu(3, 2);

    auto A_gpu = A_cpu.toGpu();
    auto B_gpu = B_cpu.toGpu();
    DenseMatrix<float, Device::GPU> C_gpu(3, 2);

    EXPECT_THROW(gemm(A_gpu, B_gpu, C_gpu), std::runtime_error);
    EXPECT_THROW(gemmAsync(A_gpu, B_gpu, C_gpu), std::runtime_error);
}

// GEMM: multiply_Larger_CpuVsGpuConsistency — random 256x256 matrices, compare CPU vs GPU
TEST(GEMM, multiply_Larger_CpuVsGpuConsistency)
{
    constexpr Index N = 256;
    DenseMatrix<float, Device::CPU> A_cpu(N, N);
    DenseMatrix<float, Device::CPU> B_cpu(N, N);

    // Fill with pseudo-random values (simple LCG for reproducibility)
    std::srand(42);
    for (Index j = 0; j < N; ++j)
    {
        for (Index i = 0; i < N; ++i)
        {
            A_cpu.setValue(i, j, static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX));
            B_cpu.setValue(i, j, static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX));
        }
    }

    auto A_gpu = A_cpu.toGpu();
    auto B_gpu = B_cpu.toGpu();

    auto C_cpu = gemm(A_cpu, B_cpu);
    auto C_gpu = gemm(A_gpu, B_gpu);
    auto C_gpu_to_cpu = C_gpu.toCpu();

    const float tolerance = 1e-3f;
    for (Index j = 0; j < N; ++j)
    {
        for (Index i = 0; i < N; ++i)
        {
            EXPECT_NEAR(C_cpu(i, j), C_gpu_to_cpu(i, j), tolerance)
                << "Mismatch at (" << i << ", " << j << ")";
        }
    }
}
#endif
