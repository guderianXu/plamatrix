#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <random>

#include <gtest/gtest.h>

#include "plamatrix/ops/small_matrix.h"

namespace plamatrix
{
namespace
{

#ifdef PLAMATRIX_WITH_CUDA

template <typename Scalar>
class SmallMatrixCudaTest : public ::testing::Test
{
protected:
    static constexpr Scalar tolerance()
    {
        if constexpr (std::is_same_v<Scalar, float>)
        {
            return Scalar(6e-4);
        }
        return Scalar(6e-11);
    }

    static DenseMatrix<Scalar, Device::CPU> makeBatch(Index rows)
    {
        DenseMatrix<Scalar, Device::CPU> compact(rows, 6);
        std::mt19937 generator(0xC0DAu + static_cast<unsigned int>(rows));
        std::uniform_real_distribution<double> distribution(-4.0, 4.0);
        for (Index row = 0; row < rows; ++row)
        {
            for (Index col = 0; col < 6; ++col)
            {
                compact(row, col) = static_cast<Scalar>(distribution(generator));
            }
        }
        if (rows > 0)
        {
            for (Index col = 0; col < 6; ++col)
            {
                compact(0, col) = col == 0 || col == 3 || col == 5 ? Scalar(3) : Scalar(0);
            }
        }
        if (rows > 1)
        {
            const std::array<Scalar, 6> repeated = {
                Scalar(3), Scalar(1), Scalar(1), Scalar(3), Scalar(1), Scalar(3)
            };
            for (Index col = 0; col < 6; ++col)
            {
                compact(1, col) = repeated[static_cast<std::size_t>(col)];
            }
        }
        return compact;
    }

    static void expectDecomposition(
        const DenseMatrix<Scalar, Device::CPU>& compact,
        const DenseMatrix<Scalar, Device::CPU>& eigenvalues,
        const DenseMatrix<Scalar, Device::CPU>& eigenvectors)
    {
        ASSERT_EQ(eigenvalues.rows(), compact.rows());
        ASSERT_EQ(eigenvalues.cols(), 3);
        ASSERT_EQ(eigenvectors.rows(), compact.rows());
        ASSERT_EQ(eigenvectors.cols(), 9);
        const Scalar tol = tolerance();
        for (Index row = 0; row < compact.rows(); ++row)
        {
            const std::array<Scalar, 9> matrix = {
                compact(row, 0), compact(row, 1), compact(row, 2),
                compact(row, 1), compact(row, 3), compact(row, 4),
                compact(row, 2), compact(row, 4), compact(row, 5)
            };
            Scalar scale = Scalar(1);
            for (Scalar value : matrix)
            {
                scale = std::max(scale, std::abs(value));
            }
            EXPECT_LE(eigenvalues(row, 0), eigenvalues(row, 1));
            EXPECT_LE(eigenvalues(row, 1), eigenvalues(row, 2));
            for (Index col = 0; col < 3; ++col)
            {
                std::array<Scalar, 3> vector = {
                    eigenvectors(row, 3 * col),
                    eigenvectors(row, 3 * col + 1),
                    eigenvectors(row, 3 * col + 2)
                };
                Index largest = 0;
                for (Index component = 1; component < 3; ++component)
                {
                    if (std::abs(vector[static_cast<std::size_t>(component)]) >
                        std::abs(vector[static_cast<std::size_t>(largest)]))
                    {
                        largest = component;
                    }
                }
                EXPECT_GE(vector[static_cast<std::size_t>(largest)], Scalar(0));
                for (Index component = 0; component < 3; ++component)
                {
                    Scalar actual = Scalar(0);
                    for (Index inner = 0; inner < 3; ++inner)
                    {
                        actual += matrix[static_cast<std::size_t>(3 * component + inner)] *
                                  vector[static_cast<std::size_t>(inner)];
                    }
                    EXPECT_NEAR(actual,
                                eigenvalues(row, col) * vector[static_cast<std::size_t>(component)],
                                tol * scale);
                }
            }
            for (Index left = 0; left < 3; ++left)
            {
                for (Index right = 0; right < 3; ++right)
                {
                    Scalar dot = Scalar(0);
                    for (Index component = 0; component < 3; ++component)
                    {
                        dot += eigenvectors(row, 3 * left + component) *
                               eigenvectors(row, 3 * right + component);
                    }
                    EXPECT_NEAR(dot, left == right ? Scalar(1) : Scalar(0), tol);
                }
            }
        }
    }

    static void expectCpuGpuAgreement(
        const SymmetricEigh3x3Result<Scalar, Device::CPU>& cpu,
        const DenseMatrix<Scalar, Device::CPU>& gpu_values,
        const DenseMatrix<Scalar, Device::CPU>& gpu_vectors)
    {
        const Scalar tol = tolerance();
        for (Index row = 0; row < cpu.eigenvalues.rows(); ++row)
        {
            for (Index col = 0; col < 3; ++col)
            {
                const Scalar scale = std::max(Scalar(1), std::abs(cpu.eigenvalues(row, col)));
                EXPECT_NEAR(gpu_values(row, col), cpu.eigenvalues(row, col), tol * scale);
            }

            Index begin = 0;
            while (begin < 3)
            {
                Index end = begin + 1;
                while (end < 3)
                {
                    const Scalar scale = std::max({Scalar(1),
                                                   std::abs(cpu.eigenvalues(row, begin)),
                                                   std::abs(cpu.eigenvalues(row, end))});
                    if (std::abs(cpu.eigenvalues(row, begin) - cpu.eigenvalues(row, end)) >
                        Scalar(512) * std::numeric_limits<Scalar>::epsilon() * scale)
                    {
                        break;
                    }
                    ++end;
                }
                if (end - begin > 1)
                {
                    for (Index i = 0; i < 3; ++i)
                    {
                        for (Index j = 0; j < 3; ++j)
                        {
                            Scalar cpu_projector = Scalar(0);
                            Scalar gpu_projector = Scalar(0);
                            for (Index col = begin; col < end; ++col)
                            {
                                cpu_projector += cpu.eigenvectors(row, 3 * col + i) *
                                                 cpu.eigenvectors(row, 3 * col + j);
                                gpu_projector += gpu_vectors(row, 3 * col + i) *
                                                 gpu_vectors(row, 3 * col + j);
                            }
                            EXPECT_NEAR(gpu_projector, cpu_projector, Scalar(4) * tol);
                        }
                    }
                }
                else
                {
                    std::array<Scalar, 3> cpu_vector = {
                        cpu.eigenvectors(row, 3 * begin),
                        cpu.eigenvectors(row, 3 * begin + 1),
                        cpu.eigenvectors(row, 3 * begin + 2)
                    };
                    std::array<Scalar, 3> gpu_vector = {
                        gpu_vectors(row, 3 * begin),
                        gpu_vectors(row, 3 * begin + 1),
                        gpu_vectors(row, 3 * begin + 2)
                    };
                    auto canonicalize = [](std::array<Scalar, 3>& vector) {
                        Index largest = 0;
                        for (Index component = 1; component < 3; ++component)
                        {
                            if (std::abs(vector[static_cast<std::size_t>(component)]) >
                                std::abs(vector[static_cast<std::size_t>(largest)]))
                            {
                                largest = component;
                            }
                        }
                        if (vector[static_cast<std::size_t>(largest)] < Scalar(0))
                        {
                            for (Scalar& component : vector)
                            {
                                component = -component;
                            }
                        }
                    };
                    canonicalize(cpu_vector);
                    canonicalize(gpu_vector);
                    for (Index component = 0; component < 3; ++component)
                    {
                        EXPECT_NEAR(gpu_vector[static_cast<std::size_t>(component)],
                                    cpu_vector[static_cast<std::size_t>(component)],
                                    Scalar(4) * tol);
                    }
                }
                begin = end;
            }
        }
    }
};

#if defined(PLAMATRIX_USE_FLOAT) && defined(PLAMATRIX_USE_DOUBLE)
using CudaScalarTypes = ::testing::Types<float, double>;
#elif defined(PLAMATRIX_USE_FLOAT)
using CudaScalarTypes = ::testing::Types<float>;
#elif defined(PLAMATRIX_USE_DOUBLE)
using CudaScalarTypes = ::testing::Types<double>;
#endif
TYPED_TEST_SUITE(SmallMatrixCudaTest, CudaScalarTypes);

TYPED_TEST(SmallMatrixCudaTest, BatchBoundariesMatchCpuAndSatisfyEigenpairInvariants)
{
    using Scalar = TypeParam;
    for (Index rows : {Index(1), Index(31), Index(32), Index(33), Index(4096)})
    {
        auto compact = TestFixture::makeBatch(rows);
        auto compact_gpu = compact.toGpu();
        auto gpu = symmetricEigh3x3Batched(compact_gpu);
        auto gpu_values = gpu.eigenvalues.toCpu();
        auto gpu_vectors = gpu.eigenvectors.toCpu();
        auto cpu = symmetricEigh3x3Batched(compact);

        TestFixture::expectDecomposition(compact, gpu_values, gpu_vectors);
        TestFixture::expectCpuGpuAgreement(cpu, gpu_values, gpu_vectors);
    }
}

TYPED_TEST(SmallMatrixCudaTest, ScaledJacobiAngleHandlesFiniteExtremeMatrix)
{
    using Scalar = TypeParam;
    const Scalar largest = std::numeric_limits<Scalar>::max();
    DenseMatrix<Scalar, Device::CPU> compact(1, 6);
    compact(0, 0) = Scalar(-0.5) * largest;
    compact(0, 1) = Scalar(0.75) * largest;
    compact(0, 2) = Scalar(0);
    compact(0, 3) = Scalar(0.5) * largest;
    compact(0, 4) = Scalar(0);
    compact(0, 5) = Scalar(0.25) * largest;

    auto gpu = symmetricEigh3x3Batched(compact.toGpu());
    auto values = gpu.eigenvalues.toCpu();
    auto vectors = gpu.eigenvectors.toCpu();

    for (Index col = 0; col < 3; ++col)
    {
        EXPECT_TRUE(std::isfinite(values(0, col)));
    }
    TestFixture::expectDecomposition(compact, values, vectors);
}

TYPED_TEST(SmallMatrixCudaTest, TangentUpdateKeepsMaximumBoundaryEigenvalueFinite)
{
    using Scalar = TypeParam;
    const Scalar maximum = std::numeric_limits<Scalar>::max();
    const Scalar half_maximum = Scalar(0.5) * maximum;
    DenseMatrix<Scalar, Device::CPU> compact(1, 6);
    compact(0, 0) = half_maximum;
    compact(0, 1) = half_maximum;
    compact(0, 2) = Scalar(0);
    compact(0, 3) = half_maximum;
    compact(0, 4) = Scalar(0);
    compact(0, 5) = Scalar(0);

    auto gpu = symmetricEigh3x3Batched(compact.toGpu());
    auto values = gpu.eigenvalues.toCpu();
    auto vectors = gpu.eigenvectors.toCpu();

    for (Index col = 0; col < 3; ++col)
    {
        EXPECT_TRUE(std::isfinite(values(0, col)));
    }
    for (Index col = 0; col < 9; ++col)
    {
        EXPECT_TRUE(std::isfinite(vectors(0, col)));
    }
    TestFixture::expectDecomposition(compact, values, vectors);
    const Scalar tolerance = TestFixture::tolerance() * maximum;
    EXPECT_NEAR(values(0, 0), Scalar(0), tolerance);
    EXPECT_NEAR(values(0, 1), Scalar(0), tolerance);
    EXPECT_NEAR(values(0, 2), maximum, tolerance);
}

TYPED_TEST(SmallMatrixCudaTest, TangentUpdateHandlesScaledOffDiagonalUnderflow)
{
    using Scalar = TypeParam;
    const Scalar maximum = std::numeric_limits<Scalar>::max();
    DenseMatrix<Scalar, Device::CPU> compact(1, 6);
    compact(0, 0) = maximum;
    compact(0, 1) = std::numeric_limits<Scalar>::denorm_min();
    compact(0, 2) = Scalar(0);
    compact(0, 3) = maximum;
    compact(0, 4) = Scalar(0);
    compact(0, 5) = Scalar(0);

    auto gpu = symmetricEigh3x3Batched(compact.toGpu());
    auto values = gpu.eigenvalues.toCpu();
    auto vectors = gpu.eigenvectors.toCpu();

    for (Index col = 0; col < 3; ++col)
    {
        EXPECT_TRUE(std::isfinite(values(0, col)));
    }
    for (Index col = 0; col < 9; ++col)
    {
        EXPECT_TRUE(std::isfinite(vectors(0, col)));
    }
    TestFixture::expectDecomposition(compact, values, vectors);
}

#endif

} // namespace
} // namespace plamatrix
