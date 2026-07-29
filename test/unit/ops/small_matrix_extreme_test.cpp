#include <array>
#include <cmath>
#include <limits>
#include <type_traits>

#include <gtest/gtest.h>

#include "plamatrix/ops/small_matrix.h"

namespace plamatrix
{
namespace
{

template <typename Scalar>
class SmallMatrixExtremeTest : public ::testing::Test
{
protected:
    static constexpr Scalar tolerance()
    {
        if constexpr (std::is_same_v<Scalar, float>)
        {
            return Scalar(2e-4);
        }
        return Scalar(2e-11);
    }

    static DenseMatrix<Scalar, Device::CPU> makeCompact(const std::array<Scalar, 6>& values)
    {
        DenseMatrix<Scalar, Device::CPU> compact(1, 6);
        for (Index col = 0; col < 6; ++col)
        {
            compact(0, col) = values[static_cast<std::size_t>(col)];
        }
        return compact;
    }

    static void expectValid(
        const DenseMatrix<Scalar, Device::CPU>& compact,
        const SymmetricEigh3x3Result<Scalar, Device::CPU>& result)
    {
        ASSERT_EQ(result.eigenvalues.rows(), 1);
        ASSERT_EQ(result.eigenvalues.cols(), 3);
        ASSERT_EQ(result.eigenvectors.rows(), 1);
        ASSERT_EQ(result.eigenvectors.cols(), 9);

        const std::array<Scalar, 9> matrix = {
            compact(0, 0), compact(0, 1), compact(0, 2),
            compact(0, 1), compact(0, 3), compact(0, 4),
            compact(0, 2), compact(0, 4), compact(0, 5)
        };
        Scalar scale = Scalar(1);
        for (Scalar value : matrix)
        {
            scale = std::max(scale, std::abs(value));
        }

        EXPECT_LE(result.eigenvalues(0, 0), result.eigenvalues(0, 1));
        EXPECT_LE(result.eigenvalues(0, 1), result.eigenvalues(0, 2));
        for (Index col = 0; col < 3; ++col)
        {
            const Scalar lambda = result.eigenvalues(0, col);
            EXPECT_TRUE(std::isfinite(lambda));
            std::array<Scalar, 3> vector = {
                result.eigenvectors(0, col * 3),
                result.eigenvectors(0, col * 3 + 1),
                result.eigenvectors(0, col * 3 + 2)
            };
            for (Scalar value : vector)
            {
                EXPECT_TRUE(std::isfinite(value));
            }
            for (Index row = 0; row < 3; ++row)
            {
                Scalar actual = Scalar(0);
                for (Index inner = 0; inner < 3; ++inner)
                {
                    actual += matrix[static_cast<std::size_t>(row * 3 + inner)] *
                              vector[static_cast<std::size_t>(inner)];
                }
                EXPECT_NEAR(actual, lambda * vector[static_cast<std::size_t>(row)],
                            tolerance() * scale);
            }
        }
        for (Index left = 0; left < 3; ++left)
        {
            for (Index right = 0; right < 3; ++right)
            {
                Scalar dot = Scalar(0);
                for (Index component = 0; component < 3; ++component)
                {
                    dot += result.eigenvectors(0, left * 3 + component) *
                           result.eigenvectors(0, right * 3 + component);
                }
                EXPECT_NEAR(dot, left == right ? Scalar(1) : Scalar(0), tolerance());
            }
        }
    }
};

#if defined(PLAMATRIX_USE_FLOAT) && defined(PLAMATRIX_USE_DOUBLE)
using ExtremeScalarTypes = ::testing::Types<float, double>;
#elif defined(PLAMATRIX_USE_FLOAT)
using ExtremeScalarTypes = ::testing::Types<float>;
#elif defined(PLAMATRIX_USE_DOUBLE)
using ExtremeScalarTypes = ::testing::Types<double>;
#endif
TYPED_TEST_SUITE(SmallMatrixExtremeTest, ExtremeScalarTypes);

TYPED_TEST(SmallMatrixExtremeTest, AvoidsTwiceOffDiagonalOverflow)
{
    using Scalar = TypeParam;
    const Scalar maximum = std::numeric_limits<Scalar>::max();
    auto compact = TestFixture::makeCompact({
        Scalar(-0.5) * maximum, Scalar(0.75) * maximum, Scalar(0),
        Scalar(0.5) * maximum, Scalar(0), Scalar(0)
    });
    auto result = symmetricEigh3x3Batched(compact);
    TestFixture::expectValid(compact, result);
    const Scalar expected = std::sqrt(Scalar(0.8125)) * maximum;
    const Scalar allowed = TestFixture::tolerance() * maximum;
    EXPECT_NEAR(result.eigenvalues(0, 0), -expected, allowed);
    EXPECT_NEAR(result.eigenvalues(0, 1), Scalar(0), allowed);
    EXPECT_NEAR(result.eigenvalues(0, 2), expected, allowed);
}

TYPED_TEST(SmallMatrixExtremeTest, AvoidsDiagonalDifferenceOverflow)
{
    using Scalar = TypeParam;
    const Scalar maximum = std::numeric_limits<Scalar>::max();
    auto compact = TestFixture::makeCompact({
        Scalar(-0.75) * maximum, Scalar(0.25) * maximum, Scalar(0),
        Scalar(0.75) * maximum, Scalar(0), Scalar(0)
    });
    auto result = symmetricEigh3x3Batched(compact);
    TestFixture::expectValid(compact, result);
    const Scalar expected = std::sqrt(Scalar(0.625)) * maximum;
    const Scalar allowed = TestFixture::tolerance() * maximum;
    EXPECT_NEAR(result.eigenvalues(0, 0), -expected, allowed);
    EXPECT_NEAR(result.eigenvalues(0, 1), Scalar(0), allowed);
    EXPECT_NEAR(result.eigenvalues(0, 2), expected, allowed);
}

TYPED_TEST(SmallMatrixExtremeTest, EqualDiagonalBlockReturnsMaximumEigenvalue)
{
    using Scalar = TypeParam;
    const Scalar maximum = std::numeric_limits<Scalar>::max();
    const Scalar half = Scalar(0.5) * maximum;
    auto compact = TestFixture::makeCompact({half, half, Scalar(0), half, Scalar(0), Scalar(0)});
    auto result = symmetricEigh3x3Batched(compact);
    TestFixture::expectValid(compact, result);
    const Scalar allowed = TestFixture::tolerance() * maximum;
    EXPECT_NEAR(result.eigenvalues(0, 0), Scalar(0), allowed);
    EXPECT_NEAR(result.eigenvalues(0, 1), Scalar(0), allowed);
    EXPECT_NEAR(result.eigenvalues(0, 2), maximum, allowed);
}

TYPED_TEST(SmallMatrixExtremeTest, UnderflowedScaledOffDiagonalKeepsOutputsFinite)
{
    using Scalar = TypeParam;
    const Scalar maximum = std::numeric_limits<Scalar>::max();
    const Scalar denormal = std::numeric_limits<Scalar>::denorm_min();
    ASSERT_NE(denormal, Scalar(0));
    auto compact = TestFixture::makeCompact({
        maximum, denormal, Scalar(0), maximum, Scalar(0), Scalar(0)
    });
    auto result = symmetricEigh3x3Batched(compact);
    TestFixture::expectValid(compact, result);
}

} // namespace
} // namespace plamatrix
