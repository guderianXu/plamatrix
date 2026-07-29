#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "plamatrix/ops/decomposition.h"
#include "plamatrix/ops/small_matrix.h"

namespace plamatrix
{
namespace
{

template <typename Scalar>
class SmallMatrixTest : public ::testing::Test
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

    static DenseMatrix<Scalar, Device::CPU> makeCompact(
        const std::vector<std::array<Scalar, 6>>& rows)
    {
        DenseMatrix<Scalar, Device::CPU> compact(static_cast<Index>(rows.size()), 6);
        for (Index row = 0; row < compact.rows(); ++row)
        {
            for (Index col = 0; col < 6; ++col)
            {
                compact(row, col) = rows[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)];
            }
        }
        return compact;
    }

    static std::array<Scalar, 9> expand(const DenseMatrix<Scalar, Device::CPU>& compact, Index row)
    {
        return {
            compact(row, 0), compact(row, 1), compact(row, 2),
            compact(row, 1), compact(row, 3), compact(row, 4),
            compact(row, 2), compact(row, 4), compact(row, 5)
        };
    }

    static void expectValidDecomposition(
        const DenseMatrix<Scalar, Device::CPU>& compact,
        const SymmetricEigh3x3Result<Scalar, Device::CPU>& result)
    {
        ASSERT_EQ(result.eigenvalues.rows(), compact.rows());
        ASSERT_EQ(result.eigenvalues.cols(), 3);
        ASSERT_EQ(result.eigenvectors.rows(), compact.rows());
        ASSERT_EQ(result.eigenvectors.cols(), 9);

        const Scalar tol = tolerance();
        for (Index row = 0; row < compact.rows(); ++row)
        {
            const auto matrix = expand(compact, row);
            EXPECT_LE(result.eigenvalues(row, 0), result.eigenvalues(row, 1));
            EXPECT_LE(result.eigenvalues(row, 1), result.eigenvalues(row, 2));

            for (Index col = 0; col < 3; ++col)
            {
                const Scalar lambda = result.eigenvalues(row, col);
                const std::array<Scalar, 3> vector = {
                    result.eigenvectors(row, col * 3),
                    result.eigenvectors(row, col * 3 + 1),
                    result.eigenvectors(row, col * 3 + 2)
                };

                Index sign_index = 0;
                for (Index component = 1; component < 3; ++component)
                {
                    if (std::abs(vector[static_cast<std::size_t>(component)]) >
                        std::abs(vector[static_cast<std::size_t>(sign_index)]))
                    {
                        sign_index = component;
                    }
                }
                EXPECT_GE(vector[static_cast<std::size_t>(sign_index)], Scalar(0));

                Scalar matrix_scale = Scalar(1);
                for (Scalar value : matrix)
                {
                    matrix_scale = std::max(matrix_scale, std::abs(value));
                }
                for (Index component = 0; component < 3; ++component)
                {
                    Scalar actual = Scalar(0);
                    for (Index inner = 0; inner < 3; ++inner)
                    {
                        actual += matrix[static_cast<std::size_t>(component * 3 + inner)] *
                                  vector[static_cast<std::size_t>(inner)];
                    }
                    EXPECT_NEAR(actual,
                                lambda * vector[static_cast<std::size_t>(component)],
                                tol * matrix_scale);
                }
            }

            for (Index left = 0; left < 3; ++left)
            {
                for (Index right = 0; right < 3; ++right)
                {
                    Scalar dot = Scalar(0);
                    for (Index component = 0; component < 3; ++component)
                    {
                        dot += result.eigenvectors(row, left * 3 + component) *
                               result.eigenvectors(row, right * 3 + component);
                    }
                    EXPECT_NEAR(dot, left == right ? Scalar(1) : Scalar(0), tol);
                }
            }
        }
    }
};

#if defined(PLAMATRIX_USE_FLOAT) && defined(PLAMATRIX_USE_DOUBLE)
using ScalarTypes = ::testing::Types<float, double>;
#elif defined(PLAMATRIX_USE_FLOAT)
using ScalarTypes = ::testing::Types<float>;
#elif defined(PLAMATRIX_USE_DOUBLE)
using ScalarTypes = ::testing::Types<double>;
#else
#error "Small matrix tests require PLAMATRIX_USE_FLOAT or PLAMATRIX_USE_DOUBLE"
#endif
TYPED_TEST_SUITE(SmallMatrixTest, ScalarTypes);

TYPED_TEST(SmallMatrixTest, DiagonalMatricesAreSortedWithCanonicalVectors)
{
    using Scalar = TypeParam;
    auto compact = TestFixture::makeCompact({
        {Scalar(5), Scalar(0), Scalar(0), Scalar(-2), Scalar(0), Scalar(1)},
        {Scalar(3), Scalar(0), Scalar(0), Scalar(3), Scalar(0), Scalar(3)}
    });

    auto result = symmetricEigh3x3Batched(compact);

    TestFixture::expectValidDecomposition(compact, result);
    EXPECT_EQ(result.eigenvalues(0, 0), Scalar(-2));
    EXPECT_EQ(result.eigenvalues(0, 1), Scalar(1));
    EXPECT_EQ(result.eigenvalues(0, 2), Scalar(5));
    for (Index col = 0; col < 9; ++col)
    {
        EXPECT_EQ(result.eigenvectors(1, col), col % 4 == 0 ? Scalar(1) : Scalar(0));
    }
}

TYPED_TEST(SmallMatrixTest, RotatedMatrixReturnsMatchingEigenpairs)
{
    using Scalar = TypeParam;
    const Scalar c = std::sqrt(Scalar(0.5));
    auto compact = TestFixture::makeCompact({
        {Scalar(2.5), Scalar(-1.5), Scalar(2) * c,
         Scalar(2.5), Scalar(-2) * c, Scalar(4)}
    });

    auto result = symmetricEigh3x3Batched(compact);

    TestFixture::expectValidDecomposition(compact, result);
    EXPECT_NEAR(result.eigenvalues(0, 0), Scalar(1), TestFixture::tolerance());
    EXPECT_NEAR(result.eigenvalues(0, 1), Scalar(2), TestFixture::tolerance());
    EXPECT_NEAR(result.eigenvalues(0, 2), Scalar(6), TestFixture::tolerance());
}

TYPED_TEST(SmallMatrixTest, RankOneAndZeroMatricesUseDeterministicOrthonormalBases)
{
    using Scalar = TypeParam;
    auto compact = TestFixture::makeCompact({
        {Scalar(1), Scalar(2), Scalar(-1), Scalar(4), Scalar(-2), Scalar(1)},
        {Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0)}
    });

    auto first = symmetricEigh3x3Batched(compact);
    auto second = symmetricEigh3x3Batched(compact);

    TestFixture::expectValidDecomposition(compact, first);
    EXPECT_NEAR(first.eigenvalues(0, 0), Scalar(0), TestFixture::tolerance());
    EXPECT_NEAR(first.eigenvalues(0, 1), Scalar(0), TestFixture::tolerance());
    EXPECT_NEAR(first.eigenvalues(0, 2), Scalar(6), TestFixture::tolerance());
    for (Index row = 0; row < compact.rows(); ++row)
    {
        for (Index col = 0; col < 9; ++col)
        {
            EXPECT_EQ(first.eigenvectors(row, col), second.eigenvectors(row, col));
        }
    }
}

TYPED_TEST(SmallMatrixTest, RepeatedEigenvalueUsesFixedAxisProjectionBasis)
{
    using Scalar = TypeParam;
    auto compact = TestFixture::makeCompact({
        {Scalar(3), Scalar(1), Scalar(1), Scalar(3), Scalar(1), Scalar(3)}
    });

    auto result = symmetricEigh3x3Batched(compact);

    TestFixture::expectValidDecomposition(compact, result);
    EXPECT_NEAR(result.eigenvalues(0, 0), Scalar(2), TestFixture::tolerance());
    EXPECT_NEAR(result.eigenvalues(0, 1), Scalar(2), TestFixture::tolerance());
    EXPECT_NEAR(result.eigenvalues(0, 2), Scalar(5), TestFixture::tolerance());
    EXPECT_GT(result.eigenvectors(0, 0), Scalar(0));
    EXPECT_NEAR(result.eigenvectors(0, 1),
                -result.eigenvectors(0, 0) / Scalar(2),
                TestFixture::tolerance());
    EXPECT_NEAR(result.eigenvectors(0, 2),
                -result.eigenvectors(0, 0) / Scalar(2),
                TestFixture::tolerance());
}

TYPED_TEST(SmallMatrixTest, EmptyBatchReturnsEmptyOutputs)
{
    using Scalar = TypeParam;
    DenseMatrix<Scalar, Device::CPU> compact(0, 6);

    auto result = symmetricEigh3x3Batched(compact);

    EXPECT_EQ(result.eigenvalues.rows(), 0);
    EXPECT_EQ(result.eigenvalues.cols(), 3);
    EXPECT_EQ(result.eigenvectors.rows(), 0);
    EXPECT_EQ(result.eigenvectors.cols(), 9);
}

TYPED_TEST(SmallMatrixTest, InvalidShapeThrowsInvalidArgument)
{
    using Scalar = TypeParam;
    DenseMatrix<Scalar, Device::CPU> too_narrow(2, 5);
    DenseMatrix<Scalar, Device::CPU> too_wide(2, 7);
    DenseMatrix<Scalar, Device::CPU> empty_wrong_shape(0, 5);

    EXPECT_THROW(symmetricEigh3x3Batched(too_narrow), std::invalid_argument);
    EXPECT_THROW(symmetricEigh3x3Batched(too_wide), std::invalid_argument);
    EXPECT_THROW(symmetricEigh3x3Batched(empty_wrong_shape), std::invalid_argument);
}

TYPED_TEST(SmallMatrixTest, NonFiniteInputIsRejectedBeforeDecomposition)
{
    using Scalar = TypeParam;
    const std::array<Scalar, 6> finite = {
        Scalar(1), Scalar(0), Scalar(0), Scalar(2), Scalar(0), Scalar(3)
    };
    auto with_nan = TestFixture::makeCompact({finite, finite});
    auto with_pos_inf = TestFixture::makeCompact({finite, finite});
    auto with_neg_inf = TestFixture::makeCompact({finite, finite});
    with_nan(1, 4) = std::numeric_limits<Scalar>::quiet_NaN();
    with_pos_inf(1, 0) = std::numeric_limits<Scalar>::infinity();
    with_neg_inf(1, 5) = -std::numeric_limits<Scalar>::infinity();

    for (const auto* input : {&with_nan, &with_pos_inf, &with_neg_inf})
    {
        try
        {
            static_cast<void>(symmetricEigh3x3Batched(*input));
            FAIL() << "non-finite input should be rejected";
        }
        catch (const std::invalid_argument& error)
        {
            const std::string message = error.what();
            EXPECT_NE(message.find("finite"), std::string::npos) << message;
            EXPECT_NE(message.find("row 1"), std::string::npos) << message;
        }
    }
}

TYPED_TEST(SmallMatrixTest, FixedRandomMatricesSatisfyEigenpairInvariants)
{
    using Scalar = TypeParam;
    constexpr Index sample_count = 12;
    std::mt19937 generator(0x5A17u);
    std::uniform_real_distribution<double> distribution(-4.0, 4.0);
    std::vector<std::array<Scalar, 6>> samples(static_cast<std::size_t>(sample_count));
    for (auto& sample : samples)
    {
        for (Scalar& value : sample)
        {
            value = static_cast<Scalar>(distribution(generator));
        }
    }
    auto compact = TestFixture::makeCompact(samples);

    auto result = symmetricEigh3x3Batched(compact);

    TestFixture::expectValidDecomposition(compact, result);
}

TYPED_TEST(SmallMatrixTest, FixedRandomBlockEigenvaluesMatchGeneralEigh)
{
    using Scalar = TypeParam;
    constexpr Index sample_count = 12;
    std::mt19937 generator(0xE193u);
    std::uniform_real_distribution<double> distribution(-4.0, 4.0);
    std::vector<std::array<Scalar, 6>> samples(static_cast<std::size_t>(sample_count));
    for (auto& sample : samples)
    {
        const Scalar diagonal = static_cast<Scalar>(distribution(generator));
        sample = {
            diagonal,
            static_cast<Scalar>(distribution(generator)),
            Scalar(0),
            diagonal,
            Scalar(0),
            static_cast<Scalar>(distribution(generator))
        };
    }
    auto compact = TestFixture::makeCompact(samples);

    auto result = symmetricEigh3x3Batched(compact);

    TestFixture::expectValidDecomposition(compact, result);
    for (Index row = 0; row < sample_count; ++row)
    {
        const auto values = TestFixture::expand(compact, row);
        DenseMatrix<Scalar, Device::CPU> full(3, 3);
        for (Index matrix_row = 0; matrix_row < 3; ++matrix_row)
        {
            for (Index matrix_col = 0; matrix_col < 3; ++matrix_col)
            {
                full(matrix_row, matrix_col) =
                    values[static_cast<std::size_t>(matrix_row * 3 + matrix_col)];
            }
        }
        auto reference_descending = eigh(full);
        for (Index eigen_index = 0; eigen_index < 3; ++eigen_index)
        {
            const Scalar expected = reference_descending(2 - eigen_index, 0);
            const Scalar scale = std::max(Scalar(1), std::abs(expected));
            EXPECT_NEAR(result.eigenvalues(row, eigen_index),
                        expected,
                        TestFixture::tolerance() * scale);
        }
    }
}

} // namespace
} // namespace plamatrix
