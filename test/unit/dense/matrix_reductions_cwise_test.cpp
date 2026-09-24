#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <type_traits>

#include "plamatrix/plamatrix.h"

namespace
{
    struct Shift
    {
        double operator()(double value) const
        {
            return value + 0.5;
        }
    };

    struct WeightedSum
    {
        double operator()(double left, double right) const
        {
            return 2.0 * left + right;
        }
    };
} // namespace

TEST(MatrixReductions, ExtremaReturnEigenStyleIndices)
{
    plamatrix::MatrixXd values(2, 3);
    values << 4.0, -2.0, 8.0, 7.0, 5.0, 11.0;

    plamatrix::Index row = -1;
    plamatrix::Index col = -1;
    EXPECT_DOUBLE_EQ(values.minCoeff(&row, &col), -2.0);
    EXPECT_EQ(row, 0);
    EXPECT_EQ(col, 1);
    EXPECT_DOUBLE_EQ(values.maxCoeff(&row, &col), 11.0);
    EXPECT_EQ(row, 1);
    EXPECT_EQ(col, 2);

    plamatrix::Vector3d vector(2.0, -4.0, 3.0);
    plamatrix::Index index = -1;
    EXPECT_DOUBLE_EQ(vector.minCoeff(&index), -4.0);
    EXPECT_EQ(index, 1);
    EXPECT_DOUBLE_EQ(vector.maxCoeff(&index), 3.0);
    EXPECT_EQ(index, 2);

    EXPECT_DOUBLE_EQ(values.block(0, 1, 2, 2).minCoeff(&row, &col), -2.0);
    EXPECT_EQ(row, 0);
    EXPECT_EQ(col, 0);
}

TEST(MatrixReductions, ScalarAndBooleanReductionsMatchEigenSemantics)
{
    plamatrix::MatrixXi values(2, 3);
    values << 1, 2, 0, -3, 4, 5;

    EXPECT_EQ(values.sum(), 9);
    EXPECT_EQ(values.prod(), 0);
    EXPECT_FALSE(values.all());
    EXPECT_TRUE(values.any());
    EXPECT_EQ(values.count(), 5);
    EXPECT_EQ(values.transpose().count(), 5);
    EXPECT_EQ(values.block(0, 0, 2, 2).prod(), -24);

    const plamatrix::MatrixXi empty(0, 0);
    EXPECT_EQ(empty.sum(), 0);
    EXPECT_EQ(empty.prod(), 1);
    EXPECT_TRUE(empty.all());
    EXPECT_FALSE(empty.any());
    EXPECT_EQ(empty.count(), 0);
}

TEST(MatrixReductions, LpAndStableNormsHandleLargeAndSmallCoefficients)
{
    const plamatrix::Vector3d values(-2.0, 3.0, -6.0);
    EXPECT_DOUBLE_EQ(values.lpNorm<1>(), 11.0);
    EXPECT_DOUBLE_EQ(values.lpNorm<plamatrix::Infinity>(), 6.0);
    EXPECT_NEAR(values.lpNorm<2>(), 7.0, 1.0e-15);
    EXPECT_NEAR(values.lpNorm<3>(), std::cbrt(251.0), 1.0e-14);

    const plamatrix::Vector2d large(1.0e308, 1.0e308);
    EXPECT_TRUE(std::isfinite(large.stableNorm()));
    EXPECT_TRUE(std::isfinite(large.blueNorm()));
    EXPECT_TRUE(std::isfinite(large.hypotNorm()));
    EXPECT_NEAR(large.stableNorm() / 1.0e308, std::sqrt(2.0), 1.0e-15);
    EXPECT_NEAR(large.blueNorm() / 1.0e308, std::sqrt(2.0), 1.0e-15);
    EXPECT_NEAR(large.hypotNorm() / 1.0e308, std::sqrt(2.0), 1.0e-15);

    const plamatrix::Vector2d small(1.0e-308, -1.0e-308);
    EXPECT_GT(small.stableNorm(), 0.0);
    EXPECT_GT(small.blueNorm(), 0.0);
    EXPECT_GT(small.hypotNorm(), 0.0);
}

TEST(MatrixCwise, BuiltinAndCustomOperationsRemainLazyAndComposable)
{
    plamatrix::Matrix2d left;
    left << -1.0, 4.0, -3.0, 2.0;
    plamatrix::Matrix2d right;
    right << 2.0, 3.0, -5.0, 8.0;

    const auto absolute = left.cwiseAbs();
    left(0, 0) = -7.0;
    EXPECT_DOUBLE_EQ(absolute(0, 0), 7.0);

    const plamatrix::Matrix2d minimum = left.cwiseMin(right);
    const plamatrix::Matrix2d maximum = left.cwiseMax(right);
    const plamatrix::Matrix2d clamped = left.cwiseMin(1.0).cwiseMax(-2.0);
    EXPECT_DOUBLE_EQ(minimum(0, 0), -7.0);
    EXPECT_DOUBLE_EQ(minimum(1, 1), 2.0);
    EXPECT_DOUBLE_EQ(maximum(0, 1), 4.0);
    EXPECT_DOUBLE_EQ(maximum(1, 1), 8.0);
    EXPECT_DOUBLE_EQ(clamped(0, 0), -2.0);
    EXPECT_DOUBLE_EQ(clamped(1, 0), -2.0);

    const plamatrix::Matrix2d shifted = left.unaryExpr(Shift{});
    const plamatrix::Matrix2d combined = left.binaryExpr(right, WeightedSum{});
    EXPECT_DOUBLE_EQ(shifted(1, 0), -2.5);
    EXPECT_DOUBLE_EQ(combined(0, 1), 11.0);

    const auto composed = (left + right).cwiseAbs().unaryExpr(Shift{});
    EXPECT_DOUBLE_EQ(composed.sum(), 32.0);
    EXPECT_DOUBLE_EQ(composed.prod(), 3681.5625);
}

TEST(MatrixCwise, NaNPropagationOptionsMatchEigenCalls)
{
    plamatrix::Vector2d left(std::numeric_limits<double>::quiet_NaN(), 4.0);
    const plamatrix::Vector2d right(3.0, std::numeric_limits<double>::quiet_NaN());

    const plamatrix::Vector2d propagated = left.cwiseMin<plamatrix::PropagateNaN>(right);
    EXPECT_TRUE(std::isnan(propagated(0)));
    EXPECT_TRUE(std::isnan(propagated(1)));

    const plamatrix::Vector2d numbers = left.cwiseMax<plamatrix::PropagateNumbers>(right);
    EXPECT_DOUBLE_EQ(numbers(0), 3.0);
    EXPECT_DOUBLE_EQ(numbers(1), 4.0);
}

TEST(MatrixCwise, ReductionsAndExpressionsCoverViewsMapsAndRefs)
{
    double storage[] = {1.0, -2.0, 3.0, 4.0, -5.0, 6.0};
    using FixedMatrix = plamatrix::Matrix<double, 2, 3>;
    plamatrix::Map<FixedMatrix> mapped(storage);
    EXPECT_DOUBLE_EQ(mapped.prod(), 720.0);
    EXPECT_DOUBLE_EQ(mapped.transpose().lpNorm<1>(), 21.0);

    plamatrix::Index row = -1;
    plamatrix::Index col = -1;
    EXPECT_DOUBLE_EQ(mapped.minCoeff(&row, &col), -5.0);
    EXPECT_EQ(row, 0);
    EXPECT_EQ(col, 2);

    plamatrix::MatrixXd owned = mapped;
    plamatrix::Ref<const plamatrix::MatrixXd> reference(owned);
    EXPECT_EQ(reference.count(), 6);
    EXPECT_TRUE(reference.cwiseAbs().isApprox(mapped.cwiseAbs()));

    const auto block = owned.block<2, 2>(0, 1);
    EXPECT_DOUBLE_EQ(block.unaryExpr(Shift{}).sum(), 10.0);
    EXPECT_DOUBLE_EQ(owned.diagonal().cwiseAbs().sum(), 5.0);
}

TEST(MatrixFuzzy, ApproximationUsesEigenNormSemantics)
{
    const plamatrix::Vector2d reference(1.0, 0.0);
    const plamatrix::Vector2d nearby(1.0, 1.0e-7);
    EXPECT_TRUE(reference.isApprox(nearby, 1.0e-6));

    const plamatrix::Vector2d zero = plamatrix::Vector2d::Zero();
    const plamatrix::Vector2d tiny(1.0e-12, 0.0);
    EXPECT_FALSE(tiny.isApprox(zero, 1.0e-6));
    EXPECT_TRUE(tiny.isMuchSmallerThan(1.0, 1.0e-6));
    EXPECT_TRUE(tiny.isMuchSmallerThan(reference, 1.0e-6));
    EXPECT_FALSE(reference.isMuchSmallerThan(tiny, 1.0e-6));

    const auto expression = (reference + nearby).cwiseAbs();
    EXPECT_TRUE(expression.isApprox(plamatrix::Vector2d(2.0, 1.0e-7), 1.0e-12));
}
