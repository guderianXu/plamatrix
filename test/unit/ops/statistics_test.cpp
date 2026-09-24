#include <gtest/gtest.h>

#include "plamatrix/internal/ops/statistics.h"

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <vector>

namespace
{

    template <typename Scalar>
    plamatrix::internal::DenseStorage<Scalar, plamatrix::internal::Device::CPU>
    makeMatrix(plamatrix::Index rows, plamatrix::Index cols, std::initializer_list<Scalar> values)
    {
        plamatrix::internal::DenseStorage<Scalar, plamatrix::internal::Device::CPU> matrix(rows, cols);
        std::copy(values.begin(), values.end(), matrix.data());
        return matrix;
    }

    template <typename Scalar> class FiniteStatisticsTest : public ::testing::Test
    {
    };

#if defined(PLAMATRIX_USE_FLOAT) && defined(PLAMATRIX_USE_DOUBLE)
    using FloatingTypes = ::testing::Types<float, double>;
#elif defined(PLAMATRIX_USE_FLOAT)
    using FloatingTypes = ::testing::Types<float>;
#else
    using FloatingTypes = ::testing::Types<double>;
#endif

    TYPED_TEST_SUITE(FiniteStatisticsTest, FloatingTypes);

} // namespace

TEST(StatisticsTest, FiniteMedianHandlesOddAndEvenCounts)
{
    const auto odd = plamatrix::internal::finiteMedian(std::vector<double>{9.0, 1.0, 2.0});
    const auto even = plamatrix::internal::finiteMedian(std::vector<double>{9.0, 1.0, 2.0, 4.0});

    ASSERT_TRUE(odd.has_value());
    ASSERT_TRUE(even.has_value());
    EXPECT_DOUBLE_EQ(*odd, 2.0);
    EXPECT_DOUBLE_EQ(*even, 3.0);
}

TEST(StatisticsTest, FiniteMedianIgnoresNonFiniteValues)
{
    const auto median = plamatrix::internal::finiteMedian(std::vector<double>{
        std::numeric_limits<double>::quiet_NaN(),
        7.0,
        std::numeric_limits<double>::infinity(),
        3.0,
    });

    ASSERT_TRUE(median.has_value());
    EXPECT_DOUBLE_EQ(*median, 5.0);
}

TEST(StatisticsTest, FiniteMedianReturnsEmptyWhenNoFiniteValueExists)
{
    const auto median = plamatrix::internal::finiteMedian(std::vector<double>{
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::infinity(),
    });

    EXPECT_FALSE(median.has_value());
}

TEST(StatisticsTest, FiniteMedianDoesNotOverflowAtFiniteExtremes)
{
    const double maximum = std::numeric_limits<double>::max();
    const auto sameSign = plamatrix::internal::finiteMedian(std::vector<double>{maximum, maximum});
    const auto oppositeSign = plamatrix::internal::finiteMedian(std::vector<double>{-maximum, maximum});

    ASSERT_TRUE(sameSign.has_value());
    ASSERT_TRUE(oppositeSign.has_value());
    EXPECT_DOUBLE_EQ(*sameSign, maximum);
    EXPECT_DOUBLE_EQ(*oppositeSign, 0.0);
}

TYPED_TEST(FiniteStatisticsTest, MasksRowsAndComputesBoundsAcrossOnlyFullyFiniteRows)
{
    using Scalar = TypeParam;
    const Scalar nan = std::numeric_limits<Scalar>::quiet_NaN();
    const Scalar infinity = std::numeric_limits<Scalar>::infinity();
    const auto input = makeMatrix<Scalar>(4,
                                          3,
                                          {
                                              Scalar(1),
                                              nan,
                                              Scalar(3),
                                              Scalar(4),
                                              Scalar(10),
                                              Scalar(20),
                                              infinity,
                                              Scalar(40),
                                              Scalar(-1),
                                              Scalar(-2),
                                              Scalar(-3),
                                              Scalar(-4),
                                          });

    const auto mask = plamatrix::internal::finiteRowMask(input);
    const auto bounds = plamatrix::internal::finiteColumnBounds(input);
    const auto combined = plamatrix::internal::finiteColumnBoundsWithMask(input);

    ASSERT_EQ(mask.rows(), 4);
    ASSERT_EQ(mask.cols(), 1);
    EXPECT_EQ(mask.data()[0], 1);
    EXPECT_EQ(mask.data()[1], 0);
    EXPECT_EQ(mask.data()[2], 0);
    EXPECT_EQ(mask.data()[3], 1);
    EXPECT_EQ(bounds.validRowCount.data()[0], 2);
    EXPECT_EQ(bounds.minimum.data()[0], Scalar(1));
    EXPECT_EQ(bounds.maximum.data()[0], Scalar(4));
    EXPECT_EQ(bounds.minimum.data()[1], Scalar(10));
    EXPECT_EQ(bounds.maximum.data()[1], Scalar(40));
    EXPECT_EQ(bounds.minimum.data()[2], Scalar(-4));
    EXPECT_EQ(bounds.maximum.data()[2], Scalar(-1));
    EXPECT_EQ(combined.rowMask.data()[0], 1);
    EXPECT_EQ(combined.rowMask.data()[1], 0);
    EXPECT_EQ(combined.rowMask.data()[2], 0);
    EXPECT_EQ(combined.rowMask.data()[3], 1);
    EXPECT_EQ(combined.validRowCount.data()[0], 2);
    EXPECT_EQ(combined.minimum.data()[0], Scalar(1));
    EXPECT_EQ(combined.maximum.data()[0], Scalar(4));
    EXPECT_EQ(combined.minimum.data()[1], Scalar(10));
    EXPECT_EQ(combined.maximum.data()[1], Scalar(40));
    EXPECT_EQ(combined.minimum.data()[2], Scalar(-4));
    EXPECT_EQ(combined.maximum.data()[2], Scalar(-1));
}

TYPED_TEST(FiniteStatisticsTest, EmptyAndAllInvalidInputsHaveExplicitSemantics)
{
    using Scalar = TypeParam;
    const Scalar nan = std::numeric_limits<Scalar>::quiet_NaN();
    const Scalar infinity = std::numeric_limits<Scalar>::infinity();
    const auto invalid = makeMatrix<Scalar>(2, 2, {nan, infinity, Scalar(1), Scalar(2)});

    const auto invalid_bounds = plamatrix::internal::finiteColumnBounds(invalid);
    const auto invalid_combined = plamatrix::internal::finiteColumnBoundsWithMask(invalid);
    EXPECT_EQ(invalid_bounds.validRowCount.data()[0], 0);
    EXPECT_TRUE(std::isnan(invalid_bounds.minimum.data()[0]));
    EXPECT_TRUE(std::isnan(invalid_bounds.maximum.data()[1]));
    EXPECT_EQ(invalid_combined.rowMask.data()[0], 0);
    EXPECT_EQ(invalid_combined.rowMask.data()[1], 0);
    EXPECT_EQ(invalid_combined.validRowCount.data()[0], 0);
    EXPECT_TRUE(std::isnan(invalid_combined.minimum.data()[0]));
    EXPECT_TRUE(std::isnan(invalid_combined.maximum.data()[1]));

    const plamatrix::internal::DenseStorage<Scalar, plamatrix::internal::Device::CPU> empty_rows(0, 3);
    const auto empty_bounds = plamatrix::internal::finiteColumnBounds(empty_rows);
    const auto empty_combined = plamatrix::internal::finiteColumnBoundsWithMask(empty_rows);
    EXPECT_EQ(empty_bounds.validRowCount.data()[0], 0);
    ASSERT_EQ(empty_bounds.minimum.cols(), 3);
    for (plamatrix::Index column = 0; column < 3; ++column)
    {
        EXPECT_TRUE(std::isnan(empty_bounds.minimum.data()[column]));
        EXPECT_TRUE(std::isnan(empty_bounds.maximum.data()[column]));
        EXPECT_TRUE(std::isnan(empty_combined.minimum.data()[column]));
        EXPECT_TRUE(std::isnan(empty_combined.maximum.data()[column]));
    }
    EXPECT_EQ(empty_combined.rowMask.rows(), 0);
    EXPECT_EQ(empty_combined.rowMask.cols(), 1);
    EXPECT_EQ(empty_combined.validRowCount.data()[0], 0);

    const plamatrix::internal::DenseStorage<Scalar, plamatrix::internal::Device::CPU> zero_columns(3, 0);
    const auto zero_mask = plamatrix::internal::finiteRowMask(zero_columns);
    const auto zero_bounds = plamatrix::internal::finiteColumnBounds(zero_columns);
    const auto zero_combined = plamatrix::internal::finiteColumnBoundsWithMask(zero_columns);
    EXPECT_EQ(zero_mask.data()[0], 1);
    EXPECT_EQ(zero_mask.data()[1], 1);
    EXPECT_EQ(zero_mask.data()[2], 1);
    EXPECT_EQ(zero_bounds.validRowCount.data()[0], 3);
    EXPECT_EQ(zero_bounds.minimum.rows(), 1);
    EXPECT_EQ(zero_bounds.minimum.cols(), 0);
    EXPECT_EQ(zero_combined.rowMask.data()[0], 1);
    EXPECT_EQ(zero_combined.rowMask.data()[1], 1);
    EXPECT_EQ(zero_combined.rowMask.data()[2], 1);
    EXPECT_EQ(zero_combined.validRowCount.data()[0], 3);
    EXPECT_EQ(zero_combined.minimum.rows(), 1);
    EXPECT_EQ(zero_combined.minimum.cols(), 0);
}
