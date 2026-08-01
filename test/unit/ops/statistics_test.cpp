#include <gtest/gtest.h>

#include "plamatrix/ops/statistics.h"

#include <cmath>
#include <limits>
#include <vector>

TEST(StatisticsTest, FiniteMedianHandlesOddAndEvenCounts)
{
    const auto odd = plamatrix::finiteMedian(std::vector<double>{9.0, 1.0, 2.0});
    const auto even = plamatrix::finiteMedian(std::vector<double>{9.0, 1.0, 2.0, 4.0});

    ASSERT_TRUE(odd.has_value());
    ASSERT_TRUE(even.has_value());
    EXPECT_DOUBLE_EQ(*odd, 2.0);
    EXPECT_DOUBLE_EQ(*even, 3.0);
}

TEST(StatisticsTest, FiniteMedianIgnoresNonFiniteValues)
{
    const auto median = plamatrix::finiteMedian(
        std::vector<double>{
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
    const auto median = plamatrix::finiteMedian(
        std::vector<double>{
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::infinity(),
        });

    EXPECT_FALSE(median.has_value());
}

TEST(StatisticsTest, FiniteMedianDoesNotOverflowAtFiniteExtremes)
{
    const double maximum = std::numeric_limits<double>::max();
    const auto sameSign =
        plamatrix::finiteMedian(
            std::vector<double>{maximum, maximum});
    const auto oppositeSign =
        plamatrix::finiteMedian(
            std::vector<double>{-maximum, maximum});

    ASSERT_TRUE(sameSign.has_value());
    ASSERT_TRUE(oppositeSign.has_value());
    EXPECT_DOUBLE_EQ(*sameSign, maximum);
    EXPECT_DOUBLE_EQ(*oppositeSign, 0.0);
}
