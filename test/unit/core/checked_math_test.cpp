#include <limits>

#include <gtest/gtest.h>

#include "plamatrix/core/checked_math.h"

namespace plamatrix
{

TEST(CheckedMath, RejectsIndexOverflow)
{
    EXPECT_THROW(detail::checkedIndexMul(std::numeric_limits<Index>::max(), 2, "test"),
                 std::overflow_error);
    EXPECT_EQ(detail::checkedIndexMul(7, 0, "test"), 0);
}

TEST(CheckedMath, RejectsSizeOverflow)
{
    EXPECT_THROW(detail::checkedSizeMul(std::numeric_limits<std::size_t>::max(), 2, "test"),
                 std::overflow_error);
    EXPECT_EQ(detail::checkedSizeMul(3, 4, "test"), 12U);
}

} // namespace plamatrix
