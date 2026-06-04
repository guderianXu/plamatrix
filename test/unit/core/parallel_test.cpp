#include <gtest/gtest.h>

#include <plamatrix/core/parallel.h>

using namespace plamatrix;

TEST(Parallel, shouldUseOpenMp_HasSmallWorkThreshold)
{
    EXPECT_FALSE(detail::shouldUseOpenMp(detail::kOpenMpWorkThreshold - 1));
    EXPECT_TRUE(detail::shouldUseOpenMp(detail::kOpenMpWorkThreshold));
}
