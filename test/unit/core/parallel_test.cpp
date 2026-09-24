#include <gtest/gtest.h>

#include <plamatrix/internal/core/parallel.h>

namespace plamatrix::internal
{

TEST(Parallel, shouldUseOpenMp_HasSmallWorkThreshold)
{
    EXPECT_FALSE(detail::shouldUseOpenMp(detail::kOpenMpWorkThreshold - 1));
    EXPECT_TRUE(detail::shouldUseOpenMp(detail::kOpenMpWorkThreshold));
}

} // namespace plamatrix::internal
