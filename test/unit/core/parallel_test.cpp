#include <gtest/gtest.h>

#include <plamatrix/core/parallel.h>

using namespace plamatrix;

TEST(Parallel, shouldUseOpenMp_HasSmallWorkThreshold)
{
    EXPECT_FALSE(detail::shouldUseOpenMp(detail::kOpenMpWorkThreshold - 1));
#ifdef PLAMATRIX_WITH_OPENMP
    EXPECT_TRUE(detail::shouldUseOpenMp(detail::kOpenMpWorkThreshold));
#else
    EXPECT_FALSE(detail::shouldUseOpenMp(detail::kOpenMpWorkThreshold));
#endif
}
