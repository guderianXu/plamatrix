#include <gtest/gtest.h>

#include "plamatrix/internal/core/error.h"

namespace plamatrix::internal
{

TEST(Error, PreservesStructuredBackendInformation)
{
    const Error error(ErrorCode::BackendFailure, "test failure", Backend::Cuda, 17);
    EXPECT_STREQ(error.what(), "test failure");
    EXPECT_EQ(error.code(), ErrorCode::BackendFailure);
    EXPECT_EQ(error.backend(), Backend::Cuda);
    EXPECT_EQ(error.nativeCode(), 17);
}

} // namespace plamatrix::internal
