#include <gtest/gtest.h>

#include "plamatrix/internal/core/cancellation.h"
#include "plamatrix/internal/core/capabilities.h"
#include "plamatrix/internal/core/diagnostics.h"
#include "plamatrix/internal/core/execution_context.h"

namespace plamatrix::internal
{

TEST(Capabilities, CpuContextReportsStableCoreCapabilities)
{
    const auto context = ExecutionContext::create();
    const auto& capabilities = context.capabilities();
    EXPECT_TRUE(capabilities.float32);
    EXPECT_TRUE(capabilities.float64);
    EXPECT_TRUE(capabilities.hostMemory);
    EXPECT_FALSE(capabilities.cooperativeMatrix);
}

TEST(CancellationToken, CanBeCancelledAndObserved)
{
    CancellationToken token;
    EXPECT_FALSE(token.isCancellationRequested());
    token.cancel();
    EXPECT_TRUE(token.isCancellationRequested());
}

TEST(Diagnostics, TraceStartsEmpty)
{
    ExecutionTrace trace;
    EXPECT_EQ(trace.operationCount(), 0U);
    trace.record("test", 1.25);
    EXPECT_EQ(trace.operationCount(), 1U);
    EXPECT_EQ(trace.entries().front().operation, "test");
}

} // namespace plamatrix::internal
