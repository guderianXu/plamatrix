#include <atomic>
#include <chrono>
#include <limits>
#include <thread>
#include <utility>

#include "indexing_test_utils.h"

#ifdef PLAMATRIX_WITH_CUDA

namespace plamatrix
{
namespace
{

using indexing_test_detail::expectGpuMatrix;
using indexing_test_detail::expectLogicErrorContaining;
using indexing_test_detail::makeMatrix;

TEST(IndexingWorkspaceCudaTest, ReuseMoveCrossStreamResetCloseAndFailedGrowthAreStrong)
{
    test::CudaStreamGuard first_stream;
    test::CudaStreamGuard second_stream;
    IndexingWorkspace workspace;
    workspace.reserveBytesAsync(512, first_stream.get());
    const std::size_t capacity = workspace.capacityBytes();
    void* const data = workspace.data();
    workspace.reserveBytesAsync(capacity, first_stream.get());
    EXPECT_EQ(workspace.capacityBytes(), capacity);
    EXPECT_EQ(workspace.data(), data);

    IndexingWorkspace moved(std::move(workspace));
    EXPECT_EQ(workspace.capacityBytes(), 0U);
    EXPECT_EQ(workspace.data(), nullptr);
    expectLogicErrorContaining([&] {
        moved.reserveBytesAsync(capacity, second_stream.get());
    }, {"different stream", "synchronize", "close"});
    EXPECT_THROW(
        moved.reserveBytesAsync(std::numeric_limits<std::size_t>::max(), first_stream.get()),
        std::runtime_error);
    static_cast<void>(cudaGetLastError());
    EXPECT_EQ(moved.capacityBytes(), capacity);
    EXPECT_EQ(moved.data(), data);

    first_stream.synchronize();
    moved.closeAsyncAllocation();
    EXPECT_EQ(moved.capacityBytes(), 0U);
    EXPECT_EQ(moved.data(), nullptr);
    EXPECT_NO_THROW(moved.closeAsyncAllocation());
    first_stream.synchronize();

    moved.reserveBytes(1024);
    void* const normal_data = moved.data();
    moved.reserveBytesAsync(1024, first_stream.get());
    first_stream.synchronize();
    expectLogicErrorContaining([&] {
        moved.reserveBytesAsync(1024, second_stream.get());
    }, {"different stream", "synchronize", "reset"});
    moved.reserveBytes(1024);
    EXPECT_EQ(moved.data(), normal_data);
    EXPECT_NO_THROW(moved.reserveBytesAsync(1024, second_stream.get()));
    second_stream.synchronize();
}

TEST(IndexingWorkspaceCudaTest, NormalResetRejectsPendingReuseStreamAndAllowsCompletedStream)
{
    test::CudaStreamGuard stream;
    IndexingWorkspace workspace;
    workspace.reserveBytes(512);
    std::atomic<bool> release{false};
    PLAMATRIX_CHECK_CUDA(cudaLaunchHostFunc(stream.get(), [](void* context) {
        auto* flag = static_cast<std::atomic<bool>*>(context);
        while (!flag->load(std::memory_order_acquire))
        {
            std::this_thread::yield();
        }
    }, &release));
    workspace.reserveBytesAsync(512, stream.get());
    std::thread fallback_release([&] {
        for (int attempt = 0; attempt < 200 && !release.load(std::memory_order_acquire); ++attempt)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        release.store(true, std::memory_order_release);
    });

    expectLogicErrorContaining([&] {
        workspace.reserveBytes(512);
    }, {"synchronize"});
    release.store(true, std::memory_order_release);
    fallback_release.join();
    stream.synchronize();
    EXPECT_NO_THROW(workspace.reserveBytes(512));
}

TEST(IndexingWorkspaceCudaTest, NormalResetRejectsUnconsumedOverflowStatus)
{
    auto counts = makeMatrix<Index>(2, 1, {
        std::numeric_limits<Index>::max(), Index(1)
    }).toGpu();
    DenseMatrix<Index, Device::GPU> output(2, 1);
    test::CudaStreamGuard stream;
    IndexingWorkspace workspace;
    workspace.reserveBytes(1024U * 1024U);
    const std::size_t capacity = workspace.capacityBytes();

    exclusiveScanAsync(counts, output, workspace, stream.get());
    stream.synchronize();
    expectLogicErrorContaining([&] {
        workspace.reserveBytes(capacity);
    }, {"checkStatus"});
    try
    {
        workspace.checkStatus("exclusiveScanAsync");
        FAIL() << "Expected preserved async overflow";
    }
    catch (const std::overflow_error& error)
    {
        EXPECT_NE(std::string(error.what()).find("source offset 1"), std::string::npos);
    }
    EXPECT_NO_THROW(workspace.reserveBytes(capacity));
}

TEST(IndexingWorkspaceCudaTest, AsyncCloseRejectsUnconsumedOutOfRangeStatus)
{
    auto input = makeMatrix<float>(1, 1, {10.0F}).toGpu();
    auto indices = makeMatrix<Index>(1, 1, {Index(1)}).toGpu();
    DenseMatrix<float, Device::GPU> output(1, 1);
    test::CudaStreamGuard stream;
    IndexingWorkspace workspace;

    gatherRowsAsync(input, indices, output, workspace, stream.get());
    stream.synchronize();
    expectLogicErrorContaining([&] {
        workspace.closeAsyncAllocation();
    }, {"checkStatus"});
    try
    {
        workspace.checkStatus("gatherRowsAsync");
        FAIL() << "Expected preserved async index error";
    }
    catch (const std::out_of_range& error)
    {
        EXPECT_NE(std::string(error.what()).find("source offset 0"), std::string::npos);
    }
    EXPECT_NO_THROW(workspace.closeAsyncAllocation());
    stream.synchronize();
}

TEST(IndexingScanCudaTest, AsyncCallDoesNotWaitForQueuedHostWork)
{
    auto counts = makeMatrix<Index>(4, 1, {1, 2, 3, 4}).toGpu();
    test::CudaStreamGuard stream;
    IndexingWorkspace workspace;
    DenseMatrix<Index, Device::GPU> output(4, 1);
    exclusiveScanAsync(counts, output, workspace, stream.get());
    stream.synchronize();
    workspace.checkStatus("exclusiveScanAsync warmup");
    std::atomic<bool> release{false};
    PLAMATRIX_CHECK_CUDA(cudaLaunchHostFunc(stream.get(), [](void* context) {
        auto* flag = static_cast<std::atomic<bool>*>(context);
        while (!flag->load(std::memory_order_acquire))
        {
            std::this_thread::yield();
        }
    }, &release));
    std::thread fallback_release([&] {
        for (int attempt = 0; attempt < 200 && !release.load(std::memory_order_acquire); ++attempt)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        release.store(true, std::memory_order_release);
    });

    const auto started = std::chrono::steady_clock::now();
    exclusiveScanAsync(counts, output, workspace, stream.get());
    const auto elapsed = std::chrono::steady_clock::now() - started;
    expectLogicErrorContaining([&] {
        workspace.checkStatus("exclusiveScanAsync");
    }, {"synchronized first"});
    release.store(true, std::memory_order_release);
    fallback_release.join();
    EXPECT_LT(elapsed, std::chrono::seconds(1));
    stream.synchronize();
    workspace.checkStatus("exclusiveScanAsync");
    expectGpuMatrix(output, 4, 1, {Index(0), Index(1), Index(3), Index(6)});
    workspace.closeAsyncAllocation();
    stream.synchronize();
}

} // namespace
} // namespace plamatrix

#endif
