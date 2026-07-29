#include <cstdint>
#include <limits>

#include "indexing_test_utils.h"

#ifdef PLAMATRIX_WITH_CUDA

namespace plamatrix
{
namespace
{

using indexing_test_detail::expectMatrix;
using indexing_test_detail::IndexingTest;
using indexing_test_detail::makeMatrix;
using indexing_test_detail::ScalarTypes;

TYPED_TEST_SUITE(IndexingTest, ScalarTypes);
using indexing_test_detail::expectGpuMatrix;

TEST(IndexingDenseUint8CudaTest, FillAndTransferSupportAllZeroAndAllOneMasks)
{
    DenseMatrix<std::uint8_t, Device::CPU> zero_cpu(5, 1);
    zero_cpu.fill(std::uint8_t(0));
    auto zero_gpu = zero_cpu.toGpu();
    expectGpuMatrix<std::uint8_t>(zero_gpu, 5, 1, {0, 0, 0, 0, 0});

    DenseMatrix<std::uint8_t, Device::GPU> one_gpu(5, 1);
    one_gpu.fill(std::uint8_t(1));
    expectGpuMatrix<std::uint8_t>(one_gpu, 5, 1, {1, 1, 1, 1, 1});
}

TEST(IndexingScanCudaTest, MatchesCpuColumnMajorAndSupportsOutputReuse)
{
    const auto counts_cpu = makeMatrix<Index>(2, 3, {2, -1, 3, 4, -2, 1});
    auto counts = counts_cpu.toGpu();
    expectGpuMatrix<Index>(exclusiveScan(counts), 2, 3, {0, 2, 1, 4, 8, 6});

    test::CudaStreamGuard stream;
    IndexingWorkspace workspace;
    DenseMatrix<Index, Device::GPU> output(2, 3);
    exclusiveScan(counts, output, workspace, stream.get());
    expectGpuMatrix<Index>(output, 2, 3, {0, 2, 1, 4, 8, 6});
    EXPECT_GT(workspace.capacityBytes(), 0U);
    workspace.closeAsyncAllocation();
    stream.synchronize();
}

TEST(IndexingScanCudaTest, ReportsLowestPositiveAndNegativeOverflowOffsets)
{
    const auto positive = makeMatrix<Index>(3, 1, {
        Index(4), std::numeric_limits<Index>::max(), Index(1)
    }).toGpu();
    const auto negative = makeMatrix<Index>(3, 1, {
        Index(-4), std::numeric_limits<Index>::min(), Index(-1)
    }).toGpu();

    try
    {
        static_cast<void>(exclusiveScan(positive));
        FAIL() << "Expected positive overflow";
    }
    catch (const std::overflow_error& error)
    {
        EXPECT_NE(std::string(error.what()).find("source offset 1"), std::string::npos);
    }
    try
    {
        static_cast<void>(exclusiveScan(negative));
        FAIL() << "Expected negative overflow";
    }
    catch (const std::overflow_error& error)
    {
        EXPECT_NE(std::string(error.what()).find("source offset 1"), std::string::npos);
    }
}

TEST(IndexingScanCudaTest, AcceptsExactLimitsAndChecksFinalAddition)
{
    const auto positive = makeMatrix<Index>(2, 1, {
        std::numeric_limits<Index>::max(), Index(0)
    }).toGpu();
    const auto negative = makeMatrix<Index>(2, 1, {
        std::numeric_limits<Index>::min(), Index(0)
    }).toGpu();
    const auto final_overflow = makeMatrix<Index>(2, 1, {
        std::numeric_limits<Index>::max(), Index(1)
    }).toGpu();

    expectGpuMatrix(exclusiveScan(positive), 2, 1, {
        Index(0), std::numeric_limits<Index>::max()
    });
    expectGpuMatrix(exclusiveScan(negative), 2, 1, {
        Index(0), std::numeric_limits<Index>::min()
    });
    EXPECT_THROW(static_cast<void>(exclusiveScan(final_overflow)), std::overflow_error);
}

TEST(IndexingScanCudaTest, PreservesEmptyGpuShape)
{
    DenseMatrix<Index, Device::CPU> empty_cpu(0, 3);

    const auto empty = exclusiveScan(empty_cpu.toGpu());

    expectGpuMatrix<Index>(empty, 0, 3, {});
}

TYPED_TEST(IndexingTest, GpuGatherMatchesCpuWithDuplicatesAndAsyncReuse)
{
    using Scalar = TypeParam;
    const auto input_cpu = makeMatrix<Scalar>(4, 3, {
        Scalar(10), Scalar(11), Scalar(12), Scalar(13),
        Scalar(20), Scalar(21), Scalar(22), Scalar(23),
        Scalar(30), Scalar(31), Scalar(32), Scalar(33)
    });
    const auto indices_cpu = makeMatrix<Index>(3, 1, {2, 0, 2});
    auto input = input_cpu.toGpu();
    auto indices = indices_cpu.toGpu();

    expectGpuMatrix(gatherRows(input, indices), 3, 3, {
        Scalar(12), Scalar(10), Scalar(12),
        Scalar(22), Scalar(20), Scalar(22),
        Scalar(32), Scalar(30), Scalar(32)
    });

    test::CudaStreamGuard stream;
    IndexingWorkspace workspace;
    DenseMatrix<Scalar, Device::GPU> first(3, 3);
    DenseMatrix<Scalar, Device::GPU> second(3, 3);
    gatherRowsAsync(input, indices, first, workspace, stream.get());
    const std::size_t capacity = workspace.capacityBytes();
    void* const data = workspace.data();
    gatherRowsAsync(input, indices, second, workspace, stream.get());
    EXPECT_EQ(workspace.capacityBytes(), capacity);
    EXPECT_EQ(workspace.data(), data);
    stream.synchronize();
    workspace.checkStatus("gatherRowsAsync");
    expectGpuMatrix(second, 3, 3, {
        Scalar(12), Scalar(10), Scalar(12),
        Scalar(22), Scalar(20), Scalar(22),
        Scalar(32), Scalar(30), Scalar(32)
    });
    workspace.closeAsyncAllocation();
    stream.synchronize();
}

TYPED_TEST(IndexingTest, GpuGatherInvalidIndexReportsLowestOffsetAndLeavesOutputUnchanged)
{
    using Scalar = TypeParam;
    auto input = makeMatrix<Scalar>(3, 2, {
        Scalar(1), Scalar(2), Scalar(3), Scalar(4), Scalar(5), Scalar(6)
    }).toGpu();
    auto indices = makeMatrix<Index>(4, 1, {1, 3, -1, 0}).toGpu();
    const auto initial = makeMatrix<Scalar>(4, 2, {
        Scalar(9), Scalar(8), Scalar(7), Scalar(6),
        Scalar(5), Scalar(4), Scalar(3), Scalar(2)
    });
    auto output = initial.toGpu();
    IndexingWorkspace workspace;

    try
    {
        gatherRows(input, indices, output, workspace);
        FAIL() << "Expected invalid gather index";
    }
    catch (const std::out_of_range& error)
    {
        EXPECT_NE(std::string(error.what()).find("source offset 1"), std::string::npos);
    }
    expectGpuMatrix(output, 4, 2, {
        Scalar(9), Scalar(8), Scalar(7), Scalar(6),
        Scalar(5), Scalar(4), Scalar(3), Scalar(2)
    });

    test::CudaStreamGuard stream;
    IndexingWorkspace async_workspace;
    auto async_output = initial.toGpu();
    gatherRowsAsync(input, indices, async_output, async_workspace, stream.get());
    stream.synchronize();
    try
    {
        async_workspace.checkStatus("gatherRowsAsync");
        FAIL() << "Expected async invalid gather index";
    }
    catch (const std::out_of_range& error)
    {
        EXPECT_NE(std::string(error.what()).find("source offset 1"), std::string::npos);
    }
    expectGpuMatrix(async_output, 4, 2, {
        Scalar(9), Scalar(8), Scalar(7), Scalar(6),
        Scalar(5), Scalar(4), Scalar(3), Scalar(2)
    });
    async_workspace.closeAsyncAllocation();
    stream.synchronize();
}

TYPED_TEST(IndexingTest, GpuAsyncStatusBatchPreservesGatherFailureUntilConsumed)
{
    using Scalar = TypeParam;
    auto input = makeMatrix<Scalar>(3, 1, {
        Scalar(10), Scalar(20), Scalar(30)
    }).toGpu();
    auto bad_indices = makeMatrix<Index>(1, 1, {Index(3)}).toGpu();
    auto valid_indices = makeMatrix<Index>(1, 1, {Index(1)}).toGpu();
    auto bad_output = makeMatrix<Scalar>(1, 1, {Scalar(-1)}).toGpu();
    auto protected_output = makeMatrix<Scalar>(1, 1, {Scalar(-2)}).toGpu();
    test::CudaStreamGuard stream;
    IndexingWorkspace workspace;

    gatherRowsAsync(
        input, bad_indices, bad_output, workspace, stream.get());
    gatherRowsAsync(
        input, valid_indices, protected_output, workspace, stream.get());
    stream.synchronize();
    EXPECT_THROW(workspace.checkStatus("gatherRowsAsync batch"), std::out_of_range);
    expectGpuMatrix(bad_output, 1, 1, {Scalar(-1)});
    expectGpuMatrix(protected_output, 1, 1, {Scalar(-2)});

    gatherRowsAsync(
        input, valid_indices, protected_output, workspace, stream.get());
    stream.synchronize();
    EXPECT_NO_THROW(workspace.checkStatus("gatherRowsAsync reused batch"));
    expectGpuMatrix(protected_output, 1, 1, {Scalar(20)});
    workspace.closeAsyncAllocation();
    stream.synchronize();
}

TYPED_TEST(IndexingTest, GpuAsyncStatusBatchPreservesOverflowAcrossSuccessfulGather)
{
    using Scalar = TypeParam;
    auto counts = makeMatrix<Index>(2, 1, {
        std::numeric_limits<Index>::max(), Index(1)
    }).toGpu();
    DenseMatrix<Index, Device::GPU> scan_output(2, 1);
    auto input = makeMatrix<Scalar>(2, 1, {Scalar(10), Scalar(20)}).toGpu();
    auto indices = makeMatrix<Index>(1, 1, {Index(1)}).toGpu();
    DenseMatrix<Scalar, Device::GPU> gather_output(1, 1);
    test::CudaStreamGuard stream;
    IndexingWorkspace workspace;

    exclusiveScanAsync(counts, scan_output, workspace, stream.get());
    gatherRowsAsync(input, indices, gather_output, workspace, stream.get());
    stream.synchronize();
    EXPECT_THROW(workspace.checkStatus("mixed indexing batch"), std::overflow_error);
    expectGpuMatrix(gather_output, 1, 1, {Scalar(20)});
    workspace.closeAsyncAllocation();
    stream.synchronize();
}

TYPED_TEST(IndexingTest, GpuAsyncStatusBatchReportsOverflowBeforeOutOfRange)
{
    using Scalar = TypeParam;
    auto input = makeMatrix<Scalar>(1, 1, {Scalar(10)}).toGpu();
    auto bad_indices = makeMatrix<Index>(1, 1, {Index(1)}).toGpu();
    DenseMatrix<Scalar, Device::GPU> gather_output(1, 1);
    auto counts = makeMatrix<Index>(2, 1, {
        std::numeric_limits<Index>::max(), Index(1)
    }).toGpu();
    DenseMatrix<Index, Device::GPU> scan_output(2, 1);
    test::CudaStreamGuard stream;
    IndexingWorkspace workspace;

    gatherRowsAsync(input, bad_indices, gather_output, workspace, stream.get());
    exclusiveScanAsync(counts, scan_output, workspace, stream.get());
    stream.synchronize();
    EXPECT_THROW(workspace.checkStatus("mixed error batch"), std::overflow_error);
    workspace.closeAsyncAllocation();
    stream.synchronize();
}

TYPED_TEST(IndexingTest, GpuGatherSupportsEmptySelection)
{
    using Scalar = TypeParam;
    DenseMatrix<Scalar, Device::CPU> input_cpu(4, 2);
    DenseMatrix<Index, Device::CPU> indices_cpu(0, 1);

    const auto gathered = gatherRows(input_cpu.toGpu(), indices_cpu.toGpu());

    expectGpuMatrix<Scalar>(gathered, 0, 2, {});
}

} // namespace
} // namespace plamatrix

#endif
