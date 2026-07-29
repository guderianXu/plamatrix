#include <cstdint>

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

TYPED_TEST(IndexingTest, GpuScatterIsDeterministicAndPreservesUntouchedRows)
{
    using Scalar = TypeParam;
    auto values = makeMatrix<Scalar>(3, 2, {
        Scalar(100), Scalar(110), Scalar(120), Scalar(200), Scalar(210), Scalar(220)
    }).toGpu();
    auto indices = makeMatrix<Index>(3, 1, {2, 0, 2}).toGpu();
    auto output = makeMatrix<Scalar>(4, 2, {
        Scalar(1), Scalar(2), Scalar(3), Scalar(4),
        Scalar(5), Scalar(6), Scalar(7), Scalar(8)
    }).toGpu();

    scatterRows(values, indices, output);

    expectGpuMatrix(output, 4, 2, {
        Scalar(110), Scalar(2), Scalar(100), Scalar(4),
        Scalar(210), Scalar(6), Scalar(200), Scalar(8)
    });
}

TYPED_TEST(IndexingTest, GpuScatterAsyncSucceedsOnNonDefaultStream)
{
    using Scalar = TypeParam;
    auto values = makeMatrix<Scalar>(2, 1, {Scalar(7), Scalar(9)}).toGpu();
    auto indices = makeMatrix<Index>(2, 1, {Index(2), Index(0)}).toGpu();
    auto output = makeMatrix<Scalar>(3, 1, {
        Scalar(-1), Scalar(-1), Scalar(-1)
    }).toGpu();
    test::CudaStreamGuard stream;
    IndexingWorkspace workspace;

    scatterRowsAsync(values, indices, output, workspace, stream.get());
    stream.synchronize();
    EXPECT_NO_THROW(workspace.checkStatus("scatterRowsAsync"));
    expectGpuMatrix(output, 3, 1, {Scalar(9), Scalar(-1), Scalar(7)});
    workspace.closeAsyncAllocation();
    stream.synchronize();
}

TYPED_TEST(IndexingTest, GpuScatterInvalidIndexLeavesOutputUnchangedIncludingZeroColumns)
{
    using Scalar = TypeParam;
    auto values = makeMatrix<Scalar>(3, 1, {Scalar(7), Scalar(8), Scalar(9)}).toGpu();
    auto indices = makeMatrix<Index>(3, 1, {0, 4, -1}).toGpu();
    auto output = makeMatrix<Scalar>(4, 1, {
        Scalar(1), Scalar(2), Scalar(3), Scalar(4)
    }).toGpu();

    try
    {
        scatterRows(values, indices, output);
        FAIL() << "Expected invalid scatter index";
    }
    catch (const std::out_of_range& error)
    {
        EXPECT_NE(std::string(error.what()).find("source offset 1"), std::string::npos);
    }
    expectGpuMatrix(output, 4, 1, {Scalar(1), Scalar(2), Scalar(3), Scalar(4)});

    DenseMatrix<Scalar, Device::CPU> zero_values_cpu(1, 0);
    DenseMatrix<Scalar, Device::CPU> zero_output_cpu(4, 0);
    auto zero_values = zero_values_cpu.toGpu();
    auto zero_output = zero_output_cpu.toGpu();
    auto bad = makeMatrix<Index>(1, 1, {4}).toGpu();
    EXPECT_THROW(scatterRows(zero_values, bad, zero_output), std::out_of_range);
}

TYPED_TEST(IndexingTest, GpuCompactCapacityAndExactResultsAreStable)
{
    using Scalar = TypeParam;
    auto input = makeMatrix<Scalar>(5, 2, {
        Scalar(10), Scalar(11), Scalar(12), Scalar(13), Scalar(14),
        Scalar(20), Scalar(21), Scalar(22), Scalar(23), Scalar(24)
    }).toGpu();
    auto mask = makeMatrix<std::uint8_t>(5, 1, {0, 2, 0, 255, 1}).toGpu();
    auto capacity = makeMatrix<Scalar>(5, 2, {
        Scalar(-1), Scalar(-1), Scalar(-1), Scalar(-1), Scalar(-1),
        Scalar(-1), Scalar(-1), Scalar(-1), Scalar(-1), Scalar(-1)
    }).toGpu();
    auto sources = makeMatrix<Index>(5, 1, {-1, -1, -1, -1, -1}).toGpu();
    DenseMatrix<Index, Device::GPU> selected_count(1, 1);
    test::CudaStreamGuard stream;
    IndexingWorkspace workspace;

    compactRowsAsync(
        input, mask, capacity, sources, selected_count, workspace, stream.get());
    stream.synchronize();
    workspace.checkStatus("compactRowsAsync");
    expectGpuMatrix(selected_count, 1, 1, {Index(3)});
    expectGpuMatrix(sources, 5, 1, {Index(1), Index(3), Index(4), Index(-1), Index(-1)});
    expectGpuMatrix(capacity, 5, 2, {
        Scalar(11), Scalar(13), Scalar(14), Scalar(-1), Scalar(-1),
        Scalar(21), Scalar(23), Scalar(24), Scalar(-1), Scalar(-1)
    });

    auto exact = compactRows(input, mask);
    expectGpuMatrix(exact.values, 3, 2, {
        Scalar(11), Scalar(13), Scalar(14), Scalar(21), Scalar(23), Scalar(24)
    });
    expectGpuMatrix(exact.sourceIndices, 3, 1, {Index(1), Index(3), Index(4)});
    workspace.closeAsyncAllocation();
    stream.synchronize();
}

TYPED_TEST(IndexingTest, GpuCompactSupportsNoneAllZeroRowsAndZeroColumns)
{
    using Scalar = TypeParam;
    auto input = makeMatrix<Scalar>(3, 2, {
        Scalar(1), Scalar(2), Scalar(3), Scalar(4), Scalar(5), Scalar(6)
    }).toGpu();
    auto none_mask = makeMatrix<std::uint8_t>(3, 1, {0, 0, 0}).toGpu();
    auto all_mask = makeMatrix<std::uint8_t>(3, 1, {1, 1, 1}).toGpu();

    auto none = compactRows(input, none_mask);
    expectGpuMatrix(none.values, 0, 2, {});
    expectGpuMatrix(none.sourceIndices, 0, 1, {});
    auto all = compactRows(input, all_mask);
    expectGpuMatrix(all.values, 3, 2, {
        Scalar(1), Scalar(2), Scalar(3), Scalar(4), Scalar(5), Scalar(6)
    });
    expectGpuMatrix(all.sourceIndices, 3, 1, {Index(0), Index(1), Index(2)});

    DenseMatrix<Scalar, Device::CPU> empty_cpu(0, 3);
    DenseMatrix<std::uint8_t, Device::CPU> empty_mask_cpu(0, 1);
    auto empty = compactRows(empty_cpu.toGpu(), empty_mask_cpu.toGpu());
    expectGpuMatrix(empty.values, 0, 3, {});
    expectGpuMatrix(empty.sourceIndices, 0, 1, {});

    DenseMatrix<Scalar, Device::CPU> zero_cols_cpu(4, 0);
    auto zero_mask = makeMatrix<std::uint8_t>(4, 1, {0, 1, 0, 2}).toGpu();
    auto zero_cols = compactRows(zero_cols_cpu.toGpu(), zero_mask);
    expectGpuMatrix(zero_cols.values, 2, 0, {});
    expectGpuMatrix(zero_cols.sourceIndices, 2, 1, {Index(1), Index(3)});
}

TYPED_TEST(IndexingTest, GpuShapeMismatchesThrowBeforeLaunch)
{
    using Scalar = TypeParam;
    DenseMatrix<Index, Device::GPU> counts(2, 2);
    DenseMatrix<Index, Device::GPU> wrong_scan_output(4, 1);
    DenseMatrix<Scalar, Device::GPU> input(3, 2);
    DenseMatrix<Index, Device::GPU> wrong_indices(2, 2);
    DenseMatrix<Index, Device::GPU> valid_indices(2, 1);
    DenseMatrix<Scalar, Device::GPU> wrong_gather_output(3, 2);
    DenseMatrix<Scalar, Device::GPU> wrong_valid_gather_output(2, 1);
    DenseMatrix<Scalar, Device::GPU> wrong_values(2, 1);
    DenseMatrix<Scalar, Device::GPU> scatter_output(3, 2);
    DenseMatrix<std::uint8_t, Device::GPU> wrong_mask(2, 1);
    DenseMatrix<std::uint8_t, Device::GPU> valid_mask(3, 1);
    DenseMatrix<Scalar, Device::GPU> capacity_output(3, 2);
    DenseMatrix<Scalar, Device::GPU> wrong_capacity_output(2, 2);
    DenseMatrix<Index, Device::GPU> source_capacity(3, 1);
    DenseMatrix<Index, Device::GPU> wrong_source_capacity(3, 2);
    DenseMatrix<Index, Device::GPU> selected_count(1, 1);
    DenseMatrix<Index, Device::GPU> wrong_selected_count(1, 2);
    IndexingWorkspace workspace;

    EXPECT_THROW(
        exclusiveScan(counts, wrong_scan_output, workspace),
        std::invalid_argument);
    EXPECT_THROW(
        gatherRows(input, wrong_indices, wrong_gather_output, workspace),
        std::invalid_argument);
    EXPECT_THROW(
        gatherRows(input, valid_indices, wrong_valid_gather_output, workspace),
        std::invalid_argument);
    EXPECT_THROW(
        scatterRows(wrong_values, wrong_indices, scatter_output, workspace),
        std::invalid_argument);
    EXPECT_THROW(
        compactRows(input, wrong_mask, input, source_capacity, selected_count, workspace),
        std::invalid_argument);
    EXPECT_THROW(
        compactRows(
            input, valid_mask, wrong_capacity_output, source_capacity, selected_count, workspace),
        std::invalid_argument);
    EXPECT_THROW(
        compactRows(
            input, valid_mask, capacity_output, wrong_source_capacity, selected_count, workspace),
        std::invalid_argument);
    EXPECT_THROW(
        compactRows(
            input, valid_mask, capacity_output, source_capacity, wrong_selected_count, workspace),
        std::invalid_argument);
    EXPECT_EQ(workspace.capacityBytes(), 0U);
}

} // namespace
} // namespace plamatrix

#endif
