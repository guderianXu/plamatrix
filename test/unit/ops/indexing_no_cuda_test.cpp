#include "indexing_test_utils.h"

#ifndef PLAMATRIX_WITH_CUDA

namespace plamatrix
{
namespace
{

using indexing_test_detail::expectMatrix;
using indexing_test_detail::IndexingTest;
using indexing_test_detail::makeMatrix;
using indexing_test_detail::ScalarTypes;

TYPED_TEST_SUITE(IndexingTest, ScalarTypes);

template <typename Function>
void expectNoCudaIndexingError(const char* operation, Function&& function)
{
    try
    {
        function();
        FAIL() << "Expected CPU-only indexing failure";
    }
    catch (const std::runtime_error& error)
    {
        const std::string message = error.what();
        EXPECT_NE(message.find(operation), std::string::npos) << message;
        EXPECT_NE(message.find("PLAMATRIX_WITH_CUDA=ON"), std::string::npos) << message;
    }
}

TYPED_TEST(IndexingTest, NoCudaGpuEntryPointsAreExplicitRuntimeStubs)
{
    using Scalar = TypeParam;
    DenseMatrix<Index, Device::GPU> counts(2, 1);
    DenseMatrix<Index, Device::GPU> indices(2, 1);
    DenseMatrix<Index, Device::GPU> scan_output(2, 1);
    DenseMatrix<Index, Device::GPU> selected_count(1, 1);
    DenseMatrix<Scalar, Device::GPU> input(2, 1);
    DenseMatrix<Scalar, Device::GPU> output(2, 1);
    DenseMatrix<std::uint8_t, Device::GPU> mask(2, 1);
    IndexingWorkspace workspace;

    expectNoCudaIndexingError("exclusiveScan", [&] {
        static_cast<void>(exclusiveScan(counts));
    });
    expectNoCudaIndexingError("exclusiveScanAsync", [&] {
        exclusiveScanAsync(counts, scan_output, workspace, nullptr);
    });
    expectNoCudaIndexingError("gatherRows", [&] {
        gatherRows(input, indices, output, workspace);
    });
    expectNoCudaIndexingError("gatherRowsAsync", [&] {
        static_cast<void>(gatherRowsAsync(input, indices, workspace, nullptr));
    });
    expectNoCudaIndexingError("scatterRows", [&] {
        scatterRows(input, indices, output);
    });
    expectNoCudaIndexingError("scatterRowsAsync", [&] {
        scatterRowsAsync(input, indices, output, workspace, nullptr);
    });
    expectNoCudaIndexingError("compactRows", [&] {
        static_cast<void>(compactRows(input, mask));
    });
    expectNoCudaIndexingError("compactRowsAsync", [&] {
        compactRowsAsync(
            input, mask, output, indices, selected_count, workspace, nullptr);
    });
}

TEST(IndexingNoCudaTest, WorkspaceOperationsHaveClearRuntimeBehavior)
{
    IndexingWorkspace workspace;
    EXPECT_EQ(workspace.capacityBytes(), 0U);
    EXPECT_EQ(workspace.data(), nullptr);
    expectNoCudaIndexingError("IndexingWorkspace::reserveBytes", [&] {
        workspace.reserveBytes(64);
    });
    expectNoCudaIndexingError("IndexingWorkspace::reserveBytesAsync", [&] {
        workspace.reserveBytesAsync(64, nullptr);
    });
    expectNoCudaIndexingError("IndexingWorkspace::checkStatus", [&] {
        workspace.checkStatus("exclusiveScanAsync");
    });
    EXPECT_NO_THROW(workspace.closeAsyncAllocation());
}

} // namespace
} // namespace plamatrix

#endif
