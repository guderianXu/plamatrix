#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>

#include "plamatrix/ops/small_matrix.h"
#include "../../support/cuda_test_utils.h"

#ifdef PLAMATRIX_WITH_CUDA

namespace plamatrix
{
namespace
{

template <typename Callable>
void expectLogicErrorContaining(Callable&& callable, const char* expected)
{
    try
    {
        callable();
        FAIL() << "Expected std::logic_error";
    }
    catch (const std::logic_error& error)
    {
        EXPECT_NE(std::string(error.what()).find(expected), std::string::npos) << error.what();
    }
}

DenseMatrix<float, Device::GPU> finiteInput()
{
    DenseMatrix<float, Device::CPU> input(4, 6);
    for (Index row = 0; row < input.rows(); ++row)
    {
        input(row, 0) = 1.0F + static_cast<float>(row);
        input(row, 1) = 0.25F;
        input(row, 2) = -0.5F;
        input(row, 3) = 2.0F + static_cast<float>(row);
        input(row, 4) = 0.75F;
        input(row, 5) = 3.0F + static_cast<float>(row);
    }
    return input.toGpu();
}

TEST(SmallMatrixWorkspaceCudaTest, StatusBlocksResetAndExplicitResetAllowsCrossStreamReuse)
{
    auto input = finiteInput();
    DenseMatrix<float, Device::GPU> values(4, 3);
    DenseMatrix<float, Device::GPU> vectors(4, 9);
    test::CudaStreamGuard first_stream;
    test::CudaStreamGuard second_stream;
    SymmetricEigh3x3Workspace workspace;
    workspace.reserveBytes(1024);
    const std::size_t capacity = workspace.capacityBytes();
    void* const data = workspace.data();

    symmetricEigh3x3BatchedAsync(input, values, vectors, workspace, first_stream.get());

    expectLogicErrorContaining([&] {
        symmetricEigh3x3BatchedAsync(
            input, values, vectors, workspace, second_stream.get());
    }, "different stream");

    first_stream.synchronize();
    expectLogicErrorContaining([&] { workspace.reserveBytes(capacity); }, "checkStatus");
    workspace.checkStatus("first stream");
    workspace.reserveBytes(capacity);
    EXPECT_EQ(workspace.capacityBytes(), capacity);
    EXPECT_EQ(workspace.data(), data);

    symmetricEigh3x3BatchedAsync(input, values, vectors, workspace, second_stream.get());
    second_stream.synchronize();
    workspace.checkStatus("second stream");
}

TEST(SmallMatrixWorkspaceCudaTest, AsyncStatusMustBeConsumedBeforeClose)
{
    DenseMatrix<float, Device::CPU> input_cpu(2, 6);
    input_cpu.fill(0.0F);
    input_cpu(1, 2) = std::numeric_limits<float>::quiet_NaN();
    auto input = input_cpu.toGpu();
    DenseMatrix<float, Device::GPU> values(2, 3);
    DenseMatrix<float, Device::GPU> vectors(2, 9);
    test::CudaStreamGuard stream;
    SymmetricEigh3x3Workspace workspace;

    symmetricEigh3x3BatchedAsync(input, values, vectors, workspace, stream.get());
    stream.synchronize();
    expectLogicErrorContaining([&] { workspace.closeAsyncAllocation(); }, "checkStatus");
    EXPECT_THROW(workspace.checkStatus("async invalid"), std::invalid_argument);
    EXPECT_NO_THROW(workspace.closeAsyncAllocation());
    EXPECT_EQ(workspace.capacityBytes(), 0U);
    EXPECT_EQ(workspace.data(), nullptr);
    stream.synchronize();
}

TEST(SmallMatrixWorkspaceCudaTest, MoveTransfersAllocationStatusAndStreamOwnership)
{
    auto input = finiteInput();
    DenseMatrix<float, Device::GPU> values(4, 3);
    DenseMatrix<float, Device::GPU> vectors(4, 9);
    test::CudaStreamGuard stream;
    test::CudaStreamGuard other_stream;
    SymmetricEigh3x3Workspace source;

    symmetricEigh3x3BatchedAsync(input, values, vectors, source, stream.get());
    const std::size_t capacity = source.capacityBytes();
    void* const data = source.data();
    SymmetricEigh3x3Workspace moved(std::move(source));

    EXPECT_EQ(source.capacityBytes(), 0U);
    EXPECT_EQ(source.data(), nullptr);
    EXPECT_EQ(moved.capacityBytes(), capacity);
    EXPECT_EQ(moved.data(), data);
    expectLogicErrorContaining([&] {
        symmetricEigh3x3BatchedAsync(
            input, values, vectors, moved, other_stream.get());
    }, "different stream");

    stream.synchronize();
    moved.checkStatus("moved");
    moved.closeAsyncAllocation();
    stream.synchronize();
}

TEST(SmallMatrixWorkspaceCudaTest, MoveAssignmentReleasesDestinationAndPreservesSourceState)
{
    test::CudaStreamGuard source_stream;
    test::CudaStreamGuard destination_stream;
    SymmetricEigh3x3Workspace source;
    SymmetricEigh3x3Workspace destination;
    source.reserveBytesAsync(64, source_stream.get());
    destination.reserveBytesAsync(128, destination_stream.get());
    void* const source_data = source.data();

    destination = std::move(source);

    EXPECT_EQ(source.data(), nullptr);
    EXPECT_EQ(source.capacityBytes(), 0U);
    EXPECT_EQ(destination.data(), source_data);
    EXPECT_EQ(destination.capacityBytes(), 64U);
    expectLogicErrorContaining([&] {
        destination.reserveBytesAsync(64, destination_stream.get());
    }, "different stream");
    destination.closeAsyncAllocation();
    source_stream.synchronize();
    destination_stream.synchronize();
}

TEST(SmallMatrixWorkspaceCudaTest, MoveAssignmentRejectsUnconsumedDestinationStatus)
{
    DenseMatrix<float, Device::CPU> invalid_cpu(4, 6);
    invalid_cpu.fill(0.0F);
    invalid_cpu(2, 1) = std::numeric_limits<float>::quiet_NaN();
    auto invalid = invalid_cpu.toGpu();
    DenseMatrix<float, Device::GPU> values(4, 3);
    DenseMatrix<float, Device::GPU> vectors(4, 9);
    test::CudaStreamGuard destination_stream;
    test::CudaStreamGuard source_stream;
    SymmetricEigh3x3Workspace destination;
    SymmetricEigh3x3Workspace source;
    symmetricEigh3x3BatchedAsync(
        invalid, values, vectors, destination, destination_stream.get());
    source.reserveBytesAsync(128, source_stream.get());
    destination_stream.synchronize();
    void* const destination_data = destination.data();
    const std::size_t destination_capacity = destination.capacityBytes();
    void* const source_data = source.data();

    try
    {
        destination = std::move(source);
        FAIL() << "Expected move assignment to reject an unconsumed destination status";
        return;
    }
    catch (const std::logic_error& error)
    {
        EXPECT_NE(std::string(error.what()).find("checkStatus"), std::string::npos)
            << error.what();
    }

    EXPECT_EQ(destination.data(), destination_data);
    EXPECT_EQ(destination.capacityBytes(), destination_capacity);
    EXPECT_EQ(source.data(), source_data);
    try
    {
        destination.checkStatus("move destination");
        FAIL() << "Expected the original destination status";
    }
    catch (const std::invalid_argument& error)
    {
        EXPECT_NE(std::string(error.what()).find("row 2"), std::string::npos) << error.what();
    }

    destination = std::move(source);
    EXPECT_EQ(destination.data(), source_data);
    EXPECT_EQ(source.data(), nullptr);
    destination.closeAsyncAllocation();
    source_stream.synchronize();
}

static_assert(!std::is_copy_constructible_v<SymmetricEigh3x3Workspace>);
static_assert(!std::is_copy_assignable_v<SymmetricEigh3x3Workspace>);
static_assert(std::is_nothrow_move_constructible_v<SymmetricEigh3x3Workspace>);
static_assert(!std::is_nothrow_move_assignable_v<SymmetricEigh3x3Workspace>);

} // namespace
} // namespace plamatrix

#endif
