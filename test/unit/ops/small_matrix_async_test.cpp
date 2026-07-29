#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <gtest/gtest.h>

#include "plamatrix/ops/small_matrix.h"
#include "../../support/cuda_test_utils.h"
#if defined(PLAMATRIX_WITH_CUDA) && defined(PLAMATRIX_SMALL_MATRIX_TEST_HOOKS)
#include "small_matrix_detail.h"
#endif

namespace plamatrix
{
namespace
{

#ifdef PLAMATRIX_WITH_CUDA

template <typename Scalar>
class SmallMatrixCudaAsyncTest : public ::testing::Test
{
protected:
    static DenseMatrix<Scalar, Device::CPU> makeBatch(Index rows)
    {
        DenseMatrix<Scalar, Device::CPU> compact(rows, 6);
        for (Index row = 0; row < rows; ++row)
        {
            compact(row, 0) = Scalar(3) + Scalar(0.01) * static_cast<Scalar>(row);
            compact(row, 1) = Scalar(0);
            compact(row, 2) = Scalar(0);
            compact(row, 3) = Scalar(1);
            compact(row, 4) = Scalar(0);
            compact(row, 5) = Scalar(2);
        }
        return compact;
    }
};

#if defined(PLAMATRIX_USE_FLOAT) && defined(PLAMATRIX_USE_DOUBLE)
using AsyncScalarTypes = ::testing::Types<float, double>;
#elif defined(PLAMATRIX_USE_FLOAT)
using AsyncScalarTypes = ::testing::Types<float>;
#elif defined(PLAMATRIX_USE_DOUBLE)
using AsyncScalarTypes = ::testing::Types<double>;
#endif
TYPED_TEST_SUITE(SmallMatrixCudaAsyncTest, AsyncScalarTypes);

TYPED_TEST(SmallMatrixCudaAsyncTest, CallerOwnedOutputsReuseStorageOnNonDefaultStream)
{
    using Scalar = TypeParam;
    auto compact = TestFixture::makeBatch(33);
    auto input = compact.toGpu();
    DenseMatrix<Scalar, Device::GPU> eigenvalues(33, 3);
    DenseMatrix<Scalar, Device::GPU> eigenvectors(33, 9);
    Scalar* const values_data = eigenvalues.data();
    Scalar* const vectors_data = eigenvectors.data();
    test::CudaStreamGuard stream;
    SymmetricEigh3x3Workspace workspace;

    symmetricEigh3x3Batched(input, eigenvalues, eigenvectors, workspace, stream.get());
    const std::size_t capacity = workspace.capacityBytes();
    void* const workspace_data = workspace.data();
    symmetricEigh3x3Batched(input, eigenvalues, eigenvectors, workspace, stream.get());

    EXPECT_EQ(eigenvalues.data(), values_data);
    EXPECT_EQ(eigenvectors.data(), vectors_data);
    EXPECT_EQ(workspace.capacityBytes(), capacity);
    EXPECT_EQ(workspace.data(), workspace_data);
    const auto host_values = eigenvalues.toCpu();
    const auto host_vectors = eigenvectors.toCpu();
    for (Index row = 0; row < compact.rows(); ++row)
    {
        EXPECT_EQ(host_values(row, 0), Scalar(1));
        EXPECT_EQ(host_values(row, 1), Scalar(2));
        EXPECT_EQ(host_values(row, 2), compact(row, 0));
        const Scalar expected[9] = {
            Scalar(0), Scalar(1), Scalar(0),
            Scalar(0), Scalar(0), Scalar(1),
            Scalar(1), Scalar(0), Scalar(0)
        };
        for (Index col = 0; col < 9; ++col)
        {
            EXPECT_EQ(host_vectors(row, col), expected[col]);
        }
    }
}

TYPED_TEST(SmallMatrixCudaAsyncTest, EmptyBatchAndShapeValidationAreSynchronous)
{
    using Scalar = TypeParam;
    DenseMatrix<Scalar, Device::CPU> empty_cpu(0, 6);
    auto empty_gpu = empty_cpu.toGpu();
    auto result = symmetricEigh3x3Batched(empty_gpu);
    EXPECT_EQ(result.eigenvalues.rows(), 0);
    EXPECT_EQ(result.eigenvalues.cols(), 3);
    EXPECT_EQ(result.eigenvectors.rows(), 0);
    EXPECT_EQ(result.eigenvectors.cols(), 9);

    DenseMatrix<Scalar, Device::GPU> values(0, 3);
    DenseMatrix<Scalar, Device::GPU> vectors(0, 9);
    SymmetricEigh3x3Workspace workspace;
    EXPECT_NO_THROW(symmetricEigh3x3BatchedAsync(
        empty_gpu, values, vectors, workspace, nullptr));
    workspace.checkStatus("empty");
    EXPECT_EQ(workspace.capacityBytes(), 0U);

    DenseMatrix<Scalar, Device::GPU> wrong_input(2, 5);
    DenseMatrix<Scalar, Device::GPU> good_input(2, 6);
    DenseMatrix<Scalar, Device::GPU> good_values(2, 3);
    DenseMatrix<Scalar, Device::GPU> good_vectors(2, 9);
    DenseMatrix<Scalar, Device::GPU> wrong_values(2, 2);
    DenseMatrix<Scalar, Device::GPU> wrong_vectors(2, 8);
    EXPECT_THROW(symmetricEigh3x3BatchedAsync(
        wrong_input, good_values, good_vectors, workspace, nullptr), std::invalid_argument);
    EXPECT_THROW(symmetricEigh3x3BatchedAsync(
        good_input, wrong_values, good_vectors, workspace, nullptr), std::invalid_argument);
    EXPECT_THROW(symmetricEigh3x3BatchedAsync(
        good_input, good_values, wrong_vectors, workspace, nullptr), std::invalid_argument);
    EXPECT_EQ(workspace.capacityBytes(), 0U);
}

TYPED_TEST(SmallMatrixCudaAsyncTest, AsyncNonFiniteRowsAreZeroedAndLowestRowIsReported)
{
    using Scalar = TypeParam;
    auto compact = TestFixture::makeBatch(8);
    compact(6, 0) = std::numeric_limits<Scalar>::quiet_NaN();
    compact(2, 4) = std::numeric_limits<Scalar>::infinity();
    compact(4, 5) = -std::numeric_limits<Scalar>::infinity();
    auto input = compact.toGpu();
    DenseMatrix<Scalar, Device::GPU> values(8, 3);
    DenseMatrix<Scalar, Device::GPU> vectors(8, 9);
    test::CudaStreamGuard stream;
    SymmetricEigh3x3Workspace workspace;

    symmetricEigh3x3BatchedAsync(input, values, vectors, workspace, stream.get());
    stream.synchronize();
    try
    {
        workspace.checkStatus("symmetricEigh3x3BatchedAsync");
        FAIL() << "Expected a non-finite input error";
    }
    catch (const std::invalid_argument& error)
    {
        EXPECT_NE(std::string(error.what()).find("row 2"), std::string::npos) << error.what();
    }

    const auto host_values = values.toCpu();
    const auto host_vectors = vectors.toCpu();
    for (Index row : {Index(2), Index(4), Index(6)})
    {
        for (Index col = 0; col < 3; ++col)
        {
            EXPECT_EQ(host_values(row, col), Scalar(0));
        }
        for (Index col = 0; col < 9; ++col)
        {
            EXPECT_EQ(host_vectors(row, col), Scalar(0));
        }
    }
    workspace.closeAsyncAllocation();
    stream.synchronize();
}

TYPED_TEST(SmallMatrixCudaAsyncTest, SynchronousWrapperRejectsLowestNonFiniteRow)
{
    using Scalar = TypeParam;
    auto compact = TestFixture::makeBatch(5);
    compact(3, 1) = std::numeric_limits<Scalar>::quiet_NaN();
    compact(1, 5) = std::numeric_limits<Scalar>::infinity();
    try
    {
        static_cast<void>(symmetricEigh3x3Batched(compact.toGpu()));
        FAIL() << "Expected a non-finite input error";
    }
    catch (const std::invalid_argument& error)
    {
        EXPECT_NE(std::string(error.what()).find("row 1"), std::string::npos) << error.what();
    }
}

#ifdef PLAMATRIX_SMALL_MATRIX_TEST_HOOKS
TEST(SmallMatrixCudaAsyncTest, BasisFailureIsZeroedAndReportedThroughWorkspaceStatus)
{
    DenseMatrix<float, Device::CPU> compact(2, 6);
    compact.fill(0.0F);
    auto input = compact.toGpu();
    DenseMatrix<float, Device::GPU> values(2, 3);
    DenseMatrix<float, Device::GPU> vectors(2, 9);
    test::CudaStreamGuard stream;
    SymmetricEigh3x3Workspace workspace;

    small_matrix_detail::setForcedBasisFailureRow(1);
    symmetricEigh3x3BatchedAsync(input, values, vectors, workspace, stream.get());
    small_matrix_detail::setForcedBasisFailureRow(-1);
    stream.synchronize();

    try
    {
        workspace.checkStatus("forced basis failure");
        FAIL() << "Expected a repeated-eigenvalue basis failure";
    }
    catch (const std::runtime_error& error)
    {
        EXPECT_NE(std::string(error.what()).find("basis"), std::string::npos) << error.what();
        EXPECT_NE(std::string(error.what()).find("row 1"), std::string::npos) << error.what();
    }
    const auto host_values = values.toCpu();
    const auto host_vectors = vectors.toCpu();
    for (Index col = 0; col < 3; ++col)
    {
        EXPECT_EQ(host_values(1, col), 0.0F);
    }
    for (Index col = 0; col < 9; ++col)
    {
        EXPECT_EQ(host_vectors(1, col), 0.0F);
    }
    workspace.closeAsyncAllocation();
    stream.synchronize();
}
#endif

#else

TEST(SmallMatrixNoCudaTest, GpuApisAndWorkspaceHaveExplicitRuntimeStubs)
{
    DenseMatrix<float, Device::GPU> input;
    DenseMatrix<float, Device::GPU> values;
    DenseMatrix<float, Device::GPU> vectors;
    SymmetricEigh3x3Workspace workspace;
    auto expect_stub = [](auto&& callable) {
        try
        {
            callable();
            FAIL() << "Expected a CUDA-disabled runtime error";
        }
        catch (const std::runtime_error& error)
        {
            const std::string message = error.what();
            EXPECT_NE(message.find("PLAMATRIX_WITH_CUDA=ON"), std::string::npos) << message;
        }
    };

    expect_stub([&] { static_cast<void>(symmetricEigh3x3Batched(input)); });
    expect_stub([&] {
        symmetricEigh3x3Batched(input, values, vectors, workspace, nullptr);
    });
    expect_stub([&] {
        symmetricEigh3x3BatchedAsync(input, values, vectors, workspace, nullptr);
    });
    expect_stub([&] { workspace.reserveBytes(64); });
    expect_stub([&] { workspace.reserveBytesAsync(64, nullptr); });
    expect_stub([&] { workspace.checkStatus("cpu-only"); });
    EXPECT_NO_THROW(workspace.closeAsyncAllocation());
}

static_assert(!std::is_copy_constructible_v<SymmetricEigh3x3Workspace>);
static_assert(!std::is_copy_assignable_v<SymmetricEigh3x3Workspace>);
static_assert(std::is_nothrow_move_constructible_v<SymmetricEigh3x3Workspace>);
static_assert(std::is_nothrow_move_assignable_v<SymmetricEigh3x3Workspace>);

#endif

} // namespace
} // namespace plamatrix
