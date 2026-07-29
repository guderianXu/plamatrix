#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>

#include <plamatrix/sparse/csr_matrix.h>

using namespace plamatrix;

TEST(CSRMatrix, constructionAndAccess)
{
    CSRMatrix<float, Device::CPU> mat(4, 4, 5);
    EXPECT_EQ(mat.rows(), 4);
    EXPECT_EQ(mat.cols(), 4);
    EXPECT_EQ(mat.nnz(), 5);
    EXPECT_NE(mat.values(), nullptr);
    EXPECT_NE(mat.colIndices(), nullptr);
    EXPECT_NE(mat.rowOffsets(), nullptr);

    // row_offsets should be zero-initialized
    for (Index r = 0; r <= 4; ++r)
    {
        EXPECT_EQ(mat.rowOffsets()[r], 0);
    }
}

TEST(CSRMatrix, construction_RejectsNegativeDimensionsAndNnz)
{
    EXPECT_THROW((CSRMatrix<float, Device::CPU>(-1, 4, 0)), std::invalid_argument);
    EXPECT_THROW((CSRMatrix<float, Device::CPU>(4, -1, 0)), std::invalid_argument);
    EXPECT_THROW((CSRMatrix<float, Device::CPU>(4, 4, -1)), std::invalid_argument);
}

TEST(CSRMatrix, construction_RejectsNonZerosWithZeroDimensions)
{
    EXPECT_THROW((CSRMatrix<float, Device::CPU>(0, 4, 1)), std::invalid_argument);
    EXPECT_THROW((CSRMatrix<float, Device::CPU>(4, 0, 1)), std::invalid_argument);
}

namespace
{
template <typename Scalar>
CSRMatrix<Scalar, Device::CPU> makeTransferInput()
{
    CSRMatrix<Scalar, Device::CPU> cpu(3, 4, 4);
    const std::array<Scalar, 4> values = {
        Scalar(1.25), Scalar(-2.5), Scalar(3.75), Scalar(4.5)};
    const std::array<Index, 4> columns = {0, 2, 1, 3};
    const std::array<Index, 4> row_offsets = {0, 2, 3, 4};
    for (Index i = 0; i < cpu.nnz(); ++i)
    {
        const auto offset = static_cast<std::size_t>(i);
        cpu.values()[i] = values[offset];
        cpu.colIndices()[i] = columns[offset];
    }
    for (Index i = 0; i <= cpu.rows(); ++i)
    {
        cpu.rowOffsets()[i] = row_offsets[static_cast<std::size_t>(i)];
    }
    return cpu;
}

template <typename Scalar>
CSRMatrix<Scalar, Device::CPU> makePinnedTransferInput()
{
    const auto input = makeTransferInput<Scalar>();
    auto pinned = CSRMatrix<Scalar, Device::CPU>::pinned(
        input.rows(), input.cols(), input.nnz());
    std::copy_n(input.values(), static_cast<std::size_t>(input.nnz()), pinned.values());
    std::copy_n(
        input.colIndices(), static_cast<std::size_t>(input.nnz()), pinned.colIndices());
    std::copy_n(
        input.rowOffsets(), static_cast<std::size_t>(input.rows()) + 1, pinned.rowOffsets());
    return pinned;
}

template <typename Scalar>
void expectTransferInput(const CSRMatrix<Scalar, Device::CPU>& actual)
{
    const std::array<Scalar, 4> values = {
        Scalar(1.25), Scalar(-2.5), Scalar(3.75), Scalar(4.5)};
    const std::array<Index, 4> columns = {0, 2, 1, 3};
    const std::array<Index, 4> row_offsets = {0, 2, 3, 4};
    ASSERT_EQ(actual.rows(), 3);
    ASSERT_EQ(actual.cols(), 4);
    ASSERT_EQ(actual.nnz(), 4);
    for (Index i = 0; i < actual.nnz(); ++i)
    {
        const auto offset = static_cast<std::size_t>(i);
        EXPECT_EQ(actual.values()[i], values[offset]);
        EXPECT_EQ(actual.colIndices()[i], columns[offset]);
    }
    for (Index i = 0; i <= actual.rows(); ++i)
    {
        EXPECT_EQ(actual.rowOffsets()[i], row_offsets[static_cast<std::size_t>(i)]);
    }
}

#ifndef PLAMATRIX_WITH_CUDA
template <typename Fn>
void expectRequiresCuda(const char* operation, Fn fn)
{
    try
    {
        fn();
        FAIL() << operation << " should reject CPU-only builds";
    }
    catch (const std::runtime_error& error)
    {
        const std::string message = error.what();
        EXPECT_NE(message.find(operation), std::string::npos);
        EXPECT_NE(message.find("PLAMATRIX_WITH_CUDA=ON"), std::string::npos);
    }
}
#endif

template <typename Matrix, typename = void>
struct CanSpoofPinned : std::false_type
{
};

template <typename Matrix>
struct CanSpoofPinned<
    Matrix,
    std::void_t<decltype(Matrix::template pinned<Device::CPU>(Index{1}, Index{1}, Index{0}))>>
    : std::true_type
{
};

template <typename Matrix, typename = void>
struct CanSpoofAsyncAllocation : std::false_type
{
};

template <typename Matrix>
struct CanSpoofAsyncAllocation<
    Matrix,
    std::void_t<decltype(Matrix::template uninitializedAsync<Device::GPU>(
        Index{1}, Index{1}, Index{0}, nullptr))>> : std::true_type
{
};

template <typename Matrix, typename = void>
struct CanSpoofAsyncClose : std::false_type
{
};

template <typename Matrix>
struct CanSpoofAsyncClose<
    Matrix,
    std::void_t<decltype(
        std::declval<Matrix&>().template closeAsyncAllocation<Device::GPU>())>>
    : std::true_type
{
};

static_assert(!CanSpoofPinned<CSRMatrix<float, Device::GPU>>::value);
static_assert(!CanSpoofAsyncAllocation<CSRMatrix<float, Device::CPU>>::value);
static_assert(!CanSpoofAsyncClose<CSRMatrix<float, Device::CPU>>::value);
} // namespace

#ifdef PLAMATRIX_WITH_CUDA
#if defined(PLAMATRIX_USE_FLOAT) && defined(PLAMATRIX_USE_DOUBLE)
using CsrTransferScalars = ::testing::Types<float, double>;
#elif defined(PLAMATRIX_USE_FLOAT)
using CsrTransferScalars = ::testing::Types<float>;
#elif defined(PLAMATRIX_USE_DOUBLE)
using CsrTransferScalars = ::testing::Types<double>;
#else
#error "CSR transfer tests require an enabled scalar type"
#endif

template <typename Scalar>
class CSRMatrixTransferTest : public ::testing::Test
{
};

TYPED_TEST_SUITE(CSRMatrixTransferTest, CsrTransferScalars);

TYPED_TEST(CSRMatrixTransferTest, synchronousRoundTripPreservesCsrData)
{
    const auto cpu = makeTransferInput<TypeParam>();
    const auto back = cpu.toGpu().toCpu();
    expectTransferInput(back);
}

TYPED_TEST(CSRMatrixTransferTest, asynchronousRoundTripUsesNonDefaultStream)
{
    const auto cpu = makePinnedTransferInput<TypeParam>();
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);

    auto gpu = cpu.toGpuAsync(stream);
    auto back = gpu.toCpuAsync(stream);
    EXPECT_TRUE(gpu.isAsyncAllocation());
    EXPECT_TRUE(back.isPinnedHost());
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    gpu.closeAsyncAllocation();
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);

    expectTransferInput(back);
}

TYPED_TEST(CSRMatrixTransferTest, asynchronousCopiesPreserveExistingOutputPointers)
{
    const auto cpu = makePinnedTransferInput<TypeParam>();
    CSRMatrix<TypeParam, Device::GPU> gpu(cpu.rows(), cpu.cols(), cpu.nnz());
    auto back = CSRMatrix<TypeParam, Device::CPU>::pinned(
        cpu.rows(), cpu.cols(), cpu.nnz());
    TypeParam* gpu_values = gpu.values();
    Index* gpu_columns = gpu.colIndices();
    Index* gpu_offsets = gpu.rowOffsets();
    TypeParam* cpu_values = back.values();
    Index* cpu_columns = back.colIndices();
    Index* cpu_offsets = back.rowOffsets();
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);

    cpu.copyToGpuAsync(gpu, stream);
    gpu.copyToCpuAsync(back, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);

    EXPECT_EQ(gpu.values(), gpu_values);
    EXPECT_EQ(gpu.colIndices(), gpu_columns);
    EXPECT_EQ(gpu.rowOffsets(), gpu_offsets);
    EXPECT_EQ(back.values(), cpu_values);
    EXPECT_EQ(back.colIndices(), cpu_columns);
    EXPECT_EQ(back.rowOffsets(), cpu_offsets);
    expectTransferInput(back);
}

TYPED_TEST(CSRMatrixTransferTest, reusedPinnedDownloadIsRevalidatedAfterAsyncOverwrite)
{
    const auto valid_cpu = makeTransferInput<TypeParam>();
    auto valid_gpu = valid_cpu.toGpu();
    auto invalid_gpu = valid_cpu.toGpu();
    const Index invalid_offsets[] = {0, 3, 2, 4};
    ASSERT_EQ(cudaMemcpy(
        invalid_gpu.rowOffsets(), invalid_offsets, sizeof(invalid_offsets),
        cudaMemcpyHostToDevice), cudaSuccess);
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    auto reused = valid_gpu.toCpuAsync(stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    ASSERT_NO_THROW(reused.validateStructure(stream));
    ASSERT_TRUE(reused.hasValidatedStructure());

    invalid_gpu.copyToCpuAsync(reused, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    EXPECT_THROW(reused.validateStructure(stream), std::invalid_argument);
    EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TYPED_TEST(CSRMatrixTransferTest, asynchronousHostToDeviceCopyRejectsPageableInput)
{
    const auto pageable = makeTransferInput<TypeParam>();
    CSRMatrix<TypeParam, Device::GPU> gpu(
        pageable.rows(), pageable.cols(), pageable.nnz());

    EXPECT_THROW(pageable.copyToGpuAsync(gpu), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(pageable.toGpuAsync()), std::invalid_argument);
}

TYPED_TEST(CSRMatrixTransferTest, asynchronousDeviceToHostCopyRejectsPageableOutput)
{
    const auto cpu = makeTransferInput<TypeParam>();
    const auto gpu = cpu.toGpu();
    CSRMatrix<TypeParam, Device::CPU> pageable(cpu.rows(), cpu.cols(), cpu.nnz());

    EXPECT_THROW(gpu.copyToCpuAsync(pageable), std::invalid_argument);
}

TYPED_TEST(CSRMatrixTransferTest, closeAsyncAllocationRejectsOrdinaryGpuStorage)
{
    CSRMatrix<TypeParam, Device::GPU> ordinary(2, 2, 1);

    EXPECT_FALSE(ordinary.isAsyncAllocation());
    EXPECT_THROW(ordinary.closeAsyncAllocation(), std::logic_error);
}

TYPED_TEST(CSRMatrixTransferTest, streamOrderedStorageRejectsCrossStreamCopies)
{
    const auto cpu = makePinnedTransferInput<TypeParam>();
    cudaStream_t owner_stream = nullptr;
    cudaStream_t other_stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&owner_stream, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaStreamCreateWithFlags(&other_stream, cudaStreamNonBlocking), cudaSuccess);

    auto gpu = cpu.toGpuAsync(owner_stream);
    EXPECT_THROW(static_cast<void>(gpu.toCpuAsync(other_stream)), std::logic_error);
    EXPECT_THROW(cpu.copyToGpuAsync(gpu, other_stream), std::logic_error);

    ASSERT_EQ(cudaStreamSynchronize(owner_stream), cudaSuccess);
    gpu.closeAsyncAllocation();
    ASSERT_EQ(cudaStreamSynchronize(owner_stream), cudaSuccess);
    EXPECT_EQ(cudaStreamDestroy(other_stream), cudaSuccess);
    EXPECT_EQ(cudaStreamDestroy(owner_stream), cudaSuccess);
}

TYPED_TEST(CSRMatrixTransferTest, ordinaryGpuStorageRejectsOverlappingCrossStreamWrites)
{
    const auto cpu = makePinnedTransferInput<TypeParam>();
    CSRMatrix<TypeParam, Device::GPU> gpu(cpu.rows(), cpu.cols(), cpu.nnz());
    cudaStream_t first_stream = nullptr;
    cudaStream_t second_stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&first_stream, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaStreamCreateWithFlags(&second_stream, cudaStreamNonBlocking), cudaSuccess);

    cpu.copyToGpuAsync(gpu, first_stream);
    EXPECT_THROW(cpu.copyToGpuAsync(gpu, second_stream), std::logic_error);

    ASSERT_EQ(cudaStreamSynchronize(first_stream), cudaSuccess);
    EXPECT_NO_THROW(gpu.validateStructure(first_stream));
    EXPECT_NO_THROW(cpu.copyToGpuAsync(gpu, second_stream));
    ASSERT_EQ(cudaStreamSynchronize(second_stream), cudaSuccess);
    EXPECT_EQ(cudaStreamDestroy(second_stream), cudaSuccess);
    EXPECT_EQ(cudaStreamDestroy(first_stream), cudaSuccess);
}

TYPED_TEST(CSRMatrixTransferTest, deviceDownloadRejectsPendingSourceFromAnotherStream)
{
    const auto cpu = makePinnedTransferInput<TypeParam>();
    CSRMatrix<TypeParam, Device::GPU> gpu(cpu.rows(), cpu.cols(), cpu.nnz());
    auto output = CSRMatrix<TypeParam, Device::CPU>::pinned(
        cpu.rows(), cpu.cols(), cpu.nnz());
    cudaStream_t first_stream = nullptr;
    cudaStream_t second_stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&first_stream, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaStreamCreateWithFlags(&second_stream, cudaStreamNonBlocking), cudaSuccess);

    cpu.copyToGpuAsync(gpu, first_stream);
    EXPECT_THROW(gpu.copyToCpuAsync(output, second_stream), std::logic_error);

    ASSERT_EQ(cudaStreamSynchronize(first_stream), cudaSuccess);
    EXPECT_NO_THROW(gpu.validateStructure(first_stream));
    EXPECT_NO_THROW(gpu.copyToCpuAsync(output, second_stream));
    ASSERT_EQ(cudaStreamSynchronize(second_stream), cudaSuccess);
    EXPECT_NO_THROW(output.validateStructure(second_stream));
    EXPECT_EQ(cudaStreamDestroy(second_stream), cudaSuccess);
    EXPECT_EQ(cudaStreamDestroy(first_stream), cudaSuccess);
}

TYPED_TEST(CSRMatrixTransferTest, pinnedDownloadRejectsOverlappingCrossStreamWrites)
{
    const auto cpu = makeTransferInput<TypeParam>();
    const auto gpu = cpu.toGpu();
    auto output = CSRMatrix<TypeParam, Device::CPU>::pinned(
        cpu.rows(), cpu.cols(), cpu.nnz());
    cudaStream_t first_stream = nullptr;
    cudaStream_t second_stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&first_stream, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaStreamCreateWithFlags(&second_stream, cudaStreamNonBlocking), cudaSuccess);

    gpu.copyToCpuAsync(output, first_stream);
    EXPECT_THROW(gpu.copyToCpuAsync(output, second_stream), std::logic_error);

    ASSERT_EQ(cudaStreamSynchronize(first_stream), cudaSuccess);
    EXPECT_NO_THROW(output.validateStructure(first_stream));
    EXPECT_NO_THROW(gpu.copyToCpuAsync(output, second_stream));
    ASSERT_EQ(cudaStreamSynchronize(second_stream), cudaSuccess);
    EXPECT_EQ(cudaStreamDestroy(second_stream), cudaSuccess);
    EXPECT_EQ(cudaStreamDestroy(first_stream), cudaSuccess);
}

TYPED_TEST(CSRMatrixTransferTest, synchronousDownloadRejectsStreamOrderedStorage)
{
    const auto cpu = makePinnedTransferInput<TypeParam>();
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    auto gpu = cpu.toGpuAsync(stream);

    EXPECT_THROW(static_cast<void>(gpu.toCpu()), std::logic_error);

    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    gpu.closeAsyncAllocation();
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TYPED_TEST(CSRMatrixTransferTest, asyncOwnershipMovesAndRepeatedCloseIsHarmless)
{
    const auto cpu = makePinnedTransferInput<TypeParam>();
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);

    auto source = cpu.toGpuAsync(stream);
    TypeParam* values = source.values();
    auto moved = std::move(source);

    EXPECT_FALSE(source.isAsyncAllocation());
    EXPECT_TRUE(moved.isAsyncAllocation());
    EXPECT_EQ(moved.values(), values);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    EXPECT_NO_THROW(moved.closeAsyncAllocation());
    EXPECT_NO_THROW(moved.closeAsyncAllocation());
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TYPED_TEST(CSRMatrixTransferTest, asyncMoveAssignmentReleasesExistingTarget)
{
    const auto cpu = makePinnedTransferInput<TypeParam>();
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    auto source = cpu.toGpuAsync(stream);
    auto target = cpu.toGpuAsync(stream);
    TypeParam* source_values = source.values();

    target = std::move(source);

    EXPECT_FALSE(source.isAsyncAllocation());
    EXPECT_TRUE(target.isAsyncAllocation());
    EXPECT_EQ(target.values(), source_values);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    target.closeAsyncAllocation();
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TYPED_TEST(CSRMatrixTransferTest, destructorFallsBackAfterOwningStreamWasDestroyed)
{
    const auto cpu = makePinnedTransferInput<TypeParam>();
    int device = 0;
    cudaMemPool_t pool = nullptr;
    std::uint64_t used_before = 0;
    std::uint64_t used_during = 0;
    std::uint64_t used_after = 0;
    ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
    ASSERT_EQ(cudaDeviceGetDefaultMemPool(&pool, device), cudaSuccess);
    ASSERT_EQ(cudaMemPoolGetAttribute(
                  pool, cudaMemPoolAttrUsedMemCurrent, &used_before),
              cudaSuccess);
    {
        cudaStream_t stream = nullptr;
        ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
        auto gpu = cpu.toGpuAsync(stream);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        ASSERT_EQ(cudaMemPoolGetAttribute(
                      pool, cudaMemPoolAttrUsedMemCurrent, &used_during),
                  cudaSuccess);
        EXPECT_GT(used_during, used_before);
        ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    }

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemPoolGetAttribute(
                  pool, cudaMemPoolAttrUsedMemCurrent, &used_after),
              cudaSuccess);
    EXPECT_EQ(used_after, used_before);
}

TYPED_TEST(CSRMatrixTransferTest, asynchronousCopiesRejectDimensionAndNnzMismatch)
{
    const auto cpu = makeTransferInput<TypeParam>();
    const auto gpu = cpu.toGpu();
    CSRMatrix<TypeParam, Device::GPU> wrong_gpu_rows(2, 4, 4);
    CSRMatrix<TypeParam, Device::GPU> wrong_gpu_cols(3, 5, 4);
    CSRMatrix<TypeParam, Device::GPU> wrong_gpu_nnz(3, 4, 3);
    CSRMatrix<TypeParam, Device::CPU> wrong_cpu_rows(2, 4, 4);
    CSRMatrix<TypeParam, Device::CPU> wrong_cpu_cols(3, 5, 4);
    CSRMatrix<TypeParam, Device::CPU> wrong_cpu_nnz(3, 4, 3);

    EXPECT_THROW(cpu.copyToGpuAsync(wrong_gpu_rows), std::runtime_error);
    EXPECT_THROW(cpu.copyToGpuAsync(wrong_gpu_cols), std::runtime_error);
    EXPECT_THROW(cpu.copyToGpuAsync(wrong_gpu_nnz), std::runtime_error);
    EXPECT_THROW(gpu.copyToCpuAsync(wrong_cpu_rows), std::runtime_error);
    EXPECT_THROW(gpu.copyToCpuAsync(wrong_cpu_cols), std::runtime_error);
    EXPECT_THROW(gpu.copyToCpuAsync(wrong_cpu_nnz), std::runtime_error);
}

TYPED_TEST(CSRMatrixTransferTest, zeroRowMatrixRoundTrips)
{
    CSRMatrix<TypeParam, Device::CPU> cpu(0, 5, 0);
    cpu.rowOffsets()[0] = 0;

    const auto gpu = cpu.toGpu();
    const auto back = gpu.toCpu();

    EXPECT_EQ(back.rows(), 0);
    EXPECT_EQ(back.cols(), 5);
    EXPECT_EQ(back.nnz(), 0);
    EXPECT_EQ(back.values(), nullptr);
    EXPECT_EQ(back.colIndices(), nullptr);
    ASSERT_NE(back.rowOffsets(), nullptr);
    EXPECT_EQ(back.rowOffsets()[0], 0);
}
#else
TEST(CSRMatrixNoCuda, transferSurfaceCompilesAndThrowsClearErrors)
{
    CSRMatrix<float, Device::CPU> cpu(2, 2, 1);
    CSRMatrix<float, Device::GPU> gpu(2, 2, 1);
    CSRMatrix<float, Device::CPU> cpu_output(2, 2, 1);
    CSRMatrix<float, Device::GPU> gpu_output(2, 2, 1);

    expectRequiresCuda("toGpu", [&]() { static_cast<void>(cpu.toGpu()); });
    expectRequiresCuda("toCpu", [&]() { static_cast<void>(gpu.toCpu()); });
    expectRequiresCuda("toGpuAsync", [&]() { static_cast<void>(cpu.toGpuAsync(nullptr)); });
    expectRequiresCuda("toCpuAsync", [&]() { static_cast<void>(gpu.toCpuAsync(nullptr)); });
    expectRequiresCuda("copyToGpuAsync", [&]() { cpu.copyToGpuAsync(gpu_output, nullptr); });
    expectRequiresCuda("copyToCpuAsync", [&]() { gpu.copyToCpuAsync(cpu_output, nullptr); });
    expectRequiresCuda("pinned", [&]() {
        static_cast<void>(CSRMatrix<float, Device::CPU>::pinned(1, 1, 0));
    });
    expectRequiresCuda("uninitializedAsync", [&]() {
        static_cast<void>(
            CSRMatrix<float, Device::GPU>::uninitializedAsync(1, 1, 0, nullptr));
    });
}
#endif
