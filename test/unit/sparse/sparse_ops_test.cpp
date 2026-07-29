#include <atomic>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include <plamatrix/plamatrix.h>

using namespace plamatrix;

static_assert(sizeof(Index) == sizeof(std::int64_t));
static_assert(std::is_same_v<Index, std::int64_t>);

#if !defined(PLAMATRIX_USE_FLOAT) && !defined(PLAMATRIX_USE_DOUBLE)
#error "Sparse operations tests require PLAMATRIX_USE_FLOAT or PLAMATRIX_USE_DOUBLE"
#endif

#ifdef PLAMATRIX_USE_DOUBLE
TEST(SparseOps, cooToCsrSortsAndCombinesDuplicatesInInsertionOrder)
{
    const std::vector<Index> row_indices{2, 0, 2, 1, 2, 0, 2};
    const std::vector<Index> col_indices{4, 3, 1, 2, 1, 3, 1};
    const std::vector<double> values{5.0, 2.0, 1.0e16, 4.0, -1.0e16, -2.0, 1.0};

    auto csr = cooToCsr(4, 5, row_indices, col_indices, values);

    ASSERT_EQ(csr.nnz(), Index{4});
    EXPECT_EQ(csr.rowOffsets()[0], Index{0});
    EXPECT_EQ(csr.rowOffsets()[1], Index{1});
    EXPECT_EQ(csr.rowOffsets()[2], Index{2});
    EXPECT_EQ(csr.rowOffsets()[3], Index{4});
    EXPECT_EQ(csr.rowOffsets()[4], Index{4});

    EXPECT_EQ(csr.colIndices()[0], Index{3});
    EXPECT_DOUBLE_EQ(csr.values()[0], 0.0);
    EXPECT_EQ(csr.colIndices()[1], Index{2});
    EXPECT_DOUBLE_EQ(csr.values()[1], 4.0);
    EXPECT_EQ(csr.colIndices()[2], Index{1});
    EXPECT_DOUBLE_EQ(csr.values()[2], 1.0);
    EXPECT_EQ(csr.colIndices()[3], Index{4});
    EXPECT_DOUBLE_EQ(csr.values()[3], 5.0);
}
#endif

#ifdef PLAMATRIX_USE_FLOAT
TEST(SparseOps, cooToCsrReusesOutputStorageForFloat)
{
    const std::vector<Index> row_indices{1, 0, 1};
    const std::vector<Index> col_indices{0, 1, 0};
    const std::vector<float> values{2.0f, 3.0f, 4.0f};
    CSRMatrix<float, Device::CPU> output(2, 2, 2);
    float* values_before = output.values();
    Index* columns_before = output.colIndices();
    Index* offsets_before = output.rowOffsets();

    cooToCsr(2, 2, row_indices, col_indices, values, output);

    EXPECT_EQ(output.values(), values_before);
    EXPECT_EQ(output.colIndices(), columns_before);
    EXPECT_EQ(output.rowOffsets(), offsets_before);
    EXPECT_EQ(output.rowOffsets()[0], Index{0});
    EXPECT_EQ(output.rowOffsets()[1], Index{1});
    EXPECT_EQ(output.rowOffsets()[2], Index{2});
    EXPECT_EQ(output.colIndices()[0], Index{1});
    EXPECT_FLOAT_EQ(output.values()[0], 3.0f);
    EXPECT_EQ(output.colIndices()[1], Index{0});
    EXPECT_FLOAT_EQ(output.values()[1], 6.0f);
}

TEST(SparseOps, cooToCsrHandlesEmptyMatrices)
{
    const std::vector<Index> indices;
    const std::vector<float> values;

    auto csr = cooToCsr(0, 5, indices, indices, values);

    EXPECT_EQ(csr.rows(), Index{0});
    EXPECT_EQ(csr.cols(), Index{5});
    EXPECT_EQ(csr.nnz(), Index{0});
    ASSERT_NE(csr.rowOffsets(), nullptr);
    EXPECT_EQ(csr.rowOffsets()[0], Index{0});
}

TEST(SparseOps, cooToCsrRejectsInvalidInputs)
{
    const std::vector<Index> one_index{0};
    const std::vector<Index> no_indices;
    const std::vector<float> one_value{1.0f};
    const std::vector<float> no_values;

    EXPECT_THROW(cooToCsr(-1, 1, no_indices, no_indices, no_values), std::invalid_argument);
    EXPECT_THROW(cooToCsr(1, -1, no_indices, no_indices, no_values), std::invalid_argument);
    EXPECT_THROW(cooToCsr(1, 1, one_index, no_indices, one_value), std::invalid_argument);
    EXPECT_THROW(cooToCsr(1, 1, one_index, one_index, no_values), std::invalid_argument);

    EXPECT_THROW(cooToCsr(1, 1, std::vector<Index>{-1}, one_index, one_value), std::out_of_range);
    EXPECT_THROW(cooToCsr(1, 1, one_index, std::vector<Index>{-1}, one_value), std::out_of_range);
    EXPECT_THROW(cooToCsr(1, 1, std::vector<Index>{1}, one_index, one_value), std::out_of_range);
    EXPECT_THROW(cooToCsr(1, 1, one_index, std::vector<Index>{1}, one_value), std::out_of_range);

    const Index beyond_int32 = static_cast<Index>(std::numeric_limits<std::int32_t>::max()) + 1;
    EXPECT_THROW(
        cooToCsr(2, 2, std::vector<Index>{beyond_int32}, one_index, one_value),
        std::out_of_range);

    CSRMatrix<float, Device::CPU> wrong_dimensions(1, 2, 1);
    EXPECT_THROW(
        cooToCsr(2, 2, one_index, one_index, one_value, wrong_dimensions),
        std::runtime_error);

    CSRMatrix<float, Device::CPU> wrong_nnz(1, 1, 2);
    EXPECT_THROW(
        cooToCsr(1, 1, one_index, one_index, one_value, wrong_nnz),
        std::runtime_error);
}

TEST(SparseOps, spmvComputesFloatResultAndReusesOutput)
{
    auto csr = cooToCsr(
        3,
        4,
        std::vector<Index>{2, 0, 1, 0},
        std::vector<Index>{3, 2, 1, 0},
        std::vector<float>{5.0f, 2.0f, -1.0f, 3.0f});
    DenseMatrix<float, Device::CPU> x(4, 1);
    x(0, 0) = 1.0f;
    x(1, 0) = 2.0f;
    x(2, 0) = 3.0f;
    x(3, 0) = 4.0f;

    auto result = spmv(csr, x);
    ASSERT_EQ(result.rows(), Index{3});
    ASSERT_EQ(result.cols(), Index{1});
    EXPECT_FLOAT_EQ(result(0, 0), 9.0f);
    EXPECT_FLOAT_EQ(result(1, 0), -2.0f);
    EXPECT_FLOAT_EQ(result(2, 0), 20.0f);

    DenseMatrix<float, Device::CPU> output(3, 1);
    output.fill(-100.0f);
    float* output_before = output.data();
    spmv(csr, x, output);

    EXPECT_EQ(output.data(), output_before);
    EXPECT_FLOAT_EQ(output(0, 0), 9.0f);
    EXPECT_FLOAT_EQ(output(1, 0), -2.0f);
    EXPECT_FLOAT_EQ(output(2, 0), 20.0f);
}
#endif

#ifdef PLAMATRIX_USE_DOUBLE
TEST(SparseOps, spmvHandlesEmptyRowsAndRejectsDimensionMismatch)
{
    auto csr = cooToCsr(
        3,
        2,
        std::vector<Index>{1},
        std::vector<Index>{0},
        std::vector<double>{2.0});
    DenseMatrix<double, Device::CPU> x(2, 1);
    x(0, 0) = 3.0;
    x(1, 0) = 4.0;

    auto result = spmv(csr, x);
    EXPECT_DOUBLE_EQ(result(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(result(1, 0), 6.0);
    EXPECT_DOUBLE_EQ(result(2, 0), 0.0);

    DenseMatrix<double, Device::CPU> wrong_rows(3, 1);
    DenseMatrix<double, Device::CPU> wrong_columns(2, 2);
    DenseMatrix<double, Device::CPU> wrong_output(2, 1);
    EXPECT_THROW(spmv(csr, wrong_rows), std::runtime_error);
    EXPECT_THROW(spmv(csr, wrong_columns), std::runtime_error);
    EXPECT_THROW(spmv(csr, x, wrong_output), std::runtime_error);
}
#endif

#ifdef PLAMATRIX_USE_FLOAT
TEST(SparseOps, spmvRejectsExactDataPointerAliasing)
{
    auto csr = cooToCsr(
        2,
        2,
        std::vector<Index>{0, 1},
        std::vector<Index>{0, 1},
        std::vector<float>{2.0f, 3.0f});
    DenseMatrix<float, Device::CPU> x(2, 1);

    EXPECT_THROW(spmv(csr, x, x), std::invalid_argument);
}
#endif

#ifdef PLAMATRIX_USE_DOUBLE
TEST(SparseOps, spmmUsesColumnMajorDenseMatricesAndReusesOutput)
{
    auto csr = cooToCsr(
        3,
        3,
        std::vector<Index>{1, 0, 2, 0},
        std::vector<Index>{1, 2, 0, 0},
        std::vector<double>{4.0, 2.0, -1.0, 3.0});
    DenseMatrix<double, Device::CPU> B(3, 2);
    B(0, 0) = 1.0;
    B(1, 0) = 2.0;
    B(2, 0) = 3.0;
    B(0, 1) = 10.0;
    B(1, 1) = 20.0;
    B(2, 1) = 30.0;

    auto result = spmm(csr, B);
    EXPECT_DOUBLE_EQ(result(0, 0), 9.0);
    EXPECT_DOUBLE_EQ(result(1, 0), 8.0);
    EXPECT_DOUBLE_EQ(result(2, 0), -1.0);
    EXPECT_DOUBLE_EQ(result(0, 1), 90.0);
    EXPECT_DOUBLE_EQ(result(1, 1), 80.0);
    EXPECT_DOUBLE_EQ(result(2, 1), -10.0);

    DenseMatrix<double, Device::CPU> output(3, 2);
    output.fill(-100.0);
    double* output_before = output.data();
    spmm(csr, B, output);

    EXPECT_EQ(output.data(), output_before);
    for (Index col = 0; col < output.cols(); ++col)
    {
        for (Index row = 0; row < output.rows(); ++row)
        {
            EXPECT_DOUBLE_EQ(output(row, col), result(row, col));
        }
    }
}
#endif

#ifdef PLAMATRIX_USE_FLOAT
TEST(SparseOps, spmmHandlesEmptyMatricesAndRejectsDimensionMismatch)
{
    const std::vector<Index> no_indices;
    const std::vector<float> no_values;
    auto csr = cooToCsr(0, 3, no_indices, no_indices, no_values);
    DenseMatrix<float, Device::CPU> B(3, 2);

    auto result = spmm(csr, B);
    EXPECT_EQ(result.rows(), Index{0});
    EXPECT_EQ(result.cols(), Index{2});

    DenseMatrix<float, Device::CPU> wrong_B(2, 2);
    DenseMatrix<float, Device::CPU> wrong_output(1, 2);
    EXPECT_THROW(spmm(csr, wrong_B), std::runtime_error);
    EXPECT_THROW(spmm(csr, B, wrong_output), std::runtime_error);
}
#endif

#ifdef PLAMATRIX_USE_DOUBLE
TEST(SparseOps, spmmRejectsExactDataPointerAliasing)
{
    auto csr = cooToCsr(
        2,
        2,
        std::vector<Index>{0, 1},
        std::vector<Index>{0, 1},
        std::vector<double>{2.0, 3.0});
    DenseMatrix<double, Device::CPU> B(2, 2);

    EXPECT_THROW(spmm(csr, B, B), std::invalid_argument);
}
#endif

#ifdef PLAMATRIX_USE_FLOAT
using CooMatrixScalar = float;
#else
using CooMatrixScalar = double;
#endif

TEST(COOMatrix, cpuToCsrUsesDuplicateCombiningConversion)
{
    COOMatrix<CooMatrixScalar, Device::CPU> matrix(2, 3);
    matrix.add(1, 2, static_cast<CooMatrixScalar>(1.5));
    matrix.add(0, 1, static_cast<CooMatrixScalar>(2.0));
    matrix.add(1, 2, static_cast<CooMatrixScalar>(2.5));

    auto csr = matrix.toCsr();

    ASSERT_EQ(csr.nnz(), Index{2});
    EXPECT_EQ(csr.rowOffsets()[0], Index{0});
    EXPECT_EQ(csr.rowOffsets()[1], Index{1});
    EXPECT_EQ(csr.rowOffsets()[2], Index{2});
    EXPECT_EQ(csr.colIndices()[0], Index{1});
    EXPECT_EQ(csr.values()[0], static_cast<CooMatrixScalar>(2.0));
    EXPECT_EQ(csr.colIndices()[1], Index{2});
    EXPECT_EQ(csr.values()[1], static_cast<CooMatrixScalar>(4.0));
}

static_assert(!std::is_copy_constructible_v<SparseOpsWorkspace>);
static_assert(!std::is_copy_assignable_v<SparseOpsWorkspace>);
static_assert(std::is_nothrow_move_constructible_v<SparseOpsWorkspace>);
static_assert(std::is_nothrow_move_assignable_v<SparseOpsWorkspace>);

#ifdef PLAMATRIX_WITH_CUDA

template <typename Scalar>
class SparseOpsGpuTest : public ::testing::Test
{
};

using SparseOpsGpuScalars = ::testing::Types<
#ifdef PLAMATRIX_USE_FLOAT
    float
#ifdef PLAMATRIX_USE_DOUBLE
    ,
#endif
#endif
#ifdef PLAMATRIX_USE_DOUBLE
    double
#endif
    >;
TYPED_TEST_SUITE(SparseOpsGpuTest, SparseOpsGpuScalars);

TYPED_TEST(SparseOpsGpuTest, deviceCooToCsrMatchesCpuAndPreservesTriplets)
{
    DenseMatrix<Index, Device::CPU> row_cpu(7, 1);
    DenseMatrix<Index, Device::CPU> col_cpu(7, 1);
    DenseMatrix<TypeParam, Device::CPU> value_cpu(7, 1);
    const std::vector<Index> rows{2, 0, 2, 1, 2, 0, 2};
    const std::vector<Index> cols{4, 3, 1, 2, 1, 3, 1};
    const TypeParam large_value = std::is_same_v<TypeParam, float>
        ? TypeParam(1.0e8) : TypeParam(1.0e16);
    const std::vector<TypeParam> values{
        TypeParam(5), TypeParam(2), large_value, TypeParam(4),
        -large_value, TypeParam(-2), TypeParam(1)};
    for (Index i = 0; i < 7; ++i)
    {
        row_cpu(i, 0) = rows[static_cast<std::size_t>(i)];
        col_cpu(i, 0) = cols[static_cast<std::size_t>(i)];
        value_cpu(i, 0) = values[static_cast<std::size_t>(i)];
    }
    const auto row_gpu = row_cpu.toGpu();
    const auto col_gpu = col_cpu.toGpu();
    const auto value_gpu = value_cpu.toGpu();
    SparseOpsWorkspace workspace;

    const auto csr_gpu = cooToCsr(4, 5, row_gpu, col_gpu, value_gpu, workspace);
    const auto csr = csr_gpu.toCpu();
    const auto rows_after = row_gpu.toCpu();
    const auto cols_after = col_gpu.toCpu();
    const auto values_after = value_gpu.toCpu();

    ASSERT_EQ(csr.nnz(), Index{4});
    EXPECT_EQ(csr.rowOffsets()[0], Index{0});
    EXPECT_EQ(csr.rowOffsets()[1], Index{1});
    EXPECT_EQ(csr.rowOffsets()[2], Index{2});
    EXPECT_EQ(csr.rowOffsets()[3], Index{4});
    EXPECT_EQ(csr.rowOffsets()[4], Index{4});
    EXPECT_EQ(csr.colIndices()[0], Index{3});
    EXPECT_EQ(csr.values()[0], TypeParam(0));
    EXPECT_EQ(csr.colIndices()[1], Index{2});
    EXPECT_EQ(csr.values()[1], TypeParam(4));
    EXPECT_EQ(csr.colIndices()[2], Index{1});
    EXPECT_EQ(csr.values()[2], TypeParam(1));
    EXPECT_EQ(csr.colIndices()[3], Index{4});
    EXPECT_EQ(csr.values()[3], TypeParam(5));
    for (Index i = 0; i < 7; ++i)
    {
        EXPECT_EQ(rows_after(i, 0), rows[static_cast<std::size_t>(i)]);
        EXPECT_EQ(cols_after(i, 0), cols[static_cast<std::size_t>(i)]);
        EXPECT_EQ(values_after(i, 0), values[static_cast<std::size_t>(i)]);
    }
}

TYPED_TEST(SparseOpsGpuTest, deviceCooToCsrAsyncUsesNonDefaultStreamAndReusesOutput)
{
    auto row_cpu = DenseMatrix<Index, Device::CPU>::pinned(5, 1);
    auto col_cpu = DenseMatrix<Index, Device::CPU>::pinned(5, 1);
    auto value_cpu = DenseMatrix<TypeParam, Device::CPU>::pinned(5, 1);
    const Index rows[] = {1, 0, 1, 2, 1};
    const Index cols[] = {0, 2, 0, 1, 0};
    const TypeParam values[] = {
        TypeParam(2), TypeParam(3), TypeParam(4), TypeParam(5), TypeParam(-1)};
    for (Index i = 0; i < 5; ++i)
    {
        row_cpu(i, 0) = rows[i];
        col_cpu(i, 0) = cols[i];
        value_cpu(i, 0) = values[i];
    }
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    auto row_gpu = DenseMatrix<Index, Device::GPU>::uninitializedAsync(5, 1, stream);
    auto col_gpu = DenseMatrix<Index, Device::GPU>::uninitializedAsync(5, 1, stream);
    auto value_gpu = DenseMatrix<TypeParam, Device::GPU>::uninitializedAsync(5, 1, stream);
    row_cpu.copyToGpuAsync(row_gpu, stream);
    col_cpu.copyToGpuAsync(col_gpu, stream);
    value_cpu.copyToGpuAsync(value_gpu, stream);
    auto output = CSRMatrix<TypeParam, Device::GPU>::uninitializedAsync(3, 3, 3, stream);
    SparseOpsWorkspace workspace;

    cooToCsrAsync(3, 3, row_gpu, col_gpu, value_gpu, output, workspace, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    workspace.checkStatus("cooToCsrAsync");
    TypeParam* values_pointer = output.values();
    Index* columns_pointer = output.colIndices();
    Index* offsets_pointer = output.rowOffsets();

    cooToCsrAsync(3, 3, row_gpu, col_gpu, value_gpu, output, workspace, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    workspace.checkStatus("cooToCsrAsync");
    EXPECT_EQ(output.values(), values_pointer);
    EXPECT_EQ(output.colIndices(), columns_pointer);
    EXPECT_EQ(output.rowOffsets(), offsets_pointer);
    auto result = output.toCpuAsync(stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    EXPECT_EQ(result.rowOffsets()[0], Index{0});
    EXPECT_EQ(result.rowOffsets()[1], Index{1});
    EXPECT_EQ(result.rowOffsets()[2], Index{2});
    EXPECT_EQ(result.rowOffsets()[3], Index{3});
    EXPECT_EQ(result.colIndices()[0], Index{2});
    EXPECT_EQ(result.values()[0], TypeParam(3));
    EXPECT_EQ(result.colIndices()[1], Index{0});
    EXPECT_EQ(result.values()[1], TypeParam(5));
    EXPECT_EQ(result.colIndices()[2], Index{1});
    EXPECT_EQ(result.values()[2], TypeParam(5));

    workspace.closeAsyncAllocation();
    row_gpu.closeAsyncAllocation();
    col_gpu.closeAsyncAllocation();
    value_gpu.closeAsyncAllocation();
    output.closeAsyncAllocation();
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TYPED_TEST(SparseOpsGpuTest, checkedAsyncCooOutputIsImmediatelyConsumableBySpmv)
{
    DenseMatrix<Index, Device::CPU> row_cpu(2, 1);
    DenseMatrix<Index, Device::CPU> col_cpu(2, 1);
    DenseMatrix<TypeParam, Device::CPU> value_cpu(2, 1);
    row_cpu(0, 0) = 0;
    row_cpu(1, 0) = 1;
    col_cpu(0, 0) = 0;
    col_cpu(1, 0) = 1;
    value_cpu(0, 0) = TypeParam(2);
    value_cpu(1, 0) = TypeParam(3);
    const auto row_gpu = row_cpu.toGpu();
    const auto col_gpu = col_cpu.toGpu();
    const auto value_gpu = value_cpu.toGpu();
    CSRMatrix<TypeParam, Device::GPU> output(2, 2, 2);
    SparseOpsWorkspace coo_workspace;

    cooToCsrAsync(
        2, 2, row_gpu, col_gpu, value_gpu, output, coo_workspace, nullptr);
    ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
    ASSERT_NO_THROW(coo_workspace.checkStatus("cooToCsrAsync"));
    EXPECT_TRUE(output.hasValidatedStructure());

    DenseMatrix<TypeParam, Device::CPU> x_cpu(2, 1);
    x_cpu(0, 0) = TypeParam(4);
    x_cpu(1, 0) = TypeParam(5);
    const auto x_gpu = x_cpu.toGpu();
    DenseMatrix<TypeParam, Device::GPU> y_gpu(2, 1);
    SparseOpsWorkspace spmv_workspace;
    EXPECT_NO_THROW(spmvAsync(output, x_gpu, y_gpu, spmv_workspace, nullptr));
    ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
    const auto y_cpu = y_gpu.toCpu();
    EXPECT_EQ(y_cpu(0, 0), TypeParam(8));
    EXPECT_EQ(y_cpu(1, 0), TypeParam(15));
}

TYPED_TEST(SparseOpsGpuTest, deviceCooToCsrAsyncDefersCoordinateAndNnzErrors)
{
    DenseMatrix<Index, Device::CPU> row_cpu(2, 1);
    DenseMatrix<Index, Device::CPU> col_cpu(2, 1);
    DenseMatrix<TypeParam, Device::CPU> value_cpu(2, 1);
    row_cpu(0, 0) = 0;
    row_cpu(1, 0) = 0;
    col_cpu(0, 0) = 1;
    col_cpu(1, 0) = 1;
    value_cpu(0, 0) = TypeParam(2);
    value_cpu(1, 0) = TypeParam(3);
    const auto row_gpu = row_cpu.toGpu();
    const auto col_gpu = col_cpu.toGpu();
    const auto value_gpu = value_cpu.toGpu();
    CSRMatrix<TypeParam, Device::GPU> wrong_nnz(2, 2, 2);
    SparseOpsWorkspace workspace;

    EXPECT_NO_THROW(cooToCsrAsync(
        2, 2, row_gpu, col_gpu, value_gpu, wrong_nnz, workspace));
    ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
    EXPECT_THROW(workspace.checkStatus("cooToCsrAsync"), std::runtime_error);

    row_cpu(1, 0) = 2;
    const auto invalid_row_gpu = row_cpu.toGpu();
    CSRMatrix<TypeParam, Device::GPU> valid_capacity(2, 2, 1);
    EXPECT_NO_THROW(cooToCsrAsync(
        2, 2, invalid_row_gpu, col_gpu, value_gpu, valid_capacity, workspace));
    ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
    EXPECT_THROW(workspace.checkStatus("cooToCsrAsync"), std::out_of_range);
}

TYPED_TEST(SparseOpsGpuTest, deviceCooToCsrRejectsNonFiniteInputAndCombinedValue)
{
    DenseMatrix<Index, Device::CPU> rows_cpu(2, 1);
    DenseMatrix<Index, Device::CPU> columns_cpu(2, 1);
    DenseMatrix<TypeParam, Device::CPU> values_cpu(2, 1);
    rows_cpu.fill(Index{0});
    columns_cpu.fill(Index{0});
    values_cpu(0, 0) = std::numeric_limits<TypeParam>::infinity();
    values_cpu(1, 0) = TypeParam(1);
    const auto rows_gpu = rows_cpu.toGpu();
    const auto columns_gpu = columns_cpu.toGpu();
    auto values_gpu = values_cpu.toGpu();
    SparseOpsWorkspace workspace;

    EXPECT_THROW(
        (void)cooToCsr(1, 1, rows_gpu, columns_gpu, values_gpu, workspace),
        std::invalid_argument);

    values_cpu.fill(std::numeric_limits<TypeParam>::max());
    values_gpu = values_cpu.toGpu();
    EXPECT_THROW(
        (void)cooToCsr(1, 1, rows_gpu, columns_gpu, values_gpu, workspace),
        std::invalid_argument);
}

TYPED_TEST(SparseOpsGpuTest, deviceCooToCsrHandlesEmptyAndRejectsUnsupportedRanges)
{
    DenseMatrix<Index, Device::GPU> no_rows(0, 1);
    DenseMatrix<Index, Device::GPU> no_cols(0, 1);
    DenseMatrix<TypeParam, Device::GPU> no_values(0, 1);
    CSRMatrix<TypeParam, Device::GPU> empty_output(3, 4, 0);
    SparseOpsWorkspace workspace;

    EXPECT_NO_THROW(cooToCsrAsync(
        3, 4, no_rows, no_cols, no_values, empty_output, workspace));
    ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
    EXPECT_NO_THROW(workspace.checkStatus("cooToCsrAsync"));
    const auto empty_cpu = empty_output.toCpu();
    EXPECT_EQ(empty_cpu.rowOffsets()[0], Index{0});
    EXPECT_EQ(empty_cpu.rowOffsets()[3], Index{0});

    const Index unsupported_rows = static_cast<Index>(std::numeric_limits<int>::max());
    EXPECT_THROW(cooToCsrAsync(
        unsupported_rows, 1, no_rows, no_cols, no_values, empty_output, workspace),
        std::overflow_error);
}

TYPED_TEST(SparseOpsGpuTest, emptyDeviceCooStatusStillRequiresStreamCompletion)
{
    DenseMatrix<Index, Device::GPU> no_rows(0, 1);
    DenseMatrix<Index, Device::GPU> no_cols(0, 1);
    DenseMatrix<TypeParam, Device::GPU> no_values(0, 1);
    CSRMatrix<TypeParam, Device::GPU> output(2, 2, 0);
    SparseOpsWorkspace workspace;
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    cooToCsrAsync(2, 2, no_rows, no_cols, no_values, output, workspace, stream);
    std::atomic<bool> release_stream{false};
    ASSERT_EQ(cudaLaunchHostFunc(
                  stream,
                  [](void* data) {
                      auto* release = static_cast<std::atomic<bool>*>(data);
                      while (!release->load(std::memory_order_acquire))
                      {
                          std::this_thread::yield();
                      }
                  },
                  &release_stream),
              cudaSuccess);

    EXPECT_THROW(workspace.checkStatus("cooToCsrAsync"), std::logic_error);
    release_stream.store(true, std::memory_order_release);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    EXPECT_NO_THROW(workspace.checkStatus("cooToCsrAsync"));
    EXPECT_NO_THROW(workspace.closeAsyncAllocation());
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TYPED_TEST(SparseOpsGpuTest, deviceCooToCsrRejectsCrossStreamOrderedTriplets)
{
    cudaStream_t owner_stream = nullptr;
    cudaStream_t other_stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&owner_stream, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaStreamCreateWithFlags(&other_stream, cudaStreamNonBlocking), cudaSuccess);
    auto row_gpu = DenseMatrix<Index, Device::GPU>::uninitializedAsync(1, 1, owner_stream);
    auto col_gpu = DenseMatrix<Index, Device::GPU>::uninitializedAsync(1, 1, owner_stream);
    auto value_gpu = DenseMatrix<TypeParam, Device::GPU>::uninitializedAsync(1, 1, owner_stream);
    auto output = CSRMatrix<TypeParam, Device::GPU>::uninitializedAsync(1, 1, 1, owner_stream);
    SparseOpsWorkspace workspace;

    EXPECT_THROW(cooToCsrAsync(
        1, 1, row_gpu, col_gpu, value_gpu, output, workspace, other_stream),
        std::logic_error);

    ASSERT_EQ(cudaStreamSynchronize(owner_stream), cudaSuccess);
    row_gpu.closeAsyncAllocation();
    col_gpu.closeAsyncAllocation();
    value_gpu.closeAsyncAllocation();
    output.closeAsyncAllocation();
    ASSERT_EQ(cudaStreamSynchronize(owner_stream), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(other_stream), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(owner_stream), cudaSuccess);
}

TYPED_TEST(SparseOpsGpuTest, spmvAsyncUsesNonDefaultStreamAndReusesWorkspace)
{
    auto csr_cpu = cooToCsr(
        3,
        4,
        std::vector<Index>{2, 0, 1, 0},
        std::vector<Index>{3, 2, 1, 0},
        std::vector<TypeParam>{TypeParam(5), TypeParam(2), TypeParam(-1), TypeParam(3)});
    DenseMatrix<TypeParam, Device::CPU> x_cpu(4, 1);
    x_cpu(0, 0) = TypeParam(1);
    x_cpu(1, 0) = TypeParam(2);
    x_cpu(2, 0) = TypeParam(3);
    x_cpu(3, 0) = TypeParam(4);
    auto csr_gpu = csr_cpu.toGpu();
    auto x_gpu = x_cpu.toGpu();
    DenseMatrix<TypeParam, Device::GPU> output(3, 1);
    SparseOpsWorkspace workspace;
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);

    spmvAsync(csr_gpu, x_gpu, output, workspace, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    const std::size_t first_capacity = workspace.capacityBytes();
    TypeParam* output_pointer = output.data();

    spmvAsync(csr_gpu, x_gpu, output, workspace, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    EXPECT_EQ(output.data(), output_pointer);
    EXPECT_EQ(workspace.capacityBytes(), first_capacity);

    const auto result = output.toCpu();
    EXPECT_EQ(result(0, 0), TypeParam(9));
    EXPECT_EQ(result(1, 0), TypeParam(-2));
    EXPECT_EQ(result(2, 0), TypeParam(20));
    EXPECT_NO_THROW(workspace.closeAsyncAllocation());
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TYPED_TEST(SparseOpsGpuTest, spmvAsyncRejectsPendingCsrFromAnotherStream)
{
    auto csr_cpu = CSRMatrix<TypeParam, Device::CPU>::pinned(2, 2, 2);
    csr_cpu.rowOffsets()[0] = 0;
    csr_cpu.rowOffsets()[1] = 1;
    csr_cpu.rowOffsets()[2] = 2;
    csr_cpu.colIndices()[0] = 0;
    csr_cpu.colIndices()[1] = 1;
    csr_cpu.values()[0] = TypeParam(2);
    csr_cpu.values()[1] = TypeParam(3);
    CSRMatrix<TypeParam, Device::GPU> csr_gpu(2, 2, 2);
    DenseMatrix<TypeParam, Device::CPU> input_cpu(2, 1);
    input_cpu.fill(TypeParam(1));
    const auto input_gpu = input_cpu.toGpu();
    DenseMatrix<TypeParam, Device::GPU> output_gpu(2, 1);
    SparseOpsWorkspace workspace;
    cudaStream_t copy_stream = nullptr;
    cudaStream_t consumer_stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaStreamCreateWithFlags(&consumer_stream, cudaStreamNonBlocking), cudaSuccess);

    csr_cpu.copyToGpuAsync(csr_gpu, copy_stream);
    EXPECT_THROW(
        spmvAsync(csr_gpu, input_gpu, output_gpu, workspace, consumer_stream),
        std::logic_error);

    ASSERT_EQ(cudaStreamSynchronize(copy_stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(consumer_stream), cudaSuccess);
    EXPECT_NO_THROW(workspace.closeAsyncAllocation());
    ASSERT_EQ(cudaStreamSynchronize(consumer_stream), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(consumer_stream), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(copy_stream), cudaSuccess);
}

TYPED_TEST(SparseOpsGpuTest, synchronousSpmvAllocatesAndReusesOutput)
{
    auto csr_cpu = cooToCsr(
        2,
        2,
        std::vector<Index>{0, 1},
        std::vector<Index>{0, 1},
        std::vector<TypeParam>{TypeParam(2), TypeParam(3)});
    DenseMatrix<TypeParam, Device::CPU> x_cpu(2, 1);
    x_cpu(0, 0) = TypeParam(4);
    x_cpu(1, 0) = TypeParam(5);
    const auto csr_gpu = csr_cpu.toGpu();
    auto x_gpu = x_cpu.toGpu();

    auto result_gpu = spmv(csr_gpu, x_gpu);
    const auto result = result_gpu.toCpu();
    EXPECT_EQ(result(0, 0), TypeParam(8));
    EXPECT_EQ(result(1, 0), TypeParam(15));

    DenseMatrix<TypeParam, Device::GPU> reused(2, 1);
    TypeParam* pointer = reused.data();
    spmv(csr_gpu, x_gpu, reused);
    EXPECT_EQ(reused.data(), pointer);
    const auto reused_cpu = reused.toCpu();
    EXPECT_EQ(reused_cpu(0, 0), TypeParam(8));
    EXPECT_EQ(reused_cpu(1, 0), TypeParam(15));
}

TYPED_TEST(SparseOpsGpuTest, spmmAsyncUsesColumnMajorDenseMatrices)
{
    auto csr_cpu = cooToCsr(
        3,
        3,
        std::vector<Index>{1, 0, 2, 0},
        std::vector<Index>{1, 2, 0, 0},
        std::vector<TypeParam>{TypeParam(4), TypeParam(2), TypeParam(-1), TypeParam(3)});
    DenseMatrix<TypeParam, Device::CPU> b_cpu(3, 2);
    b_cpu(0, 0) = TypeParam(1);
    b_cpu(1, 0) = TypeParam(2);
    b_cpu(2, 0) = TypeParam(3);
    b_cpu(0, 1) = TypeParam(10);
    b_cpu(1, 1) = TypeParam(20);
    b_cpu(2, 1) = TypeParam(30);
    const auto csr_gpu = csr_cpu.toGpu();
    const auto b_gpu = b_cpu.toGpu();
    DenseMatrix<TypeParam, Device::GPU> output(3, 2);
    SparseOpsWorkspace workspace;
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);

    spmmAsync(csr_gpu, b_gpu, output, workspace, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    const auto result = output.toCpu();
    EXPECT_EQ(result(0, 0), TypeParam(9));
    EXPECT_EQ(result(1, 0), TypeParam(8));
    EXPECT_EQ(result(2, 0), TypeParam(-1));
    EXPECT_EQ(result(0, 1), TypeParam(90));
    EXPECT_EQ(result(1, 1), TypeParam(80));
    EXPECT_EQ(result(2, 1), TypeParam(-10));
    EXPECT_NO_THROW(workspace.closeAsyncAllocation());
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TYPED_TEST(SparseOpsGpuTest, synchronousSpmmAllocatesAndReusesOutput)
{
    auto csr_cpu = cooToCsr(
        2,
        2,
        std::vector<Index>{0, 1},
        std::vector<Index>{0, 1},
        std::vector<TypeParam>{TypeParam(2), TypeParam(3)});
    DenseMatrix<TypeParam, Device::CPU> input_cpu(2, 2);
    input_cpu(0, 0) = TypeParam(4);
    input_cpu(1, 0) = TypeParam(5);
    input_cpu(0, 1) = TypeParam(6);
    input_cpu(1, 1) = TypeParam(7);
    const auto csr_gpu = csr_cpu.toGpu();
    const auto input_gpu = input_cpu.toGpu();

    auto result_gpu = spmm(csr_gpu, input_gpu);
    const auto result = result_gpu.toCpu();
    EXPECT_EQ(result(0, 0), TypeParam(8));
    EXPECT_EQ(result(1, 0), TypeParam(15));
    EXPECT_EQ(result(0, 1), TypeParam(12));
    EXPECT_EQ(result(1, 1), TypeParam(21));

    DenseMatrix<TypeParam, Device::GPU> reused(2, 2);
    TypeParam* pointer = reused.data();
    spmm(csr_gpu, input_gpu, reused);
    EXPECT_EQ(reused.data(), pointer);
    const auto reused_cpu = reused.toCpu();
    EXPECT_EQ(reused_cpu(0, 1), TypeParam(12));
    EXPECT_EQ(reused_cpu(1, 1), TypeParam(21));
}

TYPED_TEST(SparseOpsGpuTest, productsHandleEmptyMatricesAndRejectInvalidOutputs)
{
    const std::vector<Index> no_indices;
    const std::vector<TypeParam> no_values;
    const auto empty_gpu = cooToCsr(0, 3, no_indices, no_indices, no_values).toGpu();
    DenseMatrix<TypeParam, Device::CPU> b_cpu(3, 2);
    const auto b_gpu = b_cpu.toGpu();
    DenseMatrix<TypeParam, Device::GPU> empty_output(0, 2);
    SparseOpsWorkspace workspace;
    EXPECT_NO_THROW(spmmAsync(empty_gpu, b_gpu, empty_output, workspace));

    auto csr_cpu = cooToCsr(
        2,
        2,
        std::vector<Index>{0, 1},
        std::vector<Index>{0, 1},
        std::vector<TypeParam>{TypeParam(2), TypeParam(3)});
    const auto csr_gpu = csr_cpu.toGpu();
    DenseMatrix<TypeParam, Device::CPU> x_cpu(2, 1);
    auto x_gpu = x_cpu.toGpu();
    DenseMatrix<TypeParam, Device::GPU> wrong_rows(1, 1);
    DenseMatrix<TypeParam, Device::GPU> wrong_columns(2, 2);

    EXPECT_THROW(spmvAsync(csr_gpu, x_gpu, wrong_rows, workspace), std::runtime_error);
    EXPECT_THROW(spmvAsync(csr_gpu, wrong_columns, wrong_rows, workspace), std::runtime_error);
    EXPECT_THROW(spmvAsync(csr_gpu, x_gpu, x_gpu, workspace),
                 std::invalid_argument);
    EXPECT_THROW(spmmAsync(csr_gpu, wrong_columns, wrong_columns, workspace), std::invalid_argument);
}

TYPED_TEST(SparseOpsGpuTest, workspaceSwitchesBetweenMatrixAndVectorDescriptors)
{
    auto csr_cpu = cooToCsr(
        2,
        2,
        std::vector<Index>{0, 1},
        std::vector<Index>{0, 1},
        std::vector<TypeParam>{TypeParam(2), TypeParam(3)});
    DenseMatrix<TypeParam, Device::CPU> input_cpu(2, 2);
    input_cpu(0, 0) = TypeParam(4);
    input_cpu(1, 0) = TypeParam(5);
    input_cpu(0, 1) = TypeParam(6);
    input_cpu(1, 1) = TypeParam(7);
    const auto csr_gpu = csr_cpu.toGpu();
    const auto input_gpu = input_cpu.toGpu();
    DenseMatrix<TypeParam, Device::GPU> matrix_output(2, 2);
    DenseMatrix<TypeParam, Device::GPU> vector_output(2, 1);
    SparseOpsWorkspace workspace;

    spmmAsync(csr_gpu, input_gpu, matrix_output, workspace);
    DenseMatrix<TypeParam, Device::CPU> vector_cpu(2, 1);
    vector_cpu(0, 0) = TypeParam(4);
    vector_cpu(1, 0) = TypeParam(5);
    const auto vector_gpu = vector_cpu.toGpu();
    spmvAsync(csr_gpu, vector_gpu, vector_output, workspace);
    spmmAsync(csr_gpu, input_gpu, matrix_output, workspace);
    ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);

    const auto vector_result = vector_output.toCpu();
    EXPECT_EQ(vector_result(0, 0), TypeParam(8));
    EXPECT_EQ(vector_result(1, 0), TypeParam(15));
    const auto matrix_result = matrix_output.toCpu();
    EXPECT_EQ(matrix_result(0, 1), TypeParam(12));
    EXPECT_EQ(matrix_result(1, 1), TypeParam(21));
}

TYPED_TEST(SparseOpsGpuTest, asyncProductsRejectCrossStreamOrderedStorage)
{
    auto csr_cpu = CSRMatrix<TypeParam, Device::CPU>::pinned(2, 2, 2);
    csr_cpu.rowOffsets()[0] = 0;
    csr_cpu.rowOffsets()[1] = 1;
    csr_cpu.rowOffsets()[2] = 2;
    csr_cpu.colIndices()[0] = 0;
    csr_cpu.colIndices()[1] = 1;
    csr_cpu.values()[0] = TypeParam(2);
    csr_cpu.values()[1] = TypeParam(3);
    auto input_cpu = DenseMatrix<TypeParam, Device::CPU>::pinned(2, 1);
    input_cpu(0, 0) = TypeParam(4);
    input_cpu(1, 0) = TypeParam(5);
    cudaStream_t owner_stream = nullptr;
    cudaStream_t other_stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&owner_stream, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaStreamCreateWithFlags(&other_stream, cudaStreamNonBlocking), cudaSuccess);
    auto csr_gpu = csr_cpu.toGpuAsync(owner_stream);
    auto input_gpu = DenseMatrix<TypeParam, Device::GPU>::uninitializedAsync(2, 1, owner_stream);
    auto output_gpu = DenseMatrix<TypeParam, Device::GPU>::uninitializedAsync(2, 1, owner_stream);
    input_cpu.copyToGpuAsync(input_gpu, owner_stream);
    SparseOpsWorkspace workspace;

    EXPECT_THROW(
        spmvAsync(csr_gpu, input_gpu, output_gpu, workspace, other_stream),
        std::logic_error);

    ASSERT_EQ(cudaStreamSynchronize(owner_stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(other_stream), cudaSuccess);
    workspace.closeAsyncAllocation();
    csr_gpu.closeAsyncAllocation();
    input_gpu.closeAsyncAllocation();
    output_gpu.closeAsyncAllocation();
    ASSERT_EQ(cudaStreamSynchronize(owner_stream), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(other_stream), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(owner_stream), cudaSuccess);
}

TYPED_TEST(SparseOpsGpuTest, workspaceMoveCloseAndCrossStreamRulesAreStable)
{
    auto csr_cpu = cooToCsr(
        2,
        2,
        std::vector<Index>{0, 1},
        std::vector<Index>{0, 1},
        std::vector<TypeParam>{TypeParam(2), TypeParam(3)});
    DenseMatrix<TypeParam, Device::CPU> input_cpu(2, 1);
    input_cpu(0, 0) = TypeParam(4);
    input_cpu(1, 0) = TypeParam(5);
    const auto csr_gpu = csr_cpu.toGpu();
    const auto input_gpu = input_cpu.toGpu();
    DenseMatrix<TypeParam, Device::GPU> output(2, 1);
    cudaStream_t first_stream = nullptr;
    cudaStream_t second_stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&first_stream, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaStreamCreateWithFlags(&second_stream, cudaStreamNonBlocking), cudaSuccess);
    SparseOpsWorkspace workspace;
    spmvAsync(csr_gpu, input_gpu, output, workspace, first_stream);
    ASSERT_EQ(cudaStreamSynchronize(first_stream), cudaSuccess);

    SparseOpsWorkspace moved(std::move(workspace));
    EXPECT_EQ(workspace.capacityBytes(), std::size_t{0});
    EXPECT_THROW(
        spmvAsync(csr_gpu, input_gpu, output, moved, second_stream),
        std::logic_error);
    EXPECT_NO_THROW(moved.closeAsyncAllocation());
    EXPECT_NO_THROW(moved.closeAsyncAllocation());
    EXPECT_NO_THROW(spmvAsync(csr_gpu, input_gpu, output, moved, second_stream));
    ASSERT_EQ(cudaStreamSynchronize(second_stream), cudaSuccess);
    EXPECT_NO_THROW(moved.closeAsyncAllocation());
    ASSERT_EQ(cudaStreamSynchronize(first_stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(second_stream), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(second_stream), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(first_stream), cudaSuccess);
}

TYPED_TEST(SparseOpsGpuTest, workspaceRejectsCloseWhileOwnerStreamIsPending)
{
    auto csr_cpu = cooToCsr(
        1, 1, std::vector<Index>{}, std::vector<Index>{},
        std::vector<TypeParam>{});
    DenseMatrix<TypeParam, Device::CPU> input_cpu(1, 1);
    input_cpu(0, 0) = TypeParam(3);
    const auto csr_gpu = csr_cpu.toGpu();
    const auto input_gpu = input_cpu.toGpu();
    DenseMatrix<TypeParam, Device::GPU> output(1, 1);
    SparseOpsWorkspace workspace;
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    std::atomic<bool> release_stream{false};
    spmvAsync(csr_gpu, input_gpu, output, workspace, stream);
    ASSERT_EQ(cudaLaunchHostFunc(
                  stream,
                  [](void* data) {
                      auto* release = static_cast<std::atomic<bool>*>(data);
                      while (!release->load(std::memory_order_acquire))
                      {
                          std::this_thread::yield();
                      }
                  },
                  &release_stream),
              cudaSuccess);

    EXPECT_THROW(workspace.closeAsyncAllocation(), std::logic_error);
    release_stream.store(true, std::memory_order_release);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    EXPECT_NO_THROW(workspace.closeAsyncAllocation());
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

#else

TEST(SparseOpsNoCuda, gpuProductsReportUnavailableBackend)
{
    CSRMatrix<float, Device::GPU> csr(0, 0, 0);
    DenseMatrix<float, Device::GPU> input;
    DenseMatrix<float, Device::GPU> output;
    DenseMatrix<Index, Device::GPU> indices;
    SparseOpsWorkspace workspace;

    EXPECT_THROW(static_cast<void>(cooToCsr(
                     0, 0, indices, indices, input, workspace)),
                 std::runtime_error);
    EXPECT_THROW(cooToCsrAsync(
                     0, 0, indices, indices, input, csr, workspace),
                 std::runtime_error);

    EXPECT_THROW(static_cast<void>(spmv(csr, input)), std::runtime_error);
    EXPECT_THROW(spmv(csr, input, output), std::runtime_error);
    EXPECT_THROW(spmvAsync(csr, input, output, workspace), std::runtime_error);
    EXPECT_THROW(static_cast<void>(spmm(csr, input)), std::runtime_error);
    EXPECT_THROW(spmm(csr, input, output), std::runtime_error);
    EXPECT_THROW(spmmAsync(csr, input, output, workspace), std::runtime_error);
}

#endif
