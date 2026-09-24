#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <type_traits>

#include <plamatrix/internal/dense/dense_storage.h>
#include <plamatrix/internal/dense/matrix_view.h>

namespace plamatrix::internal
{

TEST(MatrixView, DenseMatrixViewMutatesOwnerAndPreservesColumnMajorLayout)
{
    DenseStorage<double, Device::CPU> matrix(3, 4);
    auto view = matrix.view();

    EXPECT_EQ(view.rows(), 3);
    EXPECT_EQ(view.cols(), 4);
    EXPECT_EQ(view.rowStride(), 1);
    EXPECT_EQ(view.colStride(), 3);
    EXPECT_TRUE(view.isContiguousColumnMajor());
    EXPECT_FALSE(view.isContiguousRowMajor());

    view(2, 3) = 17.0;
    EXPECT_DOUBLE_EQ(matrix(2, 3), 17.0);
}

TEST(MatrixView, BlockRowColumnAndTransposeRemainZeroCopy)
{
    DenseStorage<int, Device::CPU> matrix(4, 5);
    for (Index col = 0; col < matrix.cols(); ++col)
    {
        for (Index row = 0; row < matrix.rows(); ++row)
        {
            matrix(row, col) = static_cast<int>(row + 10 * col);
        }
    }

    auto block = matrix.block(1, 2, 2, 3);
    EXPECT_EQ(block.rows(), 2);
    EXPECT_EQ(block.cols(), 3);
    EXPECT_EQ(block(0, 0), 21);
    block(1, 2) = 99;
    EXPECT_EQ(matrix(2, 4), 99);

    auto row = block.row(1);
    auto col = block.col(1);
    EXPECT_EQ(row(0, 2), 99);
    EXPECT_EQ(col(1, 0), matrix(2, 3));

    auto transpose = block.transpose();
    EXPECT_EQ(transpose.rows(), 3);
    EXPECT_EQ(transpose.cols(), 2);
    EXPECT_EQ(transpose(2, 1), 99);
    transpose(0, 1) = -7;
    EXPECT_EQ(matrix(2, 2), -7);
}

TEST(MatrixView, ConstViewCannotExposeMutableReferences)
{
    DenseStorage<float, Device::CPU> matrix(2, 2);
    matrix(1, 0) = 3.5F;
    const DenseStorage<float, Device::CPU>& const_matrix = matrix;
    const auto const_view = const_matrix.view();
    ConstMatrixView<float, Device::CPU> converted = matrix.view();

    static_assert(std::is_same_v<decltype(const_view(0, 0)), const float&>);
    static_assert(std::is_same_v<decltype(converted(0, 0)), const float&>);
    EXPECT_FLOAT_EQ(const_view(1, 0), 3.5F);
    EXPECT_FLOAT_EQ(converted(1, 0), 3.5F);
}

TEST(MatrixView, MapsExternalColumnMajorRowMajorAndPaddedStorage)
{
    std::array<int, 6> column_major{1, 2, 3, 4, 5, 6};
    auto columns = makeColumnMajorView<int, Device::CPU>(column_major.data(), 2, 3);
    EXPECT_EQ(columns(1, 2), 6);

    std::array<int, 6> row_major{1, 2, 3, 4, 5, 6};
    auto rows = makeRowMajorView<int, Device::CPU>(row_major.data(), 2, 3);
    EXPECT_TRUE(rows.isContiguousRowMajor());
    EXPECT_EQ(rows(1, 2), 6);
    rows(0, 1) = 20;
    EXPECT_EQ(row_major[1], 20);

    std::array<int, 12> padded{};
    auto strided = makeMatrixView<int, Device::CPU>(padded.data(), 2, 3, 1, 4);
    strided(1, 2) = 42;
    EXPECT_EQ(padded[9], 42);
    EXPECT_FALSE(strided.isContiguousColumnMajor());
    EXPECT_FALSE(strided.isContiguousRowMajor());
}

TEST(MatrixView, EmptyAndBoundaryBlocksPreserveShapeWithoutPointerArithmetic)
{
    DenseStorage<float, Device::CPU> empty_rows(0, 4);
    auto empty_view = empty_rows.view();
    EXPECT_EQ(empty_view.rows(), 0);
    EXPECT_EQ(empty_view.cols(), 4);
    EXPECT_EQ(empty_view.data(), nullptr);

    DenseStorage<float, Device::CPU> matrix(3, 4);
    auto empty_block = matrix.block(3, 4, 0, 0);
    EXPECT_EQ(empty_block.rows(), 0);
    EXPECT_EQ(empty_block.cols(), 0);
    EXPECT_EQ(empty_block.data(), matrix.data());
}

TEST(MatrixView, RejectsInvalidLayoutsIndicesAndBlocks)
{
    std::array<float, 8> storage{};
    EXPECT_THROW((makeMatrixView<float, Device::CPU>(nullptr, 2, 2, 1, 2)), std::invalid_argument);
    EXPECT_THROW((makeMatrixView<float, Device::CPU>(storage.data(), -1, 2, 1, 2)), std::invalid_argument);
    EXPECT_THROW((makeMatrixView<float, Device::CPU>(storage.data(), 2, 2, 0, 2)), std::invalid_argument);
    EXPECT_THROW((makeMatrixView<float, Device::CPU>(storage.data(), 2, 2, std::numeric_limits<Index>::max(), 1)),
                 std::overflow_error);

    auto view = makeColumnMajorView<float, Device::CPU>(storage.data(), 2, 4);
    EXPECT_THROW(view(-1, 0), std::out_of_range);
    EXPECT_THROW(view(2, 0), std::out_of_range);
    EXPECT_THROW(view.row(2), std::out_of_range);
    EXPECT_THROW(view.col(4), std::out_of_range);
    EXPECT_THROW(view.block(1, 3, 2, 1), std::out_of_range);
}

#ifdef PLAMATRIX_WITH_CUDA
TEST(MatrixView, GpuViewsPreserveDevicePointersAndStridesWithoutHostAccess)
{
    DenseStorage<float, Device::GPU> matrix(4, 5);
    auto block = matrix.block(1, 2, 2, 3);
    EXPECT_EQ(block.data(), matrix.data() + 9);
    EXPECT_EQ(block.rows(), 2);
    EXPECT_EQ(block.cols(), 3);
    EXPECT_EQ(block.rowStride(), 1);
    EXPECT_EQ(block.colStride(), 4);
    EXPECT_EQ(block.device(), Device::GPU);

    const DenseStorage<float, Device::GPU>& const_matrix = matrix;
    auto transpose = const_matrix.transposeView();
    EXPECT_EQ(transpose.data(), matrix.data());
    EXPECT_EQ(transpose.rows(), 5);
    EXPECT_EQ(transpose.cols(), 4);
    EXPECT_EQ(transpose.rowStride(), 4);
    EXPECT_EQ(transpose.colStride(), 1);
}
#endif

} // namespace plamatrix::internal
