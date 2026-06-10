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
