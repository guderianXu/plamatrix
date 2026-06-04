#include <gtest/gtest.h>

#include <plamatrix/sparse/coo_matrix.h>
#include <plamatrix/sparse/csr_matrix.h>

using namespace plamatrix;

TEST(COOMatrix, construction)
{
    COOMatrix<float, Device::CPU> mat(4, 4);
    EXPECT_EQ(mat.rows(), 4);
    EXPECT_EQ(mat.cols(), 4);
    EXPECT_EQ(mat.nnz(), 0);
}

TEST(COOMatrix, construction_RejectsNegativeDimensions)
{
    EXPECT_THROW((COOMatrix<float, Device::CPU>(-1, 4)), std::invalid_argument);
    EXPECT_THROW((COOMatrix<float, Device::CPU>(4, -1)), std::invalid_argument);
}

TEST(COOMatrix, addAndBuild)
{
    COOMatrix<float, Device::CPU> mat(4, 4);
    mat.add(0, 0, 1.0f);
    mat.add(1, 1, 2.0f);
    mat.add(2, 2, 3.0f);
    mat.add(0, 1, 4.0f);
    mat.add(3, 3, 5.0f);

    EXPECT_EQ(mat.nnz(), 5);
}

TEST(COOMatrix, add_RejectsOutOfBoundsTriplet)
{
    COOMatrix<float, Device::CPU> mat(2, 3);

    EXPECT_THROW(mat.add(-1, 0, 1.0f), std::out_of_range);
    EXPECT_THROW(mat.add(0, -1, 1.0f), std::out_of_range);
    EXPECT_THROW(mat.add(2, 0, 1.0f), std::out_of_range);
    EXPECT_THROW(mat.add(0, 3, 1.0f), std::out_of_range);
}

TEST(COOMatrix, toCsr_Cpu)
{
    COOMatrix<float, Device::CPU> mat(4, 4);
    mat.add(0, 0, 1.0f);
    mat.add(1, 1, 2.0f);
    mat.add(2, 2, 3.0f);
    mat.add(0, 1, 4.0f);
    mat.add(3, 3, 5.0f);

    auto csr = mat.toCsr();

    EXPECT_EQ(csr.rows(), 4);
    EXPECT_EQ(csr.cols(), 4);
    EXPECT_EQ(csr.nnz(), 5);

    // Sorted by (row, col): (0,0)=1.0, (0,1)=4.0, (1,1)=2.0, (2,2)=3.0, (3,3)=5.0
    EXPECT_FLOAT_EQ(csr.values()[0], 1.0f);
    EXPECT_EQ(csr.colIndices()[0], 0);
    EXPECT_FLOAT_EQ(csr.values()[1], 4.0f);
    EXPECT_EQ(csr.colIndices()[1], 1);
    EXPECT_FLOAT_EQ(csr.values()[2], 2.0f);
    EXPECT_EQ(csr.colIndices()[2], 1);
    EXPECT_FLOAT_EQ(csr.values()[3], 3.0f);
    EXPECT_EQ(csr.colIndices()[3], 2);
    EXPECT_FLOAT_EQ(csr.values()[4], 5.0f);
    EXPECT_EQ(csr.colIndices()[4], 3);

    // row_offsets: [0, 2, 3, 4, 5]
    EXPECT_EQ(csr.rowOffsets()[0], 0);
    EXPECT_EQ(csr.rowOffsets()[1], 2);
    EXPECT_EQ(csr.rowOffsets()[2], 3);
    EXPECT_EQ(csr.rowOffsets()[3], 4);
    EXPECT_EQ(csr.rowOffsets()[4], 5);
}

TEST(COOMatrix, toCsr_Gpu)
{
    COOMatrix<float, Device::GPU> mat(3, 3);
    mat.add(0, 0, 1.0f);
    mat.add(1, 1, 2.0f);
    mat.add(2, 2, 3.0f);
    mat.add(0, 2, 4.0f);

    auto csr = mat.toCsr();

    EXPECT_EQ(csr.rows(), 3);
    EXPECT_EQ(csr.cols(), 3);
    EXPECT_EQ(csr.nnz(), 4);
    EXPECT_EQ(csr.device(), Device::GPU);
    EXPECT_NE(csr.values(), nullptr);
    EXPECT_NE(csr.colIndices(), nullptr);
    EXPECT_NE(csr.rowOffsets(), nullptr);

    // Copy CSR arrays back to CPU for verification
    std::vector<float> host_values(4);
    std::vector<Index> host_col_indices(4);
    std::vector<Index> host_row_offsets(4);

    PLAMATRIX_CHECK_CUDA(cudaMemcpy(host_values.data(), csr.values(), 4 * sizeof(float),
                                    cudaMemcpyDeviceToHost));
    PLAMATRIX_CHECK_CUDA(cudaMemcpy(host_col_indices.data(), csr.colIndices(), 4 * sizeof(Index),
                                    cudaMemcpyDeviceToHost));
    PLAMATRIX_CHECK_CUDA(cudaMemcpy(host_row_offsets.data(), csr.rowOffsets(), 4 * sizeof(Index),
                                    cudaMemcpyDeviceToHost));

    // Sorted: (0,0)=1.0, (0,2)=4.0, (1,1)=2.0, (2,2)=3.0
    EXPECT_FLOAT_EQ(host_values[0], 1.0f);
    EXPECT_EQ(host_col_indices[0], 0);
    EXPECT_FLOAT_EQ(host_values[1], 4.0f);
    EXPECT_EQ(host_col_indices[1], 2);
    EXPECT_FLOAT_EQ(host_values[2], 2.0f);
    EXPECT_EQ(host_col_indices[2], 1);
    EXPECT_FLOAT_EQ(host_values[3], 3.0f);
    EXPECT_EQ(host_col_indices[3], 2);

    EXPECT_EQ(host_row_offsets[0], 0);
    EXPECT_EQ(host_row_offsets[1], 2);  // row 0 has 2 NZ
    EXPECT_EQ(host_row_offsets[2], 3);  // row 1 has 1 NZ
    EXPECT_EQ(host_row_offsets[3], 4);  // row 2 has 1 NZ
}
