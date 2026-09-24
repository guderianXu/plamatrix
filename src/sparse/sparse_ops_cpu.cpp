#include "plamatrix/internal/sparse/sparse_ops.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace plamatrix::internal
{

namespace
{

template <typename Scalar>
struct CombinedCoo
{
    std::vector<Index> row_indices;
    std::vector<Index> col_indices;
    std::vector<Scalar> values;
};

template <typename Scalar>
CombinedCoo<Scalar> combineCoo(Index rows,
                               Index cols,
                               const std::vector<Index>& row_indices,
                               const std::vector<Index>& col_indices,
                               const std::vector<Scalar>& values)
{
    if (rows < 0 || cols < 0)
    {
        throw std::invalid_argument("cooToCsr dimensions must be non-negative");
    }
    if (row_indices.size() != col_indices.size() || row_indices.size() != values.size())
    {
        throw std::invalid_argument("cooToCsr row, column, and value arrays must have equal lengths");
    }
    if (row_indices.size() > static_cast<std::size_t>(std::numeric_limits<Index>::max()))
    {
        throw std::overflow_error("cooToCsr triplet count exceeds Index range");
    }

    const Index count = static_cast<Index>(row_indices.size());
    for (Index position = 0; position < count; ++position)
    {
        const Index row = row_indices[static_cast<std::size_t>(position)];
        const Index col = col_indices[static_cast<std::size_t>(position)];
        if (row < 0 || row >= rows || col < 0 || col >= cols)
        {
            std::ostringstream oss;
            oss << "cooToCsr coordinate out of range at triplet " << position
                << ": (" << row << ", " << col << ") for matrix " << rows << "x" << cols;
            throw std::out_of_range(oss.str());
        }
    }

    std::vector<Index> permutation(static_cast<std::size_t>(count));
    std::iota(permutation.begin(), permutation.end(), Index{0});
    std::stable_sort(permutation.begin(), permutation.end(), [&](Index left, Index right) {
        const auto left_position = static_cast<std::size_t>(left);
        const auto right_position = static_cast<std::size_t>(right);
        if (row_indices[left_position] != row_indices[right_position])
        {
            return row_indices[left_position] < row_indices[right_position];
        }
        return col_indices[left_position] < col_indices[right_position];
    });

    CombinedCoo<Scalar> combined;
    combined.row_indices.reserve(static_cast<std::size_t>(count));
    combined.col_indices.reserve(static_cast<std::size_t>(count));
    combined.values.reserve(static_cast<std::size_t>(count));

    for (Index source : permutation)
    {
        const auto source_position = static_cast<std::size_t>(source);
        const Index row = row_indices[source_position];
        const Index col = col_indices[source_position];
        if (!combined.values.empty() && combined.row_indices.back() == row
            && combined.col_indices.back() == col)
        {
            combined.values.back() += values[source_position];
        }
        else
        {
            combined.row_indices.push_back(row);
            combined.col_indices.push_back(col);
            combined.values.push_back(values[source_position]);
        }
    }

    return combined;
}

template <typename Scalar>
void writeCombinedCoo(Index rows,
                      Index cols,
                      const CombinedCoo<Scalar>& combined,
                      CsrStorage<Scalar, Device::CPU>& output)
{
    const Index combined_count = static_cast<Index>(combined.values.size());
    if (output.rows() != rows || output.cols() != cols)
    {
        std::ostringstream oss;
        oss << "cooToCsr output dimension mismatch: output is " << output.rows() << "x"
            << output.cols() << ", expected " << rows << "x" << cols;
        throw std::runtime_error(oss.str());
    }
    if (output.nnz() != combined_count)
    {
        std::ostringstream oss;
        oss << "cooToCsr output nnz mismatch: output has " << output.nnz()
            << ", expected " << combined_count;
        throw std::runtime_error(oss.str());
    }

    std::fill_n(output.rowOffsets(), static_cast<std::size_t>(rows) + 1, Index{0});
    for (Index row : combined.row_indices)
    {
        ++output.rowOffsets()[row + 1];
    }
    for (Index row = 0; row < rows; ++row)
    {
        output.rowOffsets()[row + 1] += output.rowOffsets()[row];
    }

    for (Index position = 0; position < combined_count; ++position)
    {
        const auto source_position = static_cast<std::size_t>(position);
        output.colIndices()[position] = combined.col_indices[source_position];
        output.values()[position] = combined.values[source_position];
    }
}

template <typename Scalar>
void checkSpmvDimensions(const CsrStorage<Scalar, Device::CPU>& csr,
                         const DenseStorage<Scalar, Device::CPU>& x,
                         const DenseStorage<Scalar, Device::CPU>& output)
{
    if (x.rows() != csr.cols() || x.cols() != 1)
    {
        std::ostringstream oss;
        oss << "spmv dimension mismatch: CSR is " << csr.rows() << "x" << csr.cols()
            << ", x is " << x.rows() << "x" << x.cols() << ", expected " << csr.cols() << "x1";
        throw std::runtime_error(oss.str());
    }
    if (output.rows() != csr.rows() || output.cols() != 1)
    {
        std::ostringstream oss;
        oss << "spmv output dimension mismatch: output is " << output.rows() << "x" << output.cols()
            << ", expected " << csr.rows() << "x1";
        throw std::runtime_error(oss.str());
    }
}

template <typename Scalar>
void checkSpmmDimensions(const CsrStorage<Scalar, Device::CPU>& csr,
                         const DenseStorage<Scalar, Device::CPU>& B,
                         const DenseStorage<Scalar, Device::CPU>& output)
{
    if (B.rows() != csr.cols())
    {
        std::ostringstream oss;
        oss << "spmm dimension mismatch: CSR is " << csr.rows() << "x" << csr.cols()
            << ", B is " << B.rows() << "x" << B.cols();
        throw std::runtime_error(oss.str());
    }
    if (output.rows() != csr.rows() || output.cols() != B.cols())
    {
        std::ostringstream oss;
        oss << "spmm output dimension mismatch: output is " << output.rows() << "x" << output.cols()
            << ", expected " << csr.rows() << "x" << B.cols();
        throw std::runtime_error(oss.str());
    }
}

} // namespace

template <typename Scalar>
CsrStorage<Scalar, Device::CPU> cooToCsr(Index rows,
                                        Index cols,
                                        const std::vector<Index>& row_indices,
                                        const std::vector<Index>& col_indices,
                                        const std::vector<Scalar>& values)
{
    auto combined = combineCoo(rows, cols, row_indices, col_indices, values);
    CsrStorage<Scalar, Device::CPU> output(rows, cols, static_cast<Index>(combined.values.size()));
    writeCombinedCoo(rows, cols, combined, output);
    return output;
}

template <typename Scalar>
void cooToCsr(Index rows,
              Index cols,
              const std::vector<Index>& row_indices,
              const std::vector<Index>& col_indices,
              const std::vector<Scalar>& values,
              CsrStorage<Scalar, Device::CPU>& output)
{
    const auto combined = combineCoo(rows, cols, row_indices, col_indices, values);
    writeCombinedCoo(rows, cols, combined, output);
}

template <typename Scalar>
DenseStorage<Scalar, Device::CPU> spmv(const CsrStorage<Scalar, Device::CPU>& csr,
                                      const DenseStorage<Scalar, Device::CPU>& x)
{
    DenseStorage<Scalar, Device::CPU> output(csr.rows(), 1);
    spmv(csr, x, output);
    return output;
}

template <typename Scalar>
void spmv(const CsrStorage<Scalar, Device::CPU>& csr,
          const DenseStorage<Scalar, Device::CPU>& x,
          DenseStorage<Scalar, Device::CPU>& output)
{
    checkSpmvDimensions(csr, x, output);
    if (x.data() != nullptr && x.data() == output.data())
    {
        throw std::invalid_argument("spmv input and output data must not alias");
    }
    for (Index row = 0; row < csr.rows(); ++row)
    {
        Scalar sum = Scalar{0};
        for (Index position = csr.rowOffsets()[row]; position < csr.rowOffsets()[row + 1]; ++position)
        {
            sum += csr.values()[position] * x.data()[csr.colIndices()[position]];
        }
        output.data()[row] = sum;
    }
}

template <typename Scalar>
DenseStorage<Scalar, Device::CPU> spmm(const CsrStorage<Scalar, Device::CPU>& csr,
                                      const DenseStorage<Scalar, Device::CPU>& B)
{
    DenseStorage<Scalar, Device::CPU> output(csr.rows(), B.cols());
    spmm(csr, B, output);
    return output;
}

template <typename Scalar>
void spmm(const CsrStorage<Scalar, Device::CPU>& csr,
          const DenseStorage<Scalar, Device::CPU>& B,
          DenseStorage<Scalar, Device::CPU>& output)
{
    checkSpmmDimensions(csr, B, output);
    if (B.data() != nullptr && B.data() == output.data())
    {
        throw std::invalid_argument("spmm input and output data must not alias");
    }
    for (Index col = 0; col < B.cols(); ++col)
    {
        const Scalar* b_column = B.data() + col * B.rows();
        Scalar* output_column = output.data() + col * output.rows();
        for (Index row = 0; row < csr.rows(); ++row)
        {
            Scalar sum = Scalar{0};
            for (Index position = csr.rowOffsets()[row]; position < csr.rowOffsets()[row + 1]; ++position)
            {
                sum += csr.values()[position] * b_column[csr.colIndices()[position]];
            }
            output_column[row] = sum;
        }
    }
}

#ifdef PLAMATRIX_NO_CUDA

namespace
{

[[noreturn]] void throwGpuSparseUnavailable(const char* operation)
{
    throw std::runtime_error(
        std::string(operation) + " requires PLAMATRIX_WITH_CUDA=ON");
}

} // anonymous namespace

template <typename Scalar>
CsrStorage<Scalar, Device::GPU> cooToCsr(
    Index,
    Index,
    const DenseStorage<Index, Device::GPU>&,
    const DenseStorage<Index, Device::GPU>&,
    const DenseStorage<Scalar, Device::GPU>&,
    SparseOpsWorkspace&)
{
    throwGpuSparseUnavailable("cooToCsr GPU");
}

template <typename Scalar>
void cooToCsrAsync(
    Index,
    Index,
    const DenseStorage<Index, Device::GPU>&,
    const DenseStorage<Index, Device::GPU>&,
    const DenseStorage<Scalar, Device::GPU>&,
    CsrStorage<Scalar, Device::GPU>&,
    SparseOpsWorkspace&,
    cudaStream_t)
{
    throwGpuSparseUnavailable("cooToCsrAsync");
}

template <typename Scalar>
DenseStorage<Scalar, Device::GPU> spmv(
    const CsrStorage<Scalar, Device::GPU>&,
    const DenseStorage<Scalar, Device::GPU>&)
{
    throwGpuSparseUnavailable("spmv GPU");
}

template <typename Scalar>
void spmv(const CsrStorage<Scalar, Device::GPU>&,
          const DenseStorage<Scalar, Device::GPU>&,
          DenseStorage<Scalar, Device::GPU>&)
{
    throwGpuSparseUnavailable("spmv GPU");
}

template <typename Scalar>
void spmvAsync(const CsrStorage<Scalar, Device::GPU>&,
               const DenseStorage<Scalar, Device::GPU>&,
               DenseStorage<Scalar, Device::GPU>&,
               SparseOpsWorkspace&,
               cudaStream_t)
{
    throwGpuSparseUnavailable("spmvAsync");
}

template <typename Scalar>
DenseStorage<Scalar, Device::GPU> spmm(
    const CsrStorage<Scalar, Device::GPU>&,
    const DenseStorage<Scalar, Device::GPU>&)
{
    throwGpuSparseUnavailable("spmm GPU");
}

template <typename Scalar>
void spmm(const CsrStorage<Scalar, Device::GPU>&,
          const DenseStorage<Scalar, Device::GPU>&,
          DenseStorage<Scalar, Device::GPU>&)
{
    throwGpuSparseUnavailable("spmm GPU");
}

template <typename Scalar>
void spmmAsync(const CsrStorage<Scalar, Device::GPU>&,
               const DenseStorage<Scalar, Device::GPU>&,
               DenseStorage<Scalar, Device::GPU>&,
               SparseOpsWorkspace&,
               cudaStream_t)
{
    throwGpuSparseUnavailable("spmmAsync");
}

#endif

#ifdef PLAMATRIX_USE_FLOAT
template CsrStorage<float, Device::CPU> cooToCsr<float>(
    Index,
    Index,
    const std::vector<Index>&,
    const std::vector<Index>&,
    const std::vector<float>&);
template void cooToCsr<float>(Index,
                              Index,
                              const std::vector<Index>&,
                              const std::vector<Index>&,
                              const std::vector<float>&,
                              CsrStorage<float, Device::CPU>&);
template DenseStorage<float, Device::CPU> spmv<float>(
    const CsrStorage<float, Device::CPU>&,
    const DenseStorage<float, Device::CPU>&);
template void spmv<float>(const CsrStorage<float, Device::CPU>&,
                          const DenseStorage<float, Device::CPU>&,
                          DenseStorage<float, Device::CPU>&);
template DenseStorage<float, Device::CPU> spmm<float>(
    const CsrStorage<float, Device::CPU>&,
    const DenseStorage<float, Device::CPU>&);
template void spmm<float>(const CsrStorage<float, Device::CPU>&,
                          const DenseStorage<float, Device::CPU>&,
                          DenseStorage<float, Device::CPU>&);
#ifdef PLAMATRIX_NO_CUDA
template CsrStorage<float, Device::GPU> cooToCsr<float>(
    Index, Index, const DenseStorage<Index, Device::GPU>&,
    const DenseStorage<Index, Device::GPU>&, const DenseStorage<float, Device::GPU>&,
    SparseOpsWorkspace&);
template void cooToCsrAsync<float>(
    Index, Index, const DenseStorage<Index, Device::GPU>&,
    const DenseStorage<Index, Device::GPU>&, const DenseStorage<float, Device::GPU>&,
    CsrStorage<float, Device::GPU>&, SparseOpsWorkspace&, cudaStream_t);
template DenseStorage<float, Device::GPU> spmv<float>(
    const CsrStorage<float, Device::GPU>&,
    const DenseStorage<float, Device::GPU>&);
template void spmv<float>(const CsrStorage<float, Device::GPU>&,
                          const DenseStorage<float, Device::GPU>&,
                          DenseStorage<float, Device::GPU>&);
template void spmvAsync<float>(const CsrStorage<float, Device::GPU>&,
                               const DenseStorage<float, Device::GPU>&,
                               DenseStorage<float, Device::GPU>&,
                               SparseOpsWorkspace&,
                               cudaStream_t);
template DenseStorage<float, Device::GPU> spmm<float>(
    const CsrStorage<float, Device::GPU>&,
    const DenseStorage<float, Device::GPU>&);
template void spmm<float>(const CsrStorage<float, Device::GPU>&,
                          const DenseStorage<float, Device::GPU>&,
                          DenseStorage<float, Device::GPU>&);
template void spmmAsync<float>(const CsrStorage<float, Device::GPU>&,
                               const DenseStorage<float, Device::GPU>&,
                               DenseStorage<float, Device::GPU>&,
                               SparseOpsWorkspace&,
                               cudaStream_t);
#endif
#endif

#ifdef PLAMATRIX_USE_DOUBLE
template CsrStorage<double, Device::CPU> cooToCsr<double>(
    Index,
    Index,
    const std::vector<Index>&,
    const std::vector<Index>&,
    const std::vector<double>&);
template void cooToCsr<double>(Index,
                               Index,
                               const std::vector<Index>&,
                               const std::vector<Index>&,
                               const std::vector<double>&,
                               CsrStorage<double, Device::CPU>&);
template DenseStorage<double, Device::CPU> spmv<double>(
    const CsrStorage<double, Device::CPU>&,
    const DenseStorage<double, Device::CPU>&);
template void spmv<double>(const CsrStorage<double, Device::CPU>&,
                           const DenseStorage<double, Device::CPU>&,
                           DenseStorage<double, Device::CPU>&);
template DenseStorage<double, Device::CPU> spmm<double>(
    const CsrStorage<double, Device::CPU>&,
    const DenseStorage<double, Device::CPU>&);
template void spmm<double>(const CsrStorage<double, Device::CPU>&,
                           const DenseStorage<double, Device::CPU>&,
                           DenseStorage<double, Device::CPU>&);
#ifdef PLAMATRIX_NO_CUDA
template CsrStorage<double, Device::GPU> cooToCsr<double>(
    Index, Index, const DenseStorage<Index, Device::GPU>&,
    const DenseStorage<Index, Device::GPU>&, const DenseStorage<double, Device::GPU>&,
    SparseOpsWorkspace&);
template void cooToCsrAsync<double>(
    Index, Index, const DenseStorage<Index, Device::GPU>&,
    const DenseStorage<Index, Device::GPU>&, const DenseStorage<double, Device::GPU>&,
    CsrStorage<double, Device::GPU>&, SparseOpsWorkspace&, cudaStream_t);
template DenseStorage<double, Device::GPU> spmv<double>(
    const CsrStorage<double, Device::GPU>&,
    const DenseStorage<double, Device::GPU>&);
template void spmv<double>(const CsrStorage<double, Device::GPU>&,
                           const DenseStorage<double, Device::GPU>&,
                           DenseStorage<double, Device::GPU>&);
template void spmvAsync<double>(const CsrStorage<double, Device::GPU>&,
                                const DenseStorage<double, Device::GPU>&,
                                DenseStorage<double, Device::GPU>&,
                                SparseOpsWorkspace&,
                                cudaStream_t);
template DenseStorage<double, Device::GPU> spmm<double>(
    const CsrStorage<double, Device::GPU>&,
    const DenseStorage<double, Device::GPU>&);
template void spmm<double>(const CsrStorage<double, Device::GPU>&,
                           const DenseStorage<double, Device::GPU>&,
                           DenseStorage<double, Device::GPU>&);
template void spmmAsync<double>(const CsrStorage<double, Device::GPU>&,
                                const DenseStorage<double, Device::GPU>&,
                                DenseStorage<double, Device::GPU>&,
                                SparseOpsWorkspace&,
                                cudaStream_t);
#endif
#endif

} // namespace plamatrix::internal
