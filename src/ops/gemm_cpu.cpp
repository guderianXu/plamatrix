#include <algorithm>
#include <sstream>
#include <stdexcept>

#include <omp.h>

#include "plamatrix/core/parallel.h"
#include "plamatrix/ops/gemm.h"

#ifdef PLAMATRIX_WITH_BLAS
#include "fortran_linalg.h"
#endif

namespace plamatrix
{

namespace
{

constexpr Index kRowBlockSize = 128;
constexpr Index kColumnBlockSize = 32;
constexpr Index kInnerBlockSize = 64;

template <typename Scalar>
void multiplyTile(const Scalar* A_data,
                  const Scalar* B_data,
                  Scalar* C_data,
                  Index m,
                  Index k,
                  Index row_begin,
                  Index row_end,
                  Index column_begin,
                  Index column_end)
{
    for (Index inner_begin = 0; inner_begin < k; inner_begin += kInnerBlockSize)
    {
        const Index inner_end = std::min(inner_begin + kInnerBlockSize, k);
        for (Index column = column_begin; column < column_end; ++column)
        {
            Scalar* output = C_data + column * m;
            const Scalar* right = B_data + column * k;
            for (Index inner = inner_begin; inner < inner_end; ++inner)
            {
                const Scalar right_value = right[inner];
                const Scalar* left = A_data + inner * m;
                #pragma omp simd
                for (Index row = row_begin; row < row_end; ++row)
                {
                    output[row] += left[row] * right_value;
                }
            }
        }
    }
}

template <typename Scalar>
void nativeGemm(const Scalar* A_data,
                const Scalar* B_data,
                Scalar* C_data,
                Index m,
                Index n,
                Index k)
{
    const Index row_block_count = (m + kRowBlockSize - 1) / kRowBlockSize;
    const Index column_block_count = (n + kColumnBlockSize - 1) / kColumnBlockSize;
    const Index tile_count = row_block_count * column_block_count;

    if (detail::shouldUseOpenMp(m * n * k) && tile_count > 1)
    {
        #pragma omp parallel for
        for (Index tile = 0; tile < tile_count; ++tile)
        {
            const Index column_block = tile / row_block_count;
            const Index row_block = tile % row_block_count;
            const Index row_begin = row_block * kRowBlockSize;
            const Index column_begin = column_block * kColumnBlockSize;
            multiplyTile(A_data, B_data, C_data, m, k,
                         row_begin, std::min(row_begin + kRowBlockSize, m),
                         column_begin, std::min(column_begin + kColumnBlockSize, n));
        }
        return;
    }

    multiplyTile(A_data, B_data, C_data, m, k, 0, m, 0, n);
}

} // anonymous namespace

template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> gemm(const DenseMatrix<Scalar, Device::CPU>& A,
                                       const DenseMatrix<Scalar, Device::CPU>& B)
{
    Index m = A.rows();
    Index k = A.cols();
    Index n = B.cols();

    if (k != B.rows())
    {
        std::ostringstream oss;
        oss << "GEMM dimension mismatch: A is " << m << "x" << k
            << ", B is " << B.rows() << "x" << n;
        throw std::runtime_error(oss.str());
    }

#ifdef PLAMATRIX_WITH_BLAS
    int m_int = detail::checkedLapackInt(m, "GEMM m");
    int n_int = detail::checkedLapackInt(n, "GEMM n");
    int k_int = detail::checkedLapackInt(k, "GEMM k");
#endif

    DenseMatrix<Scalar, Device::CPU> C(m, n);
    if (m == 0 || n == 0 || k == 0)
    {
        return C;
    }

    const Scalar* A_data = A.data();
    const Scalar* B_data = B.data();
    Scalar* C_data = C.data();

#ifdef PLAMATRIX_WITH_BLAS
    detail::fortranGemm(m_int, n_int, k_int, A_data, B_data, C_data);
#else
    nativeGemm(A_data, B_data, C_data, m, n, k);
#endif

    return C;
}

// Explicit template instantiations
#ifdef PLAMATRIX_USE_FLOAT
template DenseMatrix<float, Device::CPU> gemm(const DenseMatrix<float, Device::CPU>&,
                                                const DenseMatrix<float, Device::CPU>&);
#endif

#ifdef PLAMATRIX_USE_DOUBLE
template DenseMatrix<double, Device::CPU> gemm(const DenseMatrix<double, Device::CPU>&,
                                                 const DenseMatrix<double, Device::CPU>&);
#endif

} // namespace plamatrix
