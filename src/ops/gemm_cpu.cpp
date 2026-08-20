#include <algorithm>
#include <array>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <omp.h>

#include "plamatrix/core/parallel.h"
#include "plamatrix/ops/gemm.h"

#include "gemm_microkernel.h"

namespace plamatrix
{

namespace
{

inline int chooseGemmThreadCount(Index work, Index tile_count)
{
    constexpr Index work_per_thread = Index(2) * 1024 * 1024;
    const int available = std::max(1, omp_get_max_threads());
    const int useful = std::max(1, static_cast<int>(
        (work + work_per_thread - 1) / work_per_thread));
    return std::min({available, useful, std::max(1, static_cast<int>(tile_count)), 16});
}

template <typename Scalar>
void packedScalarMicrokernel(const Scalar* left,
                             const Scalar* packed_right,
                             Scalar* output,
                             Index rows,
                             Index inner_size,
                             Index columns,
                             Index row_begin,
                             Index row_end,
                             Index column_begin,
                             Index column_end)
{
    for (Index row = row_begin; row < row_end; ++row)
    {
        std::array<Scalar, 4> accumulators{};
        for (Index inner = 0; inner < inner_size; ++inner)
        {
            const Scalar left_value = left[inner * rows + row];
            const Scalar* right = packed_right +
                (column_begin / 4 * inner_size + inner) * 4;
            #pragma omp simd
            for (Index column = column_begin; column < column_end; ++column)
            {
                accumulators[static_cast<std::size_t>(column - column_begin)] +=
                    left_value * right[column - column_begin];
            }
        }
        for (Index column = column_begin; column < column_end; ++column)
        {
            output[column * rows + row] =
                accumulators[static_cast<std::size_t>(column - column_begin)];
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
    constexpr Index column_micro_size = 4;
    const Index row_block_size = std::is_same_v<Scalar, float> ? 128 : 96;
    const Index row_block_count = (m + row_block_size - 1) / row_block_size;
    const Index column_block_count = (n + column_micro_size - 1) / column_micro_size;
    const Index tile_count = row_block_count * column_block_count;
    const int thread_count = chooseGemmThreadCount(m * n * k, tile_count);
    std::vector<Scalar> packed_right(static_cast<std::size_t>(
        column_block_count * k * column_micro_size), Scalar(0));
    #pragma omp parallel for schedule(static) num_threads(thread_count) \
        if(detail::shouldUseOpenMp(k * n))
    for (Index column_block = 0; column_block < column_block_count; ++column_block)
    {
        const Index column_begin = column_block * column_micro_size;
        const Index column_end = std::min(column_begin + column_micro_size, n);
        for (Index inner = 0; inner < k; ++inner)
        {
            Scalar* destination = packed_right.data() + static_cast<std::size_t>(
                (column_block * k + inner) * column_micro_size);
            for (Index column = column_begin; column < column_end; ++column)
            {
                destination[column - column_begin] = B_data[column * k + inner];
            }
        }
    }

#ifdef PLAMATRIX_HAVE_AVX2_KERNEL
    static const bool use_avx2 = detail::cpuSupportsAvx2Fma();
#else
    constexpr bool use_avx2 = false;
#endif

    const auto multiply_tile = [&](Index tile)
    {
        const Index column_block = tile / row_block_count;
        const Index row_block = tile % row_block_count;
        const Index row_begin = row_block * row_block_size;
        const Index column_begin = column_block * column_micro_size;
        const Index row_end = std::min(row_begin + row_block_size, m);
        const Index column_end = std::min(column_begin + column_micro_size, n);
#ifdef PLAMATRIX_HAVE_AVX2_KERNEL
        if (use_avx2)
        {
            detail::packedGemmMicrokernelAvx2(
                A_data,
                packed_right.data(),
                C_data,
                m,
                k,
                n,
                row_begin,
                row_end,
                column_begin,
                column_end);
            return;
        }
#endif
        packedScalarMicrokernel(
            A_data,
            packed_right.data(),
            C_data,
            m,
            k,
            n,
            row_begin,
            row_end,
            column_begin,
            column_end);
    };

    if (detail::shouldUseOpenMp(m * n * k) && tile_count > 1)
    {
        #pragma omp parallel for schedule(static) num_threads(thread_count)
        for (Index tile = 0; tile < tile_count; ++tile)
        {
            multiply_tile(tile);
        }
        return;
    }
    for (Index tile = 0; tile < tile_count; ++tile)
    {
        multiply_tile(tile);
    }
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

    DenseMatrix<Scalar, Device::CPU> C(m, n);
    if (m == 0 || n == 0 || k == 0)
    {
        return C;
    }

    const Scalar* A_data = A.data();
    const Scalar* B_data = B.data();
    Scalar* C_data = C.data();

    nativeGemm(A_data, B_data, C_data, m, n, k);

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
