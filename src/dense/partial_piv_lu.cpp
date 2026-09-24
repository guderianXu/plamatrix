#include "plamatrix/internal/dense/partial_piv_lu.h"

#include "../ops/gemm_microkernel.h"

#ifdef PLAMATRIX_HAVE_AVX2_KERNEL
#include "partial_piv_lu_avx2.h"
#endif

namespace plamatrix::internal::detail
{
    namespace
    {
        template <typename Scalar>
        void scalarTrailingUpdate(Scalar* factors, Index size, Index block_start, Index block_end) noexcept
        {
            Index column = block_end;
            for (; column + 3 < size; column += 4)
            {
                Scalar* first = factors + column * size;
                Scalar* second = first + size;
                Scalar* third = second + size;
                Scalar* fourth = third + size;
                for (Index row = block_start; row < block_end; ++row)
                {
                    Scalar first_value = first[row];
                    Scalar second_value = second[row];
                    Scalar third_value = third[row];
                    Scalar fourth_value = fourth[row];
                    for (Index previous = block_start; previous < row; ++previous)
                    {
                        const Scalar lower = factors[previous * size + row];
                        first_value -= lower * first[previous];
                        second_value -= lower * second[previous];
                        third_value -= lower * third[previous];
                        fourth_value -= lower * fourth[previous];
                    }
                    first[row] = first_value;
                    second[row] = second_value;
                    third[row] = third_value;
                    fourth[row] = fourth_value;
                }
                for (Index previous = block_start; previous < block_end; ++previous)
                {
                    const Scalar* lower_column = factors + previous * size;
                    const Scalar first_pivot = first[previous];
                    const Scalar second_pivot = second[previous];
                    const Scalar third_pivot = third[previous];
                    const Scalar fourth_pivot = fourth[previous];
#pragma omp simd
                    for (Index row = block_end; row < size; ++row)
                    {
                        const Scalar lower = lower_column[row];
                        first[row] -= lower * first_pivot;
                        second[row] -= lower * second_pivot;
                        third[row] -= lower * third_pivot;
                        fourth[row] -= lower * fourth_pivot;
                    }
                }
            }
            for (; column < size; ++column)
            {
                Scalar* trailing_column = factors + column * size;
                for (Index row = block_start; row < block_end; ++row)
                {
                    Scalar value = trailing_column[row];
                    for (Index previous = block_start; previous < row; ++previous)
                        value -= factors[previous * size + row] * trailing_column[previous];
                    trailing_column[row] = value;
                }
                for (Index previous = block_start; previous < block_end; ++previous)
                {
                    const Scalar* lower_column = factors + previous * size;
                    const Scalar pivot_value = trailing_column[previous];
#pragma omp simd
                    for (Index row = block_end; row < size; ++row)
                        trailing_column[row] -= lower_column[row] * pivot_value;
                }
            }
        }

        template <typename Scalar>
        void trailingUpdate(Scalar* factors, Index size, Index block_start, Index block_end) noexcept
        {
#ifdef PLAMATRIX_HAVE_AVX2_KERNEL
            if (cpuSupportsAvx2Fma())
            {
                partialPivLuTrailingUpdateAvx2(factors, size, block_start, block_end);
                return;
            }
#endif
            scalarTrailingUpdate(factors, size, block_start, block_end);
        }
    } // namespace

    void partialPivLuTrailingUpdate(float* factors, Index size, Index block_start, Index block_end) noexcept
    {
        trailingUpdate(factors, size, block_start, block_end);
    }

    void partialPivLuTrailingUpdate(double* factors, Index size, Index block_start, Index block_end) noexcept
    {
        trailingUpdate(factors, size, block_start, block_end);
    }
} // namespace plamatrix::internal::detail
