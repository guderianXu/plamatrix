#include "gemm_microkernel.h"

#include <algorithm>
#include <immintrin.h>

namespace plamatrix::detail
{

#if defined(__GNUC__) || defined(__clang__)
#define PLAMATRIX_AVX2_TARGET __attribute__((target("avx2,fma")))
#else
#define PLAMATRIX_AVX2_TARGET
#endif

PLAMATRIX_AVX2_TARGET
void packedGemmMicrokernelAvx2(const double* left,
                               const double* packed_right,
                               double* output,
                               Index leading_dimension,
                               Index inner_size,
                               Index packed_columns,
                               Index row_begin,
                               Index row_end,
                               Index column_begin,
                               Index column_end)
{
    Index row = row_begin;
    for (; row + 4 <= row_end; row += 4)
    {
        __m256d accumulators[4] = {
            _mm256_setzero_pd(), _mm256_setzero_pd(),
            _mm256_setzero_pd(), _mm256_setzero_pd()};
        for (Index inner = 0; inner < inner_size; ++inner)
        {
            const __m256d left_values = _mm256_loadu_pd(
                left + inner * leading_dimension + row);
            const double* right = packed_right +
                (column_begin / 4 * inner_size + inner) * 4;
            for (Index column = column_begin; column < column_end; ++column)
            {
                accumulators[column - column_begin] = _mm256_fmadd_pd(
                    left_values,
                    _mm256_set1_pd(right[column - column_begin]),
                    accumulators[column - column_begin]);
            }
        }
        for (Index column = column_begin; column < column_end; ++column)
        {
            _mm256_storeu_pd(
                output + column * leading_dimension + row,
                accumulators[column - column_begin]);
        }
    }
    for (; row < row_end; ++row)
    {
        for (Index column = column_begin; column < column_end; ++column)
        {
            double value = 0.0;
            for (Index inner = 0; inner < inner_size; ++inner)
            {
                value += left[inner * leading_dimension + row] *
                         packed_right[(column_begin / 4 * inner_size + inner) * 4 +
                                      column - column_begin];
            }
            output[column * leading_dimension + row] = value;
        }
    }
}

PLAMATRIX_AVX2_TARGET
void packedGemmMicrokernelAvx2(const float* left,
                               const float* packed_right,
                               float* output,
                               Index leading_dimension,
                               Index inner_size,
                               Index packed_columns,
                               Index row_begin,
                               Index row_end,
                               Index column_begin,
                               Index column_end)
{
    Index row = row_begin;
    for (; row + 8 <= row_end; row += 8)
    {
        __m256 accumulators[4] = {
            _mm256_setzero_ps(), _mm256_setzero_ps(),
            _mm256_setzero_ps(), _mm256_setzero_ps()};
        for (Index inner = 0; inner < inner_size; ++inner)
        {
            const __m256 left_values = _mm256_loadu_ps(
                left + inner * leading_dimension + row);
            const float* right = packed_right +
                (column_begin / 4 * inner_size + inner) * 4;
            for (Index column = column_begin; column < column_end; ++column)
            {
                accumulators[column - column_begin] = _mm256_fmadd_ps(
                    left_values,
                    _mm256_set1_ps(right[column - column_begin]),
                    accumulators[column - column_begin]);
            }
        }
        for (Index column = column_begin; column < column_end; ++column)
        {
            _mm256_storeu_ps(
                output + column * leading_dimension + row,
                accumulators[column - column_begin]);
        }
    }
    for (; row < row_end; ++row)
    {
        for (Index column = column_begin; column < column_end; ++column)
        {
            float value = 0.0F;
            for (Index inner = 0; inner < inner_size; ++inner)
            {
                value += left[inner * leading_dimension + row] *
                         packed_right[(column_begin / 4 * inner_size + inner) * 4 +
                                      column - column_begin];
            }
            output[column * leading_dimension + row] = value;
        }
    }
}

#undef PLAMATRIX_AVX2_TARGET

} // namespace plamatrix::detail
