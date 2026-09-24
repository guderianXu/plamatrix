#include "partial_piv_lu_avx2.h"

#include <immintrin.h>
#include <omp.h>

namespace plamatrix::internal::detail
{
    namespace
    {
        template <typename Scalar>
        void
        solvePanelRows(Scalar* column, const Scalar* factors, Index size, Index block_start, Index block_end) noexcept
        {
            for (Index row = block_start; row < block_end; ++row)
            {
                Scalar value = column[row];
                for (Index previous = block_start; previous < row; ++previous)
                    value -= factors[previous * size + row] * column[previous];
                column[row] = value;
            }
        }

        void
        updateColumn(double* column, const double* lower_column, double pivot, Index row_begin, Index row_end) noexcept
        {
            const __m256d pivot_packet = _mm256_set1_pd(pivot);
            Index row = row_begin;
            for (; row + 3 < row_end; row += 4)
            {
                const __m256d lower = _mm256_loadu_pd(lower_column + row);
                const __m256d values = _mm256_loadu_pd(column + row);
                _mm256_storeu_pd(column + row, _mm256_fnmadd_pd(lower, pivot_packet, values));
            }
            for (; row < row_end; ++row)
                column[row] -= lower_column[row] * pivot;
        }

        void
        updateColumn(float* column, const float* lower_column, float pivot, Index row_begin, Index row_end) noexcept
        {
            const __m256 pivot_packet = _mm256_set1_ps(pivot);
            Index row = row_begin;
            for (; row + 7 < row_end; row += 8)
            {
                const __m256 lower = _mm256_loadu_ps(lower_column + row);
                const __m256 values = _mm256_loadu_ps(column + row);
                _mm256_storeu_ps(column + row, _mm256_fnmadd_ps(lower, pivot_packet, values));
            }
            for (; row < row_end; ++row)
                column[row] -= lower_column[row] * pivot;
        }

        void updateFourColumns(double* first,
                               double* second,
                               double* third,
                               double* fourth,
                               const double* lower_column,
                               Index pivot_row,
                               Index row_begin,
                               Index row_end) noexcept
        {
            const __m256d first_pivot = _mm256_set1_pd(first[pivot_row]);
            const __m256d second_pivot = _mm256_set1_pd(second[pivot_row]);
            const __m256d third_pivot = _mm256_set1_pd(third[pivot_row]);
            const __m256d fourth_pivot = _mm256_set1_pd(fourth[pivot_row]);
            Index row = row_begin;
            for (; row + 3 < row_end; row += 4)
            {
                const __m256d lower = _mm256_loadu_pd(lower_column + row);
                _mm256_storeu_pd(first + row, _mm256_fnmadd_pd(lower, first_pivot, _mm256_loadu_pd(first + row)));
                _mm256_storeu_pd(second + row, _mm256_fnmadd_pd(lower, second_pivot, _mm256_loadu_pd(second + row)));
                _mm256_storeu_pd(third + row, _mm256_fnmadd_pd(lower, third_pivot, _mm256_loadu_pd(third + row)));
                _mm256_storeu_pd(fourth + row, _mm256_fnmadd_pd(lower, fourth_pivot, _mm256_loadu_pd(fourth + row)));
            }
            for (; row < row_end; ++row)
            {
                const double lower = lower_column[row];
                first[row] -= lower * first[pivot_row];
                second[row] -= lower * second[pivot_row];
                third[row] -= lower * third[pivot_row];
                fourth[row] -= lower * fourth[pivot_row];
            }
        }

        void updateFourColumns(float* first,
                               float* second,
                               float* third,
                               float* fourth,
                               const float* lower_column,
                               Index pivot_row,
                               Index row_begin,
                               Index row_end) noexcept
        {
            const __m256 first_pivot = _mm256_set1_ps(first[pivot_row]);
            const __m256 second_pivot = _mm256_set1_ps(second[pivot_row]);
            const __m256 third_pivot = _mm256_set1_ps(third[pivot_row]);
            const __m256 fourth_pivot = _mm256_set1_ps(fourth[pivot_row]);
            Index row = row_begin;
            for (; row + 7 < row_end; row += 8)
            {
                const __m256 lower = _mm256_loadu_ps(lower_column + row);
                _mm256_storeu_ps(first + row, _mm256_fnmadd_ps(lower, first_pivot, _mm256_loadu_ps(first + row)));
                _mm256_storeu_ps(second + row, _mm256_fnmadd_ps(lower, second_pivot, _mm256_loadu_ps(second + row)));
                _mm256_storeu_ps(third + row, _mm256_fnmadd_ps(lower, third_pivot, _mm256_loadu_ps(third + row)));
                _mm256_storeu_ps(fourth + row, _mm256_fnmadd_ps(lower, fourth_pivot, _mm256_loadu_ps(fourth + row)));
            }
            for (; row < row_end; ++row)
            {
                const float lower = lower_column[row];
                first[row] -= lower * first[pivot_row];
                second[row] -= lower * second[pivot_row];
                third[row] -= lower * third[pivot_row];
                fourth[row] -= lower * fourth[pivot_row];
            }
        }

        template <typename Scalar>
        void updateTrailing(Scalar* factors, Index size, Index block_start, Index block_end) noexcept
        {
            const Index group_count = (size - block_end) / 4;
            const bool parallel =
                size >= 512 && group_count >= 8 && omp_get_max_threads() > 1 && omp_in_parallel() == 0;
#pragma omp parallel for schedule(static) if (parallel)
            for (Index group = 0; group < group_count; ++group)
            {
                const Index column = block_end + group * 4;
                Scalar* first = factors + column * size;
                Scalar* second = first + size;
                Scalar* third = second + size;
                Scalar* fourth = third + size;
                solvePanelRows(first, factors, size, block_start, block_end);
                solvePanelRows(second, factors, size, block_start, block_end);
                solvePanelRows(third, factors, size, block_start, block_end);
                solvePanelRows(fourth, factors, size, block_start, block_end);
                for (Index previous = block_start; previous < block_end; ++previous)
                {
                    updateFourColumns(
                        first, second, third, fourth, factors + previous * size, previous, block_end, size);
                }
            }
            for (Index column = block_end + group_count * 4; column < size; ++column)
            {
                Scalar* trailing_column = factors + column * size;
                solvePanelRows(trailing_column, factors, size, block_start, block_end);
                for (Index previous = block_start; previous < block_end; ++previous)
                {
                    updateColumn(
                        trailing_column, factors + previous * size, trailing_column[previous], block_end, size);
                }
            }
        }
    } // namespace

    void partialPivLuTrailingUpdateAvx2(float* factors, Index size, Index block_start, Index block_end) noexcept
    {
        updateTrailing(factors, size, block_start, block_end);
    }

    void partialPivLuTrailingUpdateAvx2(double* factors, Index size, Index block_start, Index block_end) noexcept
    {
        updateTrailing(factors, size, block_start, block_end);
    }
} // namespace plamatrix::internal::detail
