#include "packet_reduction_avx2.h"

#include <immintrin.h>

namespace plamatrix::internal::detail
{
    namespace
    {
        float horizontalSum(__m256 value) noexcept
        {
            const __m128 low = _mm256_castps256_ps128(value);
            const __m128 high = _mm256_extractf128_ps(value, 1);
            __m128 sum = _mm_add_ps(low, high);
            sum = _mm_hadd_ps(sum, sum);
            sum = _mm_hadd_ps(sum, sum);
            return _mm_cvtss_f32(sum);
        }

        double horizontalSum(__m256d value) noexcept
        {
            const __m128d low = _mm256_castpd256_pd128(value);
            const __m128d high = _mm256_extractf128_pd(value, 1);
            const __m128d sum = _mm_add_pd(low, high);
            const __m128d high_lane = _mm_unpackhi_pd(sum, sum);
            return _mm_cvtsd_f64(_mm_add_sd(sum, high_lane));
        }
    } // namespace

    float packetSumAvx2(const float* values, Index count) noexcept
    {
        __m256 first = _mm256_setzero_ps();
        __m256 second = _mm256_setzero_ps();
        __m256 third = _mm256_setzero_ps();
        __m256 fourth = _mm256_setzero_ps();
        Index index = 0;
        for (; index + 31 < count; index += 32)
        {
            first = _mm256_add_ps(first, _mm256_loadu_ps(values + index));
            second = _mm256_add_ps(second, _mm256_loadu_ps(values + index + 8));
            third = _mm256_add_ps(third, _mm256_loadu_ps(values + index + 16));
            fourth = _mm256_add_ps(fourth, _mm256_loadu_ps(values + index + 24));
        }
        first = _mm256_add_ps(first, second);
        third = _mm256_add_ps(third, fourth);
        float result = horizontalSum(_mm256_add_ps(first, third));
        for (; index < count; ++index)
        {
            result += values[index];
        }
        return result;
    }

    double packetSumAvx2(const double* values, Index count) noexcept
    {
        __m256d first = _mm256_setzero_pd();
        __m256d second = _mm256_setzero_pd();
        __m256d third = _mm256_setzero_pd();
        __m256d fourth = _mm256_setzero_pd();
        Index index = 0;
        for (; index + 15 < count; index += 16)
        {
            first = _mm256_add_pd(first, _mm256_loadu_pd(values + index));
            second = _mm256_add_pd(second, _mm256_loadu_pd(values + index + 4));
            third = _mm256_add_pd(third, _mm256_loadu_pd(values + index + 8));
            fourth = _mm256_add_pd(fourth, _mm256_loadu_pd(values + index + 12));
        }
        first = _mm256_add_pd(first, second);
        third = _mm256_add_pd(third, fourth);
        double result = horizontalSum(_mm256_add_pd(first, third));
        for (; index < count; ++index)
        {
            result += values[index];
        }
        return result;
    }

    float packetDotAvx2(const float* left, const float* right, Index count) noexcept
    {
        __m256 first = _mm256_setzero_ps();
        __m256 second = _mm256_setzero_ps();
        __m256 third = _mm256_setzero_ps();
        __m256 fourth = _mm256_setzero_ps();
        Index index = 0;
        for (; index + 31 < count; index += 32)
        {
            first = _mm256_fmadd_ps(_mm256_loadu_ps(left + index), _mm256_loadu_ps(right + index), first);
            second = _mm256_fmadd_ps(_mm256_loadu_ps(left + index + 8), _mm256_loadu_ps(right + index + 8), second);
            third = _mm256_fmadd_ps(_mm256_loadu_ps(left + index + 16), _mm256_loadu_ps(right + index + 16), third);
            fourth = _mm256_fmadd_ps(_mm256_loadu_ps(left + index + 24), _mm256_loadu_ps(right + index + 24), fourth);
        }
        first = _mm256_add_ps(first, second);
        third = _mm256_add_ps(third, fourth);
        float result = horizontalSum(_mm256_add_ps(first, third));
        for (; index < count; ++index)
        {
            result += left[index] * right[index];
        }
        return result;
    }

    double packetDotAvx2(const double* left, const double* right, Index count) noexcept
    {
        __m256d first = _mm256_setzero_pd();
        __m256d second = _mm256_setzero_pd();
        __m256d third = _mm256_setzero_pd();
        __m256d fourth = _mm256_setzero_pd();
        Index index = 0;
        for (; index + 15 < count; index += 16)
        {
            first = _mm256_fmadd_pd(_mm256_loadu_pd(left + index), _mm256_loadu_pd(right + index), first);
            second = _mm256_fmadd_pd(_mm256_loadu_pd(left + index + 4), _mm256_loadu_pd(right + index + 4), second);
            third = _mm256_fmadd_pd(_mm256_loadu_pd(left + index + 8), _mm256_loadu_pd(right + index + 8), third);
            fourth = _mm256_fmadd_pd(_mm256_loadu_pd(left + index + 12), _mm256_loadu_pd(right + index + 12), fourth);
        }
        first = _mm256_add_pd(first, second);
        third = _mm256_add_pd(third, fourth);
        double result = horizontalSum(_mm256_add_pd(first, third));
        for (; index < count; ++index)
        {
            result += left[index] * right[index];
        }
        return result;
    }

    float packetExpressionSumAvx2(const PacketReductionTerm<float>* terms, Index term_count, Index count) noexcept
    {
        __m256 first = _mm256_setzero_ps();
        __m256 second = _mm256_setzero_ps();
        __m256 third = _mm256_setzero_ps();
        __m256 fourth = _mm256_setzero_ps();
        Index index = 0;
        for (; index + 31 < count; index += 32)
        {
            for (Index term_index = 0; term_index < term_count; ++term_index)
            {
                const auto& term = terms[term_index];
                const __m256 scale = _mm256_set1_ps(term.scale);
                if (term.first == nullptr)
                {
                    first = _mm256_add_ps(first, scale);
                    second = _mm256_add_ps(second, scale);
                    third = _mm256_add_ps(third, scale);
                    fourth = _mm256_add_ps(fourth, scale);
                }
                else if (term.second == nullptr)
                {
                    first = _mm256_fmadd_ps(_mm256_loadu_ps(term.first + index), scale, first);
                    second = _mm256_fmadd_ps(_mm256_loadu_ps(term.first + index + 8), scale, second);
                    third = _mm256_fmadd_ps(_mm256_loadu_ps(term.first + index + 16), scale, third);
                    fourth = _mm256_fmadd_ps(_mm256_loadu_ps(term.first + index + 24), scale, fourth);
                }
                else
                {
                    const __m256 first_product =
                        _mm256_mul_ps(_mm256_loadu_ps(term.first + index), _mm256_loadu_ps(term.second + index));
                    const __m256 second_product = _mm256_mul_ps(_mm256_loadu_ps(term.first + index + 8),
                                                                _mm256_loadu_ps(term.second + index + 8));
                    const __m256 third_product = _mm256_mul_ps(_mm256_loadu_ps(term.first + index + 16),
                                                               _mm256_loadu_ps(term.second + index + 16));
                    const __m256 fourth_product = _mm256_mul_ps(_mm256_loadu_ps(term.first + index + 24),
                                                                _mm256_loadu_ps(term.second + index + 24));
                    first = _mm256_fmadd_ps(first_product, scale, first);
                    second = _mm256_fmadd_ps(second_product, scale, second);
                    third = _mm256_fmadd_ps(third_product, scale, third);
                    fourth = _mm256_fmadd_ps(fourth_product, scale, fourth);
                }
            }
        }
        first = _mm256_add_ps(first, second);
        third = _mm256_add_ps(third, fourth);
        float result = horizontalSum(_mm256_add_ps(first, third));
        for (; index < count; ++index)
        {
            float coefficient = 0.0F;
            for (Index term_index = 0; term_index < term_count; ++term_index)
            {
                const auto& term = terms[term_index];
                if (term.first == nullptr)
                    coefficient += term.scale;
                else if (term.second == nullptr)
                    coefficient += term.scale * term.first[index];
                else
                    coefficient += term.scale * term.first[index] * term.second[index];
            }
            result += coefficient;
        }
        return result;
    }

    double packetExpressionSumAvx2(const PacketReductionTerm<double>* terms, Index term_count, Index count) noexcept
    {
        __m256d first = _mm256_setzero_pd();
        __m256d second = _mm256_setzero_pd();
        __m256d third = _mm256_setzero_pd();
        __m256d fourth = _mm256_setzero_pd();
        Index index = 0;
        for (; index + 15 < count; index += 16)
        {
            for (Index term_index = 0; term_index < term_count; ++term_index)
            {
                const auto& term = terms[term_index];
                const __m256d scale = _mm256_set1_pd(term.scale);
                if (term.first == nullptr)
                {
                    first = _mm256_add_pd(first, scale);
                    second = _mm256_add_pd(second, scale);
                    third = _mm256_add_pd(third, scale);
                    fourth = _mm256_add_pd(fourth, scale);
                }
                else if (term.second == nullptr)
                {
                    first = _mm256_fmadd_pd(_mm256_loadu_pd(term.first + index), scale, first);
                    second = _mm256_fmadd_pd(_mm256_loadu_pd(term.first + index + 4), scale, second);
                    third = _mm256_fmadd_pd(_mm256_loadu_pd(term.first + index + 8), scale, third);
                    fourth = _mm256_fmadd_pd(_mm256_loadu_pd(term.first + index + 12), scale, fourth);
                }
                else
                {
                    const __m256d first_product =
                        _mm256_mul_pd(_mm256_loadu_pd(term.first + index), _mm256_loadu_pd(term.second + index));
                    const __m256d second_product = _mm256_mul_pd(_mm256_loadu_pd(term.first + index + 4),
                                                                 _mm256_loadu_pd(term.second + index + 4));
                    const __m256d third_product = _mm256_mul_pd(_mm256_loadu_pd(term.first + index + 8),
                                                                _mm256_loadu_pd(term.second + index + 8));
                    const __m256d fourth_product = _mm256_mul_pd(_mm256_loadu_pd(term.first + index + 12),
                                                                 _mm256_loadu_pd(term.second + index + 12));
                    first = _mm256_fmadd_pd(first_product, scale, first);
                    second = _mm256_fmadd_pd(second_product, scale, second);
                    third = _mm256_fmadd_pd(third_product, scale, third);
                    fourth = _mm256_fmadd_pd(fourth_product, scale, fourth);
                }
            }
        }
        first = _mm256_add_pd(first, second);
        third = _mm256_add_pd(third, fourth);
        double result = horizontalSum(_mm256_add_pd(first, third));
        for (; index < count; ++index)
        {
            double coefficient = 0.0;
            for (Index term_index = 0; term_index < term_count; ++term_index)
            {
                const auto& term = terms[term_index];
                if (term.first == nullptr)
                    coefficient += term.scale;
                else if (term.second == nullptr)
                    coefficient += term.scale * term.first[index];
                else
                    coefficient += term.scale * term.first[index] * term.second[index];
            }
            result += coefficient;
        }
        return result;
    }
} // namespace plamatrix::internal::detail
