#include "plamatrix/internal/dense/packet_reduction.h"

#include <algorithm>
#include <array>

#include <omp.h>

#include "../ops/gemm_microkernel.h"

#ifdef PLAMATRIX_HAVE_AVX2_KERNEL
#include "packet_reduction_avx2.h"
#endif

namespace plamatrix::internal::detail
{
    namespace
    {
        constexpr Index packet_threshold = 64;
        constexpr Index parallel_threshold = 262144;
        constexpr Index max_parallel_expression_terms = 16;

        template <typename Scalar> Scalar scalarSum(const Scalar* values, Index count) noexcept
        {
            Scalar result{};
#pragma omp simd reduction(+ : result)
            for (Index index = 0; index < count; ++index)
            {
                result += values[index];
            }
            return result;
        }

        template <typename Scalar> Scalar scalarDot(const Scalar* left, const Scalar* right, Index count) noexcept
        {
            Scalar result{};
#pragma omp simd reduction(+ : result)
            for (Index index = 0; index < count; ++index)
            {
                result += left[index] * right[index];
            }
            return result;
        }

        template <typename Scalar>
        Scalar scalarExpressionSum(const PacketReductionTerm<Scalar>* terms, Index term_count, Index count) noexcept
        {
            Scalar result{};
#pragma omp simd reduction(+ : result)
            for (Index index = 0; index < count; ++index)
            {
                Scalar coefficient{};
                for (Index term_index = 0; term_index < term_count; ++term_index)
                {
                    const auto& term = terms[term_index];
                    if (term.first == nullptr)
                    {
                        coefficient += term.scale;
                    }
                    else if (term.second == nullptr)
                    {
                        coefficient += term.scale * term.first[index];
                    }
                    else
                    {
                        coefficient += term.scale * term.first[index] * term.second[index];
                    }
                }
                result += coefficient;
            }
            return result;
        }

        bool shouldParallelize(Index count) noexcept
        {
            return count >= parallel_threshold && omp_get_max_threads() > 1;
        }

        template <typename Scalar> Scalar serialPacketSum(const Scalar* values, Index count) noexcept
        {
#ifdef PLAMATRIX_HAVE_AVX2_KERNEL
            if (count >= packet_threshold && cpuSupportsAvx2Fma())
                return packetSumAvx2(values, count);
#endif
            return scalarSum(values, count);
        }

        template <typename Scalar> Scalar serialPacketDot(const Scalar* left, const Scalar* right, Index count) noexcept
        {
#ifdef PLAMATRIX_HAVE_AVX2_KERNEL
            if (count >= packet_threshold && cpuSupportsAvx2Fma())
                return packetDotAvx2(left, right, count);
#endif
            return scalarDot(left, right, count);
        }

        template <typename Scalar>
        Scalar
        serialPacketExpressionSum(const PacketReductionTerm<Scalar>* terms, Index term_count, Index count) noexcept
        {
            if (term_count == 1 && terms[0].scale == Scalar{1} && terms[0].first != nullptr)
            {
                return terms[0].second == nullptr ? serialPacketSum(terms[0].first, count)
                                                  : serialPacketDot(terms[0].first, terms[0].second, count);
            }
#ifdef PLAMATRIX_HAVE_AVX2_KERNEL
            if (count >= packet_threshold && cpuSupportsAvx2Fma())
                return packetExpressionSumAvx2(terms, term_count, count);
#endif
            return scalarExpressionSum(terms, term_count, count);
        }

        template <typename Scalar, typename Operation>
        Scalar parallelReduction(Index count, const Operation& operation) noexcept
        {
            Scalar result{};
#pragma omp parallel reduction(+ : result)
            {
                const Index thread_count = static_cast<Index>(omp_get_num_threads());
                const Index thread_index = static_cast<Index>(omp_get_thread_num());
                const Index base_count = count / thread_count;
                const Index remainder = count % thread_count;
                const Index local_count = base_count + (thread_index < remainder ? 1 : 0);
                const Index begin = thread_index * base_count + std::min(thread_index, remainder);
                result += operation(begin, local_count);
            }
            return result;
        }

        template <typename Scalar>
        Scalar
        parallelPacketExpressionSum(const PacketReductionTerm<Scalar>* terms, Index term_count, Index count) noexcept
        {
            return parallelReduction<Scalar>(
                count,
                [&](Index begin, Index local_count)
                {
                    std::array<PacketReductionTerm<Scalar>, max_parallel_expression_terms> adjusted{};
                    for (Index term_index = 0; term_index < term_count; ++term_index)
                    {
                        adjusted[static_cast<std::size_t>(term_index)] = terms[term_index];
                        auto& term = adjusted[static_cast<std::size_t>(term_index)];
                        if (term.first != nullptr)
                            term.first += begin;
                        if (term.second != nullptr)
                            term.second += begin;
                    }
                    return serialPacketExpressionSum(adjusted.data(), term_count, local_count);
                });
        }
    } // namespace

    float packetSum(const float* values, Index count) noexcept
    {
        if (shouldParallelize(count))
            return parallelReduction<float>(
                count, [&](Index begin, Index local_count) { return serialPacketSum(values + begin, local_count); });
        return serialPacketSum(values, count);
    }

    double packetSum(const double* values, Index count) noexcept
    {
        if (shouldParallelize(count))
            return parallelReduction<double>(
                count, [&](Index begin, Index local_count) { return serialPacketSum(values + begin, local_count); });
        return serialPacketSum(values, count);
    }

    float packetDot(const float* left, const float* right, Index count) noexcept
    {
        if (shouldParallelize(count))
            return parallelReduction<float>(count,
                                            [&](Index begin, Index local_count)
                                            { return serialPacketDot(left + begin, right + begin, local_count); });
        return serialPacketDot(left, right, count);
    }

    double packetDot(const double* left, const double* right, Index count) noexcept
    {
        if (shouldParallelize(count))
            return parallelReduction<double>(count,
                                             [&](Index begin, Index local_count)
                                             { return serialPacketDot(left + begin, right + begin, local_count); });
        return serialPacketDot(left, right, count);
    }

    float packetSquaredNorm(const float* values, Index count) noexcept
    {
        return packetDot(values, values, count);
    }

    double packetSquaredNorm(const double* values, Index count) noexcept
    {
        return packetDot(values, values, count);
    }

    float packetExpressionSum(const PacketReductionTerm<float>* terms, Index term_count, Index count) noexcept
    {
        if (shouldParallelize(count) && term_count <= max_parallel_expression_terms)
            return parallelPacketExpressionSum(terms, term_count, count);
        return serialPacketExpressionSum(terms, term_count, count);
    }

    double packetExpressionSum(const PacketReductionTerm<double>* terms, Index term_count, Index count) noexcept
    {
        if (shouldParallelize(count) && term_count <= max_parallel_expression_terms)
            return parallelPacketExpressionSum(terms, term_count, count);
        return serialPacketExpressionSum(terms, term_count, count);
    }
} // namespace plamatrix::internal::detail
