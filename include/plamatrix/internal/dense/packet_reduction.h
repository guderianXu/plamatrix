#pragma once

#include "plamatrix/core/types.h"

namespace plamatrix::internal::detail
{
    template <typename Scalar> struct PacketReductionTerm
    {
        const Scalar* first = nullptr;
        const Scalar* second = nullptr;
        Scalar scale{};
    };

    float packetSum(const float* values, Index count) noexcept;
    double packetSum(const double* values, Index count) noexcept;

    float packetDot(const float* left, const float* right, Index count) noexcept;
    double packetDot(const double* left, const double* right, Index count) noexcept;

    float packetSquaredNorm(const float* values, Index count) noexcept;
    double packetSquaredNorm(const double* values, Index count) noexcept;

    float packetExpressionSum(const PacketReductionTerm<float>* terms, Index term_count, Index count) noexcept;
    double packetExpressionSum(const PacketReductionTerm<double>* terms, Index term_count, Index count) noexcept;
} // namespace plamatrix::internal::detail
