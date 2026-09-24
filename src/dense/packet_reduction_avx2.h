#pragma once

#include "plamatrix/internal/dense/packet_reduction.h"

namespace plamatrix::internal::detail
{
    float packetSumAvx2(const float* values, Index count) noexcept;
    double packetSumAvx2(const double* values, Index count) noexcept;

    float packetDotAvx2(const float* left, const float* right, Index count) noexcept;
    double packetDotAvx2(const double* left, const double* right, Index count) noexcept;

    float packetExpressionSumAvx2(const PacketReductionTerm<float>* terms, Index term_count, Index count) noexcept;
    double packetExpressionSumAvx2(const PacketReductionTerm<double>* terms, Index term_count, Index count) noexcept;
} // namespace plamatrix::internal::detail
