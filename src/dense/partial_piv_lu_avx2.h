#pragma once

#include "plamatrix/internal/dense/partial_piv_lu.h"

namespace plamatrix::internal::detail
{
    void partialPivLuTrailingUpdateAvx2(float* factors, Index size, Index block_start, Index block_end) noexcept;
    void partialPivLuTrailingUpdateAvx2(double* factors, Index size, Index block_start, Index block_end) noexcept;
} // namespace plamatrix::internal::detail
