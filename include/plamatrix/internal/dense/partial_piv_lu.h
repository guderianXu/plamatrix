#pragma once

#include "plamatrix/core/types.h"

namespace plamatrix::internal::detail
{
    void partialPivLuTrailingUpdate(float* factors, Index size, Index block_start, Index block_end) noexcept;
    void partialPivLuTrailingUpdate(double* factors, Index size, Index block_start, Index block_end) noexcept;
} // namespace plamatrix::internal::detail
