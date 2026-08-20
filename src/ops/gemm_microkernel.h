#pragma once

#include "plamatrix/core/types.h"

namespace plamatrix::detail
{

bool cpuSupportsAvx2Fma() noexcept;

#ifdef PLAMATRIX_HAVE_AVX2_KERNEL
void packedGemmMicrokernelAvx2(const float* left,
                               const float* packed_right,
                               float* output,
                               Index leading_dimension,
                               Index inner_size,
                               Index packed_columns,
                               Index row_begin,
                               Index row_end,
                               Index column_begin,
                               Index column_end);

void packedGemmMicrokernelAvx2(const double* left,
                               const double* packed_right,
                               double* output,
                               Index leading_dimension,
                               Index inner_size,
                               Index packed_columns,
                               Index row_begin,
                               Index row_end,
                               Index column_begin,
                               Index column_end);
#endif

} // namespace plamatrix::detail
