#pragma once

#include "plamatrix/core/types.h"

#ifdef PLAMATRIX_WITH_OPENMP
#define PLAMATRIX_OMP_PARALLEL_FOR _Pragma("omp parallel for")
#define PLAMATRIX_OMP_PARALLEL_FOR_COLLAPSE_2 _Pragma("omp parallel for collapse(2)")
#else
#define PLAMATRIX_OMP_PARALLEL_FOR
#define PLAMATRIX_OMP_PARALLEL_FOR_COLLAPSE_2
#endif

namespace plamatrix
{
namespace detail
{

/// Minimum scalar work items before entering an OpenMP parallel region.
constexpr Index kOpenMpWorkThreshold = 4096;

/// Return whether a CPU operation is large enough to justify OpenMP overhead.
inline bool shouldUseOpenMp(Index work_items)
{
#ifdef PLAMATRIX_WITH_OPENMP
    return work_items >= kOpenMpWorkThreshold;
#else
    static_cast<void>(work_items);
    return false;
#endif
}

} // namespace detail
} // namespace plamatrix
