#pragma once

#include "plamatrix/core/types.h"

namespace plamatrix
{
namespace detail
{

/// Minimum scalar work items before entering an OpenMP parallel region.
constexpr Index kOpenMpWorkThreshold = 4096;

/// Return whether a CPU operation is large enough to justify OpenMP overhead.
inline bool shouldUseOpenMp(Index work_items)
{
    return work_items >= kOpenMpWorkThreshold;
}

} // namespace detail
} // namespace plamatrix
