#pragma once

#include "plamatrix/internal/core/api.h"
#include "plamatrix/internal/core/execution_context.h"

#ifdef PLAMATRIX_WITH_CUDA
#include <cuda_runtime_api.h>

namespace plamatrix::internal::cuda
{

    /// Native CUDA stream access for backend implementations using an explicit context.
    struct PLAMATRIX_API NativeAccess
    {
        static cudaStream_t stream(ExecutionContext& context);
    };

} // namespace plamatrix::internal::cuda
#endif
