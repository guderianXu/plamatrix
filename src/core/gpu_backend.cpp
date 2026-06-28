#include "plamatrix/core/gpu_backend.h"

namespace plamatrix
{

GpuBackend gpuBackend()
{
#ifdef PLAMATRIX_WITH_CUDA
    return GpuBackend::Cuda;
#elif defined(PLAMATRIX_WITH_METAL)
    return GpuBackend::Metal;
#else
    return GpuBackend::None;
#endif
}

const char* gpuBackendName()
{
#ifdef PLAMATRIX_WITH_CUDA
    return "cuda";
#elif defined(PLAMATRIX_WITH_METAL)
    return "metal";
#else
    return "none";
#endif
}

} // namespace plamatrix
