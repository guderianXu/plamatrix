#include "plamatrix/internal/vulkan/iterative_solver.h"

#ifndef PLAMATRIX_WITH_VULKAN

#include <stdexcept>

namespace plamatrix::internal::vulkan
{

    IterativeSolverReport pcg(const CsrStorage<float, Device::CPU>&,
                              const DenseStorage<float, Device::CPU>&,
                              DenseStorage<float, Device::CPU>&,
                              const IterativeSolverOptions&)
    {
        throw std::runtime_error("Vulkan PCG requires PLAMATRIX_WITH_VULKAN=ON");
    }

    IterativeSolverReport blockPcg(const CsrStorage<float, Device::CPU>&,
                                   const DenseStorage<float, Device::CPU>&,
                                   DenseStorage<float, Device::CPU>&,
                                   const DenseStorage<float, Device::CPU>&,
                                   Index,
                                   const IterativeSolverOptions&)
    {
        throw std::runtime_error("Vulkan block PCG requires PLAMATRIX_WITH_VULKAN=ON");
    }

} // namespace plamatrix::internal::vulkan

#endif
