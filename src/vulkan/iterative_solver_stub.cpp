#include "plamatrix/vulkan/iterative_solver.h"

#ifndef PLAMATRIX_WITH_VULKAN

#include <stdexcept>

namespace plamatrix::vulkan
{

    IterativeSolverReport pcg(const CSRMatrix<float, Device::CPU>&,
                              const DenseMatrix<float, Device::CPU>&,
                              DenseMatrix<float, Device::CPU>&,
                              const IterativeSolverOptions&)
    {
        throw std::runtime_error("Vulkan PCG requires PLAMATRIX_WITH_VULKAN=ON");
    }

} // namespace plamatrix::vulkan

#endif
