#pragma once

#include "plamatrix/sparse/iterative_solver.h"
#include "plamatrix/vulkan/runtime.h"

namespace plamatrix::vulkan
{

    /// Solve an SPD CPU-owned float CSR system on the selected Vulkan compute device.
    IterativeSolverReport pcg(const CSRMatrix<float, Device::CPU>& matrix,
                              const DenseMatrix<float, Device::CPU>& rhs,
                              DenseMatrix<float, Device::CPU>& solution,
                              const IterativeSolverOptions& options = {});

} // namespace plamatrix::vulkan
