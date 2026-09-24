#pragma once

#include "plamatrix/internal/sparse/iterative_solver.h"
#include "plamatrix/internal/vulkan/runtime.h"

namespace plamatrix::internal::vulkan
{

    /// Solve an SPD CPU-owned float CSR system on the selected Vulkan compute device.
    IterativeSolverReport pcg(const CsrStorage<float, Device::CPU>& matrix,
                              const DenseStorage<float, Device::CPU>& rhs,
                              DenseStorage<float, Device::CPU>& solution,
                              const IterativeSolverOptions& options = {});

    /// Solve with caller-supplied row-major inverse diagonal blocks.
    IterativeSolverReport blockPcg(const CsrStorage<float, Device::CPU>& matrix,
                                   const DenseStorage<float, Device::CPU>& rhs,
                                   DenseStorage<float, Device::CPU>& solution,
                                   const DenseStorage<float, Device::CPU>& inverseBlocks,
                                   Index blockSize,
                                   const IterativeSolverOptions& options = {});

} // namespace plamatrix::internal::vulkan
