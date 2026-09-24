#pragma once

#ifdef PLAMATRIX_WITH_VULKAN

#include <cstdint>
#include <functional>

#include "plamatrix/internal/vulkan/iterative_solver.h"

namespace plamatrix::internal::vulkan
{

    class Buffer;
    class CommandContext;

    using DevicePcgPrefixRecorder = std::function<void(CommandContext&)>;

    IterativeSolverReport blockPcgWithDeviceValues(const CsrStorage<float, Device::CPU>& matrix,
                                                   const DenseStorage<float, Device::CPU>& rhs,
                                                   DenseStorage<float, Device::CPU>& solution,
                                                   const DenseStorage<float, Device::CPU>* inverseBlocks,
                                                   Index blockSize,
                                                   Buffer& deviceValues,
                                                   std::uint64_t deviceValuesGeneration,
                                                   std::uint64_t topologyGeneration,
                                                   const DevicePcgPrefixRecorder& prefixRecorder,
                                                   bool requireDeviceBlockJacobi,
                                                   const IterativeSolverOptions& options);

} // namespace plamatrix::internal::vulkan

#endif
