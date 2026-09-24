#pragma once

#include <array>

#include "plamatrix/internal/core/execution_policy.h"
#include "plamatrix/internal/dense/auto_backend.h"

namespace plamatrix::v1::decomposition_detail
{
    template <typename Scalar>
    internal::Backend
    selectDecompositionBackend(long double work, long double cudaThreshold, long double portableThreshold)
    {
        const auto settings = internal::currentExecutionSettings();
        if (settings.policy == internal::ExecutionPolicy::CpuOnly)
            return internal::Backend::Cpu;
        const auto supported = [](internal::Backend backend)
        { return backend != internal::Backend::Cpu && internal::detail::GpuOps<Scalar>::available(backend); };
        if (settings.policy == internal::ExecutionPolicy::GpuRequired ||
            settings.policy == internal::ExecutionPolicy::GpuPreferred)
        {
            return supported(settings.preferredGpu) ? settings.preferredGpu : internal::Backend::Cpu;
        }
        constexpr std::array candidates = {
            internal::Backend::Cuda, internal::Backend::OpenCl, internal::Backend::Vulkan};
        for (const auto backend : candidates)
        {
            const long double threshold = backend == internal::Backend::Cuda ? cudaThreshold : portableThreshold;
            if (work >= threshold && supported(backend))
                return backend;
        }
        return internal::Backend::Cpu;
    }
} // namespace plamatrix::v1::decomposition_detail
