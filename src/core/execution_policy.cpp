#include "plamatrix/internal/core/execution_policy.h"

#include "plamatrix/internal/core/error.h"

namespace plamatrix::internal::v1
{
    namespace
    {
        thread_local ExecutionSettings settings;
    }

    ExecutionSettings currentExecutionSettings() noexcept
    {
        return settings;
    }

    ScopedExecutionPolicy::ScopedExecutionPolicy(ExecutionPolicy policy, Backend preferredGpu) : _previous(settings)
    {
        if (preferredGpu == Backend::Cpu)
        {
            throw Error(ErrorCode::InvalidArgument, "Preferred GPU backend cannot be CPU");
        }
        settings = {policy, preferredGpu};
    }

    ScopedExecutionPolicy::~ScopedExecutionPolicy() noexcept
    {
        settings = _previous;
    }

} // namespace plamatrix::internal::v1
