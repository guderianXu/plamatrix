#pragma once

#include <cstddef>
#include <string_view>

#include "plamatrix/internal/core/api.h"
#include "plamatrix/internal/core/backend.h"

namespace plamatrix::internal
{

    inline namespace v1
    {

        enum class ExecutionPolicy
        {
            Auto,
            CpuOnly,
            GpuPreferred,
            GpuRequired
        };

        struct ExecutionInfo
        {
            Backend backend = Backend::Cpu;
            std::size_t bytesUploaded = 0;
            std::size_t bytesDownloaded = 0;
            bool reusedDeviceInput = false;
            std::string_view reason;
        };

        struct ExecutionSettings
        {
            ExecutionPolicy policy = ExecutionPolicy::Auto;
            Backend preferredGpu = Backend::Cuda;
        };

        /// Changes automatic backend selection for the current thread and restores it on destruction.
        class PLAMATRIX_API ScopedExecutionPolicy
        {
        public:
            explicit ScopedExecutionPolicy(ExecutionPolicy policy, Backend preferredGpu = Backend::Cuda);
            ~ScopedExecutionPolicy() noexcept;
            ScopedExecutionPolicy(const ScopedExecutionPolicy&) = delete;
            ScopedExecutionPolicy& operator=(const ScopedExecutionPolicy&) = delete;

        private:
            ExecutionSettings _previous;
        };

        PLAMATRIX_API ExecutionSettings currentExecutionSettings() noexcept;

    } // namespace v1

} // namespace plamatrix::internal
