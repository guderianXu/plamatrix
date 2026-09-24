#pragma once

#include <memory>

#include "plamatrix/internal/core/api.h"
#include "plamatrix/internal/core/backend.h"
#include "plamatrix/internal/core/capabilities.h"

namespace plamatrix::internal
{

    namespace cuda
    {
        struct NativeAccess;
    }
    namespace opencl
    {
        struct NativeAccess;
    }
    namespace vulkan
    {
        struct NativeAccess;
    }

    inline namespace v1
    {

        class PLAMATRIX_API ExecutionContext
        {
        public:
            static ExecutionContext create(DeviceId device = {});
            /// Create a stable context that can be shared by owners of resident allocations.
            static std::shared_ptr<ExecutionContext> createShared(DeviceId device = {});

            ~ExecutionContext() noexcept;
            ExecutionContext(ExecutionContext&&) = delete;
            ExecutionContext& operator=(ExecutionContext&&) = delete;
            ExecutionContext(const ExecutionContext&) = delete;
            ExecutionContext& operator=(const ExecutionContext&) = delete;

            Backend backend() const noexcept
            {
                return _device.backend;
            }

            DeviceId device() const noexcept
            {
                return _device;
            }

            const CapabilitySet& capabilities() const noexcept
            {
                return _capabilities;
            }

            void synchronize();

        private:
            friend struct cuda::NativeAccess;
            friend struct opencl::NativeAccess;
            friend struct vulkan::NativeAccess;
            struct State;

            explicit ExecutionContext(DeviceId device,
                                      CapabilitySet capabilities,
                                      std::unique_ptr<State> state) noexcept;

            DeviceId _device{};
            CapabilitySet _capabilities{};
            std::unique_ptr<State> _state;
        };

    } // namespace v1

} // namespace plamatrix::internal
