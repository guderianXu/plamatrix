#include "plamatrix/internal/core/execution_context.h"

#include "plamatrix/internal/core/error.h"

#include <cstdio>
#include <string>

#ifdef PLAMATRIX_WITH_CUDA
#include <cuda_runtime_api.h>
#include "plamatrix/internal/cuda/native.h"
#endif

#ifdef PLAMATRIX_WITH_OPENCL
#include "../opencl/device_enumeration.h"
#include "plamatrix/internal/opencl/native.h"
#include "plamatrix/internal/opencl/runtime.h"
#endif

#ifdef PLAMATRIX_WITH_VULKAN
#include "plamatrix/internal/vulkan/native.h"
#include "plamatrix/internal/vulkan/runtime.h"
#endif

namespace plamatrix::internal
{

    inline namespace v1
    {

        struct ExecutionContext::State
        {
#ifdef PLAMATRIX_WITH_CUDA
            cudaStream_t cuda_stream = nullptr;
#endif
#ifdef PLAMATRIX_WITH_OPENCL
            cl_context opencl_context = nullptr;
            cl_device_id opencl_device = nullptr;
            cl_command_queue opencl_queue = nullptr;
#endif
#ifdef PLAMATRIX_WITH_VULKAN
            std::unique_ptr<vulkan::Runtime> vulkan_runtime;
#endif

            ~State() noexcept
            {
#ifdef PLAMATRIX_WITH_CUDA
                if (cuda_stream != nullptr)
                {
                    static_cast<void>(cudaStreamDestroy(cuda_stream));
                    cuda_stream = nullptr;
                }
#endif
#ifdef PLAMATRIX_WITH_OPENCL
                if (opencl_queue != nullptr)
                {
                    static_cast<void>(clReleaseCommandQueue(opencl_queue));
                }
                if (opencl_context != nullptr)
                {
                    static_cast<void>(clReleaseContext(opencl_context));
                }
#endif
            }
        };

        ExecutionContext::~ExecutionContext() noexcept = default;

        ExecutionContext::ExecutionContext(DeviceId device,
                                           CapabilitySet capabilities,
                                           std::unique_ptr<State> state) noexcept
            : _device(device), _capabilities(capabilities), _state(std::move(state))
        {
        }

        ExecutionContext ExecutionContext::create(DeviceId device)
        {
            if (device.backend == Backend::Cpu && device.index != 0)
            {
                throw Error(ErrorCode::InvalidArgument, "CPU exposes only device index 0", device.backend);
            }
            if (device.backend == Backend::Cpu)
            {
                CapabilitySet capabilities;
                capabilities.float32 = true;
                capabilities.float64 = true;
                capabilities.hostMemory = true;
                return ExecutionContext(device, capabilities, std::make_unique<State>());
            }
            if (device.backend == Backend::Cuda)
            {
#ifdef PLAMATRIX_WITH_CUDA
                int device_count = 0;
                const auto cuda_status = cudaGetDeviceCount(&device_count);
                if (cuda_status != cudaSuccess || device.index >= static_cast<std::size_t>(device_count))
                {
                    throw Error(ErrorCode::BackendUnavailable,
                                "Requested CUDA device is unavailable",
                                device.backend,
                                static_cast<int>(cuda_status));
                }
                const auto set_status = cudaSetDevice(static_cast<int>(device.index));
                if (set_status != cudaSuccess)
                {
                    throw Error(ErrorCode::BackendFailure,
                                "Failed to select CUDA device",
                                device.backend,
                                static_cast<int>(set_status));
                }
                CapabilitySet capabilities;
                capabilities.float32 = true;
                capabilities.float64 = true;
                capabilities.deviceMemory = true;
                capabilities.asynchronousAllocation = true;
                auto state = std::make_unique<State>();
                const auto stream_status = cudaStreamCreateWithFlags(&state->cuda_stream, cudaStreamNonBlocking);
                if (stream_status != cudaSuccess)
                {
                    throw Error(ErrorCode::BackendFailure,
                                "Failed to create the CUDA ExecutionContext stream",
                                device.backend,
                                static_cast<int>(stream_status));
                }
                return ExecutionContext(device, capabilities, std::move(state));
#else
                throw Error(ErrorCode::BackendUnavailable, "PlaMatrix was built without CUDA support", device.backend);
#endif
            }
            if (device.backend == Backend::OpenCl)
            {
#ifdef PLAMATRIX_WITH_OPENCL
                const auto candidates = opencl::detail::enumerateDeviceCandidates();
                if (device.index >= candidates.size() || candidates[device.index].available == CL_FALSE ||
                    candidates[device.index].compilerAvailable == CL_FALSE)
                {
                    throw Error(
                        ErrorCode::BackendUnavailable, "Requested OpenCL device is unavailable", device.backend);
                }
                const auto& selected = candidates[device.index];
                int opencl_c_major = 0;
                int opencl_c_minor = 0;
                if (std::sscanf(selected.openClCVersion.c_str(), "OpenCL C %d.%d", &opencl_c_major, &opencl_c_minor) !=
                        2 ||
                    opencl_c_major < 1 || (opencl_c_major == 1 && opencl_c_minor < 2))
                {
                    throw Error(ErrorCode::BackendUnavailable,
                                "Requested OpenCL device lacks OpenCL C 1.2 support",
                                device.backend);
                }
                auto state = std::make_unique<State>();
                state->opencl_device = selected.device;
                const cl_context_properties properties[] = {
                    CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(selected.platform), 0};
                cl_int status = CL_SUCCESS;
                state->opencl_context =
                    clCreateContext(properties, 1, &state->opencl_device, nullptr, nullptr, &status);
                if (status != CL_SUCCESS || state->opencl_context == nullptr)
                {
                    throw Error(
                        ErrorCode::BackendFailure, "Failed to create OpenCL ExecutionContext", device.backend, status);
                }
                state->opencl_queue = clCreateCommandQueue(state->opencl_context, state->opencl_device, 0, &status);
                if (status != CL_SUCCESS || state->opencl_queue == nullptr)
                {
                    throw Error(ErrorCode::BackendFailure,
                                "Failed to create OpenCL ExecutionContext queue",
                                device.backend,
                                status);
                }
                CapabilitySet capabilities;
                capabilities.float32 = true;
                capabilities.deviceMemory = true;
                const auto extensions = opencl::detail::deviceString(state->opencl_device, CL_DEVICE_EXTENSIONS);
                capabilities.float64 = extensions.find("cl_khr_fp64") != std::string::npos ||
                                       extensions.find("cl_amd_fp64") != std::string::npos;
                return ExecutionContext(device, capabilities, std::move(state));
#else
                throw Error(
                    ErrorCode::BackendUnavailable, "PlaMatrix was built without OpenCL support", device.backend);
#endif
            }
            if (device.backend == Backend::Vulkan)
            {
#ifdef PLAMATRIX_WITH_VULKAN
                const auto devices = vulkan::enumerateVulkanDevices();
                if (device.index >= devices.size() || !devices[device.index].hasComputeQueue)
                {
                    throw Error(
                        ErrorCode::BackendUnavailable, "Requested Vulkan device is unavailable", device.backend);
                }
                auto state = std::make_unique<State>();
                try
                {
                    state->vulkan_runtime = std::make_unique<vulkan::Runtime>(device.index);
                }
                catch (const std::exception& error)
                {
                    throw Error(ErrorCode::BackendFailure, error.what(), device.backend);
                }
                CapabilitySet capabilities;
                capabilities.float32 = true;
                capabilities.deviceMemory = true;
                capabilities.commandBuffers = true;
                capabilities.cooperativeMatrix = state->vulkan_runtime->cooperativeMatrixCapabilities().enabled;
                return ExecutionContext(device, capabilities, std::move(state));
#else
                throw Error(
                    ErrorCode::BackendUnavailable, "PlaMatrix was built without Vulkan support", device.backend);
#endif
            }
            throw Error(ErrorCode::UnsupportedBackend, "Unknown PlaMatrix backend", device.backend);
        }

        std::shared_ptr<ExecutionContext> ExecutionContext::createShared(DeviceId device)
        {
            // Guaranteed copy elision keeps this non-movable object's address stable.
            return std::shared_ptr<ExecutionContext>(new ExecutionContext(create(device)));
        }

        void ExecutionContext::synchronize()
        {
            if (_device.backend == Backend::Cuda)
            {
#ifdef PLAMATRIX_WITH_CUDA
                const auto status = _state != nullptr && _state->cuda_stream != nullptr
                                        ? cudaStreamSynchronize(_state->cuda_stream)
                                        : cudaDeviceSynchronize();
                if (status != cudaSuccess)
                {
                    throw Error(ErrorCode::BackendFailure,
                                "CUDA context synchronization failed",
                                _device.backend,
                                static_cast<int>(status));
                }
#else
                throw Error(ErrorCode::BackendUnavailable, "PlaMatrix was built without CUDA support", _device.backend);
#endif
            }
            if (_device.backend == Backend::OpenCl)
            {
#ifdef PLAMATRIX_WITH_OPENCL
                const cl_int status = clFinish(_state->opencl_queue);
                if (status != CL_SUCCESS)
                {
                    throw Error(
                        ErrorCode::BackendFailure, "OpenCL context synchronization failed", _device.backend, status);
                }
#else
                throw Error(
                    ErrorCode::BackendUnavailable, "PlaMatrix was built without OpenCL support", _device.backend);
#endif
            }
            if (_device.backend == Backend::Vulkan)
            {
#ifdef PLAMATRIX_WITH_VULKAN
                const auto status = vkDeviceWaitIdle(_state->vulkan_runtime->device());
                if (status != VK_SUCCESS)
                {
                    throw Error(ErrorCode::BackendFailure,
                                "Vulkan context synchronization failed",
                                _device.backend,
                                static_cast<int>(status));
                }
#else
                throw Error(
                    ErrorCode::BackendUnavailable, "PlaMatrix was built without Vulkan support", _device.backend);
#endif
            }
        }

    } // namespace v1

#ifdef PLAMATRIX_WITH_CUDA
    namespace cuda
    {
        cudaStream_t NativeAccess::stream(ExecutionContext& context)
        {
            if (context.backend() != Backend::Cuda)
            {
                throw Error(ErrorCode::UnsupportedBackend,
                            "CUDA native stream requires a CUDA ExecutionContext",
                            context.backend());
            }
            return context._state->cuda_stream;
        }
    } // namespace cuda
#endif

#ifdef PLAMATRIX_WITH_OPENCL
    namespace opencl
    {

        cl_context NativeAccess::context(ExecutionContext& execution_context)
        {
            if (execution_context.backend() != Backend::OpenCl)
            {
                throw Error(ErrorCode::InvalidArgument,
                            "OpenCL native context requires an OpenCL ExecutionContext",
                            execution_context.backend());
            }
            return execution_context._state->opencl_context;
        }

        cl_device_id NativeAccess::device(ExecutionContext& execution_context)
        {
            if (execution_context.backend() != Backend::OpenCl)
            {
                throw Error(ErrorCode::InvalidArgument,
                            "OpenCL native device requires an OpenCL ExecutionContext",
                            execution_context.backend());
            }
            return execution_context._state->opencl_device;
        }

        cl_command_queue NativeAccess::queue(ExecutionContext& execution_context)
        {
            if (execution_context.backend() != Backend::OpenCl)
            {
                throw Error(ErrorCode::InvalidArgument,
                            "OpenCL native queue requires an OpenCL ExecutionContext",
                            execution_context.backend());
            }
            return execution_context._state->opencl_queue;
        }

    } // namespace opencl
#endif

#ifdef PLAMATRIX_WITH_VULKAN
    namespace vulkan
    {

        Runtime& NativeAccess::runtime(ExecutionContext& execution_context)
        {
            if (execution_context.backend() != Backend::Vulkan)
            {
                throw Error(ErrorCode::InvalidArgument,
                            "Vulkan native runtime requires a Vulkan ExecutionContext",
                            execution_context.backend());
            }
            return *execution_context._state->vulkan_runtime;
        }

    } // namespace vulkan
#endif

} // namespace plamatrix::internal
