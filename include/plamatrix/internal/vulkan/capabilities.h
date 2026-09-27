#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace plamatrix::internal::vulkan
{

    struct VulkanDeviceInfo
    {
        std::size_t index = 0;
        std::string name;
        std::uint32_t apiVersion = 0;
        bool hasComputeQueue = false;
    };

    /// Cooperative-matrix capabilities of the selected Vulkan device and build.
    struct CooperativeMatrixCapabilities
    {
        bool hardwareSupported = false;
        bool enabled = false;
        bool fp16InputsFp32Accumulation = false;
        std::uint32_t mSize = 0;
        std::uint32_t nSize = 0;
        std::uint32_t kSize = 0;
        std::string extensionName;
    };

#ifdef PLAMATRIX_WITH_VULKAN

    std::vector<VulkanDeviceInfo> enumerateVulkanDevices();
    bool hasUsableVulkanDevice() noexcept;
    std::string selectedVulkanDeviceName();
    CooperativeMatrixCapabilities selectedVulkanCooperativeMatrixCapabilities();

#else

    inline std::vector<VulkanDeviceInfo> enumerateVulkanDevices()
    {
        return {};
    }

    inline bool hasUsableVulkanDevice() noexcept
    {
        return false;
    }

    inline std::string selectedVulkanDeviceName()
    {
        return {};
    }

    inline CooperativeMatrixCapabilities selectedVulkanCooperativeMatrixCapabilities()
    {
        return {};
    }

#endif

} // namespace plamatrix::internal::vulkan
