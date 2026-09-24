#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#ifdef PLAMATRIX_WITH_VULKAN
#include <vulkan/vulkan.h>
#endif

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

    std::vector<VulkanDeviceInfo> enumerateVulkanDevices();
    bool hasUsableVulkanDevice() noexcept;
    std::string selectedVulkanDeviceName();
    CooperativeMatrixCapabilities selectedVulkanCooperativeMatrixCapabilities();

#ifdef PLAMATRIX_WITH_VULKAN

    class Runtime;

    enum class BufferMemory
    {
        HostVisible,
        HostVisibleCached,
        DeviceLocal
    };

    class Buffer
    {
    public:
        Buffer() = default;
        Buffer(Runtime& runtime,
               VkDeviceSize size,
               VkBufferUsageFlags usage,
               BufferMemory memory = BufferMemory::HostVisible);
        ~Buffer() noexcept;
        Buffer(const Buffer&) = delete;
        Buffer& operator=(const Buffer&) = delete;
        Buffer(Buffer&& other) noexcept;
        Buffer& operator=(Buffer&& other) noexcept;

        VkBuffer handle() const noexcept
        {
            return _buffer;
        }
        VkDeviceSize size() const noexcept
        {
            return _size;
        }
        void* map();
        void unmap() noexcept;
        bool hostVisible() const noexcept
        {
            return _hostVisible;
        }

    private:
        void reset() noexcept;
        Runtime* _runtime = nullptr;
        VkBuffer _buffer = VK_NULL_HANDLE;
        VkDeviceMemory _memory = VK_NULL_HANDLE;
        VkDeviceSize _size = 0;
        bool _hostVisible = false;
    };

    class Runtime
    {
    public:
        static Runtime& instance();
        explicit Runtime(std::size_t deviceIndex);
        ~Runtime() noexcept;
        Runtime(const Runtime&) = delete;
        Runtime& operator=(const Runtime&) = delete;

        VkInstance instanceHandle() const noexcept
        {
            return _instance;
        }
        VkPhysicalDevice physicalDevice() const noexcept
        {
            return _physicalDevice;
        }
        VkDevice device() const noexcept
        {
            return _device;
        }
        VkQueue queue() const noexcept
        {
            return _queue;
        }
        std::uint32_t queueFamily() const noexcept
        {
            return _queueFamily;
        }
        float timestampPeriodNanoseconds() const noexcept
        {
            return _timestampPeriodNanoseconds;
        }
        bool supportsSubgroupArithmetic() const noexcept
        {
            return _supportsSubgroupArithmetic;
        }
        std::uint32_t subgroupSize() const noexcept
        {
            return _subgroupSize;
        }
        const CooperativeMatrixCapabilities& cooperativeMatrixCapabilities() const noexcept
        {
            return _cooperativeMatrixCapabilities;
        }
        VkCommandPool commandPool() const noexcept
        {
            return _commandPool;
        }
        const std::string& deviceName() const noexcept
        {
            return _deviceName;
        }

        std::uint32_t findMemoryType(std::uint32_t typeBits, VkMemoryPropertyFlags properties) const;
        std::vector<std::uint32_t> loadShader(const char* name) const;
        VkCommandBuffer beginCommandBuffer() const;
        void submitAndWait(VkCommandBuffer commandBuffer) const;

    private:
        VkInstance _instance = VK_NULL_HANDLE;
        VkPhysicalDevice _physicalDevice = VK_NULL_HANDLE;
        VkDevice _device = VK_NULL_HANDLE;
        VkQueue _queue = VK_NULL_HANDLE;
        VkCommandPool _commandPool = VK_NULL_HANDLE;
        std::uint32_t _queueFamily = 0;
        float _timestampPeriodNanoseconds = 0.0f;
        std::uint32_t _subgroupSize = 0;
        bool _supportsSubgroupArithmetic = false;
        CooperativeMatrixCapabilities _cooperativeMatrixCapabilities;
        std::string _deviceName;
    };

#else

    inline bool hasUsableVulkanDevice() noexcept
    {
        return false;
    }
    inline std::string selectedVulkanDeviceName()
    {
        return {};
    }
    inline std::vector<VulkanDeviceInfo> enumerateVulkanDevices()
    {
        return {};
    }
    inline CooperativeMatrixCapabilities selectedVulkanCooperativeMatrixCapabilities()
    {
        return {};
    }

#endif

} // namespace plamatrix::internal::vulkan
