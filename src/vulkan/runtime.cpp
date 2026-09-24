#include "plamatrix/internal/vulkan/runtime.h"

#ifdef PLAMATRIX_WITH_VULKAN

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace plamatrix::internal::vulkan
{
    namespace
    {

        void checkVk(VkResult result, const char* operation)
        {
            if (result != VK_SUCCESS)
            {
                throw std::runtime_error(std::string("Vulkan ") + operation + " failed with error " +
                                         std::to_string(static_cast<int>(result)));
            }
        }

        std::uint32_t requestedDeviceIndex()
        {
            const char* value = std::getenv("PLAMATRIX_VULKAN_DEVICE_INDEX");
            if (!value || value[0] == '\0')
                return 0;
            char* end = nullptr;
            const unsigned long parsed = std::strtoul(value, &end, 10);
            if (*end != '\0' || parsed > std::numeric_limits<std::uint32_t>::max())
            {
                throw std::invalid_argument("PLAMATRIX_VULKAN_DEVICE_INDEX must be a non-negative integer");
            }
            return static_cast<std::uint32_t>(parsed);
        }

        VkInstance createInstance()
        {
            VkApplicationInfo app{};
            app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
            app.pApplicationName = "PlaMatrix";
            app.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
            app.pEngineName = "PlaMatrix";
            app.engineVersion = VK_MAKE_VERSION(0, 1, 0);
            app.apiVersion = VK_API_VERSION_1_1;

            VkInstanceCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
            info.pApplicationInfo = &app;
            VkInstance instance = VK_NULL_HANDLE;
            checkVk(vkCreateInstance(&info, nullptr, &instance), "vkCreateInstance");
            return instance;
        }

        std::vector<VkPhysicalDevice> physicalDevices(VkInstance instance)
        {
            std::uint32_t count = 0;
            checkVk(vkEnumeratePhysicalDevices(instance, &count, nullptr), "vkEnumeratePhysicalDevices");
            if (count == 0)
                return {};
            std::vector<VkPhysicalDevice> devices(count);
            checkVk(vkEnumeratePhysicalDevices(instance, &count, devices.data()), "vkEnumeratePhysicalDevices");
            return devices;
        }

        std::uint32_t computeQueueFamily(VkPhysicalDevice physical)
        {
            std::uint32_t count = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, nullptr);
            std::vector<VkQueueFamilyProperties> families(count);
            vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, families.data());
            for (std::uint32_t index = 0; index < count; ++index)
            {
                if ((families[index].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0)
                    return index;
            }
            return std::numeric_limits<std::uint32_t>::max();
        }

        bool hasDeviceExtension(VkPhysicalDevice physical, const char* name)
        {
            std::uint32_t count = 0;
            checkVk(vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, nullptr),
                    "vkEnumerateDeviceExtensionProperties");
            std::vector<VkExtensionProperties> extensions(count);
            if (count != 0)
            {
                checkVk(vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, extensions.data()),
                        "vkEnumerateDeviceExtensionProperties");
            }
            return std::any_of(extensions.begin(),
                               extensions.end(),
                               [&](const VkExtensionProperties& extension)
                               { return std::strcmp(extension.extensionName, name) == 0; });
        }

        CooperativeMatrixCapabilities queryCooperativeMatrixCapabilities(VkInstance instance, VkPhysicalDevice physical)
        {
            CooperativeMatrixCapabilities result;
#if defined(VK_KHR_cooperative_matrix) && defined(VK_KHR_shader_float16_int8) && defined(VK_KHR_vulkan_memory_model)
            if (!hasDeviceExtension(physical, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) ||
                !hasDeviceExtension(physical, VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME) ||
                !hasDeviceExtension(physical, VK_KHR_VULKAN_MEMORY_MODEL_EXTENSION_NAME))
            {
                return result;
            }

            VkPhysicalDeviceVulkanMemoryModelFeatures memory_model_features{};
            memory_model_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES;
            VkPhysicalDeviceCooperativeMatrixFeaturesKHR cooperative_features{};
            cooperative_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
            cooperative_features.pNext = &memory_model_features;
            VkPhysicalDeviceShaderFloat16Int8Features float16_features{};
            float16_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
            float16_features.pNext = &cooperative_features;
            VkPhysicalDevice16BitStorageFeatures storage_features{};
            storage_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
            storage_features.pNext = &float16_features;
            VkPhysicalDeviceFeatures2 features{};
            features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            features.pNext = &storage_features;
            vkGetPhysicalDeviceFeatures2(physical, &features);

            VkPhysicalDeviceCooperativeMatrixPropertiesKHR cooperative_properties{};
            cooperative_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
            VkPhysicalDeviceProperties2 properties{};
            properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
            properties.pNext = &cooperative_properties;
            vkGetPhysicalDeviceProperties2(physical, &properties);

            const auto query_properties = reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
                vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
            if (!query_properties || cooperative_features.cooperativeMatrix != VK_TRUE ||
                float16_features.shaderFloat16 != VK_TRUE || storage_features.storageBuffer16BitAccess != VK_TRUE ||
                memory_model_features.vulkanMemoryModel != VK_TRUE ||
                (cooperative_properties.cooperativeMatrixSupportedStages & VK_SHADER_STAGE_COMPUTE_BIT) == 0)
            {
                return result;
            }

            std::uint32_t property_count = 0;
            if (query_properties(physical, &property_count, nullptr) != VK_SUCCESS || property_count == 0)
            {
                return result;
            }
            std::vector<VkCooperativeMatrixPropertiesKHR> matrix_properties(property_count);
            for (auto& property : matrix_properties)
            {
                property.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
            }
            if (query_properties(physical, &property_count, matrix_properties.data()) != VK_SUCCESS)
            {
                return result;
            }
            const auto compatible = std::find_if(matrix_properties.begin(),
                                                 matrix_properties.end(),
                                                 [](const VkCooperativeMatrixPropertiesKHR& property)
                                                 {
                                                     return property.MSize == 16 && property.NSize == 16 &&
                                                            property.KSize == 16 &&
                                                            property.AType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                                                            property.BType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                                                            property.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                                                            property.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                                                            property.scope == VK_SCOPE_SUBGROUP_KHR;
                                                 });
            if (compatible == matrix_properties.end())
            {
                return result;
            }
            result.hardwareSupported = true;
            result.fp16InputsFp32Accumulation = true;
            result.mSize = compatible->MSize;
            result.nSize = compatible->NSize;
            result.kSize = compatible->KSize;
            result.extensionName = VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME;
#else
            static_cast<void>(instance);
            static_cast<void>(physical);
#endif
            return result;
        }

    } // namespace

    std::vector<VulkanDeviceInfo> enumerateVulkanDevices()
    {
        VkInstance instance = createInstance();
        try
        {
            const auto devices = physicalDevices(instance);
            std::vector<VulkanDeviceInfo> result;
            result.reserve(devices.size());
            for (std::size_t index = 0; index < devices.size(); ++index)
            {
                VkPhysicalDeviceProperties properties{};
                vkGetPhysicalDeviceProperties(devices[index], &properties);
                result.push_back(
                    VulkanDeviceInfo{index,
                                     properties.deviceName,
                                     properties.apiVersion,
                                     computeQueueFamily(devices[index]) != std::numeric_limits<std::uint32_t>::max()});
            }
            vkDestroyInstance(instance, nullptr);
            return result;
        }
        catch (...)
        {
            vkDestroyInstance(instance, nullptr);
            throw;
        }
    }

    bool hasUsableVulkanDevice() noexcept
    {
        try
        {
            static_cast<void>(Runtime::instance());
            return true;
        }
        catch (...)
        {
            return false;
        }
    }

    std::string selectedVulkanDeviceName()
    {
        return Runtime::instance().deviceName();
    }

    CooperativeMatrixCapabilities selectedVulkanCooperativeMatrixCapabilities()
    {
        return Runtime::instance().cooperativeMatrixCapabilities();
    }

    Runtime& Runtime::instance()
    {
        static Runtime runtime(requestedDeviceIndex());
        return runtime;
    }

    Runtime::Runtime(std::size_t deviceIndex)
    {
        _instance = createInstance();
        try
        {
            const auto devices = physicalDevices(_instance);
            const std::size_t requested = deviceIndex;
            if (requested >= devices.size())
            {
                throw std::out_of_range("Requested Vulkan device index exceeds available Vulkan devices");
            }
            _physicalDevice = devices[requested];
            _queueFamily = computeQueueFamily(_physicalDevice);
            if (_queueFamily == std::numeric_limits<std::uint32_t>::max())
            {
                throw std::runtime_error("Selected Vulkan device has no compute queue family");
            }

            VkPhysicalDeviceProperties properties{};
            vkGetPhysicalDeviceProperties(_physicalDevice, &properties);
            _deviceName = properties.deviceName;
            _timestampPeriodNanoseconds = properties.limits.timestampPeriod;
            if (properties.apiVersion >= VK_API_VERSION_1_1)
            {
                VkPhysicalDeviceSubgroupProperties subgroupProperties{};
                subgroupProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
                VkPhysicalDeviceProperties2 properties2{};
                properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
                properties2.pNext = &subgroupProperties;
                vkGetPhysicalDeviceProperties2(_physicalDevice, &properties2);
                _subgroupSize = subgroupProperties.subgroupSize;
                constexpr VkSubgroupFeatureFlags requiredSubgroupOperations =
                    VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;
                _supportsSubgroupArithmetic = (subgroupProperties.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0 &&
                                              (subgroupProperties.supportedOperations & requiredSubgroupOperations) ==
                                                  requiredSubgroupOperations &&
                                              _subgroupSize != 0 && 128u % _subgroupSize == 0;
            }

            _cooperativeMatrixCapabilities = queryCooperativeMatrixCapabilities(_instance, _physicalDevice);

            const float priority = 1.0f;
            VkDeviceQueueCreateInfo queue_info{};
            queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queue_info.queueFamilyIndex = _queueFamily;
            queue_info.queueCount = 1;
            queue_info.pQueuePriorities = &priority;
            VkDeviceCreateInfo device_info{};
            device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
            device_info.queueCreateInfoCount = 1;
            device_info.pQueueCreateInfos = &queue_info;
#if defined(PLAMATRIX_VULKAN_COOPERATIVE_MATRIX_SHADER) && defined(VK_KHR_cooperative_matrix) &&                       \
    defined(VK_KHR_shader_float16_int8) && defined(VK_KHR_vulkan_memory_model)
            std::vector<const char*> device_extensions;
            VkPhysicalDeviceVulkanMemoryModelFeatures memory_model_features{};
            VkPhysicalDeviceCooperativeMatrixFeaturesKHR cooperative_features{};
            VkPhysicalDeviceShaderFloat16Int8Features float16_features{};
            VkPhysicalDevice16BitStorageFeatures storage_features{};
            if (_cooperativeMatrixCapabilities.hardwareSupported && _subgroupSize == 32)
            {
                memory_model_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES;
                memory_model_features.vulkanMemoryModel = VK_TRUE;
                cooperative_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
                cooperative_features.cooperativeMatrix = VK_TRUE;
                cooperative_features.pNext = &memory_model_features;
                float16_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
                float16_features.shaderFloat16 = VK_TRUE;
                float16_features.pNext = &cooperative_features;
                storage_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
                storage_features.storageBuffer16BitAccess = VK_TRUE;
                storage_features.pNext = &float16_features;
                device_extensions.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
                device_extensions.push_back(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);
                device_extensions.push_back(VK_KHR_VULKAN_MEMORY_MODEL_EXTENSION_NAME);
                device_info.enabledExtensionCount = static_cast<std::uint32_t>(device_extensions.size());
                device_info.ppEnabledExtensionNames = device_extensions.data();
                device_info.pNext = &storage_features;
                _cooperativeMatrixCapabilities.enabled = true;
            }
#endif
            checkVk(vkCreateDevice(_physicalDevice, &device_info, nullptr, &_device), "vkCreateDevice");
            vkGetDeviceQueue(_device, _queueFamily, 0, &_queue);

            VkCommandPoolCreateInfo pool_info{};
            pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            pool_info.queueFamilyIndex = _queueFamily;
            checkVk(vkCreateCommandPool(_device, &pool_info, nullptr, &_commandPool), "vkCreateCommandPool");
        }
        catch (...)
        {
            if (_commandPool)
                vkDestroyCommandPool(_device, _commandPool, nullptr);
            if (_device)
                vkDestroyDevice(_device, nullptr);
            if (_instance)
                vkDestroyInstance(_instance, nullptr);
            throw;
        }
    }

    Runtime::~Runtime() noexcept
    {
        if (_device)
            vkDeviceWaitIdle(_device);
        if (_commandPool)
            vkDestroyCommandPool(_device, _commandPool, nullptr);
        if (_device)
            vkDestroyDevice(_device, nullptr);
        if (_instance)
            vkDestroyInstance(_instance, nullptr);
    }

    std::uint32_t Runtime::findMemoryType(std::uint32_t typeBits, VkMemoryPropertyFlags properties) const
    {
        VkPhysicalDeviceMemoryProperties memory_properties{};
        vkGetPhysicalDeviceMemoryProperties(_physicalDevice, &memory_properties);
        for (std::uint32_t index = 0; index < memory_properties.memoryTypeCount; ++index)
        {
            if ((typeBits & (1u << index)) != 0 &&
                (memory_properties.memoryTypes[index].propertyFlags & properties) == properties)
            {
                return index;
            }
        }
        throw std::runtime_error("Vulkan device has no compatible memory type");
    }

    std::vector<std::uint32_t> Runtime::loadShader(const char* name) const
    {
        const std::string path = std::string(PLAMATRIX_VULKAN_SHADER_DIR) + "/" + name + ".spv";
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file)
        {
            throw std::runtime_error("Unable to open Vulkan shader: " + path);
        }
        const std::streamsize byte_count = file.tellg();
        if (byte_count <= 0 || byte_count % 4 != 0)
        {
            throw std::runtime_error("Invalid SPIR-V shader size: " + path);
        }
        file.seekg(0, std::ios::beg);
        std::vector<std::uint32_t> words(static_cast<std::size_t>(byte_count) / 4);
        file.read(reinterpret_cast<char*>(words.data()), byte_count);
        if (!file)
            throw std::runtime_error("Unable to read Vulkan shader: " + path);
        return words;
    }

    VkCommandBuffer Runtime::beginCommandBuffer() const
    {
        VkCommandBufferAllocateInfo allocation{};
        allocation.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocation.commandPool = _commandPool;
        allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocation.commandBufferCount = 1;
        VkCommandBuffer command = VK_NULL_HANDLE;
        checkVk(vkAllocateCommandBuffers(_device, &allocation, &command), "vkAllocateCommandBuffers");
        VkCommandBufferBeginInfo begin{};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        try
        {
            checkVk(vkBeginCommandBuffer(command, &begin), "vkBeginCommandBuffer");
        }
        catch (...)
        {
            vkFreeCommandBuffers(_device, _commandPool, 1, &command);
            throw;
        }
        return command;
    }

    void Runtime::submitAndWait(VkCommandBuffer commandBuffer) const
    {
        checkVk(vkEndCommandBuffer(commandBuffer), "vkEndCommandBuffer");
        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &commandBuffer;
        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkFence fence = VK_NULL_HANDLE;
        checkVk(vkCreateFence(_device, &fence_info, nullptr, &fence), "vkCreateFence");
        try
        {
            checkVk(vkQueueSubmit(_queue, 1, &submit, fence), "vkQueueSubmit");
            checkVk(vkWaitForFences(_device, 1, &fence, VK_TRUE, std::numeric_limits<std::uint64_t>::max()),
                    "vkWaitForFences");
            vkDestroyFence(_device, fence, nullptr);
            vkFreeCommandBuffers(_device, _commandPool, 1, &commandBuffer);
        }
        catch (...)
        {
            vkDestroyFence(_device, fence, nullptr);
            vkFreeCommandBuffers(_device, _commandPool, 1, &commandBuffer);
            throw;
        }
    }

    Buffer::Buffer(Runtime& runtime, VkDeviceSize size, VkBufferUsageFlags usage, BufferMemory memory)
        : _runtime(&runtime), _size(size), _hostVisible(memory != BufferMemory::DeviceLocal)
    {
        if (size == 0)
            throw std::invalid_argument("Vulkan buffer size must be greater than zero");
        VkBufferCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        info.size = size;
        info.usage = usage;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        checkVk(vkCreateBuffer(runtime.device(), &info, nullptr, &_buffer), "vkCreateBuffer");
        try
        {
            VkMemoryRequirements requirements{};
            vkGetBufferMemoryRequirements(runtime.device(), _buffer, &requirements);
            VkMemoryAllocateInfo allocation{};
            allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            allocation.allocationSize = requirements.size;
            const VkMemoryPropertyFlags host_visible_properties =
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            if (memory == BufferMemory::HostVisibleCached)
            {
                try
                {
                    allocation.memoryTypeIndex = runtime.findMemoryType(
                        requirements.memoryTypeBits, host_visible_properties | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
                }
                catch (const std::runtime_error&)
                {
                    allocation.memoryTypeIndex =
                        runtime.findMemoryType(requirements.memoryTypeBits, host_visible_properties);
                }
            }
            else
            {
                const VkMemoryPropertyFlags properties =
                    memory == BufferMemory::HostVisible ? host_visible_properties : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
                allocation.memoryTypeIndex = runtime.findMemoryType(requirements.memoryTypeBits, properties);
            }
            checkVk(vkAllocateMemory(runtime.device(), &allocation, nullptr, &_memory), "vkAllocateMemory");
            checkVk(vkBindBufferMemory(runtime.device(), _buffer, _memory, 0), "vkBindBufferMemory");
        }
        catch (...)
        {
            reset();
            throw;
        }
    }

    Buffer::~Buffer() noexcept
    {
        reset();
    }

    Buffer::Buffer(Buffer&& other) noexcept
        : _runtime(std::exchange(other._runtime, nullptr)), _buffer(std::exchange(other._buffer, VK_NULL_HANDLE)),
          _memory(std::exchange(other._memory, VK_NULL_HANDLE)), _size(std::exchange(other._size, 0)),
          _hostVisible(std::exchange(other._hostVisible, false))
    {
    }

    Buffer& Buffer::operator=(Buffer&& other) noexcept
    {
        if (this != &other)
        {
            reset();
            _runtime = std::exchange(other._runtime, nullptr);
            _buffer = std::exchange(other._buffer, VK_NULL_HANDLE);
            _memory = std::exchange(other._memory, VK_NULL_HANDLE);
            _size = std::exchange(other._size, 0);
            _hostVisible = std::exchange(other._hostVisible, false);
        }
        return *this;
    }

    void* Buffer::map()
    {
        if (!_hostVisible)
            throw std::logic_error("Vulkan device-local buffer cannot be mapped");
        void* data = nullptr;
        checkVk(vkMapMemory(_runtime->device(), _memory, 0, _size, 0, &data), "vkMapMemory");
        return data;
    }

    void Buffer::unmap() noexcept
    {
        if (_runtime && _memory)
            vkUnmapMemory(_runtime->device(), _memory);
    }

    void Buffer::reset() noexcept
    {
        if (_runtime && _runtime->device())
        {
            if (_buffer)
                vkDestroyBuffer(_runtime->device(), _buffer, nullptr);
            if (_memory)
                vkFreeMemory(_runtime->device(), _memory, nullptr);
        }
        _runtime = nullptr;
        _buffer = VK_NULL_HANDLE;
        _memory = VK_NULL_HANDLE;
        _size = 0;
        _hostVisible = false;
    }

} // namespace plamatrix::internal::vulkan

#endif
