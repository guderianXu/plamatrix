#pragma once

#ifdef PLAMATRIX_WITH_VULKAN

#include <stdexcept>
#include <string>
#include <utility>

#include "plamatrix/vulkan/runtime.h"

namespace plamatrix::vulkan
{

    class ComputePipeline
    {
    public:
        ComputePipeline() = default;
        ComputePipeline(Runtime& runtime,
                        const char* shaderName,
                        std::uint32_t bindingCount,
                        std::uint32_t pushConstantSize);
        ~ComputePipeline() noexcept;
        ComputePipeline(const ComputePipeline&) = delete;
        ComputePipeline& operator=(const ComputePipeline&) = delete;
        ComputePipeline(ComputePipeline&& other) noexcept;
        ComputePipeline& operator=(ComputePipeline&& other) noexcept;

        VkPipeline pipeline() const noexcept
        {
            return _pipeline;
        }
        VkPipelineLayout layout() const noexcept
        {
            return _layout;
        }
        VkDescriptorSetLayout descriptorLayout() const noexcept
        {
            return _descriptorLayout;
        }

    private:
        void reset() noexcept;
        Runtime* _runtime = nullptr;
        VkDescriptorSetLayout _descriptorLayout = VK_NULL_HANDLE;
        VkPipelineLayout _layout = VK_NULL_HANDLE;
        VkPipeline _pipeline = VK_NULL_HANDLE;
    };

    class DescriptorPool
    {
    public:
        DescriptorPool() = default;
        explicit DescriptorPool(Runtime& runtime, std::uint32_t maxSets, std::uint32_t storageBuffers);
        ~DescriptorPool() noexcept;
        DescriptorPool(const DescriptorPool&) = delete;
        DescriptorPool& operator=(const DescriptorPool&) = delete;
        DescriptorPool(DescriptorPool&& other) noexcept;
        DescriptorPool& operator=(DescriptorPool&& other) noexcept;
        VkDescriptorPool handle() const noexcept
        {
            return _pool;
        }

    private:
        void reset() noexcept;
        Runtime* _runtime = nullptr;
        VkDescriptorPool _pool = VK_NULL_HANDLE;
    };

} // namespace plamatrix::vulkan

#endif
