#pragma once

#ifdef PLAMATRIX_WITH_VULKAN

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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
        void resetForReuse();

    private:
        void reset() noexcept;
        Runtime* _runtime = nullptr;
        VkDescriptorPool _pool = VK_NULL_HANDLE;
    };

    class CommandContext
    {
    public:
        explicit CommandContext(Runtime& runtime,
                                std::uint32_t maxSets = 64,
                                std::uint32_t storageBufferDescriptors = 512);
        ~CommandContext() noexcept;
        CommandContext(const CommandContext&) = delete;
        CommandContext& operator=(const CommandContext&) = delete;
        CommandContext(CommandContext&& other) noexcept;
        CommandContext& operator=(CommandContext&& other) noexcept;

        void begin();
        void dispatch(const ComputePipeline& pipeline,
                      const std::vector<Buffer*>& buffers,
                      std::size_t groups,
                      const void* pushData,
                      std::uint32_t pushSize);
        void submitAndWait();
        std::uint32_t pendingDispatchCount() const noexcept
        {
            return _pendingDispatches;
        }
        std::uint32_t submissionCount() const noexcept
        {
            return _submissionCount;
        }

    private:
        void reset() noexcept;
        Runtime* _runtime = nullptr;
        DescriptorPool _descriptorPool;
        VkCommandBuffer _commandBuffer = VK_NULL_HANDLE;
        VkFence _fence = VK_NULL_HANDLE;
        bool _recording = false;
        std::uint32_t _pendingDispatches = 0;
        std::uint32_t _submissionCount = 0;
    };

} // namespace plamatrix::vulkan

#endif
