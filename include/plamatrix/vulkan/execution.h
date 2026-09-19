#pragma once

#ifdef PLAMATRIX_WITH_VULKAN

#include <stdexcept>
#include <string>
#include <utility>
#include <unordered_map>
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
                        std::uint32_t pushConstantSize,
                        std::vector<VkAccessFlags> accessMasks = {});
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
        VkAccessFlags accessMask(std::uint32_t binding) const noexcept
        {
            return binding < _accessMasks.size() ? _accessMasks[binding]
                                                 : VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        }

    private:
        void reset() noexcept;
        Runtime* _runtime = nullptr;
        VkDescriptorSetLayout _descriptorLayout = VK_NULL_HANDLE;
        VkPipelineLayout _layout = VK_NULL_HANDLE;
        VkPipeline _pipeline = VK_NULL_HANDLE;
        std::vector<VkAccessFlags> _accessMasks;
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
        void copy(const Buffer& source, Buffer& destination, VkDeviceSize size);
        void submitAndWait();
        void resetSubmissionCount() noexcept
        {
            _submissionCount = 0;
            _barrierCount = 0;
            _gpuMilliseconds = 0.0;
        }
        std::uint32_t pendingDispatchCount() const noexcept
        {
            return _pendingDispatches;
        }
        std::uint32_t submissionCount() const noexcept
        {
            return _submissionCount;
        }
        std::uint32_t descriptorSetAllocations() const noexcept
        {
            return _descriptorSetAllocations;
        }
        std::uint32_t barrierCount() const noexcept
        {
            return _barrierCount;
        }
        double gpuMilliseconds() const noexcept
        {
            return _gpuMilliseconds;
        }

    private:
        struct CachedDescriptorSet
        {
            VkDescriptorSetLayout layout = VK_NULL_HANDLE;
            std::vector<VkBuffer> buffers;
            VkDescriptorSet set = VK_NULL_HANDLE;
        };

        void reset() noexcept;
        Runtime* _runtime = nullptr;
        DescriptorPool _descriptorPool;
        VkCommandBuffer _commandBuffer = VK_NULL_HANDLE;
        VkFence _fence = VK_NULL_HANDLE;
        VkQueryPool _timestampQueryPool = VK_NULL_HANDLE;
        bool _recording = false;
        std::uint32_t _pendingDispatches = 0;
        std::uint32_t _submissionCount = 0;
        std::uint32_t _descriptorSetAllocations = 0;
        std::uint32_t _barrierCount = 0;
        double _gpuMilliseconds = 0.0;
        std::unordered_map<VkBuffer, VkAccessFlags> _bufferAccess;
        std::vector<CachedDescriptorSet> _descriptorSets;
    };

} // namespace plamatrix::vulkan

#endif
