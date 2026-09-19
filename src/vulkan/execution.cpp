#include "plamatrix/vulkan/execution.h"

#ifdef PLAMATRIX_WITH_VULKAN

#include <array>
#include <limits>
#include <vector>

namespace plamatrix::vulkan
{
    namespace
    {

        void checkVk(VkResult result, const char* operation)
        {
            if (result != VK_SUCCESS)
                throw std::runtime_error(std::string("Vulkan ") + operation + " failed with error " +
                                         std::to_string(static_cast<int>(result)));
        }

    } // namespace

    ComputePipeline::ComputePipeline(Runtime& runtime,
                                     const char* shaderName,
                                     std::uint32_t bindingCount,
                                     std::uint32_t pushConstantSize)
        : _runtime(&runtime)
    {
        std::vector<VkDescriptorSetLayoutBinding> bindings(bindingCount);
        for (std::uint32_t index = 0; index < bindingCount; ++index)
        {
            bindings[index].binding = index;
            bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[index].descriptorCount = 1;
            bindings[index].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo layout_info{};
        layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout_info.bindingCount = bindingCount;
        layout_info.pBindings = bindings.data();
        checkVk(vkCreateDescriptorSetLayout(runtime.device(), &layout_info, nullptr, &_descriptorLayout),
                "vkCreateDescriptorSetLayout");

        VkPushConstantRange push{};
        push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push.offset = 0;
        push.size = pushConstantSize;
        VkPipelineLayoutCreateInfo pipeline_layout{};
        pipeline_layout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout.setLayoutCount = 1;
        pipeline_layout.pSetLayouts = &_descriptorLayout;
        pipeline_layout.pushConstantRangeCount = pushConstantSize == 0 ? 0u : 1u;
        pipeline_layout.pPushConstantRanges = pushConstantSize == 0 ? nullptr : &push;
        checkVk(vkCreatePipelineLayout(runtime.device(), &pipeline_layout, nullptr, &_layout),
                "vkCreatePipelineLayout");

        const auto words = runtime.loadShader(shaderName);
        VkShaderModuleCreateInfo shader_info{};
        shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        shader_info.codeSize = words.size() * sizeof(std::uint32_t);
        shader_info.pCode = words.data();
        VkShaderModule shader = VK_NULL_HANDLE;
        checkVk(vkCreateShaderModule(runtime.device(), &shader_info, nullptr, &shader), "vkCreateShaderModule");
        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = shader;
        stage.pName = "main";
        VkComputePipelineCreateInfo pipeline{};
        pipeline.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipeline.stage = stage;
        pipeline.layout = _layout;
        try
        {
            checkVk(vkCreateComputePipelines(runtime.device(), VK_NULL_HANDLE, 1, &pipeline, nullptr, &_pipeline),
                    "vkCreateComputePipelines");
            vkDestroyShaderModule(runtime.device(), shader, nullptr);
        }
        catch (...)
        {
            vkDestroyShaderModule(runtime.device(), shader, nullptr);
            reset();
            throw;
        }
    }

    ComputePipeline::~ComputePipeline() noexcept
    {
        reset();
    }

    ComputePipeline::ComputePipeline(ComputePipeline&& other) noexcept
        : _runtime(std::exchange(other._runtime, nullptr)),
          _descriptorLayout(std::exchange(other._descriptorLayout, VK_NULL_HANDLE)),
          _layout(std::exchange(other._layout, VK_NULL_HANDLE)),
          _pipeline(std::exchange(other._pipeline, VK_NULL_HANDLE))
    {
    }

    ComputePipeline& ComputePipeline::operator=(ComputePipeline&& other) noexcept
    {
        if (this != &other)
        {
            reset();
            _runtime = std::exchange(other._runtime, nullptr);
            _descriptorLayout = std::exchange(other._descriptorLayout, VK_NULL_HANDLE);
            _layout = std::exchange(other._layout, VK_NULL_HANDLE);
            _pipeline = std::exchange(other._pipeline, VK_NULL_HANDLE);
        }
        return *this;
    }

    void ComputePipeline::reset() noexcept
    {
        if (_runtime && _runtime->device())
        {
            if (_pipeline)
                vkDestroyPipeline(_runtime->device(), _pipeline, nullptr);
            if (_layout)
                vkDestroyPipelineLayout(_runtime->device(), _layout, nullptr);
            if (_descriptorLayout)
                vkDestroyDescriptorSetLayout(_runtime->device(), _descriptorLayout, nullptr);
        }
        _runtime = nullptr;
        _pipeline = VK_NULL_HANDLE;
        _layout = VK_NULL_HANDLE;
        _descriptorLayout = VK_NULL_HANDLE;
    }

    DescriptorPool::DescriptorPool(Runtime& runtime, std::uint32_t maxSets, std::uint32_t storageBuffers)
        : _runtime(&runtime)
    {
        VkDescriptorPoolSize size{};
        size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        size.descriptorCount = storageBuffers;
        VkDescriptorPoolCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        info.maxSets = maxSets;
        info.poolSizeCount = 1;
        info.pPoolSizes = &size;
        checkVk(vkCreateDescriptorPool(runtime.device(), &info, nullptr, &_pool), "vkCreateDescriptorPool");
    }

    DescriptorPool::~DescriptorPool() noexcept
    {
        reset();
    }

    DescriptorPool::DescriptorPool(DescriptorPool&& other) noexcept
        : _runtime(std::exchange(other._runtime, nullptr)), _pool(std::exchange(other._pool, VK_NULL_HANDLE))
    {
    }

    DescriptorPool& DescriptorPool::operator=(DescriptorPool&& other) noexcept
    {
        if (this != &other)
        {
            reset();
            _runtime = std::exchange(other._runtime, nullptr);
            _pool = std::exchange(other._pool, VK_NULL_HANDLE);
        }
        return *this;
    }

    void DescriptorPool::reset() noexcept
    {
        if (_runtime && _pool)
            vkDestroyDescriptorPool(_runtime->device(), _pool, nullptr);
        _runtime = nullptr;
        _pool = VK_NULL_HANDLE;
    }

    void DescriptorPool::resetForReuse()
    {
        if (_runtime && _pool)
            checkVk(vkResetDescriptorPool(_runtime->device(), _pool, 0), "vkResetDescriptorPool");
    }

    CommandContext::CommandContext(Runtime& runtime, std::uint32_t maxSets, std::uint32_t storageBufferDescriptors)
        : _runtime(&runtime), _descriptorPool(runtime, maxSets, storageBufferDescriptors)
    {
        VkCommandBufferAllocateInfo allocation{};
        allocation.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocation.commandPool = runtime.commandPool();
        allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocation.commandBufferCount = 1;
        checkVk(vkAllocateCommandBuffers(runtime.device(), &allocation, &_commandBuffer), "vkAllocateCommandBuffers");

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        try
        {
            checkVk(vkCreateFence(runtime.device(), &fenceInfo, nullptr, &_fence), "vkCreateFence");
        }
        catch (...)
        {
            vkFreeCommandBuffers(runtime.device(), runtime.commandPool(), 1, &_commandBuffer);
            _commandBuffer = VK_NULL_HANDLE;
            throw;
        }
    }

    CommandContext::~CommandContext() noexcept
    {
        reset();
    }

    CommandContext::CommandContext(CommandContext&& other) noexcept
        : _runtime(std::exchange(other._runtime, nullptr)), _descriptorPool(std::move(other._descriptorPool)),
          _commandBuffer(std::exchange(other._commandBuffer, VK_NULL_HANDLE)),
          _fence(std::exchange(other._fence, VK_NULL_HANDLE)), _recording(std::exchange(other._recording, false)),
          _pendingDispatches(std::exchange(other._pendingDispatches, 0)),
          _submissionCount(std::exchange(other._submissionCount, 0))
    {
    }

    CommandContext& CommandContext::operator=(CommandContext&& other) noexcept
    {
        if (this != &other)
        {
            reset();
            _runtime = std::exchange(other._runtime, nullptr);
            _descriptorPool = std::move(other._descriptorPool);
            _commandBuffer = std::exchange(other._commandBuffer, VK_NULL_HANDLE);
            _fence = std::exchange(other._fence, VK_NULL_HANDLE);
            _recording = std::exchange(other._recording, false);
            _pendingDispatches = std::exchange(other._pendingDispatches, 0);
            _submissionCount = std::exchange(other._submissionCount, 0);
        }
        return *this;
    }

    void CommandContext::begin()
    {
        if (!_runtime || !_commandBuffer || !_fence)
            throw std::runtime_error("Vulkan command context is not initialized");
        if (_recording)
            throw std::logic_error("Vulkan command context is already recording");
        _descriptorPool.resetForReuse();
        checkVk(vkResetCommandBuffer(_commandBuffer, 0), "vkResetCommandBuffer");
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        checkVk(vkBeginCommandBuffer(_commandBuffer, &beginInfo), "vkBeginCommandBuffer");
        _pendingDispatches = 0;
        _recording = true;
    }

    void CommandContext::dispatch(const ComputePipeline& pipeline,
                                  const std::vector<Buffer*>& buffers,
                                  std::size_t groups,
                                  const void* pushData,
                                  std::uint32_t pushSize)
    {
        if (!_recording)
            throw std::logic_error("Vulkan command context is not recording");
        if (groups == 0 || groups > std::numeric_limits<std::uint32_t>::max())
            throw std::invalid_argument("Vulkan dispatch group count is out of range");

        VkDescriptorSetAllocateInfo allocation{};
        allocation.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocation.descriptorPool = _descriptorPool.handle();
        allocation.descriptorSetCount = 1;
        const VkDescriptorSetLayout layout = pipeline.descriptorLayout();
        allocation.pSetLayouts = &layout;
        VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
        checkVk(vkAllocateDescriptorSets(_runtime->device(), &allocation, &descriptorSet), "vkAllocateDescriptorSets");

        std::vector<VkDescriptorBufferInfo> infos(buffers.size());
        std::vector<VkWriteDescriptorSet> writes(buffers.size());
        for (std::size_t index = 0; index < buffers.size(); ++index)
        {
            if (!buffers[index] || buffers[index]->handle() == VK_NULL_HANDLE)
                throw std::invalid_argument("Vulkan dispatch received an empty buffer");
            infos[index].buffer = buffers[index]->handle();
            infos[index].offset = 0;
            infos[index].range = buffers[index]->size();
            writes[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[index].dstSet = descriptorSet;
            writes[index].dstBinding = static_cast<std::uint32_t>(index);
            writes[index].descriptorCount = 1;
            writes[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[index].pBufferInfo = &infos[index];
        }
        vkUpdateDescriptorSets(
            _runtime->device(), static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);

        VkMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(_commandBuffer,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0,
                             1,
                             &barrier,
                             0,
                             nullptr,
                             0,
                             nullptr);
        vkCmdBindPipeline(_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline());
        vkCmdBindDescriptorSets(
            _commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.layout(), 0, 1, &descriptorSet, 0, nullptr);
        if (pushSize != 0)
            vkCmdPushConstants(_commandBuffer, pipeline.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, pushSize, pushData);
        vkCmdDispatch(_commandBuffer, static_cast<std::uint32_t>(groups), 1, 1);
        ++_pendingDispatches;
    }

    void CommandContext::copy(const Buffer& source, Buffer& destination, VkDeviceSize size)
    {
        if (!_recording)
            throw std::logic_error("Vulkan command context is not recording");
        if (size == 0 || size > source.size() || size > destination.size())
            throw std::invalid_argument("Vulkan buffer copy size is out of range");
        VkBufferCopy region{};
        region.size = size;
        vkCmdCopyBuffer(_commandBuffer, source.handle(), destination.handle(), 1, &region);
    }

    void CommandContext::submitAndWait()
    {
        if (!_recording)
            throw std::logic_error("Vulkan command context is not recording");
        try
        {
            checkVk(vkEndCommandBuffer(_commandBuffer), "vkEndCommandBuffer");
            checkVk(vkResetFences(_runtime->device(), 1, &_fence), "vkResetFences");
            VkSubmitInfo submit{};
            submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit.commandBufferCount = 1;
            submit.pCommandBuffers = &_commandBuffer;
            checkVk(vkQueueSubmit(_runtime->queue(), 1, &submit, _fence), "vkQueueSubmit");
            checkVk(vkWaitForFences(_runtime->device(), 1, &_fence, VK_TRUE, std::numeric_limits<std::uint64_t>::max()),
                    "vkWaitForFences");
            _recording = false;
            ++_submissionCount;
        }
        catch (...)
        {
            _recording = false;
            throw;
        }
    }

    void CommandContext::reset() noexcept
    {
        if (!_runtime)
            return;
        if (_recording)
            vkDeviceWaitIdle(_runtime->device());
        if (_fence)
            vkDestroyFence(_runtime->device(), _fence, nullptr);
        if (_commandBuffer)
            vkFreeCommandBuffers(_runtime->device(), _runtime->commandPool(), 1, &_commandBuffer);
        _fence = VK_NULL_HANDLE;
        _commandBuffer = VK_NULL_HANDLE;
        _recording = false;
        _pendingDispatches = 0;
        _submissionCount = 0;
        _runtime = nullptr;
    }

} // namespace plamatrix::vulkan

#endif
