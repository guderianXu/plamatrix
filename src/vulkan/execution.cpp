#include "plamatrix/internal/vulkan/execution.h"

#ifdef PLAMATRIX_WITH_VULKAN

#include <algorithm>
#include <array>
#include <limits>
#include <vector>

namespace plamatrix::internal::vulkan
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
                                     std::uint32_t pushConstantSize,
                                     std::vector<VkAccessFlags> accessMasks)
        : _runtime(&runtime), _accessMasks(std::move(accessMasks))
    {
        if (!_accessMasks.empty() && _accessMasks.size() != bindingCount)
            throw std::invalid_argument("Vulkan pipeline access mask count does not match binding count");
        if (_accessMasks.empty())
            _accessMasks.assign(bindingCount, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
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
          _pipeline(std::exchange(other._pipeline, VK_NULL_HANDLE)), _accessMasks(std::move(other._accessMasks))
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
            _accessMasks = std::move(other._accessMasks);
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
        _accessMasks.clear();
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
            VkQueryPoolCreateInfo queryInfo{};
            queryInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
            queryInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
            queryInfo.queryCount = 2;
            if (vkCreateQueryPool(runtime.device(), &queryInfo, nullptr, &_timestampQueryPool) != VK_SUCCESS)
                _timestampQueryPool = VK_NULL_HANDLE;
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
          _fence(std::exchange(other._fence, VK_NULL_HANDLE)),
          _timestampQueryPool(std::exchange(other._timestampQueryPool, VK_NULL_HANDLE)),
          _recording(std::exchange(other._recording, false)), _executable(std::exchange(other._executable, false)),
          _reusable(std::exchange(other._reusable, false)),
          _pendingDispatches(std::exchange(other._pendingDispatches, 0)),
          _submissionCount(std::exchange(other._submissionCount, 0)),
          _descriptorSetAllocations(std::exchange(other._descriptorSetAllocations, 0)),
          _recordedBarrierCount(std::exchange(other._recordedBarrierCount, 0)),
          _barrierCount(std::exchange(other._barrierCount, 0)),
          _commandBufferRecordings(std::exchange(other._commandBufferRecordings, 0)),
          _gpuMilliseconds(std::exchange(other._gpuMilliseconds, 0.0)), _bufferAccess(std::move(other._bufferAccess)),
          _descriptorSets(std::move(other._descriptorSets))
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
            _timestampQueryPool = std::exchange(other._timestampQueryPool, VK_NULL_HANDLE);
            _recording = std::exchange(other._recording, false);
            _executable = std::exchange(other._executable, false);
            _reusable = std::exchange(other._reusable, false);
            _pendingDispatches = std::exchange(other._pendingDispatches, 0);
            _submissionCount = std::exchange(other._submissionCount, 0);
            _descriptorSetAllocations = std::exchange(other._descriptorSetAllocations, 0);
            _recordedBarrierCount = std::exchange(other._recordedBarrierCount, 0);
            _barrierCount = std::exchange(other._barrierCount, 0);
            _commandBufferRecordings = std::exchange(other._commandBufferRecordings, 0);
            _gpuMilliseconds = std::exchange(other._gpuMilliseconds, 0.0);
            _bufferAccess = std::move(other._bufferAccess);
            _descriptorSets = std::move(other._descriptorSets);
        }
        return *this;
    }

    void CommandContext::begin()
    {
        beginRecording(false);
    }

    void CommandContext::beginReusable()
    {
        beginRecording(true);
    }

    void CommandContext::beginRecording(bool reusable)
    {
        if (!_runtime || !_commandBuffer || !_fence)
            throw std::runtime_error("Vulkan command context is not initialized");
        if (_recording)
            throw std::logic_error("Vulkan command context is already recording");
        _executable = false;
        _reusable = false;
        checkVk(vkResetCommandBuffer(_commandBuffer, 0), "vkResetCommandBuffer");
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = reusable ? 0u : VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        checkVk(vkBeginCommandBuffer(_commandBuffer, &beginInfo), "vkBeginCommandBuffer");
        if (_timestampQueryPool)
        {
            vkCmdResetQueryPool(_commandBuffer, _timestampQueryPool, 0, 2);
            vkCmdWriteTimestamp(_commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, _timestampQueryPool, 0);
        }
        _pendingDispatches = 0;
        _recordedBarrierCount = 0;
        _bufferAccess.clear();
        _reusable = reusable;
        _recording = true;
        ++_commandBufferRecordings;
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

        const VkDescriptorSetLayout layout = pipeline.descriptorLayout();
        std::vector<VkBuffer> handles;
        handles.reserve(buffers.size());
        for (std::size_t index = 0; index < buffers.size(); ++index)
        {
            if (!buffers[index] || buffers[index]->handle() == VK_NULL_HANDLE)
                throw std::invalid_argument("Vulkan dispatch received an empty buffer");
            handles.push_back(buffers[index]->handle());
        }
        VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
        const auto cached = std::find_if(_descriptorSets.begin(),
                                         _descriptorSets.end(),
                                         [&](const CachedDescriptorSet& entry)
                                         { return entry.layout == layout && entry.buffers == handles; });
        if (cached != _descriptorSets.end())
        {
            descriptorSet = cached->set;
        }
        else
        {
            VkDescriptorSetAllocateInfo allocation{};
            allocation.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            allocation.descriptorPool = _descriptorPool.handle();
            allocation.descriptorSetCount = 1;
            allocation.pSetLayouts = &layout;
            checkVk(vkAllocateDescriptorSets(_runtime->device(), &allocation, &descriptorSet),
                    "vkAllocateDescriptorSets");

            std::vector<VkDescriptorBufferInfo> infos(buffers.size());
            std::vector<VkWriteDescriptorSet> writes(buffers.size());
            for (std::size_t index = 0; index < buffers.size(); ++index)
            {
                infos[index].buffer = handles[index];
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
            _descriptorSets.push_back(CachedDescriptorSet{layout, handles, descriptorSet});
            ++_descriptorSetAllocations;
        }

        constexpr VkAccessFlags writeAccess = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        constexpr VkAccessFlags transferAccess = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        std::vector<VkBufferMemoryBarrier> barriers;
        VkPipelineStageFlags sourceStages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        barriers.reserve(buffers.size());
        for (std::size_t index = 0; index < buffers.size(); ++index)
        {
            const VkBuffer handle = handles[index];
            const VkAccessFlags currentAccess = pipeline.accessMask(static_cast<std::uint32_t>(index));
            const auto previous = _bufferAccess.find(handle);
            if (previous != _bufferAccess.end() && ((previous->second | currentAccess) & writeAccess) != 0)
            {
                VkBufferMemoryBarrier barrier{};
                barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                barrier.srcAccessMask = previous->second;
                barrier.dstAccessMask = currentAccess;
                barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.buffer = handle;
                barrier.offset = 0;
                barrier.size = buffers[index]->size();
                barriers.push_back(barrier);
                if ((previous->second & transferAccess) != 0)
                    sourceStages |= VK_PIPELINE_STAGE_TRANSFER_BIT;
            }
        }
        if (!barriers.empty())
        {
            vkCmdPipelineBarrier(_commandBuffer,
                                 sourceStages,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0,
                                 0,
                                 nullptr,
                                 static_cast<std::uint32_t>(barriers.size()),
                                 barriers.data(),
                                 0,
                                 nullptr);
            ++_recordedBarrierCount;
        }
        vkCmdBindPipeline(_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline());
        vkCmdBindDescriptorSets(
            _commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.layout(), 0, 1, &descriptorSet, 0, nullptr);
        if (pushSize != 0)
            vkCmdPushConstants(_commandBuffer, pipeline.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, pushSize, pushData);
        vkCmdDispatch(_commandBuffer, static_cast<std::uint32_t>(groups), 1, 1);
        for (std::size_t index = 0; index < handles.size(); ++index)
            _bufferAccess[handles[index]] = pipeline.accessMask(static_cast<std::uint32_t>(index));
        ++_pendingDispatches;
    }

    void CommandContext::previousSubmissionBarrier(VkPipelineStageFlags destinationStages,
                                                   VkAccessFlags destinationAccess)
    {
        if (!_recording)
            throw std::logic_error("Vulkan command context is not recording");
        if (destinationStages == 0 || destinationAccess == 0)
            throw std::invalid_argument("Vulkan previous-submission barrier requires destination stages and access");
        VkMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = destinationAccess;
        vkCmdPipelineBarrier(_commandBuffer,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                             destinationStages,
                             0,
                             1,
                             &barrier,
                             0,
                             nullptr,
                             0,
                             nullptr);
        ++_recordedBarrierCount;
    }

    void CommandContext::copy(const Buffer& source, Buffer& destination, VkDeviceSize size)
    {
        if (!_recording)
            throw std::logic_error("Vulkan command context is not recording");
        if (size == 0 || size > source.size() || size > destination.size())
            throw std::invalid_argument("Vulkan buffer copy size is out of range");
        constexpr VkAccessFlags writeAccess = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        std::vector<VkBufferMemoryBarrier> barriers;
        const auto addBarrier = [&](const Buffer& buffer, VkAccessFlags destinationAccess)
        {
            const auto previous = _bufferAccess.find(buffer.handle());
            if (previous == _bufferAccess.end() || ((previous->second | destinationAccess) & writeAccess) == 0)
                return;
            VkBufferMemoryBarrier barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barrier.srcAccessMask = previous->second;
            barrier.dstAccessMask = destinationAccess;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.buffer = buffer.handle();
            barrier.offset = 0;
            barrier.size = size;
            barriers.push_back(barrier);
        };
        addBarrier(source, VK_ACCESS_TRANSFER_READ_BIT);
        addBarrier(destination, VK_ACCESS_TRANSFER_WRITE_BIT);
        if (!barriers.empty())
        {
            vkCmdPipelineBarrier(_commandBuffer,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 0,
                                 0,
                                 nullptr,
                                 static_cast<std::uint32_t>(barriers.size()),
                                 barriers.data(),
                                 0,
                                 nullptr);
            ++_recordedBarrierCount;
        }
        VkBufferCopy region{};
        region.size = size;
        vkCmdCopyBuffer(_commandBuffer, source.handle(), destination.handle(), 1, &region);
        _bufferAccess[source.handle()] = VK_ACCESS_TRANSFER_READ_BIT;
        _bufferAccess[destination.handle()] = VK_ACCESS_TRANSFER_WRITE_BIT;
    }

    void CommandContext::submitAndWait()
    {
        if (!_recording && !_executable)
            throw std::logic_error("Vulkan command context has no executable recording");
        try
        {
            if (_recording)
            {
                if (_timestampQueryPool)
                    vkCmdWriteTimestamp(_commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, _timestampQueryPool, 1);
                checkVk(vkEndCommandBuffer(_commandBuffer), "vkEndCommandBuffer");
                _recording = false;
                _executable = true;
            }
            checkVk(vkResetFences(_runtime->device(), 1, &_fence), "vkResetFences");
            VkSubmitInfo submit{};
            submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit.commandBufferCount = 1;
            submit.pCommandBuffers = &_commandBuffer;
            checkVk(vkQueueSubmit(_runtime->queue(), 1, &submit, _fence), "vkQueueSubmit");
            checkVk(vkWaitForFences(_runtime->device(), 1, &_fence, VK_TRUE, std::numeric_limits<std::uint64_t>::max()),
                    "vkWaitForFences");
            if (_timestampQueryPool && _runtime->timestampPeriodNanoseconds() > 0.0f)
            {
                std::uint64_t timestamps[2] = {};
                const VkResult result = vkGetQueryPoolResults(_runtime->device(),
                                                              _timestampQueryPool,
                                                              0,
                                                              2,
                                                              sizeof(timestamps),
                                                              timestamps,
                                                              sizeof(std::uint64_t),
                                                              VK_QUERY_RESULT_64_BIT);
                if (result == VK_SUCCESS && timestamps[1] >= timestamps[0])
                {
                    const double nanoseconds = static_cast<double>(timestamps[1] - timestamps[0]) *
                                               static_cast<double>(_runtime->timestampPeriodNanoseconds());
                    _gpuMilliseconds += nanoseconds * 1.0e-6;
                }
            }
            ++_submissionCount;
            _barrierCount += _recordedBarrierCount;
            if (!_reusable)
                _executable = false;
        }
        catch (...)
        {
            _recording = false;
            _executable = false;
            throw;
        }
    }

    void CommandContext::reset() noexcept
    {
        if (!_runtime)
            return;
        if (_recording)
            vkDeviceWaitIdle(_runtime->device());
        if (_timestampQueryPool)
            vkDestroyQueryPool(_runtime->device(), _timestampQueryPool, nullptr);
        if (_fence)
            vkDestroyFence(_runtime->device(), _fence, nullptr);
        if (_commandBuffer)
            vkFreeCommandBuffers(_runtime->device(), _runtime->commandPool(), 1, &_commandBuffer);
        _fence = VK_NULL_HANDLE;
        _timestampQueryPool = VK_NULL_HANDLE;
        _commandBuffer = VK_NULL_HANDLE;
        _recording = false;
        _executable = false;
        _reusable = false;
        _pendingDispatches = 0;
        _submissionCount = 0;
        _descriptorSetAllocations = 0;
        _recordedBarrierCount = 0;
        _barrierCount = 0;
        _commandBufferRecordings = 0;
        _gpuMilliseconds = 0.0;
        _bufferAccess.clear();
        _descriptorSets.clear();
        _runtime = nullptr;
    }

} // namespace plamatrix::internal::vulkan

#endif
