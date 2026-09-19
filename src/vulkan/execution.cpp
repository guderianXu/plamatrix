#include "plamatrix/vulkan/execution.h"

#ifdef PLAMATRIX_WITH_VULKAN

#include <array>
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

} // namespace plamatrix::vulkan

#endif
