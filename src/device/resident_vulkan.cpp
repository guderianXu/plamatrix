#include "plamatrix/internal/device/detail/resident_vulkan.h"

#ifdef PLAMATRIX_WITH_VULKAN

#include <cstring>

#include "plamatrix/internal/vulkan/native.h"
#include "plamatrix/internal/vulkan/execution.h"

namespace plamatrix::internal::v1::resident_detail
{

void* allocateVulkan(ExecutionContext& context, std::size_t bytes)
{
    return new vulkan::Buffer(vulkan::NativeAccess::runtime(context),
                              bytes,
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                  VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                              vulkan::BufferMemory::HostVisible);
}

void releaseVulkan(void* handle) noexcept
{
    delete static_cast<vulkan::Buffer*>(handle);
}

void copyResidentVulkan(ExecutionContext& context, void* source, void* destination, std::size_t bytes)
{
    vulkan::CommandContext commands(vulkan::NativeAccess::runtime(context));
    commands.begin();
    commands.previousSubmissionBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                                       VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
    commands.copy(*static_cast<vulkan::Buffer*>(source), *static_cast<vulkan::Buffer*>(destination), bytes);
    commands.submitAndWait();
}

void copyFromHostVulkan(ExecutionContext& context, void* handle, const void* source, std::size_t bytes)
{
    context.synchronize();
    auto& buffer = *static_cast<vulkan::Buffer*>(handle);
    void* mapped = buffer.map();
    std::memcpy(mapped, source, bytes);
    buffer.unmap();
}

void copyToHostVulkan(ExecutionContext& context, void* handle, void* destination, std::size_t bytes)
{
    context.synchronize();
    auto& buffer = *static_cast<vulkan::Buffer*>(handle);
    void* mapped = buffer.map();
    std::memcpy(destination, mapped, bytes);
    buffer.unmap();
}

} // namespace plamatrix::internal::v1::resident_detail

#endif
