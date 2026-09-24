#pragma once

#include <cstddef>

#include "plamatrix/internal/core/execution_context.h"

namespace plamatrix::internal::v1::resident_detail
{

#ifdef PLAMATRIX_WITH_VULKAN
void* allocateVulkan(ExecutionContext& context, std::size_t bytes);
void releaseVulkan(void* handle) noexcept;
void copyResidentVulkan(ExecutionContext& context, void* source, void* destination, std::size_t bytes);
void copyFromHostVulkan(ExecutionContext& context, void* handle, const void* source, std::size_t bytes);
void copyToHostVulkan(ExecutionContext& context, void* handle, void* destination, std::size_t bytes);
#endif

} // namespace plamatrix::internal::v1::resident_detail
