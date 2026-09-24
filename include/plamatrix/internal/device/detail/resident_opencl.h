#pragma once

#include <cstddef>

#include "plamatrix/internal/core/execution_context.h"

namespace plamatrix::internal::v1::resident_detail
{

#ifdef PLAMATRIX_WITH_OPENCL
void* allocateOpenCl(ExecutionContext& context, std::size_t bytes);
void releaseOpenCl(void* handle) noexcept;
void copyResidentOpenCl(ExecutionContext& context, void* source, void* destination, std::size_t bytes);
void copyFromHostOpenCl(ExecutionContext& context, void* handle, const void* source, std::size_t bytes);
void copyToHostOpenCl(ExecutionContext& context, void* handle, void* destination, std::size_t bytes);
#endif

} // namespace plamatrix::internal::v1::resident_detail
