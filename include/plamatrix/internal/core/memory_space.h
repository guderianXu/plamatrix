#pragma once

namespace plamatrix::internal
{

inline namespace v1
{

enum class MemorySpace
{
    Host,
    HostPinned,
    Cuda,
    OpenCl,
    Vulkan
};

} // namespace v1

} // namespace plamatrix::internal
