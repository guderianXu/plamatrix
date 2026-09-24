#pragma once

#include <cstddef>

namespace plamatrix::internal
{

inline namespace v1
{

enum class Backend
{
    Cpu,
    Cuda,
    OpenCl,
    Vulkan
};

struct DeviceId
{
    Backend backend = Backend::Cpu;
    std::size_t index = 0;
};

} // namespace v1

} // namespace plamatrix::internal
