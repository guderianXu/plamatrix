#pragma once

#include <cstddef>
#include <cstdint>

namespace plamatrix
{

enum class Device : int
{
    CPU = 0,
    GPU = 1
};

using Index = std::int64_t;

} // namespace plamatrix
