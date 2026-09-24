#pragma once

#include <cstddef>

#include "plamatrix/internal/core/memory_space.h"

namespace plamatrix::internal
{

inline namespace v1
{

class MemoryResource
{
public:
    virtual ~MemoryResource() = default;

    virtual void* allocate(std::size_t bytes, std::size_t alignment) = 0;
    virtual void deallocate(void* pointer, std::size_t bytes, std::size_t alignment) noexcept = 0;
    virtual MemorySpace memorySpace() const noexcept = 0;
};

} // namespace v1

} // namespace plamatrix::internal
