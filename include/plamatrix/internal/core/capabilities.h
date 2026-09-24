#pragma once

namespace plamatrix::internal
{

inline namespace v1
{

struct CapabilitySet
{
    bool float32 = false;
    bool float64 = false;
    bool hostMemory = false;
    bool deviceMemory = false;
    bool asynchronousAllocation = false;
    bool commandBuffers = false;
    bool cooperativeMatrix = false;
};

} // namespace v1

} // namespace plamatrix::internal
