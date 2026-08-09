#include "plamatrix/opencl/runtime.h"

#include <stdexcept>

namespace plamatrix
{
namespace opencl
{

std::vector<OpenClDeviceInfo> enumerateOpenClGpuDevices()
{
    return {};
}

bool hasUsableOpenClDevice() noexcept
{
    return false;
}

void requireUsableOpenClDevice()
{
    throw std::runtime_error("PlaMatrix was built without OpenCL support");
}

std::string selectedOpenClDeviceName()
{
    return {};
}

int selectedOpenClDeviceIndex() noexcept
{
    return -1;
}

} // namespace opencl
} // namespace plamatrix
