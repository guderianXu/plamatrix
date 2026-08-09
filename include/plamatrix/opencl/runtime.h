#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#ifdef PLAMATRIX_WITH_OPENCL
#include <CL/cl.h>

#include <mutex>
#include <unordered_map>
#endif

namespace plamatrix
{
namespace opencl
{

/// Stable information for one GPU in OpenCL platform/device enumeration order.
struct OpenClDeviceInfo
{
    std::size_t index = 0;
    std::string name;
    std::string vendor;
    std::string version;
    std::uint32_t computeUnits = 0;
    bool unifiedMemory = false;
    bool available = false;
    bool compilerAvailable = false;
};

/// Enumerate GPU devices in stable OpenCL platform/device order.
/// Returns an empty vector when PlaMatrix was built without OpenCL support.
std::vector<OpenClDeviceInfo> enumerateOpenClGpuDevices();

/// Return true when the selected OpenCL GPU and its online compiler can be initialized.
bool hasUsableOpenClDevice() noexcept;

/// Initialize the selected OpenCL GPU or throw a diagnostic selection error.
void requireUsableOpenClDevice();

/// Return the selected OpenCL device name, or an empty string when none is usable.
std::string selectedOpenClDeviceName();

/// Return the selected stable GPU index, or -1 when no device is usable.
int selectedOpenClDeviceIndex() noexcept;

#ifdef PLAMATRIX_WITH_OPENCL
namespace detail
{

/// Throw std::runtime_error when an OpenCL operation does not return CL_SUCCESS.
void checkOpenCl(cl_int error, const char* operation);

/// Process-wide owner of the selected OpenCL device, context, and compiled program cache.
class OpenClRuntime
{
public:
    static OpenClRuntime& instance();

    OpenClRuntime(const OpenClRuntime&) = delete;
    OpenClRuntime& operator=(const OpenClRuntime&) = delete;

    cl_context context() const noexcept
    {
        return _context;
    }

    cl_device_id device() const noexcept
    {
        return _device;
    }

    const std::string& deviceName() const noexcept
    {
        return _deviceName;
    }

    int deviceIndex() const noexcept
    {
        return _deviceIndex;
    }

    bool supportsFp64() const noexcept
    {
        return _supportsFp64;
    }

    /// Create an in-order command queue for this runtime's context and device.
    cl_command_queue createQueue() const;

    /// Build or retrieve a cached program. Cache identity includes key, source, and options.
    /// The returned handle is borrowed from the process-wide cache and must not be released.
    cl_program program(
        const std::string& key,
        const std::string& source,
        const std::string& options = {});

private:
    OpenClRuntime();
    ~OpenClRuntime();

    cl_context _context = nullptr;
    cl_device_id _device = nullptr;
    std::string _deviceName;
    int _deviceIndex = -1;
    bool _supportsFp64 = false;
    std::mutex _programMutex;
    std::unordered_map<std::string, cl_program> _programs;
};

} // namespace detail

// Stable public names. The detail namespace names remain available for
// compatibility with the first PlaPoint adapter implementation.
using OpenClRuntime = detail::OpenClRuntime;
using detail::checkOpenCl;
#endif

} // namespace opencl
} // namespace plamatrix
