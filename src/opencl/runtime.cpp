#include "plamatrix/opencl/runtime.h"

#include "device_enumeration.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace plamatrix
{
namespace opencl
{
namespace detail
{
namespace
{

struct EnvironmentValue
{
    const char* value = nullptr;
    const char* name = nullptr;
};

std::string lowerCase(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character)
    {
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

EnvironmentValue environmentValue(const char* preferred, const char* legacy)
{
    if (const char* value = std::getenv(preferred))
    {
        return {value, preferred};
    }
    if (const char* value = std::getenv(legacy))
    {
        return {value, legacy};
    }
    return {};
}

std::string programCacheKey(
    const std::string& caller_key,
    const std::string& source,
    const std::string& options)
{
    return std::to_string(caller_key.size()) + ":" + caller_key
        + std::to_string(source.size()) + ":" + source
        + std::to_string(options.size()) + ":" + options;
}

} // namespace

void checkOpenCl(cl_int error, const char* operation)
{
    if (error != CL_SUCCESS)
    {
        throw std::runtime_error(
            std::string("OpenCL ") + (operation ? operation : "operation")
                + " failed with error " + std::to_string(error));
    }
}

OpenClRuntime& OpenClRuntime::instance()
{
    static OpenClRuntime runtime;
    return runtime;
}

OpenClRuntime::OpenClRuntime()
{
    std::vector<std::string> enumeration_diagnostics;
    std::vector<DeviceCandidate> candidates = enumerateDeviceCandidates(&enumeration_diagnostics);
    candidates.erase(
        std::remove_if(candidates.begin(), candidates.end(), [](const DeviceCandidate& candidate)
        {
            return candidate.available == CL_FALSE || candidate.compilerAvailable == CL_FALSE;
        }),
        candidates.end());
    if (candidates.empty())
    {
        std::string message = "OpenCL has no usable GPU device with an online compiler";
        if (!enumeration_diagnostics.empty())
        {
            message += "; enumeration diagnostics: ";
            for (std::size_t index = 0; index < enumeration_diagnostics.size(); ++index)
            {
                if (index != 0)
                {
                    message += "; ";
                }
                message += enumeration_diagnostics[index];
            }
        }
        throw std::runtime_error(message);
    }

    const EnvironmentValue requested_index = environmentValue(
        "PLAMATRIX_OPENCL_DEVICE_INDEX", "PLAPOINT_OPENCL_DEVICE_INDEX");
    if (requested_index.value && requested_index.value[0] != '\0'
        && std::string(requested_index.value) != "-1")
    {
        std::size_t parsed = 0;
        long long index = -1;
        try
        {
            index = std::stoll(requested_index.value, &parsed);
        }
        catch (const std::exception&)
        {
            throw std::runtime_error(
                std::string(requested_index.name)
                    + " must be -1 or a non-negative integer");
        }
        if (parsed != std::string(requested_index.value).size() || index < 0)
        {
            throw std::runtime_error(
                std::string(requested_index.name)
                    + " must be -1 or a non-negative integer");
        }
        candidates.erase(
            std::remove_if(candidates.begin(), candidates.end(), [&](const DeviceCandidate& candidate)
            {
                return candidate.index != static_cast<std::size_t>(index);
            }),
            candidates.end());
        if (candidates.empty())
        {
            throw std::runtime_error(
                std::string(requested_index.name)
                    + " does not identify a usable OpenCL GPU: " + std::to_string(index));
        }
    }
    else
    {
        const EnvironmentValue requested_device = environmentValue(
            "PLAMATRIX_OPENCL_DEVICE", "PLAPOINT_OPENCL_DEVICE");
        if (requested_device.value && requested_device.value[0] != '\0')
        {
            const std::string requested = lowerCase(requested_device.value);
            candidates.erase(
                std::remove_if(candidates.begin(), candidates.end(), [&](const DeviceCandidate& candidate)
                {
                    return lowerCase(candidate.vendor + " " + candidate.name).find(requested)
                        == std::string::npos;
                }),
                candidates.end());
            if (candidates.empty())
            {
                throw std::runtime_error(
                    std::string(requested_device.name)
                        + " did not match any usable GPU device: " + requested);
            }
        }
    }

    std::sort(candidates.begin(), candidates.end(), candidateBetter);
    if (candidates.front().index > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::runtime_error("Selected OpenCL GPU index exceeds the public integer range");
    }
    _device = candidates.front().device;
    _deviceName = candidates.front().name;
    _deviceIndex = static_cast<int>(candidates.front().index);

    const std::string extensions = deviceString(_device, CL_DEVICE_EXTENSIONS);
    _supportsFp64 = extensions.find("cl_khr_fp64") != std::string::npos
        || extensions.find("cl_amd_fp64") != std::string::npos;

    cl_int error = CL_SUCCESS;
    cl_context context = clCreateContext(nullptr, 1, &_device, nullptr, nullptr, &error);
    if (error != CL_SUCCESS)
    {
        if (context)
        {
            clReleaseContext(context);
        }
        checkOpenCl(error, "clCreateContext");
    }
    _context = context;
}

OpenClRuntime::~OpenClRuntime()
{
    for (const auto& entry : _programs)
    {
        clReleaseProgram(entry.second);
    }
    if (_context)
    {
        clReleaseContext(_context);
    }
}

cl_command_queue OpenClRuntime::createQueue() const
{
    cl_int error = CL_SUCCESS;
    cl_command_queue queue = clCreateCommandQueue(_context, _device, 0, &error);
    if (error != CL_SUCCESS)
    {
        if (queue)
        {
            clReleaseCommandQueue(queue);
        }
        checkOpenCl(error, "clCreateCommandQueue");
    }
    return queue;
}

cl_program OpenClRuntime::program(
    const std::string& key,
    const std::string& source,
    const std::string& options)
{
    const std::string cache_key = programCacheKey(key, source, options);
    std::lock_guard<std::mutex> lock(_programMutex);
    const auto found = _programs.find(cache_key);
    if (found != _programs.end())
    {
        return found->second;
    }

    const char* source_pointer = source.c_str();
    const std::size_t source_size = source.size();
    cl_int error = CL_SUCCESS;
    cl_program result = clCreateProgramWithSource(
        _context, 1, &source_pointer, &source_size, &error);
    if (error != CL_SUCCESS)
    {
        if (result)
        {
            clReleaseProgram(result);
        }
        checkOpenCl(error, "clCreateProgramWithSource");
    }

    std::string build_options = "-cl-std=CL1.2";
    if (!options.empty())
    {
        build_options += " " + options;
    }
    error = clBuildProgram(result, 1, &_device, build_options.c_str(), nullptr, nullptr);
    if (error != CL_SUCCESS)
    {
        std::size_t log_size = 0;
        clGetProgramBuildInfo(result, _device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::string log(log_size, '\0');
        if (log_size != 0)
        {
            clGetProgramBuildInfo(
                result, _device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        }
        clReleaseProgram(result);
        throw std::runtime_error(
            "OpenCL clBuildProgram failed with error " + std::to_string(error) + ": " + log);
    }

    try
    {
        _programs.emplace(cache_key, result);
    }
    catch (...)
    {
        clReleaseProgram(result);
        throw;
    }
    return result;
}

} // namespace detail

std::vector<OpenClDeviceInfo> enumerateOpenClGpuDevices()
{
    std::vector<OpenClDeviceInfo> result;
    for (const auto& candidate : detail::enumerateDeviceCandidates())
    {
        result.push_back({
            candidate.index,
            candidate.name,
            candidate.vendor,
            candidate.version,
            candidate.computeUnits,
            candidate.unifiedMemory != CL_FALSE,
            candidate.available != CL_FALSE,
            candidate.compilerAvailable != CL_FALSE});
    }
    return result;
}

bool hasUsableOpenClDevice() noexcept
{
    try
    {
        (void)detail::OpenClRuntime::instance();
        return true;
    }
    catch (...)
    {
        return false;
    }
}

void requireUsableOpenClDevice()
{
    (void)detail::OpenClRuntime::instance();
}

std::string selectedOpenClDeviceName()
{
    try
    {
        return detail::OpenClRuntime::instance().deviceName();
    }
    catch (...)
    {
        return {};
    }
}

int selectedOpenClDeviceIndex() noexcept
{
    try
    {
        return detail::OpenClRuntime::instance().deviceIndex();
    }
    catch (...)
    {
        return -1;
    }
}

} // namespace opencl
} // namespace plamatrix
