#pragma once

#ifndef PLAMATRIX_WITH_OPENCL
#error "plamatrix/opencl/execution.h requires PlaMatrix built with PLAMATRIX_WITH_OPENCL=ON"
#endif

#include "plamatrix/opencl/runtime.h"

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace plamatrix
{
namespace opencl
{
namespace detail
{

/// Move-only owner of an OpenCL command queue.
class CommandQueue
{
public:
    CommandQueue() = default;

    explicit CommandQueue(cl_command_queue queue) noexcept
        : _queue(queue)
    {
    }

    ~CommandQueue()
    {
        reset();
    }

    CommandQueue(const CommandQueue&) = delete;
    CommandQueue& operator=(const CommandQueue&) = delete;

    CommandQueue(CommandQueue&& other) noexcept
        : _queue(std::exchange(other._queue, nullptr))
    {
    }

    CommandQueue& operator=(CommandQueue&& other) noexcept
    {
        if (this != &other)
        {
            reset();
            _queue = std::exchange(other._queue, nullptr);
        }
        return *this;
    }

    cl_command_queue get() const noexcept
    {
        return _queue;
    }

    operator cl_command_queue() const noexcept
    {
        return _queue;
    }

private:
    void reset() noexcept
    {
        if (_queue)
        {
            clReleaseCommandQueue(_queue);
            _queue = nullptr;
        }
    }

    cl_command_queue _queue = nullptr;
};

/// Move-only owner of an OpenCL memory object.
class DeviceBuffer
{
public:
    DeviceBuffer() = default;

    DeviceBuffer(cl_context context, cl_mem_flags flags, std::size_t size, void* host = nullptr)
    {
        if (!context)
        {
            throw std::invalid_argument("OpenCL buffer requires a valid context");
        }
        if (size == 0)
        {
            throw std::invalid_argument("OpenCL buffer size must be greater than zero");
        }

        cl_int error = CL_SUCCESS;
        cl_mem memory = clCreateBuffer(context, flags, size, host, &error);
        if (error != CL_SUCCESS)
        {
            if (memory)
            {
                clReleaseMemObject(memory);
            }
            checkOpenCl(error, "clCreateBuffer");
        }
        _memory = memory;
        _size = size;
    }

    ~DeviceBuffer()
    {
        reset();
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : _memory(std::exchange(other._memory, nullptr)),
          _size(std::exchange(other._size, 0))
    {
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
    {
        if (this != &other)
        {
            reset();
            _memory = std::exchange(other._memory, nullptr);
            _size = std::exchange(other._size, 0);
        }
        return *this;
    }

    cl_mem get() const noexcept
    {
        return _memory;
    }

    std::size_t size() const noexcept
    {
        return _size;
    }

private:
    void reset() noexcept
    {
        if (_memory)
        {
            clReleaseMemObject(_memory);
            _memory = nullptr;
        }
        _size = 0;
    }

    cl_mem _memory = nullptr;
    std::size_t _size = 0;
};

/// Move-only owner of a kernel created from a cached OpenCL program.
class CompiledKernel
{
public:
    CompiledKernel(cl_program program, const char* name)
    {
        if (!program)
        {
            throw std::invalid_argument("OpenCL kernel requires a valid program");
        }
        if (!name || name[0] == '\0')
        {
            throw std::invalid_argument("OpenCL kernel name must not be empty");
        }

        cl_int error = CL_SUCCESS;
        cl_kernel kernel = clCreateKernel(program, name, &error);
        if (error != CL_SUCCESS)
        {
            if (kernel)
            {
                clReleaseKernel(kernel);
            }
            checkOpenCl(error, "clCreateKernel");
        }
        _kernel = kernel;
    }

    ~CompiledKernel()
    {
        reset();
    }

    CompiledKernel(const CompiledKernel&) = delete;
    CompiledKernel& operator=(const CompiledKernel&) = delete;

    CompiledKernel(CompiledKernel&& other) noexcept
        : _kernel(std::exchange(other._kernel, nullptr))
    {
    }

    CompiledKernel& operator=(CompiledKernel&& other) noexcept
    {
        if (this != &other)
        {
            reset();
            _kernel = std::exchange(other._kernel, nullptr);
        }
        return *this;
    }

    cl_kernel get() const noexcept
    {
        return _kernel;
    }

    operator cl_kernel() const noexcept
    {
        return _kernel;
    }

private:
    void reset() noexcept
    {
        if (_kernel)
        {
            clReleaseKernel(_kernel);
            _kernel = nullptr;
        }
    }

    cl_kernel _kernel = nullptr;
};

template <typename Value>
std::size_t byteSize(std::size_t count)
{
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(Value))
    {
        throw std::overflow_error("OpenCL buffer byte size overflow");
    }
    return count * sizeof(Value);
}

template <typename Value>
DeviceBuffer inputVector(OpenClRuntime& runtime, const std::vector<Value>& values)
{
    return DeviceBuffer(
        runtime.context(),
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        byteSize<Value>(values.size()),
        const_cast<Value*>(values.data()));
}

template <typename Value>
DeviceBuffer inOutVector(OpenClRuntime& runtime, std::vector<Value>& values)
{
    return DeviceBuffer(
        runtime.context(),
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        byteSize<Value>(values.size()),
        values.data());
}

template <typename Value>
void kernelArg(cl_kernel kernel, cl_uint index, const Value& value)
{
    checkOpenCl(clSetKernelArg(kernel, index, sizeof(Value), &value), "clSetKernelArg");
}

inline void kernelBufferArg(cl_kernel kernel, cl_uint index, const DeviceBuffer& buffer)
{
    const cl_mem memory = buffer.get();
    checkOpenCl(
        clSetKernelArg(kernel, index, sizeof(memory), &memory),
        "clSetKernelArg(buffer)");
}

template <typename Value>
void readVector(cl_command_queue queue, const DeviceBuffer& buffer, std::vector<Value>& values)
{
    if (values.empty())
    {
        return;
    }
    const std::size_t bytes = byteSize<Value>(values.size());
    if (bytes > buffer.size())
    {
        throw std::out_of_range("OpenCL read exceeds device buffer size");
    }
    checkOpenCl(
        clEnqueueReadBuffer(
            queue,
            buffer.get(),
            CL_TRUE,
            0,
            bytes,
            values.data(),
            0,
            nullptr,
            nullptr),
        "clEnqueueReadBuffer");
}

template <typename Scalar>
void requireFp64(OpenClRuntime& runtime)
{
    if constexpr (std::is_same_v<std::remove_cv_t<Scalar>, double>)
    {
        if (!runtime.supportsFp64())
        {
            throw std::runtime_error("Selected OpenCL device does not support double precision");
        }
    }
}

/// Return a build definition for double kernels using the caller-selected macro name.
template <typename Scalar>
std::string realBuildOptions(const std::string& double_macro = "PLAMATRIX_REAL_DOUBLE")
{
    if constexpr (std::is_same_v<std::remove_cv_t<Scalar>, double>)
    {
        if (double_macro.empty())
        {
            throw std::invalid_argument("OpenCL double-precision macro name must not be empty");
        }
        return "-D" + double_macro + "=1";
    }
    return {};
}

} // namespace detail

using detail::byteSize;
using detail::CommandQueue;
using detail::CompiledKernel;
using detail::DeviceBuffer;
using detail::inOutVector;
using detail::inputVector;
using detail::kernelArg;
using detail::kernelBufferArg;
using detail::readVector;
using detail::realBuildOptions;
using detail::requireFp64;

} // namespace opencl
} // namespace plamatrix
