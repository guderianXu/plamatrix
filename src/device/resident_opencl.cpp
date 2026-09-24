#include "plamatrix/internal/device/detail/resident_opencl.h"

#ifdef PLAMATRIX_WITH_OPENCL

#include "plamatrix/internal/core/error.h"
#include "plamatrix/internal/opencl/native.h"

namespace plamatrix::internal::v1::resident_detail
{

void* allocateOpenCl(ExecutionContext& context, std::size_t bytes)
{
    cl_int status = CL_SUCCESS;
    cl_mem memory = clCreateBuffer(opencl::NativeAccess::context(context),
                                    CL_MEM_READ_WRITE,
                                    bytes,
                                    nullptr,
                                    &status);
    if (status != CL_SUCCESS || memory == nullptr)
    {
        throw Error(ErrorCode::BackendFailure,
                    "OpenCL resident allocation failed",
                    Backend::OpenCl,
                    status);
    }
    return memory;
}

void releaseOpenCl(void* handle) noexcept
{
    if (handle != nullptr)
    {
        static_cast<void>(clReleaseMemObject(static_cast<cl_mem>(handle)));
    }
}

void copyResidentOpenCl(ExecutionContext& context, void* source, void* destination, std::size_t bytes)
{
    const cl_int status = clEnqueueCopyBuffer(opencl::NativeAccess::queue(context),
                                              static_cast<cl_mem>(source),
                                              static_cast<cl_mem>(destination),
                                              0,
                                              0,
                                              bytes,
                                              0,
                                              nullptr,
                                              nullptr);
    if (status != CL_SUCCESS)
    {
        throw Error(ErrorCode::BackendFailure, "OpenCL resident device copy failed", Backend::OpenCl, status);
    }
    context.synchronize();
}

void copyFromHostOpenCl(ExecutionContext& context, void* handle, const void* source, std::size_t bytes)
{
    const cl_int status = clEnqueueWriteBuffer(opencl::NativeAccess::queue(context),
                                               static_cast<cl_mem>(handle),
                                               CL_TRUE,
                                               0,
                                               bytes,
                                               source,
                                               0,
                                               nullptr,
                                               nullptr);
    if (status != CL_SUCCESS)
    {
        throw Error(ErrorCode::BackendFailure,
                    "OpenCL resident host-to-device copy failed",
                    Backend::OpenCl,
                    status);
    }
}

void copyToHostOpenCl(ExecutionContext& context, void* handle, void* destination, std::size_t bytes)
{
    const cl_int status = clEnqueueReadBuffer(opencl::NativeAccess::queue(context),
                                               static_cast<cl_mem>(handle),
                                               CL_TRUE,
                                               0,
                                               bytes,
                                               destination,
                                               0,
                                               nullptr,
                                               nullptr);
    if (status != CL_SUCCESS)
    {
        throw Error(ErrorCode::BackendFailure,
                    "OpenCL resident device-to-host copy failed",
                    Backend::OpenCl,
                    status);
    }
}

} // namespace plamatrix::internal::v1::resident_detail

#endif
