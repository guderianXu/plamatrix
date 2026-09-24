#pragma once

#ifndef PLAMATRIX_WITH_OPENCL
#error "plamatrix/internal/opencl/native.h requires PLAMATRIX_WITH_OPENCL=ON"
#endif

#include <CL/cl.h>

#include "plamatrix/internal/core/execution_context.h"

namespace plamatrix::internal::v1
{
template <typename Scalar> class ResidentMatrix;
template <typename Scalar> class ResidentVector;
template <typename Scalar> class ResidentCsrMatrix;
}

namespace plamatrix::internal::opencl
{

// Native handles are borrowed; the ExecutionContext must outlive their use.
struct NativeAccess
{
    static cl_context context(ExecutionContext& execution_context);
    static cl_device_id device(ExecutionContext& execution_context);
    static cl_command_queue queue(ExecutionContext& execution_context);

    template <typename Scalar> static cl_mem buffer(const ResidentMatrix<Scalar>& matrix)
    {
        return static_cast<cl_mem>(matrix._values.nativeHandle());
    }

    template <typename Scalar> static cl_mem buffer(const ResidentVector<Scalar>& vector)
    {
        return static_cast<cl_mem>(vector._values.nativeHandle());
    }

    template <typename Scalar> static cl_mem values(const ResidentCsrMatrix<Scalar>& matrix)
    {
        return static_cast<cl_mem>(matrix._values.nativeHandle());
    }

    template <typename Scalar> static cl_mem columns(const ResidentCsrMatrix<Scalar>& matrix)
    {
        return static_cast<cl_mem>(matrix._colIndices.nativeHandle());
    }

    template <typename Scalar> static cl_mem rows(const ResidentCsrMatrix<Scalar>& matrix)
    {
        return static_cast<cl_mem>(matrix._rowOffsets.nativeHandle());
    }
};

} // namespace plamatrix::internal::opencl
