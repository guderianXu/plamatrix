#pragma once

#ifndef PLAMATRIX_WITH_VULKAN
#error "plamatrix/internal/vulkan/native.h requires PLAMATRIX_WITH_VULKAN=ON"
#endif

#include "plamatrix/internal/core/execution_context.h"
#include "plamatrix/internal/vulkan/runtime.h"

namespace plamatrix::internal::v1
{
template <typename Scalar> class ResidentMatrix;
template <typename Scalar> class ResidentVector;
template <typename Scalar> class ResidentCsrMatrix;
}

namespace plamatrix::internal::vulkan
{

// Native handles are borrowed; the ExecutionContext must outlive their use.
struct NativeAccess
{
    static Runtime& runtime(ExecutionContext& context);

    template <typename Scalar> static Buffer* buffer(const ResidentMatrix<Scalar>& matrix)
    {
        return static_cast<Buffer*>(matrix._values.nativeHandle());
    }

    template <typename Scalar> static Buffer* buffer(const ResidentVector<Scalar>& vector)
    {
        return static_cast<Buffer*>(vector._values.nativeHandle());
    }

    template <typename Scalar> static Buffer* values(const ResidentCsrMatrix<Scalar>& matrix)
    {
        return static_cast<Buffer*>(matrix._values.nativeHandle());
    }

    template <typename Scalar> static Buffer* columns(const ResidentCsrMatrix<Scalar>& matrix)
    {
        return static_cast<Buffer*>(matrix._colIndices.nativeHandle());
    }

    template <typename Scalar> static Buffer* rows(const ResidentCsrMatrix<Scalar>& matrix)
    {
        return static_cast<Buffer*>(matrix._rowOffsets.nativeHandle());
    }

    template <typename Scalar> static Buffer* columns32(const ResidentCsrMatrix<Scalar>& matrix)
    {
        return static_cast<Buffer*>(matrix._vulkanColumns32.nativeHandle());
    }

    template <typename Scalar> static Buffer* rows32(const ResidentCsrMatrix<Scalar>& matrix)
    {
        return static_cast<Buffer*>(matrix._vulkanRows32.nativeHandle());
    }
};

} // namespace plamatrix::internal::vulkan
