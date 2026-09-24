#pragma once

#include <memory>
#include <vector>

#include "plamatrix/internal/core/backend.h"
#include "plamatrix/internal/core/error.h"
#include "plamatrix/internal/core/device.h"

namespace plamatrix::internal::detail
{

    void cpuGemm(const float* left, const float* right, float* output, Index rows, Index columns, Index inner);
    void cpuGemm(const double* left, const double* right, double* output, Index rows, Index columns, Index inner);

    template <typename Scalar> class GpuStorage;

    template <typename Scalar> struct GpuQrResult
    {
        std::vector<Scalar> q;
        std::vector<Scalar> r;
        Index rank = 0;
    };

    template <typename Scalar> struct GpuSvdResult
    {
        std::vector<Scalar> u;
        std::vector<Scalar> singular;
        std::vector<Scalar> v;
    };

    template <typename Scalar> struct GpuSelfAdjointEigenResult
    {
        std::vector<Scalar> values;
        std::vector<Scalar> vectors;
    };

    template <typename Scalar> struct GpuOps
    {
        static constexpr bool supported = false;
        static bool available(Backend) noexcept
        {
            return false;
        }
        static Backend backend(const GpuStorage<Scalar>&)
        {
            return Backend::Cpu;
        }
        static std::shared_ptr<GpuStorage<Scalar>> upload(Backend, const Scalar*, Index, Index)
        {
            throw Error(ErrorCode::UnsupportedOperation, "Automatic GPU execution does not support this scalar type");
        }
        static void download(const GpuStorage<Scalar>&, Scalar*)
        {
            throw Error(ErrorCode::UnsupportedOperation, "Automatic GPU execution does not support this scalar type");
        }
        static std::shared_ptr<GpuStorage<Scalar>> multiply(const GpuStorage<Scalar>&, const GpuStorage<Scalar>&)
        {
            throw Error(ErrorCode::UnsupportedOperation, "Automatic GPU execution does not support this scalar type");
        }
        static std::shared_ptr<GpuStorage<Scalar>> add(const GpuStorage<Scalar>&, const GpuStorage<Scalar>&)
        {
            throw Error(ErrorCode::UnsupportedOperation, "Automatic GPU execution does not support this scalar type");
        }
        static std::shared_ptr<GpuStorage<Scalar>> subtract(const GpuStorage<Scalar>&, const GpuStorage<Scalar>&)
        {
            throw Error(ErrorCode::UnsupportedOperation, "Automatic GPU execution does not support this scalar type");
        }
        static std::shared_ptr<GpuStorage<Scalar>> scale(const GpuStorage<Scalar>&, Scalar)
        {
            throw Error(ErrorCode::UnsupportedOperation, "Automatic GPU execution does not support this scalar type");
        }
        static std::shared_ptr<GpuStorage<Scalar>> cwiseProduct(const GpuStorage<Scalar>&, const GpuStorage<Scalar>&)
        {
            throw Error(ErrorCode::UnsupportedOperation, "Automatic GPU execution does not support this scalar type");
        }
        static std::shared_ptr<GpuStorage<Scalar>> transpose(const GpuStorage<Scalar>&)
        {
            throw Error(ErrorCode::UnsupportedOperation, "Automatic GPU execution does not support this scalar type");
        }
        static Scalar sum(const GpuStorage<Scalar>&)
        {
            throw Error(ErrorCode::UnsupportedOperation, "Automatic GPU execution does not support this scalar type");
        }
        static Scalar dot(const GpuStorage<Scalar>&, const GpuStorage<Scalar>&)
        {
            throw Error(ErrorCode::UnsupportedOperation, "Automatic GPU execution does not support this scalar type");
        }
        static GpuQrResult<Scalar> qr(Backend, const Scalar*, Index, Index)
        {
            throw Error(ErrorCode::UnsupportedOperation, "GPU QR does not support this scalar type");
        }
        static GpuSvdResult<Scalar> svd(Backend, const Scalar*, Index, Index)
        {
            throw Error(ErrorCode::UnsupportedOperation, "GPU SVD does not support this scalar type");
        }
        static GpuSelfAdjointEigenResult<Scalar> selfAdjointEigen(Backend, const Scalar*, Index)
        {
            throw Error(ErrorCode::UnsupportedOperation,
                        "GPU self-adjoint eigendecomposition does not support this scalar type");
        }
    };

#define PLAMATRIX_DECLARE_AUTO_GPU_OPS(ScalarType)                                                                     \
    template <> struct GpuOps<ScalarType>                                                                              \
    {                                                                                                                  \
        static constexpr bool supported = true;                                                                        \
        static bool available(Backend) noexcept;                                                                       \
        static Backend backend(const GpuStorage<ScalarType>&) noexcept;                                                \
        static std::shared_ptr<GpuStorage<ScalarType>> upload(Backend, const ScalarType*, Index, Index);               \
        static void download(const GpuStorage<ScalarType>&, ScalarType*);                                              \
        static std::shared_ptr<GpuStorage<ScalarType>> multiply(const GpuStorage<ScalarType>&,                         \
                                                                const GpuStorage<ScalarType>&);                        \
        static std::shared_ptr<GpuStorage<ScalarType>> add(const GpuStorage<ScalarType>&,                              \
                                                           const GpuStorage<ScalarType>&);                             \
        static std::shared_ptr<GpuStorage<ScalarType>> subtract(const GpuStorage<ScalarType>&,                         \
                                                                const GpuStorage<ScalarType>&);                        \
        static std::shared_ptr<GpuStorage<ScalarType>> scale(const GpuStorage<ScalarType>&, ScalarType);               \
        static std::shared_ptr<GpuStorage<ScalarType>> cwiseProduct(const GpuStorage<ScalarType>&,                     \
                                                                    const GpuStorage<ScalarType>&);                    \
        static std::shared_ptr<GpuStorage<ScalarType>> transpose(const GpuStorage<ScalarType>&);                       \
        static ScalarType sum(const GpuStorage<ScalarType>&);                                                          \
        static ScalarType dot(const GpuStorage<ScalarType>&, const GpuStorage<ScalarType>&);                           \
        static GpuQrResult<ScalarType> qr(Backend, const ScalarType*, Index, Index);                                   \
        static GpuSvdResult<ScalarType> svd(Backend, const ScalarType*, Index, Index);                                 \
        static GpuSelfAdjointEigenResult<ScalarType> selfAdjointEigen(Backend, const ScalarType*, Index);              \
    }

    PLAMATRIX_DECLARE_AUTO_GPU_OPS(float);
    PLAMATRIX_DECLARE_AUTO_GPU_OPS(double);

#undef PLAMATRIX_DECLARE_AUTO_GPU_OPS

} // namespace plamatrix::internal::detail
