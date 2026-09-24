#include "plamatrix/internal/dense/auto_backend.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "plamatrix/internal/core/execution_context.h"
#include "plamatrix/internal/device/device_matrix.h"

#ifdef PLAMATRIX_WITH_CUDA
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include "plamatrix/internal/dense/dense_ops.h"
#include "plamatrix/internal/dense/dense_storage.h"
#include "plamatrix/internal/dense/elementwise.h"
#include "plamatrix/internal/ops/gemm.h"
#include "plamatrix/internal/ops/reduction.h"
#endif

#ifdef PLAMATRIX_WITH_OPENCL
#include <CL/cl.h>

#include "plamatrix/internal/opencl/native.h"
#endif

#ifdef PLAMATRIX_WITH_VULKAN
#include "plamatrix/internal/vulkan/execution.h"
#include "plamatrix/internal/vulkan/native.h"
#endif

namespace plamatrix::internal::detail
{
    namespace
    {
        std::shared_ptr<ExecutionContext> sharedContext(Backend backend) noexcept
        {
            try
            {
#ifdef PLAMATRIX_WITH_OPENCL
                if (backend == Backend::OpenCl)
                {
                    static const auto context = ExecutionContext::createShared({Backend::OpenCl, 0});
                    return context;
                }
#endif
#ifdef PLAMATRIX_WITH_VULKAN
                if (backend == Backend::Vulkan)
                {
                    static const auto context = ExecutionContext::createShared({Backend::Vulkan, 0});
                    return context;
                }
#endif
            }
            catch (...)
            {
            }
            return {};
        }

        template <typename Scalar> bool backendAvailable(Backend backend) noexcept
        {
            if (backend == Backend::Cuda)
            {
#ifdef PLAMATRIX_WITH_CUDA
                int count = 0;
                return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
#else
                return false;
#endif
            }
            if (backend == Backend::OpenCl || backend == Backend::Vulkan)
            {
                const auto context = sharedContext(backend);
                if (!context)
                {
                    return false;
                }
                if constexpr (std::is_same_v<Scalar, double>)
                {
                    return backend == Backend::OpenCl && context->capabilities().float64;
                }
                return std::is_same_v<Scalar, float>;
            }
            return false;
        }

        [[noreturn]] void unavailable(Backend backend)
        {
            throw Error(ErrorCode::BackendUnavailable,
                        "Requested dense GPU backend is unavailable or lacks scalar support",
                        backend);
        }

#ifdef PLAMATRIX_WITH_CUDA
        cublasHandle_t cudaBlasHandle()
        {
            static thread_local cublasHandle_t handle = nullptr;
            static thread_local int handleDevice = -1;
            int currentDevice = 0;
            PLAMATRIX_CHECK_CUDA(cudaGetDevice(&currentDevice));
            if (handle != nullptr && handleDevice != currentDevice)
            {
                PLAMATRIX_CHECK_CUBLAS(cublasDestroy(handle));
                handle = nullptr;
            }
            if (handle == nullptr)
            {
                PLAMATRIX_CHECK_CUBLAS(cublasCreate(&handle));
                handleDevice = currentDevice;
            }
            return handle;
        }

        template <typename Scalar>
        Scalar cudaDot(const DenseStorage<Scalar, Device::GPU>& left, const DenseStorage<Scalar, Device::GPU>& right)
        {
            if (left.size() != right.size())
                throw Error(ErrorCode::InvalidArgument, "CUDA dense dot requires operands of equal size");
            if (left.size() > static_cast<Index>(std::numeric_limits<int>::max()))
                throw Error(ErrorCode::InvalidArgument, "CUDA dense dot size exceeds the cuBLAS int range");
            Scalar result{};
            cublasHandle_t handle = cudaBlasHandle();
            PLAMATRIX_CHECK_CUBLAS(cublasSetStream(handle, nullptr));
            if constexpr (std::is_same_v<Scalar, float>)
                PLAMATRIX_CHECK_CUBLAS(
                    cublasSdot(handle, static_cast<int>(left.size()), left.data(), 1, right.data(), 1, &result));
            else
                PLAMATRIX_CHECK_CUBLAS(
                    cublasDdot(handle, static_cast<int>(left.size()), left.data(), 1, right.data(), 1, &result));
            return result;
        }
#endif
    } // namespace

    template <typename Scalar> class GpuStorage
    {
    public:
#ifdef PLAMATRIX_WITH_CUDA
        explicit GpuStorage(DenseStorage<Scalar, Device::GPU>&& value)
            : selectedBackend(Backend::Cuda),
              cudaMatrix(std::make_unique<DenseStorage<Scalar, Device::GPU>>(std::move(value)))
        {
        }
#endif

        GpuStorage(Backend backend, ResidentMatrix<Scalar>&& value)
            : selectedBackend(backend), portableMatrix(std::make_unique<ResidentMatrix<Scalar>>(std::move(value)))
        {
        }

        Backend selectedBackend = Backend::Cpu;
#ifdef PLAMATRIX_WITH_CUDA
        std::unique_ptr<DenseStorage<Scalar, Device::GPU>> cudaMatrix;
#endif
        std::unique_ptr<ResidentMatrix<Scalar>> portableMatrix;
    };

    template <typename Scalar> struct JacobiPairSchedule
    {
        std::vector<Scalar> pairs;
        std::vector<std::size_t> offsets;
        std::vector<std::size_t> counts;
    };

    template <typename Scalar> JacobiPairSchedule<Scalar> makeJacobiPairSchedule(Index size)
    {
        JacobiPairSchedule<Scalar> result;
        if (size < 2)
            return result;
        const Index participantCount = size + (size & 1);
        std::vector<Index> participants(static_cast<std::size_t>(participantCount));
        for (Index index = 0; index < participantCount; ++index)
            participants[static_cast<std::size_t>(index)] = index;
        for (Index round = 0; round < participantCount - 1; ++round)
        {
            result.offsets.push_back(result.pairs.size() / 2);
            std::size_t count = 0;
            for (Index pair = 0; pair < participantCount / 2; ++pair)
            {
                const Index first = participants[static_cast<std::size_t>(pair)];
                const Index second = participants[static_cast<std::size_t>(participantCount - 1 - pair)];
                if (first < size && second < size)
                {
                    result.pairs.push_back(static_cast<Scalar>(std::min(first, second)));
                    result.pairs.push_back(static_cast<Scalar>(std::max(first, second)));
                    ++count;
                }
            }
            result.counts.push_back(count);
            const Index last = participants.back();
            for (Index index = participantCount - 1; index > 1; --index)
                participants[static_cast<std::size_t>(index)] = participants[static_cast<std::size_t>(index - 1)];
            participants[1] = last;
        }
        return result;
    }

    template <typename Scalar>
    GpuSvdResult<Scalar>
    finalizePortableSvd(const std::vector<Scalar>& work, const std::vector<Scalar>& rawV, Index rows, Index columns)
    {
        const Index minor = std::min(rows, columns);
        std::vector<Scalar> norms(static_cast<std::size_t>(columns), Scalar{});
        std::vector<Index> order(static_cast<std::size_t>(columns));
        for (Index column = 0; column < columns; ++column)
        {
            Scalar squared{};
            for (Index row = 0; row < rows; ++row)
            {
                const Scalar value = work[static_cast<std::size_t>(row + column * rows)];
                squared += value * value;
            }
            norms[static_cast<std::size_t>(column)] = std::sqrt(std::max(Scalar{}, squared));
            order[static_cast<std::size_t>(column)] = column;
        }
        std::stable_sort(order.begin(),
                         order.end(),
                         [&norms](Index left, Index right)
                         { return norms[static_cast<std::size_t>(left)] > norms[static_cast<std::size_t>(right)]; });

        GpuSvdResult<Scalar> result;
        result.u.assign(static_cast<std::size_t>(rows * rows), Scalar{});
        result.v.assign(static_cast<std::size_t>(columns * columns), Scalar{});
        result.singular.resize(static_cast<std::size_t>(minor));
        for (Index column = 0; column < columns; ++column)
        {
            const Index source = order[static_cast<std::size_t>(column)];
            for (Index row = 0; row < columns; ++row)
                result.v[static_cast<std::size_t>(row + column * columns)] =
                    rawV[static_cast<std::size_t>(row + source * columns)];
        }
        const Scalar largest = columns == 0 ? Scalar{} : norms[static_cast<std::size_t>(order.front())];
        const Scalar threshold =
            std::numeric_limits<Scalar>::epsilon() * static_cast<Scalar>(std::max(rows, columns)) * largest;
        for (Index column = 0; column < rows; ++column)
        {
            bool fromDecomposition = false;
            if (column < minor)
            {
                const Index source = order[static_cast<std::size_t>(column)];
                const Scalar norm = norms[static_cast<std::size_t>(source)];
                result.singular[static_cast<std::size_t>(column)] = norm;
                if (norm > threshold)
                {
                    for (Index row = 0; row < rows; ++row)
                        result.u[static_cast<std::size_t>(row + column * rows)] =
                            work[static_cast<std::size_t>(row + source * rows)] / norm;
                    fromDecomposition = true;
                }
            }
            if (!fromDecomposition)
            {
                Scalar norm{};
                for (Index candidate = 0; candidate < rows && !(norm > threshold); ++candidate)
                {
                    for (Index row = 0; row < rows; ++row)
                        result.u[static_cast<std::size_t>(row + column * rows)] =
                            row == candidate ? Scalar{1} : Scalar{};
                    for (int pass = 0; pass < 2; ++pass)
                    {
                        for (Index previous = 0; previous < column; ++previous)
                        {
                            Scalar projection{};
                            for (Index row = 0; row < rows; ++row)
                                projection += result.u[static_cast<std::size_t>(row + previous * rows)] *
                                              result.u[static_cast<std::size_t>(row + column * rows)];
                            for (Index row = 0; row < rows; ++row)
                                result.u[static_cast<std::size_t>(row + column * rows)] -=
                                    projection * result.u[static_cast<std::size_t>(row + previous * rows)];
                        }
                    }
                    Scalar squared{};
                    for (Index row = 0; row < rows; ++row)
                    {
                        const Scalar value = result.u[static_cast<std::size_t>(row + column * rows)];
                        squared += value * value;
                    }
                    norm = std::sqrt(std::max(Scalar{}, squared));
                }
                if (!(norm > threshold))
                    throw Error(ErrorCode::BackendFailure, "GPU SVD failed to complete an orthogonal U basis");
                for (Index row = 0; row < rows; ++row)
                    result.u[static_cast<std::size_t>(row + column * rows)] /= norm;
            }
        }
        return result;
    }

    template <typename Scalar>
    GpuSelfAdjointEigenResult<Scalar>
    finalizePortableEigen(const std::vector<Scalar>& matrix, const std::vector<Scalar>& vectors, Index size)
    {
        std::vector<Index> order(static_cast<std::size_t>(size));
        for (Index index = 0; index < size; ++index)
            order[static_cast<std::size_t>(index)] = index;
        std::stable_sort(order.begin(),
                         order.end(),
                         [&matrix, size](Index left, Index right)
                         {
                             return matrix[static_cast<std::size_t>(left + left * size)] <
                                    matrix[static_cast<std::size_t>(right + right * size)];
                         });
        GpuSelfAdjointEigenResult<Scalar> result;
        result.values.resize(static_cast<std::size_t>(size));
        result.vectors.resize(static_cast<std::size_t>(size * size));
        for (Index column = 0; column < size; ++column)
        {
            const Index source = order[static_cast<std::size_t>(column)];
            result.values[static_cast<std::size_t>(column)] = matrix[static_cast<std::size_t>(source + source * size)];
            for (Index row = 0; row < size; ++row)
                result.vectors[static_cast<std::size_t>(row + column * size)] =
                    vectors[static_cast<std::size_t>(row + source * size)];
        }
        return result;
    }

#ifdef PLAMATRIX_WITH_OPENCL
    namespace
    {
        void checkOpenCl(cl_int status, const char* operation)
        {
            if (status != CL_SUCCESS)
            {
                throw Error(
                    ErrorCode::BackendFailure, std::string("OpenCL ") + operation + " failed", Backend::OpenCl, status);
            }
        }

        constexpr const char* openclDenseKernels = R"CLC(
#ifdef USE_DOUBLE
#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif
typedef double scalar_t;
#else
typedef float scalar_t;
#endif

__kernel void binary_op(__global const scalar_t* left,
                        __global const scalar_t* right,
                        __global scalar_t* output,
                        ulong count,
                        uint operation)
{
    const ulong index = get_global_id(0);
    if (index >= count) return;
    if (operation == 0) output[index] = left[index] + right[index];
    else if (operation == 1) output[index] = left[index] - right[index];
    else output[index] = left[index] * right[index];
}

__kernel void scale_op(__global const scalar_t* input,
                       __global scalar_t* output,
                       scalar_t factor,
                       ulong count)
{
    const ulong index = get_global_id(0);
    if (index < count) output[index] = input[index] * factor;
}

__kernel void transpose_op(__global const scalar_t* input,
                           __global scalar_t* output,
                           ulong rows,
                           ulong columns)
{
    const ulong index = get_global_id(0);
    if (index >= rows * columns) return;
    const ulong row = index % rows;
    const ulong column = index / rows;
    output[column + row * columns] = input[index];
}

__kernel void gemm_naive_op(__global const scalar_t* left,
                            __global const scalar_t* right,
                            __global scalar_t* output,
                            ulong rows,
                            ulong columns,
                            ulong inner)
{
    const ulong index = get_global_id(0);
    if (index >= rows * columns) return;
    const ulong row = index % rows;
    const ulong column = index / rows;
    scalar_t sum = (scalar_t)0;
    for (ulong k = 0; k < inner; ++k)
        sum += left[row + k * rows] * right[k + column * inner];
    output[index] = sum;
}

#define GEMM_TILE_SIZE 16
__kernel void gemm_tiled_op(__global const scalar_t* left,
                            __global const scalar_t* right,
                            __global scalar_t* output,
                            ulong rows,
                            ulong columns,
                            ulong inner)
{
    const ulong local_index = get_local_id(0);
    const ulong local_row = local_index & (GEMM_TILE_SIZE - 1);
    const ulong local_column = local_index >> 4;
    const ulong tile_rows = (rows + GEMM_TILE_SIZE - 1) / GEMM_TILE_SIZE;
    const ulong tile_row = get_group_id(0) % tile_rows;
    const ulong tile_column = get_group_id(0) / tile_rows;
    const ulong row = tile_row * GEMM_TILE_SIZE + local_row;
    const ulong column = tile_column * GEMM_TILE_SIZE + local_column;
    __local scalar_t left_tile[GEMM_TILE_SIZE][GEMM_TILE_SIZE];
    __local scalar_t right_tile[GEMM_TILE_SIZE][GEMM_TILE_SIZE];
    scalar_t sum = (scalar_t)0;
    const ulong tile_count = (inner + GEMM_TILE_SIZE - 1) / GEMM_TILE_SIZE;
    for (ulong tile = 0; tile < tile_count; ++tile)
    {
        const ulong left_k = tile * GEMM_TILE_SIZE + local_column;
        const ulong right_k = tile * GEMM_TILE_SIZE + local_row;
        left_tile[local_column][local_row] = row < rows && left_k < inner ? left[row + left_k * rows] : (scalar_t)0;
        right_tile[local_column][local_row] =
            right_k < inner && column < columns ? right[right_k + column * inner] : (scalar_t)0;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (ulong k = 0; k < GEMM_TILE_SIZE; ++k)
            sum += left_tile[k][local_row] * right_tile[local_column][k];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (row < rows && column < columns)
        output[row + column * rows] = sum;
}

#define REDUCTION_SIZE 64
__kernel void reduce_op(__global const scalar_t* left,
                        __global const scalar_t* right,
                        __global scalar_t* output,
                        ulong count,
                        uint operation)
{
    __local scalar_t values[REDUCTION_SIZE];
    const ulong index = get_global_id(0);
    const uint lane = get_local_id(0);
    scalar_t value = (scalar_t)0;
    if (index < count)
        value = operation == 0 ? left[index] : left[index] * right[index];
    values[lane] = value;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = REDUCTION_SIZE / 2; stride > 0; stride >>= 1)
    {
        if (lane < stride) values[lane] += values[lane + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0) output[get_group_id(0)] = values[0];
}

__kernel void qr_op(__global const scalar_t* input,
                    __global scalar_t* qr,
                    __global scalar_t* tau,
                    __global scalar_t* q,
                    __global scalar_t* r,
                    __global scalar_t* status,
                    ulong rows,
                    ulong columns,
                    ulong step,
                    uint operation,
                    scalar_t epsilon)
{
    __local scalar_t scratch[REDUCTION_SIZE];
    __local scalar_t shared_tau;
    __local scalar_t shared_denominator;
    const uint lane = get_local_id(0);
    if (operation == 0)
    {
        const ulong index = get_global_id(0);
        const ulong qr_count = rows * columns;
        const ulong q_count = rows * rows;
        if (index < qr_count)
        {
            qr[index] = input[index];
            r[index] = (scalar_t)0;
        }
        if (index < q_count) q[index] = index % rows == index / rows ? (scalar_t)1 : (scalar_t)0;
        if (index < min(rows, columns)) tau[index] = (scalar_t)0;
        if (index == 0) status[0] = (scalar_t)0;
        return;
    }
    if (operation == 1)
    {
        scalar_t local_norm = (scalar_t)0;
        for (ulong row = step + lane; row < rows; row += REDUCTION_SIZE)
        {
            const scalar_t value = qr[row + step * rows];
            local_norm += value * value;
        }
        scratch[lane] = local_norm;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint stride = REDUCTION_SIZE / 2; stride > 0; stride >>= 1)
        {
            if (lane < stride) scratch[lane] += scratch[lane + stride];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        const scalar_t norm = sqrt(max(scratch[0], (scalar_t)0));
        if (lane == 0)
        {
            const scalar_t leading = qr[step + step * rows];
            const scalar_t alpha = leading >= (scalar_t)0 ? -norm : norm;
            const scalar_t denominator = leading - alpha;
            const scalar_t tolerance = epsilon * (scalar_t)max(rows, columns) * max((scalar_t)1, norm);
            shared_denominator = denominator;
            shared_tau = norm > tolerance && fabs(denominator) > tolerance
                             ? (scalar_t)2 /
                                   ((scalar_t)1 + max((scalar_t)0,
                                                      (norm * norm - leading * leading) /
                                                          (denominator * denominator)))
                             : (scalar_t)0;
            tau[step] = shared_tau;
            if (shared_tau != (scalar_t)0)
            {
                qr[step + step * rows] = alpha;
                status[0] += (scalar_t)1;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (shared_tau != (scalar_t)0)
            for (ulong row = step + 1 + lane; row < rows; row += REDUCTION_SIZE)
                qr[row + step * rows] /= shared_denominator;
        return;
    }
    if (operation == 2)
    {
        const ulong column = step + 1 + get_group_id(0);
        if (column >= columns || tau[step] == (scalar_t)0) return;
        scalar_t local_projection = (scalar_t)0;
        for (ulong row = step + lane; row < rows; row += REDUCTION_SIZE)
        {
            const scalar_t vector_value = row == step ? (scalar_t)1 : qr[row + step * rows];
            local_projection += vector_value * qr[row + column * rows];
        }
        scratch[lane] = local_projection;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint stride = REDUCTION_SIZE / 2; stride > 0; stride >>= 1)
        {
            if (lane < stride) scratch[lane] += scratch[lane + stride];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        const scalar_t projection = scratch[0] * tau[step];
        for (ulong row = step + lane; row < rows; row += REDUCTION_SIZE)
        {
            const scalar_t vector_value = row == step ? (scalar_t)1 : qr[row + step * rows];
            qr[row + column * rows] -= vector_value * projection;
        }
        return;
    }
    if (operation == 3)
    {
        const ulong index = get_global_id(0);
        const ulong qr_count = rows * columns;
        const ulong q_count = rows * rows;
        if (index < qr_count)
        {
            const ulong row = index % rows;
            const ulong column = index / rows;
            r[index] = row <= column ? qr[index] : (scalar_t)0;
        }
        if (index < q_count) q[index] = index % rows == index / rows ? (scalar_t)1 : (scalar_t)0;
        return;
    }
    const ulong q_column = get_group_id(0);
    scalar_t local_projection = (scalar_t)0;
    for (ulong row = step + lane; row < rows; row += REDUCTION_SIZE)
    {
        const scalar_t vector_value = row == step ? (scalar_t)1 : qr[row + step * rows];
        local_projection += vector_value * q[row + q_column * rows];
    }
    scratch[lane] = local_projection;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = REDUCTION_SIZE / 2; stride > 0; stride >>= 1)
    {
        if (lane < stride) scratch[lane] += scratch[lane + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const scalar_t projection = scratch[0] * tau[step];
    for (ulong row = step + lane; row < rows; row += REDUCTION_SIZE)
    {
        const scalar_t vector_value = row == step ? (scalar_t)1 : qr[row + step * rows];
        q[row + q_column * rows] -= vector_value * projection;
    }
}

__kernel void svd_jacobi_op(__global scalar_t* work,
                            __global scalar_t* v,
                            __global const scalar_t* pairs,
                            ulong rows,
                            ulong columns,
                            ulong pair_offset,
                            ulong pair_count,
                            scalar_t tolerance)
{
    __local scalar_t first_sum[REDUCTION_SIZE];
    __local scalar_t second_sum[REDUCTION_SIZE];
    __local scalar_t cross_sum[REDUCTION_SIZE];
    __local scalar_t shared_cosine;
    __local scalar_t shared_sine;
    const ulong local_pair = get_group_id(0);
    if (local_pair >= pair_count) return;
    const ulong pair_index = pair_offset + local_pair;
    const ulong p = (ulong)pairs[2 * pair_index];
    const ulong q_index = (ulong)pairs[2 * pair_index + 1];
    const uint lane = get_local_id(0);
    scalar_t app = (scalar_t)0;
    scalar_t aqq = (scalar_t)0;
    scalar_t apq = (scalar_t)0;
    for (ulong row = lane; row < rows; row += REDUCTION_SIZE)
    {
        const scalar_t first = work[row + p * rows];
        const scalar_t second = work[row + q_index * rows];
        app += first * first;
        aqq += second * second;
        apq += first * second;
    }
    first_sum[lane] = app;
    second_sum[lane] = aqq;
    cross_sum[lane] = apq;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = REDUCTION_SIZE / 2; stride > 0; stride >>= 1)
    {
        if (lane < stride)
        {
            first_sum[lane] += first_sum[lane + stride];
            second_sum[lane] += second_sum[lane + stride];
            cross_sum[lane] += cross_sum[lane + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0)
    {
        app = first_sum[0];
        aqq = second_sum[0];
        apq = cross_sum[0];
        const scalar_t scale = sqrt(max(app * aqq, (scalar_t)0));
        if (scale <= (scalar_t)0 || fabs(apq) <= tolerance * scale)
        {
            shared_cosine = (scalar_t)1;
            shared_sine = (scalar_t)0;
        }
        else
        {
            const scalar_t zeta = (aqq - app) / ((scalar_t)2 * apq);
            const scalar_t tangent = (zeta >= (scalar_t)0 ? (scalar_t)1 : (scalar_t)-1) /
                                     (fabs(zeta) + sqrt((scalar_t)1 + zeta * zeta));
            shared_cosine = (scalar_t)1 / sqrt((scalar_t)1 + tangent * tangent);
            shared_sine = shared_cosine * tangent;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const scalar_t cosine = shared_cosine;
    const scalar_t sine = shared_sine;
    if (sine == (scalar_t)0) return;
    for (ulong row = lane; row < rows; row += REDUCTION_SIZE)
    {
        const scalar_t first = work[row + p * rows];
        const scalar_t second = work[row + q_index * rows];
        work[row + p * rows] = cosine * first - sine * second;
        work[row + q_index * rows] = sine * first + cosine * second;
    }
    for (ulong row = lane; row < columns; row += REDUCTION_SIZE)
    {
        const scalar_t first = v[row + p * columns];
        const scalar_t second = v[row + q_index * columns];
        v[row + p * columns] = cosine * first - sine * second;
        v[row + q_index * columns] = sine * first + cosine * second;
    }
}

__kernel void eigen_jacobi_op(__global scalar_t* matrix,
                              __global scalar_t* vectors,
                              __global const scalar_t* pairs,
                              __global scalar_t* rotations,
                              ulong size,
                              ulong pair_offset,
                              ulong pair_count,
                              uint operation,
                              scalar_t tolerance)
{
    const ulong local_pair = get_group_id(0);
    if (local_pair >= pair_count) return;
    const ulong pair_index = pair_offset + local_pair;
    const ulong p = (ulong)pairs[2 * pair_index];
    const ulong q_index = (ulong)pairs[2 * pair_index + 1];
    const uint lane = get_local_id(0);
    if (operation == 0)
    {
        if (lane == 0)
        {
            const scalar_t app = matrix[p + p * size];
            const scalar_t aqq = matrix[q_index + q_index * size];
            const scalar_t apq = ((scalar_t)0.5) * (matrix[p + q_index * size] + matrix[q_index + p * size]);
            const scalar_t scale = max((scalar_t)1, max(fabs(app), fabs(aqq)));
            scalar_t cosine = (scalar_t)1;
            scalar_t sine = (scalar_t)0;
            if (fabs(apq) > tolerance * scale)
            {
                const scalar_t angle = ((scalar_t)0.5) * atan2((scalar_t)2 * apq, aqq - app);
                cosine = cos(angle);
                sine = sin(angle);
            }
            rotations[2 * local_pair] = cosine;
            rotations[2 * local_pair + 1] = sine;
        }
        return;
    }
    const scalar_t cosine = rotations[2 * local_pair];
    const scalar_t sine = rotations[2 * local_pair + 1];
    if (sine == (scalar_t)0) return;
    if (operation == 1)
    {
        for (ulong row = lane; row < size; row += REDUCTION_SIZE)
        {
            scalar_t first = matrix[row + p * size];
            scalar_t second = matrix[row + q_index * size];
            matrix[row + p * size] = cosine * first - sine * second;
            matrix[row + q_index * size] = sine * first + cosine * second;
            first = vectors[row + p * size];
            second = vectors[row + q_index * size];
            vectors[row + p * size] = cosine * first - sine * second;
            vectors[row + q_index * size] = sine * first + cosine * second;
        }
        return;
    }
    for (ulong column = lane; column < size; column += REDUCTION_SIZE)
    {
        const scalar_t first = matrix[p + column * size];
        const scalar_t second = matrix[q_index + column * size];
        matrix[p + column * size] = cosine * first - sine * second;
        matrix[q_index + column * size] = sine * first + cosine * second;
    }
}

)CLC";

        template <typename Scalar> class OpenClProgram
        {
        public:
            explicit OpenClProgram(ExecutionContext& context) : _context(&context)
            {
                const char* source = openclDenseKernels;
                cl_int status = CL_SUCCESS;
                _program =
                    clCreateProgramWithSource(opencl::NativeAccess::context(context), 1, &source, nullptr, &status);
                checkOpenCl(status, "clCreateProgramWithSource");
                const char* options = std::is_same_v<Scalar, double> ? "-DUSE_DOUBLE=1" : "";
                const cl_device_id device = opencl::NativeAccess::device(context);
                std::size_t maximum_work_group_size = 0;
                _supportsTiledGemm = clGetDeviceInfo(device,
                                                     CL_DEVICE_MAX_WORK_GROUP_SIZE,
                                                     sizeof(maximum_work_group_size),
                                                     &maximum_work_group_size,
                                                     nullptr) == CL_SUCCESS &&
                                     maximum_work_group_size >= 256;
                status = clBuildProgram(_program, 1, &device, options, nullptr, nullptr);
                if (status != CL_SUCCESS)
                {
                    std::size_t size = 0;
                    clGetProgramBuildInfo(_program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &size);
                    std::string log(size, '\0');
                    if (size != 0)
                    {
                        clGetProgramBuildInfo(_program, device, CL_PROGRAM_BUILD_LOG, size, log.data(), nullptr);
                    }
                    throw Error(
                        ErrorCode::BackendFailure, "OpenCL dense kernel build failed: " + log, Backend::OpenCl, status);
                }
            }

            ~OpenClProgram() noexcept
            {
                if (_program != nullptr)
                {
                    clReleaseProgram(_program);
                }
            }

            cl_kernel kernel(const char* name) const
            {
                cl_int status = CL_SUCCESS;
                cl_kernel result = clCreateKernel(_program, name, &status);
                checkOpenCl(status, "clCreateKernel");
                return result;
            }

            ExecutionContext& context() const noexcept
            {
                return *_context;
            }

            bool supportsTiledGemm() const noexcept
            {
                return _supportsTiledGemm;
            }

        private:
            ExecutionContext* _context;
            cl_program _program = nullptr;
            bool _supportsTiledGemm = false;
        };

        template <typename Scalar> OpenClProgram<Scalar>& openClProgram(ExecutionContext& context)
        {
            static OpenClProgram<Scalar> program(context);
            return program;
        }

        template <typename Value> void setOpenClArg(cl_kernel kernel, cl_uint index, const Value& value)
        {
            checkOpenCl(clSetKernelArg(kernel, index, sizeof(Value), &value), "clSetKernelArg");
        }

        template <typename Scalar>
        std::shared_ptr<GpuStorage<Scalar>>
        openClBinary(const GpuStorage<Scalar>& left, const GpuStorage<Scalar>& right, cl_uint operation)
        {
            auto& left_matrix = *left.portableMatrix;
            auto& right_matrix = *right.portableMatrix;
            auto& program = openClProgram<Scalar>(left_matrix.context());
            ResidentMatrix<Scalar> output(left_matrix.rows(), left_matrix.cols(), left_matrix.contextOwner());
            cl_kernel kernel = program.kernel("binary_op");
            try
            {
                const cl_mem left_buffer = opencl::NativeAccess::buffer(left_matrix);
                const cl_mem right_buffer = opencl::NativeAccess::buffer(right_matrix);
                const cl_mem output_buffer = opencl::NativeAccess::buffer(output);
                const cl_ulong count = static_cast<cl_ulong>(left_matrix.size());
                setOpenClArg(kernel, 0, left_buffer);
                setOpenClArg(kernel, 1, right_buffer);
                setOpenClArg(kernel, 2, output_buffer);
                setOpenClArg(kernel, 3, count);
                setOpenClArg(kernel, 4, operation);
                const std::size_t global = left_matrix.size();
                checkOpenCl(clEnqueueNDRangeKernel(opencl::NativeAccess::queue(left_matrix.context()),
                                                   kernel,
                                                   1,
                                                   nullptr,
                                                   &global,
                                                   nullptr,
                                                   0,
                                                   nullptr,
                                                   nullptr),
                            "clEnqueueNDRangeKernel");
                checkOpenCl(clFinish(opencl::NativeAccess::queue(left_matrix.context())), "clFinish");
                clReleaseKernel(kernel);
            }
            catch (...)
            {
                clReleaseKernel(kernel);
                throw;
            }
            return std::make_shared<GpuStorage<Scalar>>(Backend::OpenCl, std::move(output));
        }

        template <typename Scalar>
        Scalar openClReduction(const GpuStorage<Scalar>& left, const GpuStorage<Scalar>* right)
        {
            auto& matrix = *left.portableMatrix;
            if (matrix.size() == 0)
                return Scalar{};
            auto& program = openClProgram<Scalar>(matrix.context());
            constexpr std::size_t localSize = 64;
            const std::size_t maximumGroups = (matrix.size() + localSize - 1) / localSize;
            ResidentMatrix<Scalar> first(static_cast<Index>(maximumGroups), 1, matrix.contextOwner());
            ResidentMatrix<Scalar> second(static_cast<Index>(maximumGroups), 1, matrix.contextOwner());
            cl_kernel kernel = program.kernel("reduce_op");
            try
            {
                cl_mem currentLeft = opencl::NativeAccess::buffer(matrix);
                cl_mem currentRight =
                    right == nullptr ? currentLeft : opencl::NativeAccess::buffer(*right->portableMatrix);
                ResidentMatrix<Scalar>* output = &first;
                std::size_t remaining = matrix.size();
                cl_uint operation = right == nullptr ? 0U : 1U;
                do
                {
                    const std::size_t groups = (remaining + localSize - 1) / localSize;
                    const std::size_t global = groups * localSize;
                    const cl_mem outputBuffer = opencl::NativeAccess::buffer(*output);
                    const cl_ulong count = static_cast<cl_ulong>(remaining);
                    setOpenClArg(kernel, 0, currentLeft);
                    setOpenClArg(kernel, 1, currentRight);
                    setOpenClArg(kernel, 2, outputBuffer);
                    setOpenClArg(kernel, 3, count);
                    setOpenClArg(kernel, 4, operation);
                    checkOpenCl(clEnqueueNDRangeKernel(opencl::NativeAccess::queue(matrix.context()),
                                                       kernel,
                                                       1,
                                                       nullptr,
                                                       &global,
                                                       &localSize,
                                                       0,
                                                       nullptr,
                                                       nullptr),
                                "clEnqueueNDRangeKernel reduction");
                    remaining = groups;
                    currentLeft = outputBuffer;
                    currentRight = outputBuffer;
                    operation = 0U;
                    output = output == &first ? &second : &first;
                } while (remaining > 1);
                checkOpenCl(clFinish(opencl::NativeAccess::queue(matrix.context())), "clFinish reduction");
                Scalar result{};
                (output == &first ? second : first).copyToHost(&result, 1);
                clReleaseKernel(kernel);
                return result;
            }
            catch (...)
            {
                clReleaseKernel(kernel);
                throw;
            }
        }

        template <typename Scalar> GpuQrResult<Scalar> openClQr(const Scalar* inputValues, Index rows, Index columns)
        {
            const auto context = sharedContext(Backend::OpenCl);
            ResidentMatrix<Scalar> input(rows, columns, context);
            ResidentMatrix<Scalar> qr(rows, columns, context);
            ResidentMatrix<Scalar> tau(std::min(rows, columns), 1, context);
            ResidentMatrix<Scalar> q(rows, rows, context);
            ResidentMatrix<Scalar> r(rows, columns, context);
            ResidentMatrix<Scalar> status(1, 1, context);
            input.copyFromHost(inputValues, input.size());
            auto& program = openClProgram<Scalar>(input.context());
            cl_kernel kernel = program.kernel("qr_op");
            try
            {
                const cl_mem inputBuffer = opencl::NativeAccess::buffer(input);
                const cl_mem qrBuffer = opencl::NativeAccess::buffer(qr);
                const cl_mem tauBuffer = opencl::NativeAccess::buffer(tau);
                const cl_mem qBuffer = opencl::NativeAccess::buffer(q);
                const cl_mem rBuffer = opencl::NativeAccess::buffer(r);
                const cl_mem statusBuffer = opencl::NativeAccess::buffer(status);
                const cl_ulong rowCount = static_cast<cl_ulong>(rows);
                const cl_ulong columnCount = static_cast<cl_ulong>(columns);
                const Scalar epsilon = std::numeric_limits<Scalar>::epsilon();
                setOpenClArg(kernel, 0, inputBuffer);
                setOpenClArg(kernel, 1, qrBuffer);
                setOpenClArg(kernel, 2, tauBuffer);
                setOpenClArg(kernel, 3, qBuffer);
                setOpenClArg(kernel, 4, rBuffer);
                setOpenClArg(kernel, 5, statusBuffer);
                setOpenClArg(kernel, 6, rowCount);
                setOpenClArg(kernel, 7, columnCount);
                setOpenClArg(kernel, 10, epsilon);
                constexpr std::size_t localSize = 64;
                const auto enqueue = [&](Index step, cl_uint operation, std::size_t groups)
                {
                    const cl_ulong stepValue = static_cast<cl_ulong>(step);
                    setOpenClArg(kernel, 8, stepValue);
                    setOpenClArg(kernel, 9, operation);
                    const std::size_t global = groups * localSize;
                    checkOpenCl(clEnqueueNDRangeKernel(opencl::NativeAccess::queue(input.context()),
                                                       kernel,
                                                       1,
                                                       nullptr,
                                                       &global,
                                                       &localSize,
                                                       0,
                                                       nullptr,
                                                       nullptr),
                                "clEnqueueNDRangeKernel QR");
                };
                const std::size_t initialCount = std::max(input.size(), q.size());
                enqueue(0, 0, (initialCount + localSize - 1) / localSize);
                const Index steps = std::min(rows, columns);
                for (Index step = 0; step < steps; ++step)
                {
                    enqueue(step, 1, 1);
                    if (columns - step - 1 > 0)
                        enqueue(step, 2, static_cast<std::size_t>(columns - step - 1));
                }
                enqueue(0, 3, (initialCount + localSize - 1) / localSize);
                for (Index step = steps; step-- > 0;)
                    enqueue(step, 4, static_cast<std::size_t>(rows));
                checkOpenCl(clFinish(opencl::NativeAccess::queue(input.context())), "clFinish QR");
                GpuQrResult<Scalar> result;
                result.q.resize(q.size());
                result.r.resize(r.size());
                Scalar rank{};
                q.copyToHost(result.q.data(), result.q.size());
                r.copyToHost(result.r.data(), result.r.size());
                status.copyToHost(&rank, 1);
                result.rank = static_cast<Index>(std::llround(rank));
                clReleaseKernel(kernel);
                return result;
            }
            catch (...)
            {
                clReleaseKernel(kernel);
                throw;
            }
        }

        template <typename Scalar> GpuSvdResult<Scalar> openClSvd(const Scalar* inputValues, Index rows, Index columns)
        {
            const auto context = sharedContext(Backend::OpenCl);
            ResidentMatrix<Scalar> work(rows, columns, context);
            ResidentMatrix<Scalar> vectors(columns, columns, context);
            work.copyFromHost(inputValues, work.size());
            std::vector<Scalar> identity(static_cast<std::size_t>(columns * columns), Scalar{});
            for (Index index = 0; index < columns; ++index)
                identity[static_cast<std::size_t>(index + index * columns)] = Scalar{1};
            vectors.copyFromHost(identity.data(), identity.size());
            const auto schedule = makeJacobiPairSchedule<Scalar>(columns);
            if (schedule.pairs.empty())
            {
                std::vector<Scalar> hostWork(work.size());
                work.copyToHost(hostWork.data(), hostWork.size());
                return finalizePortableSvd(hostWork, identity, rows, columns);
            }
            ResidentMatrix<Scalar> pairs(static_cast<Index>(schedule.pairs.size()), 1, context);
            pairs.copyFromHost(schedule.pairs.data(), schedule.pairs.size());
            auto& program = openClProgram<Scalar>(work.context());
            cl_kernel kernel = program.kernel("svd_jacobi_op");
            try
            {
                const cl_mem workBuffer = opencl::NativeAccess::buffer(work);
                const cl_mem vectorBuffer = opencl::NativeAccess::buffer(vectors);
                const cl_ulong rowCount = static_cast<cl_ulong>(rows);
                const cl_ulong columnCount = static_cast<cl_ulong>(columns);
                const Scalar tolerance =
                    std::numeric_limits<Scalar>::epsilon() * static_cast<Scalar>(std::max(rows, columns)) * Scalar{4};
                setOpenClArg(kernel, 0, workBuffer);
                setOpenClArg(kernel, 1, vectorBuffer);
                const cl_mem pairBuffer = opencl::NativeAccess::buffer(pairs);
                setOpenClArg(kernel, 2, pairBuffer);
                setOpenClArg(kernel, 3, rowCount);
                setOpenClArg(kernel, 4, columnCount);
                setOpenClArg(kernel, 7, tolerance);
                constexpr std::size_t localSize = 64;
                const int sweeps = std::is_same_v<Scalar, float> ? 8 : 12;
                for (int sweep = 0; sweep < sweeps; ++sweep)
                {
                    for (std::size_t round = 0; round < schedule.offsets.size(); ++round)
                    {
                        const cl_ulong offset = static_cast<cl_ulong>(schedule.offsets[round]);
                        const cl_ulong count = static_cast<cl_ulong>(schedule.counts[round]);
                        setOpenClArg(kernel, 5, offset);
                        setOpenClArg(kernel, 6, count);
                        const std::size_t global = schedule.counts[round] * localSize;
                        checkOpenCl(clEnqueueNDRangeKernel(opencl::NativeAccess::queue(work.context()),
                                                           kernel,
                                                           1,
                                                           nullptr,
                                                           &global,
                                                           &localSize,
                                                           0,
                                                           nullptr,
                                                           nullptr),
                                    "clEnqueueNDRangeKernel SVD");
                    }
                }
                checkOpenCl(clFinish(opencl::NativeAccess::queue(work.context())), "clFinish SVD");
                std::vector<Scalar> hostWork(work.size());
                std::vector<Scalar> hostVectors(vectors.size());
                work.copyToHost(hostWork.data(), hostWork.size());
                vectors.copyToHost(hostVectors.data(), hostVectors.size());
                clReleaseKernel(kernel);
                return finalizePortableSvd(hostWork, hostVectors, rows, columns);
            }
            catch (...)
            {
                clReleaseKernel(kernel);
                throw;
            }
        }

        template <typename Scalar>
        GpuSelfAdjointEigenResult<Scalar> openClSelfAdjointEigen(const Scalar* inputValues, Index size)
        {
            const auto context = sharedContext(Backend::OpenCl);
            std::vector<Scalar> symmetric(static_cast<std::size_t>(size * size));
            std::vector<Scalar> identity(static_cast<std::size_t>(size * size), Scalar{});
            for (Index column = 0; column < size; ++column)
            {
                identity[static_cast<std::size_t>(column + column * size)] = Scalar{1};
                for (Index row = 0; row < size; ++row)
                    symmetric[static_cast<std::size_t>(row + column * size)] =
                        row >= column ? inputValues[row + column * size] : inputValues[column + row * size];
            }
            ResidentMatrix<Scalar> matrix(size, size, context);
            ResidentMatrix<Scalar> vectors(size, size, context);
            matrix.copyFromHost(symmetric.data(), symmetric.size());
            vectors.copyFromHost(identity.data(), identity.size());
            const auto schedule = makeJacobiPairSchedule<Scalar>(size);
            if (schedule.pairs.empty())
                return finalizePortableEigen(symmetric, identity, size);
            ResidentMatrix<Scalar> pairs(static_cast<Index>(schedule.pairs.size()), 1, context);
            ResidentMatrix<Scalar> rotations(static_cast<Index>(2 * ((size + 1) / 2)), 1, context);
            pairs.copyFromHost(schedule.pairs.data(), schedule.pairs.size());
            auto& program = openClProgram<Scalar>(matrix.context());
            cl_kernel kernel = program.kernel("eigen_jacobi_op");
            try
            {
                const cl_mem matrixBuffer = opencl::NativeAccess::buffer(matrix);
                const cl_mem vectorBuffer = opencl::NativeAccess::buffer(vectors);
                const cl_ulong matrixSize = static_cast<cl_ulong>(size);
                const Scalar tolerance = std::numeric_limits<Scalar>::epsilon() * static_cast<Scalar>(size) * Scalar{4};
                setOpenClArg(kernel, 0, matrixBuffer);
                setOpenClArg(kernel, 1, vectorBuffer);
                const cl_mem pairBuffer = opencl::NativeAccess::buffer(pairs);
                const cl_mem rotationBuffer = opencl::NativeAccess::buffer(rotations);
                setOpenClArg(kernel, 2, pairBuffer);
                setOpenClArg(kernel, 3, rotationBuffer);
                setOpenClArg(kernel, 4, matrixSize);
                setOpenClArg(kernel, 8, tolerance);
                constexpr std::size_t localSize = 64;
                const int sweeps = std::is_same_v<Scalar, float> ? 8 : 12;
                for (int sweep = 0; sweep < sweeps; ++sweep)
                {
                    for (std::size_t round = 0; round < schedule.offsets.size(); ++round)
                    {
                        const cl_ulong offset = static_cast<cl_ulong>(schedule.offsets[round]);
                        const cl_ulong count = static_cast<cl_ulong>(schedule.counts[round]);
                        setOpenClArg(kernel, 5, offset);
                        setOpenClArg(kernel, 6, count);
                        const std::size_t global = schedule.counts[round] * localSize;
                        for (cl_uint operation = 0; operation < 3; ++operation)
                        {
                            setOpenClArg(kernel, 7, operation);
                            checkOpenCl(clEnqueueNDRangeKernel(opencl::NativeAccess::queue(matrix.context()),
                                                               kernel,
                                                               1,
                                                               nullptr,
                                                               &global,
                                                               &localSize,
                                                               0,
                                                               nullptr,
                                                               nullptr),
                                        "clEnqueueNDRangeKernel self-adjoint eigen");
                        }
                    }
                }
                checkOpenCl(clFinish(opencl::NativeAccess::queue(matrix.context())), "clFinish self-adjoint eigen");
                std::vector<Scalar> hostMatrix(matrix.size());
                std::vector<Scalar> hostVectors(vectors.size());
                matrix.copyToHost(hostMatrix.data(), hostMatrix.size());
                vectors.copyToHost(hostVectors.data(), hostVectors.size());
                clReleaseKernel(kernel);
                return finalizePortableEigen(hostMatrix, hostVectors, size);
            }
            catch (...)
            {
                clReleaseKernel(kernel);
                throw;
            }
        }
    } // namespace
#endif

#ifdef PLAMATRIX_WITH_VULKAN
    namespace
    {
        struct VulkanElementwiseParameters
        {
            std::uint32_t count;
            std::uint32_t operation;
            float factor;
        };

        struct VulkanGemmParameters
        {
            std::uint32_t rows;
            std::uint32_t columns;
            std::uint32_t inner;
        };

        struct VulkanTransposeParameters
        {
            std::uint32_t rows;
            std::uint32_t columns;
        };

        struct VulkanReductionParameters
        {
            std::uint32_t count;
            std::uint32_t operation;
        };

        struct VulkanQrParameters
        {
            std::uint32_t rows;
            std::uint32_t columns;
            std::uint32_t step;
            std::uint32_t operation;
            float epsilon;
        };

        struct VulkanJacobiParameters
        {
            std::uint32_t rows;
            std::uint32_t columns;
            std::uint32_t pairOffset;
            std::uint32_t pairCount;
            float tolerance;
        };

        struct VulkanEigenParameters
        {
            std::uint32_t size;
            std::uint32_t pairOffset;
            std::uint32_t pairCount;
            std::uint32_t operation;
            float tolerance;
        };

        vulkan::ComputePipeline& vulkanDenseElementwisePipeline(vulkan::Runtime& runtime)
        {
            static vulkan::ComputePipeline pipeline(
                runtime,
                "dense_elementwise",
                3,
                sizeof(VulkanElementwiseParameters),
                {VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT});
            return pipeline;
        }

        vulkan::ComputePipeline& vulkanDenseScalePipeline(vulkan::Runtime& runtime)
        {
            static vulkan::ComputePipeline pipeline(runtime,
                                                    "dense_scale",
                                                    2,
                                                    sizeof(VulkanElementwiseParameters),
                                                    {VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT});
            return pipeline;
        }

        bool supportsTiledVulkanGemm(vulkan::Runtime& runtime)
        {
            static const bool supported = [&runtime]
            {
                VkPhysicalDeviceProperties properties{};
                vkGetPhysicalDeviceProperties(runtime.physicalDevice(), &properties);
                return properties.limits.maxComputeWorkGroupInvocations >= 256 &&
                       properties.limits.maxComputeWorkGroupSize[0] >= 256 &&
                       properties.limits.maxComputeSharedMemorySize >= 2 * 16 * 16 * sizeof(float);
            }();
            return supported;
        }

        vulkan::ComputePipeline& vulkanDenseTiledGemmPipeline(vulkan::Runtime& runtime)
        {
            static vulkan::ComputePipeline pipeline(
                runtime,
                "dense_gemm",
                3,
                sizeof(VulkanGemmParameters),
                {VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT});
            return pipeline;
        }

        vulkan::ComputePipeline& vulkanDenseNaiveGemmPipeline(vulkan::Runtime& runtime)
        {
            static vulkan::ComputePipeline pipeline(
                runtime,
                "dense_gemm_naive",
                3,
                sizeof(VulkanGemmParameters),
                {VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT});
            return pipeline;
        }

        vulkan::ComputePipeline& vulkanDenseTransposePipeline(vulkan::Runtime& runtime)
        {
            static vulkan::ComputePipeline pipeline(runtime,
                                                    "dense_transpose",
                                                    2,
                                                    sizeof(VulkanTransposeParameters),
                                                    {VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT});
            return pipeline;
        }

        vulkan::ComputePipeline& vulkanDenseReductionPipeline(vulkan::Runtime& runtime)
        {
            static vulkan::ComputePipeline pipeline(
                runtime,
                "dense_reduce",
                3,
                sizeof(VulkanReductionParameters),
                {VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT});
            return pipeline;
        }

        vulkan::ComputePipeline& vulkanDenseQrPipeline(vulkan::Runtime& runtime)
        {
            static vulkan::ComputePipeline pipeline(runtime,
                                                    "dense_qr",
                                                    6,
                                                    sizeof(VulkanQrParameters),
                                                    {VK_ACCESS_SHADER_READ_BIT,
                                                     VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                                                     VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                                                     VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                                                     VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                                                     VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT});
            return pipeline;
        }

        vulkan::ComputePipeline& vulkanDenseSvdPipeline(vulkan::Runtime& runtime)
        {
            static vulkan::ComputePipeline pipeline(runtime,
                                                    "dense_svd_jacobi",
                                                    3,
                                                    sizeof(VulkanJacobiParameters),
                                                    {VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                                                     VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                                                     VK_ACCESS_SHADER_READ_BIT});
            return pipeline;
        }

        vulkan::ComputePipeline& vulkanDenseEigenPipeline(vulkan::Runtime& runtime)
        {
            static vulkan::ComputePipeline pipeline(runtime,
                                                    "dense_eigen_jacobi",
                                                    4,
                                                    sizeof(VulkanEigenParameters),
                                                    {VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                                                     VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                                                     VK_ACCESS_SHADER_READ_BIT,
                                                     VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT});
            return pipeline;
        }

        std::uint32_t checkedU32(Index value)
        {
            if (value < 0 || static_cast<std::uint64_t>(value) > UINT32_MAX)
            {
                throw Error(ErrorCode::InvalidArgument, "Vulkan dense matrix dimension exceeds uint32 range");
            }
            return static_cast<std::uint32_t>(value);
        }

        template <typename Scalar>
        std::shared_ptr<GpuStorage<Scalar>> vulkanElementwise(const GpuStorage<Scalar>& left,
                                                              const GpuStorage<Scalar>* right,
                                                              std::uint32_t operation,
                                                              Scalar factor)
        {
            static_assert(std::is_same_v<Scalar, float>);
            auto& input = *left.portableMatrix;
            auto& runtime = vulkan::NativeAccess::runtime(input.context());
            ResidentMatrix<Scalar> output(input.rows(), input.cols(), input.contextOwner());
            auto& pipeline =
                right == nullptr ? vulkanDenseScalePipeline(runtime) : vulkanDenseElementwisePipeline(runtime);
            std::vector<vulkan::Buffer*> buffers{vulkan::NativeAccess::buffer(input)};
            if (right != nullptr)
            {
                buffers.push_back(vulkan::NativeAccess::buffer(*right->portableMatrix));
            }
            buffers.push_back(vulkan::NativeAccess::buffer(output));
            VulkanElementwiseParameters parameters{
                checkedU32(static_cast<Index>(input.size())), operation, static_cast<float>(factor)};
            vulkan::CommandContext commands(runtime);
            commands.begin();
            commands.dispatch(pipeline, buffers, (input.size() + 255) / 256, &parameters, sizeof(parameters));
            commands.submitAndWait();
            return std::make_shared<GpuStorage<Scalar>>(Backend::Vulkan, std::move(output));
        }

        float vulkanReduction(const GpuStorage<float>& left, const GpuStorage<float>* right)
        {
            auto& matrix = *left.portableMatrix;
            if (matrix.size() == 0)
                return 0.0F;
            auto& runtime = vulkan::NativeAccess::runtime(matrix.context());
            constexpr std::size_t localSize = 128;
            const std::size_t maximumGroups = (matrix.size() + 2 * localSize - 1) / (2 * localSize);
            ResidentMatrix<float> first(static_cast<Index>(maximumGroups), 1, matrix.contextOwner());
            ResidentMatrix<float> second(static_cast<Index>(maximumGroups), 1, matrix.contextOwner());
            auto& pipeline = vulkanDenseReductionPipeline(runtime);
            vulkan::Buffer* currentLeft = vulkan::NativeAccess::buffer(matrix);
            vulkan::Buffer* currentRight =
                right == nullptr ? currentLeft : vulkan::NativeAccess::buffer(*right->portableMatrix);
            ResidentMatrix<float>* output = &first;
            std::size_t remaining = matrix.size();
            std::uint32_t operation = right == nullptr ? 0U : 1U;
            vulkan::CommandContext commands(runtime);
            commands.begin();
            do
            {
                const std::size_t groups = (remaining + 2 * localSize - 1) / (2 * localSize);
                const VulkanReductionParameters parameters{checkedU32(static_cast<Index>(remaining)), operation};
                commands.dispatch(pipeline,
                                  {currentLeft, currentRight, vulkan::NativeAccess::buffer(*output)},
                                  groups,
                                  &parameters,
                                  sizeof(parameters));
                remaining = groups;
                currentLeft = vulkan::NativeAccess::buffer(*output);
                currentRight = currentLeft;
                operation = 0U;
                output = output == &first ? &second : &first;
            } while (remaining > 1);
            commands.submitAndWait();
            float result = 0.0F;
            (output == &first ? second : first).copyToHost(&result, 1);
            return result;
        }

        GpuQrResult<float> vulkanQr(const float* inputValues, Index rows, Index columns)
        {
            const auto context = sharedContext(Backend::Vulkan);
            ResidentMatrix<float> input(rows, columns, context);
            ResidentMatrix<float> qr(rows, columns, context);
            ResidentMatrix<float> tau(std::min(rows, columns), 1, context);
            ResidentMatrix<float> q(rows, rows, context);
            ResidentMatrix<float> r(rows, columns, context);
            ResidentMatrix<float> status(1, 1, context);
            input.copyFromHost(inputValues, input.size());
            auto& runtime = vulkan::NativeAccess::runtime(input.context());
            auto& pipeline = vulkanDenseQrPipeline(runtime);
            const std::vector<vulkan::Buffer*> buffers{vulkan::NativeAccess::buffer(input),
                                                       vulkan::NativeAccess::buffer(qr),
                                                       vulkan::NativeAccess::buffer(tau),
                                                       vulkan::NativeAccess::buffer(q),
                                                       vulkan::NativeAccess::buffer(r),
                                                       vulkan::NativeAccess::buffer(status)};
            vulkan::CommandContext commands(runtime);
            commands.begin();
            const std::size_t localSize = 128;
            const std::size_t initialCount = std::max(input.size(), q.size());
            const auto dispatch = [&](Index step, std::uint32_t operation, std::size_t groups)
            {
                const VulkanQrParameters parameters{checkedU32(rows),
                                                    checkedU32(columns),
                                                    checkedU32(step),
                                                    operation,
                                                    std::numeric_limits<float>::epsilon()};
                commands.dispatch(pipeline, buffers, groups, &parameters, sizeof(parameters));
            };
            dispatch(0, 0, (initialCount + localSize - 1) / localSize);
            const Index steps = std::min(rows, columns);
            for (Index step = 0; step < steps; ++step)
            {
                dispatch(step, 1, 1);
                if (columns - step - 1 > 0)
                    dispatch(step, 2, static_cast<std::size_t>(columns - step - 1));
            }
            dispatch(0, 3, (initialCount + localSize - 1) / localSize);
            for (Index step = steps; step-- > 0;)
                dispatch(step, 4, static_cast<std::size_t>(rows));
            commands.submitAndWait();
            GpuQrResult<float> result;
            result.q.resize(q.size());
            result.r.resize(r.size());
            float rank = 0.0F;
            q.copyToHost(result.q.data(), result.q.size());
            r.copyToHost(result.r.data(), result.r.size());
            status.copyToHost(&rank, 1);
            result.rank = static_cast<Index>(std::llround(rank));
            return result;
        }

        GpuSvdResult<float> vulkanSvd(const float* inputValues, Index rows, Index columns)
        {
            const auto context = sharedContext(Backend::Vulkan);
            ResidentMatrix<float> work(rows, columns, context);
            ResidentMatrix<float> vectors(columns, columns, context);
            work.copyFromHost(inputValues, work.size());
            std::vector<float> identity(static_cast<std::size_t>(columns * columns), 0.0F);
            for (Index index = 0; index < columns; ++index)
                identity[static_cast<std::size_t>(index + index * columns)] = 1.0F;
            vectors.copyFromHost(identity.data(), identity.size());
            const auto schedule = makeJacobiPairSchedule<float>(columns);
            if (schedule.pairs.empty())
            {
                std::vector<float> hostWork(work.size());
                work.copyToHost(hostWork.data(), hostWork.size());
                return finalizePortableSvd(hostWork, identity, rows, columns);
            }
            ResidentMatrix<float> pairs(static_cast<Index>(schedule.pairs.size()), 1, context);
            pairs.copyFromHost(schedule.pairs.data(), schedule.pairs.size());
            auto& runtime = vulkan::NativeAccess::runtime(work.context());
            vulkan::CommandContext commands(runtime, 64, 192);
            commands.begin();
            const float tolerance =
                std::numeric_limits<float>::epsilon() * static_cast<float>(std::max(rows, columns)) * 4.0F;
            auto& pipeline = vulkanDenseSvdPipeline(runtime);
            const std::vector<vulkan::Buffer*> buffers{vulkan::NativeAccess::buffer(work),
                                                       vulkan::NativeAccess::buffer(vectors),
                                                       vulkan::NativeAccess::buffer(pairs)};
            for (int sweep = 0; sweep < 8; ++sweep)
            {
                for (std::size_t round = 0; round < schedule.offsets.size(); ++round)
                {
                    const VulkanJacobiParameters parameters{checkedU32(rows),
                                                            checkedU32(columns),
                                                            checkedU32(static_cast<Index>(schedule.offsets[round])),
                                                            checkedU32(static_cast<Index>(schedule.counts[round])),
                                                            tolerance};
                    commands.dispatch(pipeline, buffers, schedule.counts[round], &parameters, sizeof(parameters));
                }
            }
            commands.submitAndWait();
            std::vector<float> hostWork(work.size());
            std::vector<float> hostVectors(vectors.size());
            work.copyToHost(hostWork.data(), hostWork.size());
            vectors.copyToHost(hostVectors.data(), hostVectors.size());
            return finalizePortableSvd(hostWork, hostVectors, rows, columns);
        }

        GpuSelfAdjointEigenResult<float> vulkanSelfAdjointEigen(const float* inputValues, Index size)
        {
            const auto context = sharedContext(Backend::Vulkan);
            std::vector<float> symmetric(static_cast<std::size_t>(size * size));
            std::vector<float> identity(static_cast<std::size_t>(size * size), 0.0F);
            for (Index column = 0; column < size; ++column)
            {
                identity[static_cast<std::size_t>(column + column * size)] = 1.0F;
                for (Index row = 0; row < size; ++row)
                    symmetric[static_cast<std::size_t>(row + column * size)] =
                        row >= column ? inputValues[row + column * size] : inputValues[column + row * size];
            }
            ResidentMatrix<float> matrix(size, size, context);
            ResidentMatrix<float> vectors(size, size, context);
            matrix.copyFromHost(symmetric.data(), symmetric.size());
            vectors.copyFromHost(identity.data(), identity.size());
            const auto schedule = makeJacobiPairSchedule<float>(size);
            if (schedule.pairs.empty())
                return finalizePortableEigen(symmetric, identity, size);
            ResidentMatrix<float> pairs(static_cast<Index>(schedule.pairs.size()), 1, context);
            ResidentMatrix<float> rotations(static_cast<Index>(2 * ((size + 1) / 2)), 1, context);
            pairs.copyFromHost(schedule.pairs.data(), schedule.pairs.size());
            auto& runtime = vulkan::NativeAccess::runtime(matrix.context());
            vulkan::CommandContext commands(runtime, 64, 256);
            commands.begin();
            const float tolerance = std::numeric_limits<float>::epsilon() * static_cast<float>(size) * 4.0F;
            auto& pipeline = vulkanDenseEigenPipeline(runtime);
            const std::vector<vulkan::Buffer*> buffers{vulkan::NativeAccess::buffer(matrix),
                                                       vulkan::NativeAccess::buffer(vectors),
                                                       vulkan::NativeAccess::buffer(pairs),
                                                       vulkan::NativeAccess::buffer(rotations)};
            for (int sweep = 0; sweep < 8; ++sweep)
            {
                for (std::size_t round = 0; round < schedule.offsets.size(); ++round)
                {
                    for (std::uint32_t operation = 0; operation < 3; ++operation)
                    {
                        const VulkanEigenParameters parameters{checkedU32(size),
                                                               checkedU32(static_cast<Index>(schedule.offsets[round])),
                                                               checkedU32(static_cast<Index>(schedule.counts[round])),
                                                               operation,
                                                               tolerance};
                        commands.dispatch(pipeline, buffers, schedule.counts[round], &parameters, sizeof(parameters));
                    }
                }
            }
            commands.submitAndWait();
            std::vector<float> hostMatrix(matrix.size());
            std::vector<float> hostVectors(vectors.size());
            matrix.copyToHost(hostMatrix.data(), hostMatrix.size());
            vectors.copyToHost(hostVectors.data(), hostVectors.size());
            return finalizePortableEigen(hostMatrix, hostVectors, size);
        }
    } // namespace
#endif

    template <typename Scalar>
    std::shared_ptr<GpuStorage<Scalar>> uploadGpu(Backend backend, const Scalar* source, Index rows, Index columns)
    {
        if (!backendAvailable<Scalar>(backend))
        {
            unavailable(backend);
        }
        if (backend == Backend::Cuda)
        {
#ifdef PLAMATRIX_WITH_CUDA
            auto host = DenseStorage<Scalar, Device::CPU>::uninitialized(rows, columns);
            std::memcpy(host.data(), source, static_cast<std::size_t>(rows * columns) * sizeof(Scalar));
            return std::make_shared<GpuStorage<Scalar>>(host.toGpu());
#endif
        }
        const auto context = sharedContext(backend);
        ResidentMatrix<Scalar> result(rows, columns, context);
        result.copyFromHost(source, static_cast<std::size_t>(rows * columns));
        return std::make_shared<GpuStorage<Scalar>>(backend, std::move(result));
    }

    template <typename Scalar> void downloadGpu(const GpuStorage<Scalar>& source, Scalar* destination)
    {
        if (source.selectedBackend == Backend::Cuda)
        {
#ifdef PLAMATRIX_WITH_CUDA
            auto host = source.cudaMatrix->toCpu();
            std::memcpy(destination, host.data(), static_cast<std::size_t>(host.size()) * sizeof(Scalar));
            return;
#endif
        }
        source.portableMatrix->copyToHost(destination, source.portableMatrix->size());
    }

    template <typename Scalar>
    std::shared_ptr<GpuStorage<Scalar>> multiplyGpu(const GpuStorage<Scalar>& left, const GpuStorage<Scalar>& right)
    {
        if (left.selectedBackend == Backend::Cuda)
        {
#ifdef PLAMATRIX_WITH_CUDA
            return std::make_shared<GpuStorage<Scalar>>(gemm(*left.cudaMatrix, *right.cudaMatrix));
#endif
        }
#ifdef PLAMATRIX_WITH_OPENCL
        if (left.selectedBackend == Backend::OpenCl)
        {
            auto& left_matrix = *left.portableMatrix;
            auto& right_matrix = *right.portableMatrix;
            auto& program = openClProgram<Scalar>(left_matrix.context());
            ResidentMatrix<Scalar> output(left_matrix.rows(), right_matrix.cols(), left_matrix.contextOwner());
            constexpr std::size_t tile_size = 16;
            constexpr std::size_t tile_threads = tile_size * tile_size;
            const bool tiled = program.supportsTiledGemm() && left_matrix.rows() >= tile_size &&
                               right_matrix.cols() >= tile_size && left_matrix.cols() >= tile_size;
            cl_kernel kernel = program.kernel(tiled ? "gemm_tiled_op" : "gemm_naive_op");
            try
            {
                const cl_mem left_buffer = opencl::NativeAccess::buffer(left_matrix);
                const cl_mem right_buffer = opencl::NativeAccess::buffer(right_matrix);
                const cl_mem output_buffer = opencl::NativeAccess::buffer(output);
                const cl_ulong rows = static_cast<cl_ulong>(left_matrix.rows());
                const cl_ulong columns = static_cast<cl_ulong>(right_matrix.cols());
                const cl_ulong inner = static_cast<cl_ulong>(left_matrix.cols());
                setOpenClArg(kernel, 0, left_buffer);
                setOpenClArg(kernel, 1, right_buffer);
                setOpenClArg(kernel, 2, output_buffer);
                setOpenClArg(kernel, 3, rows);
                setOpenClArg(kernel, 4, columns);
                setOpenClArg(kernel, 5, inner);
                const std::size_t tile_rows =
                    (static_cast<std::size_t>(left_matrix.rows()) + tile_size - 1) / tile_size;
                const std::size_t tile_columns =
                    (static_cast<std::size_t>(right_matrix.cols()) + tile_size - 1) / tile_size;
                const std::size_t global = tiled ? tile_rows * tile_columns * tile_threads : output.size();
                const std::size_t local = tile_threads;
                checkOpenCl(clEnqueueNDRangeKernel(opencl::NativeAccess::queue(left_matrix.context()),
                                                   kernel,
                                                   1,
                                                   nullptr,
                                                   &global,
                                                   tiled ? &local : nullptr,
                                                   0,
                                                   nullptr,
                                                   nullptr),
                            "clEnqueueNDRangeKernel");
                checkOpenCl(clFinish(opencl::NativeAccess::queue(left_matrix.context())), "clFinish");
                clReleaseKernel(kernel);
            }
            catch (...)
            {
                clReleaseKernel(kernel);
                throw;
            }
            return std::make_shared<GpuStorage<Scalar>>(Backend::OpenCl, std::move(output));
        }
#endif
#ifdef PLAMATRIX_WITH_VULKAN
        if constexpr (std::is_same_v<Scalar, float>)
        {
            if (left.selectedBackend == Backend::Vulkan)
            {
                auto& left_matrix = *left.portableMatrix;
                auto& right_matrix = *right.portableMatrix;
                auto& runtime = vulkan::NativeAccess::runtime(left_matrix.context());
                ResidentMatrix<Scalar> output(left_matrix.rows(), right_matrix.cols(), left_matrix.contextOwner());
                const bool tiled = supportsTiledVulkanGemm(runtime);
                auto& pipeline = tiled ? vulkanDenseTiledGemmPipeline(runtime) : vulkanDenseNaiveGemmPipeline(runtime);
                VulkanGemmParameters parameters{
                    checkedU32(left_matrix.rows()), checkedU32(right_matrix.cols()), checkedU32(left_matrix.cols())};
                vulkan::CommandContext commands(runtime);
                commands.begin();
                constexpr std::size_t tile_size = 16;
                const std::size_t tile_rows =
                    (static_cast<std::size_t>(left_matrix.rows()) + tile_size - 1) / tile_size;
                const std::size_t tile_columns =
                    (static_cast<std::size_t>(right_matrix.cols()) + tile_size - 1) / tile_size;
                const std::size_t scalar_groups = (output.size() + 63) / 64;
                commands.dispatch(pipeline,
                                  {vulkan::NativeAccess::buffer(left_matrix),
                                   vulkan::NativeAccess::buffer(right_matrix),
                                   vulkan::NativeAccess::buffer(output)},
                                  tiled ? tile_rows * tile_columns : scalar_groups,
                                  &parameters,
                                  sizeof(parameters));
                commands.submitAndWait();
                return std::make_shared<GpuStorage<Scalar>>(Backend::Vulkan, std::move(output));
            }
        }
#endif
        unavailable(left.selectedBackend);
    }

    template <typename Scalar>
    std::shared_ptr<GpuStorage<Scalar>>
    binaryGpu(const GpuStorage<Scalar>& left, const GpuStorage<Scalar>& right, std::uint32_t operation)
    {
        if (left.selectedBackend == Backend::Cuda)
        {
#ifdef PLAMATRIX_WITH_CUDA
            if (operation == 0)
                return std::make_shared<GpuStorage<Scalar>>(add(*left.cudaMatrix, *right.cudaMatrix));
            if (operation == 1)
                return std::make_shared<GpuStorage<Scalar>>(sub(*left.cudaMatrix, *right.cudaMatrix));
            return std::make_shared<GpuStorage<Scalar>>(hadamardMultiply(*left.cudaMatrix, *right.cudaMatrix, nullptr));
#endif
        }
#ifdef PLAMATRIX_WITH_OPENCL
        if (left.selectedBackend == Backend::OpenCl)
            return openClBinary(left, right, operation);
#endif
#ifdef PLAMATRIX_WITH_VULKAN
        if constexpr (std::is_same_v<Scalar, float>)
        {
            if (left.selectedBackend == Backend::Vulkan)
                return vulkanElementwise(left, &right, operation, Scalar{});
        }
#endif
        unavailable(left.selectedBackend);
    }

    template <typename Scalar>
    std::shared_ptr<GpuStorage<Scalar>> scaleGpu(const GpuStorage<Scalar>& input, Scalar value)
    {
        if (input.selectedBackend == Backend::Cuda)
        {
#ifdef PLAMATRIX_WITH_CUDA
            return std::make_shared<GpuStorage<Scalar>>(scalarMultiply(*input.cudaMatrix, value, nullptr));
#endif
        }
#ifdef PLAMATRIX_WITH_OPENCL
        if (input.selectedBackend == Backend::OpenCl)
        {
            auto& matrix = *input.portableMatrix;
            auto& program = openClProgram<Scalar>(matrix.context());
            ResidentMatrix<Scalar> output(matrix.rows(), matrix.cols(), matrix.contextOwner());
            cl_kernel kernel = program.kernel("scale_op");
            try
            {
                const cl_mem input_buffer = opencl::NativeAccess::buffer(matrix);
                const cl_mem output_buffer = opencl::NativeAccess::buffer(output);
                const cl_ulong count = static_cast<cl_ulong>(matrix.size());
                setOpenClArg(kernel, 0, input_buffer);
                setOpenClArg(kernel, 1, output_buffer);
                setOpenClArg(kernel, 2, value);
                setOpenClArg(kernel, 3, count);
                const std::size_t global = matrix.size();
                checkOpenCl(clEnqueueNDRangeKernel(opencl::NativeAccess::queue(matrix.context()),
                                                   kernel,
                                                   1,
                                                   nullptr,
                                                   &global,
                                                   nullptr,
                                                   0,
                                                   nullptr,
                                                   nullptr),
                            "clEnqueueNDRangeKernel");
                checkOpenCl(clFinish(opencl::NativeAccess::queue(matrix.context())), "clFinish");
                clReleaseKernel(kernel);
            }
            catch (...)
            {
                clReleaseKernel(kernel);
                throw;
            }
            return std::make_shared<GpuStorage<Scalar>>(Backend::OpenCl, std::move(output));
        }
#endif
#ifdef PLAMATRIX_WITH_VULKAN
        if constexpr (std::is_same_v<Scalar, float>)
        {
            if (input.selectedBackend == Backend::Vulkan)
                return vulkanElementwise(input, static_cast<const GpuStorage<Scalar>*>(nullptr), 3, value);
        }
#endif
        unavailable(input.selectedBackend);
    }

    template <typename Scalar> std::shared_ptr<GpuStorage<Scalar>> transposeGpu(const GpuStorage<Scalar>& input)
    {
        if (input.selectedBackend == Backend::Cuda)
        {
#ifdef PLAMATRIX_WITH_CUDA
            return std::make_shared<GpuStorage<Scalar>>(input.cudaMatrix->transpose());
#endif
        }
#ifdef PLAMATRIX_WITH_OPENCL
        if (input.selectedBackend == Backend::OpenCl)
        {
            auto& matrix = *input.portableMatrix;
            auto& program = openClProgram<Scalar>(matrix.context());
            ResidentMatrix<Scalar> output(matrix.cols(), matrix.rows(), matrix.contextOwner());
            cl_kernel kernel = program.kernel("transpose_op");
            try
            {
                const cl_mem input_buffer = opencl::NativeAccess::buffer(matrix);
                const cl_mem output_buffer = opencl::NativeAccess::buffer(output);
                const cl_ulong rows = static_cast<cl_ulong>(matrix.rows());
                const cl_ulong columns = static_cast<cl_ulong>(matrix.cols());
                setOpenClArg(kernel, 0, input_buffer);
                setOpenClArg(kernel, 1, output_buffer);
                setOpenClArg(kernel, 2, rows);
                setOpenClArg(kernel, 3, columns);
                const std::size_t global = matrix.size();
                checkOpenCl(clEnqueueNDRangeKernel(opencl::NativeAccess::queue(matrix.context()),
                                                   kernel,
                                                   1,
                                                   nullptr,
                                                   &global,
                                                   nullptr,
                                                   0,
                                                   nullptr,
                                                   nullptr),
                            "clEnqueueNDRangeKernel");
                checkOpenCl(clFinish(opencl::NativeAccess::queue(matrix.context())), "clFinish");
                clReleaseKernel(kernel);
            }
            catch (...)
            {
                clReleaseKernel(kernel);
                throw;
            }
            return std::make_shared<GpuStorage<Scalar>>(Backend::OpenCl, std::move(output));
        }
#endif
#ifdef PLAMATRIX_WITH_VULKAN
        if constexpr (std::is_same_v<Scalar, float>)
        {
            if (input.selectedBackend == Backend::Vulkan)
            {
                auto& matrix = *input.portableMatrix;
                auto& runtime = vulkan::NativeAccess::runtime(matrix.context());
                ResidentMatrix<Scalar> output(matrix.cols(), matrix.rows(), matrix.contextOwner());
                auto& pipeline = vulkanDenseTransposePipeline(runtime);
                VulkanTransposeParameters parameters{checkedU32(matrix.rows()), checkedU32(matrix.cols())};
                vulkan::CommandContext commands(runtime);
                commands.begin();
                commands.dispatch(pipeline,
                                  {vulkan::NativeAccess::buffer(matrix), vulkan::NativeAccess::buffer(output)},
                                  (matrix.size() + 255) / 256,
                                  &parameters,
                                  sizeof(parameters));
                commands.submitAndWait();
                return std::make_shared<GpuStorage<Scalar>>(Backend::Vulkan, std::move(output));
            }
        }
#endif
        unavailable(input.selectedBackend);
    }

    template <typename Scalar> Scalar sumGpu(const GpuStorage<Scalar>& input)
    {
        if (input.selectedBackend == Backend::Cuda)
        {
#ifdef PLAMATRIX_WITH_CUDA
            auto output = internal::sum(*input.cudaMatrix, ReductionAxis::All).toCpu();
            return output(0, 0);
#endif
        }
#ifdef PLAMATRIX_WITH_OPENCL
        if (input.selectedBackend == Backend::OpenCl)
            return openClReduction(input, static_cast<const GpuStorage<Scalar>*>(nullptr));
#endif
#ifdef PLAMATRIX_WITH_VULKAN
        if constexpr (std::is_same_v<Scalar, float>)
            if (input.selectedBackend == Backend::Vulkan)
                return vulkanReduction(input, nullptr);
#endif
        unavailable(input.selectedBackend);
    }

    template <typename Scalar> Scalar dotGpu(const GpuStorage<Scalar>& left, const GpuStorage<Scalar>& right)
    {
        if (left.selectedBackend != right.selectedBackend)
            throw Error(ErrorCode::InvalidArgument, "dense GPU dot requires operands on the same backend");
        if (left.selectedBackend == Backend::Cuda)
        {
#ifdef PLAMATRIX_WITH_CUDA
            return cudaDot(*left.cudaMatrix, *right.cudaMatrix);
#endif
        }
#ifdef PLAMATRIX_WITH_OPENCL
        if (left.selectedBackend == Backend::OpenCl)
            return openClReduction(left, &right);
#endif
#ifdef PLAMATRIX_WITH_VULKAN
        if constexpr (std::is_same_v<Scalar, float>)
            if (left.selectedBackend == Backend::Vulkan)
                return vulkanReduction(left, &right);
#endif
        unavailable(left.selectedBackend);
    }

    template <typename Scalar>
    GpuQrResult<Scalar> qrGpu(Backend backend, const Scalar* input, Index rows, Index columns)
    {
        if (!backendAvailable<Scalar>(backend))
            unavailable(backend);
#ifdef PLAMATRIX_WITH_OPENCL
        if (backend == Backend::OpenCl)
            return openClQr(input, rows, columns);
#endif
#ifdef PLAMATRIX_WITH_VULKAN
        if constexpr (std::is_same_v<Scalar, float>)
            if (backend == Backend::Vulkan)
                return vulkanQr(input, rows, columns);
#endif
        throw Error(ErrorCode::UnsupportedOperation, "QR is unavailable on the selected GPU backend", backend);
    }

    template <typename Scalar>
    GpuSvdResult<Scalar> svdGpu(Backend backend, const Scalar* input, Index rows, Index columns)
    {
        if (!backendAvailable<Scalar>(backend))
            unavailable(backend);
#ifdef PLAMATRIX_WITH_OPENCL
        if (backend == Backend::OpenCl)
            return openClSvd(input, rows, columns);
#endif
#ifdef PLAMATRIX_WITH_VULKAN
        if constexpr (std::is_same_v<Scalar, float>)
            if (backend == Backend::Vulkan)
                return vulkanSvd(input, rows, columns);
#endif
        throw Error(ErrorCode::UnsupportedOperation, "SVD is unavailable on the selected GPU backend", backend);
    }

    template <typename Scalar>
    GpuSelfAdjointEigenResult<Scalar> selfAdjointEigenGpu(Backend backend, const Scalar* input, Index size)
    {
        if (!backendAvailable<Scalar>(backend))
            unavailable(backend);
#ifdef PLAMATRIX_WITH_OPENCL
        if (backend == Backend::OpenCl)
            return openClSelfAdjointEigen(input, size);
#endif
#ifdef PLAMATRIX_WITH_VULKAN
        if constexpr (std::is_same_v<Scalar, float>)
            if (backend == Backend::Vulkan)
                return vulkanSelfAdjointEigen(input, size);
#endif
        throw Error(ErrorCode::UnsupportedOperation,
                    "Self-adjoint eigendecomposition is unavailable on the selected GPU backend",
                    backend);
    }

#define PLAMATRIX_DEFINE_AUTO_GPU_OPS(ScalarType)                                                                      \
    bool GpuOps<ScalarType>::available(Backend backend) noexcept                                                       \
    {                                                                                                                  \
        return backendAvailable<ScalarType>(backend);                                                                  \
    }                                                                                                                  \
    Backend GpuOps<ScalarType>::backend(const GpuStorage<ScalarType>& storage) noexcept                                \
    {                                                                                                                  \
        return storage.selectedBackend;                                                                                \
    }                                                                                                                  \
    std::shared_ptr<GpuStorage<ScalarType>> GpuOps<ScalarType>::upload(                                                \
        Backend backend, const ScalarType* source, Index rows, Index columns)                                          \
    {                                                                                                                  \
        return uploadGpu(backend, source, rows, columns);                                                              \
    }                                                                                                                  \
    void GpuOps<ScalarType>::download(const GpuStorage<ScalarType>& source, ScalarType* destination)                   \
    {                                                                                                                  \
        downloadGpu(source, destination);                                                                              \
    }                                                                                                                  \
    std::shared_ptr<GpuStorage<ScalarType>> GpuOps<ScalarType>::multiply(const GpuStorage<ScalarType>& left,           \
                                                                         const GpuStorage<ScalarType>& right)          \
    {                                                                                                                  \
        return multiplyGpu(left, right);                                                                               \
    }                                                                                                                  \
    std::shared_ptr<GpuStorage<ScalarType>> GpuOps<ScalarType>::add(const GpuStorage<ScalarType>& left,                \
                                                                    const GpuStorage<ScalarType>& right)               \
    {                                                                                                                  \
        return binaryGpu(left, right, 0);                                                                              \
    }                                                                                                                  \
    std::shared_ptr<GpuStorage<ScalarType>> GpuOps<ScalarType>::subtract(const GpuStorage<ScalarType>& left,           \
                                                                         const GpuStorage<ScalarType>& right)          \
    {                                                                                                                  \
        return binaryGpu(left, right, 1);                                                                              \
    }                                                                                                                  \
    std::shared_ptr<GpuStorage<ScalarType>> GpuOps<ScalarType>::scale(const GpuStorage<ScalarType>& input,             \
                                                                      ScalarType value)                                \
    {                                                                                                                  \
        return scaleGpu(input, value);                                                                                 \
    }                                                                                                                  \
    std::shared_ptr<GpuStorage<ScalarType>> GpuOps<ScalarType>::cwiseProduct(const GpuStorage<ScalarType>& left,       \
                                                                             const GpuStorage<ScalarType>& right)      \
    {                                                                                                                  \
        return binaryGpu(left, right, 2);                                                                              \
    }                                                                                                                  \
    std::shared_ptr<GpuStorage<ScalarType>> GpuOps<ScalarType>::transpose(const GpuStorage<ScalarType>& input)         \
    {                                                                                                                  \
        return transposeGpu(input);                                                                                    \
    }                                                                                                                  \
    ScalarType GpuOps<ScalarType>::sum(const GpuStorage<ScalarType>& input)                                            \
    {                                                                                                                  \
        return sumGpu(input);                                                                                          \
    }                                                                                                                  \
    ScalarType GpuOps<ScalarType>::dot(const GpuStorage<ScalarType>& left, const GpuStorage<ScalarType>& right)        \
    {                                                                                                                  \
        return dotGpu(left, right);                                                                                    \
    }                                                                                                                  \
    GpuQrResult<ScalarType> GpuOps<ScalarType>::qr(                                                                    \
        Backend backend, const ScalarType* input, Index rows, Index columns)                                           \
    {                                                                                                                  \
        return qrGpu(backend, input, rows, columns);                                                                   \
    }                                                                                                                  \
    GpuSvdResult<ScalarType> GpuOps<ScalarType>::svd(                                                                  \
        Backend backend, const ScalarType* input, Index rows, Index columns)                                           \
    {                                                                                                                  \
        return svdGpu(backend, input, rows, columns);                                                                  \
    }                                                                                                                  \
    GpuSelfAdjointEigenResult<ScalarType> GpuOps<ScalarType>::selfAdjointEigen(                                        \
        Backend backend, const ScalarType* input, Index size)                                                          \
    {                                                                                                                  \
        return selfAdjointEigenGpu(backend, input, size);                                                              \
    }

    PLAMATRIX_DEFINE_AUTO_GPU_OPS(float)
    PLAMATRIX_DEFINE_AUTO_GPU_OPS(double)

#undef PLAMATRIX_DEFINE_AUTO_GPU_OPS

} // namespace plamatrix::internal::detail
