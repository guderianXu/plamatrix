#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>

#include "plamatrix/core/error_check.h"
#include "plamatrix/ops/point_cloud.h"

namespace plamatrix
{

// ---- CUDA kernel: transform points ----

template <typename Scalar>
__global__ void transformPointsKernel(const Scalar* T, const Scalar* points, Scalar* result, Index N)
{
    Index i = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= N)
    {
        return;
    }

    // Read point (row i of Nx3 column-major points matrix)
    Scalar px = points[i + 0 * N];
    Scalar py = points[i + 1 * N];
    Scalar pz = points[i + 2 * N];

    // T is 4x4 column-major: T(row, col) = T[row + col * 4]
    // Compute p' = R * p + t
    result[i + 0 * N] = T[0] * px + T[4] * py + T[8] * pz + T[12];
    result[i + 1 * N] = T[1] * px + T[5] * py + T[9] * pz + T[13];
    result[i + 2 * N] = T[2] * px + T[6] * py + T[10] * pz + T[14];
}

namespace
{

int checkedCudaGrid1D(Index item_count, int block_size, const char* op)
{
    Index block_count = (item_count + block_size - 1) / block_size;
    if (block_count > static_cast<Index>(std::numeric_limits<int>::max()))
    {
        std::ostringstream oss;
        oss << op << ": item count exceeds CUDA grid range";
        throw std::runtime_error(oss.str());
    }
    return static_cast<int>(block_count);
}

template <typename Scalar>
void checkTransformInputs(const DenseMatrix<Scalar, Device::GPU>& T,
                          const DenseMatrix<Scalar, Device::GPU>& points)
{
    if (T.rows() != 4 || T.cols() != 4)
    {
        std::ostringstream oss;
        oss << "transformPoints: T must be 4x4, got " << T.rows() << "x" << T.cols();
        throw std::runtime_error(oss.str());
    }
    if (points.cols() != 3)
    {
        std::ostringstream oss;
        oss << "transformPoints: points must be Nx3, got " << points.rows() << "x" << points.cols();
        throw std::runtime_error(oss.str());
    }
}

template <typename Scalar>
void checkTransformOutput(const DenseMatrix<Scalar, Device::GPU>& output, Index point_count)
{
    if (output.rows() != point_count || output.cols() != 3)
    {
        std::ostringstream oss;
        oss << "transformPoints output dimension mismatch: output is "
            << output.rows() << "x" << output.cols()
            << ", expected " << point_count << "x3";
        throw std::runtime_error(oss.str());
    }
}

template <typename Scalar>
void checkCovarianceInputs(const DenseMatrix<Scalar, Device::GPU>& points)
{
    Index N = points.rows();
    if (N < 2)
    {
        throw std::runtime_error("covarianceMatrix: need at least 2 points");
    }
    if (points.cols() != 3)
    {
        std::ostringstream oss;
        oss << "covarianceMatrix: points must be Nx3, got " << points.rows() << "x" << points.cols();
        throw std::runtime_error(oss.str());
    }
}

template <typename Scalar>
void checkCovarianceOutput(const DenseMatrix<Scalar, Device::GPU>& output)
{
    if (output.rows() != 3 || output.cols() != 3)
    {
        std::ostringstream oss;
        oss << "covarianceMatrix output dimension mismatch: output is "
            << output.rows() << "x" << output.cols()
            << ", expected 3x3";
        throw std::runtime_error(oss.str());
    }
}

} // anonymous namespace

template <typename Scalar>
using CovarianceAccum = std::conditional_t<std::is_same_v<Scalar, float>, double, Scalar>;

template <typename Scalar, typename Accum>
__global__ void covarianceMeanPartialKernel(const Scalar* points, Accum* partial, Index N)
{
    constexpr int stat_count = 3;
    constexpr int block_size = 256;
    __shared__ Accum shared[stat_count * block_size];

    int tid = threadIdx.x;
    Accum local[stat_count] = {};

    for (Index i = static_cast<Index>(blockIdx.x) * blockDim.x + tid;
         i < N;
         i += static_cast<Index>(blockDim.x) * gridDim.x)
    {
        Accum x = static_cast<Accum>(points[i + 0 * N]);
        Accum y = static_cast<Accum>(points[i + 1 * N]);
        Accum z = static_cast<Accum>(points[i + 2 * N]);

        local[0] += x;
        local[1] += y;
        local[2] += z;
    }

    for (int stat = 0; stat < stat_count; ++stat)
    {
        shared[stat * blockDim.x + tid] = local[stat];
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            for (int stat = 0; stat < stat_count; ++stat)
            {
                shared[stat * blockDim.x + tid] += shared[stat * blockDim.x + tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        for (int stat = 0; stat < stat_count; ++stat)
        {
            partial[blockIdx.x + stat * gridDim.x] = shared[stat * blockDim.x];
        }
    }
}

template <typename Accum>
__global__ void covarianceMeanFinalizeKernel(const Accum* partial, Accum* mean, int partial_blocks, Index N)
{
    constexpr int stat_count = 3;
    constexpr int block_size = 256;
    __shared__ Accum shared[stat_count * block_size];

    int tid = threadIdx.x;
    Accum local[stat_count] = {};
    for (int i = tid; i < partial_blocks; i += blockDim.x)
    {
        for (int stat = 0; stat < stat_count; ++stat)
        {
            local[stat] += partial[i + stat * partial_blocks];
        }
    }

    for (int stat = 0; stat < stat_count; ++stat)
    {
        shared[stat * blockDim.x + tid] = local[stat];
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            for (int stat = 0; stat < stat_count; ++stat)
            {
                shared[stat * blockDim.x + tid] += shared[stat * blockDim.x + tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        Accum inv_n = static_cast<Accum>(1) / static_cast<Accum>(N);
        mean[0] = shared[0] * inv_n;
        mean[1] = shared[blockDim.x] * inv_n;
        mean[2] = shared[2 * blockDim.x] * inv_n;
    }
}

template <typename Scalar, typename Accum>
__global__ void covarianceCenteredPartialKernel(const Scalar* points, const Accum* mean, Accum* partial, Index N)
{
    constexpr int stat_count = 6;
    constexpr int block_size = 256;
    __shared__ Accum shared[stat_count * block_size];

    int tid = threadIdx.x;
    Accum local[stat_count] = {};
    Accum mean_x = mean[0];
    Accum mean_y = mean[1];
    Accum mean_z = mean[2];

    for (Index i = static_cast<Index>(blockIdx.x) * blockDim.x + tid;
         i < N;
         i += static_cast<Index>(blockDim.x) * gridDim.x)
    {
        Accum x = static_cast<Accum>(points[i + 0 * N]) - mean_x;
        Accum y = static_cast<Accum>(points[i + 1 * N]) - mean_y;
        Accum z = static_cast<Accum>(points[i + 2 * N]) - mean_z;

        local[0] += x * x;
        local[1] += x * y;
        local[2] += x * z;
        local[3] += y * y;
        local[4] += y * z;
        local[5] += z * z;
    }

    for (int stat = 0; stat < stat_count; ++stat)
    {
        shared[stat * blockDim.x + tid] = local[stat];
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            for (int stat = 0; stat < stat_count; ++stat)
            {
                shared[stat * blockDim.x + tid] += shared[stat * blockDim.x + tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        for (int stat = 0; stat < stat_count; ++stat)
        {
            partial[blockIdx.x + stat * gridDim.x] = shared[stat * blockDim.x];
        }
    }
}

template <typename Scalar, typename Accum>
__global__ void covarianceCenteredFinalizeKernel(const Accum* partial, Scalar* covariance, int partial_blocks, Index N)
{
    constexpr int stat_count = 6;
    constexpr int block_size = 256;
    __shared__ Accum shared[stat_count * block_size];

    int tid = threadIdx.x;
    Accum local[stat_count] = {};
    for (int i = tid; i < partial_blocks; i += blockDim.x)
    {
        for (int stat = 0; stat < stat_count; ++stat)
        {
            local[stat] += partial[i + stat * partial_blocks];
        }
    }

    for (int stat = 0; stat < stat_count; ++stat)
    {
        shared[stat * blockDim.x + tid] = local[stat];
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            for (int stat = 0; stat < stat_count; ++stat)
            {
                shared[stat * blockDim.x + tid] += shared[stat * blockDim.x + tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        Accum inv_n = static_cast<Accum>(1) / static_cast<Accum>(N);
        Accum c_xx = shared[0] * inv_n;
        Accum c_xy = shared[blockDim.x] * inv_n;
        Accum c_xz = shared[2 * blockDim.x] * inv_n;
        Accum c_yy = shared[3 * blockDim.x] * inv_n;
        Accum c_yz = shared[4 * blockDim.x] * inv_n;
        Accum c_zz = shared[5 * blockDim.x] * inv_n;

        covariance[0] = static_cast<Scalar>(c_xx);
        covariance[1] = static_cast<Scalar>(c_xy);
        covariance[2] = static_cast<Scalar>(c_xz);
        covariance[3] = static_cast<Scalar>(c_xy);
        covariance[4] = static_cast<Scalar>(c_yy);
        covariance[5] = static_cast<Scalar>(c_yz);
        covariance[6] = static_cast<Scalar>(c_xz);
        covariance[7] = static_cast<Scalar>(c_yz);
        covariance[8] = static_cast<Scalar>(c_zz);
    }
}

// ---- GPU implementations ----

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> rotationMatrixGpuImpl(const Vec3<Scalar>& axis, Scalar angle)
{
    // Compute Rodrigues formula on CPU, then transfer to GPU
    Scalar norm = std::sqrt(axis.x * axis.x + axis.y * axis.y + axis.z * axis.z);
    if (norm < static_cast<Scalar>(1e-12))
    {
        throw std::runtime_error("rotationMatrix: axis norm is too small");
    }

    Scalar x = axis.x / norm;
    Scalar y = axis.y / norm;
    Scalar z = axis.z / norm;

    Scalar c = std::cos(angle);
    Scalar s = std::sin(angle);
    Scalar one_minus_c = static_cast<Scalar>(1) - c;

    // Compute on CPU first
    Scalar R_cpu_data[9];
    R_cpu_data[0] = c + x * x * one_minus_c;        // R(0,0) = c + x^2*(1-c)
    R_cpu_data[1] = z * s + x * y * one_minus_c;     // R(1,0) = z*s + x*y*(1-c)
    R_cpu_data[2] = -y * s + x * z * one_minus_c;    // R(2,0) = -y*s + x*z*(1-c)
    R_cpu_data[3] = -z * s + y * x * one_minus_c;    // R(0,1) = -z*s + y*x*(1-c)
    R_cpu_data[4] = c + y * y * one_minus_c;         // R(1,1) = c + y^2*(1-c)
    R_cpu_data[5] = x * s + y * z * one_minus_c;     // R(2,1) = x*s + y*z*(1-c)
    R_cpu_data[6] = y * s + z * x * one_minus_c;     // R(0,2) = y*s + z*x*(1-c)
    R_cpu_data[7] = -x * s + z * y * one_minus_c;    // R(1,2) = -x*s + z*y*(1-c)
    R_cpu_data[8] = c + z * z * one_minus_c;         // R(2,2) = c + z^2*(1-c)

    DenseMatrix<Scalar, Device::GPU> R_gpu(3, 3);
    PLAMATRIX_CHECK_CUDA(
        cudaMemcpy(R_gpu.data(), R_cpu_data, static_cast<std::size_t>(9) * sizeof(Scalar),
                   cudaMemcpyHostToDevice));
    return R_gpu;
}

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> rigidTransformGpuImpl(const DenseMatrix<Scalar, Device::GPU>& R,
                                                         const Vec3<Scalar>& t)
{
    if (R.rows() != 3 || R.cols() != 3)
    {
        std::ostringstream oss;
        oss << "rigidTransform: R must be 3x3, got " << R.rows() << "x" << R.cols();
        throw std::runtime_error(oss.str());
    }

    // Copy R (3x3) from GPU to CPU
    Scalar R_cpu[9];
    PLAMATRIX_CHECK_CUDA(
        cudaMemcpy(R_cpu, R.data(), static_cast<std::size_t>(9) * sizeof(Scalar), cudaMemcpyDeviceToHost));

    // Build 4x4 rigid transform on CPU
    Scalar T_cpu[16];
    // Zero-initialize all elements
    for (int i = 0; i < 16; ++i)
    {
        T_cpu[i] = static_cast<Scalar>(0);
    }

    // Copy R into top-left 3x3 (column-major)
    for (Index col = 0; col < 3; ++col)
    {
        for (Index row = 0; row < 3; ++row)
        {
            T_cpu[row + col * 4] = R_cpu[row + col * 3];
        }
    }

    // Set translation
    T_cpu[0 + 3 * 4] = t.x;
    T_cpu[1 + 3 * 4] = t.y;
    T_cpu[2 + 3 * 4] = t.z;

    // Bottom row: [0, 0, 0, 1]
    T_cpu[3 + 3 * 4] = static_cast<Scalar>(1);

    // Transfer to GPU
    DenseMatrix<Scalar, Device::GPU> T_gpu(4, 4);
    PLAMATRIX_CHECK_CUDA(
        cudaMemcpy(T_gpu.data(), T_cpu, static_cast<std::size_t>(16) * sizeof(Scalar),
                   cudaMemcpyHostToDevice));
    return T_gpu;
}

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> transformPointsGpuImpl(const DenseMatrix<Scalar, Device::GPU>& T,
                                                          const DenseMatrix<Scalar, Device::GPU>& points)
{
    auto result = transformPointsAsync(T, points);
    PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(nullptr));
    return result;
}

template <typename Scalar>
void transformPointsAsync(const DenseMatrix<Scalar, Device::GPU>& T,
                          const DenseMatrix<Scalar, Device::GPU>& points,
                          DenseMatrix<Scalar, Device::GPU>& output,
                          cudaStream_t stream)
{
    checkTransformInputs(T, points);
    Index N = points.rows();
    checkTransformOutput(output, N);
    if (N == 0)
    {
        return;
    }

    constexpr int block_size = 256;
    int grid_size = checkedCudaGrid1D(N, block_size, "transformPoints");
    transformPointsKernel<<<grid_size, block_size, 0, stream>>>(T.data(), points.data(), output.data(), N);
    PLAMATRIX_CHECK_CUDA(cudaGetLastError());
}

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> transformPointsAsync(const DenseMatrix<Scalar, Device::GPU>& T,
                                                      const DenseMatrix<Scalar, Device::GPU>& points,
                                                      cudaStream_t stream)
{
    checkTransformInputs(T, points);
    DenseMatrix<Scalar, Device::GPU> result(points.rows(), 3);
    transformPointsAsync(T, points, result, stream);
    return result;
}

template <typename Scalar>
void transformPoints(const DenseMatrix<Scalar, Device::GPU>& T,
                     const DenseMatrix<Scalar, Device::GPU>& points,
                     DenseMatrix<Scalar, Device::GPU>& output,
                     cudaStream_t stream)
{
    transformPointsAsync(T, points, output, stream);
    PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
}

template <typename Scalar>
void covarianceMatrixGpuAsyncImpl(const DenseMatrix<Scalar, Device::GPU>& points,
                                  DenseMatrix<Scalar, Device::GPU>& output,
                                  GpuCovarianceWorkspace<Scalar>& workspace,
                                  cudaStream_t stream)
{
    checkCovarianceInputs(points);
    checkCovarianceOutput(output);

    Index N = points.rows();
    constexpr int block_size = 256;
    int block_count = checkedCudaGrid1D(N, block_size, "covarianceMatrix");

    using Accum = CovarianceAccum<Scalar>;
    workspace.reserveBlocks(block_count);
    DenseMatrix<Accum, Device::GPU>& partial_mean = workspace.partialMean();
    DenseMatrix<Accum, Device::GPU>& mean = workspace.mean();
    DenseMatrix<Accum, Device::GPU>& partial_covariance = workspace.partialCovariance();

    covarianceMeanPartialKernel<Scalar, Accum><<<block_count, block_size, 0, stream>>>(
        points.data(), partial_mean.data(), N);
    PLAMATRIX_CHECK_CUDA(cudaGetLastError());
    covarianceMeanFinalizeKernel<Accum><<<1, block_size, 0, stream>>>(
        partial_mean.data(), mean.data(), block_count, N);
    PLAMATRIX_CHECK_CUDA(cudaGetLastError());
    covarianceCenteredPartialKernel<Scalar, Accum><<<block_count, block_size, 0, stream>>>(
        points.data(), mean.data(), partial_covariance.data(), N);
    PLAMATRIX_CHECK_CUDA(cudaGetLastError());
    covarianceCenteredFinalizeKernel<Scalar, Accum><<<1, block_size, 0, stream>>>(
        partial_covariance.data(), output.data(), block_count, N);
    PLAMATRIX_CHECK_CUDA(cudaGetLastError());
}

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> covarianceMatrixGpuImpl(const DenseMatrix<Scalar, Device::GPU>& points)
{
    checkCovarianceInputs(points);
    DenseMatrix<Scalar, Device::GPU> output(3, 3);
    GpuCovarianceWorkspace<Scalar> workspace;
    covarianceMatrixGpuAsyncImpl(points, output, workspace, nullptr);
    PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(nullptr));
    return output;
}

template <typename Scalar>
void covarianceMatrix(const DenseMatrix<Scalar, Device::GPU>& points,
                      DenseMatrix<Scalar, Device::GPU>& output,
                      cudaStream_t stream)
{
    GpuCovarianceWorkspace<Scalar> workspace;
    covarianceMatrixGpuAsyncImpl(points, output, workspace, stream);
    PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
}

template <typename Scalar>
void covarianceMatrixAsync(const DenseMatrix<Scalar, Device::GPU>& points,
                           DenseMatrix<Scalar, Device::GPU>& output,
                           GpuCovarianceWorkspace<Scalar>& workspace,
                           cudaStream_t stream)
{
    covarianceMatrixGpuAsyncImpl(points, output, workspace, stream);
}

// ---- Explicit specializations for GPU ----

#ifdef PLAMATRIX_USE_FLOAT
template <>
DenseMatrix<float, Device::GPU> rotationMatrix<float, Device::GPU>(const Vec3<float>& axis, float angle)
{
    return rotationMatrixGpuImpl(axis, angle);
}
#endif

#ifdef PLAMATRIX_USE_DOUBLE
template <>
DenseMatrix<double, Device::GPU> rotationMatrix<double, Device::GPU>(const Vec3<double>& axis, double angle)
{
    return rotationMatrixGpuImpl(axis, angle);
}
#endif

#ifdef PLAMATRIX_USE_FLOAT
template <>
DenseMatrix<float, Device::GPU> rigidTransform<float, Device::GPU>(const DenseMatrix<float, Device::GPU>& R,
                                                                      const Vec3<float>& t)
{
    return rigidTransformGpuImpl(R, t);
}
#endif

#ifdef PLAMATRIX_USE_DOUBLE
template <>
DenseMatrix<double, Device::GPU> rigidTransform<double, Device::GPU>(const DenseMatrix<double, Device::GPU>& R,
                                                                       const Vec3<double>& t)
{
    return rigidTransformGpuImpl(R, t);
}
#endif

#ifdef PLAMATRIX_USE_FLOAT
template <>
DenseMatrix<float, Device::GPU> transformPoints<float, Device::GPU>(const DenseMatrix<float, Device::GPU>& T,
                                                                      const DenseMatrix<float, Device::GPU>& points)
{
    return transformPointsGpuImpl(T, points);
}
#endif

#ifdef PLAMATRIX_USE_DOUBLE
template <>
DenseMatrix<double, Device::GPU> transformPoints<double, Device::GPU>(const DenseMatrix<double, Device::GPU>& T,
                                                                        const DenseMatrix<double, Device::GPU>& points)
{
    return transformPointsGpuImpl(T, points);
}
#endif

#ifdef PLAMATRIX_USE_FLOAT
template DenseMatrix<float, Device::GPU> transformPointsAsync(const DenseMatrix<float, Device::GPU>&,
                                                              const DenseMatrix<float, Device::GPU>&,
                                                              cudaStream_t);

template void transformPointsAsync(const DenseMatrix<float, Device::GPU>&,
                                   const DenseMatrix<float, Device::GPU>&,
                                   DenseMatrix<float, Device::GPU>&,
                                   cudaStream_t);

template void transformPoints(const DenseMatrix<float, Device::GPU>&,
                              const DenseMatrix<float, Device::GPU>&,
                              DenseMatrix<float, Device::GPU>&,
                              cudaStream_t);
#endif

#ifdef PLAMATRIX_USE_DOUBLE
template DenseMatrix<double, Device::GPU> transformPointsAsync(const DenseMatrix<double, Device::GPU>&,
                                                               const DenseMatrix<double, Device::GPU>&,
                                                               cudaStream_t);

template void transformPointsAsync(const DenseMatrix<double, Device::GPU>&,
                                   const DenseMatrix<double, Device::GPU>&,
                                   DenseMatrix<double, Device::GPU>&,
                                   cudaStream_t);

template void transformPoints(const DenseMatrix<double, Device::GPU>&,
                              const DenseMatrix<double, Device::GPU>&,
                              DenseMatrix<double, Device::GPU>&,
                              cudaStream_t);
#endif

#ifdef PLAMATRIX_USE_FLOAT
template <>
DenseMatrix<float, Device::GPU> covarianceMatrix<float, Device::GPU>(const DenseMatrix<float, Device::GPU>& points)
{
    return covarianceMatrixGpuImpl(points);
}
#endif

#ifdef PLAMATRIX_USE_DOUBLE
template <>
DenseMatrix<double, Device::GPU> covarianceMatrix<double, Device::GPU>(const DenseMatrix<double, Device::GPU>& points)
{
    return covarianceMatrixGpuImpl(points);
}
#endif

#ifdef PLAMATRIX_USE_FLOAT
template void covarianceMatrix(const DenseMatrix<float, Device::GPU>&,
                               DenseMatrix<float, Device::GPU>&,
                               cudaStream_t);

template void covarianceMatrixAsync(const DenseMatrix<float, Device::GPU>&,
                                    DenseMatrix<float, Device::GPU>&,
                                    GpuCovarianceWorkspace<float>&,
                                    cudaStream_t);
#endif

#ifdef PLAMATRIX_USE_DOUBLE
template void covarianceMatrix(const DenseMatrix<double, Device::GPU>&,
                               DenseMatrix<double, Device::GPU>&,
                               cudaStream_t);

template void covarianceMatrixAsync(const DenseMatrix<double, Device::GPU>&,
                                    DenseMatrix<double, Device::GPU>&,
                                    GpuCovarianceWorkspace<double>&,
                                    cudaStream_t);
#endif

} // namespace plamatrix
