#include <cmath>
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
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
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

    Index N = points.rows();
    if (N == 0)
    {
        return DenseMatrix<Scalar, Device::GPU>(0, 3);
    }

    DenseMatrix<Scalar, Device::GPU> result(N, 3);

    int block_size = 256;
    int grid_size = static_cast<int>((N + block_size - 1) / block_size);

    transformPointsKernel<<<grid_size, block_size>>>(T.data(), points.data(), result.data(), N);
    PLAMATRIX_CHECK_CUDA(cudaGetLastError());

    return result;
}

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> covarianceMatrixGpuImpl(const DenseMatrix<Scalar, Device::GPU>& points)
{
    // Copy points to CPU, compute covariance on CPU, transfer result back to GPU
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

    std::size_t points_size = static_cast<std::size_t>(N * 3) * sizeof(Scalar);
    Scalar* points_cpu = new Scalar[N * 3];
    PLAMATRIX_CHECK_CUDA(cudaMemcpy(points_cpu, points.data(), points_size, cudaMemcpyDeviceToHost));

    // Compute centroid
    Scalar cx = static_cast<Scalar>(0);
    Scalar cy = static_cast<Scalar>(0);
    Scalar cz = static_cast<Scalar>(0);
    for (Index i = 0; i < N; ++i)
    {
        cx += points_cpu[i + 0 * N];
        cy += points_cpu[i + 1 * N];
        cz += points_cpu[i + 2 * N];
    }
    Scalar invN = static_cast<Scalar>(1) / static_cast<Scalar>(N);
    cx *= invN;
    cy *= invN;
    cz *= invN;

    // Compute covariance
    Scalar C_cpu[9] = {0};
    for (Index k = 0; k < N; ++k)
    {
        Scalar dx = points_cpu[k + 0 * N] - cx;
        Scalar dy = points_cpu[k + 1 * N] - cy;
        Scalar dz = points_cpu[k + 2 * N] - cz;

        C_cpu[0] += invN * dx * dx;
        C_cpu[1] += invN * dx * dy;
        C_cpu[2] += invN * dx * dz;
        C_cpu[3] += invN * dy * dx;
        C_cpu[4] += invN * dy * dy;
        C_cpu[5] += invN * dy * dz;
        C_cpu[6] += invN * dz * dx;
        C_cpu[7] += invN * dz * dy;
        C_cpu[8] += invN * dz * dz;
    }

    delete[] points_cpu;

    DenseMatrix<Scalar, Device::GPU> C_gpu(3, 3);
    PLAMATRIX_CHECK_CUDA(
        cudaMemcpy(C_gpu.data(), C_cpu, static_cast<std::size_t>(9) * sizeof(Scalar), cudaMemcpyHostToDevice));
    return C_gpu;
}

// ---- Explicit specializations for GPU ----

template <>
DenseMatrix<float, Device::GPU> rotationMatrix<float, Device::GPU>(const Vec3<float>& axis, float angle)
{
    return rotationMatrixGpuImpl(axis, angle);
}

template <>
DenseMatrix<double, Device::GPU> rotationMatrix<double, Device::GPU>(const Vec3<double>& axis, double angle)
{
    return rotationMatrixGpuImpl(axis, angle);
}

template <>
DenseMatrix<float, Device::GPU> rigidTransform<float, Device::GPU>(const DenseMatrix<float, Device::GPU>& R,
                                                                      const Vec3<float>& t)
{
    return rigidTransformGpuImpl(R, t);
}

template <>
DenseMatrix<double, Device::GPU> rigidTransform<double, Device::GPU>(const DenseMatrix<double, Device::GPU>& R,
                                                                       const Vec3<double>& t)
{
    return rigidTransformGpuImpl(R, t);
}

template <>
DenseMatrix<float, Device::GPU> transformPoints<float, Device::GPU>(const DenseMatrix<float, Device::GPU>& T,
                                                                      const DenseMatrix<float, Device::GPU>& points)
{
    return transformPointsGpuImpl(T, points);
}

template <>
DenseMatrix<double, Device::GPU> transformPoints<double, Device::GPU>(const DenseMatrix<double, Device::GPU>& T,
                                                                        const DenseMatrix<double, Device::GPU>& points)
{
    return transformPointsGpuImpl(T, points);
}

template <>
DenseMatrix<float, Device::GPU> covarianceMatrix<float, Device::GPU>(const DenseMatrix<float, Device::GPU>& points)
{
    return covarianceMatrixGpuImpl(points);
}

template <>
DenseMatrix<double, Device::GPU> covarianceMatrix<double, Device::GPU>(const DenseMatrix<double, Device::GPU>& points)
{
    return covarianceMatrixGpuImpl(points);
}

} // namespace plamatrix
