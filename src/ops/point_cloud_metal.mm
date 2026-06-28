#include <sstream>
#include <stdexcept>

#include "plamatrix/core/gpu_runtime.h"
#include "plamatrix/core/metal_context.h"
#include "plamatrix/ops/point_cloud.h"

namespace plamatrix
{

namespace
{

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

Index covarianceBlockCount(Index point_count)
{
    constexpr Index block_size = 256;
    return (point_count + block_size - 1) / block_size;
}

template <typename Scalar>
void covarianceMatrixFallback(const DenseMatrix<Scalar, Device::GPU>& points,
                              DenseMatrix<Scalar, Device::GPU>& output,
                              GpuCovarianceWorkspace<Scalar>& workspace,
                              cudaStream_t stream)
{
    checkCovarianceInputs(points);
    checkCovarianceOutput(output);
    workspace.reserveBlocks(covarianceBlockCount(points.rows()));

    auto cpu = covarianceMatrix<Scalar, Device::CPU>(points.toCpu());
    detail::gpuCopyHostToDevice(output.data(),
                                cpu.data(),
                                static_cast<std::size_t>(cpu.size()) * sizeof(Scalar),
                                stream);
}

} // namespace

#ifdef PLAMATRIX_USE_FLOAT
template <>
DenseMatrix<float, Device::GPU> rotationMatrix<float, Device::GPU>(const Vec3<float>& axis, float angle)
{
    return rotationMatrix<float, Device::CPU>(axis, angle).toGpu();
}

template <>
DenseMatrix<float, Device::GPU> rigidTransform<float, Device::GPU>(const DenseMatrix<float, Device::GPU>& R,
                                                                   const Vec3<float>& t)
{
    return rigidTransform<float, Device::CPU>(R.toCpu(), t).toGpu();
}

template <>
DenseMatrix<float, Device::GPU> transformPoints<float, Device::GPU>(
    const DenseMatrix<float, Device::GPU>& T,
    const DenseMatrix<float, Device::GPU>& points)
{
    auto result = transformPointsAsync(T, points);
    return result;
}

template <>
DenseMatrix<float, Device::GPU> covarianceMatrix<float, Device::GPU>(
    const DenseMatrix<float, Device::GPU>& points)
{
    DenseMatrix<float, Device::GPU> output(3, 3);
    GpuCovarianceWorkspace<float> workspace;
    covarianceMatrixFallback(points, output, workspace, nullptr);
    return output;
}

template <>
void transformPointsAsync(const DenseMatrix<float, Device::GPU>& T,
                          const DenseMatrix<float, Device::GPU>& points,
                          DenseMatrix<float, Device::GPU>& output,
                          cudaStream_t)
{
    checkTransformInputs(T, points);
    checkTransformOutput(output, points.rows());
    detail::metalTransformPointsFloat(T.data(), points.data(), output.data(), points.rows());
}
#endif

#ifdef PLAMATRIX_USE_DOUBLE
template <>
DenseMatrix<double, Device::GPU> rotationMatrix<double, Device::GPU>(const Vec3<double>& axis, double angle)
{
    return rotationMatrix<double, Device::CPU>(axis, angle).toGpu();
}

template <>
DenseMatrix<double, Device::GPU> rigidTransform<double, Device::GPU>(const DenseMatrix<double, Device::GPU>& R,
                                                                     const Vec3<double>& t)
{
    return rigidTransform<double, Device::CPU>(R.toCpu(), t).toGpu();
}

template <>
DenseMatrix<double, Device::GPU> transformPoints<double, Device::GPU>(
    const DenseMatrix<double, Device::GPU>& T,
    const DenseMatrix<double, Device::GPU>& points)
{
    auto result = transformPointsAsync(T, points);
    return result;
}

template <>
DenseMatrix<double, Device::GPU> covarianceMatrix<double, Device::GPU>(
    const DenseMatrix<double, Device::GPU>& points)
{
    DenseMatrix<double, Device::GPU> output(3, 3);
    GpuCovarianceWorkspace<double> workspace;
    covarianceMatrixFallback(points, output, workspace, nullptr);
    return output;
}

template <>
void transformPointsAsync(const DenseMatrix<double, Device::GPU>& T,
                          const DenseMatrix<double, Device::GPU>& points,
                          DenseMatrix<double, Device::GPU>& output,
                          cudaStream_t)
{
    checkTransformInputs(T, points);
    checkTransformOutput(output, points.rows());
    detail::metalTransformPointsDoubleFallback(T.data(), points.data(), output.data(), points.rows());
}
#endif

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> transformPointsAsync(const DenseMatrix<Scalar, Device::GPU>& T,
                                                      const DenseMatrix<Scalar, Device::GPU>& points,
                                                      cudaStream_t stream)
{
    checkTransformInputs(T, points);
    DenseMatrix<Scalar, Device::GPU> output(points.rows(), 3);
    transformPointsAsync(T, points, output, stream);
    return output;
}

template <typename Scalar>
void transformPoints(const DenseMatrix<Scalar, Device::GPU>& T,
                     const DenseMatrix<Scalar, Device::GPU>& points,
                     DenseMatrix<Scalar, Device::GPU>& output,
                     cudaStream_t stream)
{
    transformPointsAsync(T, points, output, stream);
}

template <typename Scalar>
void covarianceMatrix(const DenseMatrix<Scalar, Device::GPU>& points,
                      DenseMatrix<Scalar, Device::GPU>& output,
                      cudaStream_t stream)
{
    GpuCovarianceWorkspace<Scalar> workspace;
    covarianceMatrixFallback(points, output, workspace, stream);
}

template <typename Scalar>
void covarianceMatrixAsync(const DenseMatrix<Scalar, Device::GPU>& points,
                           DenseMatrix<Scalar, Device::GPU>& output,
                           GpuCovarianceWorkspace<Scalar>& workspace,
                           cudaStream_t stream)
{
    covarianceMatrixFallback(points, output, workspace, stream);
}

#ifdef PLAMATRIX_USE_FLOAT
template DenseMatrix<float, Device::GPU> transformPointsAsync(const DenseMatrix<float, Device::GPU>&,
                                                              const DenseMatrix<float, Device::GPU>&,
                                                              cudaStream_t);

template void transformPoints(const DenseMatrix<float, Device::GPU>&,
                              const DenseMatrix<float, Device::GPU>&,
                              DenseMatrix<float, Device::GPU>&,
                              cudaStream_t);

template void covarianceMatrix(const DenseMatrix<float, Device::GPU>&,
                               DenseMatrix<float, Device::GPU>&,
                               cudaStream_t);

template void covarianceMatrixAsync(const DenseMatrix<float, Device::GPU>&,
                                    DenseMatrix<float, Device::GPU>&,
                                    GpuCovarianceWorkspace<float>&,
                                    cudaStream_t);
#endif

#ifdef PLAMATRIX_USE_DOUBLE
template DenseMatrix<double, Device::GPU> transformPointsAsync(const DenseMatrix<double, Device::GPU>&,
                                                               const DenseMatrix<double, Device::GPU>&,
                                                               cudaStream_t);

template void transformPoints(const DenseMatrix<double, Device::GPU>&,
                              const DenseMatrix<double, Device::GPU>&,
                              DenseMatrix<double, Device::GPU>&,
                              cudaStream_t);

template void covarianceMatrix(const DenseMatrix<double, Device::GPU>&,
                               DenseMatrix<double, Device::GPU>&,
                               cudaStream_t);

template void covarianceMatrixAsync(const DenseMatrix<double, Device::GPU>&,
                                    DenseMatrix<double, Device::GPU>&,
                                    GpuCovarianceWorkspace<double>&,
                                    cudaStream_t);
#endif

} // namespace plamatrix
