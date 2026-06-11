#pragma once

#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "plamatrix/dense/dense_matrix.h"

namespace plamatrix
{

/// 3D vector (stack-allocated parameter type).
template <typename Scalar>
struct Vec3
{
    Scalar x, y, z;
};

/// Build a 3x3 rotation matrix from axis-angle using the Rodrigues formula.
/// @tparam Scalar  Element type (float or double)
/// @tparam Dev     Device for the output matrix (CPU or GPU)
/// @param axis  Rotation axis (will be normalized internally)
/// @param angle  Rotation angle in radians
/// @return  3x3 rotation matrix on the specified device
/// @throws std::runtime_error if axis norm is too small
template <typename Scalar, Device Dev>
DenseMatrix<Scalar, Dev> rotationMatrix(const Vec3<Scalar>& axis, Scalar angle);

/// Build a 4x4 rigid transform matrix: [R|t; 0 0 0 1], where R is 3x3 and t is 3x1.
/// @tparam Scalar  Element type (float or double)
/// @tparam Dev     Device (CPU or GPU)
/// @param R  3x3 rotation matrix
/// @param t  Translation vector
/// @return  4x4 rigid transform matrix on the specified device
/// @throws std::runtime_error if R is not 3x3
template <typename Scalar, Device Dev>
DenseMatrix<Scalar, Dev> rigidTransform(const DenseMatrix<Scalar, Dev>& R, const Vec3<Scalar>& t);

/// GPU scratch buffers for asynchronous point-cloud covariance.
/// Keep this object alive until the CUDA stream passed to covarianceMatrixAsync has completed.
/// @tparam Scalar  Element type (float or double)
template <typename Scalar>
class GpuCovarianceWorkspace
{
public:
    using Accum = std::conditional_t<std::is_same_v<Scalar, float>, double, Scalar>;

    /// Construct an empty workspace.
    GpuCovarianceWorkspace() = default;

    /// Move constructor.
    /// @param other  Source workspace
    GpuCovarianceWorkspace(GpuCovarianceWorkspace&& other) noexcept = default;

    /// Move assignment.
    /// @param other  Source workspace
    /// @return  Reference to this workspace
    GpuCovarianceWorkspace& operator=(GpuCovarianceWorkspace&& other) noexcept = default;

    GpuCovarianceWorkspace(const GpuCovarianceWorkspace&) = delete;
    GpuCovarianceWorkspace& operator=(const GpuCovarianceWorkspace&) = delete;

    /// Ensure workspace capacity for a covariance launch with block_count partial blocks.
    /// @param block_count  Number of CUDA reduction blocks needed by the launch
    /// @throws std::invalid_argument if block_count is negative
    void reserveBlocks(Index block_count)
    {
        if (block_count < 0)
        {
            throw std::invalid_argument("GpuCovarianceWorkspace: block_count must be non-negative");
        }
        if (block_count <= _block_capacity)
        {
            return;
        }

        if (_block_capacity > 0)
        {
            _retired_partial_mean.emplace_back(std::move(_partial_mean));
            _retired_mean.emplace_back(std::move(_mean));
            _retired_partial_covariance.emplace_back(std::move(_partial_covariance));
        }

        _partial_mean = DenseMatrix<Accum, Device::GPU>(block_count, 3);
        _mean = DenseMatrix<Accum, Device::GPU>(1, 3);
        _partial_covariance = DenseMatrix<Accum, Device::GPU>(block_count, 6);
        _block_capacity = block_count;
    }

    /// @return  Number of partial reduction blocks currently reserved.
    Index blockCapacity() const { return _block_capacity; }

    DenseMatrix<Accum, Device::GPU>& partialMean() { return _partial_mean; }
    DenseMatrix<Accum, Device::GPU>& mean() { return _mean; }
    DenseMatrix<Accum, Device::GPU>& partialCovariance() { return _partial_covariance; }

private:
    Index _block_capacity = 0;
    DenseMatrix<Accum, Device::GPU> _partial_mean;
    DenseMatrix<Accum, Device::GPU> _mean;
    DenseMatrix<Accum, Device::GPU> _partial_covariance;
    std::vector<DenseMatrix<Accum, Device::GPU>> _retired_partial_mean;
    std::vector<DenseMatrix<Accum, Device::GPU>> _retired_mean;
    std::vector<DenseMatrix<Accum, Device::GPU>> _retired_partial_covariance;
};

/// Transform N points (Nx3) by a 4x4 rigid transform matrix. Returns Nx3.
/// For each point p, computes p' = R*p + t using the 4x4 T matrix.
/// @tparam Scalar  Element type (float or double)
/// @tparam Dev     Device (CPU or GPU)
/// @param T  4x4 rigid transformation matrix
/// @param points  Nx3 input point cloud
/// @return  Nx3 transformed point cloud on the specified device
/// @throws std::runtime_error if T is not 4x4 or points is not Nx3
template <typename Scalar, Device Dev>
DenseMatrix<Scalar, Dev> transformPoints(const DenseMatrix<Scalar, Dev>& T, const DenseMatrix<Scalar, Dev>& points);

/// Transform GPU points into an existing output matrix and synchronize the CUDA stream.
/// @tparam Scalar  Element type (float or double)
/// @param T       4x4 rigid transformation matrix on GPU
/// @param points  Nx3 input point cloud on GPU
/// @param output  Nx3 output point cloud on GPU
/// @param stream  Optional CUDA stream (defaults to default stream)
/// @throws std::runtime_error if dimensions do not match
template <typename Scalar>
void transformPoints(const DenseMatrix<Scalar, Device::GPU>& T,
                     const DenseMatrix<Scalar, Device::GPU>& points,
                     DenseMatrix<Scalar, Device::GPU>& output,
                     cudaStream_t stream = nullptr);

/// Asynchronously transform GPU points into a new output matrix.
/// Caller must synchronize the stream before reading the result from another stream or host.
/// @tparam Scalar  Element type (float or double)
/// @param T       4x4 rigid transformation matrix on GPU
/// @param points  Nx3 input point cloud on GPU
/// @param stream  Optional CUDA stream (defaults to default stream)
/// @return  Nx3 transformed point cloud on GPU
/// @throws std::runtime_error if dimensions do not match
template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> transformPointsAsync(const DenseMatrix<Scalar, Device::GPU>& T,
                                                      const DenseMatrix<Scalar, Device::GPU>& points,
                                                      cudaStream_t stream = nullptr);

/// Asynchronously transform GPU points into an existing output matrix.
/// Caller must synchronize the stream before reading output.
/// @tparam Scalar  Element type (float or double)
/// @param T       4x4 rigid transformation matrix on GPU
/// @param points  Nx3 input point cloud on GPU
/// @param output  Nx3 output point cloud on GPU
/// @param stream  Optional CUDA stream (defaults to default stream)
/// @throws std::runtime_error if dimensions do not match
template <typename Scalar>
void transformPointsAsync(const DenseMatrix<Scalar, Device::GPU>& T,
                          const DenseMatrix<Scalar, Device::GPU>& points,
                          DenseMatrix<Scalar, Device::GPU>& output,
                          cudaStream_t stream = nullptr);

/// Compute the 3x3 covariance matrix for an Nx3 point cloud.
/// Computes centroid, centers the points, and returns C = (1/N) * centered^T * centered.
/// @tparam Scalar  Element type (float or double)
/// @tparam Dev     Device (CPU or GPU)
/// @param points  Nx3 input point cloud
/// @return  3x3 covariance matrix (positive semi-definite) on the specified device
/// @throws std::runtime_error if points is not Nx3 or has fewer than 2 rows
template <typename Scalar, Device Dev>
DenseMatrix<Scalar, Dev> covarianceMatrix(const DenseMatrix<Scalar, Dev>& points);

/// Compute GPU covariance into an existing 3x3 output matrix and synchronize the CUDA stream.
/// @tparam Scalar  Element type (float or double)
/// @param points  Nx3 input point cloud on GPU
/// @param output  3x3 covariance output on GPU
/// @param stream  Optional CUDA stream (defaults to default stream)
/// @throws std::runtime_error if dimensions do not match or N < 2
template <typename Scalar>
void covarianceMatrix(const DenseMatrix<Scalar, Device::GPU>& points,
                      DenseMatrix<Scalar, Device::GPU>& output,
                      cudaStream_t stream = nullptr);

/// Asynchronously compute GPU covariance into an existing 3x3 output matrix.
/// The workspace must remain alive until stream completion.
/// @tparam Scalar  Element type (float or double)
/// @param points     Nx3 input point cloud on GPU
/// @param output     3x3 covariance output on GPU
/// @param workspace  Reusable GPU reduction buffers owned by the caller
/// @param stream     Optional CUDA stream (defaults to default stream)
/// @throws std::runtime_error if dimensions do not match or N < 2
template <typename Scalar>
void covarianceMatrixAsync(const DenseMatrix<Scalar, Device::GPU>& points,
                           DenseMatrix<Scalar, Device::GPU>& output,
                           GpuCovarianceWorkspace<Scalar>& workspace,
                           cudaStream_t stream = nullptr);

#ifdef PLAMATRIX_NO_CUDA
template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> transformPoints(const DenseMatrix<Scalar, Device::GPU>&,
                                                 const DenseMatrix<Scalar, Device::GPU>&)
{
    throw std::runtime_error("transformPoints: GPU point transform requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
void transformPoints(const DenseMatrix<Scalar, Device::GPU>&,
                     const DenseMatrix<Scalar, Device::GPU>&,
                     DenseMatrix<Scalar, Device::GPU>&,
                     cudaStream_t)
{
    throw std::runtime_error("transformPoints: GPU point transform requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> transformPointsAsync(const DenseMatrix<Scalar, Device::GPU>&,
                                                      const DenseMatrix<Scalar, Device::GPU>&,
                                                      cudaStream_t)
{
    throw std::runtime_error("transformPointsAsync: GPU point transform requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
void transformPointsAsync(const DenseMatrix<Scalar, Device::GPU>&,
                          const DenseMatrix<Scalar, Device::GPU>&,
                          DenseMatrix<Scalar, Device::GPU>&,
                          cudaStream_t)
{
    throw std::runtime_error("transformPointsAsync: GPU point transform requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> covarianceMatrix(const DenseMatrix<Scalar, Device::GPU>&)
{
    throw std::runtime_error("covarianceMatrix: GPU point-cloud covariance requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
void covarianceMatrix(const DenseMatrix<Scalar, Device::GPU>&,
                      DenseMatrix<Scalar, Device::GPU>&,
                      cudaStream_t)
{
    throw std::runtime_error("covarianceMatrix: GPU point-cloud covariance requires PLAMATRIX_WITH_CUDA=ON");
}

template <typename Scalar>
void covarianceMatrixAsync(const DenseMatrix<Scalar, Device::GPU>&,
                           DenseMatrix<Scalar, Device::GPU>&,
                           GpuCovarianceWorkspace<Scalar>&,
                           cudaStream_t)
{
    throw std::runtime_error("covarianceMatrixAsync: GPU point-cloud covariance requires PLAMATRIX_WITH_CUDA=ON");
}
#endif

} // namespace plamatrix
