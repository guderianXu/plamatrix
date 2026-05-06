#pragma once

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
template <typename Scalar, Device Dev>
DenseMatrix<Scalar, Dev> rigidTransform(const DenseMatrix<Scalar, Dev>& R, const Vec3<Scalar>& t);

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

/// Compute the 3x3 covariance matrix for an Nx3 point cloud.
/// Computes centroid, centers the points, and returns C = (1/N) * centered^T * centered.
/// @tparam Scalar  Element type (float or double)
/// @tparam Dev     Device (CPU or GPU)
/// @param points  Nx3 input point cloud
/// @return  3x3 covariance matrix (positive semi-definite) on the specified device
template <typename Scalar, Device Dev>
DenseMatrix<Scalar, Dev> covarianceMatrix(const DenseMatrix<Scalar, Dev>& points);

} // namespace plamatrix
