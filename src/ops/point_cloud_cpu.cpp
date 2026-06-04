#include <cmath>
#include <sstream>
#include <stdexcept>
#include <type_traits>

#include "plamatrix/ops/point_cloud.h"

namespace plamatrix
{

// ---- Helper: rotation matrix (Rodrigues formula) ----

template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> rotationMatrixCpu(const Vec3<Scalar>& axis, Scalar angle)
{
    // Normalize axis
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

    DenseMatrix<Scalar, Device::CPU> R(3, 3);

    // Rodrigues formula: R = I + sin(theta)*K + (1-cos(theta))*K^2
    // where K is the cross-product matrix of the normalized axis
    R(0, 0) = c + x * x * one_minus_c;
    R(1, 0) = z * s + x * y * one_minus_c;
    R(2, 0) = -y * s + x * z * one_minus_c;

    R(0, 1) = -z * s + y * x * one_minus_c;
    R(1, 1) = c + y * y * one_minus_c;
    R(2, 1) = x * s + y * z * one_minus_c;

    R(0, 2) = y * s + z * x * one_minus_c;
    R(1, 2) = -x * s + z * y * one_minus_c;
    R(2, 2) = c + z * z * one_minus_c;

    return R;
}

// ---- Helper: rigid transform ----

template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> rigidTransformCpu(const DenseMatrix<Scalar, Device::CPU>& R,
                                                     const Vec3<Scalar>& t)
{
    if (R.rows() != 3 || R.cols() != 3)
    {
        std::ostringstream oss;
        oss << "rigidTransform: R must be 3x3, got " << R.rows() << "x" << R.cols();
        throw std::runtime_error(oss.str());
    }

    const Scalar* r = R.data();

    DenseMatrix<Scalar, Device::CPU> T(4, 4);
    Scalar* td = T.data();

    // Column-major layout: T(row, col) = td[row + col * 4]

    // Copy R into top-left 3x3
    for (Index col = 0; col < 3; ++col)
    {
        for (Index row = 0; row < 3; ++row)
        {
            td[row + col * 4] = r[row + col * 3];
        }
    }

    // Set translation as last column (first 3 rows)
    td[0 + 3 * 4] = t.x;
    td[1 + 3 * 4] = t.y;
    td[2 + 3 * 4] = t.z;

    // Bottom row: [0, 0, 0, 1]
    // Already zero from constructor, just set T(3,3) = 1
    td[3 + 3 * 4] = static_cast<Scalar>(1);

    return T;
}

// ---- Helper: transform points ----

template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> transformPointsCpu(const DenseMatrix<Scalar, Device::CPU>& T,
                                                      const DenseMatrix<Scalar, Device::CPU>& points)
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
    const Scalar* t_data = T.data();
    const Scalar* p_data = points.data();

    DenseMatrix<Scalar, Device::CPU> result(N, 3);
    Scalar* r_data = result.data();

    for (Index i = 0; i < N; ++i)
    {
        Scalar px = p_data[i + 0 * N];
        Scalar py = p_data[i + 1 * N];
        Scalar pz = p_data[i + 2 * N];

        // p' = R * p + t, where R is top-left 3x3, t is last column first 3 rows
        r_data[i + 0 * N] = t_data[0] * px + t_data[4] * py + t_data[8] * pz + t_data[12];
        r_data[i + 1 * N] = t_data[1] * px + t_data[5] * py + t_data[9] * pz + t_data[13];
        r_data[i + 2 * N] = t_data[2] * px + t_data[6] * py + t_data[10] * pz + t_data[14];
    }

    return result;
}

// ---- Helper: covariance matrix ----

template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> covarianceMatrixCpu(const DenseMatrix<Scalar, Device::CPU>& points)
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

    const Scalar* p = points.data();
    using Accum = std::conditional_t<std::is_same_v<Scalar, float>, double, Scalar>;

    // Compute centroid
    Accum cx = static_cast<Accum>(0);
    Accum cy = static_cast<Accum>(0);
    Accum cz = static_cast<Accum>(0);
    for (Index i = 0; i < N; ++i)
    {
        cx += static_cast<Accum>(p[i + 0 * N]);
        cy += static_cast<Accum>(p[i + 1 * N]);
        cz += static_cast<Accum>(p[i + 2 * N]);
    }
    Accum invN = static_cast<Accum>(1) / static_cast<Accum>(N);
    cx *= invN;
    cy *= invN;
    cz *= invN;

    // Compute covariance: C = (1/N) * centered^T * centered
    DenseMatrix<Scalar, Device::CPU> C(3, 3);
    Scalar* c = C.data();

    Accum accum[9] = {};
    for (Index k = 0; k < N; ++k)
    {
        Accum dx = static_cast<Accum>(p[k + 0 * N]) - cx;
        Accum dy = static_cast<Accum>(p[k + 1 * N]) - cy;
        Accum dz = static_cast<Accum>(p[k + 2 * N]) - cz;

        accum[0] += invN * dx * dx; // C(0,0)
        accum[1] += invN * dx * dy; // C(1,0)
        accum[2] += invN * dx * dz; // C(2,0)
        accum[3] += invN * dy * dx; // C(0,1)
        accum[4] += invN * dy * dy; // C(1,1)
        accum[5] += invN * dy * dz; // C(2,1)
        accum[6] += invN * dz * dx; // C(0,2)
        accum[7] += invN * dz * dy; // C(1,2)
        accum[8] += invN * dz * dz; // C(2,2)
    }

    for (int i = 0; i < 9; ++i)
    {
        c[i] = static_cast<Scalar>(accum[i]);
    }

    return C;
}

// ---- Generic template definitions with device dispatch ----

template <typename Scalar, Device Dev>
DenseMatrix<Scalar, Dev> rotationMatrix(const Vec3<Scalar>& axis, Scalar angle)
{
    if constexpr (Dev == Device::CPU)
    {
        return rotationMatrixCpu(axis, angle);
    }
    else
    {
        static_assert(Dev == Device::CPU, "GPU rotationMatrix requires CUDA backend (point_cloud.cu)");
        return DenseMatrix<Scalar, Dev>(0, 0);
    }
}

template <typename Scalar, Device Dev>
DenseMatrix<Scalar, Dev> rigidTransform(const DenseMatrix<Scalar, Dev>& R, const Vec3<Scalar>& t)
{
    if constexpr (Dev == Device::CPU)
    {
        return rigidTransformCpu(R, t);
    }
    else
    {
        static_assert(Dev == Device::CPU, "GPU rigidTransform requires CUDA backend (point_cloud.cu)");
        return DenseMatrix<Scalar, Dev>(0, 0);
    }
}

template <typename Scalar, Device Dev>
DenseMatrix<Scalar, Dev> transformPoints(const DenseMatrix<Scalar, Dev>& T, const DenseMatrix<Scalar, Dev>& points)
{
    if constexpr (Dev == Device::CPU)
    {
        return transformPointsCpu(T, points);
    }
    else
    {
        static_assert(Dev == Device::CPU, "GPU transformPoints requires CUDA backend (point_cloud.cu)");
        return DenseMatrix<Scalar, Dev>(0, 0);
    }
}

template <typename Scalar, Device Dev>
DenseMatrix<Scalar, Dev> covarianceMatrix(const DenseMatrix<Scalar, Dev>& points)
{
    if constexpr (Dev == Device::CPU)
    {
        return covarianceMatrixCpu(points);
    }
    else
    {
        static_assert(Dev == Device::CPU, "GPU covarianceMatrix requires CUDA backend (point_cloud.cu)");
        return DenseMatrix<Scalar, Dev>(0, 0);
    }
}

// ---- Explicit template instantiations for CPU ----

template DenseMatrix<float, Device::CPU> rotationMatrix(const Vec3<float>&, float);
template DenseMatrix<double, Device::CPU> rotationMatrix(const Vec3<double>&, double);

template DenseMatrix<float, Device::CPU> rigidTransform(const DenseMatrix<float, Device::CPU>&, const Vec3<float>&);
template DenseMatrix<double, Device::CPU> rigidTransform(const DenseMatrix<double, Device::CPU>&, const Vec3<double>&);

template DenseMatrix<float, Device::CPU> transformPoints(const DenseMatrix<float, Device::CPU>&,
                                                           const DenseMatrix<float, Device::CPU>&);
template DenseMatrix<double, Device::CPU> transformPoints(const DenseMatrix<double, Device::CPU>&,
                                                            const DenseMatrix<double, Device::CPU>&);

template DenseMatrix<float, Device::CPU> covarianceMatrix(const DenseMatrix<float, Device::CPU>&);
template DenseMatrix<double, Device::CPU> covarianceMatrix(const DenseMatrix<double, Device::CPU>&);

} // namespace plamatrix
