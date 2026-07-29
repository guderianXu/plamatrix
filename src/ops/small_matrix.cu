#include <cmath>
#include <limits>
#include <stdexcept>

#include "small_matrix_detail.h"

namespace plamatrix
{
namespace
{

constexpr int kBlockSize = 256;
constexpr int kJacobiSweeps = 8;
constexpr Index kNoInvalidRow = static_cast<Index>(0x7fffffffffffffffLL);

#include "small_matrix_device.cuh"

template <typename Scalar>
__device__ Scalar absValue(Scalar value)
{
    return ::fabs(value);
}

template <typename Scalar>
__device__ Scalar maxValue(Scalar left, Scalar right)
{
    return left > right ? left : right;
}

template <typename Scalar>
__device__ Scalar& element(Scalar* matrix, int row, int col)
{
    return matrix[3 * row + col];
}

template <typename Scalar>
__device__ Scalar element(const Scalar* matrix, int row, int col)
{
    return matrix[3 * row + col];
}

template <typename Scalar>
__device__ Scalar dot(const Scalar* left, const Scalar* right)
{
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

template <typename Scalar>
__device__ bool normalize(Scalar* vector, Scalar minimum_norm)
{
    const Scalar norm = ::sqrt(dot(vector, vector));
    if (norm <= minimum_norm)
    {
        return false;
    }
    for (int component = 0; component < 3; ++component)
    {
        vector[component] /= norm;
    }
    return true;
}

template <typename Scalar>
__device__ void canonicalizeSign(Scalar* vector)
{
    int largest = 0;
    for (int component = 1; component < 3; ++component)
    {
        if (absValue(vector[component]) > absValue(vector[largest]))
        {
            largest = component;
        }
    }
    if (vector[largest] < Scalar(0))
    {
        for (int component = 0; component < 3; ++component)
        {
            vector[component] = -vector[component];
        }
    }
}

template <typename Scalar>
__device__ void applyJacobiRotation(Scalar* matrix, Scalar* eigenvectors, int p, int q)
{
    const Scalar apq = element(matrix, p, q);
    const Scalar app = element(matrix, p, p);
    const Scalar aqq = element(matrix, q, q);
    const Scalar rotation_scale =
        maxValue(absValue(app), maxValue(absValue(aqq), absValue(apq)));
    if (rotation_scale == Scalar(0) || apq == Scalar(0))
    {
        return;
    }

    const Scalar scaled_app = app / rotation_scale;
    const Scalar scaled_aqq = aqq / rotation_scale;
    const Scalar scaled_apq = apq / rotation_scale;
    const Scalar x = scaled_aqq - scaled_app;
    const Scalar y = Scalar(2) * scaled_apq;
    if (y == Scalar(0))
    {
        return;
    }
    const Scalar tangent_magnitude = absValue(y) / (absValue(x) + ::hypot(x, y));
    Scalar tangent = ::copysign(tangent_magnitude, y);
    if (x < Scalar(0))
    {
        tangent = -tangent;
    }
    const Scalar cosine = Scalar(1) / ::sqrt(Scalar(1) + tangent * tangent);
    const Scalar sine = tangent * cosine;

    for (int k = 0; k < 3; ++k)
    {
        if (k == p || k == q)
        {
            continue;
        }
        const Scalar akp = element(matrix, k, p);
        const Scalar akq = element(matrix, k, q);
        element(matrix, k, p) = cosine * akp - sine * akq;
        element(matrix, p, k) = element(matrix, k, p);
        element(matrix, k, q) = sine * akp + cosine * akq;
        element(matrix, q, k) = element(matrix, k, q);
    }

    element(matrix, p, p) = ::fma(-tangent, apq, app);
    element(matrix, q, q) = ::fma(tangent, apq, aqq);
    element(matrix, p, q) = Scalar(0);
    element(matrix, q, p) = Scalar(0);

    for (int row = 0; row < 3; ++row)
    {
        const Scalar vip = element(eigenvectors, row, p);
        const Scalar viq = element(eigenvectors, row, q);
        element(eigenvectors, row, p) = cosine * vip - sine * viq;
        element(eigenvectors, row, q) = sine * vip + cosine * viq;
    }
}

template <typename Scalar>
__device__ bool eigenvaluesRepeat(Scalar left, Scalar right)
{
    const Scalar scale = maxValue(Scalar(1), maxValue(absValue(left), absValue(right)));
    return absValue(left - right) <=
           Scalar(256) * machineEpsilon<Scalar>() * scale;
}

template <typename Scalar>
__device__ bool rebuildRepeatedGroup(
    Scalar* eigenvectors, int begin, int end, bool force_failure)
{
    if (force_failure)
    {
        return false;
    }
    Scalar projector[9] = {};
    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            for (int group_col = begin; group_col < end; ++group_col)
            {
                element(projector, row, col) += element(eigenvectors, row, group_col) *
                                                element(eigenvectors, col, group_col);
            }
        }
    }

    Scalar basis[9] = {};
    int basis_count = 0;
    const Scalar minimum_norm = Scalar(64) * machineEpsilon<Scalar>();
    for (int axis = 0; axis < 3 && basis_count < end - begin; ++axis)
    {
        Scalar candidate[3] = {
            element(projector, 0, axis),
            element(projector, 1, axis),
            element(projector, 2, axis)
        };
        for (int pass = 0; pass < 2; ++pass)
        {
            for (int previous = 0; previous < basis_count; ++previous)
            {
                const Scalar projection = dot(candidate, basis + 3 * previous);
                for (int component = 0; component < 3; ++component)
                {
                    candidate[component] -= projection * basis[3 * previous + component];
                }
            }
        }
        if (normalize(candidate, minimum_norm))
        {
            for (int component = 0; component < 3; ++component)
            {
                basis[3 * basis_count + component] = candidate[component];
            }
            ++basis_count;
        }
    }

    if (basis_count == end - begin)
    {
        for (int group_col = begin; group_col < end; ++group_col)
        {
            for (int row = 0; row < 3; ++row)
            {
                element(eigenvectors, row, group_col) = basis[3 * (group_col - begin) + row];
            }
        }
        return true;
    }
    return false;
}

template <typename Scalar>
__device__ bool sortAndCanonicalize(
    Scalar* eigenvectors, Scalar* eigenvalues, bool force_basis_failure)
{
    for (int current = 1; current < 3; ++current)
    {
        const Scalar value = eigenvalues[current];
        Scalar vector[3] = {
            element(eigenvectors, 0, current),
            element(eigenvectors, 1, current),
            element(eigenvectors, 2, current)
        };
        int destination = current;
        while (destination > 0 && value < eigenvalues[destination - 1])
        {
            eigenvalues[destination] = eigenvalues[destination - 1];
            for (int row = 0; row < 3; ++row)
            {
                element(eigenvectors, row, destination) =
                    element(eigenvectors, row, destination - 1);
            }
            --destination;
        }
        eigenvalues[destination] = value;
        for (int row = 0; row < 3; ++row)
        {
            element(eigenvectors, row, destination) = vector[row];
        }
    }

    for (int begin = 0; begin < 3;)
    {
        int end = begin + 1;
        while (end < 3 && eigenvaluesRepeat(eigenvalues[begin], eigenvalues[end]))
        {
            ++end;
        }
        if (end - begin > 1)
        {
            if (!rebuildRepeatedGroup(eigenvectors, begin, end, force_basis_failure))
            {
                return false;
            }
        }
        begin = end;
    }

    for (int col = 0; col < 3; ++col)
    {
        Scalar vector[3] = {
            element(eigenvectors, 0, col),
            element(eigenvectors, 1, col),
            element(eigenvectors, 2, col)
        };
        static_cast<void>(normalize(vector, Scalar(0)));
        canonicalizeSign(vector);
        for (int row = 0; row < 3; ++row)
        {
            element(eigenvectors, row, col) = vector[row];
        }
    }
    return true;
}

template <typename Scalar>
__global__ void symmetricEigh3x3Kernel(
    const Scalar* compact,
    Index rows,
    Scalar* values,
    Scalar* vectors,
    small_matrix_detail::DeviceStatus* status,
    Index forced_basis_failure_row)
{
    const Index row = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows)
    {
        return;
    }

    Scalar packed[6];
    bool finite = true;
    for (int col = 0; col < 6; ++col)
    {
        packed[col] = compact[row + static_cast<Index>(col) * rows];
        finite = finite && ::isfinite(packed[col]);
    }
    if (!finite)
    {
        atomicMin(reinterpret_cast<unsigned long long*>(&status->nonFiniteRow),
                  static_cast<unsigned long long>(row));
        zeroOutputRow(row, rows, values, vectors);
        return;
    }

    Scalar matrix[9] = {
        packed[0], packed[1], packed[2],
        packed[1], packed[3], packed[4],
        packed[2], packed[4], packed[5]
    };
    Scalar eigenvectors[9] = {
        Scalar(1), Scalar(0), Scalar(0),
        Scalar(0), Scalar(1), Scalar(0),
        Scalar(0), Scalar(0), Scalar(1)
    };
    for (int sweep = 0; sweep < kJacobiSweeps; ++sweep)
    {
        applyJacobiRotation(matrix, eigenvectors, 0, 1);
        applyJacobiRotation(matrix, eigenvectors, 0, 2);
        applyJacobiRotation(matrix, eigenvectors, 1, 2);
    }
    Scalar eigenvalues[3] = {
        element(matrix, 0, 0), element(matrix, 1, 1), element(matrix, 2, 2)
    };
    if (!sortAndCanonicalize(eigenvectors, eigenvalues, row == forced_basis_failure_row))
    {
        atomicMin(reinterpret_cast<unsigned long long*>(&status->basisFailureRow),
                  static_cast<unsigned long long>(row));
        zeroOutputRow(row, rows, values, vectors);
        return;
    }
    for (int col = 0; col < 3; ++col)
    {
        values[row + static_cast<Index>(col) * rows] = eigenvalues[col];
        for (int component = 0; component < 3; ++component)
        {
            const int packed_col = 3 * col + component;
            vectors[row + static_cast<Index>(packed_col) * rows] =
                element(eigenvectors, component, col);
        }
    }
}

} // namespace

template <typename Scalar>
void small_matrix_detail::launchSymmetricEigh3x3(
    const DenseMatrix<Scalar, Device::GPU>& input,
    DenseMatrix<Scalar, Device::GPU>& values,
    DenseMatrix<Scalar, Device::GPU>& vectors,
    SymmetricEigh3x3Workspace& workspace,
    cudaStream_t stream)
{
    if (input.rows() == 0)
    {
        return;
    }
    const Index blocks = input.rows() / kBlockSize + (input.rows() % kBlockSize != 0 ? 1 : 0);
    if (blocks > static_cast<Index>(std::numeric_limits<int>::max()))
    {
        throw std::overflow_error("symmetricEigh3x3BatchedAsync: CUDA grid range exceeded");
    }

    workspace.reserveBytesAsync(sizeof(DeviceStatus), stream);
    auto* status = static_cast<DeviceStatus*>(workspace.data());
    if (SymmetricEigh3x3WorkspaceAccess::beginStatusBatch(workspace))
    {
        initializeStatusKernel<<<1, 1, 0, stream>>>(status);
        PLAMATRIX_CHECK_CUDA(cudaGetLastError());
    }
    Index forced_failure_row = -1;
#ifdef PLAMATRIX_SMALL_MATRIX_TEST_HOOKS
    forced_failure_row = forcedBasisFailureRow();
#endif
    symmetricEigh3x3Kernel<<<static_cast<unsigned int>(blocks), kBlockSize, 0, stream>>>(
        input.data(), input.rows(), values.data(), vectors.data(), status,
        forced_failure_row);
    PLAMATRIX_CHECK_CUDA(cudaGetLastError());
}

#define PLAMATRIX_INSTANTIATE_SMALL_MATRIX_LAUNCH(Scalar)                               \
    template void small_matrix_detail::launchSymmetricEigh3x3(                          \
        const DenseMatrix<Scalar, Device::GPU>&,                                        \
        DenseMatrix<Scalar, Device::GPU>&, DenseMatrix<Scalar, Device::GPU>&,            \
        SymmetricEigh3x3Workspace&, cudaStream_t)

#ifdef PLAMATRIX_USE_FLOAT
PLAMATRIX_INSTANTIATE_SMALL_MATRIX_LAUNCH(float);
#endif
#ifdef PLAMATRIX_USE_DOUBLE
PLAMATRIX_INSTANTIATE_SMALL_MATRIX_LAUNCH(double);
#endif

#undef PLAMATRIX_INSTANTIATE_SMALL_MATRIX_LAUNCH

} // namespace plamatrix
