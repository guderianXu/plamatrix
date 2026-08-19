#pragma once

#include "iterative_solver_cuda_detail.h"

namespace plamatrix
{
namespace iterative_solver_detail
{

template <typename Scalar>
struct Blas;

template <>
struct Blas<float>
{
    static cublasStatus_t dot(cublasHandle_t handle, int size, const float* left,
                              const float* right, float* result)
    {
        return cublasSdot(handle, size, left, 1, right, 1, result);
    }
    static cublasStatus_t copy(cublasHandle_t handle, int size, const float* input,
                               float* output)
    {
        return cublasScopy(handle, size, input, 1, output, 1);
    }
    static cublasStatus_t axpy(cublasHandle_t handle, int size, const float* alpha,
                               const float* input, float* output)
    {
        return cublasSaxpy(handle, size, alpha, input, 1, output, 1);
    }
    static cublasStatus_t scal(cublasHandle_t handle, int size, const float* alpha,
                               float* values)
    {
        return cublasSscal(handle, size, alpha, values, 1);
    }
};

template <>
struct Blas<double>
{
    static cublasStatus_t dot(cublasHandle_t handle, int size, const double* left,
                              const double* right, double* result)
    {
        return cublasDdot(handle, size, left, 1, right, 1, result);
    }
    static cublasStatus_t copy(cublasHandle_t handle, int size, const double* input,
                               double* output)
    {
        return cublasDcopy(handle, size, input, 1, output, 1);
    }
    static cublasStatus_t axpy(cublasHandle_t handle, int size, const double* alpha,
                               const double* input, double* output)
    {
        return cublasDaxpy(handle, size, alpha, input, 1, output, 1);
    }
    static cublasStatus_t scal(cublasHandle_t handle, int size, const double* alpha,
                               double* values)
    {
        return cublasDscal(handle, size, alpha, values, 1);
    }
};

template <typename Scalar>
__device__ Scalar quietNan();

template <>
__device__ float quietNan<float>()
{
    return __int_as_float(0x7fffffff);
}

template <>
__device__ double quietNan<double>()
{
    return __longlong_as_double(0x7fffffffffffffffULL);
}

template <typename Scalar>
__device__ Scalar scalarEpsilon();

template <>
__device__ float scalarEpsilon<float>()
{
    return 1.1920928955078125e-7F;
}

template <>
__device__ double scalarEpsilon<double>()
{
    return 2.2204460492503131e-16;
}

template <typename Scalar>
__global__ void subtractKernel(const Scalar* rhs, const Scalar* product,
                               Scalar* residual, Index size)
{
    const Index index = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < size)
    {
        residual[index] = rhs[index] - product[index];
    }
}

template <typename Scalar>
__global__ void jacobiKernel(const Index* row_offsets, const Index* columns,
                             const Scalar* values, Scalar* inverse, Index rows)
{
    const Index row = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows)
    {
        return;
    }
    Scalar diagonal = Scalar{0};
    Scalar row_scale = Scalar{0};
    bool found = false;
    for (Index position = row_offsets[row]; position < row_offsets[row + 1]; ++position)
    {
        const Scalar value = values[position];
        const Scalar magnitude = value < Scalar{0} ? -value : value;
        row_scale = row_scale < magnitude ? magnitude : row_scale;
        if (columns[position] == row)
        {
            diagonal += value;
            found = true;
        }
    }
    const Scalar threshold = scalarEpsilon<Scalar>() * row_scale * Scalar{16};
    const Scalar reciprocal = Scalar{1} / diagonal;
    inverse[row] = found && isfinite(diagonal) && diagonal > threshold && isfinite(reciprocal)
        ? reciprocal : quietNan<Scalar>();
}

template <typename Scalar>
__global__ void applyJacobiKernel(const Scalar* inverse, const Scalar* residual,
                                  Scalar* transformed, Index size)
{
    const Index index = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < size)
    {
        transformed[index] = inverse[index] * residual[index];
    }
}

template <typename Scalar>
__global__ void applyBlockJacobiKernel(const Scalar* inverse_blocks,
                                       const Scalar* residual,
                                       Scalar* transformed,
                                       Index size,
                                       Index block_size)
{
    const Index row = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= size)
    {
        return;
    }
    const Index block = row / block_size;
    const Index local_row = row - block * block_size;
    const Index block_offset = block * block_size * block_size;
    const Index residual_offset = block * block_size;
    Scalar value = Scalar{0};
    for (Index column = 0; column < block_size; ++column)
    {
        value += inverse_blocks[block_offset + local_row * block_size + column]
               * residual[residual_offset + column];
    }
    transformed[row] = value;
}

template <typename Scalar>
__global__ void alphaKernel(Scalar* scalars)
{
    if (scalars[kBreakdown] != Scalar{0})
    {
        scalars[kAlpha] = Scalar{0};
        scalars[kNegativeAlpha] = Scalar{0};
        return;
    }
    const Scalar rho = scalars[kRho];
    const Scalar denominator = scalars[kDenominator];
    Scalar alpha = quietNan<Scalar>();
    if (rho == Scalar{0} && denominator == Scalar{0})
    {
        alpha = Scalar{0};
    }
    else if (isfinite(rho) && isfinite(denominator) && rho > Scalar{0}
              && denominator > Scalar{0})
    {
        alpha = rho / denominator;
        if (!isfinite(alpha))
        {
            scalars[kBreakdown] = Scalar{1};
            alpha = Scalar{0};
        }
    }
    else
    {
        scalars[kBreakdown] = Scalar{1};
        alpha = Scalar{0};
    }
    scalars[kAlpha] = alpha;
    scalars[kNegativeAlpha] = -alpha;
}

template <typename Scalar>
__global__ void betaKernel(Scalar* scalars)
{
    if (scalars[kBreakdown] != Scalar{0})
    {
        scalars[kBeta] = Scalar{0};
        scalars[kAlpha] = Scalar{0};
        return;
    }
    const Scalar rho = scalars[kRho];
    const Scalar next_rho = scalars[kNextRho];
    Scalar beta = quietNan<Scalar>();
    if (rho == Scalar{0} && next_rho == Scalar{0})
    {
        beta = Scalar{0};
    }
    else if (isfinite(rho) && isfinite(next_rho) && rho > Scalar{0}
              && next_rho > Scalar{0})
    {
        beta = next_rho / rho;
        if (!isfinite(beta))
        {
            scalars[kBreakdown] = Scalar{1};
            beta = Scalar{0};
        }
    }
    else
    {
        scalars[kBreakdown] = Scalar{1};
        beta = Scalar{0};
    }
    scalars[kBeta] = beta;
    scalars[kRho] = next_rho;
    scalars[kAlpha] = Scalar{1};
}

template <typename Scalar>
__global__ void toDoubleKernel(const Scalar* input, double* output)
{
    *output = static_cast<double>(*input);
}

template <typename Scalar>
__global__ void toDoubleCheckedKernel(
    const Scalar* input, const Scalar* breakdown, double* output)
{
    *output = *breakdown == Scalar{0}
        ? static_cast<double>(*input) : quietNan<double>();
}

} // namespace iterative_solver_detail
} // namespace plamatrix
