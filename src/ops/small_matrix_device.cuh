#pragma once

template <typename Scalar>
__device__ Scalar machineEpsilon();

template <>
__device__ float machineEpsilon<float>()
{
    return 1.1920928955078125e-7F;
}

template <>
__device__ double machineEpsilon<double>()
{
    return 2.2204460492503131e-16;
}

__global__ void initializeStatusKernel(small_matrix_detail::DeviceStatus* status)
{
    status->nonFiniteRow = kNoInvalidRow;
    status->basisFailureRow = kNoInvalidRow;
}

template <typename Scalar>
__device__ void zeroOutputRow(Index row, Index rows, Scalar* values, Scalar* vectors)
{
    for (int col = 0; col < 3; ++col)
    {
        values[row + static_cast<Index>(col) * rows] = Scalar(0);
    }
    for (int col = 0; col < 9; ++col)
    {
        vectors[row + static_cast<Index>(col) * rows] = Scalar(0);
    }
}
