#pragma once

#include "plamatrix/sparse/iterative_solver.h"

namespace plamatrix
{
namespace opencl
{

/// Solve an SPD CPU-owned CSR system on the selected OpenCL GPU.
/// Matrix/vector storage is uploaded once, all PCG iterations execute on the device,
/// and the final iterate is copied back to solution.
template <typename Scalar>
IterativeSolverReport pcg(
    const CSRMatrix<Scalar, Device::CPU>& matrix,
    const DenseMatrix<Scalar, Device::CPU>& rhs,
    DenseMatrix<Scalar, Device::CPU>& solution,
    const IterativeSolverOptions& options = {});

/// Solve an SPD CPU-owned CSR system with caller-supplied inverse diagonal blocks on OpenCL.
/// inverse_blocks stores row-major blocks as a contiguous (matrix.rows() * block_size) x 1 vector.
template <typename Scalar>
IterativeSolverReport blockPcg(
    const CSRMatrix<Scalar, Device::CPU>& matrix,
    const DenseMatrix<Scalar, Device::CPU>& rhs,
    DenseMatrix<Scalar, Device::CPU>& solution,
    const DenseMatrix<Scalar, Device::CPU>& inverse_blocks,
    Index block_size,
    const IterativeSolverOptions& options = {});

} // namespace opencl
} // namespace plamatrix
