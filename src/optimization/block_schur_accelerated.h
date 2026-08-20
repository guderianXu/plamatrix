#pragma once

#include <vector>

#include "plamatrix/optimization/block_schur.h"
#include "plamatrix/sparse/csr_matrix.h"

namespace plamatrix::block_schur_detail
{

template <typename Scalar>
SchurComplementSolverReport<Scalar> solveAcceleratedReducedSchur(
    const CSRMatrix<Scalar, Device::CPU>& matrix,
    const std::vector<Scalar>& rhs,
    const std::vector<std::vector<Scalar>>& inverse_diagonal_blocks,
    Index block_size,
    const SchurComplementSolverOptions<Scalar>& options,
    SchurComplementSolverWorkspace<Scalar>& workspace,
    std::vector<Scalar>* solution);

} // namespace plamatrix::block_schur_detail
