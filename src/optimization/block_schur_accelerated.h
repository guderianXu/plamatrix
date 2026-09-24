#pragma once

#include <vector>

#include "plamatrix/internal/optimization/block_schur.h"
#include "plamatrix/internal/sparse/csr_storage.h"

namespace plamatrix::internal::block_schur_detail
{

template <typename Scalar>
SchurComplementSolverReport<Scalar> solveAcceleratedReducedSchur(
    const CsrStorage<Scalar, Device::CPU>& matrix,
    const std::vector<Scalar>& rhs,
    const std::vector<std::vector<Scalar>>& inverse_diagonal_blocks,
    Index block_size,
    const SchurComplementSolverOptions<Scalar>& options,
    SchurComplementSolverWorkspace<Scalar>& workspace,
    std::vector<Scalar>* solution);

} // namespace plamatrix::internal::block_schur_detail
