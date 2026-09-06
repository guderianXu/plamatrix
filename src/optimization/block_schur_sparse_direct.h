#pragma once

#include <memory>
#include <vector>

#include "plamatrix/optimization/block_schur.h"
#include "plamatrix/sparse/csr_matrix.h"

namespace plamatrix::block_schur_detail
{

    template <typename Scalar>
    SchurComplementSolverReport<Scalar>
    solveReducedSchurSparseDirect(const CSRMatrix<Scalar, Device::CPU>& matrix,
                                  const std::vector<Scalar>& rhs,
                                  const SchurComplementSolverOptions<Scalar>& options,
                                  Index block_size,
                                  std::shared_ptr<void>& opaque_state,
                                  std::vector<Scalar>* solution);

} // namespace plamatrix::block_schur_detail
