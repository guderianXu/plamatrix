#pragma once

#include <memory>
#include <vector>

#include "plamatrix/internal/optimization/block_schur.h"
#include "plamatrix/internal/sparse/csr_storage.h"

namespace plamatrix::internal::block_schur_detail
{

    template <typename Scalar>
    SchurComplementSolverReport<Scalar>
    solveReducedSchurSparseDirect(const CsrStorage<Scalar, Device::CPU>& matrix,
                                  const std::vector<Scalar>& rhs,
                                  const SchurComplementSolverOptions<Scalar>& options,
                                  Index block_size,
                                  std::shared_ptr<void>& opaque_state,
                                  std::vector<Scalar>* solution);

} // namespace plamatrix::internal::block_schur_detail
