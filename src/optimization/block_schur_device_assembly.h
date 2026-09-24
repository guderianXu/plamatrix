#pragma once

#include <vector>

#include "plamatrix/internal/core/device.h"
#include "plamatrix/internal/optimization/block_schur.h"
#include "plamatrix/internal/sparse/csr_storage.h"
#include "plamatrix/internal/sparse/iterative_solver.h"

namespace plamatrix::internal::block_schur_detail
{

    template <typename Scalar>
    std::vector<Scalar> assembleSchurValuesOnCuda(Index primary_size,
                                                  Index eliminated_size,
                                                  const std::vector<Scalar>& primary_diagonal,
                                                  const std::vector<Scalar>& eliminated_inverse,
                                                  const std::vector<Scalar>& primary_cross_values,
                                                  const std::vector<Scalar>& cross_values,
                                                  const std::vector<Index>& cross_eliminated_blocks,
                                                  const std::vector<Index>& base_kinds,
                                                  const std::vector<Index>& base_indices,
                                                  const std::vector<Index>& value_block_slots,
                                                  const std::vector<Index>& local_rows,
                                                  const std::vector<Index>& local_columns,
                                                  const std::vector<Index>& term_offsets,
                                                  const std::vector<Index>& term_eliminated,
                                                  const std::vector<Index>& term_left_cross,
                                                  const std::vector<Index>& term_right_cross,
                                                  SchurComplementSolverWorkspace<Scalar>& workspace,
                                                  bool upload_topology);

    template <typename Scalar>
    std::vector<Scalar> assembleSchurValuesOnOpenCl(Index primary_size,
                                                    Index eliminated_size,
                                                    const std::vector<Scalar>& primary_diagonal,
                                                    const std::vector<Scalar>& eliminated_inverse,
                                                    const std::vector<Scalar>& primary_cross_values,
                                                    const std::vector<Scalar>& cross_values,
                                                    const std::vector<Index>& base_kinds,
                                                    const std::vector<Index>& base_indices,
                                                    const std::vector<Index>& value_block_slots,
                                                    const std::vector<Index>& local_rows,
                                                    const std::vector<Index>& local_columns,
                                                    const std::vector<Index>& term_offsets,
                                                    const std::vector<Index>& term_eliminated,
                                                    const std::vector<Index>& term_left_cross,
                                                    const std::vector<Index>& term_right_cross,
                                                    SchurComplementSolverWorkspace<Scalar>& workspace,
                                                    bool upload_topology);

    template <typename Scalar>
    void copyLastCudaSchurValuesToDevice(Scalar* destination,
                                         std::size_t value_count,
                                         SchurComplementSolverWorkspace<Scalar>& workspace);

    std::vector<float> assembleSchurValuesOnVulkan(Index primary_size,
                                                   Index eliminated_size,
                                                   const std::vector<float>& primary_diagonal,
                                                   const std::vector<float>& eliminated_inverse,
                                                   const std::vector<float>& primary_cross_values,
                                                   const std::vector<float>& cross_values,
                                                   const std::vector<Index>& cross_eliminated_blocks,
                                                   const std::vector<Index>& base_kinds,
                                                   const std::vector<Index>& base_indices,
                                                   const std::vector<Index>& value_block_slots,
                                                   const std::vector<Index>& local_rows,
                                                   const std::vector<Index>& local_columns,
                                                   const std::vector<Index>& term_offsets,
                                                   const std::vector<Index>& term_left_cross,
                                                   const std::vector<Index>& term_right_cross,
                                                   SchurComplementSolverWorkspace<float>& workspace,
                                                   bool upload_topology);

    IterativeSolverReport solveLastVulkanSchurValues(const CsrStorage<float, Device::CPU>& matrix,
                                                     const DenseStorage<float, Device::CPU>& rhs,
                                                     DenseStorage<float, Device::CPU>& solution,
                                                     const DenseStorage<float, Device::CPU>* inverse_blocks,
                                                     Index block_size,
                                                     bool build_device_block_jacobi,
                                                     SchurComplementSolverWorkspace<float>& workspace,
                                                     const IterativeSolverOptions& options);

} // namespace plamatrix::internal::block_schur_detail
