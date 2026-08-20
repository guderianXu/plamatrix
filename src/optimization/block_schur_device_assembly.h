#pragma once

#include <vector>

#include "plamatrix/core/types.h"
#include "plamatrix/optimization/block_schur.h"

namespace plamatrix::block_schur_detail
{

template <typename Scalar>
std::vector<Scalar> assembleSchurValuesOnCuda(
    Index primary_size,
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
std::vector<Scalar> assembleSchurValuesOnOpenCl(
    Index primary_size,
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
void copyLastCudaSchurValuesToDevice(
    Scalar* destination,
    std::size_t value_count,
    SchurComplementSolverWorkspace<Scalar>& workspace);


} // namespace plamatrix::block_schur_detail
