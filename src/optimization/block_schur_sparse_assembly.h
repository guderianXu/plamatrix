#pragma once

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

#include "plamatrix/optimization/block_schur.h"
#include "plamatrix/sparse/csr_matrix.h"

#include "block_schur_linear_algebra.h"
#include "block_schur_device_assembly.h"

namespace plamatrix::block_schur_detail
{

struct SchurComplementSolverWorkspaceAccess
{
    template <typename Scalar>
    static bool matches(const SchurComplementSolverWorkspace<Scalar>& workspace,
                        Index primary_count,
                        Index eliminated_count,
                        Index primary_size,
                        Index eliminated_size,
                        const std::vector<Index>& topology)
    {
        return workspace._primaryBlockCount == primary_count
            && workspace._eliminatedBlockCount == eliminated_count
            && workspace._primaryBlockSize == primary_size
            && workspace._eliminatedBlockSize == eliminated_size
            && workspace._topology == topology
            && !workspace._rowOffsets.empty();
    }

    template <typename Scalar, typename PrimaryCrossBlocks, typename CrossBlocks, typename Adjacency>
    static void rebuild(SchurComplementSolverWorkspace<Scalar>& workspace,
                        Index primary_count,
                        Index eliminated_count,
                        Index primary_size,
                        Index eliminated_size,
                        const PrimaryCrossBlocks& primary_cross_blocks,
                        const CrossBlocks& cross_blocks,
                        const Adjacency& adjacency,
                        std::vector<Index> topology)
    {
        using BlockKey = std::pair<Index, Index>;
        std::map<BlockKey, Index> slot_by_key;
        for (Index block = 0; block < primary_count; ++block)
        {
            slot_by_key.emplace(BlockKey{block, block}, 0);
        }
        for (const auto& cross : primary_cross_blocks)
        {
            slot_by_key.emplace(BlockKey{cross.rowBlock, cross.columnBlock}, 0);
            slot_by_key.emplace(BlockKey{cross.columnBlock, cross.rowBlock}, 0);
        }
        for (Index eliminated = 0; eliminated < eliminated_count; ++eliminated)
        {
            const auto& entries = adjacency[static_cast<std::size_t>(eliminated)];
            for (const std::size_t left_index : entries)
            {
                for (const std::size_t right_index : entries)
                {
                    slot_by_key.emplace(
                        BlockKey{cross_blocks[left_index].primaryBlock,
                                 cross_blocks[right_index].primaryBlock},
                        0);
                }
            }
        }

        workspace._blockSlots.clear();
        workspace._blockSlots.reserve(slot_by_key.size());
        Index slot = 0;
        Index current_row = -1;
        Index column_ordinal = 0;
        for (auto& [key, value] : slot_by_key)
        {
            if (key.first != current_row)
            {
                current_row = key.first;
                column_ordinal = 0;
            }
            value = slot++;
            workspace._blockSlots.push_back(
                {key.first, key.second, column_ordinal++});
        }

        workspace._diagonalSlots.resize(static_cast<std::size_t>(primary_count));
        for (Index block = 0; block < primary_count; ++block)
        {
            workspace._diagonalSlots[static_cast<std::size_t>(block)] =
                slot_by_key.at(BlockKey{block, block});
        }
        workspace._primaryPairSlots.clear();
        workspace._primaryPairSlots.reserve(primary_cross_blocks.size() * 2);
        for (const auto& cross : primary_cross_blocks)
        {
            workspace._primaryPairSlots.push_back(
                slot_by_key.at(BlockKey{cross.rowBlock, cross.columnBlock}));
            workspace._primaryPairSlots.push_back(
                slot_by_key.at(BlockKey{cross.columnBlock, cross.rowBlock}));
        }
        workspace._eliminatedPairSlots.clear();
        workspace._eliminatedPairSlots.resize(static_cast<std::size_t>(eliminated_count));
        for (Index eliminated = 0; eliminated < eliminated_count; ++eliminated)
        {
            const auto& entries = adjacency[static_cast<std::size_t>(eliminated)];
            auto& pair_slots = workspace._eliminatedPairSlots[
                static_cast<std::size_t>(eliminated)];
            pair_slots.reserve(entries.size() * entries.size());
            for (const std::size_t left_index : entries)
            {
                for (const std::size_t right_index : entries)
                {
                    pair_slots.push_back(slot_by_key.at(
                        BlockKey{cross_blocks[left_index].primaryBlock,
                                 cross_blocks[right_index].primaryBlock}));
                }
            }
        }

        const Index dimension = primary_count * primary_size;
        workspace._rowOffsets.assign(static_cast<std::size_t>(dimension + 1), 0);
        workspace._columnIndices.clear();
        Index cursor = 0;
        for (Index block_row = 0; block_row < primary_count; ++block_row)
        {
            for (Index local_row = 0; local_row < primary_size; ++local_row)
            {
                const Index row = block_row * primary_size + local_row;
                workspace._rowOffsets[static_cast<std::size_t>(row)] = cursor;
                for (const auto& block_slot : workspace._blockSlots)
                {
                    if (block_slot.blockRow != block_row)
                    {
                        continue;
                    }
                    for (Index column = 0; column < primary_size; ++column)
                    {
                        workspace._columnIndices.push_back(
                            block_slot.blockColumn * primary_size + column);
                        ++cursor;
                    }
                }
            }
        }
        workspace._rowOffsets[static_cast<std::size_t>(dimension)] = cursor;
        workspace._primaryBlockCount = primary_count;
        workspace._eliminatedBlockCount = eliminated_count;
        workspace._primaryBlockSize = primary_size;
        workspace._eliminatedBlockSize = eliminated_size;
        workspace._topology = std::move(topology);

        const std::size_t value_count = workspace._columnIndices.size();
        workspace._valueBaseKinds.assign(value_count, 0);
        workspace._valueBaseIndices.assign(value_count, 0);
        workspace._valueBlockSlots.assign(value_count, 0);
        workspace._valueLocalRows.assign(value_count, 0);
        workspace._valueLocalColumns.assign(value_count, 0);
        for (std::size_t block_slot = 0;
             block_slot < workspace._blockSlots.size();
             ++block_slot)
        {
            for (Index row = 0; row < primary_size; ++row)
            {
                for (Index column = 0; column < primary_size; ++column)
                {
                    const Index position = valuePosition(
                        workspace, static_cast<Index>(block_slot), row, column);
                    workspace._valueBlockSlots[static_cast<std::size_t>(position)] =
                        static_cast<Index>(block_slot);
                    workspace._valueLocalRows[static_cast<std::size_t>(position)] = row;
                    workspace._valueLocalColumns[static_cast<std::size_t>(position)] = column;
                }
            }
        }
        for (Index block = 0; block < primary_count; ++block)
        {
            const Index diagonal_slot = workspace._diagonalSlots[static_cast<std::size_t>(block)];
            for (Index row = 0; row < primary_size; ++row)
            {
                for (Index column = 0; column < primary_size; ++column)
                {
                    const Index position = valuePosition(
                        workspace, diagonal_slot, row, column);
                    workspace._valueBaseKinds[static_cast<std::size_t>(position)] = 1;
                    workspace._valueBaseIndices[static_cast<std::size_t>(position)] =
                        block * primary_size * primary_size + row * primary_size + column;
                }
            }
        }
        for (std::size_t cross_index = 0;
             cross_index < primary_cross_blocks.size();
             ++cross_index)
        {
            const Index forward_slot = workspace._primaryPairSlots[cross_index * 2];
            const Index transpose_slot = workspace._primaryPairSlots[cross_index * 2 + 1];
            for (Index row = 0; row < primary_size; ++row)
            {
                for (Index column = 0; column < primary_size; ++column)
                {
                    const Index source = static_cast<Index>(cross_index) *
                        primary_size * primary_size + row * primary_size + column;
                    const Index forward = valuePosition(
                        workspace, forward_slot, row, column);
                    const Index transpose = valuePosition(
                        workspace, transpose_slot, column, row);
                    workspace._valueBaseKinds[static_cast<std::size_t>(forward)] = 2;
                    workspace._valueBaseIndices[static_cast<std::size_t>(forward)] = source;
                    workspace._valueBaseKinds[static_cast<std::size_t>(transpose)] = 2;
                    workspace._valueBaseIndices[static_cast<std::size_t>(transpose)] = source;
                }
            }
        }

        std::vector<std::vector<std::array<Index, 3>>> terms(
            workspace._blockSlots.size());
        for (Index eliminated = 0; eliminated < eliminated_count; ++eliminated)
        {
            const auto& entries = adjacency[static_cast<std::size_t>(eliminated)];
            const auto& pair_slots = workspace._eliminatedPairSlots[
                static_cast<std::size_t>(eliminated)];
            for (std::size_t left = 0; left < entries.size(); ++left)
            {
                for (std::size_t right = 0; right < entries.size(); ++right)
                {
                    const Index pair_slot = pair_slots[left * entries.size() + right];
                    terms[static_cast<std::size_t>(pair_slot)].push_back({{
                        eliminated,
                        static_cast<Index>(entries[left]),
                        static_cast<Index>(entries[right])}});
                }
            }
        }
        workspace._slotTermOffsets.assign(terms.size() + 1, 0);
        workspace._slotTermEliminated.clear();
        workspace._slotTermLeftCross.clear();
        workspace._slotTermRightCross.clear();
        for (std::size_t slot_index = 0; slot_index < terms.size(); ++slot_index)
        {
            workspace._slotTermOffsets[slot_index] = static_cast<Index>(
                workspace._slotTermEliminated.size());
            for (const auto& term : terms[slot_index])
            {
                workspace._slotTermEliminated.push_back(term[0]);
                workspace._slotTermLeftCross.push_back(term[1]);
                workspace._slotTermRightCross.push_back(term[2]);
            }
        }
        workspace._slotTermOffsets[terms.size()] = static_cast<Index>(
            workspace._slotTermEliminated.size());
        ++workspace._patternBuildCount;
    }

    template <typename Scalar>
    static Index valuePosition(const SchurComplementSolverWorkspace<Scalar>& workspace,
                               Index slot,
                               Index local_row,
                               Index local_column)
    {
        const auto& block = workspace._blockSlots[static_cast<std::size_t>(slot)];
        const Index row = block.blockRow * workspace._primaryBlockSize + local_row;
        return workspace._rowOffsets[static_cast<std::size_t>(row)]
            + block.columnOrdinal * workspace._primaryBlockSize + local_column;
    }

    template <typename Scalar>
    static const std::vector<Index>& rowOffsets(
        const SchurComplementSolverWorkspace<Scalar>& workspace)
    {
        return workspace._rowOffsets;
    }

    template <typename Scalar>
    static const std::vector<Index>& columnIndices(
        const SchurComplementSolverWorkspace<Scalar>& workspace)
    {
        return workspace._columnIndices;
    }

    template <typename Scalar>
    static Index diagonalSlot(const SchurComplementSolverWorkspace<Scalar>& workspace,
                              Index block)
    {
        return workspace._diagonalSlots[static_cast<std::size_t>(block)];
    }

    template <typename Scalar>
    static const std::vector<Index>& eliminatedPairSlots(
        const SchurComplementSolverWorkspace<Scalar>& workspace,
        Index eliminated)
    {
        return workspace._eliminatedPairSlots[static_cast<std::size_t>(eliminated)];
    }

    template <typename Scalar>
    static const std::vector<Index>& primaryPairSlots(
        const SchurComplementSolverWorkspace<Scalar>& workspace)
    {
        return workspace._primaryPairSlots;
    }

    template <typename Scalar>
    static const std::vector<Index>& valueBaseKinds(
        const SchurComplementSolverWorkspace<Scalar>& workspace)
    {
        return workspace._valueBaseKinds;
    }

    template <typename Scalar>
    static const std::vector<Index>& valueBaseIndices(
        const SchurComplementSolverWorkspace<Scalar>& workspace)
    {
        return workspace._valueBaseIndices;
    }

    template <typename Scalar>
    static const std::vector<Index>& valueLocalRows(
        const SchurComplementSolverWorkspace<Scalar>& workspace)
    {
        return workspace._valueLocalRows;
    }

    template <typename Scalar>
    static const std::vector<Index>& valueLocalColumns(
        const SchurComplementSolverWorkspace<Scalar>& workspace)
    {
        return workspace._valueLocalColumns;
    }

    template <typename Scalar>
    static const std::vector<Index>& valueBlockSlots(
        const SchurComplementSolverWorkspace<Scalar>& workspace)
    {
        return workspace._valueBlockSlots;
    }

    template <typename Scalar>
    static const std::vector<Index>& slotTermOffsets(
        const SchurComplementSolverWorkspace<Scalar>& workspace)
    {
        return workspace._slotTermOffsets;
    }

    template <typename Scalar>
    static const std::vector<Index>& slotTermEliminated(
        const SchurComplementSolverWorkspace<Scalar>& workspace)
    {
        return workspace._slotTermEliminated;
    }

    template <typename Scalar>
    static const std::vector<Index>& slotTermLeftCross(
        const SchurComplementSolverWorkspace<Scalar>& workspace)
    {
        return workspace._slotTermLeftCross;
    }

    template <typename Scalar>
    static const std::vector<Index>& slotTermRightCross(
        const SchurComplementSolverWorkspace<Scalar>& workspace)
    {
        return workspace._slotTermRightCross;
    }

    template <typename Scalar>
    static std::shared_ptr<void>& acceleratedState(
        SchurComplementSolverWorkspace<Scalar>& workspace)
    {
        return workspace._acceleratedState;
    }

    template <typename Scalar>
    static std::shared_ptr<void>& mixedPrecisionState(
        SchurComplementSolverWorkspace<Scalar>& workspace)
    {
        return workspace._mixedPrecisionState;
    }

    template <typename Scalar>
    static std::shared_ptr<void>& deviceAssemblyState(
        SchurComplementSolverWorkspace<Scalar>& workspace)
    {
        return workspace._deviceAssemblyState;
    }
};

template <typename Scalar, typename PrimaryCrossBlocks, typename CrossBlocks, typename Adjacency>
CSRMatrix<Scalar, Device::CPU> assembleReducedSchurCsr(
    Index primary_count,
    Index eliminated_count,
    Index primary_size,
    Index eliminated_size,
    const std::vector<Scalar>& primary_diagonal,
    const std::vector<std::vector<Scalar>>& eliminated_inverse,
    const PrimaryCrossBlocks& primary_cross_blocks,
    const CrossBlocks& cross_blocks,
    const Adjacency& adjacency,
    SchurComplementSolverWorkspace<Scalar>& workspace,
    bool* pattern_reused,
    SchurComplementLinearBackend backend,
    bool* assembly_on_device)
{
    std::vector<Index> topology{
        primary_count, eliminated_count, primary_size, eliminated_size};
    topology.push_back(static_cast<Index>(primary_cross_blocks.size()));
    for (const auto& cross : primary_cross_blocks)
    {
        topology.push_back(cross.rowBlock);
        topology.push_back(cross.columnBlock);
    }
    for (Index eliminated = 0; eliminated < eliminated_count; ++eliminated)
    {
        const auto& entries = adjacency[static_cast<std::size_t>(eliminated)];
        topology.push_back(static_cast<Index>(entries.size()));
        for (const std::size_t cross_index : entries)
        {
            topology.push_back(cross_blocks[cross_index].primaryBlock);
        }
    }
    const bool reused = SchurComplementSolverWorkspaceAccess::matches(
        workspace,
        primary_count,
        eliminated_count,
        primary_size,
        eliminated_size,
        topology);
    if (!reused)
    {
        SchurComplementSolverWorkspaceAccess::rebuild(
            workspace,
            primary_count,
            eliminated_count,
            primary_size,
            eliminated_size,
            primary_cross_blocks,
            cross_blocks,
            adjacency,
            std::move(topology));
    }
    if (pattern_reused)
    {
        *pattern_reused = reused;
    }

    const Index dimension = primary_count * primary_size;
    const auto& row_offsets = SchurComplementSolverWorkspaceAccess::rowOffsets(workspace);
    const auto& column_indices =
        SchurComplementSolverWorkspaceAccess::columnIndices(workspace);
    if (column_indices.size()
        > static_cast<std::size_t>(std::numeric_limits<Index>::max()))
    {
        throw std::overflow_error("Schur CSR nonzero count exceeds Index range");
    }
    CSRMatrix<Scalar, Device::CPU> matrix(
        dimension,
        dimension,
        static_cast<Index>(column_indices.size()));
    Index* matrix_row_offsets = matrix.rowOffsets();
    Index* matrix_column_indices = matrix.colIndices();
    Scalar* matrix_values = matrix.values();
    std::copy(row_offsets.begin(), row_offsets.end(), matrix_row_offsets);
    std::copy(column_indices.begin(), column_indices.end(), matrix_column_indices);
    std::fill(matrix_values, matrix_values + matrix.nnz(), Scalar(0));

    if (backend != SchurComplementLinearBackend::Cpu)
    {
        std::vector<Scalar> flattened_inverse;
        flattened_inverse.reserve(static_cast<std::size_t>(
            eliminated_count * eliminated_size * eliminated_size));
        for (const auto& inverse : eliminated_inverse)
        {
            flattened_inverse.insert(
                flattened_inverse.end(), inverse.begin(), inverse.end());
        }
        std::vector<Scalar> flattened_primary_cross;
        flattened_primary_cross.reserve(
            primary_cross_blocks.size() * static_cast<std::size_t>(primary_size * primary_size));
        for (const auto& cross : primary_cross_blocks)
        {
            flattened_primary_cross.insert(
                flattened_primary_cross.end(), cross.values.begin(), cross.values.end());
        }
        std::vector<Scalar> flattened_cross;
        flattened_cross.reserve(
            cross_blocks.size() * static_cast<std::size_t>(primary_size * eliminated_size));
        for (const auto& cross : cross_blocks)
        {
            flattened_cross.insert(
                flattened_cross.end(), cross.values.begin(), cross.values.end());
        }
        const auto& base_kinds =
            SchurComplementSolverWorkspaceAccess::valueBaseKinds(workspace);
        const auto& base_indices =
            SchurComplementSolverWorkspaceAccess::valueBaseIndices(workspace);
        const auto& value_block_slots =
            SchurComplementSolverWorkspaceAccess::valueBlockSlots(workspace);
        const auto& local_rows =
            SchurComplementSolverWorkspaceAccess::valueLocalRows(workspace);
        const auto& local_columns =
            SchurComplementSolverWorkspaceAccess::valueLocalColumns(workspace);
        const auto& term_offsets =
            SchurComplementSolverWorkspaceAccess::slotTermOffsets(workspace);
        const auto& term_eliminated =
            SchurComplementSolverWorkspaceAccess::slotTermEliminated(workspace);
        const auto& term_left_cross =
            SchurComplementSolverWorkspaceAccess::slotTermLeftCross(workspace);
        const auto& term_right_cross =
            SchurComplementSolverWorkspaceAccess::slotTermRightCross(workspace);
        std::vector<Scalar> values;
        if (backend == SchurComplementLinearBackend::Cuda)
        {
            values = assembleSchurValuesOnCuda(
                primary_size, eliminated_size, primary_diagonal, flattened_inverse,
                flattened_primary_cross, flattened_cross, base_kinds, base_indices,
                value_block_slots, local_rows, local_columns, term_offsets, term_eliminated,
                term_left_cross, term_right_cross, workspace, !reused);
        }
        else
        {
            values = assembleSchurValuesOnOpenCl(
                primary_size, eliminated_size, primary_diagonal, flattened_inverse,
                flattened_primary_cross, flattened_cross, base_kinds, base_indices,
                value_block_slots, local_rows, local_columns, term_offsets, term_eliminated,
                term_left_cross, term_right_cross, workspace, !reused);
        }
        if (backend == SchurComplementLinearBackend::OpenCl &&
            values.size() != static_cast<std::size_t>(matrix.nnz()))
        {
            throw std::runtime_error("Device Schur assembly returned an invalid value count");
        }
        if (backend == SchurComplementLinearBackend::OpenCl)
        {
            std::copy(values.begin(), values.end(), matrix_values);
        }
        if (assembly_on_device)
        {
            *assembly_on_device = true;
        }
        matrix.validateStructure();
        return matrix;
    }
    if (assembly_on_device)
    {
        *assembly_on_device = false;
    }

    auto add_value = [&](Index slot, Index row, Index column, Scalar value)
    {
        matrix_values[SchurComplementSolverWorkspaceAccess::valuePosition(
            workspace, slot, row, column)] += value;
    };
    for (Index block = 0; block < primary_count; ++block)
    {
        const Index slot = SchurComplementSolverWorkspaceAccess::diagonalSlot(
            workspace, block);
        const Scalar* source = primary_diagonal.data() + block * primary_size * primary_size;
        for (Index row = 0; row < primary_size; ++row)
        {
            for (Index column = 0; column < primary_size; ++column)
            {
                add_value(slot, row, column, source[row * primary_size + column]);
            }
        }
    }

    const auto& primary_pair_slots =
        SchurComplementSolverWorkspaceAccess::primaryPairSlots(workspace);
    for (std::size_t index = 0; index < primary_cross_blocks.size(); ++index)
    {
        const auto& cross = primary_cross_blocks[index];
        const Index forward_slot = primary_pair_slots[index * 2];
        const Index transpose_slot = primary_pair_slots[index * 2 + 1];
        for (Index row = 0; row < primary_size; ++row)
        {
            for (Index column = 0; column < primary_size; ++column)
            {
                add_value(forward_slot,
                          row,
                          column,
                          cross.values[static_cast<std::size_t>(row * primary_size + column)]);
                add_value(transpose_slot,
                          column,
                          row,
                          cross.values[static_cast<std::size_t>(row * primary_size + column)]);
            }
        }
    }

    for (Index eliminated = 0; eliminated < eliminated_count; ++eliminated)
    {
        const auto& entries = adjacency[static_cast<std::size_t>(eliminated)];
        std::vector<std::vector<Scalar>> transformed(entries.size());
        for (std::size_t left = 0; left < entries.size(); ++left)
        {
            const auto& cross = cross_blocks[entries[left]];
            transformed[left].resize(static_cast<std::size_t>(primary_size * eliminated_size));
            for (Index row = 0; row < primary_size; ++row)
            {
                multiplyMatrixVector(
                    eliminated_inverse[static_cast<std::size_t>(eliminated)].data(),
                    eliminated_size,
                    eliminated_size,
                    cross.values.data() + row * eliminated_size,
                    transformed[left].data() + row * eliminated_size);
            }
        }
        const auto& pair_slots = SchurComplementSolverWorkspaceAccess::eliminatedPairSlots(
            workspace, eliminated);
        for (std::size_t left = 0; left < entries.size(); ++left)
        {
            for (std::size_t right = 0; right < entries.size(); ++right)
            {
                const auto& right_cross = cross_blocks[entries[right]];
                const Index slot = pair_slots[left * entries.size() + right];
                for (Index row = 0; row < primary_size; ++row)
                {
                    for (Index column = 0; column < primary_size; ++column)
                    {
                        Scalar value = Scalar(0);
                        for (Index inner = 0; inner < eliminated_size; ++inner)
                        {
                            value += transformed[left][static_cast<std::size_t>(
                                         row * eliminated_size + inner)]
                                   * right_cross.values[static_cast<std::size_t>(
                                         column * eliminated_size + inner)];
                        }
                        add_value(slot, row, column, -value);
                    }
                }
            }
        }
    }
    matrix.validateStructure();
    return matrix;
}

} // namespace plamatrix::block_schur_detail
