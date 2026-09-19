#include "block_sparse_cholesky.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <queue>
#include <set>
#include <utility>

#include <omp.h>

#include "plamatrix/core/checked_math.h"

namespace plamatrix::block_schur_detail
{
    namespace
    {

        using BlockPair = std::pair<Index, Index>;

        std::vector<Index> minimumDegreeOrder(std::vector<std::set<Index>> adjacency,
                                              std::vector<std::vector<Index>>* filled_neighbors)
        {
            const Index block_count = static_cast<Index>(adjacency.size());
            using QueueEntry = std::pair<std::size_t, Index>;
            std::priority_queue<QueueEntry, std::vector<QueueEntry>, std::greater<QueueEntry>> queue;
            std::vector<bool> active(static_cast<std::size_t>(block_count), true);
            for (Index block = 0; block < block_count; ++block)
            {
                queue.emplace(adjacency[static_cast<std::size_t>(block)].size(), block);
            }

            std::vector<Index> order;
            order.reserve(static_cast<std::size_t>(block_count));
            filled_neighbors->clear();
            filled_neighbors->reserve(static_cast<std::size_t>(block_count));
            while (static_cast<Index>(order.size()) < block_count)
            {
                Index selected = -1;
                while (!queue.empty())
                {
                    const auto [degree, block] = queue.top();
                    queue.pop();
                    if (active[static_cast<std::size_t>(block)] &&
                        degree == adjacency[static_cast<std::size_t>(block)].size())
                    {
                        selected = block;
                        break;
                    }
                }
                if (selected < 0)
                {
                    return {};
                }

                std::vector<Index> neighbors;
                for (const Index neighbor : adjacency[static_cast<std::size_t>(selected)])
                {
                    if (active[static_cast<std::size_t>(neighbor)])
                    {
                        neighbors.push_back(neighbor);
                    }
                }
                for (std::size_t left = 0; left < neighbors.size(); ++left)
                {
                    for (std::size_t right = 0; right < left; ++right)
                    {
                        adjacency[static_cast<std::size_t>(neighbors[left])].insert(neighbors[right]);
                        adjacency[static_cast<std::size_t>(neighbors[right])].insert(neighbors[left]);
                    }
                }
                for (const Index neighbor : neighbors)
                {
                    adjacency[static_cast<std::size_t>(neighbor)].erase(selected);
                    queue.emplace(adjacency[static_cast<std::size_t>(neighbor)].size(), neighbor);
                }
                active[static_cast<std::size_t>(selected)] = false;
                order.push_back(selected);
                filled_neighbors->push_back(std::move(neighbors));
            }
            return order;
        }

        bool factorDiagonalBlock(Index block_size, double* block)
        {
            for (Index column = 0; column < block_size; ++column)
            {
                double diagonal = block[column * block_size + column];
                for (Index inner = 0; inner < column; ++inner)
                {
                    const double value = block[column * block_size + inner];
                    diagonal -= value * value;
                }
                if (!(diagonal > 0.0) || !std::isfinite(diagonal))
                {
                    return false;
                }
                block[column * block_size + column] = std::sqrt(diagonal);
                for (Index row = column + 1; row < block_size; ++row)
                {
                    double value = block[row * block_size + column];
                    for (Index inner = 0; inner < column; ++inner)
                    {
                        value -= block[row * block_size + inner] * block[column * block_size + inner];
                    }
                    block[row * block_size + column] = value / block[column * block_size + column];
                }
            }
            return true;
        }

        void rightSolveLowerTranspose(Index block_size, const double* diagonal, double* block)
        {
            for (Index row = 0; row < block_size; ++row)
            {
                for (Index column = 0; column < block_size; ++column)
                {
                    double value = block[row * block_size + column];
                    for (Index inner = 0; inner < column; ++inner)
                    {
                        value -= block[row * block_size + inner] * diagonal[column * block_size + inner];
                    }
                    block[row * block_size + column] = value / diagonal[column * block_size + column];
                }
            }
        }

        void subtractProduct(Index block_size, const double* left, const double* right, bool diagonal, double* target)
        {
            for (Index row = 0; row < block_size; ++row)
            {
                const Index column_end = diagonal ? row + 1 : block_size;
                for (Index column = 0; column < column_end; ++column)
                {
                    double update = 0.0;
#pragma omp simd reduction(+ : update)
                    for (Index inner = 0; inner < block_size; ++inner)
                    {
                        update += left[row * block_size + inner] * right[column * block_size + inner];
                    }
                    target[row * block_size + column] -= update;
                }
            }
        }

    } // namespace

    bool NativeBlockSparseCholesky::ensurePattern(Index dimension,
                                                  Index block_size,
                                                  const Index* row_offsets,
                                                  const Index* column_indices,
                                                  Index nonzeros,
                                                  bool* reused,
                                                  std::string* message)
    {
        if (!reused || !message)
        {
            return false;
        }
        if (matches(dimension, block_size, row_offsets, column_indices, nonzeros))
        {
            *reused = true;
            return true;
        }
        *reused = false;
        clear();
        return buildPattern(dimension, block_size, row_offsets, column_indices, nonzeros, message);
    }

    bool NativeBlockSparseCholesky::matches(
        Index dimension, Index block_size, const Index* row_offsets, const Index* column_indices, Index nonzeros) const
    {
        return _dimension == dimension && _blockSize == block_size &&
               _rowOffsets.size() == static_cast<std::size_t>(dimension + 1) &&
               _columnIndices.size() == static_cast<std::size_t>(nonzeros) &&
               std::equal(_rowOffsets.begin(), _rowOffsets.end(), row_offsets) &&
               std::equal(_columnIndices.begin(), _columnIndices.end(), column_indices);
    }

    bool NativeBlockSparseCholesky::buildPattern(Index dimension,
                                                 Index block_size,
                                                 const Index* row_offsets,
                                                 const Index* column_indices,
                                                 Index nonzeros,
                                                 std::string* message)
    {
        if (dimension <= 0 || block_size <= 0 || dimension % block_size != 0 || nonzeros < dimension || !row_offsets ||
            !column_indices || row_offsets[0] != 0 || row_offsets[dimension] != nonzeros)
        {
            *message = "native sparse Cholesky received an invalid block CSR pattern";
            return false;
        }
        _dimension = dimension;
        _blockSize = block_size;
        _blockCount = dimension / block_size;
        _rowOffsets.assign(row_offsets, row_offsets + dimension + 1);
        _columnIndices.assign(column_indices, column_indices + nonzeros);

        std::vector<std::set<Index>> adjacency(static_cast<std::size_t>(_blockCount));
        for (Index row = 0; row < dimension; ++row)
        {
            if (row_offsets[row] > row_offsets[row + 1])
            {
                *message = "native sparse Cholesky CSR row offsets are not monotonic";
                clear();
                return false;
            }
            const Index row_block = row / block_size;
            for (Index offset = row_offsets[row]; offset < row_offsets[row + 1]; ++offset)
            {
                const Index column = column_indices[offset];
                if (column < 0 || column >= dimension)
                {
                    *message = "native sparse Cholesky CSR column is out of range";
                    clear();
                    return false;
                }
                const Index column_block = column / block_size;
                if (column_block != row_block)
                {
                    adjacency[static_cast<std::size_t>(row_block)].insert(column_block);
                    adjacency[static_cast<std::size_t>(column_block)].insert(row_block);
                }
            }
        }

        std::vector<std::vector<Index>> filled_neighbors;
        _blockPermutation = minimumDegreeOrder(std::move(adjacency), &filled_neighbors);
        if (static_cast<Index>(_blockPermutation.size()) != _blockCount)
        {
            *message = "native sparse Cholesky minimum-degree ordering failed";
            clear();
            return false;
        }
        _inverseBlockPermutation.resize(static_cast<std::size_t>(_blockCount));
        for (Index new_block = 0; new_block < _blockCount; ++new_block)
        {
            _inverseBlockPermutation[static_cast<std::size_t>(_blockPermutation[static_cast<std::size_t>(new_block)])] =
                new_block;
        }

        std::map<BlockPair, std::size_t> slots;
        _factorColumnOffsets.reserve(static_cast<std::size_t>(_blockCount + 1));
        _factorColumnOffsets.push_back(0);
        for (Index column = 0; column < _blockCount; ++column)
        {
            std::vector<Index> rows{column};
            for (const Index old_neighbor : filled_neighbors[static_cast<std::size_t>(column)])
            {
                rows.push_back(_inverseBlockPermutation[static_cast<std::size_t>(old_neighbor)]);
            }
            std::sort(rows.begin() + 1, rows.end());
            for (const Index row : rows)
            {
                if (row < column)
                {
                    *message = "native sparse Cholesky produced an invalid filled block";
                    clear();
                    return false;
                }
                const std::size_t slot = _factorRowBlocks.size();
                _factorRowBlocks.push_back(row);
                slots.emplace(BlockPair{row, column}, slot);
            }
            _factorColumnOffsets.push_back(_factorRowBlocks.size());
        }

        _updateOffsets.reserve(static_cast<std::size_t>(_blockCount + 1));
        _updateOffsets.push_back(0);
        for (Index column = 0; column < _blockCount; ++column)
        {
            const std::size_t begin = _factorColumnOffsets[static_cast<std::size_t>(column)];
            const std::size_t end = _factorColumnOffsets[static_cast<std::size_t>(column + 1)];
            for (std::size_t left = begin + 1; left < end; ++left)
            {
                for (std::size_t right = begin + 1; right <= left; ++right)
                {
                    const BlockPair pair{_factorRowBlocks[left], _factorRowBlocks[right]};
                    const auto target = slots.find(pair);
                    if (target == slots.end())
                    {
                        *message = "native sparse Cholesky fill pattern is incomplete";
                        clear();
                        return false;
                    }
                    _updateTargets.push_back(target->second);
                }
            }
            _updateOffsets.push_back(_updateTargets.size());
        }

        const std::size_t block_values = static_cast<std::size_t>(
            detail::checkedIndexMul(block_size, block_size, "sparse Cholesky block values"));
        _factorValues.resize(_factorRowBlocks.size() * block_values);
        _sourceEntries.reserve(static_cast<std::size_t>(nonzeros / 2 + dimension));
        for (Index old_row = 0; old_row < dimension; ++old_row)
        {
            for (Index offset = row_offsets[old_row]; offset < row_offsets[old_row + 1]; ++offset)
            {
                const Index old_column = column_indices[offset];
                if (old_column > old_row)
                {
                    continue;
                }
                Index row_block = _inverseBlockPermutation[static_cast<std::size_t>(old_row / block_size)];
                Index column_block = _inverseBlockPermutation[static_cast<std::size_t>(old_column / block_size)];
                Index local_row = old_row % block_size;
                Index local_column = old_column % block_size;
                if (row_block < column_block)
                {
                    std::swap(row_block, column_block);
                    std::swap(local_row, local_column);
                }
                const auto slot = slots.find(BlockPair{row_block, column_block});
                if (slot == slots.end())
                {
                    *message = "native sparse Cholesky could not map a source block";
                    clear();
                    return false;
                }
                const std::size_t target =
                    slot->second * block_values + static_cast<std::size_t>(local_row * block_size + local_column);
                _sourceEntries.push_back({offset, target});
            }
        }
        _solveWork.resize(static_cast<std::size_t>(dimension));
        return true;
    }

    template <typename Scalar>
    bool NativeBlockSparseCholesky::factorizeValues(const Scalar* values, std::string* message)
    {
        if (!values || _factorColumnOffsets.empty())
        {
            *message = "native sparse Cholesky has no analyzed pattern";
            return false;
        }
        std::fill(_factorValues.begin(), _factorValues.end(), 0.0);
        for (const SourceEntry& entry : _sourceEntries)
        {
            _factorValues[entry.targetOffset] = static_cast<double>(values[entry.sourceOffset]);
        }

        const std::size_t block_values = static_cast<std::size_t>(
            detail::checkedIndexMul(_blockSize, _blockSize, "sparse Cholesky factor values"));
        _activeCoordinates.assign(static_cast<std::size_t>(_dimension), 0);
        for (Index column = 0; column < _blockCount; ++column)
        {
            const std::size_t begin = _factorColumnOffsets[static_cast<std::size_t>(column)];
            const std::size_t end = _factorColumnOffsets[static_cast<std::size_t>(column + 1)];
            for (std::size_t slot = begin; slot < end; ++slot)
            {
                const Index row_block = _factorRowBlocks[slot];
                const double* block = _factorValues.data() + slot * block_values;
                for (Index row = 0; row < _blockSize; ++row)
                {
                    for (Index local_column = 0; local_column < _blockSize; ++local_column)
                    {
                        if (block[row * _blockSize + local_column] == 0.0)
                        {
                            continue;
                        }
                        _activeCoordinates[static_cast<std::size_t>(row_block * _blockSize + row)] = 1;
                        _activeCoordinates[static_cast<std::size_t>(column * _blockSize + local_column)] = 1;
                    }
                }
            }
        }
        for (Index block = 0; block < _blockCount; ++block)
        {
            const std::size_t diagonal_slot = _factorColumnOffsets[static_cast<std::size_t>(block)];
            double* diagonal = _factorValues.data() + diagonal_slot * block_values;
            for (Index local = 0; local < _blockSize; ++local)
            {
                if (_activeCoordinates[static_cast<std::size_t>(block * _blockSize + local)] == 0)
                {
                    diagonal[local * _blockSize + local] = 1.0;
                }
            }
        }
        for (Index column = 0; column < _blockCount; ++column)
        {
            const std::size_t begin = _factorColumnOffsets[static_cast<std::size_t>(column)];
            const std::size_t end = _factorColumnOffsets[static_cast<std::size_t>(column + 1)];
            double* diagonal = _factorValues.data() + begin * block_values;
            if (!factorDiagonalBlock(_blockSize, diagonal))
            {
                *message = "native sparse Cholesky encountered a non-positive pivot";
                _factorized = false;
                return false;
            }

            const bool parallel = end - begin >= 8 && omp_in_parallel() == 0;
#pragma omp parallel for schedule(static) if (parallel)
            for (Index slot = static_cast<Index>(begin + 1); slot < static_cast<Index>(end); ++slot)
            {
                rightSolveLowerTranspose(
                    _blockSize, diagonal, _factorValues.data() + static_cast<std::size_t>(slot) * block_values);
            }

#pragma omp parallel for schedule(static) if (parallel)
            for (Index left = 1; left < static_cast<Index>(end - begin); ++left)
            {
                const std::size_t left_slot = begin + static_cast<std::size_t>(left);
                const double* left_block = _factorValues.data() + left_slot * block_values;
                for (Index right = 1; right <= left; ++right)
                {
                    const std::size_t pair_offset = static_cast<std::size_t>((left - 1) * left / 2 + right - 1);
                    const std::size_t target_slot =
                        _updateTargets[_updateOffsets[static_cast<std::size_t>(column)] + pair_offset];
                    const std::size_t right_slot = begin + static_cast<std::size_t>(right);
                    subtractProduct(_blockSize,
                                    left_block,
                                    _factorValues.data() + right_slot * block_values,
                                    left == right,
                                    _factorValues.data() + target_slot * block_values);
                }
            }
        }
        _factorized = true;
        return true;
    }

    bool NativeBlockSparseCholesky::factorize(const float* values, std::string* message)
    {
        return factorizeValues(values, message);
    }

    bool NativeBlockSparseCholesky::factorize(const double* values, std::string* message)
    {
        return factorizeValues(values, message);
    }

    bool NativeBlockSparseCholesky::solve(const double* right_hand_side, double* solution, std::string* message)
    {
        if (!_factorized || !right_hand_side || !solution)
        {
            *message = "native sparse Cholesky solve requires a valid numeric factor";
            return false;
        }
        for (Index block = 0; block < _blockCount; ++block)
        {
            const Index old_block = _blockPermutation[static_cast<std::size_t>(block)];
            for (Index local = 0; local < _blockSize; ++local)
            {
                const std::size_t permuted_index = static_cast<std::size_t>(block * _blockSize + local);
                const double value = right_hand_side[old_block * _blockSize + local];
                if (_activeCoordinates[permuted_index] == 0 && value != 0.0)
                {
                    *message = "native sparse Cholesky inactive coordinate has a non-zero right-hand side";
                    return false;
                }
                _solveWork[permuted_index] = _activeCoordinates[permuted_index] == 0 ? 0.0 : value;
            }
        }

        const std::size_t block_values = static_cast<std::size_t>(
            detail::checkedIndexMul(_blockSize, _blockSize, "sparse Cholesky solve values"));
        for (Index column = 0; column < _blockCount; ++column)
        {
            const std::size_t begin = _factorColumnOffsets[static_cast<std::size_t>(column)];
            const std::size_t end = _factorColumnOffsets[static_cast<std::size_t>(column + 1)];
            const double* diagonal = _factorValues.data() + begin * block_values;
            double* current = _solveWork.data() + column * _blockSize;
            for (Index row = 0; row < _blockSize; ++row)
            {
                double value = current[row];
                for (Index inner = 0; inner < row; ++inner)
                {
                    value -= diagonal[row * _blockSize + inner] * current[inner];
                }
                current[row] = value / diagonal[row * _blockSize + row];
            }
            for (std::size_t slot = begin + 1; slot < end; ++slot)
            {
                const double* lower = _factorValues.data() + slot * block_values;
                double* target = _solveWork.data() + _factorRowBlocks[slot] * _blockSize;
                for (Index row = 0; row < _blockSize; ++row)
                {
                    double update = 0.0;
                    for (Index inner = 0; inner < _blockSize; ++inner)
                    {
                        update += lower[row * _blockSize + inner] * current[inner];
                    }
                    target[row] -= update;
                }
            }
        }

        for (Index column = _blockCount; column-- > 0;)
        {
            const std::size_t begin = _factorColumnOffsets[static_cast<std::size_t>(column)];
            const std::size_t end = _factorColumnOffsets[static_cast<std::size_t>(column + 1)];
            const double* diagonal = _factorValues.data() + begin * block_values;
            double* current = _solveWork.data() + column * _blockSize;
            for (std::size_t slot = begin + 1; slot < end; ++slot)
            {
                const double* lower = _factorValues.data() + slot * block_values;
                const double* source = _solveWork.data() + _factorRowBlocks[slot] * _blockSize;
                for (Index row = 0; row < _blockSize; ++row)
                {
                    for (Index inner = 0; inner < _blockSize; ++inner)
                    {
                        current[inner] -= lower[row * _blockSize + inner] * source[row];
                    }
                }
            }
            for (Index row = _blockSize; row-- > 0;)
            {
                double value = current[row];
                for (Index inner = row + 1; inner < _blockSize; ++inner)
                {
                    value -= diagonal[inner * _blockSize + row] * current[inner];
                }
                current[row] = value / diagonal[row * _blockSize + row];
            }
        }

        for (Index block = 0; block < _blockCount; ++block)
        {
            const Index old_block = _blockPermutation[static_cast<std::size_t>(block)];
            for (Index local = 0; local < _blockSize; ++local)
            {
                solution[old_block * _blockSize + local] =
                    _solveWork[static_cast<std::size_t>(block * _blockSize + local)];
            }
        }
        return true;
    }

    void NativeBlockSparseCholesky::clear() noexcept
    {
        _dimension = 0;
        _blockSize = 0;
        _blockCount = 0;
        _rowOffsets.clear();
        _columnIndices.clear();
        _blockPermutation.clear();
        _inverseBlockPermutation.clear();
        _factorColumnOffsets.clear();
        _factorRowBlocks.clear();
        _updateOffsets.clear();
        _updateTargets.clear();
        _sourceEntries.clear();
        _factorValues.clear();
        _solveWork.clear();
        _activeCoordinates.clear();
        _factorized = false;
    }

} // namespace plamatrix::block_schur_detail
