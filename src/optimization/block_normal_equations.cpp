#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include "plamatrix/optimization/block_schur.h"

namespace plamatrix
{

template <typename Scalar>
BlockNormalEquations<Scalar>::BlockNormalEquations(
    Index primary_block_count,
    Index eliminated_block_count,
    Index primary_block_size,
    Index eliminated_block_size)
    : _primaryBlockCount(primary_block_count),
      _eliminatedBlockCount(eliminated_block_count),
      _primaryBlockSize(primary_block_size),
      _eliminatedBlockSize(eliminated_block_size)
{
    static_assert(std::is_floating_point_v<Scalar>,
                  "BlockNormalEquations requires a floating-point scalar");
    if (primary_block_count < 0 || eliminated_block_count < 0 ||
        primary_block_size <= 0 || eliminated_block_size <= 0)
    {
        throw std::invalid_argument(
            "BlockNormalEquations: block counts must be non-negative and block sizes positive");
    }

    _primaryDiagonal.assign(
        static_cast<std::size_t>(primary_block_count * primary_block_size * primary_block_size),
        Scalar(0));
    _eliminatedDiagonal.assign(
        static_cast<std::size_t>(
            eliminated_block_count * eliminated_block_size * eliminated_block_size),
        Scalar(0));
    _primaryGradient.assign(
        static_cast<std::size_t>(primary_block_count * primary_block_size), Scalar(0));
    _eliminatedGradient.assign(
        static_cast<std::size_t>(eliminated_block_count * eliminated_block_size), Scalar(0));
    _eliminatedAdjacency.resize(static_cast<std::size_t>(eliminated_block_count));
}

template <typename Scalar>
void BlockNormalEquations<Scalar>::validateResidual(
    const Scalar* jacobian,
    Index jacobian_size,
    const Scalar* residual,
    Index residual_size,
    Scalar weight,
    const char* operation) const
{
    if (!jacobian || !residual || residual_size <= 0)
    {
        throw std::invalid_argument(std::string(operation) +
                                    ": pointers must be non-null and residual size positive");
    }
    if (!std::isfinite(weight) || weight < Scalar(0))
    {
        throw std::invalid_argument(std::string(operation) +
                                    ": weight must be finite and non-negative");
    }
    for (Index index = 0; index < jacobian_size; ++index)
    {
        if (!std::isfinite(jacobian[index]))
        {
            throw std::invalid_argument(std::string(operation) +
                                        ": Jacobian values must be finite");
        }
    }
    for (Index index = 0; index < residual_size; ++index)
    {
        if (!std::isfinite(residual[index]))
        {
            throw std::invalid_argument(std::string(operation) +
                                        ": residual values must be finite");
        }
    }
}

template <typename Scalar>
std::size_t BlockNormalEquations<Scalar>::findOrCreateCrossBlock(
    Index primary_block,
    Index eliminated_block)
{
    auto& adjacency = _eliminatedAdjacency[static_cast<std::size_t>(eliminated_block)];
    for (const std::size_t cross_index : adjacency)
    {
        if (_crossBlocks[cross_index].primaryBlock == primary_block)
        {
            return cross_index;
        }
    }

    CrossBlock cross;
    cross.primaryBlock = primary_block;
    cross.eliminatedBlock = eliminated_block;
    cross.values.assign(
        static_cast<std::size_t>(_primaryBlockSize * _eliminatedBlockSize), Scalar(0));
    _crossBlocks.push_back(std::move(cross));
    adjacency.push_back(_crossBlocks.size() - 1);
    return _crossBlocks.size() - 1;
}

template <typename Scalar>
std::size_t BlockNormalEquations<Scalar>::findOrCreatePrimaryCrossBlock(
    Index row_block,
    Index column_block)
{
    if (row_block >= column_block)
    {
        throw std::invalid_argument(
            "BlockNormalEquations: primary cross blocks must be upper triangular");
    }
    for (std::size_t index = 0; index < _primaryCrossBlocks.size(); ++index)
    {
        const auto& cross = _primaryCrossBlocks[index];
        if (cross.rowBlock == row_block && cross.columnBlock == column_block)
        {
            return index;
        }
    }
    PrimaryCrossBlock cross;
    cross.rowBlock = row_block;
    cross.columnBlock = column_block;
    cross.values.assign(
        static_cast<std::size_t>(_primaryBlockSize * _primaryBlockSize), Scalar(0));
    _primaryCrossBlocks.push_back(std::move(cross));
    return _primaryCrossBlocks.size() - 1;
}

template <typename Scalar>
void BlockNormalEquations<Scalar>::addPrimaryTerms(
    const std::vector<Index>& primary_blocks,
    const std::vector<const Scalar*>& primary_jacobians,
    const Scalar* residual,
    Index residual_size,
    Scalar weight)
{
    if (primary_blocks.empty() || primary_blocks.size() != primary_jacobians.size())
    {
        throw std::invalid_argument(
            "BlockNormalEquations: primary block and Jacobian lists must have equal non-zero size");
    }
    for (std::size_t index = 0; index < primary_blocks.size(); ++index)
    {
        const Index block = primary_blocks[index];
        if (block < 0 || block >= _primaryBlockCount)
        {
            throw std::out_of_range("BlockNormalEquations: primary block index out of range");
        }
        if (std::find(primary_blocks.begin(), primary_blocks.begin() + index, block) !=
            primary_blocks.begin() + index)
        {
            throw std::invalid_argument("BlockNormalEquations: primary block indices must be unique");
        }
        validateResidual(primary_jacobians[index],
                         residual_size * _primaryBlockSize,
                         residual,
                         residual_size,
                         weight,
                         "BlockNormalEquations::addPrimaryTerms");
    }

    for (std::size_t block_index = 0; block_index < primary_blocks.size(); ++block_index)
    {
        const Index block = primary_blocks[block_index];
        const Scalar* jacobian = primary_jacobians[block_index];
        const Index matrix_offset = block * _primaryBlockSize * _primaryBlockSize;
        const Index gradient_offset = block * _primaryBlockSize;
        for (Index row = 0; row < residual_size; ++row)
        {
            for (Index column = 0; column < _primaryBlockSize; ++column)
            {
                const Scalar value = jacobian[row * _primaryBlockSize + column];
                _primaryGradient[static_cast<std::size_t>(gradient_offset + column)] +=
                    weight * value * residual[row];
                for (Index other = 0; other < _primaryBlockSize; ++other)
                {
                    _primaryDiagonal[static_cast<std::size_t>(
                        matrix_offset + column * _primaryBlockSize + other)] +=
                        weight * value * jacobian[row * _primaryBlockSize + other];
                }
            }
        }
    }

    for (std::size_t left = 0; left < primary_blocks.size(); ++left)
    {
        for (std::size_t right = left + 1; right < primary_blocks.size(); ++right)
        {
            const bool ordered = primary_blocks[left] < primary_blocks[right];
            const Index row_block = ordered ? primary_blocks[left] : primary_blocks[right];
            const Index column_block = ordered ? primary_blocks[right] : primary_blocks[left];
            const Scalar* row_jacobian = ordered
                ? primary_jacobians[left]
                : primary_jacobians[right];
            const Scalar* column_jacobian = ordered
                ? primary_jacobians[right]
                : primary_jacobians[left];
            auto& cross = _primaryCrossBlocks[
                findOrCreatePrimaryCrossBlock(row_block, column_block)].values;
            for (Index residual_row = 0; residual_row < residual_size; ++residual_row)
            {
                for (Index row = 0; row < _primaryBlockSize; ++row)
                {
                    const Scalar row_value =
                        row_jacobian[residual_row * _primaryBlockSize + row];
                    for (Index column = 0; column < _primaryBlockSize; ++column)
                    {
                        cross[static_cast<std::size_t>(row * _primaryBlockSize + column)] +=
                            weight * row_value *
                            column_jacobian[residual_row * _primaryBlockSize + column];
                    }
                }
            }
        }
    }
}

template <typename Scalar>
void BlockNormalEquations<Scalar>::addResidualBlock(
    Index primary_block,
    Index eliminated_block,
    const Scalar* primary_jacobian,
    const Scalar* eliminated_jacobian,
    const Scalar* residual,
    Index residual_size,
    Scalar weight)
{
    addResidualBlocks({primary_block},
                      {primary_jacobian},
                      eliminated_block,
                      eliminated_jacobian,
                      residual,
                      residual_size,
                      weight);
}

template <typename Scalar>
void BlockNormalEquations<Scalar>::addResidualBlocks(
    const std::vector<Index>& primary_blocks,
    const std::vector<const Scalar*>& primary_jacobians,
    Index eliminated_block,
    const Scalar* eliminated_jacobian,
    const Scalar* residual,
    Index residual_size,
    Scalar weight)
{
    if (eliminated_block < 0 || eliminated_block >= _eliminatedBlockCount)
    {
        throw std::out_of_range("BlockNormalEquations::addResidualBlocks: eliminated block out of range");
    }
    validateResidual(eliminated_jacobian,
                     residual_size * _eliminatedBlockSize,
                     residual,
                     residual_size,
                     weight,
                     "BlockNormalEquations::addResidualBlocks(eliminated)");
    addPrimaryTerms(primary_blocks, primary_jacobians, residual, residual_size, weight);

    const Index eliminated_matrix_offset =
        eliminated_block * _eliminatedBlockSize * _eliminatedBlockSize;
    const Index eliminated_gradient_offset = eliminated_block * _eliminatedBlockSize;
    for (std::size_t primary_index = 0; primary_index < primary_blocks.size(); ++primary_index)
    {
        auto& cross = _crossBlocks[findOrCreateCrossBlock(
            primary_blocks[primary_index], eliminated_block)].values;
        const Scalar* primary_jacobian = primary_jacobians[primary_index];
        for (Index row = 0; row < residual_size; ++row)
        {
            for (Index primary_column = 0;
                 primary_column < _primaryBlockSize;
                 ++primary_column)
            {
                const Scalar primary_value =
                    primary_jacobian[row * _primaryBlockSize + primary_column];
                for (Index eliminated_column = 0;
                     eliminated_column < _eliminatedBlockSize;
                     ++eliminated_column)
                {
                    cross[static_cast<std::size_t>(
                        primary_column * _eliminatedBlockSize + eliminated_column)] +=
                        weight * primary_value *
                        eliminated_jacobian[row * _eliminatedBlockSize + eliminated_column];
                }
            }
        }
    }

    for (Index row = 0; row < residual_size; ++row)
    {
        for (Index eliminated_col = 0; eliminated_col < _eliminatedBlockSize; ++eliminated_col)
        {
            const Scalar eliminated_value =
                eliminated_jacobian[row * _eliminatedBlockSize + eliminated_col];
            _eliminatedGradient[static_cast<std::size_t>(
                eliminated_gradient_offset + eliminated_col)] +=
                weight * eliminated_value * residual[row];
            for (Index other = 0; other < _eliminatedBlockSize; ++other)
            {
                _eliminatedDiagonal[static_cast<std::size_t>(
                    eliminated_matrix_offset + eliminated_col * _eliminatedBlockSize + other)] +=
                    weight * eliminated_value *
                    eliminated_jacobian[row * _eliminatedBlockSize + other];
            }
        }
    }
}

template <typename Scalar>
void BlockNormalEquations<Scalar>::addPrimaryResidualBlock(
    Index primary_block,
    const Scalar* primary_jacobian,
    const Scalar* residual,
    Index residual_size,
    Scalar weight)
{
    addPrimaryResidualBlocks(
        {primary_block}, {primary_jacobian}, residual, residual_size, weight);
}

template <typename Scalar>
void BlockNormalEquations<Scalar>::addPrimaryResidualBlocks(
    const std::vector<Index>& primary_blocks,
    const std::vector<const Scalar*>& primary_jacobians,
    const Scalar* residual,
    Index residual_size,
    Scalar weight)
{
    addPrimaryTerms(primary_blocks, primary_jacobians, residual, residual_size, weight);
}

template <typename Scalar>
void BlockNormalEquations<Scalar>::addEliminatedResidualBlock(
    Index eliminated_block,
    const Scalar* eliminated_jacobian,
    const Scalar* residual,
    Index residual_size,
    Scalar weight)
{
    if (eliminated_block < 0 || eliminated_block >= _eliminatedBlockCount)
    {
        throw std::out_of_range(
            "BlockNormalEquations::addEliminatedResidualBlock: block index out of range");
    }
    validateResidual(eliminated_jacobian,
                     residual_size * _eliminatedBlockSize,
                     residual,
                     residual_size,
                     weight,
                     "BlockNormalEquations::addEliminatedResidualBlock");
    const Index matrix_offset = eliminated_block * _eliminatedBlockSize * _eliminatedBlockSize;
    const Index gradient_offset = eliminated_block * _eliminatedBlockSize;
    for (Index row = 0; row < residual_size; ++row)
    {
        for (Index col = 0; col < _eliminatedBlockSize; ++col)
        {
            const Scalar value = eliminated_jacobian[row * _eliminatedBlockSize + col];
            _eliminatedGradient[static_cast<std::size_t>(gradient_offset + col)] +=
                weight * value * residual[row];
            for (Index other = 0; other < _eliminatedBlockSize; ++other)
            {
                _eliminatedDiagonal[static_cast<std::size_t>(
                    matrix_offset + col * _eliminatedBlockSize + other)] +=
                    weight * value * eliminated_jacobian[row * _eliminatedBlockSize + other];
            }
        }
    }
}

template <typename Scalar>
void BlockNormalEquations<Scalar>::mergeFrom(const BlockNormalEquations& other)
{
    if (_primaryBlockCount != other._primaryBlockCount ||
        _eliminatedBlockCount != other._eliminatedBlockCount ||
        _primaryBlockSize != other._primaryBlockSize ||
        _eliminatedBlockSize != other._eliminatedBlockSize)
    {
        throw std::invalid_argument(
            "BlockNormalEquations::mergeFrom: block layouts must match");
    }
    const auto add_values = [](std::vector<Scalar>* target,
                               const std::vector<Scalar>& source)
    {
        for (std::size_t index = 0; index < source.size(); ++index)
        {
            (*target)[index] += source[index];
        }
    };
    add_values(&_primaryDiagonal, other._primaryDiagonal);
    add_values(&_eliminatedDiagonal, other._eliminatedDiagonal);
    add_values(&_primaryGradient, other._primaryGradient);
    add_values(&_eliminatedGradient, other._eliminatedGradient);

    for (const auto& source : other._primaryCrossBlocks)
    {
        auto& target = _primaryCrossBlocks[findOrCreatePrimaryCrossBlock(
            source.rowBlock, source.columnBlock)].values;
        add_values(&target, source.values);
    }
    for (const auto& source : other._crossBlocks)
    {
        auto& target = _crossBlocks[findOrCreateCrossBlock(
            source.primaryBlock, source.eliminatedBlock)].values;
        add_values(&target, source.values);
    }
}

template <typename Scalar>
Index BlockNormalEquations<Scalar>::primaryBlockCount() const noexcept
{
    return _primaryBlockCount;
}

template <typename Scalar>
Index BlockNormalEquations<Scalar>::eliminatedBlockCount() const noexcept
{
    return _eliminatedBlockCount;
}

template <typename Scalar>
Index BlockNormalEquations<Scalar>::primaryBlockSize() const noexcept
{
    return _primaryBlockSize;
}

template <typename Scalar>
Index BlockNormalEquations<Scalar>::eliminatedBlockSize() const noexcept
{
    return _eliminatedBlockSize;
}

template class BlockNormalEquations<float>;
template class BlockNormalEquations<double>;

} // namespace plamatrix
