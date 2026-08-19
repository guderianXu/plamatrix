#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "plamatrix/core/types.h"

namespace plamatrix
{

namespace block_schur_detail
{
struct SchurComplementSolverWorkspaceAccess;
}

template <typename Scalar>
class BlockNormalEquations;

/// Linear solver used for the reduced Schur complement.
enum class SchurComplementLinearBackend
{
    Cpu,
    Cuda,
    OpenCl,
};

/// Controls the PCG solve of the reduced Schur complement.
template <typename Scalar>
struct SchurComplementSolverOptions
{
    SchurComplementLinearBackend linearBackend = SchurComplementLinearBackend::Cpu;
    /// CUDA device index, or the already selected OpenCL device index. Negative uses the default.
    int deviceIndex = -1;
    int maxIterations = 100;
    Scalar relativeTolerance = Scalar(1e-8);
    Scalar absoluteTolerance = Scalar(1e-12);
    Scalar minimumDiagonal = Scalar(1e-12);
};

/// Numerical status returned by solveDampedSchurComplement().
template <typename Scalar>
struct SchurComplementSolverReport
{
    SchurComplementLinearBackend linearBackend = SchurComplementLinearBackend::Cpu;
    bool converged = false;
    int iterations = 0;
    Scalar initialResidualNorm = Scalar(0);
    Scalar finalResidualNorm = Scalar(0);
    double schurAssemblySeconds = 0.0;
    double linearSolveSeconds = 0.0;
    bool schurPatternReused = false;
    bool schurAssemblyOnDevice = false;
    std::string deviceName;
    std::string message;
};

/// Reusable host-side structure for accelerated Schur CSR assembly.
template <typename Scalar>
class SchurComplementSolverWorkspace
{
public:
    void clear() noexcept
    {
        _primaryBlockCount = 0;
        _eliminatedBlockCount = 0;
        _primaryBlockSize = 0;
        _eliminatedBlockSize = 0;
        _topology.clear();
        _blockSlots.clear();
        _diagonalSlots.clear();
        _primaryPairSlots.clear();
        _eliminatedPairSlots.clear();
        _rowOffsets.clear();
        _columnIndices.clear();
        _valueBaseKinds.clear();
        _valueBaseIndices.clear();
        _valueBlockSlots.clear();
        _valueLocalRows.clear();
        _valueLocalColumns.clear();
        _slotTermOffsets.clear();
        _slotTermEliminated.clear();
        _slotTermLeftCross.clear();
        _slotTermRightCross.clear();
        _patternBuildCount = 0;
    }

    std::size_t patternBuildCount() const noexcept
    {
        return _patternBuildCount;
    }

private:
    struct BlockSlot
    {
        Index blockRow = 0;
        Index blockColumn = 0;
        Index columnOrdinal = 0;
    };

    Index _primaryBlockCount = 0;
    Index _eliminatedBlockCount = 0;
    Index _primaryBlockSize = 0;
    Index _eliminatedBlockSize = 0;
    std::vector<Index> _topology;
    std::vector<BlockSlot> _blockSlots;
    std::vector<Index> _diagonalSlots;
    std::vector<Index> _primaryPairSlots;
    std::vector<std::vector<Index>> _eliminatedPairSlots;
    std::vector<Index> _rowOffsets;
    std::vector<Index> _columnIndices;
    std::vector<Index> _valueBaseKinds;
    std::vector<Index> _valueBaseIndices;
    std::vector<Index> _valueBlockSlots;
    std::vector<Index> _valueLocalRows;
    std::vector<Index> _valueLocalColumns;
    std::vector<Index> _slotTermOffsets;
    std::vector<Index> _slotTermEliminated;
    std::vector<Index> _slotTermLeftCross;
    std::vector<Index> _slotTermRightCross;
    std::size_t _patternBuildCount = 0;

    friend struct block_schur_detail::SchurComplementSolverWorkspaceAccess;
};

template <typename Scalar>
SchurComplementSolverReport<Scalar> solveDampedSchurComplement(
    const BlockNormalEquations<Scalar>& equations,
    Scalar damping,
    const SchurComplementSolverOptions<Scalar>& options,
    std::vector<Scalar>* primary_step,
    std::vector<Scalar>* eliminated_step);

template <typename Scalar>
SchurComplementSolverReport<Scalar> solveDampedSchurComplement(
    const BlockNormalEquations<Scalar>& equations,
    Scalar damping,
    const SchurComplementSolverOptions<Scalar>& options,
    SchurComplementSolverWorkspace<Scalar>& workspace,
    std::vector<Scalar>* primary_step,
    std::vector<Scalar>* eliminated_step);

/**
 * @brief Block-sparse normal equations for bipartite least-squares problems.
 *
 * A residual may depend on one primary block, one eliminated block, or both.
 * Jacobians are supplied row-major as `residual_size x block_size`. Repeated
 * primary/eliminated pairs are accumulated into one cross block before Schur
 * elimination. All inputs must be finite; invalid dimensions throw.
 */
template <typename Scalar>
class BlockNormalEquations
{
public:
    BlockNormalEquations(Index primary_block_count,
                         Index eliminated_block_count,
                         Index primary_block_size,
                         Index eliminated_block_size);

    /// Add a residual depending on one primary and one eliminated block.
    void addResidualBlock(Index primary_block,
                          Index eliminated_block,
                          const Scalar* primary_jacobian,
                          const Scalar* eliminated_jacobian,
                          const Scalar* residual,
                          Index residual_size,
                          Scalar weight = Scalar(1));

    /**
     * @brief Add a residual depending on multiple primary blocks and one eliminated block.
     *
     * `primary_blocks` and `primary_jacobians` must have the same non-zero size,
     * contain unique block indices, and each Jacobian is row-major
     * `residual_size x primaryBlockSize()`.
     */
    void addResidualBlocks(const std::vector<Index>& primary_blocks,
                           const std::vector<const Scalar*>& primary_jacobians,
                           Index eliminated_block,
                           const Scalar* eliminated_jacobian,
                           const Scalar* residual,
                           Index residual_size,
                           Scalar weight = Scalar(1));

    /// Add a residual depending only on one primary block.
    void addPrimaryResidualBlock(Index primary_block,
                                 const Scalar* primary_jacobian,
                                 const Scalar* residual,
                                 Index residual_size,
                                 Scalar weight = Scalar(1));

    /// Add a residual depending on two or more primary blocks.
    void addPrimaryResidualBlocks(const std::vector<Index>& primary_blocks,
                                  const std::vector<const Scalar*>& primary_jacobians,
                                  const Scalar* residual,
                                  Index residual_size,
                                  Scalar weight = Scalar(1));

    /// Add a residual depending only on one eliminated block.
    void addEliminatedResidualBlock(Index eliminated_block,
                                    const Scalar* eliminated_jacobian,
                                    const Scalar* residual,
                                    Index residual_size,
                                    Scalar weight = Scalar(1));

    /// Return the number of primary variable blocks.
    Index primaryBlockCount() const noexcept;

    /// Return the number of eliminated variable blocks.
    Index eliminatedBlockCount() const noexcept;

    /// Return the scalar dimension of one primary block.
    Index primaryBlockSize() const noexcept;

    /// Return the scalar dimension of one eliminated block.
    Index eliminatedBlockSize() const noexcept;

private:
    struct CrossBlock
    {
        Index primaryBlock = 0;
        Index eliminatedBlock = 0;
        std::vector<Scalar> values;
    };

    struct PrimaryCrossBlock
    {
        Index rowBlock = 0;
        Index columnBlock = 0;
        std::vector<Scalar> values;
    };

    void validateResidual(const Scalar* jacobian,
                          Index jacobian_size,
                          const Scalar* residual,
                          Index residual_size,
                          Scalar weight,
                          const char* operation) const;
    std::size_t findOrCreateCrossBlock(Index primary_block, Index eliminated_block);
    std::size_t findOrCreatePrimaryCrossBlock(Index row_block, Index column_block);
    void addPrimaryTerms(const std::vector<Index>& primary_blocks,
                         const std::vector<const Scalar*>& primary_jacobians,
                         const Scalar* residual,
                         Index residual_size,
                         Scalar weight);

    Index _primaryBlockCount = 0;
    Index _eliminatedBlockCount = 0;
    Index _primaryBlockSize = 0;
    Index _eliminatedBlockSize = 0;
    std::vector<Scalar> _primaryDiagonal;
    std::vector<Scalar> _eliminatedDiagonal;
    std::vector<Scalar> _primaryGradient;
    std::vector<Scalar> _eliminatedGradient;
    std::vector<PrimaryCrossBlock> _primaryCrossBlocks;
    std::vector<CrossBlock> _crossBlocks;
    std::vector<std::vector<std::size_t>> _eliminatedAdjacency;

    friend SchurComplementSolverReport<Scalar> solveDampedSchurComplement<Scalar>(
        const BlockNormalEquations<Scalar>& equations,
        Scalar damping,
        const SchurComplementSolverOptions<Scalar>& options,
        std::vector<Scalar>* primary_step,
        std::vector<Scalar>* eliminated_step);
    friend SchurComplementSolverReport<Scalar> solveDampedSchurComplement<Scalar>(
        const BlockNormalEquations<Scalar>& equations,
        Scalar damping,
        const SchurComplementSolverOptions<Scalar>& options,
        SchurComplementSolverWorkspace<Scalar>& workspace,
        std::vector<Scalar>* primary_step,
        std::vector<Scalar>* eliminated_step);
};

extern template class BlockNormalEquations<float>;
extern template class BlockNormalEquations<double>;

extern template SchurComplementSolverReport<float> solveDampedSchurComplement(
    const BlockNormalEquations<float>&,
    float,
    const SchurComplementSolverOptions<float>&,
    std::vector<float>*,
    std::vector<float>*);
extern template SchurComplementSolverReport<double> solveDampedSchurComplement(
    const BlockNormalEquations<double>&,
    double,
    const SchurComplementSolverOptions<double>&,
    std::vector<double>*,
    std::vector<double>*);
extern template SchurComplementSolverReport<float> solveDampedSchurComplement(
    const BlockNormalEquations<float>&,
    float,
    const SchurComplementSolverOptions<float>&,
    SchurComplementSolverWorkspace<float>&,
    std::vector<float>*,
    std::vector<float>*);
extern template SchurComplementSolverReport<double> solveDampedSchurComplement(
    const BlockNormalEquations<double>&,
    double,
    const SchurComplementSolverOptions<double>&,
    SchurComplementSolverWorkspace<double>&,
    std::vector<double>*,
    std::vector<double>*);

} // namespace plamatrix
