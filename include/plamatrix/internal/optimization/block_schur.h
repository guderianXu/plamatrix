#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "plamatrix/internal/core/device.h"

namespace plamatrix::internal
{

    namespace block_schur_detail
    {
        struct SchurComplementSolverWorkspaceAccess;
    }

    template <typename Scalar> class BlockNormalEquations;

    /// Linear solver used for the reduced Schur complement.
    enum class SchurComplementLinearBackend
    {
        Cpu,
        DenseCpu,
        SparseCpu,
        Cuda,
        OpenCl,
        Vulkan,
    };

    /// Return whether the built-in CPU sparse-direct Schur solver is available.
    bool hasSparseDirectSchurSolver() noexcept;

    /// Controls the PCG solve of the reduced Schur complement.
    template <typename Scalar> struct SchurComplementSolverOptions
    {
        SchurComplementLinearBackend linearBackend = SchurComplementLinearBackend::Cpu;
        /// CUDA device index, or the already selected OpenCL device index. Vulkan uses
        /// PLAMATRIX_VULKAN_DEVICE_INDEX. Negative uses the backend default.
        int deviceIndex = -1;
        int maxIterations = 100;
        Scalar relativeTolerance = Scalar(1e-8);
        Scalar absoluteTolerance = Scalar(1e-12);
        Scalar minimumDiagonal = Scalar(1e-12);
        /// Consecutive primary blocks per CPU cluster-Jacobi preconditioner. One uses block Jacobi.
        int preconditionerClusterSize = 1;
        /// Treat a correctly sized primary_step as an initial guess instead of clearing it.
        bool useInitialGuess = false;
        /// Seed a double-precision CUDA/OpenCL solve with a guarded float PCG pass. Vulkan requires
        /// this opt-in and currently accepts float equations only because Schur assembly uses FP16
        /// cooperative-matrix inputs with FP32 accumulation.
        bool useMixedPrecision = false;
        /// Internal handoff flag for an already assembled CUDA Schur values buffer.
        bool schurValuesOnDevice = true;
    };

    /// Numerical status returned by solveDampedSchurComplement().
    template <typename Scalar> struct SchurComplementSolverReport
    {
        SchurComplementLinearBackend linearBackend = SchurComplementLinearBackend::Cpu;
        bool converged = false;
        int iterations = 0;
        Scalar initialResidualNorm = Scalar(0);
        Scalar finalResidualNorm = Scalar(0);
        /// Time spent factoring/inverting small eliminated and preconditioner blocks.
        double smallBlockInverseSeconds = 0.0;
        /// Time spent accumulating numerical Schur-complement values.
        double schurAccumulationSeconds = 0.0;
        /// Time spent building/copying the reduced CSR representation.
        double csrConversionSeconds = 0.0;
        double schurAssemblySeconds = 0.0;
        /// Time spent factoring the dense reduced system.
        double choleskyFactorizationSeconds = 0.0;
        /// Time spent in the dense forward/backward triangular solves.
        double triangularSolveSeconds = 0.0;
        /// Time spent preparing and analyzing a new sparse factorization pattern.
        double symbolicAnalysisSeconds = 0.0;
        /// Time spent checking the final reduced-system residual.
        double residualCheckSeconds = 0.0;
        double linearSolveSeconds = 0.0;
        /// Time spent recovering eliminated-variable steps.
        double backSubstitutionSeconds = 0.0;
        bool schurPatternReused = false;
        bool symbolicAnalysisReused = false;
        bool schurAssemblyOnDevice = false;
        bool mixedPrecisionUsed = false;
        bool cooperativeMatrixUsed = false;
        bool blockSpmvUsed = false;
        /// Whether Vulkan generated the 9x9 block-Jacobi preconditioner on the device.
        bool deviceBlockJacobiUsed = false;
        std::uint32_t commandSubmissions = 0;
        std::uint32_t commandBufferRecordings = 0;
        double gpuMilliseconds = 0.0;
        std::string deviceName;
        std::string preconditionerName;
        std::string message;
    };

    /// Reusable host-side structure for accelerated Schur CSR assembly.
    template <typename Scalar> class SchurComplementSolverWorkspace
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
            _hostDenseSchur.clear();
            _hostDenseReference.clear();
            _hostTransformedCross.clear();
            _hostPrimaryDiagonal.clear();
            _hostEliminatedDiagonal.clear();
            _hostEliminatedInverse.clear();
            _hostReducedRhs.clear();
            _hostScratch.clear();
            _acceleratedState.reset();
            _mixedPrecisionState.reset();
            _deviceAssemblyState.reset();
            _vulkanAssemblyState.reset();
            _sparseDirectState.reset();
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
        std::vector<Scalar> _hostDenseSchur;
        std::vector<Scalar> _hostDenseReference;
        std::vector<Scalar> _hostTransformedCross;
        std::vector<Scalar> _hostPrimaryDiagonal;
        std::vector<Scalar> _hostEliminatedDiagonal;
        std::vector<Scalar> _hostEliminatedInverse;
        std::vector<Scalar> _hostReducedRhs;
        std::vector<Scalar> _hostScratch;
        std::shared_ptr<void> _acceleratedState;
        std::shared_ptr<void> _mixedPrecisionState;
        std::shared_ptr<void> _deviceAssemblyState;
        std::shared_ptr<void> _vulkanAssemblyState;
        std::shared_ptr<void> _sparseDirectState;
        std::size_t _patternBuildCount = 0;

        friend struct block_schur_detail::SchurComplementSolverWorkspaceAccess;
    };

    template <typename Scalar>
    SchurComplementSolverReport<Scalar> solveDampedSchurComplement(const BlockNormalEquations<Scalar>& equations,
                                                                   Scalar damping,
                                                                   const SchurComplementSolverOptions<Scalar>& options,
                                                                   std::vector<Scalar>* primary_step,
                                                                   std::vector<Scalar>* eliminated_step);

    template <typename Scalar>
    SchurComplementSolverReport<Scalar> solveDampedSchurComplement(const BlockNormalEquations<Scalar>& equations,
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
    template <typename Scalar> class BlockNormalEquations
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

        /// Pointer/count variant for hot paths with a small stack-allocated block list.
        void addResidualBlocks(const Index* primary_blocks,
                               const Scalar* const* primary_jacobians,
                               std::size_t primary_block_count,
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

        /// Pointer/count variant for hot paths with a small stack-allocated block list.
        void addPrimaryResidualBlocks(const Index* primary_blocks,
                                      const Scalar* const* primary_jacobians,
                                      std::size_t primary_block_count,
                                      const Scalar* residual,
                                      Index residual_size,
                                      Scalar weight = Scalar(1));

        /// Add a residual depending only on one eliminated block.
        void addEliminatedResidualBlock(Index eliminated_block,
                                        const Scalar* eliminated_jacobian,
                                        const Scalar* residual,
                                        Index residual_size,
                                        Scalar weight = Scalar(1));

        /**
         * @brief Add an already accumulated primary gradient block.
         *
         * `values` contains `primaryBlockSize()` finite scalar entries. This
         * low-level entry point is intended for callers that eliminate local
         * variables while assembling a reduced Schur system.
         */
        void addPrimaryGradientBlock(Index primary_block, const Scalar* values);

        /**
         * @brief Add an already accumulated primary Hessian block.
         *
         * `values` is a row-major square block. Diagonal blocks are accumulated
         * directly. Off-diagonal inputs may use either block order; values are
         * transposed when the order is normalized to the stored upper triangle.
         */
        void addPrimaryHessianBlock(Index row_block, Index column_block, const Scalar* values);

        /**
         * @brief Multiply non-zero primary scalar diagonals and constrain empty columns.
         *
         * This finalizes a pre-reduced system whose local variables were already
         * eliminated with the same damping multiplier. `zero_diagonal` is assigned
         * only where the accumulated scalar diagonal is exactly zero.
         */
        void finalizePrimaryDiagonal(Scalar multiplier, Scalar zero_diagonal = Scalar(1));

        /// Deterministically accumulate another equation set with the same block layout.
        void mergeFrom(const BlockNormalEquations& other);

        /**
         * @brief Accumulate a compact shard of eliminated blocks.
         *
         * Primary blocks use the same global layout in both equation sets. Eliminated
         * block zero in `other` is mapped to `eliminated_block_offset` in this set.
         * This lets parallel callers keep only their disjoint eliminated-variable
         * range instead of replicating every eliminated block per worker.
         */
        void mergeEliminatedShardFrom(const BlockNormalEquations& other, Index eliminated_block_offset);

        /// Clear all accumulated numeric values while retaining the discovered block topology.
        void clearValues() noexcept;

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

        struct BlockPairHash
        {
            std::size_t operator()(const std::pair<Index, Index>& value) const noexcept
            {
                const std::size_t first = std::hash<Index>{}(value.first);
                const std::size_t second = std::hash<Index>{}(value.second);
                return first ^ (second + static_cast<std::size_t>(0x9e3779b9) + (first << 6) + (first >> 2));
            }
        };

        void validateResidual(const Scalar* jacobian,
                              Index jacobian_size,
                              const Scalar* residual,
                              Index residual_size,
                              Scalar weight,
                              const char* operation) const;
        std::size_t findOrCreateCrossBlock(Index primary_block, Index eliminated_block);
        std::size_t findOrCreatePrimaryCrossBlock(Index row_block, Index column_block);
        void addPrimaryTerms(const Index* primary_blocks,
                             const Scalar* const* primary_jacobians,
                             std::size_t primary_block_count,
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
        std::vector<std::vector<std::size_t>> _primaryAdjacency;
        std::vector<CrossBlock> _crossBlocks;
        std::vector<std::vector<std::size_t>> _eliminatedAdjacency;
        std::unordered_map<std::pair<Index, Index>, std::size_t, BlockPairHash> _crossBlockIndex;
        std::unordered_map<std::pair<Index, Index>, std::size_t, BlockPairHash> _primaryCrossBlockIndex;

        friend SchurComplementSolverReport<Scalar>
        solveDampedSchurComplement<Scalar>(const BlockNormalEquations<Scalar>& equations,
                                           Scalar damping,
                                           const SchurComplementSolverOptions<Scalar>& options,
                                           std::vector<Scalar>* primary_step,
                                           std::vector<Scalar>* eliminated_step);
        friend SchurComplementSolverReport<Scalar>
        solveDampedSchurComplement<Scalar>(const BlockNormalEquations<Scalar>& equations,
                                           Scalar damping,
                                           const SchurComplementSolverOptions<Scalar>& options,
                                           SchurComplementSolverWorkspace<Scalar>& workspace,
                                           std::vector<Scalar>* primary_step,
                                           std::vector<Scalar>* eliminated_step);
    };

    extern template class BlockNormalEquations<float>;
    extern template class BlockNormalEquations<double>;

    extern template SchurComplementSolverReport<float>
    solveDampedSchurComplement(const BlockNormalEquations<float>&,
                               float,
                               const SchurComplementSolverOptions<float>&,
                               std::vector<float>*,
                               std::vector<float>*);
    extern template SchurComplementSolverReport<double>
    solveDampedSchurComplement(const BlockNormalEquations<double>&,
                               double,
                               const SchurComplementSolverOptions<double>&,
                               std::vector<double>*,
                               std::vector<double>*);
    extern template SchurComplementSolverReport<float>
    solveDampedSchurComplement(const BlockNormalEquations<float>&,
                               float,
                               const SchurComplementSolverOptions<float>&,
                               SchurComplementSolverWorkspace<float>&,
                               std::vector<float>*,
                               std::vector<float>*);
    extern template SchurComplementSolverReport<double>
    solveDampedSchurComplement(const BlockNormalEquations<double>&,
                               double,
                               const SchurComplementSolverOptions<double>&,
                               SchurComplementSolverWorkspace<double>&,
                               std::vector<double>*,
                               std::vector<double>*);

} // namespace plamatrix::internal
