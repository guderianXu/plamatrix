#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "plamatrix/core/types.h"

namespace plamatrix::block_schur_detail
{

    class NativeBlockSparseCholesky
    {
    public:
        bool ensurePattern(Index dimension,
                           Index block_size,
                           const Index* row_offsets,
                           const Index* column_indices,
                           Index nonzeros,
                           bool* reused,
                           std::string* message);

        bool factorize(const float* values, std::string* message);
        bool factorize(const double* values, std::string* message);
        bool solve(const double* right_hand_side, double* solution, std::string* message);

    private:
        struct SourceEntry
        {
            Index sourceOffset = 0;
            std::size_t targetOffset = 0;
        };

        bool matches(Index dimension,
                     Index block_size,
                     const Index* row_offsets,
                     const Index* column_indices,
                     Index nonzeros) const;
        bool buildPattern(Index dimension,
                          Index block_size,
                          const Index* row_offsets,
                          const Index* column_indices,
                          Index nonzeros,
                          std::string* message);
        template <typename Scalar> bool factorizeValues(const Scalar* values, std::string* message);
        void clear() noexcept;

        Index _dimension = 0;
        Index _blockSize = 0;
        Index _blockCount = 0;
        std::vector<Index> _rowOffsets;
        std::vector<Index> _columnIndices;
        std::vector<Index> _blockPermutation;
        std::vector<Index> _inverseBlockPermutation;
        std::vector<std::size_t> _factorColumnOffsets;
        std::vector<Index> _factorRowBlocks;
        std::vector<std::size_t> _updateOffsets;
        std::vector<std::size_t> _updateTargets;
        std::vector<SourceEntry> _sourceEntries;
        std::vector<double> _factorValues;
        std::vector<double> _solveWork;
        std::vector<unsigned char> _activeCoordinates;
        bool _factorized = false;
    };

} // namespace plamatrix::block_schur_detail
