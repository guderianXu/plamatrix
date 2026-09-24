#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

#include "plamatrix/internal/ops/indexing.h"

namespace plamatrix::internal
{
    namespace indexing_detail
    {

        constexpr int kIndexingBlockSize = 256;

        enum class StatusCategory
        {
            None,
            Overflow,
            OutOfRange
        };

        struct DeviceStatusBatch
        {
            Index overflow;
            Index out_of_range;
        };

        static_assert(sizeof(DeviceStatusBatch) == 2 * sizeof(Index));

        inline Index* selectStatus(DeviceStatusBatch* status, StatusCategory category) noexcept
        {
            if (category == StatusCategory::Overflow)
            {
                return &status->overflow;
            }
            if (category == StatusCategory::OutOfRange)
            {
                return &status->out_of_range;
            }
            return nullptr;
        }

        inline std::size_t alignedOffset(std::size_t offset, std::size_t alignment)
        {
            if (alignment == 0 || (alignment & (alignment - 1)) != 0)
            {
                throw std::logic_error("indexing: workspace alignment must be a power of two");
            }
            if (offset > std::numeric_limits<std::size_t>::max() - (alignment - 1))
            {
                throw std::overflow_error("indexing: workspace alignment overflows size_t");
            }
            return (offset + alignment - 1) & ~(alignment - 1);
        }

        inline std::size_t checkedAppend(std::size_t offset, std::size_t alignment, std::size_t bytes)
        {
            const std::size_t aligned = alignedOffset(offset, alignment);
            if (aligned > std::numeric_limits<std::size_t>::max() - bytes)
            {
                throw std::overflow_error("indexing: workspace size overflows size_t");
            }
            return aligned + bytes;
        }

        inline std::size_t checkedIndexBytes(Index count, const char* operation)
        {
            if (count < 0 ||
                static_cast<std::uintmax_t>(count) > std::numeric_limits<std::size_t>::max() / sizeof(Index))
            {
                throw std::overflow_error(std::string(operation) + ": scratch size overflows size_t");
            }
            return static_cast<std::size_t>(count) * sizeof(Index);
        }

        inline unsigned int checkedGrid(Index count, const char* operation)
        {
            if (count <= 0)
            {
                return 0;
            }
            const Index blocks = count / kIndexingBlockSize + (count % kIndexingBlockSize != 0 ? 1 : 0);
            if (blocks > static_cast<Index>(std::numeric_limits<int>::max()))
            {
                throw std::overflow_error(std::string(operation) + ": CUDA grid range exceeded");
            }
            return static_cast<unsigned int>(blocks);
        }

        struct IndexingWorkspaceAccess
        {
            static bool beginStatusBatch(IndexingWorkspace& workspace) noexcept;
        };

        template <typename MatrixType>
        void requireShape(const char* operation, const MatrixType& matrix, Index rows, Index cols, const char* name)
        {
            if (matrix.rows() != rows || matrix.cols() != cols)
            {
                std::ostringstream message;
                message << operation << ": " << name << " must have shape " << rows << "x" << cols;
                throw std::invalid_argument(message.str());
            }
        }

        template <typename MatrixType> void requireIndexVector(const char* operation, const MatrixType& indices)
        {
            if (indices.cols() != 1)
            {
                throw std::invalid_argument(std::string(operation) + ": indices must have shape K x 1");
            }
        }

        void launchExclusiveScan(const DenseStorage<Index, Device::GPU>& counts,
                                 DenseStorage<Index, Device::GPU>& output,
                                 IndexingWorkspace& workspace,
                                 cudaStream_t stream);

        void launchExclusiveScan(ConstMatrixView<Index, Device::GPU> counts,
                                 MatrixView<Index, Device::GPU> output,
                                 IndexingWorkspace& workspace,
                                 cudaStream_t stream);

        template <typename Scalar>
        void launchGatherRows(ConstMatrixView<Scalar, Device::GPU> input,
                              ConstMatrixView<Index, Device::GPU> indices,
                              MatrixView<Scalar, Device::GPU> output,
                              IndexingWorkspace& workspace,
                              cudaStream_t stream);

        template <typename Scalar>
        void launchScatterRows(ConstMatrixView<Scalar, Device::GPU> values,
                               ConstMatrixView<Index, Device::GPU> indices,
                               MatrixView<Scalar, Device::GPU> output,
                               IndexingWorkspace& workspace,
                               cudaStream_t stream);

        template <typename Scalar>
        void launchCompactRows(ConstMatrixView<Scalar, Device::GPU> input,
                               ConstMatrixView<std::uint8_t, Device::GPU> keep_mask,
                               MatrixView<Scalar, Device::GPU> capacity_output,
                               MatrixView<Index, Device::GPU> capacity_source_indices,
                               MatrixView<Index, Device::GPU> selected_count,
                               IndexingWorkspace& workspace,
                               cudaStream_t stream);

    } // namespace indexing_detail
} // namespace plamatrix::internal
