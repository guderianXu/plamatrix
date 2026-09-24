#pragma once

#include <cstdint>
#include <limits>
#include <vector>

#include "plamatrix/internal/core/error.h"
#include "plamatrix/internal/core/execution_context.h"
#include "plamatrix/internal/core/memory_space.h"
#include "plamatrix/internal/device/device_matrix.h"
#include "plamatrix/internal/device/detail/resident_buffer.h"
#include "plamatrix/internal/sparse/csr_storage.h"
#include "plamatrix/sparse/sparse_matrix.h"

namespace plamatrix::internal
{

    inline namespace v1
    {

        enum class CsrIndexType
        {
            Int32,
            Int64
        };

        enum class CsrIndexBase
        {
            Zero,
            One
        };

        enum class CsrSortedness
        {
            Unknown,
            Sorted
        };

        enum class CsrDuplicatePolicy
        {
            Allow,
            Reject,
            Sum
        };

        struct CsrStructureOptions
        {
            CsrIndexType indexType = CsrIndexType::Int64;
            CsrIndexBase indexBase = CsrIndexBase::Zero;
            CsrSortedness sortedness = CsrSortedness::Unknown;
            CsrDuplicatePolicy duplicatePolicy = CsrDuplicatePolicy::Allow;
        };

        template <typename Scalar> class ResidentCsrMatrix
        {
        public:
            /// Upload Eigen-style sparse storage in canonical sorted, zero-based CSR form.
            template <int Options, typename StorageIndex>
            static ResidentCsrMatrix copyFrom(const SparseMatrix<Scalar, Options, StorageIndex>& host,
                                              ExecutionContext& context)
            {
                CsrStructureOptions options;
                options.sortedness = CsrSortedness::Sorted;
                options.duplicatePolicy = CsrDuplicatePolicy::Reject;
                return copyFrom(SparseAccess::csrSnapshot(host), context, options);
            }

            ResidentCsrMatrix(
                Index rows, Index cols, Index nnz, ExecutionContext& context, CsrStructureOptions options = {})
                : _rows(rows), _cols(cols), _context(&context), _options(options), _values(checkedNnz(nnz), context),
                  _colIndices(checkedNnz(nnz), context), _rowOffsets(checkedRows(rows), context)
            {
                if (rows < 0 || cols < 0 || nnz < 0)
                {
                    throw std::invalid_argument("ResidentCsrMatrix dimensions must be non-negative");
                }
            }

            static ResidentCsrMatrix copyFrom(const CsrStorage<Scalar, Device::CPU>& host,
                                              ExecutionContext& context,
                                              CsrStructureOptions options = {})
            {
                host.validateStructure();
                ResidentCsrMatrix result(host.rows(), host.cols(), host.nnz(), context, options);
                result._values.copyFromHost(host.values(), static_cast<std::size_t>(host.nnz()));
                result._colIndices.copyFromHost(host.colIndices(), static_cast<std::size_t>(host.nnz()));
                result._rowOffsets.copyFromHost(host.rowOffsets(), static_cast<std::size_t>(host.rows()) + 1);
                if (context.backend() == Backend::Vulkan)
                {
                    if (host.rows() > std::numeric_limits<std::uint32_t>::max() ||
                        host.cols() > std::numeric_limits<std::uint32_t>::max() ||
                        host.nnz() > std::numeric_limits<std::uint32_t>::max())
                    {
                        throw Error(ErrorCode::InvalidArgument,
                                    "Vulkan CSR dimensions exceed the shader uint32 range",
                                    Backend::Vulkan);
                    }
                    std::vector<std::uint32_t> columns(static_cast<std::size_t>(host.nnz()));
                    std::vector<std::uint32_t> rows(static_cast<std::size_t>(host.rows()) + 1);
                    for (std::size_t index = 0; index < columns.size(); ++index)
                    {
                        columns[index] = static_cast<std::uint32_t>(host.colIndices()[index]);
                    }
                    for (std::size_t index = 0; index < rows.size(); ++index)
                    {
                        rows[index] = static_cast<std::uint32_t>(host.rowOffsets()[index]);
                    }
                    result._vulkanColumns32 = resident_detail::ResidentBuffer<std::uint32_t>(columns.size(), context);
                    result._vulkanRows32 = resident_detail::ResidentBuffer<std::uint32_t>(rows.size(), context);
                    result._vulkanColumns32.copyFromHost(columns.data(), columns.size());
                    result._vulkanRows32.copyFromHost(rows.data(), rows.size());
                }
                return result;
            }

            ResidentCsrMatrix(ResidentCsrMatrix&&) noexcept = default;
            ResidentCsrMatrix& operator=(ResidentCsrMatrix&&) noexcept = default;
            ResidentCsrMatrix(const ResidentCsrMatrix&) = delete;
            ResidentCsrMatrix& operator=(const ResidentCsrMatrix&) = delete;

            Index rows() const noexcept
            {
                return _rows;
            }
            Index cols() const noexcept
            {
                return _cols;
            }
            Index nnz() const noexcept
            {
                return static_cast<Index>(_values.size());
            }
            const Scalar* values() const
            {
                return _values.data();
            }
            Scalar* values()
            {
                return _values.data();
            }
            const Index* colIndices() const
            {
                return _colIndices.data();
            }
            Index* colIndices()
            {
                return _colIndices.data();
            }
            const Index* rowOffsets() const
            {
                return _rowOffsets.data();
            }
            Index* rowOffsets()
            {
                return _rowOffsets.data();
            }

            void copyToHost(Scalar* values, Index* col_indices, Index* row_offsets) const
            {
                _values.copyToHost(values, static_cast<std::size_t>(nnz()));
                _colIndices.copyToHost(col_indices, static_cast<std::size_t>(nnz()));
                _rowOffsets.copyToHost(row_offsets, static_cast<std::size_t>(_rows) + 1);
            }
            ExecutionContext& context() const noexcept
            {
                return *_context;
            }
            MemorySpace memorySpace() const noexcept
            {
                return memorySpaceFor(_context->backend());
            }
            const CsrStructureOptions& structureOptions() const noexcept
            {
                return _options;
            }
            std::uint64_t structureRevision() const noexcept
            {
                return _structureRevision;
            }
            bool analysisValid() const noexcept
            {
                return _analysisValid;
            }

            void validateContext(const ExecutionContext& context) const
            {
                if (_context != &context)
                {
                    throw Error(ErrorCode::InvalidState,
                                "ResidentCsrMatrix belongs to a different ExecutionContext",
                                context.backend());
                }
            }

            void markStructureModified() noexcept
            {
                ++_structureRevision;
                _analysisValid = false;
            }

        private:
            friend struct opencl::NativeAccess;
            friend struct vulkan::NativeAccess;
            static std::size_t checkedNnz(Index nnz)
            {
                if (nnz < 0)
                {
                    throw std::invalid_argument("ResidentCsrMatrix nnz must be non-negative");
                }
                return static_cast<std::size_t>(nnz);
            }

            static std::size_t checkedRows(Index rows)
            {
                if (rows < 0 || rows == std::numeric_limits<Index>::max())
                {
                    throw std::invalid_argument("ResidentCsrMatrix rows are invalid");
                }
                return static_cast<std::size_t>(rows + 1);
            }

            Index _rows;
            Index _cols;
            ExecutionContext* _context;
            CsrStructureOptions _options;
            resident_detail::ResidentBuffer<Scalar> _values;
            resident_detail::ResidentBuffer<Index> _colIndices;
            resident_detail::ResidentBuffer<Index> _rowOffsets;
            resident_detail::ResidentBuffer<std::uint32_t> _vulkanColumns32;
            resident_detail::ResidentBuffer<std::uint32_t> _vulkanRows32;
            std::uint64_t _structureRevision = 0;
            bool _analysisValid = true;
        };

    } // namespace v1

} // namespace plamatrix::internal
