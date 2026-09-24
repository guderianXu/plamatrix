#pragma once

#include <memory>
#include <utility>

#include "plamatrix/internal/core/error.h"
#include "plamatrix/internal/core/execution_context.h"
#include "plamatrix/internal/core/memory_space.h"
#include "plamatrix/internal/core/shape.h"
#include "plamatrix/dense/matrix.h"
#include "plamatrix/internal/dense/matrix_view.h"
#include "plamatrix/internal/device/detail/resident_buffer.h"

namespace plamatrix::internal
{

    inline namespace v1
    {

        inline MemorySpace memorySpaceFor(Backend backend)
        {
            switch (backend)
            {
            case Backend::Cpu:
                return MemorySpace::Host;
            case Backend::Cuda:
                return MemorySpace::Cuda;
            case Backend::OpenCl:
                return MemorySpace::OpenCl;
            case Backend::Vulkan:
                return MemorySpace::Vulkan;
            }
            throw Error(ErrorCode::UnsupportedBackend, "Unknown PlaMatrix backend", backend);
        }

        template <typename Scalar> class ResidentMatrix
        {
        public:
            template <int Rows, int Cols>
            static ResidentMatrix copyFrom(const Matrix<Scalar, Rows, Cols>& host, ExecutionContext& context)
            {
                ResidentMatrix result(host.rows(), host.cols(), context);
                result.copyFromHost(host.data(), static_cast<std::size_t>(host.size()));
                return result;
            }

            template <int Rows, int Cols>
            static ResidentMatrix copyFrom(const Matrix<Scalar, Rows, Cols>& host,
                                           std::shared_ptr<ExecutionContext> context)
            {
                ResidentMatrix result(host.rows(), host.cols(), std::move(context));
                result.copyFromHost(host.data(), static_cast<std::size_t>(host.size()));
                return result;
            }

            ResidentMatrix(Index rows, Index cols, ExecutionContext& context)
                : _shape(rows, cols), _context(&context), _values(_shape.elementCount(), context)
            {
            }

            /// Retain the context when returning resident storage independently of its producer.
            ResidentMatrix(Index rows, Index cols, std::shared_ptr<ExecutionContext> context)
                : _owner(std::move(context)), _shape(rows, cols), _context(&requireContext(_owner)),
                  _values(_shape.elementCount(), *_context)
            {
            }

            ResidentMatrix(ResidentMatrix&&) noexcept = default;
            ResidentMatrix& operator=(ResidentMatrix&& other) noexcept
            {
                if (this != &other)
                {
                    ResidentMatrix moved(std::move(other));
                    swap(moved);
                }
                return *this;
            }
            ResidentMatrix(const ResidentMatrix&) = delete;
            ResidentMatrix& operator=(const ResidentMatrix&) = delete;

            void swap(ResidentMatrix& other) noexcept
            {
                std::swap(_shape, other._shape);
                std::swap(_context, other._context);
                std::swap(_values, other._values);
                _owner.swap(other._owner);
            }

            Index rows() const noexcept
            {
                return _shape.rows;
            }
            Index cols() const noexcept
            {
                return _shape.cols;
            }
            std::size_t size() const noexcept
            {
                return _values.size();
            }
            Scalar* data()
            {
                return _values.data();
            }
            const Scalar* data() const
            {
                return _values.data();
            }

            /// Borrow a contiguous CPU or CUDA view; reject a mismatched backend.
            template <Device Dev> MatrixView<Scalar, Dev> view()
            {
                validateViewBackend<Dev>();
                return MatrixView<Scalar, Dev>(data(), rows(), cols(), 1, rows());
            }

            template <Device Dev> ConstMatrixView<Scalar, Dev> view() const
            {
                validateViewBackend<Dev>();
                return ConstMatrixView<Scalar, Dev>(data(), rows(), cols(), 1, rows());
            }

            /// Deep copy in the same context, without a host staging allocation.
            ResidentMatrix clone() const
            {
                ResidentMatrix result =
                    _owner ? ResidentMatrix(rows(), cols(), _owner) : ResidentMatrix(rows(), cols(), context());
                result.copyFromResident(*this);
                return result;
            }

            /// Synchronous copy; both matrices must have identical shape and context.
            void copyFromResident(const ResidentMatrix& source)
            {
                source.validateContext(context());
                if (rows() != source.rows() || cols() != source.cols())
                {
                    throw Error(ErrorCode::InvalidArgument, "ResidentMatrix copy shape mismatch", context().backend());
                }
                _values.copyFromResident(source._values);
            }

            void copyFromHost(const Scalar* source, std::size_t count)
            {
                _values.copyFromHost(source, count);
            }

            void copyToHost(Scalar* destination, std::size_t count) const
            {
                _values.copyToHost(destination, count);
            }

            Matrix<Scalar, Dynamic, Dynamic> toHostMatrix() const
            {
                Matrix<Scalar, Dynamic, Dynamic> result(rows(), cols());
                copyToHost(result.data(), size());
                return result;
            }
            ExecutionContext& context() const noexcept
            {
                return *_context;
            }
            /// Shared ownership for independently returned results; empty for an explicitly borrowed context.
            std::shared_ptr<ExecutionContext> contextOwner() const noexcept
            {
                return _owner;
            }
            MemorySpace memorySpace() const noexcept
            {
                return memorySpaceFor(_context->backend());
            }

            void validateContext(const ExecutionContext& context) const
            {
                if (_context != &context)
                {
                    throw Error(ErrorCode::InvalidState,
                                "ResidentMatrix belongs to a different ExecutionContext",
                                context.backend());
                }
            }

        private:
            static ExecutionContext& requireContext(const std::shared_ptr<ExecutionContext>& context)
            {
                if (!context)
                {
                    throw Error(ErrorCode::InvalidArgument, "ResidentMatrix requires a non-null execution context");
                }
                return *context;
            }

            template <Device Dev> void validateViewBackend() const
            {
                constexpr Backend expected = Dev == Device::CPU ? Backend::Cpu : Backend::Cuda;
                if (context().backend() != expected)
                {
                    throw Error(ErrorCode::InvalidArgument,
                                "ResidentMatrix view device does not match its backend",
                                context().backend());
                }
            }

            friend struct opencl::NativeAccess;
            friend struct vulkan::NativeAccess;
            // Declared first so device buffers are released before their owning context.
            std::shared_ptr<ExecutionContext> _owner;
            Shape2 _shape;
            ExecutionContext* _context;
            resident_detail::ResidentBuffer<Scalar> _values;
        };

    } // namespace v1

} // namespace plamatrix::internal
