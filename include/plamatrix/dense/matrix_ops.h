#pragma once

#include <algorithm>
#include <cstddef>
#include <ostream>
#include <array>
#include <string>
#include <type_traits>

#include "plamatrix/dense/matrix.h"

namespace plamatrix::detail
{

    enum class MatrixBinaryOp
    {
        Add,
        Subtract,
        CwiseProduct
    };

    enum class DenseOperation
    {
        Product,
        Elementwise
    };

    struct DenseDecision
    {
        internal::Backend backend = internal::Backend::Cpu;
        std::string reason;

        bool useGpu() const noexcept
        {
            return backend != internal::Backend::Cpu;
        }
    };

    template <typename Scalar>
    DenseDecision chooseDenseBackend(DenseOperation operation,
                                     long double work,
                                     bool resident_input,
                                     internal::Backend resident_backend = internal::Backend::Cpu)
    {
        const auto settings = internal::currentExecutionSettings();
        if (settings.policy == internal::ExecutionPolicy::CpuOnly)
        {
            return {internal::Backend::Cpu, "CPU explicitly requested"};
        }
        if constexpr (!internal::detail::GpuOps<Scalar>::supported)
        {
            if (settings.policy == internal::ExecutionPolicy::GpuRequired)
            {
                throw internal::Error(internal::ErrorCode::UnsupportedOperation,
                                      "Dense GPU arithmetic does not support this scalar type",
                                      settings.preferredGpu);
            }
            return {internal::Backend::Cpu, "Dense GPU arithmetic does not support this scalar type"};
        }
        constexpr long double product_threshold = 128.0L * 1024.0L * 1024.0L;
        if (settings.policy == internal::ExecutionPolicy::Auto && !resident_input &&
            (operation != DenseOperation::Product || work < product_threshold))
        {
            return {internal::Backend::Cpu, "Estimated GPU transfer and launch cost exceeds the operation cost"};
        }

        const auto select = [](internal::Backend backend, const char* reason) -> DenseDecision
        {
            if (internal::detail::GpuOps<Scalar>::available(backend))
            {
                return {backend, reason};
            }
            return {};
        };
        if (settings.policy != internal::ExecutionPolicy::Auto)
        {
            const auto selected =
                select(settings.preferredGpu,
                       resident_input ? "Reused device-resident input" : "Preferred GPU selected for dense arithmetic");
            if (selected.useGpu())
            {
                return selected;
            }
            if (settings.policy == internal::ExecutionPolicy::GpuRequired)
            {
                throw internal::Error(internal::ErrorCode::BackendUnavailable,
                                      "Selected GPU backend is unavailable or lacks scalar support",
                                      settings.preferredGpu);
            }
            return {internal::Backend::Cpu, "Preferred GPU backend is unavailable; CPU selected"};
        }

        if (resident_backend != internal::Backend::Cpu)
        {
            const auto selected = select(resident_backend, "Reused device-resident input");
            if (selected.useGpu())
            {
                return selected;
            }
        }
        constexpr std::array priority = {internal::Backend::Cuda, internal::Backend::OpenCl, internal::Backend::Vulkan};
        for (const auto backend : priority)
        {
            const auto selected = select(backend, "Automatically selected GPU for dense arithmetic");
            if (selected.useGpu())
            {
                return selected;
            }
        }
        return {internal::Backend::Cpu, "No compatible GPU backend is available; CPU selected"};
    }

    template <typename Scalar>
    internal::ExecutionInfo gpuInfo(internal::Backend backend,
                                    std::string_view reason,
                                    bool left_resident,
                                    bool right_resident,
                                    std::size_t left_bytes,
                                    std::size_t right_bytes)
    {
        internal::ExecutionInfo info;
        info.backend = backend;
        info.bytesUploaded = (left_resident ? 0 : left_bytes) + (right_resident ? 0 : right_bytes);
        info.reusedDeviceInput = left_resident || right_resident;
        info.reason = reason;
        return info;
    }

    template <typename Scalar>
    void genericGemm(const Scalar* left, const Scalar* right, Scalar* output, Index rows, Index columns, Index inner)
    {
        for (Index column = 0; column < columns; ++column)
        {
            for (Index row = 0; row < rows; ++row)
            {
                Scalar sum{};
                for (Index index = 0; index < inner; ++index)
                {
                    sum += left[row + index * rows] * right[index + column * inner];
                }
                output[row + column * rows] = sum;
            }
        }
    }

} // namespace plamatrix::detail

namespace plamatrix::v1
{

    template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    template <detail::MatrixBinaryOp Op>
    Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>
    Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>::applyBinary(const Matrix& other) const
    {
        if (_rows != other._rows || _cols != other._cols)
        {
            throw internal::Error(internal::ErrorCode::InvalidArgument,
                                  "Elementwise matrix operation requires equal shapes");
        }
        Matrix result(_rows, _cols);
        const bool left_resident = isDeviceResident();
        const bool right_resident = other.isDeviceResident();
        const auto choice =
            row_major_storage
                ? detail::DenseDecision{internal::Backend::Cpu, "RowMajor dense arithmetic uses the CPU backend"}
                : detail::chooseDenseBackend<Scalar>(detail::DenseOperation::Elementwise,
                                                     static_cast<long double>(size()),
                                                     left_resident || right_resident,
                                                     left_resident ? deviceBackend() : other.deviceBackend());
        if constexpr (!row_major_storage)
        {
            if (choice.useGpu() && size() != 0)
            {
                std::shared_ptr<internal::detail::GpuStorage<Scalar>> gpu_result;
                if constexpr (Op == detail::MatrixBinaryOp::Add)
                {
                    gpu_result = internal::detail::GpuOps<Scalar>::add(gpuStorage(choice.backend),
                                                                       other.gpuStorage(choice.backend));
                }
                else if constexpr (Op == detail::MatrixBinaryOp::Subtract)
                {
                    gpu_result = internal::detail::GpuOps<Scalar>::subtract(gpuStorage(choice.backend),
                                                                            other.gpuStorage(choice.backend));
                }
                else
                {
                    gpu_result = internal::detail::GpuOps<Scalar>::cwiseProduct(gpuStorage(choice.backend),
                                                                                other.gpuStorage(choice.backend));
                }
                result.adoptGpu(std::move(gpu_result),
                                detail::gpuInfo<Scalar>(choice.backend,
                                                        choice.reason,
                                                        left_resident,
                                                        right_resident,
                                                        static_cast<std::size_t>(size()) * sizeof(Scalar),
                                                        static_cast<std::size_t>(other.size()) * sizeof(Scalar)));
                return result;
            }
        }
        result._info.bytesDownloaded = pendingDownloadBytes() + other.pendingDownloadBytes();
        const Scalar* left = data();
        const Scalar* right = other.data();
        Scalar* output = result.data();
#pragma omp simd
        for (Index index = 0; index < size(); ++index)
        {
            if constexpr (Op == detail::MatrixBinaryOp::Add)
            {
                output[index] = left[index] + right[index];
            }
            else if constexpr (Op == detail::MatrixBinaryOp::Subtract)
            {
                output[index] = left[index] - right[index];
            }
            else
            {
                output[index] = left[index] * right[index];
            }
        }
        result._info.reason = choice.reason;
        return result;
    }

    template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>
    Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>::add(const Matrix& other) const
    {
        return applyBinary<detail::MatrixBinaryOp::Add>(other);
    }

    template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>
    Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>::subtract(const Matrix& other) const
    {
        return applyBinary<detail::MatrixBinaryOp::Subtract>(other);
    }

    template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>
    Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>::cwiseProductEager(const Matrix& other) const
    {
        return applyBinary<detail::MatrixBinaryOp::CwiseProduct>(other);
    }

    template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>
    Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>::scaled(Scalar value) const
    {
        Matrix result(_rows, _cols);
        const bool resident = isDeviceResident();
        const auto choice =
            row_major_storage
                ? detail::DenseDecision{internal::Backend::Cpu, "RowMajor dense arithmetic uses the CPU backend"}
                : detail::chooseDenseBackend<Scalar>(
                      detail::DenseOperation::Elementwise, static_cast<long double>(size()), resident, deviceBackend());
        if constexpr (!row_major_storage)
        {
            if (choice.useGpu() && size() != 0)
            {
                auto gpu_result = internal::detail::GpuOps<Scalar>::scale(gpuStorage(choice.backend), value);
                result.adoptGpu(std::move(gpu_result),
                                detail::gpuInfo<Scalar>(choice.backend,
                                                        choice.reason,
                                                        resident,
                                                        true,
                                                        static_cast<std::size_t>(size()) * sizeof(Scalar),
                                                        0));
                return result;
            }
        }
        result._info.bytesDownloaded = pendingDownloadBytes();
        const Scalar* source = data();
        Scalar* output = result.data();
#pragma omp simd
        for (Index index = 0; index < size(); ++index)
        {
            output[index] = source[index] * value;
        }
        result._info.reason = choice.reason;
        return result;
    }

    template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    Matrix<Scalar, Cols, Rows> Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>::transposeMaterialized() const
    {
        Matrix<Scalar, Cols, Rows> result(_cols, _rows);
        const bool resident = isDeviceResident();
        const auto choice =
            row_major_storage
                ? detail::DenseDecision{internal::Backend::Cpu, "RowMajor transpose uses the CPU backend"}
                : detail::chooseDenseBackend<Scalar>(
                      detail::DenseOperation::Elementwise, static_cast<long double>(size()), resident, deviceBackend());
        if constexpr (!row_major_storage)
        {
            if (choice.useGpu() && size() != 0)
            {
                auto gpu_result = internal::detail::GpuOps<Scalar>::transpose(gpuStorage(choice.backend));
                result.adoptGpu(std::move(gpu_result),
                                detail::gpuInfo<Scalar>(choice.backend,
                                                        choice.reason,
                                                        resident,
                                                        true,
                                                        static_cast<std::size_t>(size()) * sizeof(Scalar),
                                                        0));
                return result;
            }
        }
        result._info.bytesDownloaded = pendingDownloadBytes();
        for (Index column = 0; column < _cols; ++column)
        {
            for (Index row = 0; row < _rows; ++row)
            {
                result(column, row) = (*this)(row, column);
            }
        }
        result._info.reason = choice.reason;
        return result;
    }

    template <typename Scalar,
              int LeftRows,
              int LeftCols,
              int LeftOptions,
              int LeftMaxRows,
              int LeftMaxCols,
              int RightRows,
              int RightCols,
              int RightOptions,
              int RightMaxRows,
              int RightMaxCols>
    Matrix<Scalar, LeftRows, RightCols>
    multiplyEager(const Matrix<Scalar, LeftRows, LeftCols, LeftOptions, LeftMaxRows, LeftMaxCols>& left,
                  const Matrix<Scalar, RightRows, RightCols, RightOptions, RightMaxRows, RightMaxCols>& right)
    {
        static_assert(LeftCols == Dynamic || RightRows == Dynamic || LeftCols == RightRows,
                      "Fixed matrix multiplication dimensions do not match");
        if (left.cols() != right.rows())
        {
            throw internal::Error(internal::ErrorCode::InvalidArgument,
                                  "Matrix multiplication dimensions do not match");
        }
        Matrix<Scalar, LeftRows, RightCols> result(left.rows(), right.cols());
        const bool left_resident = left.isDeviceResident();
        const bool right_resident = right.isDeviceResident();
        const long double work = static_cast<long double>(left.rows()) * right.cols() * left.cols();
        constexpr bool column_major_inputs =
            Matrix<Scalar, LeftRows, LeftCols, LeftOptions, LeftMaxRows, LeftMaxCols>::IsRowMajor == 0 &&
            Matrix<Scalar, RightRows, RightCols, RightOptions, RightMaxRows, RightMaxCols>::IsRowMajor == 0;
        const auto choice =
            column_major_inputs
                ? detail::chooseDenseBackend<Scalar>(detail::DenseOperation::Product,
                                                     work,
                                                     left_resident || right_resident,
                                                     left_resident ? left.deviceBackend() : right.deviceBackend())
                : detail::DenseDecision{internal::Backend::Cpu, "RowMajor matrix products use the CPU backend"};
        if constexpr (column_major_inputs)
        {
            if (choice.useGpu() && result.size() != 0 && left.cols() != 0)
            {
                auto gpu_result = internal::detail::GpuOps<Scalar>::multiply(left.gpuStorage(choice.backend),
                                                                             right.gpuStorage(choice.backend));
                result.adoptGpu(std::move(gpu_result),
                                detail::gpuInfo<Scalar>(choice.backend,
                                                        choice.reason,
                                                        left_resident,
                                                        right_resident,
                                                        static_cast<std::size_t>(left.size()) * sizeof(Scalar),
                                                        static_cast<std::size_t>(right.size()) * sizeof(Scalar)));
                return result;
            }
        }
        result._info.bytesDownloaded = left.pendingDownloadBytes() + right.pendingDownloadBytes();
        if (result.size() != 0 && left.cols() != 0)
        {
            if constexpr (column_major_inputs && (std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double>))
            {
                internal::detail::cpuGemm(
                    left.data(), right.data(), result.data(), left.rows(), right.cols(), left.cols());
            }
            else
            {
                for (Index col = 0; col < right.cols(); ++col)
                {
                    for (Index row = 0; row < left.rows(); ++row)
                    {
                        Scalar sum{};
                        for (Index inner = 0; inner < left.cols(); ++inner)
                        {
                            sum += left(row, inner) * right(inner, col);
                        }
                        result(row, col) = sum;
                    }
                }
            }
        }
        result._info.reason = choice.reason;
        return result;
    }

    /// Writes a matrix by logical rows, independent of its storage order.
    template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    std::ostream& operator<<(std::ostream& output, const Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>& matrix)
    {
        for (Index row = 0; row < matrix.rows(); ++row)
        {
            if (row != 0)
            {
                output << '\n';
            }
            for (Index col = 0; col < matrix.cols(); ++col)
            {
                if (col != 0)
                {
                    output << ' ';
                }
                output << matrix(row, col);
            }
        }
        return output;
    }

} // namespace plamatrix::v1
