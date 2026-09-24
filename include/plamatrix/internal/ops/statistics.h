#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "plamatrix/internal/ops/reduction.h"

namespace plamatrix::internal
{

    /**
     * @brief Return the median of finite samples, ignoring NaN and infinities.
     *
     * The input is passed by value for in-place selection. An input without a
     * finite sample returns std::nullopt.
     */
    template <typename Scalar> std::optional<Scalar> finiteMedian(std::vector<Scalar> values)
    {
        static_assert(std::is_floating_point_v<Scalar>, "finiteMedian requires a floating-point scalar");
        values.erase(
            std::remove_if(values.begin(), values.end(), [](const Scalar value) { return !std::isfinite(value); }),
            values.end());
        if (values.empty())
        {
            return std::nullopt;
        }

        const std::size_t middle = values.size() / 2;
        std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(middle), values.end());
        if (values.size() % 2 == 1)
        {
            return values[middle];
        }

        const Scalar lower = *std::max_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(middle));
        const Scalar upper = values[middle];
        if (std::signbit(lower) != std::signbit(upper))
        {
            return lower / Scalar(2) + upper / Scalar(2);
        }
        return lower + (upper - lower) / Scalar(2);
    }

    /// Per-column bounds over rows whose every element is finite.
    /// minimum and maximum have shape 1 x input.cols(); validRowCount has shape 1 x 1.
    /// When no row is valid, each bound is NaN and validRowCount is zero. For an N x 0
    /// input every row is vacuously finite, the bound matrices have shape 1 x 0, and the
    /// count is N.
    template <typename Scalar, Device Dev> struct FiniteColumnBoundsResult
    {
        DenseStorage<Scalar, Dev> minimum;
        DenseStorage<Scalar, Dev> maximum;
        DenseStorage<Index, Dev> validRowCount;
    };

    /// Finite-row mask and per-column bounds produced from the same mask computation.
    /// rowMask has shape input.rows() x 1; all other field semantics match
    /// FiniteColumnBoundsResult.
    template <typename Scalar, Device Dev> struct FiniteColumnBoundsWithMaskResult
    {
        DenseStorage<std::uint8_t, Dev> rowMask;
        DenseStorage<Scalar, Dev> minimum;
        DenseStorage<Scalar, Dev> maximum;
        DenseStorage<Index, Dev> validRowCount;
    };

    /// Return an input.rows() x 1 byte mask. A row is one iff every element is finite.
    template <typename Scalar>
    DenseStorage<std::uint8_t, Device::CPU> finiteRowMask(const DenseStorage<Scalar, Device::CPU>& input);

    /// Return per-column bounds after skipping rows containing NaN or infinity.
    template <typename Scalar>
    FiniteColumnBoundsResult<Scalar, Device::CPU> finiteColumnBounds(const DenseStorage<Scalar, Device::CPU>& input);

    /// Return the finite-row mask and per-column bounds without recomputing the mask.
    template <typename Scalar>
    FiniteColumnBoundsWithMaskResult<Scalar, Device::CPU>
    finiteColumnBoundsWithMask(const DenseStorage<Scalar, Device::CPU>& input);

#ifdef PLAMATRIX_WITH_CUDA

    template <typename Scalar>
    DenseStorage<std::uint8_t, Device::GPU> finiteRowMask(const DenseStorage<Scalar, Device::GPU>& input);

    template <typename Scalar>
    DenseStorage<std::uint8_t, Device::GPU> finiteRowMask(const DenseStorage<Scalar, Device::GPU>& input,
                                                         ReductionWorkspace& workspace,
                                                         cudaStream_t stream = nullptr);

    template <typename Scalar>
    void finiteRowMask(const DenseStorage<Scalar, Device::GPU>& input,
                       DenseStorage<std::uint8_t, Device::GPU>& output,
                       ReductionWorkspace& workspace,
                       cudaStream_t stream = nullptr);

    template <typename Scalar>
    DenseStorage<std::uint8_t, Device::GPU> finiteRowMaskAsync(const DenseStorage<Scalar, Device::GPU>& input,
                                                              ReductionWorkspace& workspace,
                                                              cudaStream_t stream);

    template <typename Scalar>
    void finiteRowMaskAsync(const DenseStorage<Scalar, Device::GPU>& input,
                            DenseStorage<std::uint8_t, Device::GPU>& output,
                            ReductionWorkspace& workspace,
                            cudaStream_t stream);

    template <typename Scalar>
    FiniteColumnBoundsResult<Scalar, Device::GPU> finiteColumnBounds(const DenseStorage<Scalar, Device::GPU>& input);

    template <typename Scalar>
    FiniteColumnBoundsResult<Scalar, Device::GPU> finiteColumnBounds(const DenseStorage<Scalar, Device::GPU>& input,
                                                                     ReductionWorkspace& workspace,
                                                                     cudaStream_t stream = nullptr);

    template <typename Scalar>
    void finiteColumnBounds(const DenseStorage<Scalar, Device::GPU>& input,
                            DenseStorage<Scalar, Device::GPU>& minimum,
                            DenseStorage<Scalar, Device::GPU>& maximum,
                            DenseStorage<Index, Device::GPU>& valid_row_count,
                            ReductionWorkspace& workspace,
                            cudaStream_t stream = nullptr);

    template <typename Scalar>
    FiniteColumnBoundsResult<Scalar, Device::GPU> finiteColumnBoundsAsync(const DenseStorage<Scalar, Device::GPU>& input,
                                                                          ReductionWorkspace& workspace,
                                                                          cudaStream_t stream);

    template <typename Scalar>
    void finiteColumnBoundsAsync(const DenseStorage<Scalar, Device::GPU>& input,
                                 DenseStorage<Scalar, Device::GPU>& minimum,
                                 DenseStorage<Scalar, Device::GPU>& maximum,
                                 DenseStorage<Index, Device::GPU>& valid_row_count,
                                 ReductionWorkspace& workspace,
                                 cudaStream_t stream);

    template <typename Scalar>
    FiniteColumnBoundsWithMaskResult<Scalar, Device::GPU>
    finiteColumnBoundsWithMask(const DenseStorage<Scalar, Device::GPU>& input);

    template <typename Scalar>
    FiniteColumnBoundsWithMaskResult<Scalar, Device::GPU> finiteColumnBoundsWithMask(
        const DenseStorage<Scalar, Device::GPU>& input, ReductionWorkspace& workspace, cudaStream_t stream = nullptr);

    /// Compute mask, bounds, and count into caller-owned outputs, then synchronize stream.
    template <typename Scalar>
    void finiteColumnBoundsWithMask(const DenseStorage<Scalar, Device::GPU>& input,
                                    DenseStorage<std::uint8_t, Device::GPU>& row_mask,
                                    DenseStorage<Scalar, Device::GPU>& minimum,
                                    DenseStorage<Scalar, Device::GPU>& maximum,
                                    DenseStorage<Index, Device::GPU>& valid_row_count,
                                    ReductionWorkspace& workspace,
                                    cudaStream_t stream = nullptr);

    template <typename Scalar>
    FiniteColumnBoundsWithMaskResult<Scalar, Device::GPU> finiteColumnBoundsWithMaskAsync(
        const DenseStorage<Scalar, Device::GPU>& input, ReductionWorkspace& workspace, cudaStream_t stream);

    /// Enqueue mask, bounds, and count into caller-owned outputs without synchronizing stream.
    template <typename Scalar>
    void finiteColumnBoundsWithMaskAsync(const DenseStorage<Scalar, Device::GPU>& input,
                                         DenseStorage<std::uint8_t, Device::GPU>& row_mask,
                                         DenseStorage<Scalar, Device::GPU>& minimum,
                                         DenseStorage<Scalar, Device::GPU>& maximum,
                                         DenseStorage<Index, Device::GPU>& valid_row_count,
                                         ReductionWorkspace& workspace,
                                         cudaStream_t stream);

#else

    namespace statistics_detail
    {
        [[noreturn]] inline void throwNoCuda(const char* operation)
        {
            throw std::runtime_error(std::string(operation) + " requires PLAMATRIX_WITH_CUDA=ON");
        }
    } // namespace statistics_detail

#define PLAMATRIX_DEFINE_NO_CUDA_FINITE_STATISTICS(Scalar)                                                             \
    inline DenseStorage<std::uint8_t, Device::GPU> finiteRowMask(const DenseStorage<Scalar, Device::GPU>&)               \
    {                                                                                                                  \
        statistics_detail::throwNoCuda("finiteRowMask");                                                               \
    }                                                                                                                  \
    inline DenseStorage<std::uint8_t, Device::GPU> finiteRowMask(                                                       \
        const DenseStorage<Scalar, Device::GPU>&, ReductionWorkspace&, cudaStream_t = nullptr)                          \
    {                                                                                                                  \
        statistics_detail::throwNoCuda("finiteRowMask");                                                               \
    }                                                                                                                  \
    inline void finiteRowMask(const DenseStorage<Scalar, Device::GPU>&,                                                 \
                              DenseStorage<std::uint8_t, Device::GPU>&,                                                 \
                              ReductionWorkspace&,                                                                     \
                              cudaStream_t = nullptr)                                                                  \
    {                                                                                                                  \
        statistics_detail::throwNoCuda("finiteRowMask");                                                               \
    }                                                                                                                  \
    inline DenseStorage<std::uint8_t, Device::GPU> finiteRowMaskAsync(                                                  \
        const DenseStorage<Scalar, Device::GPU>&, ReductionWorkspace&, cudaStream_t)                                    \
    {                                                                                                                  \
        statistics_detail::throwNoCuda("finiteRowMaskAsync");                                                          \
    }                                                                                                                  \
    inline void finiteRowMaskAsync(const DenseStorage<Scalar, Device::GPU>&,                                            \
                                   DenseStorage<std::uint8_t, Device::GPU>&,                                            \
                                   ReductionWorkspace&,                                                                \
                                   cudaStream_t)                                                                       \
    {                                                                                                                  \
        statistics_detail::throwNoCuda("finiteRowMaskAsync");                                                          \
    }                                                                                                                  \
    inline FiniteColumnBoundsResult<Scalar, Device::GPU> finiteColumnBounds(const DenseStorage<Scalar, Device::GPU>&)   \
    {                                                                                                                  \
        statistics_detail::throwNoCuda("finiteColumnBounds");                                                          \
    }                                                                                                                  \
    inline FiniteColumnBoundsResult<Scalar, Device::GPU> finiteColumnBounds(                                           \
        const DenseStorage<Scalar, Device::GPU>&, ReductionWorkspace&, cudaStream_t = nullptr)                          \
    {                                                                                                                  \
        statistics_detail::throwNoCuda("finiteColumnBounds");                                                          \
    }                                                                                                                  \
    inline void finiteColumnBounds(const DenseStorage<Scalar, Device::GPU>&,                                            \
                                   DenseStorage<Scalar, Device::GPU>&,                                                  \
                                   DenseStorage<Scalar, Device::GPU>&,                                                  \
                                   DenseStorage<Index, Device::GPU>&,                                                   \
                                   ReductionWorkspace&,                                                                \
                                   cudaStream_t = nullptr)                                                             \
    {                                                                                                                  \
        statistics_detail::throwNoCuda("finiteColumnBounds");                                                          \
    }                                                                                                                  \
    inline FiniteColumnBoundsResult<Scalar, Device::GPU> finiteColumnBoundsAsync(                                      \
        const DenseStorage<Scalar, Device::GPU>&, ReductionWorkspace&, cudaStream_t)                                    \
    {                                                                                                                  \
        statistics_detail::throwNoCuda("finiteColumnBoundsAsync");                                                     \
    }                                                                                                                  \
    inline void finiteColumnBoundsAsync(const DenseStorage<Scalar, Device::GPU>&,                                       \
                                        DenseStorage<Scalar, Device::GPU>&,                                             \
                                        DenseStorage<Scalar, Device::GPU>&,                                             \
                                        DenseStorage<Index, Device::GPU>&,                                              \
                                        ReductionWorkspace&,                                                           \
                                        cudaStream_t)                                                                  \
    {                                                                                                                  \
        statistics_detail::throwNoCuda("finiteColumnBoundsAsync");                                                     \
    }                                                                                                                  \
    inline FiniteColumnBoundsWithMaskResult<Scalar, Device::GPU> finiteColumnBoundsWithMask(                           \
        const DenseStorage<Scalar, Device::GPU>&)                                                                       \
    {                                                                                                                  \
        statistics_detail::throwNoCuda("finiteColumnBoundsWithMask");                                                  \
    }                                                                                                                  \
    inline FiniteColumnBoundsWithMaskResult<Scalar, Device::GPU> finiteColumnBoundsWithMask(                           \
        const DenseStorage<Scalar, Device::GPU>&, ReductionWorkspace&, cudaStream_t = nullptr)                          \
    {                                                                                                                  \
        statistics_detail::throwNoCuda("finiteColumnBoundsWithMask");                                                  \
    }                                                                                                                  \
    inline void finiteColumnBoundsWithMask(const DenseStorage<Scalar, Device::GPU>&,                                    \
                                           DenseStorage<std::uint8_t, Device::GPU>&,                                    \
                                           DenseStorage<Scalar, Device::GPU>&,                                          \
                                           DenseStorage<Scalar, Device::GPU>&,                                          \
                                           DenseStorage<Index, Device::GPU>&,                                           \
                                           ReductionWorkspace&,                                                        \
                                           cudaStream_t = nullptr)                                                     \
    {                                                                                                                  \
        statistics_detail::throwNoCuda("finiteColumnBoundsWithMask");                                                  \
    }                                                                                                                  \
    inline FiniteColumnBoundsWithMaskResult<Scalar, Device::GPU> finiteColumnBoundsWithMaskAsync(                      \
        const DenseStorage<Scalar, Device::GPU>&, ReductionWorkspace&, cudaStream_t)                                    \
    {                                                                                                                  \
        statistics_detail::throwNoCuda("finiteColumnBoundsWithMaskAsync");                                             \
    }                                                                                                                  \
    inline void finiteColumnBoundsWithMaskAsync(const DenseStorage<Scalar, Device::GPU>&,                               \
                                                DenseStorage<std::uint8_t, Device::GPU>&,                               \
                                                DenseStorage<Scalar, Device::GPU>&,                                     \
                                                DenseStorage<Scalar, Device::GPU>&,                                     \
                                                DenseStorage<Index, Device::GPU>&,                                      \
                                                ReductionWorkspace&,                                                   \
                                                cudaStream_t)                                                          \
    {                                                                                                                  \
        statistics_detail::throwNoCuda("finiteColumnBoundsWithMaskAsync");                                             \
    }

#ifdef PLAMATRIX_USE_FLOAT
    PLAMATRIX_DEFINE_NO_CUDA_FINITE_STATISTICS(float)
#endif
#ifdef PLAMATRIX_USE_DOUBLE
    PLAMATRIX_DEFINE_NO_CUDA_FINITE_STATISTICS(double)
#endif

#undef PLAMATRIX_DEFINE_NO_CUDA_FINITE_STATISTICS

#endif

    /// Caller-owned contiguous CUDA views. Input and outputs must not overlap.
    /// Async variants keep all storage and workspace live until stream completion.
    template <typename Scalar>
    void finiteRowMaskAsync(ConstMatrixView<Scalar, Device::GPU> input,
                            MatrixView<std::uint8_t, Device::GPU> output,
                            ReductionWorkspace& workspace, cudaStream_t stream);

    template <typename Scalar>
    void finiteColumnBoundsAsync(ConstMatrixView<Scalar, Device::GPU> input,
                                 MatrixView<Scalar, Device::GPU> minimum,
                                 MatrixView<Scalar, Device::GPU> maximum,
                                 MatrixView<Index, Device::GPU> valid_row_count,
                                 ReductionWorkspace& workspace,
                                 cudaStream_t stream);

    template <typename Scalar>
    void finiteColumnBoundsWithMaskAsync(ConstMatrixView<Scalar, Device::GPU> input,
                                         MatrixView<std::uint8_t, Device::GPU> row_mask,
                                         MatrixView<Scalar, Device::GPU> minimum,
                                         MatrixView<Scalar, Device::GPU> maximum,
                                         MatrixView<Index, Device::GPU> valid_row_count,
                                         ReductionWorkspace& workspace,
                                         cudaStream_t stream);

#ifndef PLAMATRIX_WITH_CUDA
    template <typename Scalar>
    void finiteRowMaskAsync(ConstMatrixView<Scalar, Device::GPU>,
                            MatrixView<std::uint8_t, Device::GPU>,
                            ReductionWorkspace&,
                            cudaStream_t)
    {
        statistics_detail::throwNoCuda("finiteRowMaskAsync");
    }

    template <typename Scalar>
    void finiteColumnBoundsAsync(ConstMatrixView<Scalar, Device::GPU>,
                                 MatrixView<Scalar, Device::GPU>,
                                 MatrixView<Scalar, Device::GPU>,
                                 MatrixView<Index, Device::GPU>,
                                 ReductionWorkspace&,
                                 cudaStream_t)
    {
        statistics_detail::throwNoCuda("finiteColumnBoundsAsync");
    }

    template <typename Scalar>
    void finiteColumnBoundsWithMaskAsync(ConstMatrixView<Scalar, Device::GPU>,
                                         MatrixView<std::uint8_t, Device::GPU>,
                                         MatrixView<Scalar, Device::GPU>,
                                         MatrixView<Scalar, Device::GPU>,
                                         MatrixView<Index, Device::GPU>,
                                         ReductionWorkspace&,
                                         cudaStream_t)
    {
        statistics_detail::throwNoCuda("finiteColumnBoundsWithMaskAsync");
    }
#endif

    template <typename Scalar>
    void finiteRowMask(ConstMatrixView<Scalar, Device::GPU> input,
                       MatrixView<std::uint8_t, Device::GPU> output,
                       ReductionWorkspace& workspace,
                       cudaStream_t stream = nullptr)
    {
        finiteRowMaskAsync(input, output, workspace, stream);
#ifdef PLAMATRIX_WITH_CUDA
        PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
#endif
    }

    template <typename Scalar>
    void finiteColumnBounds(ConstMatrixView<Scalar, Device::GPU> input,
                            MatrixView<Scalar, Device::GPU> minimum,
                            MatrixView<Scalar, Device::GPU> maximum,
                            MatrixView<Index, Device::GPU> valid_row_count,
                            ReductionWorkspace& workspace,
                            cudaStream_t stream = nullptr)
    {
        finiteColumnBoundsAsync(input, minimum, maximum, valid_row_count, workspace, stream);
#ifdef PLAMATRIX_WITH_CUDA
        PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
#endif
    }

    template <typename Scalar>
    void finiteColumnBoundsWithMask(ConstMatrixView<Scalar, Device::GPU> input,
                                    MatrixView<std::uint8_t, Device::GPU> row_mask,
                                    MatrixView<Scalar, Device::GPU> minimum,
                                    MatrixView<Scalar, Device::GPU> maximum,
                                    MatrixView<Index, Device::GPU> valid_row_count,
                                    ReductionWorkspace& workspace,
                                    cudaStream_t stream = nullptr)
    {
        finiteColumnBoundsWithMaskAsync(input, row_mask, minimum, maximum, valid_row_count, workspace, stream);
#ifdef PLAMATRIX_WITH_CUDA
        PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
#endif
    }

} // namespace plamatrix::internal
