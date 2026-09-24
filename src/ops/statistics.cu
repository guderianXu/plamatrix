#include "plamatrix/internal/ops/statistics.h"

#include <cstdint>
#include <array>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <math_constants.h>

#include "plamatrix/internal/core/error_check.h"
#include "plamatrix/internal/core/checked_math.h"

namespace plamatrix::internal
{
    namespace
    {

        constexpr int kStatisticsBlockSize = 256;

        struct ViewRange
        {
            std::uintptr_t begin;
            std::uintptr_t end;
        };

        template <typename View> ViewRange checkedViewRange(const View& view, const char* operation)
        {
            if (!view.isContiguousColumnMajor())
            {
                throw std::invalid_argument(std::string(operation) + ": views must be contiguous column-major");
            }
            const auto bytes = detail::checkedSizeMul(
                static_cast<std::size_t>(view.size()), sizeof(typename View::ValueType), operation);
            const auto begin = reinterpret_cast<std::uintptr_t>(view.data());
            if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin)
            {
                throw std::overflow_error(std::string(operation) + ": view address range overflows");
            }
            return {begin, begin + bytes};
        }

        template <typename... Views> void validateViews(const char* operation, const Views&... views)
        {
            const std::array<ViewRange, sizeof...(Views)> ranges{checkedViewRange(views, operation)...};
            for (std::size_t i = 0; i < ranges.size(); ++i)
            {
                for (std::size_t j = i + 1; j < ranges.size(); ++j)
                {
                    if (ranges[i].begin != ranges[i].end && ranges[j].begin != ranges[j].end &&
                        ranges[i].begin < ranges[j].end && ranges[j].begin < ranges[i].end)
                    {
                        throw std::invalid_argument(std::string(operation) +
                                                    ": input and output views must not overlap");
                    }
                }
            }
        }

        template <typename Scalar> __device__ Scalar positiveInfinity();

        template <> __device__ float positiveInfinity<float>()
        {
            return CUDART_INF_F;
        }

        template <> __device__ double positiveInfinity<double>()
        {
            return CUDART_INF;
        }

        template <typename Scalar> __device__ Scalar quietNaN();

        template <> __device__ float quietNaN<float>()
        {
            return CUDART_NAN_F;
        }

        template <> __device__ double quietNaN<double>()
        {
            return CUDART_NAN;
        }

        unsigned int checkedGrid(Index count, const char* operation)
        {
            if (count <= 0)
                return 0;
            const Index blocks = count / kStatisticsBlockSize + (count % kStatisticsBlockSize != 0 ? 1 : 0);
            if (blocks > static_cast<Index>(std::numeric_limits<int>::max()))
                throw std::overflow_error(std::string(operation) + ": CUDA grid range exceeded");
            return static_cast<unsigned int>(blocks);
        }

        std::size_t checkedMaskBytes(Index rows)
        {
            if (rows < 0 || static_cast<std::uintmax_t>(rows) > std::numeric_limits<std::size_t>::max())
                throw std::overflow_error("finiteColumnBounds: row mask size overflows size_t");
            return static_cast<std::size_t>(rows);
        }

        template <typename Scalar>
        __global__ void finiteRowMaskKernel(const Scalar* input, Index rows, Index columns, std::uint8_t* mask)
        {
            const Index row = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (row >= rows)
                return;
            bool finite = true;
            for (Index column = 0; column < columns; ++column)
            {
                if (!isfinite(input[row + column * rows]))
                {
                    finite = false;
                    break;
                }
            }
            mask[row] = finite ? std::uint8_t{1} : std::uint8_t{0};
        }

        template <typename Scalar>
        __global__ void finiteRowMaskAndCountKernel(
            const Scalar* input, Index rows, Index columns, std::uint8_t* mask, Index* valid_count)
        {
            const Index row = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (row >= rows)
                return;
            bool finite = true;
            for (Index column = 0; column < columns; ++column)
            {
                if (!isfinite(input[row + column * rows]))
                {
                    finite = false;
                    break;
                }
            }
            mask[row] = finite ? std::uint8_t{1} : std::uint8_t{0};
            if (finite)
            {
                atomicAdd(reinterpret_cast<unsigned long long*>(valid_count), 1ULL);
            }
        }

        template <typename Scalar>
        __global__ void finiteColumnBoundsKernel(const Scalar* input,
                                                 Index rows,
                                                 const std::uint8_t* mask,
                                                 const Index* valid_count,
                                                 Scalar* minimum,
                                                 Scalar* maximum)
        {
            __shared__ Scalar minimum_partial[kStatisticsBlockSize];
            __shared__ Scalar maximum_partial[kStatisticsBlockSize];
            const Index column = static_cast<Index>(blockIdx.x);
            Scalar local_minimum = positiveInfinity<Scalar>();
            Scalar local_maximum = -positiveInfinity<Scalar>();
            for (Index row = threadIdx.x; row < rows; row += blockDim.x)
            {
                if (mask[row] == 0)
                    continue;
                const Scalar value = input[row + column * rows];
                local_minimum = value < local_minimum ? value : local_minimum;
                local_maximum = value > local_maximum ? value : local_maximum;
            }
            minimum_partial[threadIdx.x] = local_minimum;
            maximum_partial[threadIdx.x] = local_maximum;
            __syncthreads();

            for (int offset = kStatisticsBlockSize / 2; offset > 0; offset /= 2)
            {
                if (threadIdx.x < offset)
                {
                    const Scalar other_minimum = minimum_partial[threadIdx.x + offset];
                    const Scalar other_maximum = maximum_partial[threadIdx.x + offset];
                    minimum_partial[threadIdx.x] =
                        other_minimum < minimum_partial[threadIdx.x] ? other_minimum : minimum_partial[threadIdx.x];
                    maximum_partial[threadIdx.x] =
                        other_maximum > maximum_partial[threadIdx.x] ? other_maximum : maximum_partial[threadIdx.x];
                }
                __syncthreads();
            }
            if (threadIdx.x == 0)
            {
                if (valid_count[0] == 0)
                {
                    const Scalar invalid = quietNaN<Scalar>();
                    minimum[column] = invalid;
                    maximum[column] = invalid;
                }
                else
                {
                    minimum[column] = minimum_partial[0];
                    maximum[column] = maximum_partial[0];
                }
            }
        }

        template <typename Input, typename Output>
        void launchFiniteRowMask(const Input& input,
                                 Output& output,
                                 ReductionWorkspace& workspace,
                                 cudaStream_t stream)
        {
            if (output.rows() != input.rows() || output.cols() != 1)
                throw std::invalid_argument("finiteRowMask: output must have shape input.rows() x 1");
            workspace.reserveBytesAsync(0, stream);
            const unsigned int grid = checkedGrid(input.rows(), "finiteRowMask");
            if (grid == 0)
                return;
            finiteRowMaskKernel<<<grid, kStatisticsBlockSize, 0, stream>>>(
                input.data(), input.rows(), input.cols(), output.data());
            PLAMATRIX_CHECK_CUDA(cudaGetLastError());
        }

        template <typename Input, typename Output, typename Count>
        void validateFiniteColumnBoundsOutputs(const Input& input,
                                               const Output& minimum,
                                               const Output& maximum,
                                               const Count& valid_count,
                                               const char* operation)
        {
            if (minimum.rows() != 1 || minimum.cols() != input.cols() || maximum.rows() != 1 ||
                maximum.cols() != input.cols() || valid_count.rows() != 1 || valid_count.cols() != 1)
            {
                throw std::invalid_argument(std::string(operation) +
                                            ": outputs must have shapes 1 x input.cols(), 1 x input.cols(), and 1 x 1");
            }
        }

        template <typename Input, typename Output, typename Count>
        void launchFiniteColumnBoundsFromMask(const Input& input,
                                              std::uint8_t* mask,
                                              Output& minimum,
                                              Output& maximum,
                                              Count& valid_count,
                                              cudaStream_t stream,
                                              const char* operation)
        {
            PLAMATRIX_CHECK_CUDA(cudaMemsetAsync(valid_count.data(), 0, sizeof(Index), stream));
            const unsigned int row_grid = checkedGrid(input.rows(), operation);
            if (row_grid != 0)
            {
                finiteRowMaskAndCountKernel<<<row_grid, kStatisticsBlockSize, 0, stream>>>(
                    input.data(), input.rows(), input.cols(), mask, valid_count.data());
                PLAMATRIX_CHECK_CUDA(cudaGetLastError());
            }
            if (input.cols() != 0)
            {
                if (input.cols() > static_cast<Index>(std::numeric_limits<int>::max()))
                    throw std::overflow_error(std::string(operation) + ": CUDA grid range exceeded");
                finiteColumnBoundsKernel<<<static_cast<unsigned int>(input.cols()), kStatisticsBlockSize, 0, stream>>>(
                    input.data(), input.rows(), mask, valid_count.data(), minimum.data(), maximum.data());
                PLAMATRIX_CHECK_CUDA(cudaGetLastError());
            }
        }

        template <typename Input, typename Output, typename Count>
        void launchFiniteColumnBounds(const Input& input,
                                      Output& minimum,
                                      Output& maximum,
                                      Count& valid_count,
                                      ReductionWorkspace& workspace,
                                      cudaStream_t stream)
        {
            validateFiniteColumnBoundsOutputs(input, minimum, maximum, valid_count, "finiteColumnBounds");
            workspace.reserveBytesAsync(checkedMaskBytes(input.rows()), stream);
            launchFiniteColumnBoundsFromMask(input,
                                             static_cast<std::uint8_t*>(workspace.data()),
                                             minimum,
                                             maximum,
                                             valid_count,
                                             stream,
                                             "finiteColumnBounds rows");
        }

        template <typename Input, typename Mask, typename Output, typename Count>
        void launchFiniteColumnBoundsWithMask(const Input& input,
                                              Mask& row_mask,
                                              Output& minimum,
                                              Output& maximum,
                                              Count& valid_count,
                                              ReductionWorkspace& workspace,
                                              cudaStream_t stream)
        {
            if (row_mask.rows() != input.rows() || row_mask.cols() != 1)
            {
                throw std::invalid_argument("finiteColumnBoundsWithMask: row mask must have shape input.rows() x 1");
            }
            validateFiniteColumnBoundsOutputs(input, minimum, maximum, valid_count, "finiteColumnBoundsWithMask");
            workspace.reserveBytesAsync(0, stream);
            launchFiniteColumnBoundsFromMask(
                input, row_mask.data(), minimum, maximum, valid_count, stream, "finiteColumnBoundsWithMask rows");
        }

    } // namespace

    template <typename Scalar>
    DenseStorage<std::uint8_t, Device::GPU> finiteRowMask(const DenseStorage<Scalar, Device::GPU>& input)
    {
        ReductionWorkspace workspace;
        return finiteRowMask(input, workspace, nullptr);
    }

    template <typename Scalar>
    DenseStorage<std::uint8_t, Device::GPU>
    finiteRowMask(const DenseStorage<Scalar, Device::GPU>& input, ReductionWorkspace& workspace, cudaStream_t stream)
    {
        auto output = DenseStorage<std::uint8_t, Device::GPU>::uninitialized(input.rows(), 1);
        launchFiniteRowMask(input, output, workspace, stream);
        PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
        return output;
    }

    template <typename Scalar>
    void finiteRowMask(const DenseStorage<Scalar, Device::GPU>& input,
                       DenseStorage<std::uint8_t, Device::GPU>& output,
                       ReductionWorkspace& workspace,
                       cudaStream_t stream)
    {
        launchFiniteRowMask(input, output, workspace, stream);
        PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    template <typename Scalar>
    DenseStorage<std::uint8_t, Device::GPU> finiteRowMaskAsync(const DenseStorage<Scalar, Device::GPU>& input,
                                                              ReductionWorkspace& workspace,
                                                              cudaStream_t stream)
    {
        auto output = DenseStorage<std::uint8_t, Device::GPU>::uninitializedAsync(input.rows(), 1, stream);
        launchFiniteRowMask(input, output, workspace, stream);
        return output;
    }

    template <typename Scalar>
    void finiteRowMaskAsync(const DenseStorage<Scalar, Device::GPU>& input,
                            DenseStorage<std::uint8_t, Device::GPU>& output,
                            ReductionWorkspace& workspace,
                            cudaStream_t stream)
    {
        launchFiniteRowMask(input, output, workspace, stream);
    }

    template <typename Scalar>
    FiniteColumnBoundsResult<Scalar, Device::GPU> finiteColumnBounds(const DenseStorage<Scalar, Device::GPU>& input)
    {
        ReductionWorkspace workspace;
        return finiteColumnBounds(input, workspace, nullptr);
    }

    template <typename Scalar>
    FiniteColumnBoundsResult<Scalar, Device::GPU> finiteColumnBounds(const DenseStorage<Scalar, Device::GPU>& input,
                                                                     ReductionWorkspace& workspace,
                                                                     cudaStream_t stream)
    {
        FiniteColumnBoundsResult<Scalar, Device::GPU> result{
            DenseStorage<Scalar, Device::GPU>::uninitialized(1, input.cols()),
            DenseStorage<Scalar, Device::GPU>::uninitialized(1, input.cols()),
            DenseStorage<Index, Device::GPU>::uninitialized(1, 1),
        };
        launchFiniteColumnBounds(input, result.minimum, result.maximum, result.validRowCount, workspace, stream);
        PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
        return result;
    }

    template <typename Scalar>
    void finiteColumnBounds(const DenseStorage<Scalar, Device::GPU>& input,
                            DenseStorage<Scalar, Device::GPU>& minimum,
                            DenseStorage<Scalar, Device::GPU>& maximum,
                            DenseStorage<Index, Device::GPU>& valid_row_count,
                            ReductionWorkspace& workspace,
                            cudaStream_t stream)
    {
        launchFiniteColumnBounds(input, minimum, maximum, valid_row_count, workspace, stream);
        PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    template <typename Scalar>
    FiniteColumnBoundsResult<Scalar, Device::GPU> finiteColumnBoundsAsync(const DenseStorage<Scalar, Device::GPU>& input,
                                                                          ReductionWorkspace& workspace,
                                                                          cudaStream_t stream)
    {
        FiniteColumnBoundsResult<Scalar, Device::GPU> result{
            DenseStorage<Scalar, Device::GPU>::uninitializedAsync(1, input.cols(), stream),
            DenseStorage<Scalar, Device::GPU>::uninitializedAsync(1, input.cols(), stream),
            DenseStorage<Index, Device::GPU>::uninitializedAsync(1, 1, stream),
        };
        launchFiniteColumnBounds(input, result.minimum, result.maximum, result.validRowCount, workspace, stream);
        return result;
    }

    template <typename Scalar>
    void finiteColumnBoundsAsync(const DenseStorage<Scalar, Device::GPU>& input,
                                 DenseStorage<Scalar, Device::GPU>& minimum,
                                 DenseStorage<Scalar, Device::GPU>& maximum,
                                 DenseStorage<Index, Device::GPU>& valid_row_count,
                                 ReductionWorkspace& workspace,
                                 cudaStream_t stream)
    {
        launchFiniteColumnBounds(input, minimum, maximum, valid_row_count, workspace, stream);
    }

    template <typename Scalar>
    FiniteColumnBoundsWithMaskResult<Scalar, Device::GPU>
    finiteColumnBoundsWithMask(const DenseStorage<Scalar, Device::GPU>& input)
    {
        ReductionWorkspace workspace;
        return finiteColumnBoundsWithMask(input, workspace, nullptr);
    }

    template <typename Scalar>
    FiniteColumnBoundsWithMaskResult<Scalar, Device::GPU> finiteColumnBoundsWithMask(
        const DenseStorage<Scalar, Device::GPU>& input, ReductionWorkspace& workspace, cudaStream_t stream)
    {
        FiniteColumnBoundsWithMaskResult<Scalar, Device::GPU> result{
            DenseStorage<std::uint8_t, Device::GPU>::uninitialized(input.rows(), 1),
            DenseStorage<Scalar, Device::GPU>::uninitialized(1, input.cols()),
            DenseStorage<Scalar, Device::GPU>::uninitialized(1, input.cols()),
            DenseStorage<Index, Device::GPU>::uninitialized(1, 1),
        };
        launchFiniteColumnBoundsWithMask(
            input, result.rowMask, result.minimum, result.maximum, result.validRowCount, workspace, stream);
        PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
        return result;
    }

    template <typename Scalar>
    void finiteColumnBoundsWithMask(const DenseStorage<Scalar, Device::GPU>& input,
                                    DenseStorage<std::uint8_t, Device::GPU>& row_mask,
                                    DenseStorage<Scalar, Device::GPU>& minimum,
                                    DenseStorage<Scalar, Device::GPU>& maximum,
                                    DenseStorage<Index, Device::GPU>& valid_row_count,
                                    ReductionWorkspace& workspace,
                                    cudaStream_t stream)
    {
        launchFiniteColumnBoundsWithMask(input, row_mask, minimum, maximum, valid_row_count, workspace, stream);
        PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    template <typename Scalar>
    FiniteColumnBoundsWithMaskResult<Scalar, Device::GPU> finiteColumnBoundsWithMaskAsync(
        const DenseStorage<Scalar, Device::GPU>& input, ReductionWorkspace& workspace, cudaStream_t stream)
    {
        FiniteColumnBoundsWithMaskResult<Scalar, Device::GPU> result{
            DenseStorage<std::uint8_t, Device::GPU>::uninitializedAsync(input.rows(), 1, stream),
            DenseStorage<Scalar, Device::GPU>::uninitializedAsync(1, input.cols(), stream),
            DenseStorage<Scalar, Device::GPU>::uninitializedAsync(1, input.cols(), stream),
            DenseStorage<Index, Device::GPU>::uninitializedAsync(1, 1, stream),
        };
        launchFiniteColumnBoundsWithMask(
            input, result.rowMask, result.minimum, result.maximum, result.validRowCount, workspace, stream);
        return result;
    }

    template <typename Scalar>
    void finiteColumnBoundsWithMaskAsync(const DenseStorage<Scalar, Device::GPU>& input,
                                         DenseStorage<std::uint8_t, Device::GPU>& row_mask,
                                         DenseStorage<Scalar, Device::GPU>& minimum,
                                         DenseStorage<Scalar, Device::GPU>& maximum,
                                         DenseStorage<Index, Device::GPU>& valid_row_count,
                                         ReductionWorkspace& workspace,
                                         cudaStream_t stream)
    {
        launchFiniteColumnBoundsWithMask(input, row_mask, minimum, maximum, valid_row_count, workspace, stream);
    }

    template <typename Scalar>
    void finiteRowMaskAsync(ConstMatrixView<Scalar, Device::GPU> input,
                            MatrixView<std::uint8_t, Device::GPU> output,
                            ReductionWorkspace& workspace,
                            cudaStream_t stream)
    {
        validateViews("finiteRowMask", input, output);
        launchFiniteRowMask(input, output, workspace, stream);
    }

    template <typename Scalar>
    void finiteColumnBoundsAsync(ConstMatrixView<Scalar, Device::GPU> input,
                                 MatrixView<Scalar, Device::GPU> minimum,
                                 MatrixView<Scalar, Device::GPU> maximum,
                                 MatrixView<Index, Device::GPU> valid_row_count,
                                 ReductionWorkspace& workspace,
                                 cudaStream_t stream)
    {
        validateViews("finiteColumnBounds", input, minimum, maximum, valid_row_count);
        launchFiniteColumnBounds(input, minimum, maximum, valid_row_count, workspace, stream);
    }

    template <typename Scalar>
    void finiteColumnBoundsWithMaskAsync(ConstMatrixView<Scalar, Device::GPU> input,
                                         MatrixView<std::uint8_t, Device::GPU> row_mask,
                                         MatrixView<Scalar, Device::GPU> minimum,
                                         MatrixView<Scalar, Device::GPU> maximum,
                                         MatrixView<Index, Device::GPU> valid_row_count,
                                         ReductionWorkspace& workspace,
                                         cudaStream_t stream)
    {
        validateViews("finiteColumnBoundsWithMask", input, row_mask, minimum, maximum, valid_row_count);
        launchFiniteColumnBoundsWithMask(input, row_mask, minimum, maximum, valid_row_count, workspace, stream);
    }

#define PLAMATRIX_INSTANTIATE_GPU_FINITE_STATISTICS(Scalar)                                                            \
    template void finiteRowMaskAsync(ConstMatrixView<Scalar, Device::GPU>,                                             \
                                     MatrixView<std::uint8_t, Device::GPU>,                                            \
                                     ReductionWorkspace&,                                                              \
                                     cudaStream_t);                                                                    \
    template void finiteColumnBoundsAsync(ConstMatrixView<Scalar, Device::GPU>,                                        \
                                          MatrixView<Scalar, Device::GPU>,                                             \
                                          MatrixView<Scalar, Device::GPU>,                                             \
                                          MatrixView<Index, Device::GPU>,                                              \
                                          ReductionWorkspace&,                                                         \
                                          cudaStream_t);                                                               \
    template void finiteColumnBoundsWithMaskAsync(ConstMatrixView<Scalar, Device::GPU>,                                \
                                                  MatrixView<std::uint8_t, Device::GPU>,                               \
                                                  MatrixView<Scalar, Device::GPU>,                                     \
                                                  MatrixView<Scalar, Device::GPU>,                                     \
                                                  MatrixView<Index, Device::GPU>,                                      \
                                                  ReductionWorkspace&,                                                 \
                                                  cudaStream_t);                                                       \
    template DenseStorage<std::uint8_t, Device::GPU> finiteRowMask(const DenseStorage<Scalar, Device::GPU>&);            \
    template DenseStorage<std::uint8_t, Device::GPU> finiteRowMask(                                                     \
        const DenseStorage<Scalar, Device::GPU>&, ReductionWorkspace&, cudaStream_t);                                   \
    template void finiteRowMask(const DenseStorage<Scalar, Device::GPU>&,                                               \
                                DenseStorage<std::uint8_t, Device::GPU>&,                                               \
                                ReductionWorkspace&,                                                                   \
                                cudaStream_t);                                                                         \
    template DenseStorage<std::uint8_t, Device::GPU> finiteRowMaskAsync(                                                \
        const DenseStorage<Scalar, Device::GPU>&, ReductionWorkspace&, cudaStream_t);                                   \
    template void finiteRowMaskAsync(const DenseStorage<Scalar, Device::GPU>&,                                          \
                                     DenseStorage<std::uint8_t, Device::GPU>&,                                          \
                                     ReductionWorkspace&,                                                              \
                                     cudaStream_t);                                                                    \
    template FiniteColumnBoundsResult<Scalar, Device::GPU> finiteColumnBounds(                                         \
        const DenseStorage<Scalar, Device::GPU>&);                                                                      \
    template FiniteColumnBoundsResult<Scalar, Device::GPU> finiteColumnBounds(                                         \
        const DenseStorage<Scalar, Device::GPU>&, ReductionWorkspace&, cudaStream_t);                                   \
    template void finiteColumnBounds(const DenseStorage<Scalar, Device::GPU>&,                                          \
                                     DenseStorage<Scalar, Device::GPU>&,                                                \
                                     DenseStorage<Scalar, Device::GPU>&,                                                \
                                     DenseStorage<Index, Device::GPU>&,                                                 \
                                     ReductionWorkspace&,                                                              \
                                     cudaStream_t);                                                                    \
    template FiniteColumnBoundsResult<Scalar, Device::GPU> finiteColumnBoundsAsync(                                    \
        const DenseStorage<Scalar, Device::GPU>&, ReductionWorkspace&, cudaStream_t);                                   \
    template void finiteColumnBoundsAsync(const DenseStorage<Scalar, Device::GPU>&,                                     \
                                          DenseStorage<Scalar, Device::GPU>&,                                           \
                                          DenseStorage<Scalar, Device::GPU>&,                                           \
                                          DenseStorage<Index, Device::GPU>&,                                            \
                                          ReductionWorkspace&,                                                         \
                                          cudaStream_t);                                                               \
    template FiniteColumnBoundsWithMaskResult<Scalar, Device::GPU> finiteColumnBoundsWithMask(                         \
        const DenseStorage<Scalar, Device::GPU>&);                                                                      \
    template FiniteColumnBoundsWithMaskResult<Scalar, Device::GPU> finiteColumnBoundsWithMask(                         \
        const DenseStorage<Scalar, Device::GPU>&, ReductionWorkspace&, cudaStream_t);                                   \
    template void finiteColumnBoundsWithMask(const DenseStorage<Scalar, Device::GPU>&,                                  \
                                             DenseStorage<std::uint8_t, Device::GPU>&,                                  \
                                             DenseStorage<Scalar, Device::GPU>&,                                        \
                                             DenseStorage<Scalar, Device::GPU>&,                                        \
                                             DenseStorage<Index, Device::GPU>&,                                         \
                                             ReductionWorkspace&,                                                      \
                                             cudaStream_t);                                                            \
    template FiniteColumnBoundsWithMaskResult<Scalar, Device::GPU> finiteColumnBoundsWithMaskAsync(                    \
        const DenseStorage<Scalar, Device::GPU>&, ReductionWorkspace&, cudaStream_t);                                   \
    template void finiteColumnBoundsWithMaskAsync(const DenseStorage<Scalar, Device::GPU>&,                             \
                                                  DenseStorage<std::uint8_t, Device::GPU>&,                             \
                                                  DenseStorage<Scalar, Device::GPU>&,                                   \
                                                  DenseStorage<Scalar, Device::GPU>&,                                   \
                                                  DenseStorage<Index, Device::GPU>&,                                    \
                                                  ReductionWorkspace&,                                                 \
                                                  cudaStream_t)

#ifdef PLAMATRIX_USE_FLOAT
    PLAMATRIX_INSTANTIATE_GPU_FINITE_STATISTICS(float);
#endif
#ifdef PLAMATRIX_USE_DOUBLE
    PLAMATRIX_INSTANTIATE_GPU_FINITE_STATISTICS(double);
#endif

#undef PLAMATRIX_INSTANTIATE_GPU_FINITE_STATISTICS

} // namespace plamatrix::internal
