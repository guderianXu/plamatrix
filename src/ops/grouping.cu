#include "plamatrix/internal/ops/grouping.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include <cub/cub.cuh>

#include "plamatrix/internal/core/error_check.h"

namespace plamatrix::internal
{
    namespace
    {

        template <typename Matrix> void requireColumnVector(const char* operation, const Matrix& input)
        {
            if (input.cols() != 1)
            {
                throw std::invalid_argument(std::string(operation) + ": input must be a column vector");
            }
        }

        int checkedCubCount(Index count, const char* operation)
        {
            if (count < 0 || count > static_cast<Index>(std::numeric_limits<int>::max()))
            {
                throw std::overflow_error(std::string(operation) + ": CUB item count exceeds int range");
            }
            return static_cast<int>(count);
        }

        std::size_t appendWorkspaceRegion(std::size_t& offset,
                                          std::size_t alignment,
                                          std::size_t element_count,
                                          std::size_t element_size)
        {
            const std::size_t maximum = std::numeric_limits<std::size_t>::max();
            if (offset > maximum - (alignment - 1))
            {
                throw std::overflow_error("sortByKey: workspace alignment overflows size_t");
            }
            const std::size_t aligned = (offset + alignment - 1) & ~(alignment - 1);
            if (element_count != 0 && element_size > (maximum - aligned) / element_count)
            {
                throw std::overflow_error("sortByKey: workspace size overflows size_t");
            }
            offset = aligned + element_count * element_size;
            return aligned;
        }

        template <typename View> void requireContiguousColumnVector(const char* operation, const View& input)
        {
            requireColumnVector(operation, input);
            if (!input.isContiguousColumnMajor())
            {
                throw std::invalid_argument(std::string(operation) + ": MatrixView must be contiguous column-major");
            }
        }

        template <typename LeftView, typename RightView> bool viewsOverlap(const LeftView& left, const RightView& right)
        {
            if (left.rows() == 0 || right.rows() == 0)
            {
                return false;
            }
            if (static_cast<std::size_t>(left.rows()) >
                    std::numeric_limits<std::size_t>::max() / sizeof(typename LeftView::ValueType) ||
                static_cast<std::size_t>(right.rows()) >
                    std::numeric_limits<std::size_t>::max() / sizeof(typename RightView::ValueType))
            {
                throw std::overflow_error("grouping MatrixView byte extent overflows size_t");
            }
            const std::size_t left_bytes = static_cast<std::size_t>(left.rows()) * sizeof(typename LeftView::ValueType);
            const std::size_t right_bytes =
                static_cast<std::size_t>(right.rows()) * sizeof(typename RightView::ValueType);
            const auto left_address = reinterpret_cast<std::uintptr_t>(left.data());
            const auto right_address = reinterpret_cast<std::uintptr_t>(right.data());
            return left_address <= right_address ? right_address - left_address < left_bytes
                                                 : left_address - right_address < right_bytes;
        }

        void validateOperation(GroupReduction operation)
        {
            switch (operation)
            {
            case GroupReduction::Sum:
            case GroupReduction::Minimum:
            case GroupReduction::Maximum:
                return;
            }
            throw std::invalid_argument("group reduction operation is invalid");
        }

        template <typename Scalar> Scalar emptyValue(GroupReduction operation)
        {
            switch (operation)
            {
            case GroupReduction::Sum:
                return Scalar(0);
            case GroupReduction::Minimum:
                return std::numeric_limits<Scalar>::max();
            case GroupReduction::Maximum:
                return std::numeric_limits<Scalar>::lowest();
            }
            throw std::invalid_argument("group reduction operation is invalid");
        }

        struct SumOperation
        {
            template <typename Value>
            __host__ __device__ constexpr Value operator()(const Value& left, const Value& right) const
            {
                return left + right;
            }
        };

        struct MinimumOperation
        {
            template <typename Value>
            __host__ __device__ constexpr Value operator()(const Value& left, const Value& right) const
            {
                return right < left ? right : left;
            }
        };

        struct MaximumOperation
        {
            template <typename Value>
            __host__ __device__ constexpr Value operator()(const Value& left, const Value& right) const
            {
                return left < right ? right : left;
            }
        };

        template <typename Key, typename Value>
        void requireKeyValueShapes(const char* operation,
                                   const DenseStorage<Key, Device::GPU>& keys,
                                   const DenseStorage<Value, Device::GPU>& values)
        {
            requireColumnVector(operation, keys);
            requireColumnVector(operation, values);
            if (keys.rows() != values.rows())
            {
                throw std::invalid_argument(std::string(operation) +
                                            ": keys and values must have the same number of rows");
            }
        }

        template <typename Key, typename Value>
        void launchSortByKey(ConstMatrixView<Key, Device::GPU> keys,
                             ConstMatrixView<Value, Device::GPU> values,
                             MatrixView<Key, Device::GPU> sorted_keys,
                             MatrixView<Value, Device::GPU> sorted_values,
                             GroupingWorkspace& workspace,
                             cudaStream_t stream)
        {
            requireContiguousColumnVector("sortByKey", keys);
            requireContiguousColumnVector("sortByKey", values);
            requireContiguousColumnVector("sortByKey", sorted_keys);
            requireContiguousColumnVector("sortByKey", sorted_values);
            if (keys.rows() != values.rows() || sorted_keys.rows() != keys.rows() ||
                sorted_values.rows() != values.rows())
            {
                throw std::invalid_argument("sortByKey: outputs must match the input column-vector shapes");
            }
            if (viewsOverlap(sorted_keys, keys) || viewsOverlap(sorted_keys, values) ||
                viewsOverlap(sorted_values, keys) || viewsOverlap(sorted_values, values) ||
                viewsOverlap(sorted_keys, sorted_values))
            {
                throw std::invalid_argument("sortByKey: input and output storage must not overlap");
            }

            const int count = checkedCubCount(keys.rows(), "sortByKey");
            if (count == 0)
            {
                workspace.reserveBytesAsync(0, stream);
                return;
            }
            std::size_t temporary_bytes = 0;
            PLAMATRIX_CHECK_CUDA(cub::DeviceRadixSort::SortPairs(nullptr,
                                                                 temporary_bytes,
                                                                 keys.data(),
                                                                 sorted_keys.data(),
                                                                 values.data(),
                                                                 sorted_values.data(),
                                                                 count,
                                                                 0,
                                                                 static_cast<int>(sizeof(Key) * 8),
                                                                 stream));
            workspace.reserveBytesAsync(temporary_bytes, stream);
            PLAMATRIX_CHECK_CUDA(cub::DeviceRadixSort::SortPairs(workspace.data(),
                                                                 temporary_bytes,
                                                                 keys.data(),
                                                                 sorted_keys.data(),
                                                                 values.data(),
                                                                 sorted_values.data(),
                                                                 count,
                                                                 0,
                                                                 static_cast<int>(sizeof(Key) * 8),
                                                                 stream));
        }

        template <typename Key, typename Value>
        void launchSortByKey(const DenseStorage<Key, Device::GPU>& keys,
                             const DenseStorage<Value, Device::GPU>& values,
                             DenseStorage<Key, Device::GPU>& sorted_keys,
                             DenseStorage<Value, Device::GPU>& sorted_values,
                             GroupingWorkspace& workspace,
                             cudaStream_t stream)
        {
            launchSortByKey(keys.view(), values.view(), sorted_keys.view(), sorted_values.view(), workspace, stream);
        }

        constexpr int kInt32Key3BlockSize = 256;

        enum class Int32Key3Component : int
        {
            X,
            Y,
            Z
        };

        __device__ std::int32_t keyComponent(const Int32Key3& key, Int32Key3Component component)
        {
            switch (component)
            {
            case Int32Key3Component::X:
                return key.x;
            case Int32Key3Component::Y:
                return key.y;
            case Int32Key3Component::Z:
                return key.z;
            }
            return 0;
        }

        __global__ void
        initializeInt32Key3Permutation(const Int32Key3* keys, std::int32_t* components, Index* permutation, int count)
        {
            const int offset = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (offset >= count)
            {
                return;
            }
            components[offset] = keys[offset].z;
            permutation[offset] = static_cast<Index>(offset);
        }

        __global__ void extractInt32Key3Component(const Int32Key3* keys,
                                                  const Index* permutation,
                                                  std::int32_t* components,
                                                  int count,
                                                  Int32Key3Component component)
        {
            const int offset = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (offset >= count)
            {
                return;
            }
            components[offset] = keyComponent(keys[permutation[offset]], component);
        }

        template <typename Value>
        __global__ void gatherInt32Key3AndValue(const Int32Key3* keys,
                                                const Value* values,
                                                const Index* permutation,
                                                Int32Key3* sorted_keys,
                                                Value* sorted_values,
                                                int count)
        {
            const int offset = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (offset >= count)
            {
                return;
            }
            const Index source = permutation[offset];
            sorted_keys[offset] = keys[source];
            sorted_values[offset] = values[source];
        }

        template <typename Value>
        void launchInt32Key3Sort(ConstMatrixView<Int32Key3, Device::GPU> keys,
                                 ConstMatrixView<Value, Device::GPU> values,
                                 MatrixView<Int32Key3, Device::GPU> sorted_keys,
                                 MatrixView<Value, Device::GPU> sorted_values,
                                 GroupingWorkspace& workspace,
                                 cudaStream_t stream)
        {
            requireContiguousColumnVector("sortByKey", keys);
            requireContiguousColumnVector("sortByKey", values);
            requireContiguousColumnVector("sortByKey", sorted_keys);
            requireContiguousColumnVector("sortByKey", sorted_values);
            if (keys.rows() != values.rows() || sorted_keys.rows() != keys.rows() ||
                sorted_values.rows() != values.rows())
            {
                throw std::invalid_argument("sortByKey: inputs and outputs must have matching row counts");
            }
            if (viewsOverlap(sorted_keys, keys) || viewsOverlap(sorted_keys, values) ||
                viewsOverlap(sorted_values, keys) || viewsOverlap(sorted_values, values) ||
                viewsOverlap(sorted_keys, sorted_values))
            {
                throw std::invalid_argument("sortByKey: input and output storage must not overlap");
            }

            const int count = checkedCubCount(keys.rows(), "sortByKey");
            if (count == 0)
            {
                workspace.reserveBytesAsync(0, stream);
                return;
            }

            std::size_t temporary_bytes = 0;
            auto* null_component = static_cast<std::int32_t*>(nullptr);
            auto* null_permutation = static_cast<Index*>(nullptr);
            PLAMATRIX_CHECK_CUDA(cub::DeviceRadixSort::SortPairs(nullptr,
                                                                 temporary_bytes,
                                                                 null_component,
                                                                 null_component,
                                                                 null_permutation,
                                                                 null_permutation,
                                                                 count,
                                                                 0,
                                                                 32,
                                                                 stream));

            std::size_t total_bytes = temporary_bytes;
            const std::size_t component_a_offset = appendWorkspaceRegion(
                total_bytes, alignof(std::int32_t), static_cast<std::size_t>(count), sizeof(std::int32_t));
            const std::size_t component_b_offset = appendWorkspaceRegion(
                total_bytes, alignof(std::int32_t), static_cast<std::size_t>(count), sizeof(std::int32_t));
            const std::size_t permutation_a_offset =
                appendWorkspaceRegion(total_bytes, alignof(Index), static_cast<std::size_t>(count), sizeof(Index));
            const std::size_t permutation_b_offset =
                appendWorkspaceRegion(total_bytes, alignof(Index), static_cast<std::size_t>(count), sizeof(Index));
            workspace.reserveBytesAsync(total_bytes, stream);

            auto* base = static_cast<std::uint8_t*>(workspace.data());
            auto* component_a = reinterpret_cast<std::int32_t*>(base + component_a_offset);
            auto* component_b = reinterpret_cast<std::int32_t*>(base + component_b_offset);
            auto* permutation_a = reinterpret_cast<Index*>(base + permutation_a_offset);
            auto* permutation_b = reinterpret_cast<Index*>(base + permutation_b_offset);
            const unsigned int blocks =
                static_cast<unsigned int>(count / kInt32Key3BlockSize + (count % kInt32Key3BlockSize != 0 ? 1 : 0));

            initializeInt32Key3Permutation<<<blocks, kInt32Key3BlockSize, 0, stream>>>(
                keys.data(), component_a, permutation_a, count);
            PLAMATRIX_CHECK_CUDA(cudaGetLastError());
            PLAMATRIX_CHECK_CUDA(cub::DeviceRadixSort::SortPairs(workspace.data(),
                                                                 temporary_bytes,
                                                                 component_a,
                                                                 component_b,
                                                                 permutation_a,
                                                                 permutation_b,
                                                                 count,
                                                                 0,
                                                                 32,
                                                                 stream));

            extractInt32Key3Component<<<blocks, kInt32Key3BlockSize, 0, stream>>>(
                keys.data(), permutation_b, component_a, count, Int32Key3Component::Y);
            PLAMATRIX_CHECK_CUDA(cudaGetLastError());
            PLAMATRIX_CHECK_CUDA(cub::DeviceRadixSort::SortPairs(workspace.data(),
                                                                 temporary_bytes,
                                                                 component_a,
                                                                 component_b,
                                                                 permutation_b,
                                                                 permutation_a,
                                                                 count,
                                                                 0,
                                                                 32,
                                                                 stream));

            extractInt32Key3Component<<<blocks, kInt32Key3BlockSize, 0, stream>>>(
                keys.data(), permutation_a, component_a, count, Int32Key3Component::X);
            PLAMATRIX_CHECK_CUDA(cudaGetLastError());
            PLAMATRIX_CHECK_CUDA(cub::DeviceRadixSort::SortPairs(workspace.data(),
                                                                 temporary_bytes,
                                                                 component_a,
                                                                 component_b,
                                                                 permutation_a,
                                                                 permutation_b,
                                                                 count,
                                                                 0,
                                                                 32,
                                                                 stream));

            gatherInt32Key3AndValue<<<blocks, kInt32Key3BlockSize, 0, stream>>>(
                keys.data(), values.data(), permutation_b, sorted_keys.data(), sorted_values.data(), count);
            PLAMATRIX_CHECK_CUDA(cudaGetLastError());
        }

        template <typename Value>
        void launchSortByKey(const DenseStorage<Int32Key3, Device::GPU>& keys,
                             const DenseStorage<Value, Device::GPU>& values,
                             DenseStorage<Int32Key3, Device::GPU>& sorted_keys,
                             DenseStorage<Value, Device::GPU>& sorted_values,
                             GroupingWorkspace& workspace,
                             cudaStream_t stream)
        {
            launchInt32Key3Sort(
                keys.view(), values.view(), sorted_keys.view(), sorted_values.view(), workspace, stream);
        }

        template <typename Key>
        void launchRunLengthEncodeRaw(const Key* keys,
                                      Index rows,
                                      Key* unique_keys,
                                      Index* counts,
                                      Index* run_count,
                                      GroupingWorkspace& workspace,
                                      cudaStream_t stream)
        {
            const int count = checkedCubCount(rows, "runLengthEncode");
            if (count == 0)
            {
                workspace.reserveBytesAsync(0, stream);
                PLAMATRIX_CHECK_CUDA(cudaMemsetAsync(run_count, 0, sizeof(Index), stream));
                return;
            }
            std::size_t temporary_bytes = 0;
            PLAMATRIX_CHECK_CUDA(cub::DeviceRunLengthEncode::Encode(
                nullptr, temporary_bytes, keys, unique_keys, counts, run_count, count, stream));
            workspace.reserveBytesAsync(temporary_bytes, stream);
            PLAMATRIX_CHECK_CUDA(cub::DeviceRunLengthEncode::Encode(
                workspace.data(), temporary_bytes, keys, unique_keys, counts, run_count, count, stream));
        }

        template <typename Key>
        void launchRunLengthEncode(const DenseStorage<Key, Device::GPU>& keys,
                                   DenseStorage<Key, Device::GPU>& unique_keys,
                                   DenseStorage<Index, Device::GPU>& counts,
                                   DenseStorage<Index, Device::GPU>& run_count,
                                   GroupingWorkspace& workspace,
                                   cudaStream_t stream)
        {
            requireColumnVector("runLengthEncode", keys);
            if (unique_keys.rows() != keys.rows() || unique_keys.cols() != 1 || counts.rows() != keys.rows() ||
                counts.cols() != 1 || run_count.rows() != 1 || run_count.cols() != 1)
            {
                throw std::invalid_argument(
                    "runLengthEncode: outputs must be input-sized column capacities and a 1 x 1 run count");
            }
            if (keys.rows() != 0 && unique_keys.data() == keys.data())
            {
                throw std::invalid_argument("runLengthEncode: in-place encoding is not supported");
            }

            launchRunLengthEncodeRaw(
                keys.data(), keys.rows(), unique_keys.data(), counts.data(), run_count.data(), workspace, stream);
        }

        template <typename Key>
        void launchRunLengthEncode(ConstMatrixView<Key, Device::GPU> keys,
                                   MatrixView<Key, Device::GPU> unique_keys,
                                   MatrixView<Index, Device::GPU> counts,
                                   MatrixView<Index, Device::GPU> run_count,
                                   GroupingWorkspace& workspace,
                                   cudaStream_t stream)
        {
            requireContiguousColumnVector("runLengthEncode", keys);
            requireContiguousColumnVector("runLengthEncode", unique_keys);
            requireContiguousColumnVector("runLengthEncode", counts);
            requireContiguousColumnVector("runLengthEncode", run_count);
            if (unique_keys.rows() != keys.rows() || counts.rows() != keys.rows() || run_count.rows() != 1)
            {
                throw std::invalid_argument(
                    "runLengthEncode: outputs must be input-sized column capacities and a 1 x 1 run count");
            }
            if (viewsOverlap(unique_keys, keys) || viewsOverlap(counts, keys) || viewsOverlap(run_count, keys) ||
                viewsOverlap(unique_keys, counts) || viewsOverlap(unique_keys, run_count) ||
                viewsOverlap(counts, run_count))
            {
                throw std::invalid_argument("runLengthEncode: input and output storage must not overlap");
            }
            launchRunLengthEncodeRaw(
                keys.data(), keys.rows(), unique_keys.data(), counts.data(), run_count.data(), workspace, stream);
        }

        template <typename Key, typename Value, typename Operation>
        cudaError_t cubReduceByKey(void* temporary,
                                   std::size_t& temporary_bytes,
                                   const DenseStorage<Key, Device::GPU>& keys,
                                   const DenseStorage<Value, Device::GPU>& values,
                                   DenseStorage<Key, Device::GPU>& unique_keys,
                                   DenseStorage<Value, Device::GPU>& aggregates,
                                   DenseStorage<Index, Device::GPU>& run_count,
                                   Operation operation,
                                   int count,
                                   cudaStream_t stream)
        {
            return cub::DeviceReduce::ReduceByKey(temporary,
                                                  temporary_bytes,
                                                  keys.data(),
                                                  unique_keys.data(),
                                                  values.data(),
                                                  aggregates.data(),
                                                  run_count.data(),
                                                  operation,
                                                  count,
                                                  stream);
        }

        template <typename Key, typename Value>
        void launchReduceByKey(const DenseStorage<Key, Device::GPU>& keys,
                               const DenseStorage<Value, Device::GPU>& values,
                               GroupReduction operation,
                               DenseStorage<Key, Device::GPU>& unique_keys,
                               DenseStorage<Value, Device::GPU>& aggregates,
                               DenseStorage<Index, Device::GPU>& run_count,
                               GroupingWorkspace& workspace,
                               cudaStream_t stream)
        {
            requireKeyValueShapes("reduceByKey", keys, values);
            validateOperation(operation);
            if (unique_keys.rows() != keys.rows() || unique_keys.cols() != 1 || aggregates.rows() != values.rows() ||
                aggregates.cols() != 1 || run_count.rows() != 1 || run_count.cols() != 1)
            {
                throw std::invalid_argument(
                    "reduceByKey: outputs must be input-sized column capacities and a 1 x 1 run count");
            }
            if (keys.rows() != 0 && (unique_keys.data() == keys.data() || aggregates.data() == values.data()))
            {
                throw std::invalid_argument("reduceByKey: in-place reduction is not supported");
            }

            const int count = checkedCubCount(keys.rows(), "reduceByKey");
            if (count == 0)
            {
                workspace.reserveBytesAsync(0, stream);
                PLAMATRIX_CHECK_CUDA(cudaMemsetAsync(run_count.data(), 0, sizeof(Index), stream));
                return;
            }
            std::size_t temporary_bytes = 0;
            switch (operation)
            {
            case GroupReduction::Sum:
                PLAMATRIX_CHECK_CUDA(cubReduceByKey(nullptr,
                                                    temporary_bytes,
                                                    keys,
                                                    values,
                                                    unique_keys,
                                                    aggregates,
                                                    run_count,
                                                    SumOperation{},
                                                    count,
                                                    stream));
                break;
            case GroupReduction::Minimum:
                PLAMATRIX_CHECK_CUDA(cubReduceByKey(nullptr,
                                                    temporary_bytes,
                                                    keys,
                                                    values,
                                                    unique_keys,
                                                    aggregates,
                                                    run_count,
                                                    MinimumOperation{},
                                                    count,
                                                    stream));
                break;
            case GroupReduction::Maximum:
                PLAMATRIX_CHECK_CUDA(cubReduceByKey(nullptr,
                                                    temporary_bytes,
                                                    keys,
                                                    values,
                                                    unique_keys,
                                                    aggregates,
                                                    run_count,
                                                    MaximumOperation{},
                                                    count,
                                                    stream));
                break;
            }
            workspace.reserveBytesAsync(temporary_bytes, stream);
            switch (operation)
            {
            case GroupReduction::Sum:
                PLAMATRIX_CHECK_CUDA(cubReduceByKey(workspace.data(),
                                                    temporary_bytes,
                                                    keys,
                                                    values,
                                                    unique_keys,
                                                    aggregates,
                                                    run_count,
                                                    SumOperation{},
                                                    count,
                                                    stream));
                break;
            case GroupReduction::Minimum:
                PLAMATRIX_CHECK_CUDA(cubReduceByKey(workspace.data(),
                                                    temporary_bytes,
                                                    keys,
                                                    values,
                                                    unique_keys,
                                                    aggregates,
                                                    run_count,
                                                    MinimumOperation{},
                                                    count,
                                                    stream));
                break;
            case GroupReduction::Maximum:
                PLAMATRIX_CHECK_CUDA(cubReduceByKey(workspace.data(),
                                                    temporary_bytes,
                                                    keys,
                                                    values,
                                                    unique_keys,
                                                    aggregates,
                                                    run_count,
                                                    MaximumOperation{},
                                                    count,
                                                    stream));
                break;
            }
        }

        template <typename Value, typename Operation>
        cudaError_t cubSegmentedReduce(void* temporary,
                                       std::size_t& temporary_bytes,
                                       ConstMatrixView<Value, Device::GPU> values,
                                       ConstMatrixView<Index, Device::GPU> offsets,
                                       MatrixView<Value, Device::GPU> output,
                                       Operation operation,
                                       Value initial,
                                       int segments,
                                       cudaStream_t stream)
        {
            return cub::DeviceSegmentedReduce::Reduce(temporary,
                                                      temporary_bytes,
                                                      values.data(),
                                                      output.data(),
                                                      segments,
                                                      offsets.data(),
                                                      offsets.data() + 1,
                                                      operation,
                                                      initial,
                                                      stream);
        }

        template <typename Value>
        void launchSegmentedReduce(ConstMatrixView<Value, Device::GPU> values,
                                   ConstMatrixView<Index, Device::GPU> offsets,
                                   GroupReduction operation,
                                   MatrixView<Value, Device::GPU> output,
                                   GroupingWorkspace& workspace,
                                   cudaStream_t stream)
        {
            requireContiguousColumnVector("segmentedReduce", values);
            requireContiguousColumnVector("segmentedReduce", offsets);
            requireContiguousColumnVector("segmentedReduce", output);
            validateOperation(operation);
            if (offsets.rows() == 0)
            {
                throw std::invalid_argument("segmentedReduce: offsets must contain at least one element");
            }
            if (output.rows() != offsets.rows() - 1 || output.cols() != 1)
            {
                throw std::invalid_argument("segmentedReduce: output must have shape (offsets.rows() - 1) x 1");
            }
            if (viewsOverlap(output, values) || viewsOverlap(output, offsets))
            {
                throw std::invalid_argument("segmentedReduce: input and output storage must not overlap");
            }
            static_cast<void>(checkedCubCount(values.rows(), "segmentedReduce values"));
            const int segments = checkedCubCount(output.rows(), "segmentedReduce segments");
            if (segments == 0)
            {
                workspace.reserveBytesAsync(0, stream);
                return;
            }

            std::size_t temporary_bytes = 0;
            switch (operation)
            {
            case GroupReduction::Sum:
                PLAMATRIX_CHECK_CUDA(cubSegmentedReduce(nullptr,
                                                        temporary_bytes,
                                                        values,
                                                        offsets,
                                                        output,
                                                        SumOperation{},
                                                        emptyValue<Value>(operation),
                                                        segments,
                                                        stream));
                break;
            case GroupReduction::Minimum:
                PLAMATRIX_CHECK_CUDA(cubSegmentedReduce(nullptr,
                                                        temporary_bytes,
                                                        values,
                                                        offsets,
                                                        output,
                                                        MinimumOperation{},
                                                        emptyValue<Value>(operation),
                                                        segments,
                                                        stream));
                break;
            case GroupReduction::Maximum:
                PLAMATRIX_CHECK_CUDA(cubSegmentedReduce(nullptr,
                                                        temporary_bytes,
                                                        values,
                                                        offsets,
                                                        output,
                                                        MaximumOperation{},
                                                        emptyValue<Value>(operation),
                                                        segments,
                                                        stream));
                break;
            }
            workspace.reserveBytesAsync(temporary_bytes, stream);
            switch (operation)
            {
            case GroupReduction::Sum:
                PLAMATRIX_CHECK_CUDA(cubSegmentedReduce(workspace.data(),
                                                        temporary_bytes,
                                                        values,
                                                        offsets,
                                                        output,
                                                        SumOperation{},
                                                        emptyValue<Value>(operation),
                                                        segments,
                                                        stream));
                break;
            case GroupReduction::Minimum:
                PLAMATRIX_CHECK_CUDA(cubSegmentedReduce(workspace.data(),
                                                        temporary_bytes,
                                                        values,
                                                        offsets,
                                                        output,
                                                        MinimumOperation{},
                                                        emptyValue<Value>(operation),
                                                        segments,
                                                        stream));
                break;
            case GroupReduction::Maximum:
                PLAMATRIX_CHECK_CUDA(cubSegmentedReduce(workspace.data(),
                                                        temporary_bytes,
                                                        values,
                                                        offsets,
                                                        output,
                                                        MaximumOperation{},
                                                        emptyValue<Value>(operation),
                                                        segments,
                                                        stream));
                break;
            }
        }

        template <typename Value>
        void launchSegmentedReduce(const DenseStorage<Value, Device::GPU>& values,
                                   const DenseStorage<Index, Device::GPU>& offsets,
                                   GroupReduction operation,
                                   DenseStorage<Value, Device::GPU>& output,
                                   GroupingWorkspace& workspace,
                                   cudaStream_t stream)
        {
            launchSegmentedReduce(values.view(), offsets.view(), operation, output.view(), workspace, stream);
        }

        template <typename Key>
        RunLengthEncodeResult<Key, Device::GPU>
        copyExactRunLengthResult(RunLengthEncodeResult<Key, Device::GPU>& capacity, cudaStream_t stream)
        {
            const Index run_count = capacity.runCount.toCpu().data()[0];
            if (run_count < 0 || run_count > capacity.uniqueKeys.rows())
            {
                throw std::runtime_error("runLengthEncode: CUB returned an invalid run count");
            }
            RunLengthEncodeResult<Key, Device::GPU> result{
                DenseStorage<Key, Device::GPU>::uninitialized(run_count, 1),
                DenseStorage<Index, Device::GPU>::uninitialized(run_count, 1),
                DenseStorage<Index, Device::GPU>::uninitialized(1, 1),
            };
            PLAMATRIX_CHECK_CUDA(cudaMemcpyAsync(
                result.runCount.data(), capacity.runCount.data(), sizeof(Index), cudaMemcpyDeviceToDevice, stream));
            if (run_count != 0)
            {
                PLAMATRIX_CHECK_CUDA(cudaMemcpyAsync(result.uniqueKeys.data(),
                                                     capacity.uniqueKeys.data(),
                                                     static_cast<std::size_t>(run_count) * sizeof(Key),
                                                     cudaMemcpyDeviceToDevice,
                                                     stream));
                PLAMATRIX_CHECK_CUDA(cudaMemcpyAsync(result.counts.data(),
                                                     capacity.counts.data(),
                                                     static_cast<std::size_t>(run_count) * sizeof(Index),
                                                     cudaMemcpyDeviceToDevice,
                                                     stream));
            }
            PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
            return result;
        }

        template <typename Key, typename Value>
        ReduceByKeyResult<Key, Value, Device::GPU>
        copyExactReduceResult(ReduceByKeyResult<Key, Value, Device::GPU>& capacity, cudaStream_t stream)
        {
            const Index run_count = capacity.runCount.toCpu().data()[0];
            if (run_count < 0 || run_count > capacity.uniqueKeys.rows())
            {
                throw std::runtime_error("reduceByKey: CUB returned an invalid run count");
            }
            ReduceByKeyResult<Key, Value, Device::GPU> result{
                DenseStorage<Key, Device::GPU>::uninitialized(run_count, 1),
                DenseStorage<Value, Device::GPU>::uninitialized(run_count, 1),
                DenseStorage<Index, Device::GPU>::uninitialized(1, 1),
            };
            PLAMATRIX_CHECK_CUDA(cudaMemcpyAsync(
                result.runCount.data(), capacity.runCount.data(), sizeof(Index), cudaMemcpyDeviceToDevice, stream));
            if (run_count != 0)
            {
                PLAMATRIX_CHECK_CUDA(cudaMemcpyAsync(result.uniqueKeys.data(),
                                                     capacity.uniqueKeys.data(),
                                                     static_cast<std::size_t>(run_count) * sizeof(Key),
                                                     cudaMemcpyDeviceToDevice,
                                                     stream));
                PLAMATRIX_CHECK_CUDA(cudaMemcpyAsync(result.aggregates.data(),
                                                     capacity.aggregates.data(),
                                                     static_cast<std::size_t>(run_count) * sizeof(Value),
                                                     cudaMemcpyDeviceToDevice,
                                                     stream));
            }
            PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
            return result;
        }

    } // namespace

    template <typename Key, typename Value>
    SortedKeyValueResult<Key, Value, Device::GPU> sortByKey(const DenseStorage<Key, Device::GPU>& keys,
                                                            const DenseStorage<Value, Device::GPU>& values)
    {
        GroupingWorkspace workspace;
        return sortByKey(keys, values, workspace, nullptr);
    }

    template <typename Key, typename Value>
    SortedKeyValueResult<Key, Value, Device::GPU> sortByKey(const DenseStorage<Key, Device::GPU>& keys,
                                                            const DenseStorage<Value, Device::GPU>& values,
                                                            GroupingWorkspace& workspace,
                                                            cudaStream_t stream)
    {
        SortedKeyValueResult<Key, Value, Device::GPU> result{
            DenseStorage<Key, Device::GPU>::uninitialized(keys.rows(), 1),
            DenseStorage<Value, Device::GPU>::uninitialized(values.rows(), 1),
        };
        launchSortByKey(keys, values, result.keys, result.values, workspace, stream);
        PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
        return result;
    }

    template <typename Key, typename Value>
    void sortByKey(const DenseStorage<Key, Device::GPU>& keys,
                   const DenseStorage<Value, Device::GPU>& values,
                   DenseStorage<Key, Device::GPU>& sorted_keys,
                   DenseStorage<Value, Device::GPU>& sorted_values,
                   GroupingWorkspace& workspace,
                   cudaStream_t stream)
    {
        launchSortByKey(keys, values, sorted_keys, sorted_values, workspace, stream);
        PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    template <typename Key, typename Value>
    SortedKeyValueResult<Key, Value, Device::GPU> sortByKeyAsync(const DenseStorage<Key, Device::GPU>& keys,
                                                                 const DenseStorage<Value, Device::GPU>& values,
                                                                 GroupingWorkspace& workspace,
                                                                 cudaStream_t stream)
    {
        SortedKeyValueResult<Key, Value, Device::GPU> result{
            DenseStorage<Key, Device::GPU>::uninitializedAsync(keys.rows(), 1, stream),
            DenseStorage<Value, Device::GPU>::uninitializedAsync(values.rows(), 1, stream),
        };
        launchSortByKey(keys, values, result.keys, result.values, workspace, stream);
        return result;
    }

    template <typename Key, typename Value>
    void sortByKeyAsync(const DenseStorage<Key, Device::GPU>& keys,
                        const DenseStorage<Value, Device::GPU>& values,
                        DenseStorage<Key, Device::GPU>& sorted_keys,
                        DenseStorage<Value, Device::GPU>& sorted_values,
                        GroupingWorkspace& workspace,
                        cudaStream_t stream)
    {
        launchSortByKey(keys, values, sorted_keys, sorted_values, workspace, stream);
    }

    template <typename Value>
    void sortByKey(ConstMatrixView<Int32Key3, Device::GPU> keys,
                   ConstMatrixView<Value, Device::GPU> values,
                   MatrixView<Int32Key3, Device::GPU> sorted_keys,
                   MatrixView<Value, Device::GPU> sorted_values,
                   GroupingWorkspace& workspace,
                   cudaStream_t stream)
    {
        launchInt32Key3Sort(keys, values, sorted_keys, sorted_values, workspace, stream);
        PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    template <typename Value>
    void sortByKeyAsync(ConstMatrixView<Int32Key3, Device::GPU> keys,
                        ConstMatrixView<Value, Device::GPU> values,
                        MatrixView<Int32Key3, Device::GPU> sorted_keys,
                        MatrixView<Value, Device::GPU> sorted_values,
                        GroupingWorkspace& workspace,
                        cudaStream_t stream)
    {
        launchInt32Key3Sort(keys, values, sorted_keys, sorted_values, workspace, stream);
    }

    template <typename Key, typename Value>
    void sortByKey(ConstMatrixView<Key, Device::GPU> keys,
                   ConstMatrixView<Value, Device::GPU> values,
                   MatrixView<Key, Device::GPU> sorted_keys,
                   MatrixView<Value, Device::GPU> sorted_values,
                   GroupingWorkspace& workspace,
                   cudaStream_t stream)
    {
        launchSortByKey(keys, values, sorted_keys, sorted_values, workspace, stream);
        PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    template <typename Key, typename Value>
    void sortByKeyAsync(ConstMatrixView<Key, Device::GPU> keys,
                        ConstMatrixView<Value, Device::GPU> values,
                        MatrixView<Key, Device::GPU> sorted_keys,
                        MatrixView<Value, Device::GPU> sorted_values,
                        GroupingWorkspace& workspace,
                        cudaStream_t stream)
    {
        launchSortByKey(keys, values, sorted_keys, sorted_values, workspace, stream);
    }

    template <typename Key>
    RunLengthEncodeResult<Key, Device::GPU> runLengthEncode(const DenseStorage<Key, Device::GPU>& keys)
    {
        GroupingWorkspace workspace;
        return runLengthEncode(keys, workspace, nullptr);
    }

    template <typename Key>
    RunLengthEncodeResult<Key, Device::GPU>
    runLengthEncode(const DenseStorage<Key, Device::GPU>& keys, GroupingWorkspace& workspace, cudaStream_t stream)
    {
        RunLengthEncodeResult<Key, Device::GPU> capacity{
            DenseStorage<Key, Device::GPU>::uninitialized(keys.rows(), 1),
            DenseStorage<Index, Device::GPU>::uninitialized(keys.rows(), 1),
            DenseStorage<Index, Device::GPU>::uninitialized(1, 1),
        };
        launchRunLengthEncode(keys, capacity.uniqueKeys, capacity.counts, capacity.runCount, workspace, stream);
        PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
        return copyExactRunLengthResult(capacity, stream);
    }

    template <typename Key>
    void runLengthEncode(const DenseStorage<Key, Device::GPU>& keys,
                         DenseStorage<Key, Device::GPU>& unique_keys,
                         DenseStorage<Index, Device::GPU>& counts,
                         DenseStorage<Index, Device::GPU>& run_count,
                         GroupingWorkspace& workspace,
                         cudaStream_t stream)
    {
        launchRunLengthEncode(keys, unique_keys, counts, run_count, workspace, stream);
        PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    template <typename Key>
    RunLengthEncodeResult<Key, Device::GPU>
    runLengthEncodeAsync(const DenseStorage<Key, Device::GPU>& keys, GroupingWorkspace& workspace, cudaStream_t stream)
    {
        RunLengthEncodeResult<Key, Device::GPU> result{
            DenseStorage<Key, Device::GPU>::uninitializedAsync(keys.rows(), 1, stream),
            DenseStorage<Index, Device::GPU>::uninitializedAsync(keys.rows(), 1, stream),
            DenseStorage<Index, Device::GPU>::uninitializedAsync(1, 1, stream),
        };
        launchRunLengthEncode(keys, result.uniqueKeys, result.counts, result.runCount, workspace, stream);
        return result;
    }

    template <typename Key>
    void runLengthEncodeAsync(const DenseStorage<Key, Device::GPU>& keys,
                              DenseStorage<Key, Device::GPU>& unique_keys,
                              DenseStorage<Index, Device::GPU>& counts,
                              DenseStorage<Index, Device::GPU>& run_count,
                              GroupingWorkspace& workspace,
                              cudaStream_t stream)
    {
        launchRunLengthEncode(keys, unique_keys, counts, run_count, workspace, stream);
    }

    void runLengthEncode(ConstMatrixView<Int32Key3, Device::GPU> keys,
                         MatrixView<Int32Key3, Device::GPU> unique_keys,
                         MatrixView<Index, Device::GPU> counts,
                         MatrixView<Index, Device::GPU> run_count,
                         GroupingWorkspace& workspace,
                         cudaStream_t stream)
    {
        launchRunLengthEncode(keys, unique_keys, counts, run_count, workspace, stream);
        PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    void runLengthEncodeAsync(ConstMatrixView<Int32Key3, Device::GPU> keys,
                              MatrixView<Int32Key3, Device::GPU> unique_keys,
                              MatrixView<Index, Device::GPU> counts,
                              MatrixView<Index, Device::GPU> run_count,
                              GroupingWorkspace& workspace,
                              cudaStream_t stream)
    {
        launchRunLengthEncode(keys, unique_keys, counts, run_count, workspace, stream);
    }

    template <typename Key>
    void runLengthEncode(ConstMatrixView<Key, Device::GPU> keys,
                         MatrixView<Key, Device::GPU> unique_keys,
                         MatrixView<Index, Device::GPU> counts,
                         MatrixView<Index, Device::GPU> run_count,
                         GroupingWorkspace& workspace,
                         cudaStream_t stream)
    {
        launchRunLengthEncode(keys, unique_keys, counts, run_count, workspace, stream);
        PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    template <typename Key>
    void runLengthEncodeAsync(ConstMatrixView<Key, Device::GPU> keys,
                              MatrixView<Key, Device::GPU> unique_keys,
                              MatrixView<Index, Device::GPU> counts,
                              MatrixView<Index, Device::GPU> run_count,
                              GroupingWorkspace& workspace,
                              cudaStream_t stream)
    {
        launchRunLengthEncode(keys, unique_keys, counts, run_count, workspace, stream);
    }

    template <typename Key, typename Value>
    ReduceByKeyResult<Key, Value, Device::GPU> reduceByKey(const DenseStorage<Key, Device::GPU>& keys,
                                                           const DenseStorage<Value, Device::GPU>& values,
                                                           GroupReduction operation)
    {
        GroupingWorkspace workspace;
        return reduceByKey(keys, values, operation, workspace, nullptr);
    }

    template <typename Key, typename Value>
    ReduceByKeyResult<Key, Value, Device::GPU> reduceByKey(const DenseStorage<Key, Device::GPU>& keys,
                                                           const DenseStorage<Value, Device::GPU>& values,
                                                           GroupReduction operation,
                                                           GroupingWorkspace& workspace,
                                                           cudaStream_t stream)
    {
        ReduceByKeyResult<Key, Value, Device::GPU> capacity{
            DenseStorage<Key, Device::GPU>::uninitialized(keys.rows(), 1),
            DenseStorage<Value, Device::GPU>::uninitialized(values.rows(), 1),
            DenseStorage<Index, Device::GPU>::uninitialized(1, 1),
        };
        launchReduceByKey(
            keys, values, operation, capacity.uniqueKeys, capacity.aggregates, capacity.runCount, workspace, stream);
        PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
        return copyExactReduceResult(capacity, stream);
    }

    template <typename Key, typename Value>
    void reduceByKey(const DenseStorage<Key, Device::GPU>& keys,
                     const DenseStorage<Value, Device::GPU>& values,
                     GroupReduction operation,
                     DenseStorage<Key, Device::GPU>& unique_keys,
                     DenseStorage<Value, Device::GPU>& aggregates,
                     DenseStorage<Index, Device::GPU>& run_count,
                     GroupingWorkspace& workspace,
                     cudaStream_t stream)
    {
        launchReduceByKey(keys, values, operation, unique_keys, aggregates, run_count, workspace, stream);
        PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    template <typename Key, typename Value>
    ReduceByKeyResult<Key, Value, Device::GPU> reduceByKeyAsync(const DenseStorage<Key, Device::GPU>& keys,
                                                                const DenseStorage<Value, Device::GPU>& values,
                                                                GroupReduction operation,
                                                                GroupingWorkspace& workspace,
                                                                cudaStream_t stream)
    {
        ReduceByKeyResult<Key, Value, Device::GPU> result{
            DenseStorage<Key, Device::GPU>::uninitializedAsync(keys.rows(), 1, stream),
            DenseStorage<Value, Device::GPU>::uninitializedAsync(values.rows(), 1, stream),
            DenseStorage<Index, Device::GPU>::uninitializedAsync(1, 1, stream),
        };
        launchReduceByKey(
            keys, values, operation, result.uniqueKeys, result.aggregates, result.runCount, workspace, stream);
        return result;
    }

    template <typename Key, typename Value>
    void reduceByKeyAsync(const DenseStorage<Key, Device::GPU>& keys,
                          const DenseStorage<Value, Device::GPU>& values,
                          GroupReduction operation,
                          DenseStorage<Key, Device::GPU>& unique_keys,
                          DenseStorage<Value, Device::GPU>& aggregates,
                          DenseStorage<Index, Device::GPU>& run_count,
                          GroupingWorkspace& workspace,
                          cudaStream_t stream)
    {
        launchReduceByKey(keys, values, operation, unique_keys, aggregates, run_count, workspace, stream);
    }

    template <typename Value>
    DenseStorage<Value, Device::GPU> segmentedReduce(const DenseStorage<Value, Device::GPU>& values,
                                                    const DenseStorage<Index, Device::GPU>& offsets,
                                                    GroupReduction operation)
    {
        GroupingWorkspace workspace;
        return segmentedReduce(values, offsets, operation, workspace, nullptr);
    }

    template <typename Value>
    DenseStorage<Value, Device::GPU> segmentedReduce(const DenseStorage<Value, Device::GPU>& values,
                                                    const DenseStorage<Index, Device::GPU>& offsets,
                                                    GroupReduction operation,
                                                    GroupingWorkspace& workspace,
                                                    cudaStream_t stream)
    {
        requireColumnVector("segmentedReduce", values);
        requireColumnVector("segmentedReduce", offsets);
        validateOperation(operation);
        if (offsets.rows() == 0)
        {
            throw std::invalid_argument("segmentedReduce: offsets must contain at least one element");
        }
        auto output = DenseStorage<Value, Device::GPU>::uninitialized(offsets.rows() - 1, 1);
        launchSegmentedReduce(values, offsets, operation, output, workspace, stream);
        PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
        return output;
    }

    template <typename Value>
    void segmentedReduce(const DenseStorage<Value, Device::GPU>& values,
                         const DenseStorage<Index, Device::GPU>& offsets,
                         GroupReduction operation,
                         DenseStorage<Value, Device::GPU>& output,
                         GroupingWorkspace& workspace,
                         cudaStream_t stream)
    {
        launchSegmentedReduce(values, offsets, operation, output, workspace, stream);
        PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    template <typename Value>
    DenseStorage<Value, Device::GPU> segmentedReduceAsync(const DenseStorage<Value, Device::GPU>& values,
                                                         const DenseStorage<Index, Device::GPU>& offsets,
                                                         GroupReduction operation,
                                                         GroupingWorkspace& workspace,
                                                         cudaStream_t stream)
    {
        requireColumnVector("segmentedReduce", values);
        requireColumnVector("segmentedReduce", offsets);
        validateOperation(operation);
        if (offsets.rows() == 0)
        {
            throw std::invalid_argument("segmentedReduce: offsets must contain at least one element");
        }
        auto output = DenseStorage<Value, Device::GPU>::uninitializedAsync(offsets.rows() - 1, 1, stream);
        launchSegmentedReduce(values, offsets, operation, output, workspace, stream);
        return output;
    }

    template <typename Value>
    void segmentedReduceAsync(const DenseStorage<Value, Device::GPU>& values,
                              const DenseStorage<Index, Device::GPU>& offsets,
                              GroupReduction operation,
                              DenseStorage<Value, Device::GPU>& output,
                              GroupingWorkspace& workspace,
                              cudaStream_t stream)
    {
        launchSegmentedReduce(values, offsets, operation, output, workspace, stream);
    }

    template <typename Value>
    void segmentedReduce(ConstMatrixView<Value, Device::GPU> values,
                         ConstMatrixView<Index, Device::GPU> offsets,
                         GroupReduction operation,
                         MatrixView<Value, Device::GPU> output,
                         GroupingWorkspace& workspace,
                         cudaStream_t stream)
    {
        launchSegmentedReduce(values, offsets, operation, output, workspace, stream);
        PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    template <typename Value>
    void segmentedReduceAsync(ConstMatrixView<Value, Device::GPU> values,
                              ConstMatrixView<Index, Device::GPU> offsets,
                              GroupReduction operation,
                              MatrixView<Value, Device::GPU> output,
                              GroupingWorkspace& workspace,
                              cudaStream_t stream)
    {
        launchSegmentedReduce(values, offsets, operation, output, workspace, stream);
    }

#define PLAMATRIX_INSTANTIATE_GPU_KEY_VALUE(Key, Value)                                                                \
    template SortedKeyValueResult<Key, Value, Device::GPU> sortByKey(const DenseStorage<Key, Device::GPU>&,             \
                                                                     const DenseStorage<Value, Device::GPU>&);          \
    template SortedKeyValueResult<Key, Value, Device::GPU> sortByKey(const DenseStorage<Key, Device::GPU>&,             \
                                                                     const DenseStorage<Value, Device::GPU>&,           \
                                                                     GroupingWorkspace&,                               \
                                                                     cudaStream_t);                                    \
    template void sortByKey(const DenseStorage<Key, Device::GPU>&,                                                      \
                            const DenseStorage<Value, Device::GPU>&,                                                    \
                            DenseStorage<Key, Device::GPU>&,                                                            \
                            DenseStorage<Value, Device::GPU>&,                                                          \
                            GroupingWorkspace&,                                                                        \
                            cudaStream_t);                                                                             \
    template SortedKeyValueResult<Key, Value, Device::GPU> sortByKeyAsync(const DenseStorage<Key, Device::GPU>&,        \
                                                                          const DenseStorage<Value, Device::GPU>&,      \
                                                                          GroupingWorkspace&,                          \
                                                                          cudaStream_t);                               \
    template void sortByKeyAsync(const DenseStorage<Key, Device::GPU>&,                                                 \
                                 const DenseStorage<Value, Device::GPU>&,                                               \
                                 DenseStorage<Key, Device::GPU>&,                                                       \
                                 DenseStorage<Value, Device::GPU>&,                                                     \
                                 GroupingWorkspace&,                                                                   \
                                 cudaStream_t);                                                                        \
    template void sortByKey(ConstMatrixView<Key, Device::GPU>,                                                         \
                            ConstMatrixView<Value, Device::GPU>,                                                       \
                            MatrixView<Key, Device::GPU>,                                                              \
                            MatrixView<Value, Device::GPU>,                                                            \
                            GroupingWorkspace&,                                                                        \
                            cudaStream_t);                                                                             \
    template void sortByKeyAsync(ConstMatrixView<Key, Device::GPU>,                                                    \
                                 ConstMatrixView<Value, Device::GPU>,                                                  \
                                 MatrixView<Key, Device::GPU>,                                                         \
                                 MatrixView<Value, Device::GPU>,                                                       \
                                 GroupingWorkspace&,                                                                   \
                                 cudaStream_t);                                                                        \
    template ReduceByKeyResult<Key, Value, Device::GPU> reduceByKey(                                                   \
        const DenseStorage<Key, Device::GPU>&, const DenseStorage<Value, Device::GPU>&, GroupReduction);                 \
    template ReduceByKeyResult<Key, Value, Device::GPU> reduceByKey(const DenseStorage<Key, Device::GPU>&,              \
                                                                    const DenseStorage<Value, Device::GPU>&,            \
                                                                    GroupReduction,                                    \
                                                                    GroupingWorkspace&,                                \
                                                                    cudaStream_t);                                     \
    template void reduceByKey(const DenseStorage<Key, Device::GPU>&,                                                    \
                              const DenseStorage<Value, Device::GPU>&,                                                  \
                              GroupReduction,                                                                          \
                              DenseStorage<Key, Device::GPU>&,                                                          \
                              DenseStorage<Value, Device::GPU>&,                                                        \
                              DenseStorage<Index, Device::GPU>&,                                                        \
                              GroupingWorkspace&,                                                                      \
                              cudaStream_t);                                                                           \
    template ReduceByKeyResult<Key, Value, Device::GPU> reduceByKeyAsync(const DenseStorage<Key, Device::GPU>&,         \
                                                                         const DenseStorage<Value, Device::GPU>&,       \
                                                                         GroupReduction,                               \
                                                                         GroupingWorkspace&,                           \
                                                                         cudaStream_t);                                \
    template void reduceByKeyAsync(const DenseStorage<Key, Device::GPU>&,                                               \
                                   const DenseStorage<Value, Device::GPU>&,                                             \
                                   GroupReduction,                                                                     \
                                   DenseStorage<Key, Device::GPU>&,                                                     \
                                   DenseStorage<Value, Device::GPU>&,                                                   \
                                   DenseStorage<Index, Device::GPU>&,                                                   \
                                   GroupingWorkspace&,                                                                 \
                                   cudaStream_t)

#define PLAMATRIX_INSTANTIATE_GPU_KEY(Key)                                                                             \
    template RunLengthEncodeResult<Key, Device::GPU> runLengthEncode(const DenseStorage<Key, Device::GPU>&);            \
    template RunLengthEncodeResult<Key, Device::GPU> runLengthEncode(                                                  \
        const DenseStorage<Key, Device::GPU>&, GroupingWorkspace&, cudaStream_t);                                       \
    template void runLengthEncode(const DenseStorage<Key, Device::GPU>&,                                                \
                                  DenseStorage<Key, Device::GPU>&,                                                      \
                                  DenseStorage<Index, Device::GPU>&,                                                    \
                                  DenseStorage<Index, Device::GPU>&,                                                    \
                                  GroupingWorkspace&,                                                                  \
                                  cudaStream_t);                                                                       \
    template RunLengthEncodeResult<Key, Device::GPU> runLengthEncodeAsync(                                             \
        const DenseStorage<Key, Device::GPU>&, GroupingWorkspace&, cudaStream_t);                                       \
    template void runLengthEncodeAsync(const DenseStorage<Key, Device::GPU>&,                                           \
                                       DenseStorage<Key, Device::GPU>&,                                                 \
                                       DenseStorage<Index, Device::GPU>&,                                               \
                                       DenseStorage<Index, Device::GPU>&,                                               \
                                       GroupingWorkspace&,                                                             \
                                       cudaStream_t);                                                                  \
    template void runLengthEncode(ConstMatrixView<Key, Device::GPU>,                                                   \
                                  MatrixView<Key, Device::GPU>,                                                        \
                                  MatrixView<Index, Device::GPU>,                                                      \
                                  MatrixView<Index, Device::GPU>,                                                      \
                                  GroupingWorkspace&,                                                                  \
                                  cudaStream_t);                                                                       \
    template void runLengthEncodeAsync(ConstMatrixView<Key, Device::GPU>,                                              \
                                       MatrixView<Key, Device::GPU>,                                                   \
                                       MatrixView<Index, Device::GPU>,                                                 \
                                       MatrixView<Index, Device::GPU>,                                                 \
                                       GroupingWorkspace&,                                                             \
                                       cudaStream_t);                                                                  \
    PLAMATRIX_INSTANTIATE_GPU_KEY_VALUE(Key, float);                                                                   \
    PLAMATRIX_INSTANTIATE_GPU_KEY_VALUE(Key, double);                                                                  \
    PLAMATRIX_INSTANTIATE_GPU_KEY_VALUE(Key, Index)

    PLAMATRIX_INSTANTIATE_GPU_KEY(std::uint32_t);
    PLAMATRIX_INSTANTIATE_GPU_KEY(std::uint64_t);
    PLAMATRIX_INSTANTIATE_GPU_KEY(Index);

#define PLAMATRIX_INSTANTIATE_GPU_INT32_KEY3_VALUE(Value)                                                              \
    template SortedKeyValueResult<Int32Key3, Value, Device::GPU> sortByKey(const DenseStorage<Int32Key3, Device::GPU>&, \
                                                                           const DenseStorage<Value, Device::GPU>&);    \
    template SortedKeyValueResult<Int32Key3, Value, Device::GPU> sortByKey(const DenseStorage<Int32Key3, Device::GPU>&, \
                                                                           const DenseStorage<Value, Device::GPU>&,     \
                                                                           GroupingWorkspace&,                         \
                                                                           cudaStream_t);                              \
    template void sortByKey(const DenseStorage<Int32Key3, Device::GPU>&,                                                \
                            const DenseStorage<Value, Device::GPU>&,                                                    \
                            DenseStorage<Int32Key3, Device::GPU>&,                                                      \
                            DenseStorage<Value, Device::GPU>&,                                                          \
                            GroupingWorkspace&,                                                                        \
                            cudaStream_t);                                                                             \
    template SortedKeyValueResult<Int32Key3, Value, Device::GPU> sortByKeyAsync(                                       \
        const DenseStorage<Int32Key3, Device::GPU>&,                                                                    \
        const DenseStorage<Value, Device::GPU>&,                                                                        \
        GroupingWorkspace&,                                                                                            \
        cudaStream_t);                                                                                                 \
    template void sortByKeyAsync(const DenseStorage<Int32Key3, Device::GPU>&,                                           \
                                 const DenseStorage<Value, Device::GPU>&,                                               \
                                 DenseStorage<Int32Key3, Device::GPU>&,                                                 \
                                 DenseStorage<Value, Device::GPU>&,                                                     \
                                 GroupingWorkspace&,                                                                   \
                                 cudaStream_t);                                                                        \
    template void sortByKey(ConstMatrixView<Int32Key3, Device::GPU>,                                                   \
                            ConstMatrixView<Value, Device::GPU>,                                                       \
                            MatrixView<Int32Key3, Device::GPU>,                                                        \
                            MatrixView<Value, Device::GPU>,                                                            \
                            GroupingWorkspace&,                                                                        \
                            cudaStream_t);                                                                             \
    template void sortByKeyAsync(ConstMatrixView<Int32Key3, Device::GPU>,                                              \
                                 ConstMatrixView<Value, Device::GPU>,                                                  \
                                 MatrixView<Int32Key3, Device::GPU>,                                                   \
                                 MatrixView<Value, Device::GPU>,                                                       \
                                 GroupingWorkspace&,                                                                   \
                                 cudaStream_t)

    PLAMATRIX_INSTANTIATE_GPU_INT32_KEY3_VALUE(std::int32_t);
    PLAMATRIX_INSTANTIATE_GPU_INT32_KEY3_VALUE(Index);

    template RunLengthEncodeResult<Int32Key3, Device::GPU> runLengthEncode(const DenseStorage<Int32Key3, Device::GPU>&);
    template RunLengthEncodeResult<Int32Key3, Device::GPU>
    runLengthEncode(const DenseStorage<Int32Key3, Device::GPU>&, GroupingWorkspace&, cudaStream_t);
    template void runLengthEncode(const DenseStorage<Int32Key3, Device::GPU>&,
                                  DenseStorage<Int32Key3, Device::GPU>&,
                                  DenseStorage<Index, Device::GPU>&,
                                  DenseStorage<Index, Device::GPU>&,
                                  GroupingWorkspace&,
                                  cudaStream_t);
    template RunLengthEncodeResult<Int32Key3, Device::GPU>
    runLengthEncodeAsync(const DenseStorage<Int32Key3, Device::GPU>&, GroupingWorkspace&, cudaStream_t);
    template void runLengthEncodeAsync(const DenseStorage<Int32Key3, Device::GPU>&,
                                       DenseStorage<Int32Key3, Device::GPU>&,
                                       DenseStorage<Index, Device::GPU>&,
                                       DenseStorage<Index, Device::GPU>&,
                                       GroupingWorkspace&,
                                       cudaStream_t);

#define PLAMATRIX_INSTANTIATE_GPU_SEGMENTED(Value)                                                                     \
    template DenseStorage<Value, Device::GPU> segmentedReduce(                                                          \
        const DenseStorage<Value, Device::GPU>&, const DenseStorage<Index, Device::GPU>&, GroupReduction);               \
    template DenseStorage<Value, Device::GPU> segmentedReduce(const DenseStorage<Value, Device::GPU>&,                   \
                                                             const DenseStorage<Index, Device::GPU>&,                   \
                                                             GroupReduction,                                           \
                                                             GroupingWorkspace&,                                       \
                                                             cudaStream_t);                                            \
    template void segmentedReduce(const DenseStorage<Value, Device::GPU>&,                                              \
                                  const DenseStorage<Index, Device::GPU>&,                                              \
                                  GroupReduction,                                                                      \
                                  DenseStorage<Value, Device::GPU>&,                                                    \
                                  GroupingWorkspace&,                                                                  \
                                  cudaStream_t);                                                                       \
    template DenseStorage<Value, Device::GPU> segmentedReduceAsync(const DenseStorage<Value, Device::GPU>&,              \
                                                                  const DenseStorage<Index, Device::GPU>&,              \
                                                                  GroupReduction,                                      \
                                                                  GroupingWorkspace&,                                  \
                                                                  cudaStream_t);                                       \
    template void segmentedReduceAsync(const DenseStorage<Value, Device::GPU>&,                                         \
                                       const DenseStorage<Index, Device::GPU>&,                                         \
                                       GroupReduction,                                                                 \
                                       DenseStorage<Value, Device::GPU>&,                                               \
                                       GroupingWorkspace&,                                                             \
                                       cudaStream_t);                                                                  \
    template void segmentedReduce(ConstMatrixView<Value, Device::GPU>,                                                 \
                                  ConstMatrixView<Index, Device::GPU>,                                                 \
                                  GroupReduction,                                                                      \
                                  MatrixView<Value, Device::GPU>,                                                      \
                                  GroupingWorkspace&,                                                                  \
                                  cudaStream_t);                                                                       \
    template void segmentedReduceAsync(ConstMatrixView<Value, Device::GPU>,                                            \
                                       ConstMatrixView<Index, Device::GPU>,                                            \
                                       GroupReduction,                                                                 \
                                       MatrixView<Value, Device::GPU>,                                                 \
                                       GroupingWorkspace&,                                                             \
                                       cudaStream_t)

    PLAMATRIX_INSTANTIATE_GPU_SEGMENTED(float);
    PLAMATRIX_INSTANTIATE_GPU_SEGMENTED(double);
    PLAMATRIX_INSTANTIATE_GPU_SEGMENTED(Index);

#undef PLAMATRIX_INSTANTIATE_GPU_SEGMENTED
#undef PLAMATRIX_INSTANTIATE_GPU_INT32_KEY3_VALUE
#undef PLAMATRIX_INSTANTIATE_GPU_KEY
#undef PLAMATRIX_INSTANTIATE_GPU_KEY_VALUE

} // namespace plamatrix::internal
