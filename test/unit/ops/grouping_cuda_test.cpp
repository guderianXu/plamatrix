#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "plamatrix/internal/ops/grouping.h"
#include "../../support/cuda_test_utils.h"

#ifdef PLAMATRIX_WITH_CUDA

namespace plamatrix::internal
{
    namespace
    {

        template <typename Scalar> DenseStorage<Scalar, Device::CPU> makeVector(std::initializer_list<Scalar> values)
        {
            DenseStorage<Scalar, Device::CPU> result(static_cast<Index>(values.size()), 1);
            std::copy(values.begin(), values.end(), result.data());
            return result;
        }

        template <typename Scalar>
        void expectGpuVector(const DenseStorage<Scalar, Device::GPU>& actual, std::initializer_list<Scalar> expected)
        {
            const auto cpu = actual.toCpu();
            ASSERT_EQ(cpu.rows(), static_cast<Index>(expected.size()));
            ASSERT_EQ(cpu.cols(), 1);
            Index offset = 0;
            for (const Scalar value : expected)
            {
                EXPECT_EQ(cpu.data()[offset++], value);
            }
        }

        TEST(GroupingCudaTest, StableSortEncodeReduceAndSegmentedReduceMatchCpu)
        {
            const auto keys = makeVector<std::uint32_t>({3, 1, 3, 2, 1}).toGpu();
            const auto values = makeVector<float>({30.0F, 10.0F, 31.0F, 20.0F, 11.0F}).toGpu();

            const auto sorted = sortByKey(keys, values);
            expectGpuVector(sorted.keys, {1U, 1U, 2U, 3U, 3U});
            expectGpuVector(sorted.values, {10.0F, 11.0F, 20.0F, 30.0F, 31.0F});

            const auto runs = runLengthEncode(sorted.keys);
            expectGpuVector(runs.uniqueKeys, {1U, 2U, 3U});
            expectGpuVector(runs.counts, {Index(2), Index(1), Index(2)});
            EXPECT_EQ(runs.runCount.toCpu().data()[0], 3);

            const auto reduced = reduceByKey(sorted.keys, sorted.values, GroupReduction::Sum);
            const auto minimum = reduceByKey(sorted.keys, sorted.values, GroupReduction::Minimum);
            expectGpuVector(reduced.uniqueKeys, {1U, 2U, 3U});
            expectGpuVector(reduced.aggregates, {21.0F, 20.0F, 61.0F});
            expectGpuVector(minimum.aggregates, {10.0F, 20.0F, 30.0F});
            EXPECT_EQ(reduced.runCount.toCpu().data()[0], 3);

            const auto offsets = makeVector<Index>({0, 2, 2, 5}).toGpu();
            expectGpuVector(segmentedReduce(sorted.values, offsets, GroupReduction::Maximum),
                            {11.0F, std::numeric_limits<float>::lowest(), 31.0F});
            expectGpuVector(segmentedReduce(sorted.values, offsets, GroupReduction::Minimum),
                            {10.0F, std::numeric_limits<float>::max(), 20.0F});
        }

        TEST(GroupingCudaTest, AsyncAllocatingAndOutputReuseShareNonBlockingStreamWorkspace)
        {
            test::CudaStreamGuard stream(cudaStreamNonBlocking);
            GroupingWorkspace workspace;
            const auto keys = makeVector<Index>({2, 1, 1, 3}).toGpu();
            const auto values = makeVector<Index>({20, 10, 11, 30}).toGpu();
            const auto offsets = makeVector<Index>({0, 2, 4}).toGpu();

            auto sorted = sortByKeyAsync(keys, values, workspace, stream.get());
            auto runs = runLengthEncodeAsync(sorted.keys, workspace, stream.get());
            auto reduced =
                reduceByKeyAsync(sorted.keys, sorted.values, GroupReduction::Maximum, workspace, stream.get());
            auto segmented = segmentedReduceAsync(sorted.values, offsets, GroupReduction::Sum, workspace, stream.get());

            DenseStorage<Index, Device::GPU> reused_keys(4, 1);
            DenseStorage<Index, Device::GPU> reused_values(4, 1);
            DenseStorage<Index, Device::GPU> reused_unique(4, 1);
            DenseStorage<Index, Device::GPU> reused_counts(4, 1);
            DenseStorage<Index, Device::GPU> reused_run_count(1, 1);
            DenseStorage<Index, Device::GPU> reused_aggregates(4, 1);
            DenseStorage<Index, Device::GPU> reused_segmented(2, 1);
            sortByKeyAsync(keys, values, reused_keys, reused_values, workspace, stream.get());
            runLengthEncodeAsync(reused_keys, reused_unique, reused_counts, reused_run_count, workspace, stream.get());
            reduceByKeyAsync(reused_keys,
                             reused_values,
                             GroupReduction::Sum,
                             reused_unique,
                             reused_aggregates,
                             reused_run_count,
                             workspace,
                             stream.get());
            segmentedReduceAsync(
                reused_values, offsets, GroupReduction::Maximum, reused_segmented, workspace, stream.get());
            stream.synchronize();

            expectGpuVector(sorted.keys, {Index(1), Index(1), Index(2), Index(3)});
            expectGpuVector(sorted.values, {Index(10), Index(11), Index(20), Index(30)});
            EXPECT_EQ(runs.runCount.toCpu().data()[0], 3);
            const auto run_keys = runs.uniqueKeys.toCpu();
            const auto run_counts = runs.counts.toCpu();
            EXPECT_EQ(run_keys.data()[0], 1);
            EXPECT_EQ(run_keys.data()[2], 3);
            EXPECT_EQ(run_counts.data()[0], 2);
            EXPECT_EQ(run_counts.data()[2], 1);
            EXPECT_EQ(reduced.runCount.toCpu().data()[0], 3);
            const auto reduced_values = reduced.aggregates.toCpu();
            EXPECT_EQ(reduced_values.data()[0], 11);
            EXPECT_EQ(reduced_values.data()[2], 30);
            expectGpuVector(segmented, {Index(21), Index(50)});
            EXPECT_EQ(reused_run_count.toCpu().data()[0], 3);
            const auto reused_reduced_values = reused_aggregates.toCpu();
            EXPECT_EQ(reused_reduced_values.data()[0], 21);
            EXPECT_EQ(reused_reduced_values.data()[2], 30);
            expectGpuVector(reused_segmented, {Index(11), Index(30)});

            sorted.keys.closeAsyncAllocation();
            sorted.values.closeAsyncAllocation();
            runs.uniqueKeys.closeAsyncAllocation();
            runs.counts.closeAsyncAllocation();
            runs.runCount.closeAsyncAllocation();
            reduced.uniqueKeys.closeAsyncAllocation();
            reduced.aggregates.closeAsyncAllocation();
            reduced.runCount.closeAsyncAllocation();
            segmented.closeAsyncAllocation();
            workspace.closeAsyncAllocation();
            stream.synchronize();
        }

        TEST(GroupingCudaTest, SupportsSignedAndUnsigned64BitKeysAndEmptyInputs)
        {
            const auto unsigned_sorted =
                sortByKey(makeVector<std::uint64_t>({2, 1}).toGpu(), makeVector<double>({20.0, 10.0}).toGpu());
            expectGpuVector(unsigned_sorted.values, {10.0, 20.0});

            const auto signed_sorted =
                sortByKey(makeVector<Index>({-1, 2, -1}).toGpu(), makeVector<Index>({1, 2, 3}).toGpu());
            expectGpuVector(signed_sorted.keys, {Index(-1), Index(-1), Index(2)});
            expectGpuVector(signed_sorted.values, {Index(1), Index(3), Index(2)});

            const DenseStorage<std::uint32_t, Device::CPU> empty_cpu(0, 1);
            const auto empty_gpu = empty_cpu.toGpu();
            const auto empty_runs = runLengthEncode(empty_gpu);
            EXPECT_EQ(empty_runs.uniqueKeys.rows(), 0);
            EXPECT_EQ(empty_runs.runCount.toCpu().data()[0], 0);

            const DenseStorage<float, Device::CPU> empty_values_cpu(0, 1);
            const auto empty_values = empty_values_cpu.toGpu();
            const auto empty_sorted = sortByKey(empty_gpu, empty_values);
            const auto empty_reduced = reduceByKey(empty_gpu, empty_values, GroupReduction::Sum);
            const auto empty_offsets = makeVector<Index>({0}).toGpu();
            const auto empty_segments = segmentedReduce(empty_values, empty_offsets, GroupReduction::Sum);
            EXPECT_EQ(empty_sorted.keys.rows(), 0);
            EXPECT_EQ(empty_reduced.runCount.toCpu().data()[0], 0);
            EXPECT_EQ(empty_segments.rows(), 0);
        }

        TEST(GroupingCudaTest, Int32Key3DenseSortAndEncodePreserveSignedLexicographicOrder)
        {
            constexpr std::int32_t minimum = std::numeric_limits<std::int32_t>::min();
            constexpr std::int32_t maximum = std::numeric_limits<std::int32_t>::max();
            const Int32Key3 extreme_low{minimum, maximum, -1};
            const Int32Key3 duplicate{-1, 5, 7};
            const Int32Key3 zero_negative_y{0, -1, maximum};
            const Int32Key3 zero_minimum_z{0, 0, minimum};
            const Int32Key3 zero{0, 0, 0};
            const Int32Key3 extreme_high{maximum, minimum, 0};
            const auto keys = makeVector<Int32Key3>({zero,
                                                     duplicate,
                                                     extreme_low,
                                                     duplicate,
                                                     zero_negative_y,
                                                     extreme_high,
                                                     zero_minimum_z,
                                                     extreme_low})
                                  .toGpu();
            const auto values = makeVector<Index>({0, 1, 2, 3, 4, 5, 6, 7}).toGpu();
            const auto values32 = makeVector<std::int32_t>({0, 1, 2, 3, 4, 5, 6, 7}).toGpu();

            const auto sorted = sortByKey(keys, values);
            const auto sorted32 = sortByKey(keys, values32);
            expectGpuVector(
                sorted.keys,
                {extreme_low, extreme_low, duplicate, duplicate, zero_negative_y, zero_minimum_z, zero, extreme_high});
            expectGpuVector(sorted.values,
                            {Index(2), Index(7), Index(1), Index(3), Index(4), Index(6), Index(0), Index(5)});
            expectGpuVector(sorted32.values,
                            {std::int32_t(2),
                             std::int32_t(7),
                             std::int32_t(1),
                             std::int32_t(3),
                             std::int32_t(4),
                             std::int32_t(6),
                             std::int32_t(0),
                             std::int32_t(5)});

            const auto runs = runLengthEncode(sorted.keys);
            expectGpuVector(runs.uniqueKeys,
                            {extreme_low, duplicate, zero_negative_y, zero_minimum_z, zero, extreme_high});
            expectGpuVector(runs.counts, {Index(2), Index(2), Index(1), Index(1), Index(1), Index(1)});
            EXPECT_EQ(runs.runCount.toCpu().data()[0], 6);

            const DenseStorage<Int32Key3, Device::CPU> empty_keys_cpu(0, 1);
            const DenseStorage<Index, Device::CPU> empty_values_cpu(0, 1);
            const auto empty_keys = empty_keys_cpu.toGpu();
            const auto empty_values = empty_values_cpu.toGpu();
            const auto empty_sorted = sortByKey(empty_keys, empty_values);
            const auto empty_runs = runLengthEncode(empty_keys);
            EXPECT_EQ(empty_sorted.keys.rows(), 0);
            EXPECT_EQ(empty_runs.uniqueKeys.rows(), 0);
            EXPECT_EQ(empty_runs.runCount.toCpu().data()[0], 0);
        }

        TEST(GroupingCudaTest, Int32Key3ViewsUseExternalByteStorageOnNonDefaultStream)
        {
            constexpr Index count = 8;
            constexpr std::int32_t minimum = std::numeric_limits<std::int32_t>::min();
            constexpr std::int32_t maximum = std::numeric_limits<std::int32_t>::max();
            const Int32Key3 extreme_low{minimum, maximum, -1};
            const Int32Key3 duplicate{-1, 5, 7};
            const Int32Key3 zero_negative_y{0, -1, maximum};
            const Int32Key3 zero_minimum_z{0, 0, minimum};
            const Int32Key3 zero{0, 0, 0};
            const Int32Key3 extreme_high{maximum, minimum, 0};
            const auto host_keys = makeVector<Int32Key3>(
                {zero, duplicate, extreme_low, duplicate, zero_negative_y, extreme_high, zero_minimum_z, extreme_low});
            const auto host_values = makeVector<std::int32_t>({0, 1, 2, 3, 4, 5, 6, 7});

            std::size_t byte_count = 0;
            const auto reserve = [&](std::size_t alignment, std::size_t bytes)
            {
                byte_count = (byte_count + alignment - 1) & ~(alignment - 1);
                const std::size_t offset = byte_count;
                byte_count += bytes;
                return offset;
            };
            const std::size_t input_keys_offset =
                reserve(alignof(Int32Key3), static_cast<std::size_t>(count) * sizeof(Int32Key3));
            const std::size_t input_values_offset =
                reserve(alignof(std::int32_t), static_cast<std::size_t>(count) * sizeof(std::int32_t));
            const std::size_t sorted_keys_offset =
                reserve(alignof(Int32Key3), static_cast<std::size_t>(count) * sizeof(Int32Key3));
            const std::size_t sorted_values_offset =
                reserve(alignof(std::int32_t), static_cast<std::size_t>(count) * sizeof(std::int32_t));
            const std::size_t unique_keys_offset =
                reserve(alignof(Int32Key3), static_cast<std::size_t>(count) * sizeof(Int32Key3));
            const std::size_t counts_offset = reserve(alignof(Index), static_cast<std::size_t>(count) * sizeof(Index));
            const std::size_t run_count_offset = reserve(alignof(Index), sizeof(Index));

            auto storage = DenseStorage<std::uint8_t, Device::GPU>::uninitialized(static_cast<Index>(byte_count), 1);
            auto* base = storage.data();
            auto* input_keys = reinterpret_cast<Int32Key3*>(base + input_keys_offset);
            auto* input_values = reinterpret_cast<std::int32_t*>(base + input_values_offset);
            auto* sorted_keys = reinterpret_cast<Int32Key3*>(base + sorted_keys_offset);
            auto* sorted_values = reinterpret_cast<std::int32_t*>(base + sorted_values_offset);
            auto* unique_keys = reinterpret_cast<Int32Key3*>(base + unique_keys_offset);
            auto* counts = reinterpret_cast<Index*>(base + counts_offset);
            auto* run_count = reinterpret_cast<Index*>(base + run_count_offset);

            test::CudaStreamGuard stream(cudaStreamNonBlocking);
            ASSERT_EQ(cudaMemcpyAsync(input_keys,
                                      host_keys.data(),
                                      static_cast<std::size_t>(count) * sizeof(Int32Key3),
                                      cudaMemcpyHostToDevice,
                                      stream.get()),
                      cudaSuccess);
            ASSERT_EQ(cudaMemcpyAsync(input_values,
                                      host_values.data(),
                                      static_cast<std::size_t>(count) * sizeof(std::int32_t),
                                      cudaMemcpyHostToDevice,
                                      stream.get()),
                      cudaSuccess);

            const auto key_input =
                makeColumnMajorView<Int32Key3, Device::GPU>(static_cast<const Int32Key3*>(input_keys), count, 1);
            const auto value_input = makeColumnMajorView<std::int32_t, Device::GPU>(
                static_cast<const std::int32_t*>(input_values), count, 1);
            auto key_output = makeColumnMajorView<Int32Key3, Device::GPU>(sorted_keys, count, 1);
            auto value_output = makeColumnMajorView<std::int32_t, Device::GPU>(sorted_values, count, 1);
            auto unique_output = makeColumnMajorView<Int32Key3, Device::GPU>(unique_keys, count, 1);
            auto counts_output = makeColumnMajorView<Index, Device::GPU>(counts, count, 1);
            auto run_count_output = makeColumnMajorView<Index, Device::GPU>(run_count, 1, 1);
            GroupingWorkspace workspace;

            sortByKey(key_input, value_input, key_output, value_output, workspace, stream.get());
            runLengthEncode(ConstMatrixView<Int32Key3, Device::GPU>(key_output),
                            unique_output,
                            counts_output,
                            run_count_output,
                            workspace,
                            stream.get());
            sortByKeyAsync(key_input, value_input, key_output, value_output, workspace, stream.get());
            sortByKeyAsync(key_input, value_input, key_output, value_output, workspace, stream.get());
            runLengthEncodeAsync(ConstMatrixView<Int32Key3, Device::GPU>(key_output),
                                 unique_output,
                                 counts_output,
                                 run_count_output,
                                 workspace,
                                 stream.get());
            stream.synchronize();

            std::vector<Int32Key3> actual_keys(static_cast<std::size_t>(count));
            std::vector<std::int32_t> actual_values(static_cast<std::size_t>(count));
            std::vector<Int32Key3> actual_unique(static_cast<std::size_t>(count));
            std::vector<Index> actual_counts(static_cast<std::size_t>(count));
            Index actual_run_count = 0;
            ASSERT_EQ(
                cudaMemcpy(
                    actual_keys.data(), sorted_keys, actual_keys.size() * sizeof(Int32Key3), cudaMemcpyDeviceToHost),
                cudaSuccess);
            ASSERT_EQ(cudaMemcpy(actual_values.data(),
                                 sorted_values,
                                 actual_values.size() * sizeof(std::int32_t),
                                 cudaMemcpyDeviceToHost),
                      cudaSuccess);
            ASSERT_EQ(cudaMemcpy(actual_unique.data(),
                                 unique_keys,
                                 actual_unique.size() * sizeof(Int32Key3),
                                 cudaMemcpyDeviceToHost),
                      cudaSuccess);
            ASSERT_EQ(
                cudaMemcpy(actual_counts.data(), counts, actual_counts.size() * sizeof(Index), cudaMemcpyDeviceToHost),
                cudaSuccess);
            ASSERT_EQ(cudaMemcpy(&actual_run_count, run_count, sizeof(Index), cudaMemcpyDeviceToHost), cudaSuccess);

            const std::vector<Int32Key3> expected_keys{
                extreme_low,
                extreme_low,
                duplicate,
                duplicate,
                zero_negative_y,
                zero_minimum_z,
                zero,
                extreme_high,
            };
            const std::vector<std::int32_t> expected_values{2, 7, 1, 3, 4, 6, 0, 5};
            const std::vector<Int32Key3> expected_unique{
                extreme_low, duplicate, zero_negative_y, zero_minimum_z, zero, extreme_high};
            const std::vector<Index> expected_counts{2, 2, 1, 1, 1, 1};
            EXPECT_EQ(actual_keys, expected_keys);
            EXPECT_EQ(actual_values, expected_values);
            ASSERT_EQ(actual_run_count, static_cast<Index>(expected_unique.size()));
            EXPECT_TRUE(std::equal(expected_unique.begin(), expected_unique.end(), actual_unique.begin()));
            EXPECT_TRUE(std::equal(expected_counts.begin(), expected_counts.end(), actual_counts.begin()));

            workspace.closeAsyncAllocation();
            stream.synchronize();
        }

        TEST(GroupingCudaTest, NumericViewsSortEncodeAndSegmentWithoutAllocatingResults)
        {
            constexpr Index count = 4;
            const auto host_keys = makeVector<std::uint64_t>({3, 1, 3, 1});
            const auto host_values = makeVector<Index>({30, 10, 31, 11});
            const auto host_offsets = makeVector<Index>({0, 2, 4});

            std::size_t byte_count = 0;
            const auto reserve = [&](std::size_t alignment, std::size_t bytes)
            {
                byte_count = (byte_count + alignment - 1) & ~(alignment - 1);
                const std::size_t offset = byte_count;
                byte_count += bytes;
                return offset;
            };
            const auto key_offset = reserve(alignof(std::uint64_t), count * sizeof(std::uint64_t));
            const auto value_offset = reserve(alignof(Index), count * sizeof(Index));
            const auto sorted_key_offset = reserve(alignof(std::uint64_t), count * sizeof(std::uint64_t));
            const auto sorted_value_offset = reserve(alignof(Index), count * sizeof(Index));
            const auto unique_offset = reserve(alignof(std::uint64_t), count * sizeof(std::uint64_t));
            const auto counts_offset = reserve(alignof(Index), count * sizeof(Index));
            const auto run_count_offset = reserve(alignof(Index), sizeof(Index));
            const auto offsets_offset = reserve(alignof(Index), 3 * sizeof(Index));
            const auto reduced_offset = reserve(alignof(Index), 2 * sizeof(Index));

            auto storage = DenseStorage<std::uint8_t, Device::GPU>::uninitialized(static_cast<Index>(byte_count), 1);
            auto* base = storage.data();
            auto* keys = reinterpret_cast<std::uint64_t*>(base + key_offset);
            auto* values = reinterpret_cast<Index*>(base + value_offset);
            auto* sorted_keys = reinterpret_cast<std::uint64_t*>(base + sorted_key_offset);
            auto* sorted_values = reinterpret_cast<Index*>(base + sorted_value_offset);
            auto* unique_keys = reinterpret_cast<std::uint64_t*>(base + unique_offset);
            auto* counts = reinterpret_cast<Index*>(base + counts_offset);
            auto* run_count = reinterpret_cast<Index*>(base + run_count_offset);
            auto* offsets = reinterpret_cast<Index*>(base + offsets_offset);
            auto* reduced = reinterpret_cast<Index*>(base + reduced_offset);

            test::CudaStreamGuard stream(cudaStreamNonBlocking);
            ASSERT_EQ(cudaMemcpyAsync(
                          keys, host_keys.data(), count * sizeof(std::uint64_t), cudaMemcpyHostToDevice, stream.get()),
                      cudaSuccess);
            ASSERT_EQ(cudaMemcpyAsync(
                          values, host_values.data(), count * sizeof(Index), cudaMemcpyHostToDevice, stream.get()),
                      cudaSuccess);
            ASSERT_EQ(
                cudaMemcpyAsync(offsets, host_offsets.data(), 3 * sizeof(Index), cudaMemcpyHostToDevice, stream.get()),
                cudaSuccess);

            const auto key_input =
                makeColumnMajorView<std::uint64_t, Device::GPU>(static_cast<const std::uint64_t*>(keys), count, 1);
            const auto value_input =
                makeColumnMajorView<Index, Device::GPU>(static_cast<const Index*>(values), count, 1);
            auto key_output = makeColumnMajorView<std::uint64_t, Device::GPU>(sorted_keys, count, 1);
            auto value_output = makeColumnMajorView<Index, Device::GPU>(sorted_values, count, 1);
            auto unique_output = makeColumnMajorView<std::uint64_t, Device::GPU>(unique_keys, count, 1);
            auto counts_output = makeColumnMajorView<Index, Device::GPU>(counts, count, 1);
            auto run_count_output = makeColumnMajorView<Index, Device::GPU>(run_count, 1, 1);
            const auto offsets_input =
                makeColumnMajorView<Index, Device::GPU>(static_cast<const Index*>(offsets), 3, 1);
            auto reduced_output = makeColumnMajorView<Index, Device::GPU>(reduced, 2, 1);
            GroupingWorkspace workspace;

            sortByKeyAsync(key_input, value_input, key_output, value_output, workspace, stream.get());
            runLengthEncodeAsync(ConstMatrixView<std::uint64_t, Device::GPU>(key_output),
                                 unique_output,
                                 counts_output,
                                 run_count_output,
                                 workspace,
                                 stream.get());
            segmentedReduceAsync(ConstMatrixView<Index, Device::GPU>(value_output),
                                 offsets_input,
                                 GroupReduction::Sum,
                                 reduced_output,
                                 workspace,
                                 stream.get());
            stream.synchronize();

            std::vector<std::uint64_t> actual_keys(count);
            std::vector<Index> actual_values(count);
            std::vector<std::uint64_t> actual_unique(count);
            std::vector<Index> actual_counts(count);
            std::vector<Index> actual_reduced(2);
            Index actual_run_count = 0;
            ASSERT_EQ(
                cudaMemcpy(actual_keys.data(), sorted_keys, count * sizeof(std::uint64_t), cudaMemcpyDeviceToHost),
                cudaSuccess);
            ASSERT_EQ(cudaMemcpy(actual_values.data(), sorted_values, count * sizeof(Index), cudaMemcpyDeviceToHost),
                      cudaSuccess);
            ASSERT_EQ(
                cudaMemcpy(actual_unique.data(), unique_keys, count * sizeof(std::uint64_t), cudaMemcpyDeviceToHost),
                cudaSuccess);
            ASSERT_EQ(cudaMemcpy(actual_counts.data(), counts, count * sizeof(Index), cudaMemcpyDeviceToHost),
                      cudaSuccess);
            ASSERT_EQ(cudaMemcpy(actual_reduced.data(), reduced, 2 * sizeof(Index), cudaMemcpyDeviceToHost),
                      cudaSuccess);
            ASSERT_EQ(cudaMemcpy(&actual_run_count, run_count, sizeof(Index), cudaMemcpyDeviceToHost), cudaSuccess);
            EXPECT_EQ(actual_keys, (std::vector<std::uint64_t>{1, 1, 3, 3}));
            EXPECT_EQ(actual_values, (std::vector<Index>{10, 11, 30, 31}));
            EXPECT_EQ(actual_run_count, 2);
            EXPECT_EQ(actual_unique[0], 1);
            EXPECT_EQ(actual_unique[1], 3);
            EXPECT_EQ(actual_counts[0], 2);
            EXPECT_EQ(actual_counts[1], 2);
            EXPECT_EQ(actual_reduced, (std::vector<Index>{21, 61}));

            sortByKey(key_input, value_input, key_output, value_output, workspace, stream.get());
            runLengthEncode(ConstMatrixView<std::uint64_t, Device::GPU>(key_output),
                            unique_output,
                            counts_output,
                            run_count_output,
                            workspace,
                            stream.get());
            segmentedReduce(ConstMatrixView<Index, Device::GPU>(value_output),
                            offsets_input,
                            GroupReduction::Maximum,
                            reduced_output,
                            workspace,
                            stream.get());
            ASSERT_EQ(cudaMemcpy(actual_reduced.data(), reduced, 2 * sizeof(Index), cudaMemcpyDeviceToHost),
                      cudaSuccess);
            EXPECT_EQ(actual_reduced, (std::vector<Index>{11, 31}));

            EXPECT_THROW(sortByKey(key_input,
                                   value_input,
                                   makeColumnMajorView<std::uint64_t, Device::GPU>(keys, count, 1),
                                   value_output,
                                   workspace,
                                   stream.get()),
                         std::invalid_argument);
            EXPECT_THROW(
                sortByKey(key_input,
                          makeColumnMajorView<Index, Device::GPU>(static_cast<const Index*>(values), count - 1, 1),
                          key_output,
                          value_output,
                          workspace,
                          stream.get()),
                std::invalid_argument);
            EXPECT_THROW(sortByKey(key_input,
                                   value_input,
                                   key_output,
                                   MatrixView<Index, Device::GPU>(sorted_values, count, 1, 2, count * 2),
                                   workspace,
                                   stream.get()),
                         std::invalid_argument);
            EXPECT_THROW(runLengthEncode(ConstMatrixView<std::uint64_t, Device::GPU>(key_output),
                                         key_output,
                                         counts_output,
                                         run_count_output,
                                         workspace,
                                         stream.get()),
                         std::invalid_argument);
            EXPECT_THROW(runLengthEncode(ConstMatrixView<std::uint64_t, Device::GPU>(key_output),
                                         unique_output,
                                         makeColumnMajorView<Index, Device::GPU>(counts, count - 1, 1),
                                         run_count_output,
                                         workspace,
                                         stream.get()),
                         std::invalid_argument);
            EXPECT_THROW(segmentedReduce(ConstMatrixView<Index, Device::GPU>(value_output),
                                         offsets_input,
                                         GroupReduction::Sum,
                                         makeColumnMajorView<Index, Device::GPU>(sorted_values, 2, 1),
                                         workspace,
                                         stream.get()),
                         std::invalid_argument);
            EXPECT_THROW(segmentedReduce(ConstMatrixView<Index, Device::GPU>(value_output),
                                         offsets_input,
                                         GroupReduction::Sum,
                                         makeColumnMajorView<Index, Device::GPU>(reduced, 1, 1),
                                         workspace,
                                         stream.get()),
                         std::invalid_argument);
            workspace.closeAsyncAllocation();
            stream.synchronize();
        }

        TEST(GroupingCudaTest, EmptyNumericViewsPreserveZeroRunCount)
        {
            test::CudaStreamGuard stream(cudaStreamNonBlocking);
            GroupingWorkspace workspace;
            auto run_count_storage = DenseStorage<Index, Device::GPU>::uninitialized(1, 1);
            const auto offset_storage = makeVector<Index>({0}).toGpu();
            const auto keys =
                makeColumnMajorView<std::uint32_t, Device::GPU>(static_cast<const std::uint32_t*>(nullptr), 0, 1);
            const auto values = makeColumnMajorView<float, Device::GPU>(static_cast<const float*>(nullptr), 0, 1);
            auto sorted_keys =
                makeColumnMajorView<std::uint32_t, Device::GPU>(static_cast<std::uint32_t*>(nullptr), 0, 1);
            auto sorted_values = makeColumnMajorView<float, Device::GPU>(static_cast<float*>(nullptr), 0, 1);
            auto counts = makeColumnMajorView<Index, Device::GPU>(static_cast<Index*>(nullptr), 0, 1);
            auto run_count = makeColumnMajorView<Index, Device::GPU>(run_count_storage.data(), 1, 1);
            auto empty_reduced = makeColumnMajorView<float, Device::GPU>(static_cast<float*>(nullptr), 0, 1);

            sortByKeyAsync(keys, values, sorted_keys, sorted_values, workspace, stream.get());
            runLengthEncodeAsync(keys, sorted_keys, counts, run_count, workspace, stream.get());
            segmentedReduceAsync(
                values, offset_storage.view(), GroupReduction::Sum, empty_reduced, workspace, stream.get());
            stream.synchronize();
            EXPECT_EQ(run_count_storage.toCpu().data()[0], 0);
            workspace.closeAsyncAllocation();
            stream.synchronize();
        }

    } // namespace
} // namespace plamatrix::internal

#endif
