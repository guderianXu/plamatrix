#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>

#include <gtest/gtest.h>

#include "plamatrix/internal/ops/grouping.h"

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
        void expectVector(const DenseStorage<Scalar, Device::CPU>& actual, std::initializer_list<Scalar> expected)
        {
            ASSERT_EQ(actual.rows(), static_cast<Index>(expected.size()));
            ASSERT_EQ(actual.cols(), 1);
            Index offset = 0;
            for (const Scalar value : expected)
            {
                EXPECT_EQ(actual.data()[offset++], value);
            }
        }

        TEST(GroupingTest, StableSortPreservesDuplicateInputOrder)
        {
            const auto keys = makeVector<std::uint32_t>({3, 1, 3, 2, 1});
            const auto values = makeVector<Index>({0, 1, 2, 3, 4});

            const auto result = sortByKey(keys, values);

            expectVector(result.keys, {1U, 1U, 2U, 3U, 3U});
            expectVector(result.values, {Index(1), Index(4), Index(3), Index(0), Index(2)});
        }

        TEST(GroupingTest, RunLengthEncodePreservesRunsAndHandlesEmptyInput)
        {
            const auto keys = makeVector<std::uint64_t>({1, 1, 4, 4, 4, 2});
            const auto result = runLengthEncode(keys);

            expectVector(result.uniqueKeys, {std::uint64_t(1), std::uint64_t(4), std::uint64_t(2)});
            expectVector(result.counts, {Index(2), Index(3), Index(1)});
            EXPECT_EQ(result.runCount.data()[0], 3);

            const DenseStorage<std::uint64_t, Device::CPU> empty(0, 1);
            const auto empty_result = runLengthEncode(empty);
            EXPECT_EQ(empty_result.uniqueKeys.rows(), 0);
            EXPECT_EQ(empty_result.counts.rows(), 0);
            EXPECT_EQ(empty_result.runCount.data()[0], 0);

            const DenseStorage<float, Device::CPU> empty_values(0, 1);
            const auto empty_sorted = sortByKey(empty, empty_values);
            const auto empty_reduced = reduceByKey(empty, empty_values, GroupReduction::Sum);
            const auto empty_segments = segmentedReduce(empty_values, makeVector<Index>({0}), GroupReduction::Sum);
            EXPECT_EQ(empty_sorted.keys.rows(), 0);
            EXPECT_EQ(empty_reduced.runCount.data()[0], 0);
            EXPECT_EQ(empty_segments.rows(), 0);
        }

        TEST(GroupingTest, Int32Key3SortAndRunLengthEncodeAreLexicographicStableAndCollisionFree)
        {
            constexpr std::int32_t minimum = std::numeric_limits<std::int32_t>::min();
            constexpr std::int32_t maximum = std::numeric_limits<std::int32_t>::max();
            const Int32Key3 extreme_low{minimum, maximum, -1};
            const Int32Key3 duplicate{-1, 5, 7};
            const Int32Key3 zero_negative_y{0, -1, maximum};
            const Int32Key3 zero_minimum_z{0, 0, minimum};
            const Int32Key3 zero{0, 0, 0};
            const Int32Key3 extreme_high{maximum, minimum, 0};
            const auto keys = makeVector<Int32Key3>(
                {zero, duplicate, extreme_low, duplicate, zero_negative_y, extreme_high, zero_minimum_z, extreme_low});
            const auto values = makeVector<Index>({0, 1, 2, 3, 4, 5, 6, 7});

            const auto sorted = sortByKey(keys, values);
            const auto values32 = makeVector<std::int32_t>({0, 1, 2, 3, 4, 5, 6, 7});
            const auto sorted32 = sortByKey(keys, values32);

            expectVector(
                sorted.keys,
                {extreme_low, extreme_low, duplicate, duplicate, zero_negative_y, zero_minimum_z, zero, extreme_high});
            expectVector(sorted.values,
                         {Index(2), Index(7), Index(1), Index(3), Index(4), Index(6), Index(0), Index(5)});
            expectVector(sorted32.values,
                         {std::int32_t(2),
                          std::int32_t(7),
                          std::int32_t(1),
                          std::int32_t(3),
                          std::int32_t(4),
                          std::int32_t(6),
                          std::int32_t(0),
                          std::int32_t(5)});

            const auto runs = runLengthEncode(sorted.keys);
            expectVector(runs.uniqueKeys,
                         {extreme_low, duplicate, zero_negative_y, zero_minimum_z, zero, extreme_high});
            expectVector(runs.counts, {Index(2), Index(2), Index(1), Index(1), Index(1), Index(1)});
            EXPECT_EQ(runs.runCount.data()[0], 6);

            const DenseStorage<Int32Key3, Device::CPU> empty_keys(0, 1);
            const DenseStorage<Index, Device::CPU> empty_values(0, 1);
            const auto empty_sorted = sortByKey(empty_keys, empty_values);
            const auto empty_runs = runLengthEncode(empty_keys);
            EXPECT_EQ(empty_sorted.keys.rows(), 0);
            EXPECT_EQ(empty_runs.uniqueKeys.rows(), 0);
            EXPECT_EQ(empty_runs.runCount.data()[0], 0);
        }

#ifndef PLAMATRIX_WITH_CUDA
        TEST(GroupingTest, Int32Key3ViewEntryPointsHaveNoCudaStubs)
        {
            const auto keys = makeColumnMajorView<Int32Key3, Device::GPU>(static_cast<const Int32Key3*>(nullptr), 0, 1);
            const auto values =
                makeColumnMajorView<std::int32_t, Device::GPU>(static_cast<const std::int32_t*>(nullptr), 0, 1);
            auto sorted_keys = makeColumnMajorView<Int32Key3, Device::GPU>(static_cast<Int32Key3*>(nullptr), 0, 1);
            auto sorted_values =
                makeColumnMajorView<std::int32_t, Device::GPU>(static_cast<std::int32_t*>(nullptr), 0, 1);
            auto counts = makeColumnMajorView<Index, Device::GPU>(static_cast<Index*>(nullptr), 0, 1);
            Index run_count_storage = 0;
            auto run_count = makeColumnMajorView<Index, Device::GPU>(&run_count_storage, 1, 1);
            GroupingWorkspace workspace;

            EXPECT_THROW(sortByKey(keys, values, sorted_keys, sorted_values, workspace), std::runtime_error);
            EXPECT_THROW(runLengthEncode(keys, sorted_keys, counts, run_count, workspace), std::runtime_error);
        }

        TEST(GroupingTest, NumericViewEntryPointsHaveNoCudaStubs)
        {
            const auto keys =
                makeColumnMajorView<std::uint64_t, Device::GPU>(static_cast<const std::uint64_t*>(nullptr), 0, 1);
            const auto values = makeColumnMajorView<Index, Device::GPU>(static_cast<const Index*>(nullptr), 0, 1);
            auto sorted_keys =
                makeColumnMajorView<std::uint64_t, Device::GPU>(static_cast<std::uint64_t*>(nullptr), 0, 1);
            auto sorted_values = makeColumnMajorView<Index, Device::GPU>(static_cast<Index*>(nullptr), 0, 1);
            auto counts = makeColumnMajorView<Index, Device::GPU>(static_cast<Index*>(nullptr), 0, 1);
            Index run_count_storage = 0;
            auto run_count = makeColumnMajorView<Index, Device::GPU>(&run_count_storage, 1, 1);
            const auto offsets =
                makeColumnMajorView<Index, Device::GPU>(static_cast<const Index*>(&run_count_storage), 1, 1);
            GroupingWorkspace workspace;

            EXPECT_THROW(sortByKey(keys, values, sorted_keys, sorted_values, workspace), std::runtime_error);
            EXPECT_THROW(sortByKeyAsync(keys, values, sorted_keys, sorted_values, workspace, nullptr),
                         std::runtime_error);
            EXPECT_THROW(runLengthEncode(keys, sorted_keys, counts, run_count, workspace), std::runtime_error);
            EXPECT_THROW(runLengthEncodeAsync(keys, sorted_keys, counts, run_count, workspace, nullptr),
                         std::runtime_error);
            EXPECT_THROW(segmentedReduce(values, offsets, GroupReduction::Sum, sorted_values, workspace),
                         std::runtime_error);
            EXPECT_THROW(segmentedReduceAsync(values, offsets, GroupReduction::Sum, sorted_values, workspace, nullptr),
                         std::runtime_error);
        }
#endif

        TEST(GroupingTest, ReduceByKeySupportsSumMinimumAndMaximum)
        {
            const auto keys = makeVector<Index>({1, 1, 2, 2, 2, 5});
            const auto values = makeVector<double>({3.0, -1.0, 4.0, 2.0, 8.0, 7.0});

            const auto sum_result = reduceByKey(keys, values, GroupReduction::Sum);
            const auto minimum_result = reduceByKey(keys, values, GroupReduction::Minimum);
            const auto maximum_result = reduceByKey(keys, values, GroupReduction::Maximum);

            expectVector(sum_result.uniqueKeys, {Index(1), Index(2), Index(5)});
            expectVector(sum_result.aggregates, {2.0, 14.0, 7.0});
            expectVector(minimum_result.aggregates, {-1.0, 2.0, 7.0});
            expectVector(maximum_result.aggregates, {3.0, 8.0, 7.0});
            EXPECT_EQ(sum_result.runCount.data()[0], 3);
        }

        TEST(GroupingTest, SegmentedReduceDefinesEmptySegmentIdentities)
        {
            const auto values = makeVector<Index>({3, -1, 4, 2, 8});
            const auto offsets = makeVector<Index>({0, 2, 2, 5});

            expectVector(segmentedReduce(values, offsets, GroupReduction::Sum), {Index(2), Index(0), Index(14)});
            expectVector(segmentedReduce(values, offsets, GroupReduction::Minimum),
                         {Index(-1), std::numeric_limits<Index>::max(), Index(2)});
            expectVector(segmentedReduce(values, offsets, GroupReduction::Maximum),
                         {Index(3), std::numeric_limits<Index>::lowest(), Index(8)});
        }

        TEST(GroupingTest, RejectsInvalidShapesOffsetsAndOperations)
        {
            DenseStorage<std::uint32_t, Device::CPU> matrix_keys(2, 2);
            const auto values = makeVector<float>({1.0F, 2.0F});
            EXPECT_THROW(sortByKey(matrix_keys, values), std::invalid_argument);

            const auto segment_values = makeVector<float>({1.0F, 2.0F, 3.0F});
            EXPECT_THROW(segmentedReduce(segment_values, makeVector<Index>({1, 3}), GroupReduction::Sum),
                         std::invalid_argument);
            EXPECT_THROW(segmentedReduce(segment_values, makeVector<Index>({0, 2, 1, 3}), GroupReduction::Sum),
                         std::invalid_argument);
            EXPECT_THROW(segmentedReduce(segment_values, makeVector<Index>({0, 3}), static_cast<GroupReduction>(99)),
                         std::invalid_argument);
        }

        TEST(GroupingTest, SupportedKeyValueCombinationsAreLinkable)
        {
            const auto unsigned_result = sortByKey(makeVector<std::uint64_t>({2, 1}), makeVector<double>({20.0, 10.0}));
            expectVector(unsigned_result.values, {10.0, 20.0});

            const auto signed_result = sortByKey(makeVector<Index>({2, 1}), makeVector<Index>({20, 10}));
            expectVector(signed_result.values, {Index(10), Index(20)});
        }

    } // namespace
} // namespace plamatrix::internal
