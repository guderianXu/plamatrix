#include <cstdint>
#include <limits>
#include <type_traits>

#include "indexing_test_utils.h"

namespace plamatrix
{
namespace
{

using indexing_test_detail::expectMatrix;
using indexing_test_detail::IndexingTest;
using indexing_test_detail::makeMatrix;
using indexing_test_detail::ScalarTypes;

TYPED_TEST_SUITE(IndexingTest, ScalarTypes);

static_assert(!std::is_copy_constructible_v<IndexingWorkspace>);
static_assert(!std::is_copy_assignable_v<IndexingWorkspace>);
static_assert(std::is_nothrow_move_constructible_v<IndexingWorkspace>);
static_assert(std::is_nothrow_move_assignable_v<IndexingWorkspace>);
static_assert(std::is_nothrow_destructible_v<IndexingWorkspace>);

TEST(IndexingScanTest, UsesExclusiveColumnMajorLinearOrderAndAllowsNegativeCounts)
{
    const auto counts = makeMatrix<Index>(2, 3, {2, -1, 3, 4, -2, 1});

    const auto offsets = exclusiveScan(counts);

    expectMatrix(offsets, 2, 3, {
        Index(0), Index(2), Index(1), Index(4), Index(8), Index(6)
    });
}

TEST(IndexingScanTest, PreservesEmptyShape)
{
    const DenseMatrix<Index, Device::CPU> counts(0, 3);

    const auto offsets = exclusiveScan(counts);

    expectMatrix(offsets, 0, 3, {});
}

TEST(IndexingScanTest, RejectsPositiveOverflow)
{
    const auto counts = makeMatrix<Index>(2, 1, {
        std::numeric_limits<Index>::max(), Index(1)
    });

    EXPECT_THROW(static_cast<void>(exclusiveScan(counts)), std::overflow_error);
}

TEST(IndexingScanTest, RejectsNegativeOverflow)
{
    const auto counts = makeMatrix<Index>(2, 1, {
        std::numeric_limits<Index>::min(), Index(-1)
    });

    EXPECT_THROW(static_cast<void>(exclusiveScan(counts)), std::overflow_error);
}

TEST(IndexingScanTest, AcceptsExactPositiveAndNegativeLimitsFollowedByZero)
{
    const auto positive = makeMatrix<Index>(2, 1, {
        std::numeric_limits<Index>::max(), Index(0)
    });
    const auto negative = makeMatrix<Index>(2, 1, {
        std::numeric_limits<Index>::min(), Index(0)
    });

    const auto positive_offsets = exclusiveScan(positive);
    const auto negative_offsets = exclusiveScan(negative);

    expectMatrix(positive_offsets, 2, 1, {
        Index(0), std::numeric_limits<Index>::max()
    });
    expectMatrix(negative_offsets, 2, 1, {
        Index(0), std::numeric_limits<Index>::min()
    });
}

TYPED_TEST(IndexingTest, GatherRowsPreservesOrderDuplicatesAndAllColumns)
{
    using Scalar = TypeParam;
    const auto input = makeMatrix<Scalar>(4, 3, {
        Scalar(10), Scalar(11), Scalar(12), Scalar(13),
        Scalar(20), Scalar(21), Scalar(22), Scalar(23),
        Scalar(30), Scalar(31), Scalar(32), Scalar(33)
    });
    const auto indices = makeMatrix<Index>(3, 1, {2, 0, 2});

    const auto gathered = gatherRows(input, indices);

    expectMatrix(gathered, 3, 3, {
        Scalar(12), Scalar(10), Scalar(12),
        Scalar(22), Scalar(20), Scalar(22),
        Scalar(32), Scalar(30), Scalar(32)
    });
}

TYPED_TEST(IndexingTest, GatherRowsSupportsEmptySelection)
{
    using Scalar = TypeParam;
    const DenseMatrix<Scalar, Device::CPU> input(4, 2);
    const DenseMatrix<Index, Device::CPU> indices(0, 1);

    const auto gathered = gatherRows(input, indices);

    expectMatrix(gathered, 0, 2, {});
}

TYPED_TEST(IndexingTest, GatherRowsSupportsMaximumRowsWithZeroColumns)
{
    using Scalar = TypeParam;
    const Index maximum = std::numeric_limits<Index>::max();
    const DenseMatrix<Scalar, Device::CPU> input(maximum, 0);
    const auto indices = makeMatrix<Index>(2, 1, {maximum - 1, Index(0)});

    const auto gathered = gatherRows(input, indices);

    expectMatrix(gathered, 2, 0, {});
}

TYPED_TEST(IndexingTest, GatherRowsRejectsOutOfRangeIndices)
{
    using Scalar = TypeParam;
    const DenseMatrix<Scalar, Device::CPU> input(3, 1);
    const auto negative = makeMatrix<Index>(1, 1, {-1});
    const auto too_large = makeMatrix<Index>(1, 1, {3});

    EXPECT_THROW(static_cast<void>(gatherRows(input, negative)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(gatherRows(input, too_large)), std::out_of_range);
}

TYPED_TEST(IndexingTest, GatherRowsRequiresColumnVectorIndices)
{
    using Scalar = TypeParam;
    const DenseMatrix<Scalar, Device::CPU> input(3, 1);
    const DenseMatrix<Index, Device::CPU> indices(1, 2);

    EXPECT_THROW(static_cast<void>(gatherRows(input, indices)), std::invalid_argument);
}

TYPED_TEST(IndexingTest, ScatterRowsUsesLowestSourceAndPreservesUntouchedRows)
{
    using Scalar = TypeParam;
    const auto values = makeMatrix<Scalar>(3, 2, {
        Scalar(100), Scalar(110), Scalar(120),
        Scalar(200), Scalar(210), Scalar(220)
    });
    const auto indices = makeMatrix<Index>(3, 1, {2, 0, 2});
    auto output = makeMatrix<Scalar>(4, 2, {
        Scalar(1), Scalar(2), Scalar(3), Scalar(4),
        Scalar(5), Scalar(6), Scalar(7), Scalar(8)
    });

    scatterRows(values, indices, output);

    expectMatrix(output, 4, 2, {
        Scalar(110), Scalar(2), Scalar(100), Scalar(4),
        Scalar(210), Scalar(6), Scalar(200), Scalar(8)
    });
}

TYPED_TEST(IndexingTest, ScatterRowsSupportsEmptySelection)
{
    using Scalar = TypeParam;
    const DenseMatrix<Scalar, Device::CPU> values(0, 2);
    const DenseMatrix<Index, Device::CPU> indices(0, 1);
    auto output = makeMatrix<Scalar>(2, 2, {
        Scalar(1), Scalar(2), Scalar(3), Scalar(4)
    });

    EXPECT_NO_THROW(scatterRows(values, indices, output));
    expectMatrix(output, 2, 2, {
        Scalar(1), Scalar(2), Scalar(3), Scalar(4)
    });
}

TYPED_TEST(IndexingTest, ScatterRowsDoesNotAllocateByOutputRowsForZeroColumns)
{
    using Scalar = TypeParam;
    const Index maximum = std::numeric_limits<Index>::max();
    const DenseMatrix<Scalar, Device::CPU> values(2, 0);
    const auto indices = makeMatrix<Index>(2, 1, {maximum - 1, Index(0)});
    DenseMatrix<Scalar, Device::CPU> output(maximum, 0);

    EXPECT_NO_THROW(scatterRows(values, indices, output));
    expectMatrix(output, maximum, 0, {});
}

TYPED_TEST(IndexingTest, ScatterRowsRejectsShapeMismatches)
{
    using Scalar = TypeParam;
    const DenseMatrix<Scalar, Device::CPU> wrong_rows(2, 2);
    const DenseMatrix<Scalar, Device::CPU> wrong_cols(3, 1);
    const DenseMatrix<Index, Device::CPU> indices(3, 1);
    const DenseMatrix<Index, Device::CPU> wrong_indices(1, 3);
    DenseMatrix<Scalar, Device::CPU> output(4, 2);

    EXPECT_THROW(scatterRows(wrong_rows, indices, output), std::invalid_argument);
    EXPECT_THROW(scatterRows(wrong_cols, indices, output), std::invalid_argument);
    EXPECT_THROW(scatterRows(wrong_rows, wrong_indices, output), std::invalid_argument);
}

TYPED_TEST(IndexingTest, ScatterRowsValidatesAllIndicesBeforeWriting)
{
    using Scalar = TypeParam;
    const auto values = makeMatrix<Scalar>(2, 2, {
        Scalar(100), Scalar(110), Scalar(200), Scalar(210)
    });
    const auto indices = makeMatrix<Index>(2, 1, {0, 3});
    auto output = makeMatrix<Scalar>(3, 2, {
        Scalar(1), Scalar(2), Scalar(3), Scalar(4), Scalar(5), Scalar(6)
    });

    EXPECT_THROW(scatterRows(values, indices, output), std::out_of_range);
    expectMatrix(output, 3, 2, {
        Scalar(1), Scalar(2), Scalar(3), Scalar(4), Scalar(5), Scalar(6)
    });
}

TYPED_TEST(IndexingTest, CompactRowsIsStableAndTreatsNonzeroMaskAsKeep)
{
    using Scalar = TypeParam;
    const auto input = makeMatrix<Scalar>(5, 2, {
        Scalar(10), Scalar(11), Scalar(12), Scalar(13), Scalar(14),
        Scalar(20), Scalar(21), Scalar(22), Scalar(23), Scalar(24)
    });
    const auto mask = makeMatrix<std::uint8_t>(5, 1, {0, 2, 0, 255, 1});

    const auto compacted = compactRows(input, mask);

    expectMatrix(compacted.values, 3, 2, {
        Scalar(11), Scalar(13), Scalar(14),
        Scalar(21), Scalar(23), Scalar(24)
    });
    expectMatrix(compacted.sourceIndices, 3, 1, {Index(1), Index(3), Index(4)});
}

TYPED_TEST(IndexingTest, CompactRowsSupportsNoneAndAllMasks)
{
    using Scalar = TypeParam;
    const auto input = makeMatrix<Scalar>(3, 2, {
        Scalar(1), Scalar(2), Scalar(3), Scalar(4), Scalar(5), Scalar(6)
    });
    const auto none_mask = makeMatrix<std::uint8_t>(3, 1, {0, 0, 0});
    const auto all_mask = makeMatrix<std::uint8_t>(3, 1, {1, 1, 1});

    const auto none = compactRows(input, none_mask);
    const auto all = compactRows(input, all_mask);

    expectMatrix(none.values, 0, 2, {});
    expectMatrix(none.sourceIndices, 0, 1, {});
    expectMatrix(all.values, 3, 2, {
        Scalar(1), Scalar(2), Scalar(3), Scalar(4), Scalar(5), Scalar(6)
    });
    expectMatrix(all.sourceIndices, 3, 1, {Index(0), Index(1), Index(2)});
}

TYPED_TEST(IndexingTest, CompactRowsSupportsEmptyInput)
{
    using Scalar = TypeParam;
    const DenseMatrix<Scalar, Device::CPU> input(0, 3);
    const DenseMatrix<std::uint8_t, Device::CPU> mask(0, 1);

    const auto compacted = compactRows(input, mask);

    expectMatrix(compacted.values, 0, 3, {});
    expectMatrix(compacted.sourceIndices, 0, 1, {});
}

TYPED_TEST(IndexingTest, CompactRowsPreservesSelectedIndicesWithZeroColumns)
{
    using Scalar = TypeParam;
    const DenseMatrix<Scalar, Device::CPU> input(4, 0);
    const auto mask = makeMatrix<std::uint8_t>(4, 1, {0, 1, 0, 2});

    const auto compacted = compactRows(input, mask);

    expectMatrix(compacted.values, 2, 0, {});
    expectMatrix(compacted.sourceIndices, 2, 1, {Index(1), Index(3)});
}

TYPED_TEST(IndexingTest, CompactRowsRejectsMaskShapeMismatch)
{
    using Scalar = TypeParam;
    const DenseMatrix<Scalar, Device::CPU> input(3, 2);
    const DenseMatrix<std::uint8_t, Device::CPU> wrong_rows(2, 1);
    const DenseMatrix<std::uint8_t, Device::CPU> wrong_cols(3, 2);

    EXPECT_THROW(static_cast<void>(compactRows(input, wrong_rows)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(compactRows(input, wrong_cols)), std::invalid_argument);
}

} // namespace
} // namespace plamatrix
