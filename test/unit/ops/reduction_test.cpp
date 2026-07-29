#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>

#include "plamatrix/core/parallel.h"
#include "plamatrix/ops/reduction.h"
#include "../../support/cuda_test_utils.h"

namespace plamatrix
{
namespace
{

template <typename Scalar>
class ReductionTest : public ::testing::Test
{
protected:
    static DenseMatrix<Scalar, Device::CPU> makeMatrix(
        Index rows,
        Index cols,
        std::initializer_list<Scalar> values)
    {
        DenseMatrix<Scalar, Device::CPU> matrix(rows, cols);
        if (matrix.size() != static_cast<Index>(values.size()))
        {
            throw std::invalid_argument("makeMatrix: initializer length must match matrix size");
        }
        Index index = 0;
        for (Scalar value : values)
        {
            matrix.data()[index++] = value;
        }
        return matrix;
    }

    template <typename Value>
    static void expectValues(const DenseMatrix<Value, Device::CPU>& matrix,
                             std::initializer_list<Value> expected)
    {
        ASSERT_EQ(matrix.size(), static_cast<Index>(expected.size()));
        Index index = 0;
        for (Value value : expected)
        {
            EXPECT_EQ(matrix.data()[index++], value);
        }
    }

    template <typename Value>
    static void expectShape(const DenseMatrix<Value, Device::CPU>& matrix,
                            Index rows,
                            Index cols)
    {
        EXPECT_EQ(matrix.rows(), rows);
        EXPECT_EQ(matrix.cols(), cols);
    }
};

#if defined(PLAMATRIX_USE_FLOAT) && defined(PLAMATRIX_USE_DOUBLE)
using ScalarTypes = ::testing::Types<float, double>;
#elif defined(PLAMATRIX_USE_FLOAT)
using ScalarTypes = ::testing::Types<float>;
#elif defined(PLAMATRIX_USE_DOUBLE)
using ScalarTypes = ::testing::Types<double>;
#else
#error "Reduction tests require PLAMATRIX_USE_FLOAT or PLAMATRIX_USE_DOUBLE"
#endif
TYPED_TEST_SUITE(ReductionTest, ScalarTypes);

static_assert(!std::is_copy_constructible_v<ReductionWorkspace>);
static_assert(!std::is_copy_assignable_v<ReductionWorkspace>);
static_assert(std::is_nothrow_move_constructible_v<ReductionWorkspace>);
static_assert(std::is_nothrow_move_assignable_v<ReductionWorkspace>);
static_assert(std::is_nothrow_destructible_v<ReductionWorkspace>);

#ifdef PLAMATRIX_WITH_CUDA
template <typename Scalar>
void expectGpuReductionResult(
    const DenseMatrix<Scalar, Device::GPU>& actual,
    const DenseMatrix<Scalar, Device::CPU>& expected,
    Scalar tolerance = Scalar(0))
{
    const auto actual_cpu = actual.toCpu();
    ASSERT_EQ(actual_cpu.rows(), expected.rows());
    ASSERT_EQ(actual_cpu.cols(), expected.cols());
    for (Index index = 0; index < expected.size(); ++index)
    {
        if (std::isnan(expected.data()[index]))
        {
            EXPECT_TRUE(std::isnan(actual_cpu.data()[index]));
        }
        else if (std::isinf(expected.data()[index]))
        {
            EXPECT_EQ(actual_cpu.data()[index], expected.data()[index]);
        }
        else if (tolerance == Scalar(0))
        {
            EXPECT_EQ(actual_cpu.data()[index], expected.data()[index]);
        }
        else
        {
            const Scalar scale = std::max(Scalar(1), std::abs(expected.data()[index]));
            EXPECT_NEAR(actual_cpu.data()[index], expected.data()[index], tolerance * scale);
        }
    }
}

void expectGpuIndices(
    const DenseMatrix<Index, Device::GPU>& actual,
    const DenseMatrix<Index, Device::CPU>& expected)
{
    const auto actual_cpu = actual.toCpu();
    ASSERT_EQ(actual_cpu.rows(), expected.rows());
    ASSERT_EQ(actual_cpu.cols(), expected.cols());
    for (Index index = 0; index < expected.size(); ++index)
    {
        EXPECT_EQ(actual_cpu.data()[index], expected.data()[index]);
    }
}

template <typename Callable>
void expectLogicErrorContaining(
    Callable&& callable,
    std::initializer_list<const char*> expected_fragments)
{
    try
    {
        std::forward<Callable>(callable)();
        FAIL() << "operation should throw std::logic_error";
    }
    catch (const std::logic_error& error)
    {
        const std::string message = error.what();
        for (const char* fragment : expected_fragments)
        {
            EXPECT_NE(message.find(fragment), std::string::npos) << message;
        }
    }
    catch (...)
    {
        FAIL() << "operation threw an unexpected exception type";
    }
}
#else
template <typename Callable>
void expectNoCudaReductionError(const char* operation, Callable&& callable)
{
    try
    {
        std::forward<Callable>(callable)();
        FAIL() << operation << " should reject CPU-only builds";
    }
    catch (const std::runtime_error& error)
    {
        const std::string message = error.what();
        EXPECT_NE(message.find(operation), std::string::npos) << message;
        EXPECT_NE(message.find("PLAMATRIX_WITH_CUDA=ON"), std::string::npos) << message;
    }
}
#endif

TYPED_TEST(ReductionTest, AllAxisUsesColumnMajorLinearIndices)
{
    using Scalar = TypeParam;
    auto input = TestFixture::makeMatrix(2, 3, {
        Scalar(1), Scalar(-2), Scalar(3), Scalar(-4), Scalar(5), Scalar(-6)
    });

    auto summed = sum(input, ReductionAxis::All);
    auto averaged = mean(input, ReductionAxis::All);
    auto minimum = min(input, ReductionAxis::All);
    auto maximum = max(input, ReductionAxis::All);
    auto min_indexed = argMin(input, ReductionAxis::All);
    auto max_indexed = argMax(input, ReductionAxis::All);

    TestFixture::expectShape(summed, 1, 1);
    TestFixture::expectShape(min_indexed.values, 1, 1);
    TestFixture::expectShape(min_indexed.indices, 1, 1);
    TestFixture::expectValues(summed, {Scalar(-3)});
    TestFixture::expectValues(averaged, {Scalar(-0.5)});
    TestFixture::expectValues(minimum, {Scalar(-6)});
    TestFixture::expectValues(maximum, {Scalar(5)});
    TestFixture::expectValues(min_indexed.values, {Scalar(-6)});
    TestFixture::expectValues(min_indexed.indices, {Index(5)});
    TestFixture::expectValues(max_indexed.values, {Scalar(5)});
    TestFixture::expectValues(max_indexed.indices, {Index(4)});
}

TYPED_TEST(ReductionTest, RowsAxisReducesColumns)
{
    using Scalar = TypeParam;
    auto input = TestFixture::makeMatrix(2, 3, {
        Scalar(1), Scalar(-2), Scalar(3), Scalar(-4), Scalar(5), Scalar(-6)
    });

    auto summed = sum(input, ReductionAxis::Rows);
    auto averaged = mean(input, ReductionAxis::Rows);
    auto minimum = min(input, ReductionAxis::Rows);
    auto maximum = max(input, ReductionAxis::Rows);
    auto min_indexed = argMin(input, ReductionAxis::Rows);
    auto max_indexed = argMax(input, ReductionAxis::Rows);

    TestFixture::expectShape(summed, 2, 1);
    TestFixture::expectShape(min_indexed.indices, 2, 1);
    TestFixture::expectValues(summed, {Scalar(9), Scalar(-12)});
    TestFixture::expectValues(averaged, {Scalar(3), Scalar(-4)});
    TestFixture::expectValues(minimum, {Scalar(1), Scalar(-6)});
    TestFixture::expectValues(maximum, {Scalar(5), Scalar(-2)});
    TestFixture::expectValues(min_indexed.values, {Scalar(1), Scalar(-6)});
    TestFixture::expectValues(min_indexed.indices, {Index(0), Index(2)});
    TestFixture::expectValues(max_indexed.values, {Scalar(5), Scalar(-2)});
    TestFixture::expectValues(max_indexed.indices, {Index(2), Index(0)});
}

TYPED_TEST(ReductionTest, ColumnsAxisReducesRows)
{
    using Scalar = TypeParam;
    auto input = TestFixture::makeMatrix(2, 3, {
        Scalar(1), Scalar(-2), Scalar(3), Scalar(-4), Scalar(5), Scalar(-6)
    });

    auto summed = sum(input, ReductionAxis::Columns);
    auto averaged = mean(input, ReductionAxis::Columns);
    auto minimum = min(input, ReductionAxis::Columns);
    auto maximum = max(input, ReductionAxis::Columns);
    auto min_indexed = argMin(input, ReductionAxis::Columns);
    auto max_indexed = argMax(input, ReductionAxis::Columns);

    TestFixture::expectShape(summed, 1, 3);
    TestFixture::expectShape(min_indexed.indices, 1, 3);
    TestFixture::expectValues(summed, {Scalar(-1), Scalar(-1), Scalar(-1)});
    TestFixture::expectValues(averaged, {Scalar(-0.5), Scalar(-0.5), Scalar(-0.5)});
    TestFixture::expectValues(minimum, {Scalar(-2), Scalar(-4), Scalar(-6)});
    TestFixture::expectValues(maximum, {Scalar(1), Scalar(3), Scalar(5)});
    TestFixture::expectValues(min_indexed.values, {Scalar(-2), Scalar(-4), Scalar(-6)});
    TestFixture::expectValues(min_indexed.indices, {Index(1), Index(1), Index(1)});
    TestFixture::expectValues(max_indexed.values, {Scalar(1), Scalar(3), Scalar(5)});
    TestFixture::expectValues(max_indexed.indices, {Index(0), Index(0), Index(0)});
}

TYPED_TEST(ReductionTest, EmptyReducedLanesFollowOperationRules)
{
    using Scalar = TypeParam;
    DenseMatrix<Scalar, Device::CPU> zero_by_three(0, 3);
    DenseMatrix<Scalar, Device::CPU> three_by_zero(3, 0);

    auto all_sum = sum(zero_by_three, ReductionAxis::All);
    auto column_sums = sum(zero_by_three, ReductionAxis::Columns);
    auto row_sums = sum(three_by_zero, ReductionAxis::Rows);
    TestFixture::expectShape(all_sum, 1, 1);
    TestFixture::expectValues(all_sum, {Scalar(0)});
    TestFixture::expectShape(column_sums, 1, 3);
    TestFixture::expectValues(column_sums, {Scalar(0), Scalar(0), Scalar(0)});
    TestFixture::expectShape(row_sums, 3, 1);
    TestFixture::expectValues(row_sums, {Scalar(0), Scalar(0), Scalar(0)});

    EXPECT_THROW(static_cast<void>(mean(zero_by_three, ReductionAxis::All)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(min(zero_by_three, ReductionAxis::Columns)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(max(three_by_zero, ReductionAxis::Rows)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(argMin(zero_by_three, ReductionAxis::Columns)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(argMax(three_by_zero, ReductionAxis::Rows)), std::invalid_argument);
}

TYPED_TEST(ReductionTest, ZeroOutputLanesReturnCorrectShapes)
{
    using Scalar = TypeParam;
    DenseMatrix<Scalar, Device::CPU> zero_by_three(0, 3);
    DenseMatrix<Scalar, Device::CPU> three_by_zero(3, 0);
    DenseMatrix<Scalar, Device::CPU> zero_by_zero(0, 0);

    for (const auto* input : {&zero_by_three, &zero_by_zero})
    {
        TestFixture::expectShape(sum(*input, ReductionAxis::Rows), 0, 1);
        TestFixture::expectShape(mean(*input, ReductionAxis::Rows), 0, 1);
        TestFixture::expectShape(min(*input, ReductionAxis::Rows), 0, 1);
        TestFixture::expectShape(max(*input, ReductionAxis::Rows), 0, 1);
        TestFixture::expectShape(argMin(*input, ReductionAxis::Rows).indices, 0, 1);
        TestFixture::expectShape(argMax(*input, ReductionAxis::Rows).values, 0, 1);
    }
    for (const auto* input : {&three_by_zero, &zero_by_zero})
    {
        TestFixture::expectShape(sum(*input, ReductionAxis::Columns), 1, 0);
        TestFixture::expectShape(mean(*input, ReductionAxis::Columns), 1, 0);
        TestFixture::expectShape(min(*input, ReductionAxis::Columns), 1, 0);
        TestFixture::expectShape(max(*input, ReductionAxis::Columns), 1, 0);
        TestFixture::expectShape(argMin(*input, ReductionAxis::Columns).values, 1, 0);
        TestFixture::expectShape(argMax(*input, ReductionAxis::Columns).indices, 1, 0);
    }
}

TYPED_TEST(ReductionTest, ZeroByZeroAllRejectsNonSumReductions)
{
    using Scalar = TypeParam;
    DenseMatrix<Scalar, Device::CPU> input(0, 0);

    TestFixture::expectValues(sum(input, ReductionAxis::All), {Scalar(0)});
    EXPECT_THROW(static_cast<void>(mean(input, ReductionAxis::All)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(min(input, ReductionAxis::All)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(max(input, ReductionAxis::All)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(argMin(input, ReductionAxis::All)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(argMax(input, ReductionAxis::All)), std::invalid_argument);
}

TYPED_TEST(ReductionTest, NaNsPropagateAndLowestNaNIndexWins)
{
    using Scalar = TypeParam;
    const Scalar nan = std::numeric_limits<Scalar>::quiet_NaN();
    auto input = TestFixture::makeMatrix(2, 3, {
        Scalar(4), nan, nan, Scalar(2), nan, nan
    });

    for (ReductionAxis axis : {ReductionAxis::All, ReductionAxis::Rows, ReductionAxis::Columns})
    {
        auto summed = sum(input, axis);
        auto averaged = mean(input, axis);
        auto minimum = min(input, axis);
        auto maximum = max(input, axis);
        for (Index index = 0; index < summed.size(); ++index)
        {
            EXPECT_TRUE(std::isnan(summed.data()[index]));
            EXPECT_TRUE(std::isnan(averaged.data()[index]));
            EXPECT_TRUE(std::isnan(minimum.data()[index]));
            EXPECT_TRUE(std::isnan(maximum.data()[index]));
        }
    }

    auto all_min = argMin(input, ReductionAxis::All);
    auto all_max = argMax(input, ReductionAxis::All);
    EXPECT_TRUE(std::isnan(all_min.values.data()[0]));
    EXPECT_TRUE(std::isnan(all_max.values.data()[0]));
    EXPECT_EQ(all_min.indices.data()[0], 1);
    EXPECT_EQ(all_max.indices.data()[0], 1);
    TestFixture::expectValues(argMin(input, ReductionAxis::Rows).indices, {Index(1), Index(0)});
    TestFixture::expectValues(argMax(input, ReductionAxis::Rows).indices, {Index(1), Index(0)});
    TestFixture::expectValues(argMin(input, ReductionAxis::Columns).indices, {Index(1), Index(0), Index(0)});
    TestFixture::expectValues(argMax(input, ReductionAxis::Columns).indices, {Index(1), Index(0), Index(0)});
}

TYPED_TEST(ReductionTest, FiniteTiesChooseLowestReducedIndex)
{
    using Scalar = TypeParam;
    auto input = TestFixture::makeMatrix(3, 3, {
        Scalar(-5), Scalar(7), Scalar(2),
        Scalar(-5), Scalar(7), Scalar(9),
        Scalar(4), Scalar(7), Scalar(9)
    });

    TestFixture::expectValues(argMin(input, ReductionAxis::Rows).indices,
                              {Index(0), Index(0), Index(0)});
    TestFixture::expectValues(argMax(input, ReductionAxis::Rows).indices,
                              {Index(2), Index(0), Index(1)});
    TestFixture::expectValues(argMin(input, ReductionAxis::Columns).indices,
                              {Index(0), Index(0), Index(0)});
    TestFixture::expectValues(argMax(input, ReductionAxis::Columns).indices,
                              {Index(1), Index(2), Index(2)});
    EXPECT_EQ(argMin(input, ReductionAxis::All).indices.data()[0], 0);
    EXPECT_EQ(argMax(input, ReductionAxis::All).indices.data()[0], 5);
}

TYPED_TEST(ReductionTest, InfinitiesAndNegativeValuesUseIeeeComparisons)
{
    using Scalar = TypeParam;
    const Scalar infinity = std::numeric_limits<Scalar>::infinity();
    auto input = TestFixture::makeMatrix(2, 2, {
        -infinity, Scalar(-4), infinity, Scalar(-9)
    });

    TestFixture::expectValues(min(input, ReductionAxis::Rows), {-infinity, Scalar(-9)});
    TestFixture::expectValues(max(input, ReductionAxis::Rows), {infinity, Scalar(-4)});
    TestFixture::expectValues(argMin(input, ReductionAxis::Rows).indices, {Index(0), Index(1)});
    TestFixture::expectValues(argMax(input, ReductionAxis::Rows).indices, {Index(1), Index(0)});
    EXPECT_EQ(min(input, ReductionAxis::All).data()[0], -infinity);
    EXPECT_EQ(max(input, ReductionAxis::All).data()[0], infinity);
}

TYPED_TEST(ReductionTest, InvalidAxisIsRejectedByEveryOperation)
{
    using Scalar = TypeParam;
    DenseMatrix<Scalar, Device::CPU> input(0, 0);
    const auto invalid = static_cast<ReductionAxis>(99);

    EXPECT_THROW(static_cast<void>(sum(input, invalid)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(mean(input, invalid)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(min(input, invalid)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(max(input, invalid)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(argMin(input, invalid)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(argMax(input, invalid)), std::invalid_argument);
}

TYPED_TEST(ReductionTest, OpenMpSizedRowsKeepSerialLaneResults)
{
    using Scalar = TypeParam;
    constexpr Index lane_count = detail::kOpenMpWorkThreshold;
    DenseMatrix<Scalar, Device::CPU> input(lane_count, 3);
    for (Index row = 0; row < lane_count; ++row)
    {
        input(row, 0) = Scalar(row);
        input(row, 1) = Scalar(-row - 1);
        input(row, 2) = Scalar(row + 2);
    }

    auto summed = sum(input, ReductionAxis::Rows);
    auto averaged = mean(input, ReductionAxis::Rows);
    auto minimum = min(input, ReductionAxis::Rows);
    auto maximum = max(input, ReductionAxis::Rows);
    auto min_indexed = argMin(input, ReductionAxis::Rows);
    auto max_indexed = argMax(input, ReductionAxis::Rows);
    for (Index row = 0; row < lane_count; ++row)
    {
        EXPECT_EQ(summed(row, 0), Scalar(row + 1));
        EXPECT_EQ(averaged(row, 0), static_cast<Scalar>(static_cast<double>(row + 1) / 3.0));
        EXPECT_EQ(minimum(row, 0), Scalar(-row - 1));
        EXPECT_EQ(maximum(row, 0), Scalar(row + 2));
        EXPECT_EQ(min_indexed.values(row, 0), Scalar(-row - 1));
        EXPECT_EQ(min_indexed.indices(row, 0), 1);
        EXPECT_EQ(max_indexed.values(row, 0), Scalar(row + 2));
        EXPECT_EQ(max_indexed.indices(row, 0), 2);
    }
}

TYPED_TEST(ReductionTest, MeanOfLargestFinitePairRemainsFinite)
{
    using Scalar = TypeParam;
    const Scalar largest = std::numeric_limits<Scalar>::max();
    auto input = TestFixture::makeMatrix(1, 2, {largest, largest});
    auto column_input = input.transpose();

    EXPECT_EQ(mean(input, ReductionAxis::All).data()[0], largest);
    EXPECT_EQ(mean(input, ReductionAxis::Rows).data()[0], largest);
    EXPECT_EQ(mean(column_input, ReductionAxis::Columns).data()[0], largest);
}

TYPED_TEST(ReductionTest, MeanPreservesLargePositiveNegativeCancellation)
{
    using Scalar = TypeParam;
    const Scalar largest = std::numeric_limits<Scalar>::max();
    const Scalar expected = Scalar(4) / Scalar(3);
    const Scalar tolerance = std::numeric_limits<Scalar>::epsilon() * Scalar(8);
    auto input = TestFixture::makeMatrix(1, 3, {largest, Scalar(4), -largest});
    auto column_input = input.transpose();

    EXPECT_NEAR(mean(input, ReductionAxis::All).data()[0], expected, tolerance);
    EXPECT_NEAR(mean(input, ReductionAxis::Rows).data()[0], expected, tolerance);
    EXPECT_NEAR(mean(column_input, ReductionAxis::Columns).data()[0], expected, tolerance);
}

TYPED_TEST(ReductionTest, MeanDetectsPartialAdditionLoss)
{
    using Scalar = TypeParam;
    const Scalar large = std::ldexp(Scalar(1), 53);
    auto input = TestFixture::makeMatrix(1, 3, {large, Scalar(3), -large});
    auto column_input = input.transpose();

    EXPECT_EQ(mean(input, ReductionAxis::All).data()[0], Scalar(1));
    EXPECT_EQ(mean(input, ReductionAxis::Rows).data()[0], Scalar(1));
    EXPECT_EQ(mean(column_input, ReductionAxis::Columns).data()[0], Scalar(1));
}

TYPED_TEST(ReductionTest, MeanHandlesInfinitiesAndNaNsPerLane)
{
    using Scalar = TypeParam;
    const Scalar infinity = std::numeric_limits<Scalar>::infinity();
    const Scalar nan = std::numeric_limits<Scalar>::quiet_NaN();
    auto input = TestFixture::makeMatrix(5, 2, {
        infinity, -infinity, infinity, nan, Scalar(1),
        Scalar(4), Scalar(-3), -infinity, Scalar(2), Scalar(2)
    });

    auto result = mean(input, ReductionAxis::Rows);

    EXPECT_EQ(result(0, 0), infinity);
    EXPECT_EQ(result(1, 0), -infinity);
    EXPECT_TRUE(std::isnan(result(2, 0)));
    EXPECT_TRUE(std::isnan(result(3, 0)));
    EXPECT_EQ(result(4, 0), Scalar(1.5));
    EXPECT_TRUE(std::isnan(mean(input, ReductionAxis::All).data()[0]));
}

TYPED_TEST(ReductionTest, MeanRetainsRegularPrecisionAcrossAxes)
{
    using Scalar = TypeParam;
    const Scalar tolerance = std::numeric_limits<Scalar>::epsilon() * Scalar(8);
    auto input = TestFixture::makeMatrix(2, 3, {
        Scalar(1), Scalar(2), Scalar(2), Scalar(4), Scalar(2), Scalar(8)
    });

    auto all = mean(input, ReductionAxis::All);
    auto rows = mean(input, ReductionAxis::Rows);
    auto columns = mean(input, ReductionAxis::Columns);

    EXPECT_NEAR(all.data()[0], Scalar(19) / Scalar(6), tolerance);
    EXPECT_NEAR(rows(0, 0), Scalar(5) / Scalar(3), tolerance);
    EXPECT_NEAR(rows(1, 0), Scalar(14) / Scalar(3), tolerance);
    EXPECT_NEAR(columns(0, 0), Scalar(1.5), tolerance);
    EXPECT_NEAR(columns(0, 1), Scalar(3), tolerance);
    EXPECT_NEAR(columns(0, 2), Scalar(5), tolerance);
}

#ifdef PLAMATRIX_WITH_CUDA
TYPED_TEST(ReductionTest, GpuValueReductionFamiliesMatchCpuForEveryAxis)
{
    using Scalar = TypeParam;
    auto input_cpu = TestFixture::makeMatrix(3, 4, {
        Scalar(3), Scalar(-2), Scalar(7), Scalar(4), Scalar(5), Scalar(-1),
        Scalar(8), Scalar(6), Scalar(-3), Scalar(2), Scalar(9), Scalar(-4)
    });
    auto input = input_cpu.toGpu();
    test::CudaStreamGuard stream;
    const Scalar tolerance = std::numeric_limits<Scalar>::epsilon() * Scalar(64);

    for (ReductionAxis axis : {ReductionAxis::All, ReductionAxis::Rows, ReductionAxis::Columns})
    {
#define EXERCISE_VALUE_REDUCTION(OP, ASYNC_OP, EXPECTED, TOLERANCE)                     \
        {                                                                                \
            ReductionWorkspace workspace;                                                \
            auto convenience = OP(input, axis);                                           \
            auto sync_allocated = OP(input, axis, workspace, stream.get());               \
            DenseMatrix<Scalar, Device::GPU> sync_reused(                                 \
                EXPECTED.rows(), EXPECTED.cols());                                        \
            OP(input, axis, sync_reused, workspace, stream.get());                        \
            auto async_allocated = ASYNC_OP(input, axis, workspace, stream.get());        \
            DenseMatrix<Scalar, Device::GPU> async_reused(                                \
                EXPECTED.rows(), EXPECTED.cols());                                        \
            ASYNC_OP(input, axis, async_reused, workspace, stream.get());                 \
            stream.synchronize();                                                         \
            expectGpuReductionResult(convenience, EXPECTED, TOLERANCE);                   \
            expectGpuReductionResult(sync_allocated, EXPECTED, TOLERANCE);                \
            expectGpuReductionResult(sync_reused, EXPECTED, TOLERANCE);                   \
            expectGpuReductionResult(async_allocated, EXPECTED, TOLERANCE);               \
            expectGpuReductionResult(async_reused, EXPECTED, TOLERANCE);                  \
            async_allocated.closeAsyncAllocation();                                       \
            stream.synchronize();                                                         \
        }

        const auto expected_sum = sum(input_cpu, axis);
        const auto expected_mean = mean(input_cpu, axis);
        const auto expected_min = min(input_cpu, axis);
        const auto expected_max = max(input_cpu, axis);
        EXERCISE_VALUE_REDUCTION(sum, sumAsync, expected_sum, tolerance);
        EXERCISE_VALUE_REDUCTION(mean, meanAsync, expected_mean, tolerance);
        EXERCISE_VALUE_REDUCTION(min, minAsync, expected_min, Scalar(0));
        EXERCISE_VALUE_REDUCTION(max, maxAsync, expected_max, Scalar(0));
#undef EXERCISE_VALUE_REDUCTION
    }
}

TYPED_TEST(ReductionTest, GpuIndexedReductionFamiliesMatchCpuForEveryAxis)
{
    using Scalar = TypeParam;
    auto input_cpu = TestFixture::makeMatrix(3, 4, {
        Scalar(3), Scalar(-2), Scalar(7), Scalar(4), Scalar(5), Scalar(-1),
        Scalar(8), Scalar(6), Scalar(-3), Scalar(2), Scalar(9), Scalar(-4)
    });
    auto input = input_cpu.toGpu();
    test::CudaStreamGuard stream;

    for (ReductionAxis axis : {ReductionAxis::All, ReductionAxis::Rows, ReductionAxis::Columns})
    {
#define EXERCISE_INDEXED_REDUCTION(OP, ASYNC_OP, EXPECTED)                               \
        {                                                                                \
            ReductionWorkspace workspace;                                                \
            auto convenience = OP(input, axis);                                           \
            auto sync_allocated = OP(input, axis, workspace, stream.get());               \
            DenseMatrix<Scalar, Device::GPU> sync_values(                                 \
                EXPECTED.values.rows(), EXPECTED.values.cols());                          \
            DenseMatrix<Index, Device::GPU> sync_indices(                                 \
                EXPECTED.indices.rows(), EXPECTED.indices.cols());                        \
            OP(input, axis, sync_values, sync_indices, workspace, stream.get());          \
            auto async_allocated = ASYNC_OP(input, axis, workspace, stream.get());        \
            DenseMatrix<Scalar, Device::GPU> async_values(                                \
                EXPECTED.values.rows(), EXPECTED.values.cols());                          \
            DenseMatrix<Index, Device::GPU> async_indices(                                \
                EXPECTED.indices.rows(), EXPECTED.indices.cols());                        \
            ASYNC_OP(input, axis, async_values, async_indices, workspace, stream.get());  \
            stream.synchronize();                                                         \
            expectGpuReductionResult(convenience.values, EXPECTED.values);                \
            expectGpuIndices(convenience.indices, EXPECTED.indices);                      \
            expectGpuReductionResult(sync_allocated.values, EXPECTED.values);             \
            expectGpuIndices(sync_allocated.indices, EXPECTED.indices);                   \
            expectGpuReductionResult(sync_values, EXPECTED.values);                       \
            expectGpuIndices(sync_indices, EXPECTED.indices);                            \
            expectGpuReductionResult(async_allocated.values, EXPECTED.values);            \
            expectGpuIndices(async_allocated.indices, EXPECTED.indices);                  \
            expectGpuReductionResult(async_values, EXPECTED.values);                      \
            expectGpuIndices(async_indices, EXPECTED.indices);                           \
            async_allocated.values.closeAsyncAllocation();                               \
            async_allocated.indices.closeAsyncAllocation();                              \
            stream.synchronize();                                                         \
        }

        const auto expected_min = argMin(input_cpu, axis);
        const auto expected_max = argMax(input_cpu, axis);
        EXERCISE_INDEXED_REDUCTION(argMin, argMinAsync, expected_min);
        EXERCISE_INDEXED_REDUCTION(argMax, argMaxAsync, expected_max);
#undef EXERCISE_INDEXED_REDUCTION
    }
}

TYPED_TEST(ReductionTest, GpuWorkspaceGrowsMovesAndClosesAccordingToProvenance)
{
    ReductionWorkspace workspace;
    EXPECT_EQ(workspace.capacityBytes(), 0U);
    EXPECT_EQ(workspace.data(), nullptr);

    workspace.reserveBytes(256);
    ASSERT_GE(workspace.capacityBytes(), 256U);
    ASSERT_NE(workspace.data(), nullptr);
    const std::size_t original_capacity = workspace.capacityBytes();
    void* const original_data = workspace.data();
    workspace.reserveBytes(256);
    workspace.reserveBytes(128);
    EXPECT_EQ(workspace.capacityBytes(), original_capacity);
    EXPECT_EQ(workspace.data(), original_data);

    workspace.reserveBytes(original_capacity + 1);
    EXPECT_GT(workspace.capacityBytes(), original_capacity);
    EXPECT_THROW(workspace.closeAsyncAllocation(), std::logic_error);

    ReductionWorkspace moved(std::move(workspace));
    EXPECT_EQ(workspace.capacityBytes(), 0U);
    EXPECT_EQ(workspace.data(), nullptr);
    EXPECT_NE(moved.data(), nullptr);

    test::CudaStreamGuard stream;
    ReductionWorkspace async_workspace;
    async_workspace.reserveBytesAsync(512, stream.get());
    ASSERT_GE(async_workspace.capacityBytes(), 512U);
    ASSERT_NE(async_workspace.data(), nullptr);
    ReductionWorkspace async_moved;
    async_moved = std::move(async_workspace);
    EXPECT_EQ(async_workspace.capacityBytes(), 0U);
    EXPECT_EQ(async_workspace.data(), nullptr);
    EXPECT_NO_THROW(async_moved.closeAsyncAllocation());
    EXPECT_EQ(async_moved.capacityBytes(), 0U);
    EXPECT_EQ(async_moved.data(), nullptr);
    EXPECT_NO_THROW(async_moved.closeAsyncAllocation());
    stream.synchronize();
}

TYPED_TEST(ReductionTest, GpuAsyncWorkspaceReserveIsGrowOnlyAndPreservesResults)
{
    using Scalar = TypeParam;
    auto input_cpu = TestFixture::makeMatrix(3, 4, {
        Scalar(3), Scalar(-2), Scalar(7), Scalar(4), Scalar(5), Scalar(-1),
        Scalar(8), Scalar(6), Scalar(-3), Scalar(2), Scalar(9), Scalar(-4)
    });
    auto input = input_cpu.toGpu();
    test::CudaStreamGuard stream;
    ReductionWorkspace workspace;

    auto before_growth = sumAsync(
        input, ReductionAxis::All, workspace, stream.get());
    const std::size_t original_capacity = workspace.capacityBytes();
    void* const original_data = workspace.data();
    ASSERT_GT(original_capacity, 0U);
    ASSERT_NE(original_data, nullptr);

    workspace.reserveBytesAsync(original_capacity, stream.get());
    workspace.reserveBytesAsync(original_capacity / 2, stream.get());
    EXPECT_EQ(workspace.capacityBytes(), original_capacity);
    EXPECT_EQ(workspace.data(), original_data);

    const std::size_t requested_capacity = original_capacity + 4096U;
    workspace.reserveBytesAsync(requested_capacity, stream.get());
    EXPECT_GE(workspace.capacityBytes(), requested_capacity);
    EXPECT_GT(workspace.capacityBytes(), original_capacity);
    const std::size_t grown_capacity = workspace.capacityBytes();

    auto after_growth = sumAsync(
        input, ReductionAxis::All, workspace, stream.get());
    EXPECT_EQ(workspace.capacityBytes(), grown_capacity);
    stream.synchronize();

    const auto expected = sum(input_cpu, ReductionAxis::All);
    expectGpuReductionResult(before_growth, expected);
    expectGpuReductionResult(after_growth, expected);

    before_growth.closeAsyncAllocation();
    after_growth.closeAsyncAllocation();
    workspace.closeAsyncAllocation();
    stream.synchronize();
}

TYPED_TEST(ReductionTest, GpuAsyncWorkspaceSupportsSameStreamSequentialReuse)
{
    using Scalar = TypeParam;
    auto input_cpu = TestFixture::makeMatrix(2, 4, {
        Scalar(1), Scalar(5), Scalar(2), Scalar(6),
        Scalar(3), Scalar(7), Scalar(4), Scalar(8)
    });
    auto input = input_cpu.toGpu();
    test::CudaStreamGuard stream;
    ReductionWorkspace workspace;

    auto first = meanAsync(input, ReductionAxis::All, workspace, stream.get());
    const std::size_t first_capacity = workspace.capacityBytes();
    void* const first_data = workspace.data();
    auto second = meanAsync(input, ReductionAxis::All, workspace, stream.get());

    EXPECT_EQ(workspace.capacityBytes(), first_capacity);
    EXPECT_EQ(workspace.data(), first_data);
    stream.synchronize();

    const auto expected = mean(input_cpu, ReductionAxis::All);
    expectGpuReductionResult(first, expected);
    expectGpuReductionResult(second, expected);

    first.closeAsyncAllocation();
    second.closeAsyncAllocation();
    workspace.closeAsyncAllocation();
    stream.synchronize();
}

TYPED_TEST(ReductionTest, GpuAsyncWorkspaceRejectsCrossStreamReuseUntilClosed)
{
    using Scalar = TypeParam;
    auto input_cpu = TestFixture::makeMatrix(2, 3, {
        Scalar(1), Scalar(4), Scalar(2), Scalar(5), Scalar(3), Scalar(6)
    });
    auto input = input_cpu.toGpu();
    test::CudaStreamGuard first_stream;
    test::CudaStreamGuard second_stream;
    ReductionWorkspace workspace;
    DenseMatrix<Scalar, Device::GPU> rejected_output(1, 1);

    auto first = sumAsync(
        input, ReductionAxis::All, workspace, first_stream.get());
    expectLogicErrorContaining([&] {
        sumAsync(input, ReductionAxis::All, rejected_output, workspace, second_stream.get());
    }, {"different stream", "synchronize", "close"});

    first_stream.synchronize();
    expectGpuReductionResult(first, sum(input_cpu, ReductionAxis::All));
    first.closeAsyncAllocation();
    workspace.closeAsyncAllocation();
    first_stream.synchronize();

    auto second = sumAsync(
        input, ReductionAxis::All, workspace, second_stream.get());
    second_stream.synchronize();
    expectGpuReductionResult(second, sum(input_cpu, ReductionAxis::All));
    second.closeAsyncAllocation();
    workspace.closeAsyncAllocation();
    second_stream.synchronize();
}

TYPED_TEST(ReductionTest, GpuNormalWorkspaceSupportsReuseButRejectsAsyncGrowth)
{
    using Scalar = TypeParam;
    auto input_cpu = TestFixture::makeMatrix(2, 3, {
        Scalar(1), Scalar(4), Scalar(2), Scalar(5), Scalar(3), Scalar(6)
    });
    auto input = input_cpu.toGpu();
    test::CudaStreamGuard stream;
    ReductionWorkspace sufficient_workspace;
    constexpr std::size_t reserved_capacity = 1024U * 1024U;
    sufficient_workspace.reserveBytes(reserved_capacity);
    void* const reserved_data = sufficient_workspace.data();

    auto result = sumAsync(
        input, ReductionAxis::All, sufficient_workspace, stream.get());
    EXPECT_EQ(sufficient_workspace.capacityBytes(), reserved_capacity);
    EXPECT_EQ(sufficient_workspace.data(), reserved_data);
    stream.synchronize();
    expectGpuReductionResult(result, sum(input_cpu, ReductionAxis::All));
    result.closeAsyncAllocation();
    stream.synchronize();

    ReductionWorkspace undersized_workspace;
    undersized_workspace.reserveBytes(1);
    const std::size_t original_capacity = undersized_workspace.capacityBytes();
    void* const original_data = undersized_workspace.data();
    DenseMatrix<Scalar, Device::GPU> rejected_output(1, 1);
    expectLogicErrorContaining([&] {
        sumAsync(input, ReductionAxis::All, rejected_output,
                 undersized_workspace, stream.get());
    }, {"cannot grow", "reserveBytes", "before launching"});
    EXPECT_EQ(undersized_workspace.capacityBytes(), original_capacity);
    EXPECT_EQ(undersized_workspace.data(), original_data);
}

TEST(ReductionWorkspaceCudaTest, AsyncAllocationRequiresCheckedCloseBeforeSyncReserve)
{
    test::CudaStreamGuard stream;
    ReductionWorkspace workspace;
    workspace.reserveBytesAsync(512, stream.get());
    const std::size_t async_capacity = workspace.capacityBytes();
    void* const async_data = workspace.data();

    expectLogicErrorContaining([&] {
        workspace.reserveBytes(1024);
    }, {"cannot replace", "closeAsyncAllocation"});
    EXPECT_EQ(workspace.capacityBytes(), async_capacity);
    EXPECT_EQ(workspace.data(), async_data);

    workspace.closeAsyncAllocation();
    EXPECT_EQ(workspace.capacityBytes(), 0U);
    EXPECT_EQ(workspace.data(), nullptr);
    EXPECT_NO_THROW(workspace.closeAsyncAllocation());
    stream.synchronize();

    workspace.reserveBytes(1024);
    EXPECT_GE(workspace.capacityBytes(), 1024U);
    EXPECT_NE(workspace.data(), nullptr);
}

TEST(ReductionWorkspaceCudaTest, MovePreservesStreamProvenanceAndFailedGrowState)
{
    test::CudaStreamGuard owning_stream;
    test::CudaStreamGuard other_stream;
    ReductionWorkspace source;
    source.reserveBytesAsync(512, owning_stream.get());
    ReductionWorkspace moved(std::move(source));
    const std::size_t original_capacity = moved.capacityBytes();
    void* const original_data = moved.data();

    EXPECT_EQ(source.capacityBytes(), 0U);
    EXPECT_EQ(source.data(), nullptr);
    expectLogicErrorContaining([&] {
        moved.reserveBytesAsync(original_capacity, other_stream.get());
    }, {"different stream", "synchronize", "close"});

    EXPECT_THROW(
        moved.reserveBytesAsync(std::numeric_limits<std::size_t>::max(), owning_stream.get()),
        std::runtime_error);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
    EXPECT_EQ(moved.capacityBytes(), original_capacity);
    EXPECT_EQ(moved.data(), original_data);
    EXPECT_NO_THROW(moved.reserveBytesAsync(original_capacity / 2, owning_stream.get()));

    owning_stream.synchronize();
    moved.closeAsyncAllocation();
    EXPECT_NO_THROW(moved.closeAsyncAllocation());
    owning_stream.synchronize();
}

#ifdef PLAMATRIX_USE_DOUBLE
TEST(ReductionDoubleTest, GpuMeanKeepsMaxFiniteLanesFiniteAcrossAxes)
{
    const double largest = std::numeric_limits<double>::max();
    for (Index reduction_length : {Index(3), Index(257)})
    {
        DenseMatrix<double, Device::CPU> all_input(1, reduction_length);
        DenseMatrix<double, Device::CPU> rows_input(2, reduction_length);
        DenseMatrix<double, Device::CPU> columns_input(reduction_length, 2);
        all_input.fill(largest);
        rows_input.fill(largest);
        columns_input.fill(largest);

        expectGpuReductionResult(
            mean(all_input.toGpu(), ReductionAxis::All),
            mean(all_input, ReductionAxis::All));
        expectGpuReductionResult(
            mean(rows_input.toGpu(), ReductionAxis::Rows),
            mean(rows_input, ReductionAxis::Rows));
        expectGpuReductionResult(
            mean(columns_input.toGpu(), ReductionAxis::Columns),
            mean(columns_input, ReductionAxis::Columns));
    }
}
#endif

TYPED_TEST(ReductionTest, GpuScaledMeanMatchesCpuForSpecialValues)
{
    using Scalar = TypeParam;
    const Scalar largest = std::numeric_limits<Scalar>::max();
    const Scalar infinity = std::numeric_limits<Scalar>::infinity();
    const Scalar nan = std::numeric_limits<Scalar>::quiet_NaN();
    const Scalar tolerance = std::numeric_limits<Scalar>::epsilon() * Scalar(64);
    for (Index reduction_length : {Index(3), Index(257)})
    {
        DenseMatrix<Scalar, Device::CPU> rows_input(5, reduction_length);
        rows_input.fill(Scalar(0));
        for (Index col = 0; col < reduction_length; ++col)
        {
            rows_input(0, col) = -largest;
            rows_input(2, col) = Scalar(1);
            rows_input(3, col) = Scalar(2);
            rows_input(4, col) = Scalar(0);
        }
        rows_input(1, 0) = largest;
        rows_input(1, 1) = -largest;
        rows_input(1, 2) = Scalar(4);
        rows_input(2, 0) = nan;
        rows_input(3, 0) = infinity;
        rows_input(4, 0) = infinity;
        rows_input(4, 1) = -infinity;

        const auto rows_expected = mean(rows_input, ReductionAxis::Rows);
        expectGpuReductionResult(
            mean(rows_input.toGpu(), ReductionAxis::Rows), rows_expected, tolerance);

        const auto columns_input = rows_input.transpose();
        const auto columns_expected = mean(columns_input, ReductionAxis::Columns);
        expectGpuReductionResult(
            mean(columns_input.toGpu(), ReductionAxis::Columns),
            columns_expected,
            tolerance);

        DenseMatrix<Scalar, Device::CPU> all_input(1, reduction_length);
        const auto check_all = [&] {
            expectGpuReductionResult(
                mean(all_input.toGpu(), ReductionAxis::All),
                mean(all_input, ReductionAxis::All),
                tolerance);
        };

        all_input.fill(-largest);
        check_all();
        all_input.fill(Scalar(0));
        all_input(0, 0) = largest;
        all_input(0, 1) = -largest;
        all_input(0, 2) = Scalar(4);
        check_all();
        all_input.fill(Scalar(1));
        all_input(0, 0) = nan;
        check_all();
        all_input.fill(Scalar(2));
        all_input(0, 0) = infinity;
        check_all();
        all_input.fill(Scalar(0));
        all_input(0, 0) = infinity;
        all_input(0, 1) = -infinity;
        check_all();
    }
}

TEST(ReductionWorkspaceCudaTest, MoveAssignmentReleasesDestinationAndKeepsSourceProvenance)
{
    test::CudaStreamGuard source_stream;
    test::CudaStreamGuard destination_stream;
    ReductionWorkspace source;
    ReductionWorkspace destination;
    int device = 0;
    cudaMemPool_t memory_pool = nullptr;
    PLAMATRIX_CHECK_CUDA(cudaGetDevice(&device));
    PLAMATRIX_CHECK_CUDA(cudaDeviceGetDefaultMemPool(&memory_pool, device));
    std::uint64_t used_before = 0;
    PLAMATRIX_CHECK_CUDA(cudaMemPoolGetAttribute(
        memory_pool, cudaMemPoolAttrUsedMemCurrent, &used_before));
    source.reserveBytesAsync(512, source_stream.get());
    destination.reserveBytesAsync(256, destination_stream.get());
    source_stream.synchronize();
    destination_stream.synchronize();
    std::uint64_t used_after_allocations = 0;
    PLAMATRIX_CHECK_CUDA(cudaMemPoolGetAttribute(
        memory_pool, cudaMemPoolAttrUsedMemCurrent, &used_after_allocations));
    EXPECT_GT(used_after_allocations, used_before);
    const std::size_t source_capacity = source.capacityBytes();
    void* const source_data = source.data();

    destination = std::move(source);
    destination_stream.synchronize();
    std::uint64_t used_after_assignment = 0;
    PLAMATRIX_CHECK_CUDA(cudaMemPoolGetAttribute(
        memory_pool, cudaMemPoolAttrUsedMemCurrent, &used_after_assignment));
    EXPECT_LT(used_after_assignment, used_after_allocations);
    EXPECT_EQ(source.capacityBytes(), 0U);
    EXPECT_EQ(source.data(), nullptr);
    EXPECT_EQ(destination.capacityBytes(), source_capacity);
    EXPECT_EQ(destination.data(), source_data);
    EXPECT_NO_THROW(destination.reserveBytesAsync(source_capacity, source_stream.get()));
    expectLogicErrorContaining([&] {
        destination.reserveBytesAsync(source_capacity, destination_stream.get());
    }, {"different stream", "synchronize", "close"});

    source_stream.synchronize();
    destination_stream.synchronize();
    destination.closeAsyncAllocation();
    source_stream.synchronize();
}

TYPED_TEST(ReductionTest, GpuNormalWorkspaceResetAllowsCrossStreamReuse)
{
    using Scalar = TypeParam;
    auto input_cpu = TestFixture::makeMatrix(2, 3, {
        Scalar(1), Scalar(4), Scalar(2), Scalar(5), Scalar(3), Scalar(6)
    });
    auto input = input_cpu.toGpu();
    test::CudaStreamGuard first_stream;
    test::CudaStreamGuard second_stream;
    ReductionWorkspace workspace;
    workspace.reserveBytes(1024U * 1024U);

    auto first = sumAsync(
        input, ReductionAxis::All, workspace, first_stream.get());
    first_stream.synchronize();
    expectGpuReductionResult(first, sum(input_cpu, ReductionAxis::All));
    first.closeAsyncAllocation();
    first_stream.synchronize();

    DenseMatrix<Scalar, Device::GPU> rejected_output(1, 1);
    expectLogicErrorContaining([&] {
        sumAsync(input, ReductionAxis::All, rejected_output, workspace, second_stream.get());
    }, {"different stream", "synchronize", "reset"});

    const std::size_t capacity = workspace.capacityBytes();
    void* const data = workspace.data();
    workspace.reserveBytes(capacity);
    EXPECT_EQ(workspace.capacityBytes(), capacity);
    EXPECT_EQ(workspace.data(), data);

    auto second = sumAsync(
        input, ReductionAxis::All, workspace, second_stream.get());
    second_stream.synchronize();
    expectGpuReductionResult(second, sum(input_cpu, ReductionAxis::All));
    second.closeAsyncAllocation();
    second_stream.synchronize();
}

TYPED_TEST(ReductionTest, GpuNaNsPropagateAndIndexedReductionsChooseLowestOffset)
{
    using Scalar = TypeParam;
    const Scalar nan = std::numeric_limits<Scalar>::quiet_NaN();
    auto input_cpu = TestFixture::makeMatrix(2, 4, {
        Scalar(5), Scalar(-3), nan, Scalar(4), nan, Scalar(-3), Scalar(-1), Scalar(4)
    });
    auto input = input_cpu.toGpu();

    expectGpuReductionResult(sum(input, ReductionAxis::Rows),
                             sum(input_cpu, ReductionAxis::Rows));
    expectGpuReductionResult(mean(input, ReductionAxis::Rows),
                             mean(input_cpu, ReductionAxis::Rows));
    expectGpuReductionResult(min(input, ReductionAxis::Rows),
                             min(input_cpu, ReductionAxis::Rows));
    expectGpuReductionResult(max(input, ReductionAxis::Rows),
                             max(input_cpu, ReductionAxis::Rows));

    const auto minimum = argMin(input, ReductionAxis::Rows);
    const auto maximum = argMax(input, ReductionAxis::Rows);
    expectGpuReductionResult(minimum.values, argMin(input_cpu, ReductionAxis::Rows).values);
    expectGpuReductionResult(maximum.values, argMax(input_cpu, ReductionAxis::Rows).values);
    TestFixture::expectValues(minimum.indices.toCpu(), {Index(1), Index(0)});
    TestFixture::expectValues(maximum.indices.toCpu(), {Index(1), Index(1)});
}

TYPED_TEST(ReductionTest, GpuEmptyDimensionsAndErrorsMatchCpuContract)
{
    using Scalar = TypeParam;
    DenseMatrix<Scalar, Device::CPU> zero_by_three_cpu(0, 3);
    DenseMatrix<Scalar, Device::CPU> three_by_zero_cpu(3, 0);
    auto zero_by_three = zero_by_three_cpu.toGpu();
    auto three_by_zero = three_by_zero_cpu.toGpu();

    const auto empty_rows = mean(zero_by_three, ReductionAxis::Rows);
    EXPECT_EQ(empty_rows.rows(), 0);
    EXPECT_EQ(empty_rows.cols(), 1);
    const auto empty_columns = argMax(three_by_zero, ReductionAxis::Columns);
    EXPECT_EQ(empty_columns.values.rows(), 1);
    EXPECT_EQ(empty_columns.values.cols(), 0);
    EXPECT_EQ(empty_columns.indices.rows(), 1);
    EXPECT_EQ(empty_columns.indices.cols(), 0);

    expectGpuReductionResult(sum(zero_by_three, ReductionAxis::All),
                             sum(zero_by_three_cpu, ReductionAxis::All));
    expectGpuReductionResult(sum(zero_by_three, ReductionAxis::Columns),
                             sum(zero_by_three_cpu, ReductionAxis::Columns));
    expectGpuReductionResult(sum(three_by_zero, ReductionAxis::Rows),
                             sum(three_by_zero_cpu, ReductionAxis::Rows));

    EXPECT_THROW(static_cast<void>(mean(zero_by_three, ReductionAxis::All)),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(min(zero_by_three, ReductionAxis::Columns)),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(max(three_by_zero, ReductionAxis::Rows)),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(argMin(three_by_zero, ReductionAxis::All)),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(argMax(zero_by_three, ReductionAxis::Columns)),
                 std::invalid_argument);

    DenseMatrix<Scalar, Device::GPU> input(0, 0);
    const auto invalid = static_cast<ReductionAxis>(99);
    EXPECT_THROW(static_cast<void>(sum(input, invalid)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(mean(input, invalid)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(min(input, invalid)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(max(input, invalid)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(argMin(input, invalid)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(argMax(input, invalid)), std::invalid_argument);
}

TYPED_TEST(ReductionTest, GpuAsyncReductionsUseIndependentWorkspacesAndStreams)
{
    using Scalar = TypeParam;
    auto first_cpu = TestFixture::makeMatrix(2, 3, {
        Scalar(1), Scalar(4), Scalar(2), Scalar(5), Scalar(3), Scalar(6)
    });
    auto second_cpu = TestFixture::makeMatrix(3, 2, {
        Scalar(-1), Scalar(8), Scalar(2), Scalar(7), Scalar(0), Scalar(7)
    });
    auto first = first_cpu.toGpu();
    auto second = second_cpu.toGpu();
    test::CudaStreamGuard first_stream;
    test::CudaStreamGuard second_stream;
    ReductionWorkspace first_workspace;
    ReductionWorkspace second_workspace;

    auto first_result = meanAsync(
        first, ReductionAxis::All, first_workspace, first_stream.get());
    auto second_result = argMaxAsync(
        second, ReductionAxis::All, second_workspace, second_stream.get());
    EXPECT_GT(first_workspace.capacityBytes(), 0U);
    EXPECT_GT(second_workspace.capacityBytes(), 0U);
    EXPECT_NE(first_workspace.data(), second_workspace.data());
    first_stream.synchronize();
    second_stream.synchronize();

    expectGpuReductionResult(first_result, mean(first_cpu, ReductionAxis::All));
    expectGpuReductionResult(second_result.values, argMax(second_cpu, ReductionAxis::All).values);
    expectGpuIndices(second_result.indices, argMax(second_cpu, ReductionAxis::All).indices);

    first_result.closeAsyncAllocation();
    second_result.values.closeAsyncAllocation();
    second_result.indices.closeAsyncAllocation();
    first_workspace.closeAsyncAllocation();
    second_workspace.closeAsyncAllocation();
    first_stream.synchronize();
    second_stream.synchronize();
}
#else
TYPED_TEST(ReductionTest, NoCudaReductionOverloadFamiliesReportOperationAndBuildOption)
{
    using Scalar = TypeParam;
    DenseMatrix<Scalar, Device::GPU> input(2, 3);
    DenseMatrix<Scalar, Device::GPU> values(2, 1);
    DenseMatrix<Index, Device::GPU> indices(2, 1);
    ReductionWorkspace workspace;
    cudaStream_t stream = nullptr;

#define EXPECT_NO_CUDA_VALUE_REDUCTION(OP, ASYNC_OP)                                    \
    expectNoCudaReductionError(#OP, [&] { static_cast<void>(OP(input, ReductionAxis::Rows)); }); \
    expectNoCudaReductionError(#OP, [&] {                                                \
        static_cast<void>(OP(input, ReductionAxis::Rows, workspace, stream));            \
    });                                                                                  \
    expectNoCudaReductionError(#OP, [&] {                                                \
        OP(input, ReductionAxis::Rows, values, workspace, stream);                       \
    });                                                                                  \
    expectNoCudaReductionError(#ASYNC_OP, [&] {                                          \
        static_cast<void>(ASYNC_OP(input, ReductionAxis::Rows, workspace, stream));      \
    });                                                                                  \
    expectNoCudaReductionError(#ASYNC_OP, [&] {                                          \
        ASYNC_OP(input, ReductionAxis::Rows, values, workspace, stream);                 \
    })

    EXPECT_NO_CUDA_VALUE_REDUCTION(sum, sumAsync);
    EXPECT_NO_CUDA_VALUE_REDUCTION(mean, meanAsync);
    EXPECT_NO_CUDA_VALUE_REDUCTION(min, minAsync);
    EXPECT_NO_CUDA_VALUE_REDUCTION(max, maxAsync);
#undef EXPECT_NO_CUDA_VALUE_REDUCTION

#define EXPECT_NO_CUDA_INDEXED_REDUCTION(OP, ASYNC_OP)                                  \
    expectNoCudaReductionError(#OP, [&] { static_cast<void>(OP(input, ReductionAxis::Rows)); }); \
    expectNoCudaReductionError(#OP, [&] {                                                \
        static_cast<void>(OP(input, ReductionAxis::Rows, workspace, stream));            \
    });                                                                                  \
    expectNoCudaReductionError(#OP, [&] {                                                \
        OP(input, ReductionAxis::Rows, values, indices, workspace, stream);              \
    });                                                                                  \
    expectNoCudaReductionError(#ASYNC_OP, [&] {                                          \
        static_cast<void>(ASYNC_OP(input, ReductionAxis::Rows, workspace, stream));      \
    });                                                                                  \
    expectNoCudaReductionError(#ASYNC_OP, [&] {                                          \
        ASYNC_OP(input, ReductionAxis::Rows, values, indices, workspace, stream);        \
    })

    EXPECT_NO_CUDA_INDEXED_REDUCTION(argMin, argMinAsync);
    EXPECT_NO_CUDA_INDEXED_REDUCTION(argMax, argMaxAsync);
#undef EXPECT_NO_CUDA_INDEXED_REDUCTION
}

TEST(ReductionNoCudaTest, WorkspaceOperationsHaveExplicitCpuOnlyBehavior)
{
    ReductionWorkspace workspace;
    EXPECT_EQ(workspace.capacityBytes(), 0U);
    EXPECT_EQ(workspace.data(), nullptr);
    const ReductionWorkspace& const_workspace = workspace;
    EXPECT_EQ(const_workspace.data(), nullptr);

    expectNoCudaReductionError("ReductionWorkspace::reserveBytes", [&] {
        workspace.reserveBytes(64);
    });
    expectNoCudaReductionError("ReductionWorkspace::reserveBytesAsync", [&] {
        workspace.reserveBytesAsync(64, nullptr);
    });
    EXPECT_NO_THROW(workspace.closeAsyncAllocation());

    ReductionWorkspace moved(std::move(workspace));
    EXPECT_EQ(workspace.capacityBytes(), 0U);
    EXPECT_EQ(moved.capacityBytes(), 0U);
}
#endif

#ifdef PLAMATRIX_USE_FLOAT
TEST(ReductionFloatTest, SumAndMeanAccumulateInDouble)
{
    DenseMatrix<float, Device::CPU> input(1, 3);
    input(0, 0) = 100000000.0f;
    input(0, 1) = 1.0f;
    input(0, 2) = -100000000.0f;
    const double reference = static_cast<double>(input(0, 0))
                           + static_cast<double>(input(0, 1))
                           + static_cast<double>(input(0, 2));

    EXPECT_EQ(sum(input, ReductionAxis::Rows).data()[0], static_cast<float>(reference));
    EXPECT_EQ(mean(input, ReductionAxis::Rows).data()[0], static_cast<float>(reference / 3.0));
    EXPECT_EQ(sum(input, ReductionAxis::All).data()[0], static_cast<float>(reference));

    auto column_input = input.transpose();
    EXPECT_EQ(sum(column_input, ReductionAxis::Columns).data()[0], static_cast<float>(reference));
    EXPECT_EQ(mean(column_input, ReductionAxis::Columns).data()[0], static_cast<float>(reference / 3.0));
}

TEST(ReductionFloatTest, MeanCastsOnlyAfterDivision)
{
    const float largest = std::numeric_limits<float>::max();
    DenseMatrix<float, Device::CPU> input(1, 2);
    input(0, 0) = largest;
    input(0, 1) = largest;

    EXPECT_EQ(mean(input, ReductionAxis::All).data()[0], largest);
    EXPECT_EQ(mean(input, ReductionAxis::Rows).data()[0], largest);
}
#endif

} // anonymous namespace
} // namespace plamatrix
