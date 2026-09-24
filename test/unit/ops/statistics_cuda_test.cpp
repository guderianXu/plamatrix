#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>

#include <gtest/gtest.h>

#include "plamatrix/internal/ops/statistics.h"
#include "plamatrix/internal/device/device_matrix.h"
#include "../../support/cuda_test_utils.h"

#ifdef PLAMATRIX_WITH_CUDA

namespace plamatrix::internal
{
    namespace
    {

        template <typename Scalar>
        DenseStorage<Scalar, Device::CPU> makeMatrix(Index rows, Index cols, std::initializer_list<Scalar> values)
        {
            DenseStorage<Scalar, Device::CPU> matrix(rows, cols);
            std::copy(values.begin(), values.end(), matrix.data());
            return matrix;
        }

        template <typename Scalar> class FiniteStatisticsCudaTest : public ::testing::Test
        {
        };

#if defined(PLAMATRIX_USE_FLOAT) && defined(PLAMATRIX_USE_DOUBLE)
        using FloatingTypes = ::testing::Types<float, double>;
#elif defined(PLAMATRIX_USE_FLOAT)
        using FloatingTypes = ::testing::Types<float>;
#else
        using FloatingTypes = ::testing::Types<double>;
#endif

        TYPED_TEST_SUITE(FiniteStatisticsCudaTest, FloatingTypes);

        template <typename Scalar> MatrixView<Scalar, Device::GPU> residentView(ResidentMatrix<Scalar>& matrix)
        {
            return {matrix.data(), matrix.rows(), matrix.cols(), 1, matrix.rows()};
        }

        TYPED_TEST(FiniteStatisticsCudaTest, ResidentViewsUseExistingKernelsWithoutDenseMatrix)
        {
            using Scalar = TypeParam;
            auto context = ExecutionContext::create({Backend::Cuda, 0});
            Matrix<Scalar, Dynamic, Dynamic> host(3, 2);
            host << Scalar(2), Scalar(20), std::numeric_limits<Scalar>::quiet_NaN(), Scalar(40), Scalar(-1), Scalar(10);
            auto input = ResidentMatrix<Scalar>::copyFrom(host, context);
            ResidentMatrix<std::uint8_t> mask(3, 1, context);
            ResidentMatrix<Scalar> minimum(1, 2, context);
            ResidentMatrix<Scalar> maximum(1, 2, context);
            ResidentMatrix<Index> count(1, 1, context);
            test::CudaStreamGuard stream(cudaStreamNonBlocking);
            ReductionWorkspace workspace;
            const auto input_view = residentView(input).asConst();

            finiteRowMask(input_view, residentView(mask), workspace, stream.get());
            EXPECT_EQ(mask.toHostMatrix()(1, 0), 0);
            finiteColumnBoundsAsync(
                input_view, residentView(minimum), residentView(maximum), residentView(count), workspace, stream.get());
            stream.synchronize();
            EXPECT_EQ(count.toHostMatrix()(0, 0), 2);
            EXPECT_EQ(minimum.toHostMatrix()(0, 0), Scalar(-1));
            EXPECT_EQ(maximum.toHostMatrix()(0, 1), Scalar(20));
            finiteColumnBoundsWithMask(input_view,
                                       residentView(mask),
                                       residentView(minimum),
                                       residentView(maximum),
                                       residentView(count),
                                       workspace,
                                       stream.get());
            EXPECT_EQ(mask.toHostMatrix()(2, 0), 1);
            EXPECT_EQ(count.toHostMatrix()(0, 0), 2);
            workspace.closeAsyncAllocation();
            stream.synchronize();
        }

        TYPED_TEST(FiniteStatisticsCudaTest, ViewsRejectStridesAndOverlapBeforeWriting)
        {
            using Scalar = TypeParam;
            auto context = ExecutionContext::create({Backend::Cuda, 0});
            ResidentMatrix<Scalar> storage(4, 3, context);
            ResidentMatrix<std::uint8_t> mask(3, 1, context);
            ResidentMatrix<Index> count(1, 1, context);
            ReductionWorkspace workspace;
            ConstMatrixView<Scalar, Device::GPU> strided(storage.data(), 3, 2, 1, 4);
            EXPECT_THROW(finiteRowMask(strided, residentView(mask), workspace), std::invalid_argument);
            const auto input = residentView(storage).asConst();
            MatrixView<Scalar, Device::GPU> minimum(storage.data() + 1, 1, 3, 1, 1);
            ResidentMatrix<Scalar> maximum(1, 3, context);
            EXPECT_THROW(finiteColumnBounds(input, minimum, residentView(maximum), residentView(count), workspace),
                         std::invalid_argument);
        }

        TYPED_TEST(FiniteStatisticsCudaTest, MatchesCpuForNanInfinityAndAllInvalidRows)
        {
            using Scalar = TypeParam;
            const Scalar nan = std::numeric_limits<Scalar>::quiet_NaN();
            const Scalar infinity = std::numeric_limits<Scalar>::infinity();
            const auto input = makeMatrix<Scalar>(4,
                                                  2,
                                                  {
                                                      Scalar(1),
                                                      nan,
                                                      Scalar(3),
                                                      Scalar(4),
                                                      Scalar(10),
                                                      Scalar(20),
                                                      infinity,
                                                      Scalar(40),
                                                  });
            const auto gpu_input = input.toGpu();

            const auto mask = finiteRowMask(gpu_input).toCpu();
            const auto bounds = finiteColumnBounds(gpu_input);
            const auto combined = finiteColumnBoundsWithMask(gpu_input);
            const auto minimum = bounds.minimum.toCpu();
            const auto maximum = bounds.maximum.toCpu();
            const auto count = bounds.validRowCount.toCpu();
            const auto combined_mask = combined.rowMask.toCpu();
            const auto combined_minimum = combined.minimum.toCpu();
            const auto combined_maximum = combined.maximum.toCpu();
            const auto combined_count = combined.validRowCount.toCpu();

            EXPECT_EQ(mask.data()[0], 1);
            EXPECT_EQ(mask.data()[1], 0);
            EXPECT_EQ(mask.data()[2], 0);
            EXPECT_EQ(mask.data()[3], 1);
            EXPECT_EQ(count.data()[0], 2);
            EXPECT_EQ(minimum.data()[0], Scalar(1));
            EXPECT_EQ(maximum.data()[0], Scalar(4));
            EXPECT_EQ(minimum.data()[1], Scalar(10));
            EXPECT_EQ(maximum.data()[1], Scalar(40));
            EXPECT_EQ(combined_mask.data()[0], 1);
            EXPECT_EQ(combined_mask.data()[1], 0);
            EXPECT_EQ(combined_mask.data()[2], 0);
            EXPECT_EQ(combined_mask.data()[3], 1);
            EXPECT_EQ(combined_count.data()[0], 2);
            EXPECT_EQ(combined_minimum.data()[0], Scalar(1));
            EXPECT_EQ(combined_maximum.data()[0], Scalar(4));
            EXPECT_EQ(combined_minimum.data()[1], Scalar(10));
            EXPECT_EQ(combined_maximum.data()[1], Scalar(40));

            const auto invalid = makeMatrix<Scalar>(2, 1, {nan, infinity}).toGpu();
            const auto invalid_bounds = finiteColumnBounds(invalid);
            const auto invalid_combined = finiteColumnBoundsWithMask(invalid);
            EXPECT_EQ(invalid_bounds.validRowCount.toCpu().data()[0], 0);
            EXPECT_TRUE(std::isnan(invalid_bounds.minimum.toCpu().data()[0]));
            EXPECT_TRUE(std::isnan(invalid_bounds.maximum.toCpu().data()[0]));
            EXPECT_EQ(invalid_combined.rowMask.toCpu().data()[0], 0);
            EXPECT_EQ(invalid_combined.rowMask.toCpu().data()[1], 0);
            EXPECT_EQ(invalid_combined.validRowCount.toCpu().data()[0], 0);
            EXPECT_TRUE(std::isnan(invalid_combined.minimum.toCpu().data()[0]));
            EXPECT_TRUE(std::isnan(invalid_combined.maximum.toCpu().data()[0]));
        }

        TYPED_TEST(FiniteStatisticsCudaTest, CallerOwnedSynchronousOutputsSupportNonDefaultStream)
        {
            using Scalar = TypeParam;
            const Scalar nan = std::numeric_limits<Scalar>::quiet_NaN();
            const auto input = makeMatrix<Scalar>(3,
                                                  2,
                                                  {
                                                      Scalar(1),
                                                      nan,
                                                      Scalar(3),
                                                      Scalar(10),
                                                      Scalar(20),
                                                      Scalar(30),
                                                  })
                                   .toGpu();
            test::CudaStreamGuard stream(cudaStreamNonBlocking);
            ReductionWorkspace workspace;
            DenseStorage<std::uint8_t, Device::GPU> mask(3, 1);
            DenseStorage<Scalar, Device::GPU> minimum(1, 2);
            DenseStorage<Scalar, Device::GPU> maximum(1, 2);
            DenseStorage<Index, Device::GPU> count(1, 1);

            finiteColumnBoundsWithMask(input, mask, minimum, maximum, count, workspace, stream.get());

            const auto cpu_mask = mask.toCpu();
            EXPECT_EQ(cpu_mask.data()[0], 1);
            EXPECT_EQ(cpu_mask.data()[1], 0);
            EXPECT_EQ(cpu_mask.data()[2], 1);
            EXPECT_EQ(count.toCpu().data()[0], 2);
            EXPECT_EQ(minimum.toCpu().data()[0], Scalar(1));
            EXPECT_EQ(maximum.toCpu().data()[0], Scalar(3));
            EXPECT_EQ(minimum.toCpu().data()[1], Scalar(10));
            EXPECT_EQ(maximum.toCpu().data()[1], Scalar(30));
        }

        TYPED_TEST(FiniteStatisticsCudaTest, AsyncWorkspaceReusesStorageWithoutImplicitSynchronization)
        {
            using Scalar = TypeParam;
            auto input = makeMatrix<Scalar>(4,
                                            2,
                                            {
                                                Scalar(1),
                                                Scalar(2),
                                                Scalar(3),
                                                Scalar(4),
                                                Scalar(10),
                                                Scalar(20),
                                                Scalar(30),
                                                Scalar(40),
                                            })
                             .toGpu();
            test::CudaStreamGuard stream(cudaStreamNonBlocking);
            ReductionWorkspace workspace;
            DenseStorage<std::uint8_t, Device::GPU> mask(4, 1);
            DenseStorage<Scalar, Device::GPU> minimum(1, 2);
            DenseStorage<Scalar, Device::GPU> maximum(1, 2);
            DenseStorage<Index, Device::GPU> count(1, 1);

            finiteRowMaskAsync(input, mask, workspace, stream.get());
            finiteColumnBoundsAsync(input, minimum, maximum, count, workspace, stream.get());
            finiteColumnBoundsAsync(input, minimum, maximum, count, workspace, stream.get());
            finiteColumnBoundsWithMaskAsync(input, mask, minimum, maximum, count, workspace, stream.get());
            auto allocated_mask = finiteRowMaskAsync(input, workspace, stream.get());
            auto allocated_bounds = finiteColumnBoundsAsync(input, workspace, stream.get());
            auto allocated_combined = finiteColumnBoundsWithMaskAsync(input, workspace, stream.get());
            stream.synchronize();

            const auto cpu_mask = mask.toCpu();
            const auto cpu_minimum = minimum.toCpu();
            const auto cpu_maximum = maximum.toCpu();
            EXPECT_EQ(cpu_mask.data()[0], 1);
            EXPECT_EQ(cpu_mask.data()[3], 1);
            EXPECT_EQ(count.toCpu().data()[0], 4);
            EXPECT_EQ(cpu_minimum.data()[0], Scalar(1));
            EXPECT_EQ(cpu_maximum.data()[0], Scalar(4));
            EXPECT_EQ(cpu_minimum.data()[1], Scalar(10));
            EXPECT_EQ(cpu_maximum.data()[1], Scalar(40));
            EXPECT_EQ(allocated_mask.toCpu().data()[2], 1);
            EXPECT_EQ(allocated_bounds.validRowCount.toCpu().data()[0], 4);
            EXPECT_EQ(allocated_bounds.minimum.toCpu().data()[1], Scalar(10));
            EXPECT_EQ(allocated_bounds.maximum.toCpu().data()[1], Scalar(40));
            EXPECT_EQ(allocated_combined.rowMask.toCpu().data()[2], 1);
            EXPECT_EQ(allocated_combined.validRowCount.toCpu().data()[0], 4);
            EXPECT_EQ(allocated_combined.minimum.toCpu().data()[1], Scalar(10));
            EXPECT_EQ(allocated_combined.maximum.toCpu().data()[1], Scalar(40));

            allocated_mask.closeAsyncAllocation();
            allocated_bounds.minimum.closeAsyncAllocation();
            allocated_bounds.maximum.closeAsyncAllocation();
            allocated_bounds.validRowCount.closeAsyncAllocation();
            allocated_combined.rowMask.closeAsyncAllocation();
            allocated_combined.minimum.closeAsyncAllocation();
            allocated_combined.maximum.closeAsyncAllocation();
            allocated_combined.validRowCount.closeAsyncAllocation();
            workspace.closeAsyncAllocation();
            stream.synchronize();
        }

        TYPED_TEST(FiniteStatisticsCudaTest, EmptyRowsAndZeroColumnsMatchCpuSemantics)
        {
            using Scalar = TypeParam;
            DenseStorage<Scalar, Device::CPU> empty_rows_cpu(0, 3);
            const auto empty_bounds = finiteColumnBounds(empty_rows_cpu.toGpu());
            const auto empty_combined = finiteColumnBoundsWithMask(empty_rows_cpu.toGpu());
            EXPECT_EQ(empty_bounds.validRowCount.toCpu().data()[0], 0);
            const auto empty_minimum = empty_bounds.minimum.toCpu();
            const auto empty_combined_minimum = empty_combined.minimum.toCpu();
            for (Index column = 0; column < 3; ++column)
            {
                EXPECT_TRUE(std::isnan(empty_minimum.data()[column]));
                EXPECT_TRUE(std::isnan(empty_combined_minimum.data()[column]));
            }
            EXPECT_EQ(empty_combined.rowMask.rows(), 0);
            EXPECT_EQ(empty_combined.validRowCount.toCpu().data()[0], 0);

            DenseStorage<Scalar, Device::CPU> zero_columns_cpu(3, 0);
            const auto zero_columns = zero_columns_cpu.toGpu();
            const auto zero_mask = finiteRowMask(zero_columns).toCpu();
            const auto zero_bounds = finiteColumnBounds(zero_columns);
            const auto zero_combined = finiteColumnBoundsWithMask(zero_columns);
            EXPECT_EQ(zero_mask.data()[0], 1);
            EXPECT_EQ(zero_mask.data()[2], 1);
            EXPECT_EQ(zero_bounds.validRowCount.toCpu().data()[0], 3);
            EXPECT_EQ(zero_bounds.minimum.cols(), 0);
            EXPECT_EQ(zero_combined.rowMask.toCpu().data()[0], 1);
            EXPECT_EQ(zero_combined.rowMask.toCpu().data()[2], 1);
            EXPECT_EQ(zero_combined.validRowCount.toCpu().data()[0], 3);
            EXPECT_EQ(zero_combined.minimum.cols(), 0);
        }

    } // namespace
} // namespace plamatrix::internal

#else

TEST(FiniteStatisticsViewsNoCuda, RejectsDeviceOperationsExplicitly)
{
    plamatrix::internal::ConstMatrixView<float, plamatrix::internal::Device::GPU> input;
    plamatrix::internal::MatrixView<float, plamatrix::internal::Device::GPU> bounds;
    plamatrix::internal::MatrixView<std::uint8_t, plamatrix::internal::Device::GPU> mask;
    plamatrix::internal::MatrixView<plamatrix::Index, plamatrix::internal::Device::GPU> count;
    plamatrix::internal::ReductionWorkspace workspace;
    EXPECT_THROW(plamatrix::internal::finiteRowMask(input, mask, workspace), std::runtime_error);
    EXPECT_THROW(plamatrix::internal::finiteColumnBounds(input, bounds, bounds, count, workspace), std::runtime_error);
    EXPECT_THROW(plamatrix::internal::finiteColumnBoundsWithMask(input, mask, bounds, bounds, count, workspace),
                 std::runtime_error);
}

#endif
