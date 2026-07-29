#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>

#include <gtest/gtest.h>

#include "plamatrix/core/parallel.h"
#include "plamatrix/dense/elementwise.h"
#include "support/cuda_test_utils.h"

namespace plamatrix
{
namespace
{

template <typename Scalar>
class ElementwiseTest : public ::testing::Test
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

    static void expectValues(const DenseMatrix<Scalar, Device::CPU>& matrix,
                             std::initializer_list<Scalar> expected)
    {
        ASSERT_EQ(matrix.size(), static_cast<Index>(expected.size()));
        Index index = 0;
        for (Scalar value : expected)
        {
            EXPECT_EQ(matrix.data()[index++], value);
        }
    }

    static void expectEmpty(const DenseMatrix<Scalar, Device::CPU>& matrix)
    {
        EXPECT_EQ(matrix.rows(), 0);
        EXPECT_EQ(matrix.cols(), 3);
        EXPECT_EQ(matrix.size(), 0);
        EXPECT_EQ(matrix.data(), nullptr);
    }

    static void expectGpuValues(const DenseMatrix<Scalar, Device::GPU>& matrix,
                                std::initializer_list<Scalar> expected)
    {
        expectValues(matrix.toCpu(), expected);
    }

    static void expectGpuEmpty(const DenseMatrix<Scalar, Device::GPU>& matrix)
    {
        expectEmpty(matrix.toCpu());
    }
};

using ScalarTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(ElementwiseTest, ScalarTypes);

#ifndef PLAMATRIX_WITH_CUDA
template <typename Callable>
void expectNoCudaElementwiseError(const char* operation, Callable&& callable)
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

TYPED_TEST(ElementwiseTest, MakeMatrixRejectsIncorrectInitializerLength)
{
    using Scalar = TypeParam;

    EXPECT_THROW(
        static_cast<void>(TestFixture::makeMatrix(2, 2, {Scalar(1), Scalar(2), Scalar(3)})),
        std::invalid_argument);
}

TYPED_TEST(ElementwiseTest, ScalarOperationsAllocateExpectedResults)
{
    using Scalar = TypeParam;
    auto input = TestFixture::makeMatrix(2, 3, {
        Scalar(-2), Scalar(-0.5), Scalar(0), Scalar(1), Scalar(2), Scalar(4)
    });

    auto scaled = plamatrix::scalarMultiply(input, Scalar(2));
    auto shifted = plamatrix::scalarAdd(input, Scalar(-1));
    auto divided = plamatrix::scalarDivide(input, Scalar(2));

    TestFixture::expectValues(scaled, {
        Scalar(-4), Scalar(-1), Scalar(0), Scalar(2), Scalar(4), Scalar(8)
    });
    TestFixture::expectValues(shifted, {
        Scalar(-3), Scalar(-1.5), Scalar(-1), Scalar(0), Scalar(1), Scalar(3)
    });
    TestFixture::expectValues(divided, {
        Scalar(-1), Scalar(-0.25), Scalar(0), Scalar(0.5), Scalar(1), Scalar(2)
    });
}

TYPED_TEST(ElementwiseTest, HadamardOperationsAllocateExpectedResults)
{
    using Scalar = TypeParam;
    auto input = TestFixture::makeMatrix(2, 3, {
        Scalar(-2), Scalar(-0.5), Scalar(0), Scalar(1), Scalar(2), Scalar(4)
    });
    auto other = TestFixture::makeMatrix(2, 3, {
        Scalar(1), Scalar(2), Scalar(-4), Scalar(0.5), Scalar(-2), Scalar(8)
    });

    auto product = plamatrix::hadamardMultiply(input, other);
    auto quotient = plamatrix::hadamardDivide(input, other);

    TestFixture::expectValues(product, {
        Scalar(-2), Scalar(-1), Scalar(0), Scalar(0.5), Scalar(-4), Scalar(32)
    });
    TestFixture::expectValues(quotient, {
        Scalar(-2), Scalar(-0.25), Scalar(0), Scalar(2), Scalar(-1), Scalar(0.5)
    });
}

TYPED_TEST(ElementwiseTest, UnaryAndClampOperationsAllocateExpectedResults)
{
    using Scalar = TypeParam;
    auto input = TestFixture::makeMatrix(2, 3, {
        Scalar(-2), Scalar(-0.5), Scalar(0), Scalar(1), Scalar(2), Scalar(4)
    });
    auto non_negative = TestFixture::makeMatrix(2, 3, {
        Scalar(0), Scalar(0.25), Scalar(1), Scalar(4), Scalar(9), Scalar(16)
    });

    auto absolute = plamatrix::absElements(input);
    auto rooted = plamatrix::sqrtElements(non_negative);
    auto clipped = plamatrix::clampElements(input, Scalar(-1), Scalar(1));

    TestFixture::expectValues(absolute, {
        Scalar(2), Scalar(0.5), Scalar(0), Scalar(1), Scalar(2), Scalar(4)
    });
    TestFixture::expectValues(rooted, {
        Scalar(0), Scalar(0.5), Scalar(1), Scalar(2), Scalar(3), Scalar(4)
    });
    TestFixture::expectValues(clipped, {
        Scalar(-1), Scalar(-0.5), Scalar(0), Scalar(1), Scalar(1), Scalar(1)
    });
}

TYPED_TEST(ElementwiseTest, EmptyMatricesPreserveDimensions)
{
    using Scalar = TypeParam;
    DenseMatrix<Scalar, Device::CPU> input(0, 3);
    DenseMatrix<Scalar, Device::CPU> other(0, 3);

    TestFixture::expectEmpty(plamatrix::scalarMultiply(input, Scalar(2)));
    TestFixture::expectEmpty(plamatrix::scalarAdd(input, Scalar(-1)));
    TestFixture::expectEmpty(plamatrix::scalarDivide(input, Scalar(2)));
    TestFixture::expectEmpty(plamatrix::hadamardMultiply(input, other));
    TestFixture::expectEmpty(plamatrix::hadamardDivide(input, other));
    TestFixture::expectEmpty(plamatrix::absElements(input));
    TestFixture::expectEmpty(plamatrix::sqrtElements(input));
    TestFixture::expectEmpty(plamatrix::clampElements(input, Scalar(-1), Scalar(1)));
}

TYPED_TEST(ElementwiseTest, RejectsInvalidArguments)
{
    using Scalar = TypeParam;
    DenseMatrix<Scalar, Device::CPU> input(2, 3);
    DenseMatrix<Scalar, Device::CPU> mismatch(3, 2);

    EXPECT_THROW(static_cast<void>(plamatrix::hadamardMultiply(input, mismatch)), std::runtime_error);
    EXPECT_THROW(static_cast<void>(plamatrix::hadamardDivide(input, mismatch)), std::runtime_error);
    EXPECT_THROW(static_cast<void>(plamatrix::scalarDivide(input, Scalar(0))), std::domain_error);
    EXPECT_THROW(
        static_cast<void>(plamatrix::clampElements(input, Scalar(2), Scalar(1))),
        std::invalid_argument);
}

TYPED_TEST(ElementwiseTest, HadamardDivisionByZeroUsesIeeeSemantics)
{
    using Scalar = TypeParam;
    auto numerator = TestFixture::makeMatrix(3, 1, {
        Scalar(1), Scalar(-1), Scalar(0)
    });
    auto denominator = TestFixture::makeMatrix(3, 1, {
        Scalar(0), Scalar(0), Scalar(0)
    });

    auto quotient = plamatrix::hadamardDivide(numerator, denominator);

    EXPECT_TRUE(std::isinf(quotient.data()[0]));
    EXPECT_FALSE(std::signbit(quotient.data()[0]));
    EXPECT_TRUE(std::isinf(quotient.data()[1]));
    EXPECT_TRUE(std::signbit(quotient.data()[1]));
    EXPECT_TRUE(std::isnan(quotient.data()[2]));
}

TYPED_TEST(ElementwiseTest, OpenMpBranchProcessesEveryElement)
{
    using Scalar = TypeParam;
    constexpr Index element_count = detail::kOpenMpWorkThreshold;
    DenseMatrix<Scalar, Device::CPU> input(element_count, 1);
    DenseMatrix<Scalar, Device::CPU> other(element_count, 1);

    ASSERT_TRUE(detail::shouldUseOpenMp(element_count));
    for (Index index = 0; index < element_count; ++index)
    {
        input.data()[index] = Scalar(static_cast<int>(index % 17) - 8);
        other.data()[index] = Scalar(index % 5 + 1);
    }

    auto absolute = plamatrix::absElements(input);
    auto product = plamatrix::hadamardMultiply(input, other);

    ASSERT_EQ(absolute.size(), element_count);
    ASSERT_EQ(product.size(), element_count);
    for (Index index = 0; index < element_count; ++index)
    {
        EXPECT_EQ(absolute.data()[index], std::abs(input.data()[index]));
        EXPECT_EQ(product.data()[index], input.data()[index] * other.data()[index]);
    }
}

#ifdef PLAMATRIX_WITH_CUDA
TYPED_TEST(ElementwiseTest, GpuSynchronousOperationsSupportAllocationAndOutputReuse)
{
    using Scalar = TypeParam;
    auto input_cpu = TestFixture::makeMatrix(2, 3, {
        Scalar(-2), Scalar(-0.5), Scalar(0), Scalar(1), Scalar(2), Scalar(4)
    });
    auto other_cpu = TestFixture::makeMatrix(2, 3, {
        Scalar(1), Scalar(2), Scalar(-4), Scalar(0.5), Scalar(-2), Scalar(8)
    });
    auto non_negative_cpu = TestFixture::makeMatrix(2, 3, {
        Scalar(0), Scalar(0.25), Scalar(1), Scalar(4), Scalar(9), Scalar(16)
    });
    auto input = input_cpu.toGpu();
    auto other = other_cpu.toGpu();
    auto non_negative = non_negative_cpu.toGpu();

    auto scaled = scalarMultiply(input, Scalar(2));
    auto shifted = scalarAdd(input, Scalar(-1));
    auto divided = scalarDivide(input, Scalar(2));
    auto product = hadamardMultiply(input, other);
    auto quotient = hadamardDivide(input, other);
    auto absolute = absElements(input);
    auto rooted = sqrtElements(non_negative);
    auto clipped = clampElements(input, Scalar(-1), Scalar(1));

    DenseMatrix<Scalar, Device::GPU> scaled_reuse(2, 3);
    DenseMatrix<Scalar, Device::GPU> shifted_reuse(2, 3);
    DenseMatrix<Scalar, Device::GPU> divided_reuse(2, 3);
    DenseMatrix<Scalar, Device::GPU> product_reuse(2, 3);
    DenseMatrix<Scalar, Device::GPU> quotient_reuse(2, 3);
    DenseMatrix<Scalar, Device::GPU> absolute_reuse(2, 3);
    DenseMatrix<Scalar, Device::GPU> rooted_reuse(2, 3);
    DenseMatrix<Scalar, Device::GPU> clipped_reuse(2, 3);
    test::CudaStreamGuard stream;
    scalarMultiply(input, Scalar(2), scaled_reuse, stream.get());
    scalarAdd(input, Scalar(-1), shifted_reuse, stream.get());
    scalarDivide(input, Scalar(2), divided_reuse, stream.get());
    hadamardMultiply(input, other, product_reuse, stream.get());
    hadamardDivide(input, other, quotient_reuse, stream.get());
    absElements(input, absolute_reuse, stream.get());
    sqrtElements(non_negative, rooted_reuse, stream.get());
    clampElements(input, Scalar(-1), Scalar(1), clipped_reuse, stream.get());
    stream.destroy();

    const std::initializer_list<Scalar> scaled_expected = {
        Scalar(-4), Scalar(-1), Scalar(0), Scalar(2), Scalar(4), Scalar(8)
    };
    const std::initializer_list<Scalar> shifted_expected = {
        Scalar(-3), Scalar(-1.5), Scalar(-1), Scalar(0), Scalar(1), Scalar(3)
    };
    const std::initializer_list<Scalar> divided_expected = {
        Scalar(-1), Scalar(-0.25), Scalar(0), Scalar(0.5), Scalar(1), Scalar(2)
    };
    const std::initializer_list<Scalar> product_expected = {
        Scalar(-2), Scalar(-1), Scalar(0), Scalar(0.5), Scalar(-4), Scalar(32)
    };
    const std::initializer_list<Scalar> quotient_expected = {
        Scalar(-2), Scalar(-0.25), Scalar(0), Scalar(2), Scalar(-1), Scalar(0.5)
    };
    const std::initializer_list<Scalar> absolute_expected = {
        Scalar(2), Scalar(0.5), Scalar(0), Scalar(1), Scalar(2), Scalar(4)
    };
    const std::initializer_list<Scalar> rooted_expected = {
        Scalar(0), Scalar(0.5), Scalar(1), Scalar(2), Scalar(3), Scalar(4)
    };
    const std::initializer_list<Scalar> clipped_expected = {
        Scalar(-1), Scalar(-0.5), Scalar(0), Scalar(1), Scalar(1), Scalar(1)
    };
    TestFixture::expectGpuValues(scaled, scaled_expected);
    TestFixture::expectGpuValues(scaled_reuse, scaled_expected);
    TestFixture::expectGpuValues(shifted, shifted_expected);
    TestFixture::expectGpuValues(shifted_reuse, shifted_expected);
    TestFixture::expectGpuValues(divided, divided_expected);
    TestFixture::expectGpuValues(divided_reuse, divided_expected);
    TestFixture::expectGpuValues(product, product_expected);
    TestFixture::expectGpuValues(product_reuse, product_expected);
    TestFixture::expectGpuValues(quotient, quotient_expected);
    TestFixture::expectGpuValues(quotient_reuse, quotient_expected);
    TestFixture::expectGpuValues(absolute, absolute_expected);
    TestFixture::expectGpuValues(absolute_reuse, absolute_expected);
    TestFixture::expectGpuValues(rooted, rooted_expected);
    TestFixture::expectGpuValues(rooted_reuse, rooted_expected);
    TestFixture::expectGpuValues(clipped, clipped_expected);
    TestFixture::expectGpuValues(clipped_reuse, clipped_expected);
}

TYPED_TEST(ElementwiseTest, GpuSynchronousAllocatingResultsOutliveExplicitStream)
{
    using Scalar = TypeParam;
    auto input_cpu = TestFixture::makeMatrix(2, 3, {
        Scalar(-2), Scalar(-0.5), Scalar(0), Scalar(1), Scalar(2), Scalar(4)
    });
    auto other_cpu = TestFixture::makeMatrix(2, 3, {
        Scalar(1), Scalar(2), Scalar(-4), Scalar(0.5), Scalar(-2), Scalar(8)
    });
    auto input = input_cpu.toGpu();
    auto other = other_cpu.toGpu();
    test::CudaStreamGuard stream;

    auto scalar_result = scalarMultiply(input, Scalar(2), stream.get());
    auto unary_result = absElements(input, stream.get());
    auto binary_result = hadamardMultiply(input, other, stream.get());
    auto clamp_result = clampElements(input, Scalar(-1), Scalar(1), stream.get());

    stream.destroy();

    TestFixture::expectGpuValues(scalar_result, {
        Scalar(-4), Scalar(-1), Scalar(0), Scalar(2), Scalar(4), Scalar(8)
    });
    TestFixture::expectGpuValues(unary_result, {
        Scalar(2), Scalar(0.5), Scalar(0), Scalar(1), Scalar(2), Scalar(4)
    });
    TestFixture::expectGpuValues(binary_result, {
        Scalar(-2), Scalar(-1), Scalar(0), Scalar(0.5), Scalar(-4), Scalar(32)
    });
    TestFixture::expectGpuValues(clamp_result, {
        Scalar(-1), Scalar(-0.5), Scalar(0), Scalar(1), Scalar(1), Scalar(1)
    });
}

TYPED_TEST(ElementwiseTest, GpuAsyncOperationsUseTwoIndependentStreams)
{
    using Scalar = TypeParam;
    auto input_cpu = TestFixture::makeMatrix(2, 3, {
        Scalar(-2), Scalar(-0.5), Scalar(0), Scalar(1), Scalar(2), Scalar(4)
    });
    auto other_cpu = TestFixture::makeMatrix(2, 3, {
        Scalar(1), Scalar(2), Scalar(-4), Scalar(0.5), Scalar(-2), Scalar(8)
    });
    auto non_negative_cpu = TestFixture::makeMatrix(2, 3, {
        Scalar(0), Scalar(0.25), Scalar(1), Scalar(4), Scalar(9), Scalar(16)
    });
    auto input = input_cpu.toGpu();
    auto other = other_cpu.toGpu();
    auto non_negative = non_negative_cpu.toGpu();

    test::CudaStreamGuard first_stream;
    test::CudaStreamGuard second_stream;

    auto scaled = scalarMultiplyAsync(input, Scalar(2), first_stream.get());
    auto shifted = scalarAddAsync(input, Scalar(-1), second_stream.get());
    auto divided = scalarDivideAsync(input, Scalar(2), first_stream.get());
    auto product = hadamardMultiplyAsync(input, other, first_stream.get());
    auto quotient = hadamardDivideAsync(input, other, second_stream.get());
    auto absolute = absElementsAsync(input, first_stream.get());
    auto rooted = sqrtElementsAsync(non_negative, second_stream.get());
    auto clipped = clampElementsAsync(input, Scalar(-1), Scalar(1), second_stream.get());

    DenseMatrix<Scalar, Device::GPU> scaled_reuse(2, 3);
    DenseMatrix<Scalar, Device::GPU> shifted_reuse(2, 3);
    DenseMatrix<Scalar, Device::GPU> divided_reuse(2, 3);
    DenseMatrix<Scalar, Device::GPU> product_reuse(2, 3);
    DenseMatrix<Scalar, Device::GPU> quotient_reuse(2, 3);
    DenseMatrix<Scalar, Device::GPU> absolute_reuse(2, 3);
    DenseMatrix<Scalar, Device::GPU> rooted_reuse(2, 3);
    DenseMatrix<Scalar, Device::GPU> clipped_reuse(2, 3);
    scalarMultiplyAsync(input, Scalar(2), scaled_reuse, first_stream.get());
    scalarAddAsync(input, Scalar(-1), shifted_reuse, second_stream.get());
    scalarDivideAsync(input, Scalar(2), divided_reuse, first_stream.get());
    hadamardMultiplyAsync(input, other, product_reuse, first_stream.get());
    hadamardDivideAsync(input, other, quotient_reuse, second_stream.get());
    absElementsAsync(input, absolute_reuse, first_stream.get());
    sqrtElementsAsync(non_negative, rooted_reuse, second_stream.get());
    clampElementsAsync(input, Scalar(-1), Scalar(1), clipped_reuse, second_stream.get());

    first_stream.synchronize();
    second_stream.synchronize();

    TestFixture::expectGpuValues(scaled, {
        Scalar(-4), Scalar(-1), Scalar(0), Scalar(2), Scalar(4), Scalar(8)
    });
    TestFixture::expectGpuValues(scaled_reuse, {
        Scalar(-4), Scalar(-1), Scalar(0), Scalar(2), Scalar(4), Scalar(8)
    });
    TestFixture::expectGpuValues(shifted, {
        Scalar(-3), Scalar(-1.5), Scalar(-1), Scalar(0), Scalar(1), Scalar(3)
    });
    TestFixture::expectGpuValues(shifted_reuse, {
        Scalar(-3), Scalar(-1.5), Scalar(-1), Scalar(0), Scalar(1), Scalar(3)
    });
    TestFixture::expectGpuValues(divided, {
        Scalar(-1), Scalar(-0.25), Scalar(0), Scalar(0.5), Scalar(1), Scalar(2)
    });
    TestFixture::expectGpuValues(divided_reuse, {
        Scalar(-1), Scalar(-0.25), Scalar(0), Scalar(0.5), Scalar(1), Scalar(2)
    });
    TestFixture::expectGpuValues(product, {
        Scalar(-2), Scalar(-1), Scalar(0), Scalar(0.5), Scalar(-4), Scalar(32)
    });
    TestFixture::expectGpuValues(product_reuse, {
        Scalar(-2), Scalar(-1), Scalar(0), Scalar(0.5), Scalar(-4), Scalar(32)
    });
    TestFixture::expectGpuValues(quotient, {
        Scalar(-2), Scalar(-0.25), Scalar(0), Scalar(2), Scalar(-1), Scalar(0.5)
    });
    TestFixture::expectGpuValues(quotient_reuse, {
        Scalar(-2), Scalar(-0.25), Scalar(0), Scalar(2), Scalar(-1), Scalar(0.5)
    });
    TestFixture::expectGpuValues(absolute, {
        Scalar(2), Scalar(0.5), Scalar(0), Scalar(1), Scalar(2), Scalar(4)
    });
    TestFixture::expectGpuValues(absolute_reuse, {
        Scalar(2), Scalar(0.5), Scalar(0), Scalar(1), Scalar(2), Scalar(4)
    });
    TestFixture::expectGpuValues(rooted, {
        Scalar(0), Scalar(0.5), Scalar(1), Scalar(2), Scalar(3), Scalar(4)
    });
    TestFixture::expectGpuValues(rooted_reuse, {
        Scalar(0), Scalar(0.5), Scalar(1), Scalar(2), Scalar(3), Scalar(4)
    });
    TestFixture::expectGpuValues(clipped, {
        Scalar(-1), Scalar(-0.5), Scalar(0), Scalar(1), Scalar(1), Scalar(1)
    });
    TestFixture::expectGpuValues(clipped_reuse, {
        Scalar(-1), Scalar(-0.5), Scalar(0), Scalar(1), Scalar(1), Scalar(1)
    });

    EXPECT_NO_THROW(scaled.closeAsyncAllocation());
    EXPECT_NO_THROW(shifted.closeAsyncAllocation());
    EXPECT_NO_THROW(divided.closeAsyncAllocation());
    EXPECT_NO_THROW(product.closeAsyncAllocation());
    EXPECT_NO_THROW(quotient.closeAsyncAllocation());
    EXPECT_NO_THROW(absolute.closeAsyncAllocation());
    EXPECT_NO_THROW(rooted.closeAsyncAllocation());
    EXPECT_NO_THROW(clipped.closeAsyncAllocation());
    first_stream.synchronize();
    second_stream.synchronize();
}

TYPED_TEST(ElementwiseTest, GpuOperationsPreserveEmptyDimensions)
{
    using Scalar = TypeParam;
    DenseMatrix<Scalar, Device::CPU> input_cpu(0, 3);
    DenseMatrix<Scalar, Device::CPU> other_cpu(0, 3);
    auto input = input_cpu.toGpu();
    auto other = other_cpu.toGpu();

    TestFixture::expectGpuEmpty(scalarMultiply(input, Scalar(2)));
    TestFixture::expectGpuEmpty(scalarAdd(input, Scalar(-1)));
    TestFixture::expectGpuEmpty(scalarDivide(input, Scalar(2)));
    TestFixture::expectGpuEmpty(hadamardMultiply(input, other));
    TestFixture::expectGpuEmpty(hadamardDivide(input, other));
    TestFixture::expectGpuEmpty(absElements(input));
    TestFixture::expectGpuEmpty(sqrtElements(input));
    TestFixture::expectGpuEmpty(clampElements(input, Scalar(-1), Scalar(1)));
    TestFixture::expectGpuEmpty(scalarMultiplyAsync(input, Scalar(2)));
    TestFixture::expectGpuEmpty(scalarAddAsync(input, Scalar(-1)));
    TestFixture::expectGpuEmpty(scalarDivideAsync(input, Scalar(2)));
    TestFixture::expectGpuEmpty(hadamardMultiplyAsync(input, other));
    TestFixture::expectGpuEmpty(hadamardDivideAsync(input, other));
    TestFixture::expectGpuEmpty(absElementsAsync(input));
    TestFixture::expectGpuEmpty(sqrtElementsAsync(input));
    TestFixture::expectGpuEmpty(clampElementsAsync(input, Scalar(-1), Scalar(1)));
}

TYPED_TEST(ElementwiseTest, GpuOperationsRejectHostKnownInvalidArguments)
{
    using Scalar = TypeParam;
    DenseMatrix<Scalar, Device::GPU> input(2, 3);
    DenseMatrix<Scalar, Device::GPU> mismatch(3, 2);
    DenseMatrix<Scalar, Device::GPU> wrong_output(3, 2);

    EXPECT_THROW(static_cast<void>(hadamardMultiply(input, mismatch)), std::runtime_error);
    EXPECT_THROW(static_cast<void>(hadamardDivideAsync(input, mismatch)), std::runtime_error);
    EXPECT_THROW(scalarAdd(input, Scalar(1), wrong_output), std::runtime_error);
    EXPECT_THROW(absElementsAsync(input, wrong_output), std::runtime_error);
    EXPECT_THROW(static_cast<void>(scalarDivide(input, Scalar(0))), std::domain_error);
    EXPECT_THROW(
        static_cast<void>(scalarDivideAsync(input, Scalar(0))),
        std::domain_error);
    EXPECT_THROW(
        static_cast<void>(clampElements(input, Scalar(2), Scalar(1))),
        std::invalid_argument);
    EXPECT_THROW(
        static_cast<void>(clampElementsAsync(input, Scalar(2), Scalar(1))),
        std::invalid_argument);
}

TYPED_TEST(ElementwiseTest, GpuHadamardDivisionByZeroUsesIeeeSemantics)
{
    using Scalar = TypeParam;
    auto numerator_cpu = TestFixture::makeMatrix(3, 1, {
        Scalar(1), Scalar(-1), Scalar(0)
    });
    auto denominator_cpu = TestFixture::makeMatrix(3, 1, {
        Scalar(0), Scalar(0), Scalar(0)
    });
    auto numerator = numerator_cpu.toGpu();
    auto denominator = denominator_cpu.toGpu();

    auto sync_result = hadamardDivide(numerator, denominator).toCpu();
    auto async_gpu = hadamardDivideAsync(numerator, denominator);
    PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(nullptr));
    auto async_result = async_gpu.toCpu();

    for (const auto* result : {&sync_result, &async_result})
    {
        EXPECT_TRUE(std::isinf(result->data()[0]));
        EXPECT_FALSE(std::signbit(result->data()[0]));
        EXPECT_TRUE(std::isinf(result->data()[1]));
        EXPECT_TRUE(std::signbit(result->data()[1]));
        EXPECT_TRUE(std::isnan(result->data()[2]));
    }
}
#else
TYPED_TEST(ElementwiseTest, NoCudaElementwiseEntryPointsThrowRuntimeError)
{
    using Scalar = TypeParam;
    DenseMatrix<Scalar, Device::GPU> input(2, 3);
    DenseMatrix<Scalar, Device::GPU> other(2, 3);
    DenseMatrix<Scalar, Device::GPU> output(2, 3);

    expectNoCudaElementwiseError("scalarMultiply", [&] {
        static_cast<void>(scalarMultiplyAsync(input, Scalar(2)));
    });
    expectNoCudaElementwiseError("scalarMultiply", [&] {
        scalarMultiplyAsync(input, Scalar(2), output);
    });
    expectNoCudaElementwiseError("scalarMultiply", [&] {
        static_cast<void>(scalarMultiply(input, Scalar(2)));
    });
    expectNoCudaElementwiseError("scalarMultiply", [&] {
        scalarMultiply(input, Scalar(2), output);
    });

    expectNoCudaElementwiseError("scalarAdd", [&] {
        static_cast<void>(scalarAddAsync(input, Scalar(1)));
    });
    expectNoCudaElementwiseError("scalarAdd", [&] {
        scalarAddAsync(input, Scalar(1), output);
    });
    expectNoCudaElementwiseError("scalarAdd", [&] {
        static_cast<void>(scalarAdd(input, Scalar(1)));
    });
    expectNoCudaElementwiseError("scalarAdd", [&] {
        scalarAdd(input, Scalar(1), output);
    });

    expectNoCudaElementwiseError("scalarDivide", [&] {
        static_cast<void>(scalarDivideAsync(input, Scalar(2)));
    });
    expectNoCudaElementwiseError("scalarDivide", [&] {
        scalarDivideAsync(input, Scalar(2), output);
    });
    expectNoCudaElementwiseError("scalarDivide", [&] {
        static_cast<void>(scalarDivide(input, Scalar(2)));
    });
    expectNoCudaElementwiseError("scalarDivide", [&] {
        scalarDivide(input, Scalar(2), output);
    });

    expectNoCudaElementwiseError("hadamardMultiply", [&] {
        static_cast<void>(hadamardMultiplyAsync(input, other));
    });
    expectNoCudaElementwiseError("hadamardMultiply", [&] {
        hadamardMultiplyAsync(input, other, output);
    });
    expectNoCudaElementwiseError("hadamardMultiply", [&] {
        static_cast<void>(hadamardMultiply(input, other));
    });
    expectNoCudaElementwiseError("hadamardMultiply", [&] {
        hadamardMultiply(input, other, output);
    });

    expectNoCudaElementwiseError("hadamardDivide", [&] {
        static_cast<void>(hadamardDivideAsync(input, other));
    });
    expectNoCudaElementwiseError("hadamardDivide", [&] {
        hadamardDivideAsync(input, other, output);
    });
    expectNoCudaElementwiseError("hadamardDivide", [&] {
        static_cast<void>(hadamardDivide(input, other));
    });
    expectNoCudaElementwiseError("hadamardDivide", [&] {
        hadamardDivide(input, other, output);
    });

    expectNoCudaElementwiseError("absElements", [&] {
        static_cast<void>(absElementsAsync(input));
    });
    expectNoCudaElementwiseError("absElements", [&] {
        absElementsAsync(input, output);
    });
    expectNoCudaElementwiseError("absElements", [&] {
        static_cast<void>(absElements(input));
    });
    expectNoCudaElementwiseError("absElements", [&] {
        absElements(input, output);
    });

    expectNoCudaElementwiseError("sqrtElements", [&] {
        static_cast<void>(sqrtElementsAsync(input));
    });
    expectNoCudaElementwiseError("sqrtElements", [&] {
        sqrtElementsAsync(input, output);
    });
    expectNoCudaElementwiseError("sqrtElements", [&] {
        static_cast<void>(sqrtElements(input));
    });
    expectNoCudaElementwiseError("sqrtElements", [&] {
        sqrtElements(input, output);
    });

    expectNoCudaElementwiseError("clampElements", [&] {
        static_cast<void>(clampElementsAsync(input, Scalar(-1), Scalar(1)));
    });
    expectNoCudaElementwiseError("clampElements", [&] {
        clampElementsAsync(input, Scalar(-1), Scalar(1), output);
    });
    expectNoCudaElementwiseError("clampElements", [&] {
        static_cast<void>(clampElements(input, Scalar(-1), Scalar(1)));
    });
    expectNoCudaElementwiseError("clampElements", [&] {
        clampElements(input, Scalar(-1), Scalar(1), output);
    });
}
#endif

} // anonymous namespace
} // namespace plamatrix
