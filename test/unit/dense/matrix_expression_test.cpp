#include <cmath>
#include <limits>
#include <type_traits>
#include <sstream>

#include <gtest/gtest.h>

#ifdef PLAMATRIX_WITH_CUDA
#include <cuda_runtime_api.h>
#endif

#include "plamatrix/dense/matrix.h"

static_assert(std::is_same_v<typename plamatrix::ScalarBinaryOpTraits<float, double>::ReturnType, double>);

TEST(MatrixExpression, MatrixNormUsesEveryCoefficientForOwnedMappedAndLazyInputs)
{
    plamatrix::MatrixXd matrix(2, 3);
    matrix << 1.0, -2.0, 3.0, -4.0, 5.0, -6.0;
    EXPECT_DOUBLE_EQ(matrix.squaredNorm(), 91.0);
    EXPECT_DOUBLE_EQ(matrix.norm(), std::sqrt(91.0));
    EXPECT_DOUBLE_EQ((2.0 * matrix).squaredNorm(), 364.0);
    EXPECT_DOUBLE_EQ(matrix.transpose().norm(), std::sqrt(91.0));
    EXPECT_THROW(matrix.dot(matrix), plamatrix::internal::Error);

    double storage[] = {1.0, -4.0, 999.0, -2.0, 5.0, 999.0, 3.0, -6.0};
    using StridedMap = plamatrix::
        Map<plamatrix::MatrixXd, plamatrix::Unaligned, plamatrix::Stride<plamatrix::Dynamic, plamatrix::Dynamic>>;
    StridedMap mapped(storage, 2, 3, plamatrix::Stride<plamatrix::Dynamic, plamatrix::Dynamic>(3, 1));
    EXPECT_DOUBLE_EQ(mapped.squaredNorm(), 91.0);
    EXPECT_DOUBLE_EQ(mapped.norm(), std::sqrt(91.0));
    const plamatrix::MatrixXd empty(0, 3);
    EXPECT_DOUBLE_EQ(empty.norm(), 0.0);
    EXPECT_DOUBLE_EQ((empty + empty).squaredNorm(), 0.0);
}

TEST(MatrixExpression, PacketReductionsHandleTailsForFloatAndDouble)
{
    plamatrix::VectorXd left(137);
    plamatrix::VectorXd right(137);
    double expected_sum = 0.0;
    double expected_dot = 0.0;
    double expected_squared_norm = 0.0;
    for (plamatrix::Index index = 0; index < left.size(); ++index)
    {
        left(index) = static_cast<double>((index % 17) - 8) * 0.25;
        right(index) = static_cast<double>((index % 13) - 6) * 0.5;
        expected_sum += left(index);
        expected_dot += left(index) * right(index);
        expected_squared_norm += left(index) * left(index);
    }
    EXPECT_NEAR(left.sum(), expected_sum, 1.0e-12);
    EXPECT_NEAR(left.dot(right), expected_dot, 1.0e-12);
    EXPECT_NEAR(left.squaredNorm(), expected_squared_norm, 1.0e-12);
    EXPECT_NEAR((left.array() * right.array()).sum(), expected_dot, 1.0e-12);

    plamatrix::VectorXf float_left(259);
    plamatrix::VectorXf float_right(259);
    float expected_float_dot = 0.0F;
    for (plamatrix::Index index = 0; index < float_left.size(); ++index)
    {
        float_left(index) = static_cast<float>((index % 11) - 5) * 0.125F;
        float_right(index) = static_cast<float>((index % 7) - 3) * 0.25F;
        expected_float_dot += float_left(index) * float_right(index);
    }
    EXPECT_NEAR(float_left.dot(float_right), expected_float_dot, 1.0e-5F);
    EXPECT_NEAR((float_left.array() * float_right.array()).sum(), expected_float_dot, 1.0e-5F);
}

TEST(MatrixExpression, PacketReductionFusesMultiNodeArrayTrees)
{
    constexpr plamatrix::Index count = 139;
    plamatrix::VectorXd first(count);
    plamatrix::VectorXd second(count);
    plamatrix::VectorXd third(count);
    plamatrix::VectorXd fourth(count);
    double expected = 0.0;
    for (plamatrix::Index index = 0; index < count; ++index)
    {
        first(index) = static_cast<double>((index % 17) - 8) * 0.125;
        second(index) = static_cast<double>((index % 13) - 6) * 0.25;
        third(index) = static_cast<double>((index % 11) - 5) * 0.5;
        fourth(index) = static_cast<double>((index % 7) - 3) * 0.75;
        expected += 1.25 * first(index) - 0.75 * second(index) + third(index) * fourth(index) + 0.5;
    }

    const double actual = (1.25 * first.array() - second.array() * 0.75 + third.array() * fourth.array() + 0.5).sum();
    EXPECT_NEAR(actual, expected, 1.0e-11);

    plamatrix::VectorXf float_first(263);
    plamatrix::VectorXf float_second(263);
    plamatrix::VectorXf float_third(263);
    float expected_float = 0.0F;
    for (plamatrix::Index index = 0; index < float_first.size(); ++index)
    {
        float_first(index) = static_cast<float>((index % 9) - 4) * 0.25F;
        float_second(index) = static_cast<float>((index % 5) - 2) * 0.5F;
        float_third(index) = static_cast<float>((index % 3) - 1) * 0.75F;
        expected_float += float_first(index) * float_first(index) + float_second(index) * float_third(index);
    }
    EXPECT_NEAR(
        (float_first.array().square() + float_second.array() * float_third.array()).sum(), expected_float, 2.0e-4F);
}

TEST(MatrixExpression, MixedScalarsUseScalarBinaryOpTraits)
{
    plamatrix::Matrix2f left;
    left << 1.0F, 2.0F, 3.0F, 4.0F;
    plamatrix::Matrix2d right;
    right << 0.5, 1.5, 2.5, 3.5;

    const auto sum_expression = left + right;
    static_assert(std::is_same_v<typename decltype(sum_expression)::Scalar, double>);
    const plamatrix::Matrix2d sum = sum_expression;
    EXPECT_DOUBLE_EQ(sum(0, 0), 1.5);
    EXPECT_DOUBLE_EQ(sum(1, 1), 7.5);

    const auto product_expression = left * right;
    static_assert(std::is_same_v<typename decltype(product_expression)::Scalar, double>);
    const plamatrix::Matrix2d product = product_expression;
    EXPECT_DOUBLE_EQ(product(0, 0), 5.5);
    EXPECT_DOUBLE_EQ(product(1, 1), 18.5);

    plamatrix::Vector2f first(1.0F, 2.0F);
    plamatrix::Vector2d second(3.0, 4.0);
    static_assert(std::is_same_v<decltype(first.dot(second)), double>);
    EXPECT_DOUBLE_EQ(first.dot(second), 11.0);
}

TEST(MatrixExpression, ComplexNormDotConjugateAndAdjointMatchEigenSemantics)
{
    using Complex = std::complex<double>;
    plamatrix::Matrix<Complex, 2, 1> left(Complex{1.0, 2.0}, Complex{3.0, -1.0});
    plamatrix::Matrix<Complex, 2, 1> right(Complex{2.0, -1.0}, Complex{-1.0, 4.0});
    EXPECT_EQ(left.sum(), Complex(4.0, 1.0));
    EXPECT_NEAR(left.squaredNorm(), 15.0, 1.0e-14);
    EXPECT_NEAR(left.norm(), std::sqrt(15.0), 1.0e-14);
    EXPECT_NEAR(
        std::abs(left.dot(right) - (std::conj(left(0)) * right(0) + std::conj(left(1)) * right(1))), 0.0, 1.0e-14);

    plamatrix::Matrix<Complex, 2, 2> matrix;
    matrix << Complex{1.0, 2.0}, Complex{3.0, 4.0}, Complex{5.0, -1.0}, Complex{-2.0, 6.0};
    const auto adjoint = matrix.adjoint();
    EXPECT_EQ(adjoint(1, 0), std::conj(matrix(0, 1)));
    EXPECT_EQ(adjoint(0, 1), std::conj(matrix(1, 0)));
    EXPECT_TRUE(matrix.conjugate().allFinite());
}

TEST(MatrixExpression, NestedElementwiseOperationsRemainLazyUntilEvaluation)
{
    plamatrix::Matrix2d left = plamatrix::Matrix2d::Ones();
    plamatrix::Matrix2d right = plamatrix::Matrix2d::Constant(2.0);
    auto expression = left + right - 2.0 * left;
    static_assert(decltype(expression)::RowsAtCompileTime == 2);
    static_assert(decltype(expression)::ColsAtCompileTime == 2);
    EXPECT_DOUBLE_EQ(expression(0, 0), 1.0);
    left(0, 0) = 4.0;
    EXPECT_DOUBLE_EQ(expression(0, 0), -2.0);
    const plamatrix::Matrix2d result = expression;
    EXPECT_DOUBLE_EQ(result(0, 0), -2.0);
    EXPECT_DOUBLE_EQ(result(1, 1), 1.0);
    EXPECT_EQ(plamatrix::internal::MatrixAccess::executionInfo(result).reason, "Fused CPU expression evaluation");
    std::ostringstream printed;
    printed << expression;
    EXPECT_EQ(printed.str(), "-2 1\n1 1");
}

TEST(MatrixExpression, CapturesTemporaryMatricesAndExpressionsByValue)
{
    const plamatrix::Matrix2d identity = plamatrix::Matrix2d::Identity();
    auto expression = (plamatrix::Matrix2d::Ones() + identity).transpose() * identity;
    const plamatrix::Matrix2d result = expression;
    EXPECT_DOUBLE_EQ(result(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(result(1, 0), 1.0);
}

TEST(MatrixExpression, ProductAndTransposeComposeWithElementwiseOperations)
{
    plamatrix::MatrixXd left(3, 2);
    left << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
    plamatrix::Matrix2d expected;
    expected << 36.0, 44.0, 44.0, 57.0;
    const plamatrix::Matrix2d result = left.transpose() * left + plamatrix::Matrix2d::Identity();
    EXPECT_TRUE(result.isApprox(expected, 1e-12));
}

TEST(MatrixExpression, OrdinaryAssignmentSnapshotsAliasesAndNoaliasWritesDestination)
{
    plamatrix::Matrix2d matrix;
    matrix << 1.0, 2.0, 3.0, 4.0;
    const plamatrix::Matrix2d original = matrix;
    matrix = matrix.transpose() * matrix;
    const plamatrix::Matrix2d expected = original.transpose() * original;
    EXPECT_TRUE(matrix.isApprox(expected, 1e-12));

    plamatrix::Matrix2d destination;
    destination.noalias() = original.transpose() * original;
    EXPECT_TRUE(destination.isApprox(expected, 1e-12));
    EXPECT_EQ(plamatrix::internal::MatrixAccess::executionInfo(destination).reason,
              "CPU noalias expression assignment");
}

TEST(MatrixExpression, FixedBlocksAndDynamicViewsParticipateWithoutCopies)
{
    plamatrix::MatrixXd source = plamatrix::MatrixXd::Zero(4, 4);
    source.block<2, 2>(0, 0) = plamatrix::Matrix2d::Identity();
    const auto expression = source.block<2, 2>(0, 0) + source.block<2, 2>(0, 0).transpose();
    static_assert(decltype(expression)::RowsAtCompileTime == 2);
    plamatrix::Matrix2d result = expression;
    EXPECT_DOUBLE_EQ(result(0, 0), 2.0);
    source(0, 0) = 3.0;
    result = expression;
    EXPECT_DOUBLE_EQ(result(0, 0), 6.0);
}

TEST(MatrixExpression, CoefficientwiseDivisionReductionsAndCompoundAssignmentCompose)
{
    plamatrix::Matrix2d left = plamatrix::Matrix2d::Constant(4.0);
    plamatrix::Matrix2d right = plamatrix::Matrix2d::Constant(2.0);
    const auto expression = (left.cwiseProduct(right) + left) / 2.0;
    EXPECT_DOUBLE_EQ(expression.sum(), 24.0);
    EXPECT_DOUBLE_EQ(expression.mean(), 6.0);
    left += expression;
    EXPECT_DOUBLE_EQ(left(0, 0), 10.0);

    plamatrix::Matrix<int, 2, 1> integers(5, -9);
    const auto quotient = (integers + integers) / 2;
    EXPECT_EQ(quotient(0), 5);
    EXPECT_EQ(quotient(1), -9);
    EXPECT_THROW(static_cast<void>(integers / 0), plamatrix::internal::Error);
}

TEST(MatrixExpression, ArrayOperationsComposeWithoutEagerMatrices)
{
    plamatrix::Matrix2d left;
    left << 1.0, 2.0, 3.0, 4.0;
    const plamatrix::Matrix2d right = plamatrix::Matrix2d::Constant(2.0);
    auto expression = (left + right).array().square() + left.array() * right.array();
    static_assert(decltype(expression)::RowsAtCompileTime == 2);
    EXPECT_DOUBLE_EQ(expression(0, 0), 11.0);
    left(0, 0) = 5.0;
    EXPECT_DOUBLE_EQ(expression(0, 0), 59.0);
    EXPECT_DOUBLE_EQ(expression.maxCoeff(), 59.0);
    EXPECT_DOUBLE_EQ(expression.minCoeff(), 20.0);
    const plamatrix::Matrix2d result = expression;
    EXPECT_DOUBLE_EQ(result(0, 0), 59.0);
    EXPECT_DOUBLE_EQ(result(1, 1), 44.0);

    const plamatrix::Matrix2d shifted = 3.0 + (left.array() - 1.0);
    EXPECT_DOUBLE_EQ(shifted(0, 0), 7.0);
    const plamatrix::Matrix2d quotient = (left + right).array() / right.array();
    EXPECT_DOUBLE_EQ(quotient(0, 0), 3.5);
    const plamatrix::Matrix2d temporary = plamatrix::Matrix2d::Ones().array().square();
    EXPECT_DOUBLE_EQ(temporary(1, 1), 1.0);

    plamatrix::Matrix<int, 2, 1> numerator(4, 8);
    plamatrix::Matrix<int, 2, 1> denominator(2, 0);
    EXPECT_THROW(static_cast<void>((numerator.array() / denominator.array()).eval()), plamatrix::internal::Error);
}

TEST(MatrixExpression, AxiswiseReductionsAndBroadcastComposeWithExpressions)
{
    plamatrix::Matrix<double, 2, 3> values;
    values << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
    const plamatrix::Matrix<double, 2, 3> doubled = values + values;
    auto sums = (values + doubled).rowwise().sum();
    static_assert(decltype(sums)::RowsAtCompileTime == 2);
    static_assert(decltype(sums)::ColsAtCompileTime == 1);
    EXPECT_DOUBLE_EQ(sums(0), 18.0);
    EXPECT_DOUBLE_EQ(sums(1), 45.0);
    EXPECT_DOUBLE_EQ((values + doubled).colwise().mean()(0, 2), 13.5);

    plamatrix::RowVectorXd offset(3);
    offset << 10.0, 20.0, 30.0;
    const auto broadcast = ((values + doubled).rowwise() + offset).array().square();
    const plamatrix::Matrix<double, 2, 3> result = broadcast;
    EXPECT_DOUBLE_EQ(result(0, 0), 169.0);
    EXPECT_DOUBLE_EQ(result(1, 2), 2304.0);
    values = values.colwise() - plamatrix::Vector2d(1.0, 2.0);
    EXPECT_DOUBLE_EQ(values(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(values(1, 2), 4.0);
    EXPECT_THROW(static_cast<void>(values.rowwise() + plamatrix::RowVectorXd::Ones(2)), plamatrix::internal::Error);
    EXPECT_THROW(static_cast<void>(plamatrix::MatrixXd(2, 0).rowwise().mean()), plamatrix::internal::Error);
}

TEST(MatrixExpression, UnsupportedArrayRequiredDeviceFailsExplicitly)
{
    plamatrix::Matrix2d values = plamatrix::Matrix2d::Ones();
    plamatrix::internal::ScopedExecutionPolicy required(plamatrix::internal::ExecutionPolicy::GpuRequired);
    EXPECT_THROW(static_cast<void>(plamatrix::Matrix2d(values.array().square())), plamatrix::internal::Error);
    EXPECT_THROW(static_cast<void>(plamatrix::Vector2d(values.rowwise().sum())), plamatrix::internal::Error);
}

TEST(MatrixExpression, VectorAndMatrixQueriesWorkBeforeMaterialization)
{
    const plamatrix::Vector3d left(1.0, 2.0, 3.0);
    const plamatrix::Vector3d right(3.0, 2.0, 1.0);
    const auto vector = left + right;
    EXPECT_DOUBLE_EQ(vector.dot(left), 24.0);
    EXPECT_DOUBLE_EQ(vector.sum(), 12.0);
    EXPECT_DOUBLE_EQ(vector.squaredNorm(), 48.0);
    EXPECT_NEAR(vector.norm(), std::sqrt(48.0), 1e-12);
    EXPECT_TRUE(vector.isApprox(plamatrix::Vector3d(4.0, 4.0, 4.0)));
    EXPECT_TRUE(vector.allFinite());
    EXPECT_THROW(static_cast<void>(vector.dot(plamatrix::Matrix2d::Identity())), plamatrix::internal::Error);

    const plamatrix::Matrix2d identity = plamatrix::Matrix2d::Identity();
    const auto doubled = identity + identity;
    EXPECT_DOUBLE_EQ(doubled.trace(), 4.0);
    EXPECT_FALSE(doubled.isApprox(identity));
    plamatrix::Matrix2d nonfinite = identity;
    nonfinite(0, 0) = std::numeric_limits<double>::infinity();
    EXPECT_FALSE((nonfinite + identity).allFinite());
}

TEST(MatrixExpression, ContiguousExpressionReductionsMatchMaterializedResults)
{
    constexpr plamatrix::Index size = 16384;
    plamatrix::VectorXd left(size);
    plamatrix::VectorXd right(size);
    plamatrix::VectorXd weights(size);
    for (plamatrix::Index index = 0; index < size; ++index)
    {
        left(index) = static_cast<double>((index % 17) - 8) * 0.25;
        right(index) = static_cast<double>((index % 13) - 6) * 0.125;
        weights(index) = static_cast<double>((index % 7) + 1) * 0.5;
    }

    const auto expression = 1.25 * left - 0.75 * right;
    const plamatrix::VectorXd materialized = expression;
    EXPECT_NEAR(expression.sum(), materialized.sum(), 1.0e-10);
    EXPECT_NEAR(expression.dot(weights), materialized.dot(weights), 1.0e-10);
    EXPECT_NEAR(expression.squaredNorm(), materialized.squaredNorm(), 1.0e-10);
}

TEST(MatrixExpression, ArrayMathComposesLazilyWithExpressions)
{
    plamatrix::Matrix2d values;
    values << 1.0, 4.0, 9.0, 16.0;
    auto roots = values.array().sqrt();
    static_assert(decltype(roots)::RowsAtCompileTime == 2);
    EXPECT_DOUBLE_EQ(roots(0, 0), 1.0);
    values(0, 0) = 25.0;
    EXPECT_DOUBLE_EQ(roots(0, 0), 5.0);

    const plamatrix::Matrix2d recovered = roots.square();
    EXPECT_TRUE(recovered.isApprox(values));
    const auto logarithms = values.array().log();
    const plamatrix::Matrix2d exponentials = values.array().log().exp();
    EXPECT_TRUE(exponentials.isApprox(values, 1e-12));
    EXPECT_TRUE(logarithms.exp().isApprox(values, 1e-12));
    const plamatrix::Matrix2d negatives = -1.0 * values;
    const plamatrix::Matrix2d magnitudes = negatives.array().abs();
    EXPECT_TRUE(magnitudes.isApprox(values));

    plamatrix::Matrix<int, 2, 1> signed_values(-3, 4);
    const plamatrix::Matrix<int, 2, 1> absolute = signed_values.array().abs();
    EXPECT_EQ(absolute(0), 3);
    EXPECT_EQ(absolute(1), 4);
    signed_values(0) = std::numeric_limits<int>::min();
    EXPECT_THROW(static_cast<void>(signed_values.array().abs().eval()), plamatrix::internal::Error);
}

TEST(MatrixExpression, GpuPreferredKeepsComposedResultsResident)
{
#ifdef PLAMATRIX_WITH_CUDA
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
    {
        GTEST_SKIP() << "CUDA device unavailable";
    }
    plamatrix::MatrixXf left = plamatrix::MatrixXf::Ones(16, 16);
    plamatrix::MatrixXf right = plamatrix::MatrixXf::Identity(16, 16);
    plamatrix::internal::ScopedExecutionPolicy preferred(plamatrix::internal::ExecutionPolicy::GpuPreferred);
    plamatrix::MatrixXf result = left * right + left;
    EXPECT_EQ(plamatrix::internal::MatrixAccess::executionInfo(result).backend, plamatrix::internal::Backend::Cuda);
    EXPECT_TRUE(plamatrix::internal::MatrixAccess::isDeviceResident(result));
    plamatrix::MatrixXf squared = result.array().square();
    EXPECT_EQ(plamatrix::internal::MatrixAccess::executionInfo(squared).backend, plamatrix::internal::Backend::Cpu);
    EXPECT_GT(plamatrix::internal::MatrixAccess::executionInfo(squared).bytesDownloaded, 0U);
    EXPECT_FLOAT_EQ(squared(0, 0), 4.0f);
    const plamatrix::MatrixXf& host_result = result;
    EXPECT_FLOAT_EQ(host_result(0, 0), 2.0f);
    EXPECT_FLOAT_EQ(host_result(15, 15), 2.0f);
#else
    GTEST_SKIP() << "PlaMatrix was built without CUDA";
#endif
}

TEST(MatrixExpression, NoAliasUsesCpuForSmallAutomaticProduct)
{
    plamatrix::Matrix2f left = plamatrix::Matrix2f::Ones();
    plamatrix::Matrix2f right = plamatrix::Matrix2f::Identity();
    plamatrix::Matrix2f result;
    result.noalias() = left * right;
    EXPECT_EQ(plamatrix::internal::MatrixAccess::executionInfo(result).backend, plamatrix::internal::Backend::Cpu);
    EXPECT_TRUE(result.isApprox(left));
}

TEST(MatrixExpression, NoAliasRequiredDeviceRejectsUnsupportedArrayOperation)
{
    plamatrix::Matrix2f input = plamatrix::Matrix2f::Ones();
    plamatrix::Matrix2f result;
    plamatrix::internal::ScopedExecutionPolicy required(plamatrix::internal::ExecutionPolicy::GpuRequired);
    EXPECT_THROW(result.noalias() = input.array().square(), plamatrix::internal::Error);
}

TEST(MatrixExpression, NoAliasKeepsCudaResultsResidentAcrossStaticAndDynamicShapes)
{
#ifdef PLAMATRIX_WITH_CUDA
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
    {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    plamatrix::MatrixXf dynamic_left = plamatrix::MatrixXf::Ones(2, 2);
    plamatrix::MatrixXf dynamic_right = plamatrix::MatrixXf::Identity(2, 2);
    plamatrix::Matrix2f fixed_result;
    plamatrix::MatrixXf dynamic_result;
    {
        plamatrix::internal::ScopedExecutionPolicy required(plamatrix::internal::ExecutionPolicy::GpuRequired);
        fixed_result.noalias() = dynamic_left * dynamic_right;
        dynamic_result.noalias() = plamatrix::Matrix2f::Ones() * plamatrix::Matrix2f::Identity();
    }
    EXPECT_EQ(plamatrix::internal::MatrixAccess::executionInfo(fixed_result).backend,
              plamatrix::internal::Backend::Cuda);
    EXPECT_EQ(plamatrix::internal::MatrixAccess::executionInfo(dynamic_result).backend,
              plamatrix::internal::Backend::Cuda);
    EXPECT_TRUE(plamatrix::internal::MatrixAccess::isDeviceResident(fixed_result));
    EXPECT_TRUE(plamatrix::internal::MatrixAccess::isDeviceResident(dynamic_result));
    EXPECT_EQ(plamatrix::internal::MatrixAccess::executionInfo(fixed_result).bytesDownloaded, 0U);
    EXPECT_EQ(plamatrix::internal::MatrixAccess::executionInfo(dynamic_result).bytesDownloaded, 0U);
    EXPECT_TRUE(fixed_result.isApprox(plamatrix::Matrix2f::Ones()));
    EXPECT_TRUE(dynamic_result.isApprox(plamatrix::MatrixXf::Ones(2, 2)));

    plamatrix::internal::ScopedExecutionPolicy preferred(plamatrix::internal::ExecutionPolicy::GpuPreferred);
    plamatrix::MatrixXf composed;
    composed.noalias() = fixed_result * dynamic_result + dynamic_result;
    EXPECT_EQ(plamatrix::internal::MatrixAccess::executionInfo(composed).backend, plamatrix::internal::Backend::Cuda);
    EXPECT_TRUE(plamatrix::internal::MatrixAccess::isDeviceResident(composed));
    EXPECT_FLOAT_EQ(static_cast<const plamatrix::MatrixXf&>(composed)(0, 0), 3.0f);

    plamatrix::MatrixXf large_left = plamatrix::MatrixXf::Ones(512, 512);
    plamatrix::MatrixXf large_right = plamatrix::MatrixXf::Identity(512, 512);
    plamatrix::MatrixXf automatic_result;
    {
        plamatrix::internal::ScopedExecutionPolicy automatic(plamatrix::internal::ExecutionPolicy::Auto);
        automatic_result.noalias() = large_left * large_right;
    }
    EXPECT_EQ(plamatrix::internal::MatrixAccess::executionInfo(automatic_result).backend,
              plamatrix::internal::Backend::Cuda);
    EXPECT_TRUE(plamatrix::internal::MatrixAccess::isDeviceResident(automatic_result));
    EXPECT_FLOAT_EQ(static_cast<const plamatrix::MatrixXf&>(automatic_result)(0, 0), 1.0f);
#else
    GTEST_SKIP() << "PlaMatrix was built without CUDA";
#endif
}
