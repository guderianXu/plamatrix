#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>

#include "plamatrix/dense/matrix.h"

TEST(EigenStyleExtended, CommaInitializerAcceptsConvertibleScalarsAlongsideBlocks)
{
    plamatrix::Matrix2d matrix;
    matrix << 1, 2.0F, 3L, 4.0;
    EXPECT_DOUBLE_EQ(matrix(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(matrix(1, 1), 4.0);
    plamatrix::Matrix<plamatrix::Index, plamatrix::Dynamic, plamatrix::Dynamic> indices(3, 1);
    indices << 2, 0, -1;
    EXPECT_EQ(indices(2, 0), -1);
    plamatrix::MatrixXd blocks(1, 4);
    const plamatrix::Matrix<double, 1, 2> pair(2.0, 3.0);
    blocks << 1, pair, 4.0F;
    EXPECT_DOUBLE_EQ(blocks.sum(), 10.0);
}

TEST(EigenStyleExtended, LdltSolvesSymmetricPositiveDefiniteSystem)
{
    plamatrix::Matrix3d coefficients;
    coefficients << 4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0;
    const plamatrix::Vector3d expected(1.0, 2.0, -1.0);
    const auto factor = coefficients.ldlt();
    const auto solved = factor.solve(coefficients * expected);
    for (plamatrix::Index row = 0; row < 3; ++row)
    {
        EXPECT_NEAR(solved(row), expected(row), 1e-12);
    }
    const auto zero_factor = plamatrix::Matrix3d::Zero().ldlt();
    EXPECT_TRUE(zero_factor.isPositive());
    EXPECT_TRUE(zero_factor.isNegative());
    EXPECT_TRUE(zero_factor.solve(plamatrix::Vector3d::Zero()).isApprox(plamatrix::Vector3d::Zero()));
    plamatrix::Matrix3d nonsymmetric = coefficients;
    nonsymmetric(0, 1) = 9.0;
    EXPECT_TRUE(nonsymmetric.ldlt().solve(coefficients * expected).isApprox(expected));
    nonsymmetric(0, 1) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_TRUE(nonsymmetric.ldlt().solve(coefficients * expected).isApprox(expected));
    nonsymmetric(1, 0) = 9.0;
    nonsymmetric(0, 1) = 1.0;
    const plamatrix::LDLT<plamatrix::Matrix3d, plamatrix::Upper> upper_factor(nonsymmetric);
    EXPECT_TRUE(upper_factor.solve(coefficients * expected).isApprox(expected));
}

TEST(EigenStyleExtended, CoefficientAndStrideAccessReflectColumnMajorStorage)
{
    plamatrix::MatrixXd matrix(2, 3);
    matrix << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
    EXPECT_EQ(matrix.innerSize(), 2);
    EXPECT_EQ(matrix.outerSize(), 3);
    EXPECT_EQ(matrix.innerStride(), 1);
    EXPECT_EQ(matrix.outerStride(), 2);
    EXPECT_DOUBLE_EQ(matrix.coeff(1, 2), 6.0);
    EXPECT_DOUBLE_EQ(matrix.data()[5], 6.0);
    matrix.coeffRef(1, 2) = 9.0;
    EXPECT_DOUBLE_EQ(matrix(1, 2), 9.0);

    const plamatrix::MatrixXd& constant = matrix;
    EXPECT_DOUBLE_EQ(constant.coeff(1, 2), 9.0);

    plamatrix::Vector3d vector(1.0, 2.0, 3.0);
    EXPECT_DOUBLE_EQ(vector.coeff(1), 2.0);
    vector.coeffRef(1) = 4.0;
    EXPECT_DOUBLE_EQ(vector[1], 4.0);
    const plamatrix::Vector3d& constant_vector = vector;
    EXPECT_DOUBLE_EQ(constant_vector[1], 4.0);
    EXPECT_THROW(matrix[0], std::out_of_range);

    plamatrix::RowVectorXd row_vector(3);
    row_vector << 1.0, 2.0, 3.0;
    EXPECT_EQ(row_vector.innerSize(), 3);
    EXPECT_EQ(row_vector.outerSize(), 1);
    EXPECT_EQ(row_vector.innerStride(), 1);
    EXPECT_EQ(row_vector.outerStride(), 3);
    EXPECT_DOUBLE_EQ(row_vector.data()[2], row_vector(0, 2));
}

TEST(EigenStyleExtended, LdltPivotsAndSolvesSemidefiniteSystems)
{
    plamatrix::Matrix3d coefficients;
    coefficients << 0.0, 0.0, 0.0, 0.0, 4.0, 1.0, 0.0, 1.0, 1.0;
    const auto factor = coefficients.ldlt();
    EXPECT_TRUE(factor.isPositive());
    EXPECT_FALSE(factor.isNegative());
    EXPECT_DOUBLE_EQ(factor.vectorD()(0), 4.0);
    plamatrix::Vector3d right(0.0, 9.0, 3.0);
    const plamatrix::Vector3d solved = factor.solve(right);
    EXPECT_TRUE((coefficients * solved).isApprox(right));
    EXPECT_DOUBLE_EQ(solved(0), 0.0);

    plamatrix::Matrix2d negative;
    negative << -4.0, 0.0, 0.0, 0.0;
    const auto negative_factor = negative.ldlt();
    EXPECT_FALSE(negative_factor.isPositive());
    EXPECT_TRUE(negative_factor.isNegative());
    EXPECT_TRUE(negative_factor.solve(plamatrix::Vector2d(-8.0, 0.0)).isApprox(plamatrix::Vector2d(2.0, 0.0)));

    plamatrix::Matrix2d rank_deficient;
    rank_deficient << 1.0, 1.0, 1.0, 1.0;
    const auto rank_factor = rank_deficient.ldlt();
    const plamatrix::Vector2d rank_rhs(2.0, 2.0);
    EXPECT_TRUE((rank_deficient * rank_factor.solve(rank_rhs)).isApprox(rank_rhs));
    EXPECT_THROW(rank_factor.solve(plamatrix::Vector2d(2.0, 3.0)), plamatrix::internal::Error);

    plamatrix::Matrix2d indefinite;
    indefinite << 0.0, 1.0, 1.0, 0.0;
    EXPECT_THROW(indefinite.ldlt(), plamatrix::internal::Error);
}

TEST(EigenStyleExtended, PublicFactorizationTypesUseEigenTemplateSignatures)
{
    static_assert(std::is_same_v<plamatrix::Matrix3d::Scalar, double>);
    static_assert(std::is_same_v<plamatrix::Matrix3d::RealScalar, double>);
    static_assert(std::is_same_v<plamatrix::Matrix3d::PlainObject, plamatrix::Matrix3d>);
    static_assert(std::is_same_v<decltype(std::declval<const plamatrix::Matrix3d&>().partialPivLu()),
                                 plamatrix::PartialPivLU<plamatrix::Matrix3d>>);
    static_assert(std::is_same_v<decltype(std::declval<const plamatrix::Matrix3d&>().lu()),
                                 plamatrix::PartialPivLU<plamatrix::Matrix3d>>);
    static_assert(std::is_same_v<decltype(std::declval<const plamatrix::Matrix3d&>().partialPivLu<plamatrix::Index>()),
                                 plamatrix::PartialPivLU<plamatrix::Matrix3d, plamatrix::Index>>);
    static_assert(std::is_same_v<decltype(std::declval<const plamatrix::Matrix3d&>().ldlt()),
                                 plamatrix::LDLT<plamatrix::Matrix3d>>);
    static_assert(std::is_same_v<decltype(std::declval<const plamatrix::MatrixXd&>().colPivHouseholderQr()),
                                 plamatrix::ColPivHouseholderQR<plamatrix::MatrixXd>>);
    static_assert(
        std::is_same_v<decltype(std::declval<const plamatrix::MatrixXd&>().colPivHouseholderQr<plamatrix::Index>()),
                       plamatrix::ColPivHouseholderQR<plamatrix::MatrixXd, plamatrix::Index>>);
    static_assert(std::is_same_v<decltype(std::declval<const plamatrix::MatrixXd&>().jacobiSvd()),
                                 plamatrix::JacobiSVD<plamatrix::MatrixXd>>);
    static_assert(
        std::is_same_v<decltype(std::declval<const plamatrix::MatrixXd&>()
                                    .jacobiSvd<plamatrix::ComputeThinU | plamatrix::ComputeThinV>()),
                       plamatrix::JacobiSVD<plamatrix::MatrixXd, plamatrix::ComputeThinU | plamatrix::ComputeThinV>>);

    plamatrix::Matrix2d coefficients;
    coefficients << 3.0, 1.0, 1.0, 2.0;
    const plamatrix::Vector2d right(4.0, 3.0);
    const plamatrix::PartialPivLU<plamatrix::Matrix2d> lu(coefficients);
    const plamatrix::LDLT<plamatrix::Matrix2d> ldlt(coefficients);
    const plamatrix::ColPivHouseholderQR<plamatrix::Matrix2d> qr(coefficients);
    const plamatrix::JacobiSVD<plamatrix::Matrix2d> svd(coefficients);
    const auto thin_svd = coefficients.jacobiSvd<plamatrix::ComputeThinU | plamatrix::ComputeThinV>();
    const auto wide_index_lu = coefficients.lu<plamatrix::Index>();
    const auto wide_index_qr = coefficients.colPivHouseholderQr<plamatrix::Index>();
    const plamatrix::Vector2d expected(1.0, 1.0);
    EXPECT_TRUE(lu.solve(right).isApprox(expected));
    EXPECT_TRUE(ldlt.solve(right).isApprox(expected));
    EXPECT_TRUE(qr.solve(right).isApprox(expected));
    EXPECT_TRUE(svd.solve(right).isApprox(expected));
    EXPECT_TRUE(thin_svd.solve(right).isApprox(expected));
    EXPECT_TRUE(wide_index_lu.solve(right).isApprox(expected));
    EXPECT_TRUE(wide_index_qr.solve(right).isApprox(expected));
}

TEST(EigenStyleExtended, ColumnPivotedQrSolvesTallLeastSquares)
{
    plamatrix::MatrixXd design(4, 2);
    design << 1.0, 10.0, 2.0, 0.0, 3.0, 1.0, 4.0, 2.0;
    plamatrix::VectorXd rhs(4);
    rhs << 31.0, 2.0, 6.0, 10.0;
    const auto solved = design.colPivHouseholderQr().solve(rhs);
    EXPECT_NEAR(solved(0), 1.0, 1e-12);
    EXPECT_NEAR(solved(1), 3.0, 1e-12);
    EXPECT_EQ(plamatrix::MatrixXd::Zero(2, 3).colPivHouseholderQr().rank(), 0);
    EXPECT_EQ(plamatrix::MatrixXd::Zero(3, 2).colPivHouseholderQr().rank(), 0);
}

TEST(EigenStyleExtended, FloatSolversHandleMultipleRightHandSides)
{
    plamatrix::Matrix2f coefficients;
    coefficients << 3.0f, 1.0f, 1.0f, 2.0f;
    const plamatrix::Matrix2f expected = plamatrix::Matrix2f::Identity();
    const auto right = coefficients * expected;
    EXPECT_TRUE(coefficients.ldlt().solve(right).isApprox(expected, 1e-5f));
    EXPECT_TRUE(coefficients.colPivHouseholderQr().solve(right).isApprox(expected, 1e-5f));
    EXPECT_TRUE(coefficients.jacobiSvd().solve(right).isApprox(expected, 1e-5f));

    plamatrix::internal::ScopedExecutionPolicy required(plamatrix::internal::ExecutionPolicy::GpuRequired,
                                                        plamatrix::internal::Backend::OpenCl);
    EXPECT_THROW(coefficients.ldlt(), plamatrix::internal::Error);
    EXPECT_THROW(coefficients.colPivHouseholderQr(), plamatrix::internal::Error);
    if (plamatrix::internal::detail::GpuOps<float>::available(plamatrix::internal::Backend::OpenCl))
    {
        const auto svd = coefficients.jacobiSvd();
        EXPECT_EQ(svd.backend(), plamatrix::internal::Backend::OpenCl);
        EXPECT_TRUE(svd.solve(right).isApprox(expected, 1e-5f));
    }
    else
    {
        EXPECT_THROW(coefficients.jacobiSvd(), plamatrix::internal::Error);
    }
}

TEST(EigenStyleExtended, JacobiSvdReturnsMinimumNormForRankDeficientAndWideSystems)
{
    plamatrix::MatrixXd rank_deficient(3, 2);
    rank_deficient << 1.0, 2.0, 2.0, 4.0, 3.0, 6.0;
    const plamatrix::VectorXd rhs = []
    {
        plamatrix::VectorXd values(3);
        values << 5.0, 10.0, 15.0;
        return values;
    }();
    const auto solved = rank_deficient.jacobiSvd().solve(rhs);
    EXPECT_NEAR(solved(0), 1.0, 1e-9);
    EXPECT_NEAR(solved(1), 2.0, 1e-9);
    const auto rank_svd = rank_deficient.jacobiSvd();
    const auto singular = rank_svd.singularValues();
    EXPECT_GE(singular(0), singular(1));
    EXPECT_NEAR(singular(1), 0.0, 1e-10);

    plamatrix::MatrixXd regular(2, 2);
    regular << 3.0, 1.0, 0.0, 2.0;
    const auto regular_svd = regular.jacobiSvd();
    const auto left = regular_svd.matrixU();
    const auto right = regular_svd.matrixV();
    const auto values = regular_svd.singularValues();
    plamatrix::MatrixXd reconstructed = plamatrix::MatrixXd::Zero(2, 2);
    for (plamatrix::Index component = 0; component < 2; ++component)
    {
        for (plamatrix::Index row = 0; row < 2; ++row)
        {
            for (plamatrix::Index col = 0; col < 2; ++col)
            {
                reconstructed(row, col) += left(row, component) * values(component) * right(col, component);
            }
        }
    }
    EXPECT_TRUE(reconstructed.isApprox(regular, 1e-10));

    plamatrix::MatrixXd wide(2, 3);
    wide << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0;
    const auto wide_result = wide.jacobiSvd().solve(plamatrix::Vector2d(2.0, 3.0));
    EXPECT_NEAR(wide_result(0), 2.0, 1e-12);
    EXPECT_NEAR(wide_result(1), 3.0, 1e-12);
    EXPECT_NEAR(wide_result(2), 0.0, 1e-12);
}

TEST(EigenStyleExtended, FixedBlockViewsComposeWithMatrixOperations)
{
    plamatrix::MatrixXd source = plamatrix::MatrixXd::Zero(4, 4);
    source.block<2, 2>(1, 1) = plamatrix::Matrix2d::Identity();
    const auto block = source.block<2, 2>(1, 1);
    static_assert(decltype(block)::RowsAtCompileTime == 2);
    static_assert(decltype(block)::ColsAtCompileTime == 2);
    const auto product = block * plamatrix::Matrix2d::Identity();
    static_assert(decltype(product)::RowsAtCompileTime == 2);
    EXPECT_DOUBLE_EQ(product(1, 1), 1.0);

    const auto gram = source.transpose() * source;
    EXPECT_DOUBLE_EQ(gram(2, 2), 1.0);
    const plamatrix::Matrix2d copied(block);
    EXPECT_DOUBLE_EQ(copied(0, 0), 1.0);

    plamatrix::MatrixXd target = plamatrix::MatrixXd::Zero(4, 4);
    auto destination = target.block<2, 2>(0, 0);
    destination = block;
    EXPECT_DOUBLE_EQ(target(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(target(1, 1), 1.0);
}

TEST(EigenStyleExtended, ReductionsArrayAndBroadcastAreAvailableOnMatrix)
{
    plamatrix::MatrixXd values(2, 3);
    values << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
    EXPECT_DOUBLE_EQ(values.sum(), 21.0);
    EXPECT_DOUBLE_EQ(values.mean(), 3.5);
    EXPECT_DOUBLE_EQ(values.minCoeff(), 1.0);
    EXPECT_DOUBLE_EQ(values.maxCoeff(), 6.0);
    EXPECT_DOUBLE_EQ(values.trace(), 6.0);
    EXPECT_DOUBLE_EQ(values.rowwise().sum()(0), 6.0);
    EXPECT_DOUBLE_EQ(values.colwise().mean()(0, 2), 4.5);
    EXPECT_DOUBLE_EQ(values.array().square()(1, 2), 36.0);

    const auto columns = values.colwise() + plamatrix::Vector2d(10.0, 20.0);
    EXPECT_DOUBLE_EQ(columns(1, 2), 26.0);
    plamatrix::RowVectorXd offsets(3);
    offsets << 1.0, 2.0, 3.0;
    const auto rows = values.rowwise() + offsets;
    EXPECT_DOUBLE_EQ(rows(1, 2), 9.0);
    EXPECT_TRUE(std::isnan(plamatrix::MatrixXd().mean()));
}

TEST(EigenStyleExtended, DeferredNoAliasProductAndConveniences)
{
    plamatrix::Matrix3d coefficients;
    coefficients << 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0;
    plamatrix::Matrix3d result;
    result.noalias() = coefficients.transpose() * coefficients;
    EXPECT_DOUBLE_EQ(result(0, 0), 4.0);
    EXPECT_DOUBLE_EQ(result(2, 2), 16.0);
    EXPECT_EQ(plamatrix::internal::MatrixAccess::executionInfo(result).reason, "CPU noalias expression assignment");

    plamatrix::MatrixXd resized = plamatrix::MatrixXd::Ones(2, 2);
    resized(0, 1) = 7.0;
    resized.conservativeResize(3, 4);
    EXPECT_DOUBLE_EQ(resized(0, 1), 7.0);
    EXPECT_DOUBLE_EQ(resized(2, 3), 0.0);
    const auto converted = resized.cast<float>();
    EXPECT_FLOAT_EQ(converted(0, 1), 7.0f);
    EXPECT_TRUE(resized.isApprox(plamatrix::MatrixXd(resized)));
    EXPECT_FALSE(resized.isApprox(plamatrix::MatrixXd::Zero(3, 4)));

    const auto random = plamatrix::Matrix3d::Random();
    EXPECT_GE(random.minCoeff(), -1.0);
    EXPECT_LE(random.maxCoeff(), 1.0);

    plamatrix::Matrix2d tiled;
    tiled << plamatrix::Matrix<double, 2, 1>(1.0, 2.0), plamatrix::Matrix<double, 2, 1>(3.0, 4.0);
    EXPECT_DOUBLE_EQ(tiled(1, 0), 2.0);
    EXPECT_DOUBLE_EQ(tiled(0, 1), 3.0);
}
