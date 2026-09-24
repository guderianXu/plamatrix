#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <type_traits>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <Eigen/QR>
#include <Eigen/Sparse>

#include <plamatrix/plamatrix.h>
#include <plamatrix/internal/backend.h>

namespace plamatrix::internal
{

    namespace
    {

        template <typename Scalar>
        using EigenMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

        template <typename Scalar>
        Eigen::Map<const EigenMatrix<Scalar>> mapEigen(const DenseStorage<Scalar, Device::CPU>& matrix)
        {
            return Eigen::Map<const EigenMatrix<Scalar>>(matrix.data(), matrix.rows(), matrix.cols());
        }

        template <typename Scalar>
        DenseStorage<Scalar, Device::CPU> deterministicMatrix(Index rows, Index cols, Scalar diagonal = Scalar(0))
        {
            DenseStorage<Scalar, Device::CPU> matrix(rows, cols);
            for (Index col = 0; col < cols; ++col)
            {
                for (Index row = 0; row < rows; ++row)
                {
                    const int seed = static_cast<int>((row * 17 + col * 29 + 11) % 37) - 18;
                    matrix(row, col) = Scalar(seed) / Scalar(13);
                }
            }
            for (Index index = 0; index < std::min(rows, cols); ++index)
            {
                matrix(index, index) += diagonal;
            }
            return matrix;
        }

        template <typename Scalar> Scalar tolerance()
        {
            return std::is_same_v<Scalar, float> ? Scalar(3e-4) : Scalar(2e-10);
        }

        struct ReferenceShift
        {
            double operator()(double value) const
            {
                return value + 0.25;
            }
        };

        struct ReferenceBlend
        {
            double operator()(double left, double right) const
            {
                return left - 2.0 * right;
            }
        };

        template <typename Scalar> class EigenReferenceTest : public ::testing::Test
        {
        };

#if defined(PLAMATRIX_USE_FLOAT) && defined(PLAMATRIX_USE_DOUBLE)
        using ReferenceScalars = ::testing::Types<float, double>;
#elif defined(PLAMATRIX_USE_FLOAT)
        using ReferenceScalars = ::testing::Types<float>;
#elif defined(PLAMATRIX_USE_DOUBLE)
        using ReferenceScalars = ::testing::Types<double>;
#else
#error "Eigen reference tests require at least one PlaMatrix scalar type"
#endif
        TYPED_TEST_SUITE(EigenReferenceTest, ReferenceScalars);

        TYPED_TEST(EigenReferenceTest, GemmMatchesEigenAcrossRectangularShapes)
        {
            using Scalar = TypeParam;
            for (const auto shape :
                 {std::array<Index, 3>{1, 1, 1}, std::array<Index, 3>{7, 5, 9}, std::array<Index, 3>{19, 23, 11}})
            {
                auto left = deterministicMatrix<Scalar>(shape[0], shape[1]);
                auto right = deterministicMatrix<Scalar>(shape[1], shape[2]);
                auto actual = gemm(left, right);
                EigenMatrix<Scalar> expected = (mapEigen(left) * mapEigen(right)).eval();

                for (Index col = 0; col < actual.cols(); ++col)
                {
                    for (Index row = 0; row < actual.rows(); ++row)
                    {
                        EXPECT_NEAR(actual(row, col), expected(row, col), tolerance<Scalar>() * Scalar(shape[1]));
                    }
                }
            }
        }

        TYPED_TEST(EigenReferenceTest, AxpbyMatchesFusedEigenExpressionAcrossRectangularShapes)
        {
            using Scalar = TypeParam;
            constexpr Scalar alpha = Scalar(1.25);
            constexpr Scalar beta = Scalar(-0.75);
            for (const auto shape :
                 {std::array<Index, 2>{1, 1}, std::array<Index, 2>{7, 5}, std::array<Index, 2>{19, 23}})
            {
                auto lhs = deterministicMatrix<Scalar>(shape[0], shape[1]);
                auto rhs = deterministicMatrix<Scalar>(shape[0], shape[1], Scalar(0.5));
                auto actual = axpby(alpha, lhs, beta, rhs);
                EigenMatrix<Scalar> expected = (alpha * mapEigen(lhs) + beta * mapEigen(rhs)).eval();

                for (Index col = 0; col < actual.cols(); ++col)
                {
                    for (Index row = 0; row < actual.rows(); ++row)
                    {
                        EXPECT_NEAR(actual(row, col), expected(row, col), tolerance<Scalar>());
                    }
                }
            }
        }

        TYPED_TEST(EigenReferenceTest, ThreeTermCombinationMatchesFusedEigenExpression)
        {
            using Scalar = TypeParam;
            constexpr Scalar alpha = Scalar(1.25);
            constexpr Scalar beta = Scalar(-0.75);
            constexpr Scalar gamma = Scalar(0.5);
            for (const auto shape :
                 {std::array<Index, 2>{1, 1}, std::array<Index, 2>{7, 5}, std::array<Index, 2>{19, 23}})
            {
                auto first = deterministicMatrix<Scalar>(shape[0], shape[1]);
                auto second = deterministicMatrix<Scalar>(shape[0], shape[1], Scalar(0.5));
                auto third = deterministicMatrix<Scalar>(shape[0], shape[1], Scalar(-0.25));
                auto actual = linearCombination(alpha, first, beta, second, gamma, third);
                EigenMatrix<Scalar> expected =
                    (alpha * mapEigen(first) + beta * mapEigen(second) + gamma * mapEigen(third)).eval();

                for (Index index = 0; index < actual.size(); ++index)
                {
                    EXPECT_NEAR(actual.data()[index], expected.data()[index], tolerance<Scalar>());
                }
            }
        }

        TYPED_TEST(EigenReferenceTest, DotAndSquaredNormMatchEigenWithoutIntermediates)
        {
            using Scalar = TypeParam;
            auto lhs = deterministicMatrix<Scalar>(37, 29);
            auto rhs = deterministicMatrix<Scalar>(37, 29, Scalar(0.5));
            const Scalar scale = Scalar(lhs.size());

            const Scalar eigen_dot = mapEigen(lhs).cwiseProduct(mapEigen(rhs)).sum();
            EXPECT_NEAR(dot(lhs, rhs), eigen_dot, tolerance<Scalar>() * scale);
            EXPECT_NEAR(squaredNorm(lhs), mapEigen(lhs).squaredNorm(), tolerance<Scalar>() * scale);
        }

        TYPED_TEST(EigenReferenceTest, PartialPivotSolveMatchesEigenAndResidual)
        {
            using Scalar = TypeParam;
            auto coefficients = deterministicMatrix<Scalar>(12, 12, Scalar(20));
            auto rhs = deterministicMatrix<Scalar>(12, 3);
            auto actual = solve<Scalar, Device::CPU>(coefficients, rhs);
            EigenMatrix<Scalar> expected = mapEigen(coefficients).partialPivLu().solve(mapEigen(rhs));

            const Scalar relative_error = (mapEigen(actual) - expected).norm() / std::max(Scalar(1), expected.norm());
            const Scalar relative_residual = (mapEigen(coefficients) * mapEigen(actual) - mapEigen(rhs)).norm() /
                                             std::max(Scalar(1), mapEigen(rhs).norm());
            EXPECT_LT(relative_error, tolerance<Scalar>());
            EXPECT_LT(relative_residual, tolerance<Scalar>());
        }

        TYPED_TEST(EigenReferenceTest, DecompositionsMatchEigenSpectrum)
        {
            using Scalar = TypeParam;
            auto input = deterministicMatrix<Scalar>(8, 5);
            auto [u, singular_values, vt] = svd(input);
            Eigen::JacobiSVD<EigenMatrix<Scalar>> eigen_svd(mapEigen(input), Eigen::ComputeFullU | Eigen::ComputeFullV);
            ASSERT_EQ(singular_values.rows(), eigen_svd.singularValues().size());
            for (Index index = 0; index < singular_values.rows(); ++index)
            {
                EXPECT_NEAR(
                    singular_values(index, 0), eigen_svd.singularValues()(index), tolerance<Scalar>() * Scalar(20));
            }

            auto symmetric_seed = deterministicMatrix<Scalar>(6, 6);
            auto symmetric = gemm(symmetric_seed.transpose(), symmetric_seed);
            auto actual_eigenvalues = eigh(symmetric);
            Eigen::SelfAdjointEigenSolver<EigenMatrix<Scalar>> eigen_solver(mapEigen(symmetric));
            ASSERT_EQ(eigen_solver.info(), Eigen::Success);
            for (Index index = 0; index < actual_eigenvalues.rows(); ++index)
            {
                const Index reversed = actual_eigenvalues.rows() - index - 1;
                EXPECT_NEAR(actual_eigenvalues(index, 0),
                            eigen_solver.eigenvalues()(reversed),
                            tolerance<Scalar>() * Scalar(50));
            }
        }

        TYPED_TEST(EigenReferenceTest, SparseSpmvMatchesEigenWithDuplicateTriplets)
        {
            using Scalar = TypeParam;
            constexpr Index rows = 17;
            std::vector<Index> row_indices;
            std::vector<Index> col_indices;
            std::vector<Scalar> values;
            std::vector<Eigen::Triplet<Scalar, Index>> triplets;
            for (Index row = 0; row < rows; ++row)
            {
                const auto append = [&](Index col, Scalar value)
                {
                    row_indices.push_back(row);
                    col_indices.push_back(col);
                    values.push_back(value);
                    triplets.emplace_back(row, col, value);
                };
                append(row, Scalar(4));
                append(row, Scalar(0.25));
                if (row != 0)
                {
                    append(row - 1, Scalar(-1));
                }
                if (row + 1 != rows)
                {
                    append(row + 1, Scalar(-1));
                }
            }

            auto sparse = cooToCsr(rows, rows, row_indices, col_indices, values);
            auto input = deterministicMatrix<Scalar>(rows, 1);
            auto actual = spmv(sparse, input);

            Eigen::SparseMatrix<Scalar, Eigen::RowMajor, Index> eigen_sparse(rows, rows);
            eigen_sparse.setFromTriplets(triplets.begin(), triplets.end());
            EigenMatrix<Scalar> expected = eigen_sparse * mapEigen(input);
            for (Index row = 0; row < rows; ++row)
            {
                EXPECT_NEAR(actual(row, 0), expected(row, 0), tolerance<Scalar>());
            }
        }

        TEST(EigenReference, MatrixViewsMatchEigenMapsWithoutCopies)
        {
            std::vector<double> row_major(5 * 7);
            for (std::size_t index = 0; index < row_major.size(); ++index)
            {
                row_major[index] = static_cast<double>(index);
            }
            auto view = makeRowMajorView<double, Device::CPU>(row_major.data(), 5, 7);
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eigen(
                row_major.data(), 5, 7);

            view.block(1, 2, 3, 4)(2, 3) = -9.0;
            EXPECT_DOUBLE_EQ(eigen(3, 5), -9.0);
            eigen(2, 4) = 123.0;
            EXPECT_DOUBLE_EQ(view(2, 4), 123.0);
        }

        TEST(EigenReference, StructuredDenseExpressionsMatchEigen)
        {
            plamatrix::MatrixXd actual(2, 3);
            actual << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
            Eigen::MatrixXd expected(2, 3);
            expected << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;

            const auto actual_col_reshape = actual.reshaped(3, 2).eval();
            const Eigen::MatrixXd expected_col_reshape = expected.reshaped(3, 2);
            const auto actual_row_reshape = actual.reshaped<plamatrix::RowMajor>(3, 2).eval();
            const Eigen::MatrixXd expected_row_reshape = expected.reshaped<Eigen::RowMajor>(3, 2);
            for (Index col = 0; col < 2; ++col)
            {
                for (Index row = 0; row < 3; ++row)
                {
                    EXPECT_DOUBLE_EQ(actual_col_reshape(row, col), expected_col_reshape(row, col));
                    EXPECT_DOUBLE_EQ(actual_row_reshape(row, col), expected_row_reshape(row, col));
                }
            }

            const auto actual_replicated = actual.replicate<2, 2>().eval();
            const Eigen::MatrixXd expected_replicated = expected.replicate<2, 2>();
            const auto actual_reversed = actual.reverse().eval();
            const Eigen::MatrixXd expected_reversed = expected.reverse();
            for (Index col = 0; col < actual_replicated.cols(); ++col)
            {
                for (Index row = 0; row < actual_replicated.rows(); ++row)
                {
                    EXPECT_DOUBLE_EQ(actual_replicated(row, col), expected_replicated(row, col));
                }
            }
            for (Index col = 0; col < actual.cols(); ++col)
            {
                for (Index row = 0; row < actual.rows(); ++row)
                {
                    EXPECT_DOUBLE_EQ(actual_reversed(row, col), expected_reversed(row, col));
                }
            }

            plamatrix::Matrix3d actual_square;
            actual_square << 2.0, 9.0, 8.0, 1.0, 3.0, 7.0, 4.0, 5.0, 6.0;
            Eigen::Matrix3d expected_square;
            expected_square << 2.0, 9.0, 8.0, 1.0, 3.0, 7.0, 4.0, 5.0, 6.0;
            const plamatrix::Matrix3d actual_lower = actual_square.triangularView<plamatrix::Lower>();
            const Eigen::Matrix3d expected_lower = expected_square.triangularView<Eigen::Lower>();
            const plamatrix::Matrix3d actual_symmetric = actual_square.selfadjointView<plamatrix::Lower>();
            const Eigen::Matrix3d expected_symmetric = expected_square.selfadjointView<Eigen::Lower>();
            for (Index col = 0; col < 3; ++col)
            {
                for (Index row = 0; row < 3; ++row)
                {
                    EXPECT_DOUBLE_EQ(actual_lower(row, col), expected_lower(row, col));
                    EXPECT_DOUBLE_EQ(actual_symmetric(row, col), expected_symmetric(row, col));
                }
            }

            actual.transposeInPlace();
            expected.transposeInPlace();
            ASSERT_EQ(actual.rows(), expected.rows());
            ASSERT_EQ(actual.cols(), expected.cols());
            for (Index col = 0; col < actual.cols(); ++col)
            {
                for (Index row = 0; row < actual.rows(); ++row)
                {
                    EXPECT_DOUBLE_EQ(actual(row, col), expected(row, col));
                }
            }
        }

        TEST(EigenReference, DenseReductionsNormsAndCwiseExpressionsMatchEigen)
        {
            plamatrix::MatrixXd actual(2, 3);
            actual << -4.0, 2.0, 7.0, 3.0, -5.0, 6.0;
            Eigen::MatrixXd expected(2, 3);
            expected << -4.0, 2.0, 7.0, 3.0, -5.0, 6.0;

            plamatrix::Index actual_row = -1;
            plamatrix::Index actual_col = -1;
            Eigen::Index expected_row = -1;
            Eigen::Index expected_col = -1;
            EXPECT_DOUBLE_EQ(actual.minCoeff(&actual_row, &actual_col),
                             expected.minCoeff(&expected_row, &expected_col));
            EXPECT_EQ(actual_row, expected_row);
            EXPECT_EQ(actual_col, expected_col);
            EXPECT_DOUBLE_EQ(actual.maxCoeff(&actual_row, &actual_col),
                             expected.maxCoeff(&expected_row, &expected_col));
            EXPECT_EQ(actual_row, expected_row);
            EXPECT_EQ(actual_col, expected_col);

            EXPECT_DOUBLE_EQ(actual.sum(), expected.sum());
            EXPECT_DOUBLE_EQ(actual.mean(), expected.mean());
            EXPECT_DOUBLE_EQ(actual.prod(), expected.prod());
            EXPECT_EQ(actual.all(), expected.all());
            EXPECT_EQ(actual.any(), expected.any());
            EXPECT_EQ(actual.count(), expected.count());
            EXPECT_NEAR(actual.lpNorm<1>(), expected.lpNorm<1>(), 1.0e-14);
            EXPECT_NEAR(actual.lpNorm<3>(), expected.lpNorm<3>(), 1.0e-14);
            EXPECT_NEAR(actual.lpNorm<plamatrix::Infinity>(), expected.lpNorm<Eigen::Infinity>(), 1.0e-14);
            EXPECT_NEAR(actual.stableNorm(), expected.stableNorm(), 1.0e-14);
            EXPECT_NEAR(actual.blueNorm(), expected.blueNorm(), 1.0e-14);
            EXPECT_NEAR(actual.hypotNorm(), expected.hypotNorm(), 1.0e-14);

            const plamatrix::MatrixXd actual_other = plamatrix::MatrixXd::Constant(2, 3, 1.5);
            const Eigen::MatrixXd expected_other = Eigen::MatrixXd::Constant(2, 3, 1.5);
            const plamatrix::MatrixXd actual_abs = actual.cwiseAbs();
            const Eigen::MatrixXd expected_abs = expected.cwiseAbs();
            const plamatrix::MatrixXd actual_min = actual.cwiseMin(actual_other);
            const Eigen::MatrixXd expected_min = expected.cwiseMin(expected_other);
            const plamatrix::MatrixXd actual_max = actual.cwiseMax(0.5);
            const Eigen::MatrixXd expected_max = expected.cwiseMax(0.5);
            const plamatrix::MatrixXd actual_unary = actual.unaryExpr(ReferenceShift{});
            const Eigen::MatrixXd expected_unary = expected.unaryExpr(ReferenceShift{});
            const plamatrix::MatrixXd actual_binary = actual.binaryExpr(actual_other, ReferenceBlend{});
            const Eigen::MatrixXd expected_binary = expected.binaryExpr(expected_other, ReferenceBlend{});
            for (Index col = 0; col < actual.cols(); ++col)
            {
                for (Index row = 0; row < actual.rows(); ++row)
                {
                    EXPECT_DOUBLE_EQ(actual_abs(row, col), expected_abs(row, col));
                    EXPECT_DOUBLE_EQ(actual_min(row, col), expected_min(row, col));
                    EXPECT_DOUBLE_EQ(actual_max(row, col), expected_max(row, col));
                    EXPECT_DOUBLE_EQ(actual_unary(row, col), expected_unary(row, col));
                    EXPECT_DOUBLE_EQ(actual_binary(row, col), expected_binary(row, col));
                }
            }

            const plamatrix::MatrixXd actual_near = actual * (1.0 + 1.0e-13);
            const Eigen::MatrixXd expected_near = expected * (1.0 + 1.0e-13);
            EXPECT_EQ(actual.isApprox(actual_near), expected.isApprox(expected_near));
            EXPECT_EQ(actual.isMuchSmallerThan(actual_near, 2.0), expected.isMuchSmallerThan(expected_near, 2.0));
            EXPECT_EQ(actual.isMuchSmallerThan(100.0, 0.1), expected.isMuchSmallerThan(100.0, 0.1));
        }

        TEST(EigenReference, GeometryQuaternionTransformAndEulerAnglesMatchEigen)
        {
            const plamatrix::Vector3d axis = plamatrix::Vector3d(1.0, -2.0, 0.5).normalized();
            const Eigen::Vector3d eigen_axis(axis.x(), axis.y(), axis.z());
            const plamatrix::Quaterniond quaternion(plamatrix::AngleAxisd(0.65, axis));
            const Eigen::Quaterniond eigen_quaternion(Eigen::AngleAxisd(0.65, eigen_axis));
            const auto rotation = quaternion.toRotationMatrix();
            const Eigen::Matrix3d eigen_rotation = eigen_quaternion.toRotationMatrix();
            for (Index row = 0; row < 3; ++row)
                for (Index col = 0; col < 3; ++col)
                    EXPECT_NEAR(rotation(row, col), eigen_rotation(row, col), 1.0e-12);

            const auto actual_angles = rotation.canonicalEulerAngles(2, 1, 0);
            const auto eigen_angles = eigen_rotation.canonicalEulerAngles(2, 1, 0);
            for (Index index = 0; index < 3; ++index)
                EXPECT_NEAR(actual_angles(index), eigen_angles(index), 1.0e-12);

            const plamatrix::Quaterniond target(plamatrix::AngleAxisd(-1.1, plamatrix::Vector3d::UnitY()));
            const Eigen::Quaterniond eigen_target(Eigen::AngleAxisd(-1.1, Eigen::Vector3d::UnitY()));
            const auto interpolated = quaternion.slerp(0.35, target);
            const auto eigen_interpolated = eigen_quaternion.slerp(0.35, eigen_target);
            EXPECT_NEAR(
                std::abs(interpolated.dot(plamatrix::Quaterniond(
                    eigen_interpolated.w(), eigen_interpolated.x(), eigen_interpolated.y(), eigen_interpolated.z()))),
                1.0,
                1.0e-12);

            plamatrix::Affine3d transform = plamatrix::Translation3d(2.0, -1.0, 3.0) * quaternion;
            Eigen::Affine3d eigen_transform = Eigen::Translation3d(2.0, -1.0, 3.0) * eigen_quaternion;
            transform.scale(plamatrix::Vector3d(2.0, 3.0, 4.0));
            eigen_transform.scale(Eigen::Vector3d(2.0, 3.0, 4.0));
            for (Index row = 0; row < 4; ++row)
                for (Index col = 0; col < 4; ++col)
                    EXPECT_NEAR(transform.matrix()(row, col), eigen_transform.matrix()(row, col), 1.0e-12);
            const auto extracted_rotation = transform.rotation();
            const Eigen::Matrix3d eigen_extracted_rotation = eigen_transform.rotation();
            for (Index row = 0; row < 3; ++row)
                for (Index col = 0; col < 3; ++col)
                    EXPECT_NEAR(extracted_rotation(row, col), eigen_extracted_rotation(row, col), 1.0e-12);
        }

        TEST(EigenReference, UmeyamaMatchesEigenForSimilarityAndRigidFits)
        {
            plamatrix::Matrix<double, 3, 5> source;
            source << 0.0, 1.0, -2.0, 0.5, 3.0, 0.0, 2.0, 1.0, -1.0, 0.25, 0.0, -1.0, 0.5, 2.0, 1.5;
            Eigen::Matrix<double, 3, 5> eigen_source;
            for (Index row = 0; row < 3; ++row)
                for (Index col = 0; col < 5; ++col)
                    eigen_source(row, col) = source(row, col);

            const plamatrix::Matrix3d rotation =
                plamatrix::AngleAxisd(0.35, plamatrix::Vector3d(1.0, 2.0, -1.0).normalized()).toRotationMatrix();
            const plamatrix::Vector3d translation(2.0, -3.0, 1.5);
            plamatrix::Matrix<double, 3, 5> destination;
            Eigen::Matrix<double, 3, 5> eigen_destination;
            for (Index col = 0; col < 5; ++col)
                for (Index row = 0; row < 3; ++row)
                {
                    double value = translation(row);
                    for (Index inner = 0; inner < 3; ++inner)
                        value += 1.75 * rotation(row, inner) * source(inner, col);
                    destination(row, col) = value;
                    eigen_destination(row, col) = value;
                }

            for (const bool with_scaling : {false, true})
            {
                const auto actual = plamatrix::umeyama(source, destination, with_scaling);
                const Eigen::Matrix4d expected = Eigen::umeyama(eigen_source, eigen_destination, with_scaling);
                for (Index row = 0; row < 4; ++row)
                    for (Index col = 0; col < 4; ++col)
                        EXPECT_NEAR(actual(row, col), expected(row, col), 2.0e-10);
            }
        }

        TEST(EigenReference, CompressedSparseStorageAndMapMatchEigenLayout)
        {
            plamatrix::SparseMatrix<double> actual(3, 3);
            Eigen::SparseMatrix<double> expected(3, 3);
            actual.startVec(0);
            actual.insertBack(0, 0) = 2.0;
            actual.insertBack(2, 0) = 3.0;
            actual.startVec(1);
            actual.insertBack(1, 1) = 4.0;
            actual.startVec(2);
            actual.insertBack(0, 2) = 5.0;
            actual.insertBack(2, 2) = 6.0;
            actual.finalize();
            expected.insert(0, 0) = 2.0;
            expected.insert(2, 0) = 3.0;
            expected.insert(1, 1) = 4.0;
            expected.insert(0, 2) = 5.0;
            expected.insert(2, 2) = 6.0;
            expected.makeCompressed();

            ASSERT_EQ(actual.nonZeros(), expected.nonZeros());
            for (Eigen::Index outer = 0; outer <= expected.outerSize(); ++outer)
                EXPECT_EQ(actual.outerIndexPtr()[outer], expected.outerIndexPtr()[outer]);
            for (Eigen::Index index = 0; index < expected.nonZeros(); ++index)
            {
                EXPECT_EQ(actual.innerIndexPtr()[index], expected.innerIndexPtr()[index]);
                EXPECT_DOUBLE_EQ(actual.valuePtr()[index], expected.valuePtr()[index]);
            }

            plamatrix::Map<plamatrix::SparseMatrix<double>> mapped(
                3, 3, actual.nonZeros(), actual.outerIndexPtr(), actual.innerIndexPtr(), actual.valuePtr());
            mapped.valuePtr()[1] = 7.0;
            EXPECT_DOUBLE_EQ(actual.coeff(2, 0), 7.0);
        }

        TEST(EigenReference, AdvancedDecompositionsAndComplexSemanticsMatchEigen)
        {
            plamatrix::Matrix3d actual;
            actual << 4.0, 1.0, -2.0, 1.0, 2.0, 0.5, -2.0, 0.5, 3.0;
            Eigen::Matrix3d expected;
            expected << 4.0, 1.0, -2.0, 1.0, 2.0, 0.5, -2.0, 0.5, 3.0;
            EXPECT_NEAR(actual.partialPivLu().rcond(), expected.partialPivLu().rcond(), 1.0e-12);

            const plamatrix::HessenbergDecomposition<plamatrix::Matrix3d> actual_hessenberg(actual);
            const Eigen::HessenbergDecomposition<Eigen::Matrix3d> expected_hessenberg(expected);
            EXPECT_TRUE(
                (actual_hessenberg.matrixQ() * actual_hessenberg.matrixH() * actual_hessenberg.matrixQ().transpose())
                    .isApprox(actual, 1.0e-11));
            EXPECT_TRUE((expected_hessenberg.matrixQ() * expected_hessenberg.matrixH() *
                         expected_hessenberg.matrixQ().transpose())
                            .isApprox(expected, 1.0e-11));

            plamatrix::Matrix<double, 3, 2> actual_rank_deficient;
            actual_rank_deficient << 1.0, 2.0, 2.0, 4.0, 3.0, 6.0;
            Eigen::Matrix<double, 3, 2> expected_rank_deficient;
            expected_rank_deficient << 1.0, 2.0, 2.0, 4.0, 3.0, 6.0;
            const auto actual_cod = actual_rank_deficient.completeOrthogonalDecomposition();
            const auto expected_cod = expected_rank_deficient.completeOrthogonalDecomposition();
            EXPECT_EQ(actual_cod.rank(), expected_cod.rank());
            EXPECT_TRUE((actual_rank_deficient * actual_cod.pseudoInverse() * actual_rank_deficient)
                            .isApprox(actual_rank_deficient, 1.0e-10));

            using Complex = std::complex<double>;
            plamatrix::Matrix<Complex, 2, 2> actual_complex;
            actual_complex << Complex{1.0, 1.0}, Complex{2.0, -1.0}, Complex{0.5, 1.0}, Complex{3.0, -2.0};
            Eigen::Matrix2cd expected_complex;
            expected_complex << Complex{1.0, 1.0}, Complex{2.0, -1.0}, Complex{0.5, 1.0}, Complex{3.0, -2.0};
            EXPECT_NEAR(actual_complex.norm(), expected_complex.norm(), 1.0e-12);
            plamatrix::Matrix<std::complex<double>, 2, 1> actual_vector;
            actual_vector << Complex{1.0, 2.0}, Complex{-3.0, 0.5};
            Eigen::Vector2cd expected_vector;
            expected_vector << Complex{1.0, 2.0}, Complex{-3.0, 0.5};
            EXPECT_NEAR(
                std::abs(actual_vector.dot(actual_vector) - expected_vector.dot(expected_vector)), 0.0, 1.0e-12);
            const auto actual_adjoint = actual_complex.adjoint();
            for (Index col = 0; col < 2; ++col)
                for (Index row = 0; row < 2; ++row)
                    EXPECT_NEAR(
                        std::abs(actual_adjoint(row, col) - expected_complex.adjoint()(row, col)), 0.0, 1.0e-12);
        }

        TEST(EigenReference, GeneralizedQzEigenvaluesMatchEigen)
        {
            constexpr Index size = 5;
            plamatrix::MatrixXd actual_first(size, size);
            plamatrix::MatrixXd actual_second(size, size);
            Eigen::MatrixXd expected_first(size, size);
            Eigen::MatrixXd expected_second(size, size);
            for (Index column = 0; column < size; ++column)
            {
                for (Index row = 0; row < size; ++row)
                {
                    const double first = std::sin(0.31 * (1 + row + 3 * column));
                    const double second =
                        std::cos(0.23 * (2 + 2 * row - column)) + (row == column ? static_cast<double>(size + 2) : 0.0);
                    actual_first(row, column) = expected_first(row, column) = first;
                    actual_second(row, column) = expected_second(row, column) = second;
                }
            }

            const plamatrix::GeneralizedEigenSolver<plamatrix::MatrixXd> actual(actual_first, actual_second);
            const Eigen::GeneralizedEigenSolver<Eigen::MatrixXd> expected(expected_first, expected_second);
            ASSERT_EQ(actual.info(), plamatrix::Success);
            ASSERT_EQ(expected.info(), Eigen::Success);

            std::vector<std::complex<double>> actual_values;
            std::vector<std::complex<double>> expected_values;
            for (Index index = 0; index < size; ++index)
            {
                actual_values.push_back(actual.eigenvalues()(index));
                expected_values.push_back(expected.eigenvalues()(index));
            }
            const auto order = [](const auto& left, const auto& right)
            { return left.real() != right.real() ? left.real() < right.real() : left.imag() < right.imag(); };
            std::sort(actual_values.begin(), actual_values.end(), order);
            std::sort(expected_values.begin(), expected_values.end(), order);
            for (Index index = 0; index < size; ++index)
                EXPECT_NEAR(std::abs(actual_values[static_cast<std::size_t>(index)] -
                                     expected_values[static_cast<std::size_t>(index)]),
                            0.0,
                            2.0e-8);
        }

    } // namespace

} // namespace plamatrix::internal
