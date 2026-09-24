#include <cmath>
#include <type_traits>

#include <gtest/gtest.h>

#include <plamatrix/plamatrix.h>

namespace
{
    constexpr double Pi = 3.141592653589793238462643383279502884;

    template <typename MatrixType> void expectIdentity(const MatrixType& matrix, double tolerance = 1.0e-10)
    {
        ASSERT_EQ(matrix.rows(), matrix.cols());
        for (plamatrix::Index row = 0; row < matrix.rows(); ++row)
            for (plamatrix::Index column = 0; column < matrix.cols(); ++column)
                EXPECT_NEAR(matrix(row, column), row == column ? 1.0 : 0.0, tolerance);
    }
} // namespace

TEST(Geometry, EigenNamesAndTransformModesArePublic)
{
    static_assert(std::is_base_of_v<plamatrix::QuaternionBase<plamatrix::Quaterniond>, plamatrix::Quaterniond>);
    static_assert(std::is_base_of_v<plamatrix::RotationBase<plamatrix::AngleAxisd, 3>, plamatrix::AngleAxisd>);
    static_assert(std::is_same_v<plamatrix::Isometry3d, plamatrix::Transform<double, 3, plamatrix::Isometry>>);
    static_assert(std::is_same_v<plamatrix::Affine3d, plamatrix::Transform<double, 3, plamatrix::Affine>>);
    static_assert(std::is_same_v<plamatrix::Translation3d, plamatrix::Translation<double, 3>>);

    EXPECT_EQ(plamatrix::Isometry3d::Identity().matrix().rows(), 4);
    EXPECT_EQ(plamatrix::AffineCompact3d::Identity().matrix().rows(), 3);
    EXPECT_EQ(plamatrix::Projective3d::Identity().matrix().cols(), 4);
}

TEST(Geometry, QuaternionAngleAxisAndRotationMatrixRoundTrip)
{
    const plamatrix::AngleAxisd angle_axis(Pi / 3.0, plamatrix::Vector3d::UnitZ());
    const plamatrix::Quaterniond quaternion(angle_axis);
    EXPECT_NEAR(quaternion.w(), std::cos(Pi / 6.0), 1.0e-12);
    EXPECT_NEAR(quaternion.z(), std::sin(Pi / 6.0), 1.0e-12);
    EXPECT_DOUBLE_EQ(quaternion.coeffs()(3), quaternion.w());

    const plamatrix::Matrix3d rotation = quaternion.toRotationMatrix();
    const plamatrix::Quaterniond restored(rotation);
    EXPECT_NEAR(quaternion.angularDistance(restored), 0.0, 1.0e-12);
    const plamatrix::AngleAxisd restored_angle(restored);
    EXPECT_NEAR(restored_angle.angle(), Pi / 3.0, 1.0e-12);
    EXPECT_TRUE(restored_angle.axis().isApprox(plamatrix::Vector3d::UnitZ(), 1.0e-12));

    const plamatrix::Vector3d rotated = quaternion * plamatrix::Vector3d::UnitX();
    EXPECT_NEAR(rotated.x(), 0.5, 1.0e-12);
    EXPECT_NEAR(rotated.y(), std::sqrt(3.0) / 2.0, 1.0e-12);
    EXPECT_NEAR(rotated.z(), 0.0, 1.0e-12);
}

TEST(Geometry, QuaternionSlerpAndTwoVectorRotationUseShortestArc)
{
    const plamatrix::Quaterniond identity = plamatrix::Quaterniond::Identity();
    const plamatrix::Quaterniond half_turn(plamatrix::AngleAxisd(Pi, plamatrix::Vector3d::UnitZ()));
    const plamatrix::Quaterniond halfway = identity.slerp(0.5, half_turn);
    const plamatrix::Vector3d rotated = halfway * plamatrix::Vector3d::UnitX();
    EXPECT_NEAR(rotated.x(), 0.0, 1.0e-12);
    EXPECT_NEAR(rotated.y(), 1.0, 1.0e-12);

    plamatrix::Quaterniond between;
    between.setFromTwoVectors(plamatrix::Vector3d::UnitX(), -1.0 * plamatrix::Vector3d::UnitX());
    const plamatrix::Vector3d opposite = between * plamatrix::Vector3d::UnitX();
    EXPECT_TRUE(opposite.isApprox(plamatrix::Vector3d(-1.0, 0.0, 0.0), 1.0e-12));

    const auto factory =
        plamatrix::Quaterniond::FromTwoVectors(plamatrix::Vector3d::UnitX(), plamatrix::Vector3d::UnitY());
    EXPECT_TRUE((factory * plamatrix::Vector3d::UnitX()).isApprox(plamatrix::Vector3d::UnitY(), 1.0e-12));
    const auto scalar_last = plamatrix::Quaterniond::FromCoeffsScalarLast(0.1, 0.2, 0.3, 0.4);
    const plamatrix::Vector4d coefficients(0.1, 0.2, 0.3, 0.4);
    EXPECT_EQ(scalar_last, plamatrix::Quaterniond(coefficients));
    EXPECT_EQ(scalar_last, plamatrix::Quaterniond::FromCoeffsScalarFirst(0.4, 0.1, 0.2, 0.3));
    EXPECT_NEAR(plamatrix::Quaterniond::UnitRandom().norm(), 1.0, 1.0e-12);
}

TEST(Geometry, TranslationTransformCompositionAndInverseMatchEigenSemantics)
{
    const plamatrix::Translation3d translation(1.0, -2.0, 3.0);
    const plamatrix::AngleAxisd rotation(Pi / 2.0, plamatrix::Vector3d::UnitZ());
    const plamatrix::Isometry3d pose = translation * rotation;
    const plamatrix::Vector3d transformed = pose * plamatrix::Vector3d(2.0, 0.0, -1.0);
    EXPECT_TRUE(transformed.isApprox(plamatrix::Vector3d(1.0, 0.0, 2.0), 1.0e-12));
    EXPECT_TRUE((pose.inverse() * transformed).isApprox(plamatrix::Vector3d(2.0, 0.0, -1.0), 1.0e-12));
    expectIdentity((pose * pose.inverse()).matrix());

    plamatrix::Affine3d affine = plamatrix::Affine3d::Identity();
    affine.translate(plamatrix::Vector3d(2.0, 3.0, 4.0));
    affine.scale(2.0);
    EXPECT_TRUE((affine * plamatrix::Vector3d(1.0, 1.0, 1.0)).isApprox(plamatrix::Vector3d(4.0, 5.0, 6.0), 1.0e-12));
    EXPECT_TRUE((affine.inverse(plamatrix::Affine) * (affine * plamatrix::Vector3d(1.0, 1.0, 1.0)))
                    .isApprox(plamatrix::Vector3d(1.0, 1.0, 1.0), 1.0e-12));
}

TEST(Geometry, AffineRotationScalingDecompositionMatchesEigenSemantics)
{
    const plamatrix::Matrix3d expected_rotation =
        plamatrix::AngleAxisd(0.45, plamatrix::Vector3d(1.0, -2.0, 0.5).normalized()).toRotationMatrix();
    const plamatrix::Vector3d factors(2.0, 3.0, 4.0);
    plamatrix::Affine3d affine(expected_rotation);
    affine.scale(factors);

    EXPECT_TRUE(affine.rotation().isApprox(expected_rotation, 1.0e-10));
    plamatrix::Matrix3d rotation;
    plamatrix::Matrix3d scaling;
    affine.computeRotationScaling(&rotation, &scaling);
    EXPECT_TRUE(rotation.isApprox(expected_rotation, 1.0e-10));
    EXPECT_NEAR(scaling(0, 0), factors(0), 1.0e-10);
    EXPECT_NEAR(scaling(1, 1), factors(1), 1.0e-10);
    EXPECT_NEAR(scaling(2, 2), factors(2), 1.0e-10);

    plamatrix::Matrix3d scaling_first;
    plamatrix::Matrix3d rotation_second;
    affine.computeScalingRotation(&scaling_first, &rotation_second);
    EXPECT_TRUE(rotation_second.isApprox(expected_rotation, 1.0e-10));
    EXPECT_TRUE((scaling_first * rotation_second).isApprox(plamatrix::Matrix3d(affine.linear()), 1.0e-10));
}

TEST(Geometry, EulerAnglesReconstructTaitBryanAndProperEulerRotations)
{
    const plamatrix::Quaterniond tait_bryan = plamatrix::AngleAxisd(0.4, plamatrix::Vector3d::UnitZ()) *
                                              plamatrix::AngleAxisd(-0.2, plamatrix::Vector3d::UnitY()) *
                                              plamatrix::AngleAxisd(0.3, plamatrix::Vector3d::UnitX());
    const plamatrix::Matrix3d rotation = tait_bryan.toRotationMatrix();
    const plamatrix::Vector3d angles = rotation.canonicalEulerAngles(2, 1, 0);
    const plamatrix::Quaterniond reconstructed = plamatrix::AngleAxisd(angles[0], plamatrix::Vector3d::UnitZ()) *
                                                 plamatrix::AngleAxisd(angles[1], plamatrix::Vector3d::UnitY()) *
                                                 plamatrix::AngleAxisd(angles[2], plamatrix::Vector3d::UnitX());
    EXPECT_NEAR(tait_bryan.angularDistance(reconstructed), 0.0, 1.0e-12);

    const plamatrix::Quaterniond proper = plamatrix::AngleAxisd(0.25, plamatrix::Vector3d::UnitZ()) *
                                          plamatrix::AngleAxisd(0.7, plamatrix::Vector3d::UnitX()) *
                                          plamatrix::AngleAxisd(-0.4, plamatrix::Vector3d::UnitZ());
    const plamatrix::Vector3d proper_angles = proper.toRotationMatrix().eulerAngles(2, 0, 2);
    const plamatrix::Quaterniond proper_reconstructed =
        plamatrix::AngleAxisd(proper_angles[0], plamatrix::Vector3d::UnitZ()) *
        plamatrix::AngleAxisd(proper_angles[1], plamatrix::Vector3d::UnitX()) *
        plamatrix::AngleAxisd(proper_angles[2], plamatrix::Vector3d::UnitZ());
    EXPECT_NEAR(proper.angularDistance(proper_reconstructed), 0.0, 1.0e-12);
    EXPECT_THROW(rotation.eulerAngles(3, 1, 0), std::invalid_argument);
}

TEST(Geometry, UmeyamaRecoversRigidAndSimilarityTransforms)
{
    plamatrix::Matrix<double, 3, 5> source;
    source << 0.0, 1.0, -2.0, 0.5, 3.0, 0.0, 2.0, 1.0, -1.0, 0.25, 0.0, -1.0, 0.5, 2.0, 1.5;
    const plamatrix::Matrix3d rotation =
        plamatrix::AngleAxisd(0.35, plamatrix::Vector3d(1.0, 2.0, -1.0).normalized()).toRotationMatrix();
    const plamatrix::Vector3d translation(2.0, -3.0, 1.5);
    constexpr double scale = 1.75;
    plamatrix::Matrix<double, 3, 5> destination;
    for (plamatrix::Index column = 0; column < source.cols(); ++column)
        for (plamatrix::Index row = 0; row < 3; ++row)
        {
            double value = translation(row);
            for (plamatrix::Index inner = 0; inner < 3; ++inner)
                value += scale * rotation(row, inner) * source(inner, column);
            destination(row, column) = value;
        }

    const auto similarity = plamatrix::umeyama(source, destination, true);
    for (plamatrix::Index row = 0; row < 3; ++row)
    {
        EXPECT_NEAR(similarity(row, 3), translation(row), 1.0e-10);
        for (plamatrix::Index column = 0; column < 3; ++column)
            EXPECT_NEAR(similarity(row, column), scale * rotation(row, column), 1.0e-10);
    }

    plamatrix::Matrix<double, 3, 5> rigid_destination;
    for (plamatrix::Index column = 0; column < source.cols(); ++column)
        for (plamatrix::Index row = 0; row < 3; ++row)
        {
            double value = translation(row);
            for (plamatrix::Index inner = 0; inner < 3; ++inner)
                value += rotation(row, inner) * source(inner, column);
            rigid_destination(row, column) = value;
        }
    const auto rigid = plamatrix::umeyama(source, rigid_destination, false);
    for (plamatrix::Index row = 0; row < 3; ++row)
        for (plamatrix::Index column = 0; column < 3; ++column)
            EXPECT_NEAR(rigid(row, column), rotation(row, column), 1.0e-10);
}
