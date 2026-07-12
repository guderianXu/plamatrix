#include <gtest/gtest.h>

#include <array>
#include <limits>

#include <plamatrix/ops/vector.h>

TEST(Vector3, ConvertsArraysAndAppliesArithmetic)
{
    const plamatrix::Vec3<double> a(std::array<double, 3>{1.0, 2.0, 3.0});
    const plamatrix::Vec3<double> b{4.0, -1.0, 2.0};
    const auto sum = a + b;
    const auto difference = a - b;
    const auto scaled = 2.0 * a;

    EXPECT_EQ(sum.toArray(), (std::array<double, 3>{5.0, 1.0, 5.0}));
    EXPECT_EQ(difference.toArray(), (std::array<double, 3>{-3.0, 3.0, 1.0}));
    EXPECT_EQ(scaled.toArray(), (std::array<double, 3>{2.0, 4.0, 6.0}));
    EXPECT_EQ((a / 2.0).toArray(), (std::array<double, 3>{0.5, 1.0, 1.5}));
}

TEST(Vector3, AppliesCompoundArithmetic)
{
    plamatrix::Vec3<double> value{1.0, 2.0, 3.0};
    value += plamatrix::Vec3<double>{2.0, 1.0, -1.0};
    value -= plamatrix::Vec3<double>{1.0, 1.0, 1.0};

    EXPECT_EQ(value.toArray(), (std::array<double, 3>{2.0, 2.0, 1.0}));
    EXPECT_EQ((value * 3.0).toArray(), (std::array<double, 3>{6.0, 6.0, 3.0}));
}

TEST(Vector3, ComputesProductsAndNorms)
{
    const plamatrix::Vec3<double> x{1.0, 0.0, 0.0};
    const plamatrix::Vec3<double> y{0.0, 1.0, 0.0};
    EXPECT_DOUBLE_EQ(plamatrix::dot(x, y), 0.0);
    EXPECT_EQ(plamatrix::cross(x, y).toArray(), (std::array<double, 3>{0.0, 0.0, 1.0}));

    const plamatrix::Vec3<double> value{2.0, 3.0, 6.0};
    EXPECT_DOUBLE_EQ(plamatrix::squaredNorm(value), 49.0);
    EXPECT_DOUBLE_EQ(plamatrix::norm(value), 7.0);
}

TEST(Vector3, NormalizesAndPreservesNearZeroVectors)
{
    const auto unit = plamatrix::normalized(plamatrix::Vec3<double>{0.0, 3.0, 4.0});
    EXPECT_NEAR(unit.y, 0.6, 1.0e-12);
    EXPECT_NEAR(unit.z, 0.8, 1.0e-12);

    const plamatrix::Vec3<double> tiny{1.0e-15, 0.0, 0.0};
    EXPECT_EQ(plamatrix::normalized(tiny, 1.0e-12).toArray(), tiny.toArray());
}

TEST(Vector3, DetectsNonFiniteComponents)
{
    EXPECT_TRUE(plamatrix::isFinite(plamatrix::Vec3<double>{1.0, 2.0, 3.0}));
    EXPECT_FALSE(plamatrix::isFinite(plamatrix::Vec3<double>{
        std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0}));
    EXPECT_FALSE(plamatrix::isFinite(plamatrix::Vec3<double>{
        0.0, std::numeric_limits<double>::infinity(), 0.0}));
}
