#include <limits>

#include <gtest/gtest.h>

#include "plamatrix/dense/matrix_self_adjoint.h"

TEST(SelfAdjointEigenSolver, UsesLowerTriangleAndReturnsAscendingEigenpairs)
{
    plamatrix::Matrix3d input;
    input << 3.0, 99.0, 99.0, 1.0, 3.0, 99.0, 0.0, 0.0, 1.0;
    const plamatrix::SelfAdjointEigenSolver<plamatrix::Matrix3d> solver(input);
    const auto& values = solver.eigenvalues();
    EXPECT_NEAR(values(0), 1.0, 1e-12);
    EXPECT_NEAR(values(1), 2.0, 1e-12);
    EXPECT_NEAR(values(2), 4.0, 1e-12);

    plamatrix::Matrix3d diagonal = plamatrix::Matrix3d::Zero();
    for (plamatrix::Index index = 0; index < 3; ++index)
    {
        diagonal(index, index) = values(index);
    }
    plamatrix::Matrix3d symmetric = input;
    symmetric(0, 1) = input(1, 0);
    symmetric(0, 2) = input(2, 0);
    symmetric(1, 2) = input(2, 1);
    const plamatrix::Matrix3d reconstructed = solver.eigenvectors() * diagonal * solver.eigenvectors().transpose();
    EXPECT_TRUE(reconstructed.isApprox(symmetric, 1e-10));
}

TEST(SelfAdjointEigenSolver, RejectsNonFiniteInputAndRequiredDevice)
{
    plamatrix::Matrix3f input = plamatrix::Matrix3f::Identity();
    input(1, 0) = std::numeric_limits<float>::infinity();
    EXPECT_EQ((plamatrix::SelfAdjointEigenSolver<plamatrix::Matrix3f>(input).info()), plamatrix::InvalidInput);

    input(1, 0) = 0.0f;
    plamatrix::internal::ScopedExecutionPolicy required(plamatrix::internal::ExecutionPolicy::GpuRequired,
                                                        plamatrix::internal::Backend::OpenCl);
    if (plamatrix::internal::detail::GpuOps<float>::available(plamatrix::internal::Backend::OpenCl))
    {
        const plamatrix::SelfAdjointEigenSolver<plamatrix::Matrix3f> solver(input);
        EXPECT_EQ(solver.backend(), plamatrix::internal::Backend::OpenCl);
        EXPECT_EQ(solver.info(), plamatrix::Success);
        EXPECT_TRUE(solver.eigenvalues().isApprox(plamatrix::Vector3f::Ones(), 1e-5f));
        EXPECT_TRUE((solver.eigenvectors() * solver.eigenvectors().transpose())
                        .isApprox(plamatrix::Matrix3f::Identity(), 1e-5f));
    }
    else
    {
        EXPECT_THROW((plamatrix::SelfAdjointEigenSolver<plamatrix::Matrix3f>(input)), plamatrix::internal::Error);
    }
}
