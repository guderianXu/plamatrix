#include <algorithm>
#include <complex>
#include <type_traits>

#include <gtest/gtest.h>

#include "plamatrix/plamatrix.h"

TEST(DenseDecompositions, MatrixMembersUseEigenNamesAndTypes)
{
    using Matrix = plamatrix::Matrix3d;
    static_assert(std::is_same_v<decltype(std::declval<const Matrix&>().llt()), plamatrix::LLT<Matrix>>);
    static_assert(std::is_same_v<decltype(std::declval<const Matrix&>().fullPivLu()), plamatrix::FullPivLU<Matrix>>);
    static_assert(
        std::is_same_v<decltype(std::declval<const Matrix&>().householderQr()), plamatrix::HouseholderQR<Matrix>>);
    static_assert(std::is_same_v<decltype(std::declval<const Matrix&>().bdcSvd()), plamatrix::BDCSVD<Matrix>>);
}

TEST(DenseDecompositions, LuRcondEstimatesReciprocalOneNormCondition)
{
    plamatrix::Matrix2d diagonal;
    diagonal << 2.0, 0.0, 0.0, 1.0;
    EXPECT_NEAR(diagonal.partialPivLu().rcond(), 0.5, 1.0e-15);
    EXPECT_NEAR(diagonal.fullPivLu().rcond(), 0.5, 1.0e-15);

    plamatrix::Matrix2d singular;
    singular << 1.0, 0.0, 0.0, 0.0;
    EXPECT_DOUBLE_EQ(singular.fullPivLu().rcond(), 0.0);
}

TEST(DenseDecompositions, HessenbergAndTridiagonalizationReconstructInput)
{
    plamatrix::Matrix4d general;
    general << 4.0, 1.0, -2.0, 2.0, 1.0, 2.0, 0.0, 1.0, -2.0, 0.0, 3.0, -2.0, 2.0, 1.0, -2.0, -1.0;
    plamatrix::HessenbergDecomposition<plamatrix::Matrix4d> hessenberg(general);
    ASSERT_EQ(hessenberg.info(), plamatrix::Success);
    const auto q = hessenberg.matrixQ();
    EXPECT_TRUE((q * hessenberg.matrixH() * q.transpose()).isApprox(general, 1.0e-10));
    for (plamatrix::Index col = 0; col < 4; ++col)
        for (plamatrix::Index row = col + 2; row < 4; ++row)
            EXPECT_NEAR(hessenberg.matrixH()(row, col), 0.0, 1.0e-12);

    plamatrix::Tridiagonalization<plamatrix::Matrix4d> tridiagonal(general);
    ASSERT_EQ(tridiagonal.info(), plamatrix::Success);
    const auto tq = tridiagonal.matrixQ();
    EXPECT_TRUE((tq * tridiagonal.matrixT() * tq.transpose()).isApprox(general, 1.0e-10));
    EXPECT_EQ(tridiagonal.diagonal().size(), 4);
    EXPECT_EQ(tridiagonal.subDiagonal().size(), 3);
}

TEST(DenseDecompositions, RealAndComplexSchurExposeEigenStyleFactors)
{
    plamatrix::Matrix3d real;
    real << 1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 0.0, 0.0, 6.0;
    plamatrix::RealSchur<plamatrix::Matrix3d> real_schur(real);
    ASSERT_EQ(real_schur.info(), plamatrix::Success);
    EXPECT_TRUE(
        (real_schur.matrixU() * real_schur.matrixT() * real_schur.matrixU().transpose()).isApprox(real, 1.0e-10));
    const plamatrix::HessenbergDecomposition<plamatrix::Matrix3d> hessenberg(real);
    plamatrix::RealSchur<plamatrix::Matrix3d> real_from_hessenberg;
    real_from_hessenberg.computeFromHessenberg(hessenberg.matrixH(), hessenberg.matrixQ(), true);
    ASSERT_EQ(real_from_hessenberg.info(), plamatrix::Success);
    EXPECT_TRUE(
        (real_from_hessenberg.matrixU() * real_from_hessenberg.matrixT() * real_from_hessenberg.matrixU().transpose())
            .isApprox(real, 1.0e-10));

    using Complex = std::complex<double>;
    plamatrix::Matrix<Complex, 2, 2> complex;
    complex << Complex{1.0, 1.0}, Complex{2.0, -1.0}, Complex{0.5, 1.0}, Complex{3.0, 2.0};
    plamatrix::ComplexSchur<decltype(complex)> complex_schur(complex);
    ASSERT_EQ(complex_schur.info(), plamatrix::Success);
    EXPECT_TRUE((complex_schur.matrixU() * complex_schur.matrixT() * complex_schur.matrixU().adjoint())
                    .isApprox(complex, 1.0e-9));
    EXPECT_NEAR(std::abs(complex_schur.matrixT()(1, 0)), 0.0, 1.0e-9);
    plamatrix::ComplexSchur<decltype(complex)> complex_from_hessenberg;
    complex_from_hessenberg.computeFromHessenberg(complex, decltype(complex)::Identity(), true);
    ASSERT_EQ(complex_from_hessenberg.info(), plamatrix::Success);
    EXPECT_TRUE((complex_from_hessenberg.matrixU() * complex_from_hessenberg.matrixT() *
                 complex_from_hessenberg.matrixU().adjoint())
                    .isApprox(complex, 1.0e-9));
}

TEST(DenseDecompositions, CompleteOrthogonalDecompositionAndComplexEigenSolverWork)
{
    plamatrix::Matrix<double, 3, 2> rank_deficient;
    rank_deficient << 1.0, 2.0, 2.0, 4.0, 3.0, 6.0;
    const auto cod = rank_deficient.completeOrthogonalDecomposition();
    ASSERT_EQ(cod.info(), plamatrix::Success);
    EXPECT_EQ(cod.rank(), 1);
    const auto pseudo = cod.pseudoInverse();
    EXPECT_TRUE((rank_deficient * pseudo * rank_deficient).isApprox(rank_deficient, 1.0e-9));

    using Complex = std::complex<double>;
    plamatrix::Matrix<Complex, 2, 2> matrix;
    matrix << Complex{1.0, 1.0}, Complex{2.0, -1.0}, Complex{0.5, 1.0}, Complex{3.0, -2.0};
    plamatrix::ComplexEigenSolver<decltype(matrix)> solver(matrix);
    ASSERT_EQ(solver.info(), plamatrix::Success);
    for (plamatrix::Index eigen = 0; eigen < 2; ++eigen)
    {
        const auto vector = solver.eigenvectors().col(eigen).eval();
        EXPECT_TRUE((matrix * vector).isApprox(solver.eigenvalues()(eigen) * vector, 1.0e-7));
    }
}

TEST(DenseDecompositions, LltAndFullPivLuSolveAndReportStructure)
{
    plamatrix::Matrix3d positive;
    positive << 4.0, 1.0, 1.0, 1.0, 3.0, 0.0, 1.0, 0.0, 2.0;
    plamatrix::Vector3d expected(1.0, 2.0, -1.0);
    const auto right = positive * expected;
    const auto llt = positive.llt();
    ASSERT_EQ(llt.info(), plamatrix::Success);
    EXPECT_TRUE(llt.solve(right).isApprox(expected, 1e-11));
    EXPECT_TRUE((llt.matrixL() * llt.matrixU()).isApprox(positive, 1e-11));

    plamatrix::Matrix3d pivoted;
    pivoted << 0.0, 2.0, 1.0, 3.0, 0.0, 4.0, 5.0, 6.0, 0.0;
    const auto lu = pivoted.fullPivLu();
    EXPECT_EQ(lu.rank(), 3);
    EXPECT_TRUE(lu.isInvertible());
    EXPECT_TRUE(lu.solve(pivoted * expected).isApprox(expected, 1e-11));

    plamatrix::MatrixXd deficient(2, 3);
    deficient << 1.0, 2.0, 3.0, 2.0, 4.0, 6.0;
    const auto deficient_lu = deficient.fullPivLu();
    ASSERT_EQ(deficient_lu.rank(), 1);
    const auto null_space = deficient_lu.kernel();
    EXPECT_EQ(null_space.cols(), 2);
    EXPECT_TRUE((deficient * null_space).isApprox(plamatrix::MatrixXd::Zero(2, 2), 1e-11));
}

TEST(DenseDecompositions, QrSupportsWideAndRankDeficientInputs)
{
    plamatrix::MatrixXd tall(4, 2);
    tall << 1.0, 2.0, 2.0, 0.0, 3.0, 1.0, 4.0, 3.0;
    const auto qr = tall.householderQr();
    const auto q = qr.householderQ();
    EXPECT_TRUE((q.transpose() * q).isApprox(plamatrix::MatrixXd::Identity(4, 4), 1e-11));

    plamatrix::MatrixXd wide(2, 3);
    wide << 1.0, 2.0, 3.0, 2.0, 4.0, 6.0;
    const auto pivoted = wide.colPivHouseholderQr();
    EXPECT_EQ(pivoted.rank(), 1);
    EXPECT_EQ(pivoted.dimensionOfKernel(), 2);
}

TEST(DenseDecompositions, SvdSupportsFullThinAndLargeMatrixEntryPoint)
{
    plamatrix::MatrixXd input(3, 2);
    input << 1.0, 0.0, 0.0, 2.0, 2.0, 0.0;
    const plamatrix::JacobiSVD<plamatrix::MatrixXd, plamatrix::ComputeFullU | plamatrix::ComputeFullV> full(input);
    ASSERT_EQ(full.info(), plamatrix::Success);
    EXPECT_EQ(full.matrixU().rows(), 3);
    EXPECT_EQ(full.matrixU().cols(), 3);
    EXPECT_EQ(full.matrixV().rows(), 2);
    EXPECT_EQ(full.matrixV().cols(), 2);

    const auto thin = input.jacobiSvd<plamatrix::ComputeThinU | plamatrix::ComputeThinV>();
    EXPECT_EQ(thin.matrixU().cols(), 2);
    EXPECT_EQ(thin.matrixV().cols(), 2);
    const auto large = input.bdcSvd<plamatrix::ComputeThinU | plamatrix::ComputeThinV>();
    EXPECT_EQ(large.info(), plamatrix::Success);
    EXPECT_TRUE(large.singularValues().isApprox(thin.singularValues(), 1e-11));
    EXPECT_THROW((plamatrix::JacobiSVD<plamatrix::MatrixXd>(input, plamatrix::ComputeFullU | plamatrix::ComputeThinU)),
                 std::invalid_argument);
}

TEST(DenseDecompositions, ComplexLuQrAndSvdUseHermitianSemantics)
{
    using Complex = std::complex<double>;
    using Matrix = plamatrix::Matrix<Complex, 2, 2>;
    Matrix input;
    input << Complex{2.0, 1.0}, Complex{1.0, -1.0}, Complex{3.0, 2.0}, Complex{4.0, -1.0};
    plamatrix::Matrix<Complex, 2, 1> expected;
    expected << Complex{1.0, 2.0}, Complex{-2.0, 1.0};
    const auto right = input * expected;

    EXPECT_TRUE(input.partialPivLu().solve(right).isApprox(expected, 1.0e-12));
    EXPECT_TRUE(input.fullPivLu().solve(right).isApprox(expected, 1.0e-12));
    const auto qr = input.householderQr();
    EXPECT_TRUE(qr.solve(right).isApprox(expected, 1.0e-12));
    EXPECT_TRUE((qr.householderQ().adjoint() * qr.householderQ()).isApprox(Matrix::Identity(), 1.0e-12));
    EXPECT_TRUE(input.colPivHouseholderQr().solve(right).isApprox(expected, 1.0e-12));

    const auto svd = input.jacobiSvd<plamatrix::ComputeFullU | plamatrix::ComputeFullV>();
    ASSERT_EQ(svd.info(), plamatrix::Success);
    EXPECT_TRUE((svd.matrixU() * svd.singularValues().asDiagonal() * svd.matrixV().adjoint()).isApprox(input, 1.0e-11));
    EXPECT_TRUE(svd.solve(right).isApprox(expected, 1.0e-10));
}

TEST(DenseDecompositions, ComplexQrAndSvdHandleRectangularMatrices)
{
    using Complex = std::complex<double>;
    for (const auto shape : {std::pair{5, 3}, std::pair{3, 5}})
    {
        plamatrix::MatrixXcd input(shape.first, shape.second);
        for (plamatrix::Index column = 0; column < input.cols(); ++column)
            for (plamatrix::Index row = 0; row < input.rows(); ++row)
                input(row, column) = Complex{0.25 * (1 + row + 2 * column), 0.2 * (row - column)};

        const auto qr = input.householderQr();
        ASSERT_EQ(qr.info(), plamatrix::Success);
        const auto q = qr.householderQ();
        EXPECT_TRUE((q.adjoint() * q).isApprox(plamatrix::MatrixXcd::Identity(input.rows(), input.rows()), 1.0e-11));

        const auto svd = input.jacobiSvd<plamatrix::ComputeFullU | plamatrix::ComputeFullV>();
        ASSERT_EQ(svd.info(), plamatrix::Success);
        const plamatrix::Index minor = std::min(input.rows(), input.cols());
        const auto reconstructed = svd.matrixU().block(0, 0, input.rows(), minor) * svd.singularValues().asDiagonal() *
                                   svd.matrixV().block(0, 0, input.cols(), minor).adjoint();
        EXPECT_TRUE(reconstructed.isApprox(input, 1.0e-10));
        EXPECT_TRUE((svd.matrixU().adjoint() * svd.matrixU())
                        .isApprox(plamatrix::MatrixXcd::Identity(svd.matrixU().cols(), svd.matrixU().cols()), 1.0e-10));
        EXPECT_TRUE((svd.matrixV().adjoint() * svd.matrixV())
                        .isApprox(plamatrix::MatrixXcd::Identity(svd.matrixV().cols(), svd.matrixV().cols()), 1.0e-10));
    }
}

TEST(DenseDecompositions, RealQzReconstructsDeterministicPencilsAcrossSizes)
{
    for (plamatrix::Index size = 2; size <= 8; ++size)
    {
        plamatrix::MatrixXd first(size, size);
        plamatrix::MatrixXd second(size, size);
        for (plamatrix::Index column = 0; column < size; ++column)
        {
            for (plamatrix::Index row = 0; row < size; ++row)
            {
                first(row, column) = std::sin(0.37 * (1 + row + 3 * column));
                second(row, column) = std::cos(0.29 * (2 + 2 * row - column));
            }
            second(column, column) += static_cast<double>(size + 1);
        }

        const plamatrix::RealQZ<plamatrix::MatrixXd> qz(first, second);
        ASSERT_EQ(qz.info(), plamatrix::Success) << "size=" << size;
        EXPECT_TRUE((qz.matrixQ().transpose() * first * qz.matrixZ()).isApprox(qz.matrixS(), 2.0e-9))
            << "size=" << size;
        EXPECT_TRUE((qz.matrixQ().transpose() * second * qz.matrixZ()).isApprox(qz.matrixT(), 2.0e-9))
            << "size=" << size;
        EXPECT_TRUE(
            (qz.matrixQ().transpose() * qz.matrixQ()).isApprox(plamatrix::MatrixXd::Identity(size, size), 2.0e-10));
        EXPECT_TRUE(
            (qz.matrixZ().transpose() * qz.matrixZ()).isApprox(plamatrix::MatrixXd::Identity(size, size), 2.0e-10));
    }
}

TEST(DenseDecompositions, RealQzUsesOrthogonalEquivalenceAndHandlesSingularMetric)
{
    plamatrix::Matrix4d first;
    first << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 2.0, -1.0, 3.0, 5.0, 1.0, 0.0, 2.0, 4.0;
    plamatrix::Matrix4d second;
    second << 4.0, 1.0, 0.0, 2.0, 1.0, 3.0, 1.0, 0.0, 0.0, 1.0, 2.0, 1.0, 2.0, 0.0, 1.0, 3.0;
    for (plamatrix::Index row = 0; row < second.rows(); ++row)
        second(row, 3) = 0.0;

    const plamatrix::RealQZ<plamatrix::Matrix4d> qz(first, second);
    ASSERT_EQ(qz.info(), plamatrix::Success);
    EXPECT_TRUE((qz.matrixQ().transpose() * first * qz.matrixZ()).isApprox(qz.matrixS(), 1.0e-10));
    EXPECT_TRUE((qz.matrixQ().transpose() * second * qz.matrixZ()).isApprox(qz.matrixT(), 1.0e-10));
    EXPECT_TRUE((qz.matrixQ().transpose() * qz.matrixQ()).isApprox(plamatrix::Matrix4d::Identity(), 1.0e-11));
    EXPECT_TRUE((qz.matrixZ().transpose() * qz.matrixZ()).isApprox(plamatrix::Matrix4d::Identity(), 1.0e-11));
    for (plamatrix::Index row = 1; row < 4; ++row)
    {
        EXPECT_NEAR(qz.matrixT()(row, row - 1), 0.0, 1.0e-12);
        for (plamatrix::Index col = 0; col + 1 < row; ++col)
            EXPECT_NEAR(qz.matrixS()(row, col), 0.0, 1.0e-12);
    }

    const plamatrix::GeneralizedEigenSolver<plamatrix::Matrix4d> eigen(first, second);
    ASSERT_EQ(eigen.info(), plamatrix::Success);
    bool has_infinite_root = false;
    for (plamatrix::Index index = 0; index < 4; ++index)
        has_infinite_root = has_infinite_root || std::abs(eigen.betas()(index)) < 1.0e-12;
    EXPECT_TRUE(has_infinite_root);
    for (plamatrix::Index column = 0; column < 4; ++column)
    {
        const auto vector = eigen.eigenvectors().col(column).eval();
        const auto residual = eigen.betas()(column) * (first.cast<std::complex<double>>() * vector) -
                              eigen.alphas()(column) * (second.cast<std::complex<double>>() * vector);
        EXPECT_NEAR(residual.norm(), 0.0, 1.0e-8);
    }
}

TEST(DenseDecompositions, ArbitrarySelfAdjointAndGeneralizedSelfAdjointWork)
{
    plamatrix::MatrixXd diagonal = plamatrix::MatrixXd::Zero(5, 5);
    plamatrix::MatrixXd metric = plamatrix::MatrixXd::Zero(5, 5);
    for (plamatrix::Index index = 0; index < 5; ++index)
    {
        diagonal(index, index) = static_cast<double>(index + 1);
        metric(index, index) = 2.0;
    }
    const plamatrix::SelfAdjointEigenSolver<plamatrix::MatrixXd> ordinary(diagonal);
    ASSERT_EQ(ordinary.info(), plamatrix::Success);
    EXPECT_DOUBLE_EQ(ordinary.eigenvalues()(0), 1.0);
    EXPECT_DOUBLE_EQ(ordinary.eigenvalues()(4), 5.0);

    const plamatrix::GeneralizedSelfAdjointEigenSolver<plamatrix::MatrixXd> generalized(diagonal, metric);
    ASSERT_EQ(generalized.info(), plamatrix::Success);
    EXPECT_NEAR(generalized.eigenvalues()(0), 0.5, 1e-12);
    EXPECT_NEAR(generalized.eigenvalues()(4), 2.5, 1e-12);
    plamatrix::MatrixXd expected_sqrt = plamatrix::MatrixXd::Zero(5, 5);
    for (plamatrix::Index index = 0; index < 5; ++index)
        expected_sqrt(index, index) = std::sqrt(diagonal(index, index));
    EXPECT_TRUE(ordinary.operatorSqrt().isApprox(expected_sqrt, 1e-11));
}

TEST(DenseDecompositions, GeneralAndGeneralizedEigenSolversReturnComplexPairs)
{
    plamatrix::Matrix2d rotation;
    rotation << 0.0, -1.0, 1.0, 0.0;
    const plamatrix::EigenSolver<plamatrix::Matrix2d> solver(rotation);
    ASSERT_EQ(solver.info(), plamatrix::Success);
    EXPECT_NEAR(std::abs(solver.eigenvalues()(0)), 1.0, 1e-10);
    EXPECT_NEAR(std::abs(solver.eigenvalues()(1)), 1.0, 1e-10);
    EXPECT_NEAR(solver.eigenvalues()(0).real(), 0.0, 1e-10);
    EXPECT_NEAR(std::abs(solver.eigenvalues()(0).imag()), 1.0, 1e-10);

    plamatrix::Matrix3d companion;
    companion << 0.0, 0.0, -1.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0;
    const plamatrix::EigenSolver<plamatrix::Matrix3d> companion_solver(companion);
    ASSERT_EQ(companion_solver.info(), plamatrix::Success);
    for (plamatrix::Index eigen = 0; eigen < 3; ++eigen)
    {
        for (plamatrix::Index row = 0; row < 3; ++row)
        {
            std::complex<double> residual{};
            for (plamatrix::Index column = 0; column < 3; ++column)
                residual += companion(row, column) * companion_solver.eigenvectors()(column, eigen);
            residual -= companion_solver.eigenvalues()(eigen) * companion_solver.eigenvectors()(row, eigen);
            EXPECT_NEAR(std::abs(residual), 0.0, 1e-8);
        }
    }

    plamatrix::Matrix2d metric;
    metric << 2.0, 0.0, 0.0, 4.0;
    plamatrix::Matrix2d diagonal;
    diagonal << 6.0, 0.0, 0.0, 20.0;
    const plamatrix::GeneralizedEigenSolver<plamatrix::Matrix2d> generalized(diagonal, metric);
    ASSERT_EQ(generalized.info(), plamatrix::Success);
    auto values = generalized.eigenvalues();
    const double first = values(0).real();
    const double second = values(1).real();
    EXPECT_NEAR(std::min(first, second), 3.0, 1e-10);
    EXPECT_NEAR(std::max(first, second), 5.0, 1e-10);

    plamatrix::Matrix2d singular_metric;
    singular_metric << 1.0, 0.0, 0.0, 0.0;
    const plamatrix::GeneralizedEigenSolver<plamatrix::Matrix2d> singular_generalized(diagonal, singular_metric);
    ASSERT_EQ(singular_generalized.info(), plamatrix::Success);
    EXPECT_DOUBLE_EQ(singular_generalized.betas()(0), 1.0);
    EXPECT_DOUBLE_EQ(singular_generalized.betas()(1), 0.0);
    EXPECT_NEAR(singular_generalized.alphas()(0).real(), 6.0, 1.0e-10);
}

TEST(DenseDecompositions, GpuAcceleratesNativeCudaDecompositions)
{
    plamatrix::MatrixXf positive = plamatrix::MatrixXf::Identity(8, 8);
    for (plamatrix::Index index = 0; index < positive.rows(); ++index)
        positive(index, index) += index;
    plamatrix::internal::ScopedExecutionPolicy required(plamatrix::internal::ExecutionPolicy::GpuRequired,
                                                        plamatrix::internal::Backend::Cuda);

    const auto llt = positive.llt();
    ASSERT_EQ(llt.info(), plamatrix::Success);
    EXPECT_EQ(llt.backend(), plamatrix::internal::Backend::Cuda);

    const auto qr = positive.householderQr();
    ASSERT_EQ(qr.info(), plamatrix::Success);
    EXPECT_EQ(qr.backend(), plamatrix::internal::Backend::Cuda);

    const auto svd = positive.bdcSvd<plamatrix::ComputeThinU | plamatrix::ComputeThinV>();
    ASSERT_EQ(svd.info(), plamatrix::Success);
    EXPECT_EQ(svd.backend(), plamatrix::internal::Backend::Cuda);

    const plamatrix::SelfAdjointEigenSolver<plamatrix::MatrixXf> eigen(positive);
    ASSERT_EQ(eigen.info(), plamatrix::Success);
    EXPECT_EQ(eigen.backend(), plamatrix::internal::Backend::Cuda);
}

TEST(DenseDecompositions, PortableBackendsExecuteQrSvdAndSelfAdjointEigen)
{
    using plamatrix::internal::Backend;
    using plamatrix::internal::ExecutionPolicy;
    using plamatrix::internal::ScopedExecutionPolicy;
    using plamatrix::internal::detail::GpuOps;

    for (const auto backend : {Backend::OpenCl, Backend::Vulkan})
    {
        if (!GpuOps<float>::available(backend))
            continue;
        plamatrix::MatrixXf input(5, 3);
        for (plamatrix::Index column = 0; column < input.cols(); ++column)
            for (plamatrix::Index row = 0; row < input.rows(); ++row)
                input(row, column) =
                    std::sin(0.31F * static_cast<float>(1 + row + 3 * column)) + (row == column ? 1.0F : 0.0F);
        plamatrix::MatrixXf symmetric = input * input.transpose();
        for (plamatrix::Index index = 0; index < symmetric.rows(); ++index)
            symmetric(index, index) += 0.5F;
        plamatrix::VectorXf expected(3);
        expected << 0.5F, -1.25F, 2.0F;
        const auto right = input * expected;

        ScopedExecutionPolicy required(ExecutionPolicy::GpuRequired, backend);
        const auto qr = input.householderQr();
        ASSERT_EQ(qr.info(), plamatrix::Success);
        EXPECT_EQ(qr.backend(), backend);
        EXPECT_TRUE((qr.householderQ() * qr.matrixQR()).isApprox(input, 3.0e-4F));
        EXPECT_TRUE(
            (qr.householderQ().transpose() * qr.householderQ()).isApprox(plamatrix::MatrixXf::Identity(5, 5), 3.0e-4F));
        EXPECT_TRUE(qr.solve(right).isApprox(expected, 8.0e-4F));

        const auto svd = input.jacobiSvd<plamatrix::ComputeFullU | plamatrix::ComputeFullV>();
        ASSERT_EQ(svd.info(), plamatrix::Success);
        EXPECT_EQ(svd.backend(), backend);
        const auto reconstructed =
            svd.matrixU().block(0, 0, 5, 3) * svd.singularValues().asDiagonal() * svd.matrixV().transpose();
        EXPECT_TRUE(reconstructed.isApprox(input, 8.0e-4F));
        EXPECT_TRUE((svd.matrixU().transpose() * svd.matrixU()).isApprox(plamatrix::MatrixXf::Identity(5, 5), 8.0e-4F));
        EXPECT_TRUE((svd.matrixV().transpose() * svd.matrixV()).isApprox(plamatrix::MatrixXf::Identity(3, 3), 8.0e-4F));
        EXPECT_TRUE(svd.solve(right).isApprox(expected, 8.0e-4F));

        const plamatrix::SelfAdjointEigenSolver<plamatrix::MatrixXf> eigen(symmetric);
        ASSERT_EQ(eigen.info(), plamatrix::Success);
        EXPECT_EQ(eigen.backend(), backend);
        EXPECT_TRUE((eigen.eigenvectors() * eigen.eigenvalues().asDiagonal() * eigen.eigenvectors().transpose())
                        .isApprox(symmetric, 1.0e-3F));
        EXPECT_TRUE((eigen.eigenvectors().transpose() * eigen.eigenvectors())
                        .isApprox(plamatrix::MatrixXf::Identity(5, 5), 1.0e-3F));
    }
}

TEST(DenseDecompositions, PortableBackendsHandleWideRankDeficientAndFp64Inputs)
{
    using plamatrix::internal::Backend;
    using plamatrix::internal::ExecutionPolicy;
    using plamatrix::internal::ScopedExecutionPolicy;
    using plamatrix::internal::detail::GpuOps;

    for (const auto backend : {Backend::OpenCl, Backend::Vulkan})
    {
        if (!GpuOps<float>::available(backend))
            continue;
        SCOPED_TRACE(backend == Backend::OpenCl ? "OpenCL" : "Vulkan");
        ScopedExecutionPolicy required(ExecutionPolicy::GpuRequired, backend);
        for (const auto shape : {std::pair{7, 4}, std::pair{4, 7}})
        {
            plamatrix::MatrixXf input(shape.first, shape.second);
            for (plamatrix::Index column = 0; column < input.cols(); ++column)
                for (plamatrix::Index row = 0; row < input.rows(); ++row)
                    input(row, column) =
                        std::cos(0.17F * static_cast<float>(2 + 2 * row - column)) + (row == column ? 0.75F : 0.0F);
            const auto qr = input.householderQr();
            ASSERT_EQ(qr.info(), plamatrix::Success);
            EXPECT_EQ(qr.backend(), backend);
            EXPECT_TRUE((qr.householderQ() * qr.matrixQR()).isApprox(input, 8.0e-4F));

            const auto svd = input.jacobiSvd<plamatrix::ComputeFullU | plamatrix::ComputeFullV>();
            ASSERT_EQ(svd.info(), plamatrix::Success);
            EXPECT_EQ(svd.backend(), backend);
            const plamatrix::Index minor = std::min(input.rows(), input.cols());
            const auto reconstructed = svd.matrixU().block(0, 0, input.rows(), minor) *
                                       svd.singularValues().asDiagonal() *
                                       svd.matrixV().block(0, 0, input.cols(), minor).transpose();
            EXPECT_TRUE(reconstructed.isApprox(input, 1.5e-3F));
        }

        plamatrix::MatrixXf deficient(8, 4);
        for (plamatrix::Index row = 0; row < deficient.rows(); ++row)
        {
            deficient(row, 0) = static_cast<float>(row + 1);
            deficient(row, 1) = 2.0F * deficient(row, 0);
            deficient(row, 2) = std::sin(0.3F * static_cast<float>(row));
            deficient(row, 3) = deficient(row, 0) - deficient(row, 2);
        }
        const auto deficientSvd = deficient.jacobiSvd<plamatrix::ComputeThinU | plamatrix::ComputeThinV>();
        const auto reconstructed =
            deficientSvd.matrixU() * deficientSvd.singularValues().asDiagonal() * deficientSvd.matrixV().transpose();
        EXPECT_TRUE(reconstructed.isApprox(deficient, 2.0e-3F));
        EXPECT_EQ(deficientSvd.rank(), 2);

        plamatrix::MatrixXf symmetric(9, 9);
        for (plamatrix::Index column = 0; column < symmetric.cols(); ++column)
            for (plamatrix::Index row = 0; row < symmetric.rows(); ++row)
                symmetric(row, column) = row == column ? 4.0F + static_cast<float>(row)
                                                       : 0.1F * static_cast<float>(1 + std::min(row, column));
        const plamatrix::SelfAdjointEigenSolver<plamatrix::MatrixXf> eigen(symmetric);
        ASSERT_EQ(eigen.info(), plamatrix::Success);
        EXPECT_TRUE((eigen.eigenvectors() * eigen.eigenvalues().asDiagonal() * eigen.eigenvectors().transpose())
                        .isApprox(symmetric, 2.0e-3F));
    }

    if (GpuOps<double>::available(Backend::OpenCl))
    {
        ScopedExecutionPolicy required(ExecutionPolicy::GpuRequired, Backend::OpenCl);
        plamatrix::MatrixXd input(6, 4);
        for (plamatrix::Index column = 0; column < input.cols(); ++column)
            for (plamatrix::Index row = 0; row < input.rows(); ++row)
                input(row, column) =
                    std::sin(0.23 * static_cast<double>(1 + row + 2 * column)) + (row == column ? 1.0 : 0.0);
        const auto qr = input.householderQr();
        EXPECT_EQ(qr.backend(), Backend::OpenCl);
        EXPECT_TRUE((qr.householderQ() * qr.matrixQR()).isApprox(input, 1.0e-10));
        const auto svd = input.jacobiSvd<plamatrix::ComputeThinU | plamatrix::ComputeThinV>();
        EXPECT_EQ(svd.backend(), Backend::OpenCl);
        EXPECT_TRUE(
            (svd.matrixU() * svd.singularValues().asDiagonal() * svd.matrixV().transpose()).isApprox(input, 2.0e-9));
    }
}

TEST(DenseDecompositions, PortableBackendsMaintainMediumMatrixAccuracy)
{
    using plamatrix::internal::Backend;
    using plamatrix::internal::ExecutionPolicy;
    using plamatrix::internal::ScopedExecutionPolicy;
    using plamatrix::internal::detail::GpuOps;

    plamatrix::MatrixXf input(32, 16);
    for (plamatrix::Index column = 0; column < input.cols(); ++column)
    {
        for (plamatrix::Index row = 0; row < input.rows(); ++row)
        {
            input(row, column) = std::sin(0.071F * static_cast<float>(1 + row + 3 * column)) +
                                 std::cos(0.043F * static_cast<float>(2 + 2 * row - column)) +
                                 (row == column ? 0.5F : 0.0F);
        }
    }
    plamatrix::MatrixXf symmetric(24, 24);
    for (plamatrix::Index column = 0; column < symmetric.cols(); ++column)
    {
        for (plamatrix::Index row = 0; row < symmetric.rows(); ++row)
        {
            symmetric(row, column) =
                row == column ? 8.0F + static_cast<float>(row) : 0.025F * static_cast<float>(1 + std::min(row, column));
        }
    }

    for (const auto backend : {Backend::OpenCl, Backend::Vulkan})
    {
        if (!GpuOps<float>::available(backend))
            continue;
        SCOPED_TRACE(backend == Backend::OpenCl ? "OpenCL" : "Vulkan");
        ScopedExecutionPolicy required(ExecutionPolicy::GpuRequired, backend);

        const auto svd = input.jacobiSvd<plamatrix::ComputeThinU | plamatrix::ComputeThinV>();
        const auto reconstructed = svd.matrixU() * svd.singularValues().asDiagonal() * svd.matrixV().transpose();
        EXPECT_LT((reconstructed - input).norm() / input.norm(), 2.0e-4F);
        EXPECT_TRUE(
            (svd.matrixU().transpose() * svd.matrixU()).isApprox(plamatrix::MatrixXf::Identity(16, 16), 2.0e-3F));
        EXPECT_TRUE(
            (svd.matrixV().transpose() * svd.matrixV()).isApprox(plamatrix::MatrixXf::Identity(16, 16), 2.0e-3F));

        const plamatrix::SelfAdjointEigenSolver<plamatrix::MatrixXf> eigen(symmetric);
        const auto eigenReconstructed =
            eigen.eigenvectors() * eigen.eigenvalues().asDiagonal() * eigen.eigenvectors().transpose();
        EXPECT_LT((eigenReconstructed - symmetric).norm() / symmetric.norm(), 5.0e-4F);
        EXPECT_TRUE((eigen.eigenvectors().transpose() * eigen.eigenvectors())
                        .isApprox(plamatrix::MatrixXf::Identity(24, 24), 2.0e-3F));
    }
}

TEST(DenseDecompositions, AutomaticPolicyUsesPortableDecompositionsAtMeasuredCrossover)
{
    using plamatrix::internal::Backend;
    using plamatrix::internal::ExecutionPolicy;
    using plamatrix::internal::ScopedExecutionPolicy;
    using plamatrix::internal::detail::GpuOps;

    const Backend expected = GpuOps<float>::available(Backend::OpenCl)
                                 ? Backend::OpenCl
                                 : (GpuOps<float>::available(Backend::Vulkan) ? Backend::Vulkan : Backend::Cpu);
    ScopedExecutionPolicy automatic(ExecutionPolicy::Auto, Backend::Cuda);

    plamatrix::MatrixXf input(48, 24);
    for (plamatrix::Index column = 0; column < input.cols(); ++column)
        for (plamatrix::Index row = 0; row < input.rows(); ++row)
            input(row, column) = std::sin(0.031F * static_cast<float>(1 + row + 2 * column));
    const auto svd = input.jacobiSvd<plamatrix::ComputeThinU | plamatrix::ComputeThinV>();
    EXPECT_EQ(svd.backend(), expected);

    plamatrix::MatrixXf symmetric(64, 64);
    for (plamatrix::Index column = 0; column < symmetric.cols(); ++column)
        for (plamatrix::Index row = 0; row < symmetric.rows(); ++row)
            symmetric(row, column) =
                row == column ? 10.0F + static_cast<float>(row) : 0.01F * static_cast<float>(1 + std::min(row, column));
    const plamatrix::SelfAdjointEigenSolver<plamatrix::MatrixXf> eigen(symmetric);
    EXPECT_EQ(eigen.backend(), expected);
}
