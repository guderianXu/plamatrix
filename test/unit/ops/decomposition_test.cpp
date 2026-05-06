#include <cmath>
#include <cstdlib>

#include <gtest/gtest.h>

#include <plamatrix/ops/decomposition.h>
#include <plamatrix/dense/dense_ops.h>

using namespace plamatrix;

namespace
{

/// Compute A = U * diag(S) * Vt to verify SVD reconstruction.
/// @param U   m×m orthogonal matrix
/// @param S   min(m,n) × 1 column vector of singular values
/// @param Vt  n×n orthogonal matrix
/// @return  m×n matrix (reconstruction of original)
template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> reconstructFromSvd(const DenseMatrix<Scalar, Device::CPU>& U,
                                                     const DenseMatrix<Scalar, Device::CPU>& S,
                                                     const DenseMatrix<Scalar, Device::CPU>& Vt)
{
    // Compute U * diag(S): each column of U is scaled by S[row, 0]
    Index m = U.rows();
    Index n = S.rows();   // min(m, n)
    DenseMatrix<Scalar, Device::CPU> US(m, n);
    for (Index j = 0; j < n; ++j)
    {
        Scalar sj = S(j, 0);
        for (Index i = 0; i < m; ++i)
        {
            US(i, j) = U(i, j) * sj;
        }
    }

    // Multiply US * Vt: result is m × (Vt.cols()) = m × n_orig
    Index n_orig = Vt.cols();
    DenseMatrix<Scalar, Device::CPU> result(m, n_orig);
    for (Index j = 0; j < n_orig; ++j)
    {
        for (Index i = 0; i < m; ++i)
        {
            Scalar sum = Scalar(0);
            for (Index k = 0; k < n; ++k)
            {
                sum += US(i, k) * Vt(k, j);
            }
            result(i, j) = sum;
        }
    }
    return result;
}

/// Check orthogonality: U^T * U ≈ I
template <typename Scalar>
bool isOrthogonal(const DenseMatrix<Scalar, Device::CPU>& U, Scalar tolerance)
{
    Index n = U.cols();
    for (Index j = 0; j < n; ++j)
    {
        for (Index i = 0; i < n; ++i)
        {
            Scalar dot = Scalar(0);
            for (Index k = 0; k < U.rows(); ++k)
            {
                dot += U(k, i) * U(k, j);
            }
            Scalar expected = (i == j) ? Scalar(1) : Scalar(0);
            if (std::abs(dot - expected) > tolerance)
            {
                return false;
            }
        }
    }
    return true;
}

} // anonymous namespace

// SVD: decompose_3x3_Cpu — verify reconstruction and orthogonality
TEST(SVD, decompose_3x3_Cpu)
{
    // Known 3x3 matrix:
    // A = [ 4  1  2 ]
    //     [ 2  3  1 ]
    //     [ 1  2  4 ]
    DenseMatrix<double, Device::CPU> A(3, 3);
    A.setValue(0, 0, 4.0);
    A.setValue(1, 0, 2.0);
    A.setValue(2, 0, 1.0);
    A.setValue(0, 1, 1.0);
    A.setValue(1, 1, 3.0);
    A.setValue(2, 1, 2.0);
    A.setValue(0, 2, 2.0);
    A.setValue(1, 2, 1.0);
    A.setValue(2, 2, 4.0);

    auto [U, S, Vt] = svd(A);

    // Check dimensions
    EXPECT_EQ(U.rows(), 3);
    EXPECT_EQ(U.cols(), 3);
    EXPECT_EQ(S.rows(), 3);
    EXPECT_EQ(S.cols(), 1);
    EXPECT_EQ(Vt.rows(), 3);
    EXPECT_EQ(Vt.cols(), 3);

    // Check singular values are sorted descending
    EXPECT_GE(S(0, 0), S(1, 0));
    EXPECT_GE(S(1, 0), S(2, 0));

    // Check singular values are positive
    EXPECT_GT(S(0, 0), 0.0);
    EXPECT_GT(S(1, 0), 0.0);
    EXPECT_GT(S(2, 0), 0.0);

    // Check orthogonality of U: U^T * U ≈ I
    EXPECT_TRUE(isOrthogonal(U, 1e-9));

    // Check orthogonality of Vt: Vt * Vt^T ≈ I
    EXPECT_TRUE(isOrthogonal(Vt, 1e-9));

    // Check reconstruction: U * diag(S) * Vt ≈ A
    auto A_recon = reconstructFromSvd(U, S, Vt);

    const double tol = 1e-9;
    for (Index j = 0; j < 3; ++j)
    {
        for (Index i = 0; i < 3; ++i)
        {
            EXPECT_NEAR(A(i, j), A_recon(i, j), tol)
                << "Mismatch at (" << i << ", " << j << ")";
        }
    }
}

// SVD: decompose_3x3_Gpu — same matrix on GPU via cuSOLVER
TEST(SVD, decompose_3x3_Gpu)
{
    DenseMatrix<double, Device::CPU> A_cpu(3, 3);
    A_cpu.setValue(0, 0, 4.0);
    A_cpu.setValue(1, 0, 2.0);
    A_cpu.setValue(2, 0, 1.0);
    A_cpu.setValue(0, 1, 1.0);
    A_cpu.setValue(1, 1, 3.0);
    A_cpu.setValue(2, 1, 2.0);
    A_cpu.setValue(0, 2, 2.0);
    A_cpu.setValue(1, 2, 1.0);
    A_cpu.setValue(2, 2, 4.0);

    auto A_gpu = A_cpu.toGpu();
    auto [U_gpu, S_gpu, Vt_gpu] = svd(A_gpu);

    auto U = U_gpu.toCpu();
    auto S = S_gpu.toCpu();
    auto Vt = Vt_gpu.toCpu();

    // Check dimensions
    EXPECT_EQ(U.rows(), 3);
    EXPECT_EQ(U.cols(), 3);
    EXPECT_EQ(S.rows(), 3);
    EXPECT_EQ(S.cols(), 1);
    EXPECT_EQ(Vt.rows(), 3);
    EXPECT_EQ(Vt.cols(), 3);

    // Check singular values are sorted descending
    EXPECT_GE(S(0, 0), S(1, 0));
    EXPECT_GE(S(1, 0), S(2, 0));

    // Check singular values are positive
    EXPECT_GT(S(0, 0), 0.0);
    EXPECT_GT(S(1, 0), 0.0);
    EXPECT_GT(S(2, 0), 0.0);

    // Check orthogonality of U
    EXPECT_TRUE(isOrthogonal(U, 1e-6));

    // Check orthogonality of Vt
    EXPECT_TRUE(isOrthogonal(Vt, 1e-6));

    // Check reconstruction: U * diag(S) * Vt ≈ A
    auto A_recon = reconstructFromSvd(U, S, Vt);

    const double tol = 1e-6;
    for (Index j = 0; j < 3; ++j)
    {
        for (Index i = 0; i < 3; ++i)
        {
            EXPECT_NEAR(A_cpu(i, j), A_recon(i, j), tol)
                << "Mismatch at (" << i << ", " << j << ")";
        }
    }
}

// QR: decompose_3x2_Cpu — verify Q orthogonal, R upper triangular, A ≈ Q * R
TEST(QR, decompose_3x2_Cpu)
{
    // 3x2 matrix: A = [1 4; 2 5; 3 6] (column-major)
    DenseMatrix<double, Device::CPU> A(3, 2);
    A.setValue(0, 0, 1.0);
    A.setValue(1, 0, 2.0);
    A.setValue(2, 0, 3.0);
    A.setValue(0, 1, 4.0);
    A.setValue(1, 1, 5.0);
    A.setValue(2, 1, 6.0);

    auto [Q, R] = qr(A);

    // Q should be 3x3 (square, orthogonal)
    EXPECT_EQ(Q.rows(), 3);
    EXPECT_EQ(Q.cols(), 3);
    // R should be 3x2 (upper triangular)
    EXPECT_EQ(R.rows(), 3);
    EXPECT_EQ(R.cols(), 2);

    // Check Q is orthogonal: Q^T * Q ≈ I
    EXPECT_TRUE(isOrthogonal(Q, 1e-9));

    // Check R is upper triangular
    for (Index j = 0; j < 2; ++j)
    {
        for (Index i = j + 1; i < 3; ++i)
        {
            EXPECT_NEAR(R(i, j), 0.0, 1e-9)
                << "R not upper triangular at (" << i << ", " << j << ")";
        }
    }

    // Check A ≈ Q * R (3x3 * 3x2 = 3x2)
    const double tol = 1e-9;
    for (Index j = 0; j < 2; ++j)
    {
        for (Index i = 0; i < 3; ++i)
        {
            double qr_sum = 0.0;
            for (Index k = 0; k < 3; ++k)
            {
                qr_sum += Q(i, k) * R(k, j);
            }
            EXPECT_NEAR(A(i, j), qr_sum, tol)
                << "QR mismatch at (" << i << ", " << j << ")";
        }
    }
}

// QR: decompose_3x2_Gpu — same matrix on GPU via cuSOLVER
TEST(QR, decompose_3x2_Gpu)
{
    DenseMatrix<double, Device::CPU> A_cpu(3, 2);
    A_cpu.setValue(0, 0, 1.0);
    A_cpu.setValue(1, 0, 2.0);
    A_cpu.setValue(2, 0, 3.0);
    A_cpu.setValue(0, 1, 4.0);
    A_cpu.setValue(1, 1, 5.0);
    A_cpu.setValue(2, 1, 6.0);

    auto A_gpu = A_cpu.toGpu();
    auto [Q_gpu, R_gpu] = qr(A_gpu);

    auto Q = Q_gpu.toCpu();
    auto R = R_gpu.toCpu();

    // Q should be 3x3, R should be 3x2
    EXPECT_EQ(Q.rows(), 3);
    EXPECT_EQ(Q.cols(), 3);
    EXPECT_EQ(R.rows(), 3);
    EXPECT_EQ(R.cols(), 2);

    // Check Q is orthogonal
    EXPECT_TRUE(isOrthogonal(Q, 1e-6));

    // Check R is upper triangular
    for (Index j = 0; j < 2; ++j)
    {
        for (Index i = j + 1; i < 3; ++i)
        {
            EXPECT_NEAR(R(i, j), 0.0, 1e-6)
                << "R not upper triangular at (" << i << ", " << j << ")";
        }
    }

    // Check A ≈ Q * R
    const double tol = 1e-6;
    for (Index j = 0; j < 2; ++j)
    {
        for (Index i = 0; i < 3; ++i)
        {
            double qr_sum = 0.0;
            for (Index k = 0; k < 3; ++k)
            {
                qr_sum += Q(i, k) * R(k, j);
            }
            EXPECT_NEAR(A_cpu(i, j), qr_sum, tol)
                << "QR mismatch at (" << i << ", " << j << ")";
        }
    }
}

// Eigh: symmetric_2x2_Cpu — verify eigenvalues
TEST(Eigh, symmetric_2x2_Cpu)
{
    // Symmetric 2x2 matrix: A = [2 1; 1 2], eigenvalues = 3, 1
    DenseMatrix<double, Device::CPU> A(2, 2);
    A.setValue(0, 0, 2.0);
    A.setValue(1, 0, 1.0);
    A.setValue(0, 1, 1.0);
    A.setValue(1, 1, 2.0);

    auto eigvals = eigh(A);

    // Should return a column vector of eigenvalues
    EXPECT_EQ(eigvals.rows(), 2);
    EXPECT_EQ(eigvals.cols(), 1);

    // Eigenvalues should be sorted descending
    EXPECT_GE(eigvals(0, 0), eigvals(1, 0));

    // Check values: λ₁ = 3, λ₂ = 1
    EXPECT_NEAR(eigvals(0, 0), 3.0, 1e-9);
    EXPECT_NEAR(eigvals(1, 0), 1.0, 1e-9);
}

// Eigh: symmetric_2x2_Gpu — same matrix on GPU via cuSOLVER
TEST(Eigh, symmetric_2x2_Gpu)
{
    DenseMatrix<double, Device::CPU> A_cpu(2, 2);
    A_cpu.setValue(0, 0, 2.0);
    A_cpu.setValue(1, 0, 1.0);
    A_cpu.setValue(0, 1, 1.0);
    A_cpu.setValue(1, 1, 2.0);

    auto A_gpu = A_cpu.toGpu();
    auto eigvals_gpu = eigh(A_gpu);
    auto eigvals = eigvals_gpu.toCpu();

    // Should return a column vector of eigenvalues
    EXPECT_EQ(eigvals.rows(), 2);
    EXPECT_EQ(eigvals.cols(), 1);

    // Eigenvalues should be sorted descending
    EXPECT_GE(eigvals(0, 0), eigvals(1, 0));

    // Check values: λ₁ = 3, λ₂ = 1
    EXPECT_NEAR(eigvals(0, 0), 3.0, 1e-6);
    EXPECT_NEAR(eigvals(1, 0), 1.0, 1e-6);
}
