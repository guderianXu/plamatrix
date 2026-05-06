#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "plamatrix/ops/decomposition.h"

namespace plamatrix
{

namespace
{

/// Maximum number of Jacobi sweeps for convergence
constexpr int maxJacobiSweeps = 100;

/// Convergence tolerance for Jacobi sweeps
template <typename Scalar>
Scalar jacobiTolerance()
{
    return static_cast<Scalar>(1e-12);
}

/// Compute sign of value: returns 1.0 if val >= 0, -1.0 otherwise
template <typename Scalar>
Scalar sign(Scalar val)
{
    return (val >= Scalar(0)) ? Scalar(1) : Scalar(-1);
}

} // anonymous namespace

template <typename Scalar>
std::tuple<DenseMatrix<Scalar, Device::CPU>, DenseMatrix<Scalar, Device::CPU>, DenseMatrix<Scalar, Device::CPU>>
svd(const DenseMatrix<Scalar, Device::CPU>& A)
{
    Index m = A.rows();
    Index n = A.cols();

    if (m == 0 || n == 0)
    {
        throw std::runtime_error("SVD: input matrix has zero dimensions");
    }

    // U = copy of A (m x n), we work in-place on U (columns will become left singular vectors)
    DenseMatrix<Scalar, Device::CPU> U(m, n);
    for (Index j = 0; j < n; ++j)
    {
        for (Index i = 0; i < m; ++i)
        {
            U(i, j) = A(i, j);
        }
    }

    // V = identity (n x n), stored transposed for efficiency
    DenseMatrix<Scalar, Device::CPU> Vt(n, n);
    for (Index j = 0; j < n; ++j)
    {
        Vt(j, j) = Scalar(1);
    }

    // Singular values vector
    DenseMatrix<Scalar, Device::CPU> S(n, 1);

    // Jacobi sweeps
    Scalar epsilon = jacobiTolerance<Scalar>();

    for (int sweep = 0; sweep < maxJacobiSweeps; ++sweep)
    {
        Scalar max_off_diag = Scalar(0);

        for (Index j1 = 0; j1 < n; ++j1)
        {
            for (Index j2 = j1 + 1; j2 < n; ++j2)
            {
                // Compute column norms and dot product
                Scalar a = Scalar(0);
                Scalar b = Scalar(0);
                Scalar c = Scalar(0);

                for (Index i = 0; i < m; ++i)
                {
                    Scalar u_i_j1 = U(i, j1);
                    Scalar u_i_j2 = U(i, j2);
                    a += u_i_j1 * u_i_j1;
                    b += u_i_j2 * u_i_j2;
                    c += u_i_j1 * u_i_j2;
                }

                max_off_diag = std::max(max_off_diag, std::abs(c));

                // Skip if off-diagonal element is negligible
                Scalar scale = a * b;
                if (scale > Scalar(0))
                {
                    if (std::abs(c) / std::sqrt(scale) < epsilon)
                    {
                        continue;
                    }
                }

                // Compute Givens rotation to zero out c
                // Standard Jacobi: tau = (a - b) / (2*c), where a,b are diagonal entries
                Scalar tau = (a - b) / (Scalar(2) * c);
                Scalar t = sign(tau) / (std::abs(tau) + std::sqrt(Scalar(1) + tau * tau));
                Scalar cs = Scalar(1) / std::sqrt(Scalar(1) + t * t);
                Scalar sn = cs * t;

                // Apply rotation to columns j1, j2 of U
                for (Index i = 0; i < m; ++i)
                {
                    Scalar u_i_j1 = U(i, j1);
                    Scalar u_i_j2 = U(i, j2);
                    U(i, j1) = cs * u_i_j1 + sn * u_i_j2;
                    U(i, j2) = -sn * u_i_j1 + cs * u_i_j2;
                }

                // Apply rotation to rows j1, j2 of Vt
                for (Index k = 0; k < n; ++k)
                {
                    Scalar v_j1_k = Vt(j1, k);
                    Scalar v_j2_k = Vt(j2, k);
                    Vt(j1, k) = cs * v_j1_k + sn * v_j2_k;
                    Vt(j2, k) = -sn * v_j1_k + cs * v_j2_k;
                }
            }
        }

        // Check convergence
        if (max_off_diag < epsilon)
        {
            break;
        }
    }

    // Compute singular values = column norms of U
    for (Index j = 0; j < n; ++j)
    {
        Scalar norm = Scalar(0);
        for (Index i = 0; i < m; ++i)
        {
            Scalar val = U(i, j);
            norm += val * val;
        }
        S(j, 0) = std::sqrt(norm);
    }

    // Normalize columns of U (divide by singular values)
    for (Index j = 0; j < n; ++j)
    {
        Scalar sj = S(j, 0);
        if (sj > epsilon)
        {
            Scalar inv_sj = Scalar(1) / sj;
            for (Index i = 0; i < m; ++i)
            {
                U(i, j) *= inv_sj;
            }
        }
        else
        {
            // Zero singular value: set column to zero (already normalized)
            for (Index i = 0; i < m; ++i)
            {
                U(i, j) = Scalar(0);
            }
        }
    }

    // Sort singular values in descending order, permute U and Vt accordingly
    // Create index array and sort by singular value
    std::vector<Index> perm(n);
    for (Index j = 0; j < n; ++j)
    {
        perm[j] = j;
    }
    std::sort(perm.begin(), perm.end(), [&S](Index a, Index b) {
        return S(a, 0) > S(b, 0);
    });

    // Apply permutation to S, U, Vt
    // We need to create temporary copies since the permutation may have cycles
    DenseMatrix<Scalar, Device::CPU> S_sorted(n, 1);
    DenseMatrix<Scalar, Device::CPU> U_sorted(m, n);
    DenseMatrix<Scalar, Device::CPU> Vt_sorted(n, n);

    for (Index j = 0; j < n; ++j)
    {
        Index src = perm[j];
        S_sorted(j, 0) = S(src, 0);
        for (Index i = 0; i < m; ++i)
        {
            U_sorted(i, j) = U(i, src);
        }
        for (Index k = 0; k < n; ++k)
        {
            Vt_sorted(j, k) = Vt(src, k);
        }
    }

    return {std::move(U_sorted), std::move(S_sorted), std::move(Vt_sorted)};
}

// Explicit template instantiations
template std::tuple<DenseMatrix<float, Device::CPU>, DenseMatrix<float, Device::CPU>, DenseMatrix<float, Device::CPU>>
svd(const DenseMatrix<float, Device::CPU>&);
template std::tuple<DenseMatrix<double, Device::CPU>, DenseMatrix<double, Device::CPU>, DenseMatrix<double, Device::CPU>>
svd(const DenseMatrix<double, Device::CPU>&);

} // namespace plamatrix
