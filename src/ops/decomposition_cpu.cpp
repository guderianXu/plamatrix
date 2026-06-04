#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "plamatrix/ops/decomposition.h"

#ifdef PLAMATRIX_WITH_LAPACK
#include "fortran_linalg.h"
#endif

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

template <typename Scalar>
bool normalizeColumnCandidate(DenseMatrix<Scalar, Device::CPU>& U,
                              Index col,
                              std::vector<Scalar>& candidate,
                              Scalar epsilon)
{
    Index rows = U.rows();
    for (Index prev = 0; prev < col; ++prev)
    {
        Scalar dot = Scalar(0);
        for (Index i = 0; i < rows; ++i)
        {
            dot += candidate[static_cast<std::size_t>(i)] * U(i, prev);
        }
        for (Index i = 0; i < rows; ++i)
        {
            candidate[static_cast<std::size_t>(i)] -= dot * U(i, prev);
        }
    }

    Scalar norm = Scalar(0);
    for (Index i = 0; i < rows; ++i)
    {
        Scalar v = candidate[static_cast<std::size_t>(i)];
        norm += v * v;
    }
    norm = std::sqrt(norm);
    if (norm <= epsilon)
    {
        return false;
    }

    Scalar inv_norm = Scalar(1) / norm;
    for (Index i = 0; i < rows; ++i)
    {
        U(i, col) = candidate[static_cast<std::size_t>(i)] * inv_norm;
    }
    return true;
}

template <typename Scalar>
void setOrthonormalColumn(DenseMatrix<Scalar, Device::CPU>& U,
                          Index col,
                          std::vector<Scalar> candidate,
                          Scalar epsilon)
{
    if (normalizeColumnCandidate(U, col, candidate, epsilon))
    {
        return;
    }

    Index rows = U.rows();
    for (Index basis = 0; basis < rows; ++basis)
    {
        std::fill(candidate.begin(), candidate.end(), Scalar(0));
        candidate[static_cast<std::size_t>(basis)] = Scalar(1);
        if (normalizeColumnCandidate(U, col, candidate, epsilon))
        {
            return;
        }
    }

    throw std::runtime_error("SVD: failed to construct orthonormal U basis");
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

#ifdef PLAMATRIX_WITH_LAPACK
    int m_int = detail::checkedLapackInt(m, "SVD m");
    int n_int = detail::checkedLapackInt(n, "SVD n");
    Index min_mn = (m < n) ? m : n;

    DenseMatrix<Scalar, Device::CPU> A_work(m, n);
    for (Index j = 0; j < n; ++j)
    {
        for (Index i = 0; i < m; ++i)
        {
            A_work(i, j) = A(i, j);
        }
    }

    DenseMatrix<Scalar, Device::CPU> U_full(m, m);
    DenseMatrix<Scalar, Device::CPU> S_compact(min_mn, 1);
    DenseMatrix<Scalar, Device::CPU> Vt(n, n);

    detail::fortranGesvd(m_int, n_int, A_work.data(), S_compact.data(), U_full.data(), Vt.data());
    return {std::move(U_full), std::move(S_compact), std::move(Vt)};
#else

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

                // Skip if either column pair cannot produce a stable rotation.
                Scalar scale = a * b;
                if (scale <= epsilon * epsilon)
                {
                    continue;
                }
                if (std::abs(c) / std::sqrt(scale) < epsilon)
                {
                    continue;
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

    Index compact_singular_count = (m < n) ? m : n;
    DenseMatrix<Scalar, Device::CPU> S_compact(compact_singular_count, 1);
    for (Index i = 0; i < compact_singular_count; ++i)
    {
        S_compact(i, 0) = S_sorted(i, 0);
    }

    DenseMatrix<Scalar, Device::CPU> U_full(m, m);
    for (Index col = 0; col < m; ++col)
    {
        std::vector<Scalar> candidate(static_cast<std::size_t>(m), Scalar(0));
        if (col < n)
        {
            for (Index row = 0; row < m; ++row)
            {
                candidate[static_cast<std::size_t>(row)] = U_sorted(row, col);
            }
        }
        setOrthonormalColumn(U_full, col, std::move(candidate), epsilon);
    }

    return {std::move(U_full), std::move(S_compact), std::move(Vt_sorted)};
#endif
}

// Explicit template instantiations
template std::tuple<DenseMatrix<float, Device::CPU>, DenseMatrix<float, Device::CPU>, DenseMatrix<float, Device::CPU>>
svd(const DenseMatrix<float, Device::CPU>&);
template std::tuple<DenseMatrix<double, Device::CPU>, DenseMatrix<double, Device::CPU>, DenseMatrix<double, Device::CPU>>
svd(const DenseMatrix<double, Device::CPU>&);

template <typename Scalar>
std::tuple<DenseMatrix<Scalar, Device::CPU>, DenseMatrix<Scalar, Device::CPU>>
qr(const DenseMatrix<Scalar, Device::CPU>& A)
{
    Index m = A.rows();
    Index n = A.cols();

    if (m == 0 || n == 0)
    {
        throw std::runtime_error("QR: input matrix has zero dimensions");
    }

    // R = copy of A, work in place
    DenseMatrix<Scalar, Device::CPU> R(m, n);
    for (Index j = 0; j < n; ++j)
    {
        for (Index i = 0; i < m; ++i)
        {
            R(i, j) = A(i, j);
        }
    }

    Index p = (m < n) ? m : n;  // min(m, n)
    Scalar epsilon = static_cast<Scalar>(1e-15);

    // Store tau for each reflector
    std::vector<Scalar> tau(p, Scalar(0));

    // Householder reflections
    for (Index k = 0; k < p; ++k)
    {
        Index len = m - k;

        // Compute norm of column x = R[k:m, k]
        Scalar norm_x = Scalar(0);
        for (Index i = k; i < m; ++i)
        {
            norm_x += R(i, k) * R(i, k);
        }
        norm_x = std::sqrt(norm_x);

        if (norm_x < epsilon)
        {
            tau[k] = Scalar(0);
            continue;
        }

        Scalar x0 = R(k, k);
        Scalar alpha = -sign(x0) * norm_x;
        Scalar v0 = x0 - alpha;

        // Store alpha on diagonal (this becomes R(k,k))
        R(k, k) = alpha;

        // Normalize v and store below diagonal (v[i] = x[i] / v0)
        for (Index i = 1; i < static_cast<Index>(len); ++i)
        {
            R(k + i, k) = R(k + i, k) / v0;
        }

        // tau = (alpha - x0) / alpha
        tau[k] = (alpha - x0) / alpha;

        // Apply Householder to trailing submatrix
        for (Index j = k + 1; j < n; ++j)
        {
            // dot = v^T * R[k:m, j]  (v[0] = 1)
            Scalar dot = R(k, j);
            for (Index i = 1; i < static_cast<Index>(len); ++i)
            {
                dot += R(k + i, k) * R(k + i, j);
            }

            // R[k:m, j] -= tau * dot * v
            R(k, j) -= tau[k] * dot;
            for (Index i = 1; i < static_cast<Index>(len); ++i)
            {
                R(k + i, j) -= tau[k] * dot * R(k + i, k);
            }
        }
    }

    // Build Q = I
    DenseMatrix<Scalar, Device::CPU> Q(m, m);
    for (Index i = 0; i < m; ++i)
    {
        Q(i, i) = Scalar(1);
    }

    // Apply H_0, H_1, ..., H_{p-1} to Q from the RIGHT
    for (Index k = 0; k < p; ++k)
    {
        Scalar tau_k = tau[k];
        if (tau_k == Scalar(0))
        {
            continue;
        }

        Index len = m - k;

        for (Index r = 0; r < m; ++r)
        {
            // dot = v^T * Q[r, k:m]  (v[0] = 1)
            Scalar dot = Q(r, k);
            for (Index i = 1; i < static_cast<Index>(len); ++i)
            {
                dot += R(k + i, k) * Q(r, k + i);
            }

            // Q[r, k:m] -= tau * dot * v
            Q(r, k) -= tau_k * dot;
            for (Index i = 1; i < static_cast<Index>(len); ++i)
            {
                Q(r, k + i) -= tau_k * dot * R(k + i, k);
            }
        }
    }

    // Zero out the lower triangular part of R (where Householder vectors were stored)
    for (Index j = 0; j < p; ++j)
    {
        for (Index i = j + 1; i < m; ++i)
        {
            R(i, j) = Scalar(0);
        }
    }

    return {std::move(Q), std::move(R)};
}

template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> eigh(const DenseMatrix<Scalar, Device::CPU>& A)
{
    Index n = A.rows();

    if (n == 0)
    {
        throw std::runtime_error("Eigh: input matrix has zero dimensions");
    }

    if (A.cols() != n)
    {
        throw std::runtime_error("Eigh: input matrix must be square");
    }

#ifdef PLAMATRIX_WITH_LAPACK
    int n_int = detail::checkedLapackInt(n, "Eigh n");
    DenseMatrix<Scalar, Device::CPU> A_work(n, n);
    for (Index j = 0; j < n; ++j)
    {
        for (Index i = 0; i < n; ++i)
        {
            A_work(i, j) = A(i, j);
        }
    }

    std::vector<Scalar> eigenvalues(static_cast<std::size_t>(n));
    detail::fortranSyev(n_int, A_work.data(), eigenvalues.data());

    DenseMatrix<Scalar, Device::CPU> eigvals(n, 1);
    for (Index i = 0; i < n; ++i)
    {
        eigvals(i, 0) = eigenvalues[static_cast<std::size_t>(n - 1 - i)];
    }
    return eigvals;
#else

    // Copy A to working matrix
    DenseMatrix<Scalar, Device::CPU> A_work(n, n);
    for (Index j = 0; j < n; ++j)
    {
        for (Index i = 0; i < n; ++i)
        {
            A_work(i, j) = A(i, j);
        }
    }

    Scalar epsilon = static_cast<Scalar>(1e-12);
    constexpr int maxSweeps = 100;

    for (int sweep = 0; sweep < maxSweeps; ++sweep)
    {
        Scalar max_off = Scalar(0);

        for (Index p = 0; p < n - 1; ++p)
        {
            for (Index q = p + 1; q < n; ++q)
            {
                Scalar apq = A_work(p, q);
                if (std::abs(apq) < epsilon)
                {
                    continue;
                }

                max_off = std::max(max_off, std::abs(apq));

                Scalar app = A_work(p, p);
                Scalar aqq = A_work(q, q);

                // Compute Jacobi rotation
                Scalar tau_val = (aqq - app) / (Scalar(2) * apq);
                Scalar t = -sign(tau_val) / (std::abs(tau_val) + std::sqrt(Scalar(1) + tau_val * tau_val));
                Scalar c = Scalar(1) / std::sqrt(Scalar(1) + t * t);
                Scalar s = c * t;

                // Update diagonal entries
                A_work(p, p) = c * c * app + s * s * aqq - Scalar(2) * c * s * apq;
                A_work(q, q) = s * s * app + c * c * aqq + Scalar(2) * c * s * apq;
                A_work(p, q) = Scalar(0);
                A_work(q, p) = Scalar(0);

                // Update off-diagonal elements in rows/cols p and q
                for (Index k = 0; k < n; ++k)
                {
                    if (k == p || k == q)
                    {
                        continue;
                    }

                    Scalar akp = A_work(k, p);
                    Scalar akq = A_work(k, q);
                    A_work(k, p) = c * akp - s * akq;
                    A_work(p, k) = A_work(k, p);
                    A_work(k, q) = s * akp + c * akq;
                    A_work(q, k) = A_work(k, q);
                }
            }
        }

        if (max_off < epsilon)
        {
            break;
        }
    }

    // Extract eigenvalues from diagonal
    DenseMatrix<Scalar, Device::CPU> eigvals(n, 1);
    for (Index i = 0; i < n; ++i)
    {
        eigvals(i, 0) = A_work(i, i);
    }

    // Sort eigenvalues in descending order
    for (Index i = 0; i < n - 1; ++i)
    {
        Index max_idx = i;
        for (Index j = i + 1; j < n; ++j)
        {
            if (eigvals(j, 0) > eigvals(max_idx, 0))
            {
                max_idx = j;
            }
        }
        if (max_idx != i)
        {
            Scalar tmp = eigvals(i, 0);
            eigvals(i, 0) = eigvals(max_idx, 0);
            eigvals(max_idx, 0) = tmp;
        }
    }

    return eigvals;
#endif
}

// Explicit template instantiations for qr
template std::tuple<DenseMatrix<float, Device::CPU>, DenseMatrix<float, Device::CPU>>
qr(const DenseMatrix<float, Device::CPU>&);
template std::tuple<DenseMatrix<double, Device::CPU>, DenseMatrix<double, Device::CPU>>
qr(const DenseMatrix<double, Device::CPU>&);

// Explicit template instantiations for eigh
template DenseMatrix<float, Device::CPU> eigh(const DenseMatrix<float, Device::CPU>&);
template DenseMatrix<double, Device::CPU> eigh(const DenseMatrix<double, Device::CPU>&);

} // namespace plamatrix
