#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <omp.h>

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
    return Scalar(8) * std::numeric_limits<Scalar>::epsilon();
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
    for (int pass = 0; pass < 2; ++pass)
    {
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

template <typename Scalar>
void applyOneSidedJacobiPair(DenseMatrix<Scalar, Device::CPU>& u,
                             DenseMatrix<Scalar, Device::CPU>& vt,
                             Index first,
                             Index second,
                             Scalar epsilon,
                             Scalar* maximum_correlation)
{
    Scalar first_norm = Scalar(0);
    Scalar second_norm = Scalar(0);
    Scalar cross = Scalar(0);
    for (Index row = 0; row < u.rows(); ++row)
    {
        const Scalar left = u(row, first);
        const Scalar right = u(row, second);
        first_norm += left * left;
        second_norm += right * right;
        cross += left * right;
    }
    const Scalar scale = first_norm * second_norm;
    if (scale <= std::numeric_limits<Scalar>::min())
    {
        return;
    }
    const Scalar correlation = std::abs(cross) / std::sqrt(scale);
    *maximum_correlation = std::max(*maximum_correlation, correlation);
    if (correlation <= epsilon)
    {
        return;
    }
    const Scalar tau = (first_norm - second_norm) / (Scalar(2) * cross);
    const Scalar tangent = sign(tau) /
        (std::abs(tau) + std::sqrt(Scalar(1) + tau * tau));
    const Scalar cosine = Scalar(1) / std::sqrt(Scalar(1) + tangent * tangent);
    const Scalar sine = cosine * tangent;
    for (Index row = 0; row < u.rows(); ++row)
    {
        const Scalar left = u(row, first);
        const Scalar right = u(row, second);
        u(row, first) = cosine * left + sine * right;
        u(row, second) = -sine * left + cosine * right;
    }
    for (Index column = 0; column < vt.cols(); ++column)
    {
        const Scalar left = vt(first, column);
        const Scalar right = vt(second, column);
        vt(first, column) = cosine * left + sine * right;
        vt(second, column) = -sine * left + cosine * right;
    }
}

template <typename Scalar>
void parallelRoundRobinJacobi(DenseMatrix<Scalar, Device::CPU>& u,
                              DenseMatrix<Scalar, Device::CPU>& vt,
                              Scalar epsilon)
{
    const Index column_count = u.cols();
    const Index participant_count = column_count + column_count % 2;
    std::vector<Index> participants(static_cast<std::size_t>(participant_count), -1);
    for (Index column = 0; column < column_count; ++column)
    {
        participants[static_cast<std::size_t>(column)] = column;
    }
    for (int sweep = 0; sweep < maxJacobiSweeps; ++sweep)
    {
        Scalar maximum_correlation = Scalar(0);
        for (Index round = 0; round < participant_count - 1; ++round)
        {
            Scalar round_maximum = Scalar(0);
            #pragma omp parallel for schedule(static) reduction(max:round_maximum) \
                if(column_count >= 64 && !omp_in_parallel())
            for (Index pair = 0; pair < participant_count / 2; ++pair)
            {
                const Index first = participants[static_cast<std::size_t>(pair)];
                const Index second = participants[static_cast<std::size_t>(
                    participant_count - 1 - pair)];
                if (first >= 0 && second >= 0)
                {
                    Scalar pair_maximum = Scalar(0);
                    applyOneSidedJacobiPair(
                        u, vt, first, second, epsilon, &pair_maximum);
                    round_maximum = std::max(round_maximum, pair_maximum);
                }
            }
            maximum_correlation = std::max(maximum_correlation, round_maximum);
            const Index last = participants.back();
            for (Index index = participant_count - 1; index > 1; --index)
            {
                participants[static_cast<std::size_t>(index)] =
                    participants[static_cast<std::size_t>(index - 1)];
            }
            participants[1] = last;
        }
        if (maximum_correlation <= epsilon)
        {
            break;
        }
    }
}

template <typename Scalar>
std::vector<Scalar> symmetricEigenvaluesHouseholderQl(
    const DenseMatrix<Scalar, Device::CPU>& input)
{
    const Index dimension = input.rows();
    std::vector<Scalar> matrix(static_cast<std::size_t>(dimension * dimension));
    const auto at = [&](Index row, Index column) -> Scalar&
    {
        return matrix[static_cast<std::size_t>(row * dimension + column)];
    };
    for (Index row = 0; row < dimension; ++row)
    {
        for (Index column = 0; column < dimension; ++column)
        {
            at(row, column) = input(row, column);
        }
    }
    std::vector<Scalar> diagonal(static_cast<std::size_t>(dimension));
    std::vector<Scalar> off_diagonal(static_cast<std::size_t>(dimension), Scalar(0));
    std::vector<Scalar> reflector(static_cast<std::size_t>(dimension));
    std::vector<Scalar> product(static_cast<std::size_t>(dimension));
    for (Index column = 0; column + 2 < dimension; ++column)
    {
        const Index begin = column + 1;
        Scalar norm = Scalar(0);
        for (Index row = begin; row < dimension; ++row)
        {
            norm = std::hypot(norm, at(row, column));
        }
        if (norm <= std::numeric_limits<Scalar>::min())
        {
            continue;
        }
        const Scalar transformed = -std::copysign(norm, at(begin, column));
        off_diagonal[static_cast<std::size_t>(begin)] = transformed;
        std::fill(reflector.begin(), reflector.end(), Scalar(0));
        for (Index row = begin; row < dimension; ++row)
        {
            reflector[static_cast<std::size_t>(row)] = at(row, column);
        }
        reflector[static_cast<std::size_t>(begin)] -= transformed;
        Scalar reflector_norm = Scalar(0);
        for (Index row = begin; row < dimension; ++row)
        {
            const Scalar value = reflector[static_cast<std::size_t>(row)];
            reflector_norm += value * value;
        }
        const Scalar beta = Scalar(2) / reflector_norm;
        std::fill(product.begin(), product.end(), Scalar(0));
        for (Index row = begin; row < dimension; ++row)
        {
            for (Index inner = begin; inner < dimension; ++inner)
            {
                product[static_cast<std::size_t>(row)] +=
                    beta * at(row, inner) * reflector[static_cast<std::size_t>(inner)];
            }
        }
        Scalar correction = Scalar(0);
        for (Index row = begin; row < dimension; ++row)
        {
            correction += reflector[static_cast<std::size_t>(row)] *
                          product[static_cast<std::size_t>(row)];
        }
        correction *= -Scalar(0.5) * beta;
        for (Index row = begin; row < dimension; ++row)
        {
            product[static_cast<std::size_t>(row)] +=
                correction * reflector[static_cast<std::size_t>(row)];
        }
        for (Index row = begin; row < dimension; ++row)
        {
            for (Index inner = begin; inner <= row; ++inner)
            {
                at(row, inner) -=
                    reflector[static_cast<std::size_t>(row)] *
                        product[static_cast<std::size_t>(inner)] +
                    product[static_cast<std::size_t>(row)] *
                        reflector[static_cast<std::size_t>(inner)];
                at(inner, row) = at(row, inner);
            }
        }
    }
    for (Index index = 0; index < dimension; ++index)
    {
        diagonal[static_cast<std::size_t>(index)] = at(index, index);
        if (index > 0 && off_diagonal[static_cast<std::size_t>(index)] == Scalar(0))
        {
            off_diagonal[static_cast<std::size_t>(index)] = at(index, index - 1);
        }
    }
    for (Index index = 1; index < dimension; ++index)
    {
        off_diagonal[static_cast<std::size_t>(index - 1)] =
            off_diagonal[static_cast<std::size_t>(index)];
    }
    off_diagonal.back() = Scalar(0);

    const Scalar epsilon = std::numeric_limits<Scalar>::epsilon();
    for (Index left = 0; left < dimension; ++left)
    {
        int iterations = 0;
        while (true)
        {
            Index right = left;
            while (right + 1 < dimension)
            {
                const Scalar scale = std::abs(diagonal[static_cast<std::size_t>(right)]) +
                                     std::abs(diagonal[static_cast<std::size_t>(right + 1)]);
                if (std::abs(off_diagonal[static_cast<std::size_t>(right)]) <=
                    epsilon * std::max(Scalar(1), scale))
                {
                    break;
                }
                ++right;
            }
            if (right == left)
            {
                break;
            }
            if (++iterations > 128)
            {
                throw std::runtime_error("Eigh: implicit QL iteration did not converge");
            }
            Scalar shift = (diagonal[static_cast<std::size_t>(left + 1)] -
                            diagonal[static_cast<std::size_t>(left)]) /
                           (Scalar(2) * off_diagonal[static_cast<std::size_t>(left)]);
            Scalar root = std::hypot(shift, Scalar(1));
            shift = diagonal[static_cast<std::size_t>(right)] -
                    diagonal[static_cast<std::size_t>(left)] +
                    off_diagonal[static_cast<std::size_t>(left)] /
                        (shift + std::copysign(root, shift));
            Scalar sine = Scalar(1);
            Scalar cosine = Scalar(1);
            Scalar correction = Scalar(0);
            for (Index offset = 0; offset < right - left; ++offset)
            {
                const Index index = right - 1 - offset;
                const Scalar left_value = sine *
                    off_diagonal[static_cast<std::size_t>(index)];
                const Scalar right_value = cosine *
                    off_diagonal[static_cast<std::size_t>(index)];
                if (std::abs(left_value) >= std::abs(shift))
                {
                    cosine = shift / left_value;
                    root = std::hypot(cosine, Scalar(1));
                    off_diagonal[static_cast<std::size_t>(index + 1)] = left_value * root;
                    sine = Scalar(1) / root;
                    cosine *= sine;
                }
                else
                {
                    sine = left_value / shift;
                    root = std::hypot(sine, Scalar(1));
                    off_diagonal[static_cast<std::size_t>(index + 1)] = shift * root;
                    cosine = Scalar(1) / root;
                    sine *= cosine;
                }
                shift = diagonal[static_cast<std::size_t>(index + 1)] - correction;
                root = (diagonal[static_cast<std::size_t>(index)] - shift) * sine +
                       Scalar(2) * cosine * right_value;
                correction = sine * root;
                diagonal[static_cast<std::size_t>(index + 1)] = shift + correction;
                shift = cosine * root - right_value;
            }
            diagonal[static_cast<std::size_t>(left)] -= correction;
            off_diagonal[static_cast<std::size_t>(left)] = shift;
            off_diagonal[static_cast<std::size_t>(right)] = Scalar(0);
        }
    }
    std::sort(diagonal.begin(), diagonal.end(), std::greater<Scalar>());
    return diagonal;
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

    if (n >= 64)
    {
        parallelRoundRobinJacobi(U, Vt, epsilon);
    }
    else
    {
        for (int sweep = 0; sweep < maxJacobiSweeps; ++sweep)
        {
            Scalar max_correlation = Scalar(0);
            for (Index first = 0; first < n; ++first)
            {
                for (Index second = first + 1; second < n; ++second)
                {
                    applyOneSidedJacobiPair(
                        U, Vt, first, second, epsilon, &max_correlation);
                }
            }
            if (max_correlation <= epsilon)
            {
                break;
            }
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
}

// Explicit template instantiations
#ifdef PLAMATRIX_USE_FLOAT
template std::tuple<DenseMatrix<float, Device::CPU>, DenseMatrix<float, Device::CPU>, DenseMatrix<float, Device::CPU>>
svd(const DenseMatrix<float, Device::CPU>&);
#endif

#ifdef PLAMATRIX_USE_DOUBLE
template std::tuple<DenseMatrix<double, Device::CPU>, DenseMatrix<double, Device::CPU>, DenseMatrix<double, Device::CPU>>
svd(const DenseMatrix<double, Device::CPU>&);
#endif

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
    Scalar matrix_scale = Scalar(0);
    for (Index column = 0; column < n; ++column)
    {
        for (Index row = 0; row < m; ++row)
        {
            matrix_scale = std::max(matrix_scale, std::abs(R(row, column)));
        }
    }
    const Scalar epsilon = std::numeric_limits<Scalar>::epsilon() *
                           std::max(Scalar(1), matrix_scale) *
                           static_cast<Scalar>(std::max(m, n));

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
            norm_x = std::hypot(norm_x, R(i, k));
        }

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
        #pragma omp parallel for schedule(static) \
            if(n - k >= 64 && !omp_in_parallel())
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

        #pragma omp parallel for schedule(static) \
            if(m >= 128 && !omp_in_parallel())
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

    const auto eigenvalues = symmetricEigenvaluesHouseholderQl(A);
    DenseMatrix<Scalar, Device::CPU> eigvals(n, 1);
    for (Index i = 0; i < n; ++i)
    {
        eigvals(i, 0) = eigenvalues[static_cast<std::size_t>(i)];
    }

    return eigvals;
}

// Explicit template instantiations for qr
#ifdef PLAMATRIX_USE_FLOAT
template std::tuple<DenseMatrix<float, Device::CPU>, DenseMatrix<float, Device::CPU>>
qr(const DenseMatrix<float, Device::CPU>&);
#endif

#ifdef PLAMATRIX_USE_DOUBLE
template std::tuple<DenseMatrix<double, Device::CPU>, DenseMatrix<double, Device::CPU>>
qr(const DenseMatrix<double, Device::CPU>&);
#endif

// Explicit template instantiations for eigh
#ifdef PLAMATRIX_USE_FLOAT
template DenseMatrix<float, Device::CPU> eigh(const DenseMatrix<float, Device::CPU>&);
#endif

#ifdef PLAMATRIX_USE_DOUBLE
template DenseMatrix<double, Device::CPU> eigh(const DenseMatrix<double, Device::CPU>&);
#endif

} // namespace plamatrix
