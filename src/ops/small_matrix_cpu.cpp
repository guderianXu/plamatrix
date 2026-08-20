#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "plamatrix/ops/small_matrix.h"

namespace plamatrix
{
namespace
{

template <typename Scalar>
using Matrix3 = std::array<Scalar, 9>;

template <typename Scalar>
using Vector3 = std::array<Scalar, 3>;

constexpr int jacobiSweeps = 8;

template <typename Scalar>
Scalar& element(Matrix3<Scalar>& matrix, Index row, Index col)
{
    return matrix[static_cast<std::size_t>(row * 3 + col)];
}

template <typename Scalar>
Scalar element(const Matrix3<Scalar>& matrix, Index row, Index col)
{
    return matrix[static_cast<std::size_t>(row * 3 + col)];
}

template <typename Scalar>
void validateInput(const DenseMatrix<Scalar, Device::CPU>& compact_matrices)
{
    if (compact_matrices.cols() != 6)
    {
        std::ostringstream message;
        message << "symmetricEigh3x3Batched: expected an N x 6 compact matrix, got "
                << compact_matrices.rows() << " x " << compact_matrices.cols();
        throw std::invalid_argument(message.str());
    }

    for (Index row = 0; row < compact_matrices.rows(); ++row)
    {
        for (Index col = 0; col < compact_matrices.cols(); ++col)
        {
            if (!std::isfinite(compact_matrices(row, col)))
            {
                std::ostringstream message;
                message << "symmetricEigh3x3Batched: input must be finite; non-finite value at row "
                        << row << ", column " << col;
                throw std::invalid_argument(message.str());
            }
        }
    }
}

template <typename Scalar>
void applyJacobiRotation(Matrix3<Scalar>& matrix,
                         Matrix3<Scalar>& eigenvectors,
                         Index p,
                         Index q)
{
    const Scalar apq = element(matrix, p, q);
    const Scalar app = element(matrix, p, p);
    const Scalar aqq = element(matrix, q, q);
    const Scalar rotation_scale = std::max({std::abs(app), std::abs(aqq), std::abs(apq)});
    if (rotation_scale == Scalar(0) || apq == Scalar(0))
    {
        return;
    }

    const Scalar scaled_app = app / rotation_scale;
    const Scalar scaled_aqq = aqq / rotation_scale;
    const Scalar scaled_apq = apq / rotation_scale;
    const Scalar x = scaled_aqq - scaled_app;
    const Scalar y = Scalar(2) * scaled_apq;
    if (y == Scalar(0))
    {
        return;
    }
    const Scalar tangent_magnitude = std::abs(y) / (std::abs(x) + std::hypot(x, y));
    Scalar tangent = std::copysign(tangent_magnitude, y);
    if (x < Scalar(0))
    {
        tangent = -tangent;
    }
    const Scalar cosine = Scalar(1) / std::sqrt(Scalar(1) + tangent * tangent);
    const Scalar sine = tangent * cosine;

    for (Index k = 0; k < 3; ++k)
    {
        if (k == p || k == q)
        {
            continue;
        }
        const Scalar akp = element(matrix, k, p);
        const Scalar akq = element(matrix, k, q);
        element(matrix, k, p) = cosine * akp - sine * akq;
        element(matrix, p, k) = element(matrix, k, p);
        element(matrix, k, q) = sine * akp + cosine * akq;
        element(matrix, q, k) = element(matrix, k, q);
    }

    element(matrix, p, p) = std::fma(-tangent, apq, app);
    element(matrix, q, q) = std::fma(tangent, apq, aqq);
    element(matrix, p, q) = Scalar(0);
    element(matrix, q, p) = Scalar(0);

    for (Index row = 0; row < 3; ++row)
    {
        const Scalar vip = element(eigenvectors, row, p);
        const Scalar viq = element(eigenvectors, row, q);
        element(eigenvectors, row, p) = cosine * vip - sine * viq;
        element(eigenvectors, row, q) = sine * vip + cosine * viq;
    }
}

template <typename Scalar>
Scalar dot(const Vector3<Scalar>& left, const Vector3<Scalar>& right)
{
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

template <typename Scalar>
bool normalize(Vector3<Scalar>& vector, Scalar minimum_norm)
{
    const Scalar norm = std::sqrt(dot(vector, vector));
    if (norm <= minimum_norm)
    {
        return false;
    }
    for (Scalar& component : vector)
    {
        component /= norm;
    }
    return true;
}

template <typename Scalar>
void canonicalizeSign(Vector3<Scalar>& vector)
{
    Index largest = 0;
    for (Index component = 1; component < 3; ++component)
    {
        if (std::abs(vector[static_cast<std::size_t>(component)]) >
            std::abs(vector[static_cast<std::size_t>(largest)]))
        {
            largest = component;
        }
    }
    if (vector[static_cast<std::size_t>(largest)] < Scalar(0))
    {
        for (Scalar& component : vector)
        {
            component = -component;
        }
    }
}

template <typename Scalar>
bool eigenvaluesRepeat(Scalar left, Scalar right)
{
    const Scalar scale = std::max({Scalar(1), std::abs(left), std::abs(right)});
    return std::abs(left - right) <= Scalar(256) * std::numeric_limits<Scalar>::epsilon() * scale;
}

template <typename Scalar>
void rebuildRepeatedGroup(Matrix3<Scalar>& eigenvectors, Index begin, Index end)
{
    Matrix3<Scalar> projector{};
    for (Index row = 0; row < 3; ++row)
    {
        for (Index col = 0; col < 3; ++col)
        {
            for (Index group_col = begin; group_col < end; ++group_col)
            {
                element(projector, row, col) += element(eigenvectors, row, group_col) *
                                                element(eigenvectors, col, group_col);
            }
        }
    }

    std::array<Vector3<Scalar>, 3> basis{};
    Index basis_count = 0;
    const Scalar minimum_norm = Scalar(64) * std::numeric_limits<Scalar>::epsilon();
    for (Index axis = 0; axis < 3 && basis_count < end - begin; ++axis)
    {
        Vector3<Scalar> candidate = {
            element(projector, 0, axis),
            element(projector, 1, axis),
            element(projector, 2, axis)
        };
        for (int pass = 0; pass < 2; ++pass)
        {
            for (Index previous = 0; previous < basis_count; ++previous)
            {
                const Scalar projection = dot(candidate, basis[static_cast<std::size_t>(previous)]);
                for (Index component = 0; component < 3; ++component)
                {
                    candidate[static_cast<std::size_t>(component)] -=
                        projection * basis[static_cast<std::size_t>(previous)][static_cast<std::size_t>(component)];
                }
            }
        }
        if (normalize(candidate, minimum_norm))
        {
            basis[static_cast<std::size_t>(basis_count++)] = candidate;
        }
    }

    if (basis_count != end - begin)
    {
        throw std::runtime_error(
            "symmetricEigh3x3Batched: failed to construct a repeated-eigenvalue basis");
    }
    for (Index group_col = begin; group_col < end; ++group_col)
    {
        const auto& vector = basis[static_cast<std::size_t>(group_col - begin)];
        for (Index row = 0; row < 3; ++row)
        {
            element(eigenvectors, row, group_col) = vector[static_cast<std::size_t>(row)];
        }
    }
}

template <typename Scalar>
void sortAndCanonicalize(Matrix3<Scalar>& eigenvectors, Vector3<Scalar>& eigenvalues)
{
    std::array<Index, 3> order = {0, 1, 2};
    std::stable_sort(order.begin(), order.end(), [&eigenvalues](Index left, Index right) {
        return eigenvalues[static_cast<std::size_t>(left)] < eigenvalues[static_cast<std::size_t>(right)];
    });

    const Vector3<Scalar> unsorted_values = eigenvalues;
    const Matrix3<Scalar> unsorted_vectors = eigenvectors;
    for (Index col = 0; col < 3; ++col)
    {
        const Index source = order[static_cast<std::size_t>(col)];
        eigenvalues[static_cast<std::size_t>(col)] = unsorted_values[static_cast<std::size_t>(source)];
        for (Index row = 0; row < 3; ++row)
        {
            element(eigenvectors, row, col) = element(unsorted_vectors, row, source);
        }
    }

    for (Index begin = 0; begin < 3;)
    {
        Index end = begin + 1;
        while (end < 3 && eigenvaluesRepeat(eigenvalues[static_cast<std::size_t>(begin)],
                                             eigenvalues[static_cast<std::size_t>(end)]))
        {
            ++end;
        }
        if (end - begin > 1)
        {
            rebuildRepeatedGroup(eigenvectors, begin, end);
        }
        begin = end;
    }

    for (Index col = 0; col < 3; ++col)
    {
        Vector3<Scalar> vector = {
            element(eigenvectors, 0, col),
            element(eigenvectors, 1, col),
            element(eigenvectors, 2, col)
        };
        if (!normalize(vector, Scalar(0)))
        {
            throw std::runtime_error("symmetricEigh3x3Batched: computed a zero eigenvector");
        }
        canonicalizeSign(vector);
        for (Index row = 0; row < 3; ++row)
        {
            element(eigenvectors, row, col) = vector[static_cast<std::size_t>(row)];
        }
    }
}

template <typename Scalar>
void decomposeCompact(const std::array<Scalar, 6>& compact,
                      Vector3<Scalar>* eigenvalues_output,
                      Matrix3<Scalar>* eigenvectors_output)
{
    Matrix3<Scalar> matrix = {
        compact[0], compact[1], compact[2],
        compact[1], compact[3], compact[4],
        compact[2], compact[4], compact[5]
    };
    Matrix3<Scalar> eigenvectors = {
        Scalar(1), Scalar(0), Scalar(0),
        Scalar(0), Scalar(1), Scalar(0),
        Scalar(0), Scalar(0), Scalar(1)
    };

    for (int sweep = 0; sweep < jacobiSweeps; ++sweep)
    {
        applyJacobiRotation(matrix, eigenvectors, 0, 1);
        applyJacobiRotation(matrix, eigenvectors, 0, 2);
        applyJacobiRotation(matrix, eigenvectors, 1, 2);
    }

    *eigenvalues_output = {
        element(matrix, 0, 0), element(matrix, 1, 1), element(matrix, 2, 2)
    };
    sortAndCanonicalize(eigenvectors, *eigenvalues_output);
    *eigenvectors_output = eigenvectors;
}

template <typename Scalar>
void decomposeRow(const DenseMatrix<Scalar, Device::CPU>& compact_matrices,
                  Index row,
                  DenseMatrix<Scalar, Device::CPU>& eigenvalues_output,
                  DenseMatrix<Scalar, Device::CPU>& eigenvectors_output)
{
    const std::array<Scalar, 6> compact = {
        compact_matrices(row, 0), compact_matrices(row, 1), compact_matrices(row, 2),
        compact_matrices(row, 3), compact_matrices(row, 4), compact_matrices(row, 5)};
    Vector3<Scalar> eigenvalues{};
    Matrix3<Scalar> eigenvectors{};
    decomposeCompact(compact, &eigenvalues, &eigenvectors);
    for (Index col = 0; col < 3; ++col)
    {
        eigenvalues_output(row, col) = eigenvalues[static_cast<std::size_t>(col)];
        for (Index component = 0; component < 3; ++component)
        {
            eigenvectors_output(row, col * 3 + component) = element(eigenvectors, component, col);
        }
    }
}

} // namespace

template <typename Scalar>
void symmetricEigh3x3(const std::array<Scalar, 6>& compact_matrix,
                      std::array<Scalar, 3>* eigenvalues,
                      std::array<Scalar, 9>* eigenvectors)
{
    if (!eigenvalues || !eigenvectors)
    {
        throw std::invalid_argument("symmetricEigh3x3: outputs must be non-null");
    }
    for (const Scalar value : compact_matrix)
    {
        if (!std::isfinite(value))
        {
            throw std::invalid_argument("symmetricEigh3x3: input must be finite");
        }
    }
    Matrix3<Scalar> internal_eigenvectors{};
    decomposeCompact(compact_matrix, eigenvalues, &internal_eigenvectors);
    for (Index column = 0; column < 3; ++column)
    {
        for (Index component = 0; component < 3; ++component)
        {
            (*eigenvectors)[static_cast<std::size_t>(column * 3 + component)] =
                element(internal_eigenvectors, component, column);
        }
    }
}

template <typename Scalar>
void svd3x3(const std::array<Scalar, 9>& matrix,
            std::array<Scalar, 9>* u,
            std::array<Scalar, 3>* singular_values,
            std::array<Scalar, 9>* vt)
{
    if (!u || !singular_values || !vt)
    {
        throw std::invalid_argument("svd3x3: outputs must be non-null");
    }
    for (const Scalar value : matrix)
    {
        if (!std::isfinite(value))
        {
            throw std::invalid_argument("svd3x3: input must be finite");
        }
    }

    std::array<Scalar, 6> normal{};
    normal[0] = matrix[0] * matrix[0] + matrix[3] * matrix[3] + matrix[6] * matrix[6];
    normal[1] = matrix[0] * matrix[1] + matrix[3] * matrix[4] + matrix[6] * matrix[7];
    normal[2] = matrix[0] * matrix[2] + matrix[3] * matrix[5] + matrix[6] * matrix[8];
    normal[3] = matrix[1] * matrix[1] + matrix[4] * matrix[4] + matrix[7] * matrix[7];
    normal[4] = matrix[1] * matrix[2] + matrix[4] * matrix[5] + matrix[7] * matrix[8];
    normal[5] = matrix[2] * matrix[2] + matrix[5] * matrix[5] + matrix[8] * matrix[8];

    Vector3<Scalar> eigenvalues{};
    Matrix3<Scalar> eigenvectors{};
    decomposeCompact(normal, &eigenvalues, &eigenvectors);
    Scalar maximum_singular = Scalar(0);
    for (Index column = 0; column < 3; ++column)
    {
        const Index source = 2 - column;
        (*singular_values)[static_cast<std::size_t>(column)] = std::sqrt(
            std::max(Scalar(0), eigenvalues[static_cast<std::size_t>(source)]));
        maximum_singular = std::max(
            maximum_singular, (*singular_values)[static_cast<std::size_t>(column)]);
        for (Index component = 0; component < 3; ++component)
        {
            (*vt)[static_cast<std::size_t>(column * 3 + component)] =
                element(eigenvectors, component, source);
        }
    }

    u->fill(Scalar(0));
    const Scalar threshold = std::max(
        Scalar(1), maximum_singular) * Scalar(128) * std::numeric_limits<Scalar>::epsilon();
    for (Index column = 0; column < 3; ++column)
    {
        Vector3<Scalar> candidate{};
        const Scalar singular = (*singular_values)[static_cast<std::size_t>(column)];
        if (singular > threshold)
        {
            for (Index row = 0; row < 3; ++row)
            {
                for (Index inner = 0; inner < 3; ++inner)
                {
                    candidate[static_cast<std::size_t>(row)] +=
                        matrix[static_cast<std::size_t>(row * 3 + inner)] *
                        (*vt)[static_cast<std::size_t>(column * 3 + inner)];
                }
                candidate[static_cast<std::size_t>(row)] /= singular;
            }
        }
        for (Index previous = 0; previous < column; ++previous)
        {
            Scalar projection = Scalar(0);
            for (Index row = 0; row < 3; ++row)
            {
                projection += candidate[static_cast<std::size_t>(row)] *
                              (*u)[static_cast<std::size_t>(row * 3 + previous)];
            }
            for (Index row = 0; row < 3; ++row)
            {
                candidate[static_cast<std::size_t>(row)] -=
                    projection * (*u)[static_cast<std::size_t>(row * 3 + previous)];
            }
        }
        if (!normalize(candidate, threshold))
        {
            for (Index axis = 0; axis < 3; ++axis)
            {
                candidate = {Scalar(0), Scalar(0), Scalar(0)};
                candidate[static_cast<std::size_t>(axis)] = Scalar(1);
                for (Index previous = 0; previous < column; ++previous)
                {
                    Scalar projection = Scalar(0);
                    for (Index row = 0; row < 3; ++row)
                    {
                        projection += candidate[static_cast<std::size_t>(row)] *
                                      (*u)[static_cast<std::size_t>(row * 3 + previous)];
                    }
                    for (Index row = 0; row < 3; ++row)
                    {
                        candidate[static_cast<std::size_t>(row)] -=
                            projection * (*u)[static_cast<std::size_t>(row * 3 + previous)];
                    }
                }
                if (normalize(candidate, threshold))
                {
                    break;
                }
            }
        }
        for (Index row = 0; row < 3; ++row)
        {
            (*u)[static_cast<std::size_t>(row * 3 + column)] =
                candidate[static_cast<std::size_t>(row)];
        }
    }
}

template <typename Scalar>
SymmetricEigh3x3Result<Scalar, Device::CPU> symmetricEigh3x3Batched(
    const DenseMatrix<Scalar, Device::CPU>& compact_matrices)
{
    validateInput(compact_matrices);

    SymmetricEigh3x3Result<Scalar, Device::CPU> result{
        DenseMatrix<Scalar, Device::CPU>(compact_matrices.rows(), 3),
        DenseMatrix<Scalar, Device::CPU>(compact_matrices.rows(), 9)
    };
    for (Index row = 0; row < compact_matrices.rows(); ++row)
    {
        decomposeRow(compact_matrices, row, result.eigenvalues, result.eigenvectors);
    }
    return result;
}

#ifdef PLAMATRIX_USE_FLOAT
template void symmetricEigh3x3(const std::array<float, 6>&,
                               std::array<float, 3>*,
                               std::array<float, 9>*);
template void svd3x3(const std::array<float, 9>&,
                     std::array<float, 9>*,
                     std::array<float, 3>*,
                     std::array<float, 9>*);
template SymmetricEigh3x3Result<float, Device::CPU> symmetricEigh3x3Batched(
    const DenseMatrix<float, Device::CPU>&);
#endif

#ifdef PLAMATRIX_USE_DOUBLE
template void symmetricEigh3x3(const std::array<double, 6>&,
                               std::array<double, 3>*,
                               std::array<double, 9>*);
template void svd3x3(const std::array<double, 9>&,
                     std::array<double, 9>*,
                     std::array<double, 3>*,
                     std::array<double, 9>*);
template SymmetricEigh3x3Result<double, Device::CPU> symmetricEigh3x3Batched(
    const DenseMatrix<double, Device::CPU>&);
#endif

} // namespace plamatrix
