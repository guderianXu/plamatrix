#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <vector>

#include "plamatrix/dense/schur.h"

namespace plamatrix::v1
{
    template <typename Real, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    class ComplexEigenSolver<Matrix<std::complex<Real>, Rows, Cols, Options, MaxRows, MaxCols>>
    {
    public:
        using Complex = std::complex<Real>;
        using MatrixType = Matrix<Complex, Rows, Cols, Options, MaxRows, MaxCols>;
        using EigenvalueType = Matrix<Complex, Cols, 1, ColMajor, MaxCols, 1>;
        using EigenvectorsType = MatrixType;

        ComplexEigenSolver() = default;
        explicit ComplexEigenSolver(Index size) : _values(size, 1), _vectors(size, size)
        {
        }
        explicit ComplexEigenSolver(const MatrixType& matrix, bool computeEigenvectors = true)
        {
            compute(matrix, computeEigenvectors);
        }

        ComplexEigenSolver& compute(const MatrixType& matrix, bool computeEigenvectors = true)
        {
            _info = InvalidInput;
            _vectorsComputed = computeEigenvectors;
            if (matrix.rows() != matrix.cols() || matrix.rows() == 0)
            {
                return *this;
            }
            const Index n = matrix.rows();
            const auto at = [n](Index row, Index col) { return static_cast<std::size_t>(row + col * n); };
            std::vector<Complex> original(static_cast<std::size_t>(n * n));
            Real scale{};
            for (Index col = 0; col < n; ++col)
            {
                for (Index row = 0; row < n; ++row)
                {
                    const Complex value = matrix(row, col);
                    if (!std::isfinite(value.real()) || !std::isfinite(value.imag()))
                        return *this;
                    original[at(row, col)] = value;
                    scale = std::max(scale, std::abs(value));
                }
            }
            ComplexSchur<MatrixType> schur(matrix, false);
            _info = schur.info();
            if (_info != Success)
            {
                return *this;
            }
            _values.resize(n, 1);
            _vectors.resize(n, n);
            for (Index index = 0; index < n; ++index)
            {
                _values(index) = schur.matrixT()(index, index);
            }
            if (!computeEigenvectors)
            {
                return *this;
            }

            const Real perturbation = std::sqrt(std::numeric_limits<Real>::epsilon()) * std::max(Real{1}, scale);
            for (Index eigen = 0; eigen < n; ++eigen)
            {
                std::vector<Complex> vector(static_cast<std::size_t>(n), Complex{1});
                for (int iteration = 0; iteration < 12; ++iteration)
                {
                    auto shifted = original;
                    for (Index index = 0; index < n; ++index)
                    {
                        shifted[at(index, index)] -= _values(eigen) + Complex{perturbation};
                    }
                    if (!decomposition_detail::solveComplexSystem(shifted, vector, n))
                    {
                        _info = NumericalIssue;
                        return *this;
                    }
                    Real norm{};
                    for (const Complex value : vector)
                        norm = std::hypot(norm, std::abs(value));
                    if (!(norm > Real{}))
                    {
                        _info = NumericalIssue;
                        return *this;
                    }
                    for (Complex& value : vector)
                        value /= norm;
                }
                for (Index row = 0; row < n; ++row)
                {
                    _vectors(row, eigen) = vector[static_cast<std::size_t>(row)];
                }
            }
            return *this;
        }

        const EigenvalueType& eigenvalues() const noexcept
        {
            return _values;
        }
        const EigenvectorsType& eigenvectors() const
        {
            if (!_vectorsComputed)
                throw std::logic_error("eigenvectors were not requested");
            return _vectors;
        }
        ComputationInfo info() const noexcept
        {
            return _info;
        }

    private:
        EigenvalueType _values;
        EigenvectorsType _vectors;
        ComputationInfo _info = InvalidInput;
        bool _vectorsComputed = false;
    };
} // namespace plamatrix::v1
