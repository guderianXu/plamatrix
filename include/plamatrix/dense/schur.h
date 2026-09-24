#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <type_traits>
#include <vector>

#include "plamatrix/dense/eigen_solvers.h"
#include "plamatrix/dense/matrix_structural_decompositions.h"
#include "plamatrix/dense/qr.h"

namespace plamatrix::v1
{
    template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    class RealSchur<Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>>
    {
        static_assert(std::is_floating_point_v<Scalar>, "RealSchur requires a real scalar");

    public:
        using MatrixType = Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>;
        RealSchur() = default;
        explicit RealSchur(Index size) : _matrixT(size, size), _matrixU(size, size)
        {
        }
        explicit RealSchur(const MatrixType& matrix, bool computeU = true)
        {
            compute(matrix, computeU);
        }

        template <typename HessMatrixType, typename OrthMatrixType>
        RealSchur&
        computeFromHessenberg(const HessMatrixType& matrixH, const OrthMatrixType& matrixQ, bool computeU = true)
        {
            if (matrixH.rows() != matrixH.cols() || matrixQ.rows() != matrixH.rows() ||
                matrixQ.cols() != matrixH.cols())
            {
                _info = InvalidInput;
                return *this;
            }
            const MatrixType reconstructed = (matrixQ * matrixH * matrixQ.transpose()).eval();
            return compute(reconstructed, computeU);
        }

        RealSchur& compute(const MatrixType& matrix, bool computeU = true)
        {
            _info = InvalidInput;
            _computeU = computeU;
            if (matrix.rows() != matrix.cols() || matrix.rows() == 0 || !matrix.allFinite())
            {
                return *this;
            }
            HessenbergDecomposition<MatrixType> hessenberg(matrix);
            if (hessenberg.info() != Success)
            {
                return *this;
            }
            _matrixT = hessenberg.matrixH();
            _matrixU = computeU ? hessenberg.matrixQ() : MatrixType::Identity(matrix.rows(), matrix.rows());
            const Index n = matrix.rows();
            const Index maximum = _maxIterations >= 0 ? _maxIterations : std::max<Index>(1, 80 * n);
            const Scalar epsilon = std::numeric_limits<Scalar>::epsilon();
            for (Index iteration = 0; iteration < maximum; ++iteration)
            {
                Scalar scale{};
                for (Index index = 0; index < n; ++index)
                {
                    scale = std::max(scale, std::abs(_matrixT(index, index)));
                }
                for (Index row = 1; row < n; ++row)
                {
                    const Scalar local = std::abs(_matrixT(row - 1, row - 1)) + std::abs(_matrixT(row, row));
                    if (std::abs(_matrixT(row, row - 1)) <= epsilon * Scalar{64} * std::max(Scalar{1}, local + scale))
                    {
                        _matrixT(row, row - 1) = Scalar{};
                    }
                }
                bool quasi_triangular = true;
                Index block_size = 1;
                for (Index row = 1; row < n; ++row)
                {
                    if (_matrixT(row, row - 1) != Scalar{})
                    {
                        ++block_size;
                        quasi_triangular = quasi_triangular && block_size <= 2;
                    }
                    else
                    {
                        block_size = 1;
                    }
                }
                if (quasi_triangular)
                {
                    _info = Success;
                    return *this;
                }

                Scalar shift = _matrixT(n - 1, n - 1);
                if (n > 1)
                {
                    const Scalar a = _matrixT(n - 2, n - 2);
                    const Scalar b = _matrixT(n - 2, n - 1);
                    const Scalar c = _matrixT(n - 1, n - 2);
                    const Scalar d = _matrixT(n - 1, n - 1);
                    const Scalar discriminant = (a - d) * (a - d) + Scalar{4} * b * c;
                    if (discriminant >= Scalar{})
                    {
                        const Scalar root = std::sqrt(discriminant);
                        const Scalar first = Scalar{0.5} * (a + d + root);
                        const Scalar second = Scalar{0.5} * (a + d - root);
                        shift = std::abs(first - d) < std::abs(second - d) ? first : second;
                    }
                    else
                    {
                        shift = Scalar{0.5} * (a + d);
                    }
                }
                Matrix<Scalar, Dynamic, Dynamic> shifted(_matrixT);
                for (Index index = 0; index < n; ++index)
                {
                    shifted(index, index) -= shift;
                }
                HouseholderQR<Matrix<Scalar, Dynamic, Dynamic>> qr(shifted);
                const auto q = qr.householderQ();
                Matrix<Scalar, Dynamic, Dynamic> r = Matrix<Scalar, Dynamic, Dynamic>::Zero(n, n);
                for (Index col = 0; col < n; ++col)
                {
                    for (Index row = 0; row <= col; ++row)
                    {
                        r(row, col) = qr.matrixQR()(row, col);
                    }
                }
                _matrixT = (r * q).eval();
                for (Index index = 0; index < n; ++index)
                {
                    _matrixT(index, index) += shift;
                }
                if (computeU)
                {
                    _matrixU = (_matrixU * q).eval();
                }
            }
            _info = NoConvergence;
            return *this;
        }

        RealSchur& setMaxIterations(Index value)
        {
            if (value < 0)
                throw std::invalid_argument("RealSchur maximum iterations must be non-negative");
            _maxIterations = value;
            return *this;
        }
        Index getMaxIterations() const noexcept
        {
            return _maxIterations;
        }
        const MatrixType& matrixT() const noexcept
        {
            return _matrixT;
        }
        const MatrixType& matrixU() const
        {
            if (!_computeU)
                throw std::logic_error("RealSchur matrixU was not requested");
            return _matrixU;
        }
        ComputationInfo info() const noexcept
        {
            return _info;
        }

    private:
        MatrixType _matrixT;
        MatrixType _matrixU;
        Index _maxIterations = -1;
        ComputationInfo _info = InvalidInput;
        bool _computeU = false;
    };

    template <typename Real, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    class ComplexSchur<Matrix<std::complex<Real>, Rows, Cols, Options, MaxRows, MaxCols>>
    {
    public:
        using Complex = std::complex<Real>;
        using MatrixType = Matrix<Complex, Rows, Cols, Options, MaxRows, MaxCols>;
        ComplexSchur() = default;
        explicit ComplexSchur(Index size) : _matrixT(size, size), _matrixU(size, size)
        {
        }
        explicit ComplexSchur(const MatrixType& matrix, bool computeU = true)
        {
            compute(matrix, computeU);
        }

        template <typename HessMatrixType, typename OrthMatrixType>
        ComplexSchur&
        computeFromHessenberg(const HessMatrixType& matrixH, const OrthMatrixType& matrixQ, bool computeU = true)
        {
            if (matrixH.rows() != matrixH.cols() || matrixQ.rows() != matrixH.rows() ||
                matrixQ.cols() != matrixH.cols())
            {
                _info = InvalidInput;
                return *this;
            }
            const MatrixType reconstructed = (matrixQ * matrixH * matrixQ.adjoint()).eval();
            return compute(reconstructed, computeU);
        }

        ComplexSchur& compute(const MatrixType& matrix, bool computeU = true)
        {
            _info = InvalidInput;
            _computeU = computeU;
            if (matrix.rows() != matrix.cols() || matrix.rows() == 0)
            {
                return *this;
            }
            const Index n = matrix.rows();
            const auto at = [n](Index row, Index col) { return static_cast<std::size_t>(row + col * n); };
            std::vector<Complex> work(static_cast<std::size_t>(n * n));
            std::vector<Complex> accumulated(static_cast<std::size_t>(n * n));
            for (Index col = 0; col < n; ++col)
            {
                for (Index row = 0; row < n; ++row)
                {
                    const Complex value = matrix(row, col);
                    if (!std::isfinite(value.real()) || !std::isfinite(value.imag()))
                        return *this;
                    work[at(row, col)] = value;
                    accumulated[at(row, col)] = row == col ? Complex{1} : Complex{};
                }
            }
            const Index maximum = _maxIterations >= 0 ? _maxIterations : std::max<Index>(1, 160 * n);
            const Real epsilon = std::numeric_limits<Real>::epsilon();
            for (Index iteration = 0; iteration < maximum; ++iteration)
            {
                Real off_diagonal{};
                Real scale{};
                for (Index col = 0; col < n; ++col)
                {
                    scale = std::max(scale, std::abs(work[at(col, col)]));
                    for (Index row = col + 1; row < n; ++row)
                    {
                        off_diagonal = std::max(off_diagonal, std::abs(work[at(row, col)]));
                    }
                }
                if (off_diagonal <= epsilon * Real{128} * std::max(Real{1}, scale))
                {
                    _info = Success;
                    break;
                }
                const Complex shift = work[at(n - 1, n - 1)];
                for (Index index = 0; index < n; ++index)
                    work[at(index, index)] -= shift;
                std::vector<Complex> q;
                std::vector<Complex> r;
                decomposition_detail::complexQr(work, n, q, r);
                work = decomposition_detail::multiplyComplex(r, q, n);
                for (Index index = 0; index < n; ++index)
                    work[at(index, index)] += shift;
                if (computeU)
                    accumulated = decomposition_detail::multiplyComplex(accumulated, q, n);
            }
            _matrixT.resize(n, n);
            _matrixU.resize(n, n);
            for (Index col = 0; col < n; ++col)
            {
                for (Index row = 0; row < n; ++row)
                {
                    _matrixT(row, col) = work[at(row, col)];
                    _matrixU(row, col) = accumulated[at(row, col)];
                }
            }
            if (_info == InvalidInput)
                _info = NoConvergence;
            return *this;
        }

        ComplexSchur& setMaxIterations(Index value)
        {
            if (value < 0)
                throw std::invalid_argument("ComplexSchur maximum iterations must be non-negative");
            _maxIterations = value;
            return *this;
        }
        Index getMaxIterations() const noexcept
        {
            return _maxIterations;
        }
        const MatrixType& matrixT() const noexcept
        {
            return _matrixT;
        }
        const MatrixType& matrixU() const
        {
            if (!_computeU)
                throw std::logic_error("ComplexSchur matrixU was not requested");
            return _matrixU;
        }
        ComputationInfo info() const noexcept
        {
            return _info;
        }

    private:
        MatrixType _matrixT;
        MatrixType _matrixU;
        Index _maxIterations = -1;
        ComputationInfo _info = InvalidInput;
        bool _computeU = false;
    };
} // namespace plamatrix::v1
