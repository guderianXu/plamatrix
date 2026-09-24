#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

#include "plamatrix/dense/matrix.h"

namespace plamatrix::v1
{
    namespace structural_detail
    {
        template <typename Scalar, typename MatrixType>
        Scalar householderSimilarity(MatrixType& matrix, Matrix<Scalar, Dynamic, Dynamic>& q, Index column)
        {
            const Index n = matrix.rows();
            const Index first = column + 1;
            Scalar norm{};
            for (Index row = first; row < n; ++row)
            {
                norm = std::hypot(norm, matrix(row, column));
            }
            if (norm == Scalar{})
            {
                return Scalar{};
            }
            std::vector<Scalar> vector(static_cast<std::size_t>(n - first));
            for (Index row = first; row < n; ++row)
            {
                vector[static_cast<std::size_t>(row - first)] = matrix(row, column);
            }
            vector.front() += std::copysign(norm, vector.front());
            Scalar squared_norm{};
            for (const Scalar value : vector)
            {
                squared_norm += value * value;
            }
            if (squared_norm == Scalar{})
            {
                return Scalar{};
            }
            const Scalar tau = Scalar{2} / squared_norm;

            for (Index col = column; col < n; ++col)
            {
                Scalar projection{};
                for (Index row = first; row < n; ++row)
                {
                    projection += vector[static_cast<std::size_t>(row - first)] * matrix(row, col);
                }
                projection *= tau;
                for (Index row = first; row < n; ++row)
                {
                    matrix(row, col) -= vector[static_cast<std::size_t>(row - first)] * projection;
                }
            }
            for (Index row = 0; row < n; ++row)
            {
                Scalar projection{};
                for (Index col = first; col < n; ++col)
                {
                    projection += matrix(row, col) * vector[static_cast<std::size_t>(col - first)];
                }
                projection *= tau;
                for (Index col = first; col < n; ++col)
                {
                    matrix(row, col) -= projection * vector[static_cast<std::size_t>(col - first)];
                }
            }
            for (Index row = 0; row < n; ++row)
            {
                Scalar projection{};
                for (Index col = first; col < n; ++col)
                {
                    projection += q(row, col) * vector[static_cast<std::size_t>(col - first)];
                }
                projection *= tau;
                for (Index col = first; col < n; ++col)
                {
                    q(row, col) -= projection * vector[static_cast<std::size_t>(col - first)];
                }
            }
            for (Index row = first + 1; row < n; ++row)
            {
                matrix(row, column) = Scalar{};
            }
            return tau;
        }
    } // namespace structural_detail

    template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    class HessenbergDecomposition<Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>>
    {
        static_assert(std::is_floating_point_v<Scalar>, "HessenbergDecomposition requires a real scalar");

    public:
        using MatrixType = Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>;
        using CoeffVectorType = Matrix<Scalar, (Rows == Dynamic ? Dynamic : Rows - 1), 1>;

        HessenbergDecomposition() = default;
        explicit HessenbergDecomposition(Index size)
            : _h(size, size), _q(size, size), _coefficients(std::max<Index>(0, size - 1), 1)
        {
        }
        explicit HessenbergDecomposition(const MatrixType& matrix)
        {
            compute(matrix);
        }

        HessenbergDecomposition& compute(const MatrixType& matrix)
        {
            _info = InvalidInput;
            if (matrix.rows() != matrix.cols() || matrix.rows() == 0 || !matrix.allFinite())
            {
                return *this;
            }
            _h = matrix;
            _q = Matrix<Scalar, Dynamic, Dynamic>::Identity(matrix.rows(), matrix.rows());
            _coefficients.resize(std::max<Index>(0, matrix.rows() - 1), 1);
            _coefficients.setZero();
            for (Index column = 0; column + 2 < matrix.rows(); ++column)
            {
                _coefficients(column) = structural_detail::householderSimilarity<Scalar>(_h, _q, column);
            }
            _info = Success;
            return *this;
        }

        const MatrixType& matrixH() const noexcept
        {
            return _h;
        }
        const CoeffVectorType& householderCoefficients() const noexcept
        {
            return _coefficients;
        }
        MatrixType matrixQ() const
        {
            return MatrixType(_q);
        }
        ComputationInfo info() const noexcept
        {
            return _info;
        }

    private:
        MatrixType _h;
        Matrix<Scalar, Dynamic, Dynamic> _q;
        CoeffVectorType _coefficients;
        ComputationInfo _info = InvalidInput;
    };

    template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    class Tridiagonalization<Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>>
    {
        static_assert(std::is_floating_point_v<Scalar>, "Tridiagonalization requires a real scalar");

    public:
        using MatrixType = Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>;
        using DiagonalType = Matrix<Scalar, Rows, 1, ColMajor, MaxRows, 1>;
        using SubDiagonalType = Matrix<Scalar, (Rows == Dynamic ? Dynamic : Rows - 1), 1>;

        Tridiagonalization() = default;
        explicit Tridiagonalization(Index size)
            : _matrix(size, size), _q(size, size), _diagonal(size, 1), _subDiagonal(std::max<Index>(0, size - 1), 1)
        {
        }
        explicit Tridiagonalization(const MatrixType& matrix)
        {
            compute(matrix);
        }

        Tridiagonalization& compute(const MatrixType& matrix)
        {
            HessenbergDecomposition<MatrixType> hessenberg(matrix);
            _info = hessenberg.info();
            if (_info != Success)
            {
                return *this;
            }
            _matrix = hessenberg.matrixH();
            _q = hessenberg.matrixQ();
            _diagonal.resize(matrix.rows(), 1);
            _subDiagonal.resize(std::max<Index>(0, matrix.rows() - 1), 1);
            for (Index col = 0; col < matrix.cols(); ++col)
            {
                for (Index row = 0; row < matrix.rows(); ++row)
                {
                    if (std::abs(row - col) > 1)
                    {
                        _matrix(row, col) = Scalar{};
                    }
                }
                _diagonal(col) = _matrix(col, col);
                if (col + 1 < matrix.rows())
                {
                    const Scalar value = Scalar{0.5} * (_matrix(col + 1, col) + _matrix(col, col + 1));
                    _matrix(col + 1, col) = value;
                    _matrix(col, col + 1) = value;
                    _subDiagonal(col) = value;
                }
            }
            return *this;
        }

        const MatrixType& matrixT() const noexcept
        {
            return _matrix;
        }
        const MatrixType& packedMatrix() const noexcept
        {
            return _matrix;
        }
        const DiagonalType& diagonal() const noexcept
        {
            return _diagonal;
        }
        const SubDiagonalType& subDiagonal() const noexcept
        {
            return _subDiagonal;
        }
        MatrixType matrixQ() const
        {
            return _q;
        }
        ComputationInfo info() const noexcept
        {
            return _info;
        }

    private:
        MatrixType _matrix;
        MatrixType _q;
        DiagonalType _diagonal;
        SubDiagonalType _subDiagonal;
        ComputationInfo _info = InvalidInput;
    };
} // namespace plamatrix::v1
