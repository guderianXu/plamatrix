#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "plamatrix/dense/permutation_matrix.h"
#include "plamatrix/dense/svd.h"

namespace plamatrix::v1
{
    template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    class CompleteOrthogonalDecomposition<Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>>
        : public SolverBase<CompleteOrthogonalDecomposition<Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>>>
    {
        static_assert(std::is_floating_point_v<Scalar>,
                      "CompleteOrthogonalDecomposition requires a real floating-point scalar");

    public:
        using MatrixType = Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>;
        using PermutationType = PermutationMatrix<Cols, MaxCols, DefaultPermutationIndex>;

        CompleteOrthogonalDecomposition() = default;
        explicit CompleteOrthogonalDecomposition(Index rows, Index cols) : _matrixQTZ(rows, cols), _svd(rows, cols)
        {
        }
        explicit CompleteOrthogonalDecomposition(const MatrixType& matrix)
        {
            compute(matrix);
        }

        CompleteOrthogonalDecomposition& compute(const MatrixType& matrix)
        {
            _svd.compute(matrix, ComputeFullU | ComputeFullV);
            _info = _svd.info();
            if (_info != Success)
            {
                return *this;
            }
            _rows = matrix.rows();
            _cols = matrix.cols();
            _matrixQTZ.resize(_rows, _cols);
            _matrixQTZ.setZero();
            for (Index index = 0; index < _svd.singularValues().size(); ++index)
            {
                _matrixQTZ(index, index) = _svd.singularValues()(index);
            }
            std::vector<DefaultPermutationIndex> indices(static_cast<std::size_t>(_cols));
            std::iota(indices.begin(), indices.end(), DefaultPermutationIndex{});
            _permutation = PermutationType(indices);
            return *this;
        }

        CompleteOrthogonalDecomposition& setThreshold(Scalar value)
        {
            if (!(value >= Scalar{}) || !std::isfinite(value))
            {
                throw std::invalid_argument(
                    "CompleteOrthogonalDecomposition threshold must be finite and non-negative");
            }
            _svd.setThreshold(value);
            return *this;
        }
        Scalar threshold() const
        {
            return _svd.threshold();
        }
        Index rank() const
        {
            return _svd.rank();
        }
        Index dimensionOfKernel() const
        {
            return _cols - rank();
        }
        bool isInjective() const
        {
            return rank() == _cols;
        }
        bool isSurjective() const
        {
            return rank() == _rows;
        }
        bool isInvertible() const
        {
            return _rows == _cols && isInjective();
        }
        ComputationInfo info() const noexcept
        {
            return _info;
        }

        const MatrixType& matrixQTZ() const noexcept
        {
            return _matrixQTZ;
        }
        const PermutationType& colsPermutation() const noexcept
        {
            return _permutation;
        }
        auto householderQ() const
        {
            return _svd.matrixU();
        }
        auto matrixZ() const
        {
            return _svd.matrixV().transpose().eval();
        }

        template <typename Rhs> auto _solve(const Rhs& right) const
        {
            if (_info != Success)
            {
                throw std::logic_error("CompleteOrthogonalDecomposition has not been computed successfully");
            }
            return _svd.solve(right);
        }
        template <typename Node> auto solve(const DenseExpression<Node>& right) const
        {
            return _solve(right.eval());
        }
        template <typename Rhs> auto solve(const MatrixBase<Rhs>& right) const
        {
            return _solve(right.derived());
        }

        Matrix<Scalar, Cols, Rows, (Options & RowMajor) ? ColMajor : RowMajor, MaxCols, MaxRows> pseudoInverse() const
        {
            using Result = Matrix<Scalar, Cols, Rows, (Options & RowMajor) ? ColMajor : RowMajor, MaxCols, MaxRows>;
            Result result(_cols, _rows);
            result.setZero();
            const auto& u = _svd.matrixU();
            const auto& v = _svd.matrixV();
            for (Index component = 0; component < _svd.singularValues().size(); ++component)
            {
                const Scalar singular = _svd.singularValues()(component);
                if (singular <= threshold())
                    continue;
                const Scalar inverse = Scalar{1} / singular;
                for (Index col = 0; col < _rows; ++col)
                {
                    for (Index row = 0; row < _cols; ++row)
                    {
                        result(row, col) += v(row, component) * inverse * u(col, component);
                    }
                }
            }
            return result;
        }

        Scalar absDeterminant() const
        {
            if (_rows != _cols)
                throw std::logic_error("absDeterminant requires a square matrix");
            Scalar result{1};
            for (Index index = 0; index < _svd.singularValues().size(); ++index)
            {
                result *= _svd.singularValues()(index);
            }
            return result;
        }
        Scalar logAbsDeterminant() const
        {
            if (_rows != _cols)
                throw std::logic_error("logAbsDeterminant requires a square matrix");
            Scalar result{};
            for (Index index = 0; index < _svd.singularValues().size(); ++index)
            {
                result += std::log(_svd.singularValues()(index));
            }
            return result;
        }

    private:
        JacobiSVD<MatrixType, ComputeFullU | ComputeFullV> _svd;
        MatrixType _matrixQTZ;
        PermutationType _permutation;
        Index _rows = 0;
        Index _cols = 0;
        ComputationInfo _info = InvalidInput;
    };

    template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    CompleteOrthogonalDecomposition<Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>>
    Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>::completeOrthogonalDecomposition() const
    {
        return CompleteOrthogonalDecomposition<Matrix>(*this);
    }
} // namespace plamatrix::v1
