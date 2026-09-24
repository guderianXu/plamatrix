#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "plamatrix/dense/qr.h"

namespace plamatrix::v1
{
    /// Real generalized Schur decomposition computed by orthogonal equivalence transformations.
    /// Produces Q^T A Z = S and Q^T B Z = T, with S quasi-upper-triangular and T upper-triangular.
    template <typename MatrixType> class RealQZ;

    template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    class RealQZ<Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>>
    {
        static_assert(std::is_floating_point_v<Scalar>, "RealQZ requires a real floating-point scalar");

    public:
        using MatrixType = Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>;

        RealQZ() = default;
        explicit RealQZ(Index size) : _s(size, size), _t(size, size), _q(size, size), _z(size, size)
        {
        }
        RealQZ(const MatrixType& first, const MatrixType& second, bool computeQz = true)
        {
            compute(first, second, computeQz);
        }

        RealQZ& compute(const MatrixType& first, const MatrixType& second, bool computeQz = true)
        {
            _initialized = true;
            _computeQz = computeQz;
            _iterations = 0;
            if (first.rows() != first.cols() || second.rows() != second.cols() || first.rows() != second.rows() ||
                first.rows() == 0 || !first.allFinite() || !second.allFinite())
            {
                _info = InvalidInput;
                return *this;
            }
            if (internal::currentExecutionSettings().policy == internal::ExecutionPolicy::GpuRequired)
                throw internal::Error(internal::ErrorCode::UnsupportedOperation,
                                      "RealQZ is unavailable on the selected GPU backend",
                                      internal::currentExecutionSettings().preferredGpu);

            const Index size = first.rows();
            const HouseholderQR<MatrixType> qr(second);
            _q = qr.householderQ();
            _t = (_q.transpose() * second).eval();
            _s = (_q.transpose() * first).eval();
            _z = MatrixType::Identity(size, size);
            for (Index row = 1; row < size; ++row)
                for (Index col = 0; col < row; ++col)
                    _t(row, col) = Scalar{};

            reduceToHessenbergTriangular();
            iterate();
            return *this;
        }

        const MatrixType& matrixS() const
        {
            requireInitialized();
            return _s;
        }
        const MatrixType& matrixT() const
        {
            requireInitialized();
            return _t;
        }
        const MatrixType& matrixQ() const
        {
            requireVectors();
            return _q;
        }
        const MatrixType& matrixZ() const
        {
            requireVectors();
            return _z;
        }
        ComputationInfo info() const noexcept
        {
            return _info;
        }
        Index iterations() const noexcept
        {
            return _iterations;
        }
        RealQZ& setMaxIterations(Index value)
        {
            if (value <= 0)
                throw std::invalid_argument("RealQZ maximum iterations must be positive");
            _maxIterations = value;
            return *this;
        }

    private:
        struct Rotation
        {
            Scalar c = Scalar{1};
            Scalar s = Scalar{};
        };

        static Rotation leftRotation(Scalar first, Scalar second)
        {
            const Scalar norm = std::hypot(first, second);
            return norm == Scalar{} ? Rotation{} : Rotation{first / norm, second / norm};
        }
        static Rotation rightRotationToKillFirst(Scalar first, Scalar second)
        {
            const Scalar norm = std::hypot(first, second);
            return norm == Scalar{} ? Rotation{} : Rotation{second / norm, -first / norm};
        }
        static void
        applyLeftTranspose(MatrixType& matrix, Index first, Index second, const Rotation& rotation, Index beginColumn)
        {
            for (Index col = beginColumn; col < matrix.cols(); ++col)
            {
                const Scalar upper = matrix(first, col);
                const Scalar lower = matrix(second, col);
                matrix(first, col) = rotation.c * upper + rotation.s * lower;
                matrix(second, col) = -rotation.s * upper + rotation.c * lower;
            }
        }
        static void applyRight(MatrixType& matrix, Index first, Index second, const Rotation& rotation, Index endRow)
        {
            for (Index row = 0; row < endRow; ++row)
            {
                const Scalar left = matrix(row, first);
                const Scalar right = matrix(row, second);
                matrix(row, first) = rotation.c * left + rotation.s * right;
                matrix(row, second) = -rotation.s * left + rotation.c * right;
            }
        }
        void accumulateQ(Index first, Index second, const Rotation& rotation)
        {
            if (_computeQz)
                applyRight(_q, first, second, rotation, _q.rows());
        }
        void accumulateZ(Index first, Index second, const Rotation& rotation)
        {
            if (_computeQz)
                applyRight(_z, first, second, rotation, _z.rows());
        }

        void reduceToHessenbergTriangular()
        {
            const Index size = _s.rows();
            for (Index col = 0; col + 2 < size; ++col)
            {
                for (Index row = size - 1; row >= col + 2; --row)
                {
                    const Rotation left = leftRotation(_s(row - 1, col), _s(row, col));
                    applyLeftTranspose(_s, row - 1, row, left, col);
                    applyLeftTranspose(_t, row - 1, row, left, row - 1);
                    accumulateQ(row - 1, row, left);
                    _s(row, col) = Scalar{};

                    const Rotation right = rightRotationToKillFirst(_t(row, row - 1), _t(row, row));
                    applyRight(_s, row - 1, row, right, size);
                    applyRight(_t, row - 1, row, right, row + 1);
                    accumulateZ(row - 1, row, right);
                    _t(row, row - 1) = Scalar{};
                }
            }
        }

        Scalar shift(Index lower) const
        {
            const Index upper = lower - 1;
            const Scalar s00 = _s(upper, upper);
            const Scalar s01 = _s(upper, lower);
            const Scalar s10 = _s(lower, upper);
            const Scalar s11 = _s(lower, lower);
            const Scalar t00 = _t(upper, upper);
            const Scalar t01 = _t(upper, lower);
            const Scalar t11 = _t(lower, lower);
            const Scalar quadratic = t00 * t11;
            const Scalar linear = -s00 * t11 - s11 * t00 + s10 * t01;
            const Scalar constant = s00 * s11 - s01 * s10;
            const Scalar scale = std::abs(quadratic) + std::abs(linear) + std::abs(constant);
            const Scalar tiny = std::numeric_limits<Scalar>::epsilon() * std::max(Scalar{1}, scale);
            if (std::abs(quadratic) <= tiny)
                return std::abs(linear) <= tiny ? Scalar{} : -constant / linear;
            const Scalar discriminant = linear * linear - Scalar{4} * quadratic * constant;
            if (discriminant < Scalar{})
                return -linear / (Scalar{2} * quadratic);
            const Scalar root = std::sqrt(discriminant);
            const Scalar first = (-linear + root) / (Scalar{2} * quadratic);
            const Scalar second = (-linear - root) / (Scalar{2} * quadratic);
            if (std::abs(t11) <= tiny)
                return std::abs(first) < std::abs(second) ? first : second;
            const Scalar estimate = s11 / t11;
            return std::abs(first - estimate) < std::abs(second - estimate) ? first : second;
        }

        void qzSweep(Index first, Index last, Scalar value)
        {
            Scalar x = _s(first, first) - value * _t(first, first);
            Scalar y = _s(first + 1, first);
            for (Index index = first; index < last; ++index)
            {
                const Rotation left = leftRotation(x, y);
                applyLeftTranspose(_s, index, index + 1, left, std::max(first, index - 1));
                applyLeftTranspose(_t, index, index + 1, left, index);
                accumulateQ(index, index + 1, left);
                if (index > first)
                    _s(index + 1, index - 1) = Scalar{};

                const Rotation right = rightRotationToKillFirst(_t(index + 1, index), _t(index + 1, index + 1));
                applyRight(_s, index, index + 1, right, std::min(last + 2, _s.rows()));
                applyRight(_t, index, index + 1, right, index + 2);
                accumulateZ(index, index + 1, right);
                _t(index + 1, index) = Scalar{};
                if (index + 1 < last)
                {
                    x = _s(index + 1, index);
                    y = _s(index + 2, index);
                }
            }
        }

        void isolateInfiniteRoot(Index last)
        {
            const Rotation right = rightRotationToKillFirst(_s(last, last - 1), _s(last, last));
            applyRight(_s, last - 1, last, right, _s.rows());
            applyRight(_t, last - 1, last, right, _t.rows());
            accumulateZ(last - 1, last, right);
            _s(last, last - 1) = Scalar{};
            _t(last, last - 1) = Scalar{};
        }

        void iterate()
        {
            const Scalar epsilon = std::numeric_limits<Scalar>::epsilon();
            Scalar normS{};
            Scalar normT{};
            for (Index col = 0; col < _s.cols(); ++col)
                for (Index row = 0; row < _s.rows(); ++row)
                {
                    normS += std::abs(_s(row, col));
                    normT += std::abs(_t(row, col));
                }
            Index last = _s.rows() - 1;
            Index localIterations = 0;
            while (last > 0)
            {
                const Scalar scale = std::abs(_s(last - 1, last - 1)) + std::abs(_s(last, last));
                if (std::abs(_s(last, last - 1)) <= epsilon * std::max(normS, scale))
                {
                    _s(last, last - 1) = Scalar{};
                    --last;
                    localIterations = 0;
                    continue;
                }
                Index first = last - 1;
                while (first > 0)
                {
                    const Scalar localScale = std::abs(_s(first - 1, first - 1)) + std::abs(_s(first, first));
                    if (std::abs(_s(first, first - 1)) <= epsilon * std::max(normS, localScale))
                    {
                        _s(first, first - 1) = Scalar{};
                        break;
                    }
                    --first;
                }
                if (first == last - 1)
                {
                    last -= 2;
                    localIterations = 0;
                    continue;
                }
                const Scalar tTolerance = epsilon * std::max(normT, Scalar{1});
                if (std::abs(_t(last, last)) <= tTolerance)
                {
                    isolateInfiniteRoot(last);
                    --last;
                    localIterations = 0;
                    continue;
                }
                if (localIterations >= _maxIterations)
                {
                    _info = NoConvergence;
                    return;
                }
                qzSweep(first, last, shift(last));
                ++localIterations;
                ++_iterations;
            }
            for (Index row = 1; row < _t.rows(); ++row)
                for (Index col = 0; col < row; ++col)
                    _t(row, col) = Scalar{};
            _info = Success;
        }

        void requireInitialized() const
        {
            if (!_initialized)
                throw std::logic_error("RealQZ is not initialized");
        }
        void requireVectors() const
        {
            requireInitialized();
            if (!_computeQz)
                throw std::logic_error("RealQZ was computed without Q and Z");
        }

        MatrixType _s;
        MatrixType _t;
        MatrixType _q;
        MatrixType _z;
        ComputationInfo _info = InvalidInput;
        Index _maxIterations = 400;
        Index _iterations = 0;
        bool _initialized = false;
        bool _computeQz = true;
    };
} // namespace plamatrix::v1
