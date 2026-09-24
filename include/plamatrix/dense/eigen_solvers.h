#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <numeric>
#include <vector>

#include "plamatrix/dense/cholesky_lu.h"
#include "plamatrix/dense/detail/decomposition_backend.h"
#include "plamatrix/dense/svd.h"
#include "plamatrix/dense/real_qz.h"

namespace plamatrix::v1
{
    namespace decomposition_detail
    {
        template <typename Scalar, typename MatrixType, typename ValuesType, typename VectorsType>
        ComputationInfo symmetricJacobi(const MatrixType& input, ValuesType& values, VectorsType& vectors)
        {
            const Index n = input.rows();
            MatrixType work(n, n);
            vectors = VectorsType::Identity(n, n);
            for (Index column = 0; column < n; ++column)
            {
                for (Index row = 0; row < n; ++row)
                {
                    work(row, column) = row >= column ? input(row, column) : input(column, row);
                }
            }
            const Scalar epsilon = std::numeric_limits<Scalar>::epsilon();
            bool converged = n < 2;
            for (Index sweep = 0; sweep < 64 * std::max<Index>(n, 1); ++sweep)
            {
                Index p = 0;
                Index q = 0;
                Scalar largest{};
                Scalar scale{};
                for (Index column = 0; column < n; ++column)
                {
                    scale = std::max(scale, std::abs(work(column, column)));
                    for (Index row = column + 1; row < n; ++row)
                    {
                        if (std::abs(work(row, column)) > largest)
                        {
                            largest = std::abs(work(row, column));
                            p = column;
                            q = row;
                        }
                    }
                }
                if (largest <= epsilon * static_cast<Scalar>(std::max<Index>(n, 1)) * std::max(Scalar{1}, scale))
                {
                    converged = true;
                    break;
                }
                const Scalar angle = Scalar{0.5} * std::atan2(Scalar{2} * work(q, p), work(q, q) - work(p, p));
                const Scalar cosine = std::cos(angle);
                const Scalar sine = std::sin(angle);
                for (Index index = 0; index < n; ++index)
                {
                    const Scalar first = work(index, p);
                    const Scalar second = work(index, q);
                    work(index, p) = cosine * first - sine * second;
                    work(index, q) = sine * first + cosine * second;
                }
                for (Index index = 0; index < n; ++index)
                {
                    const Scalar first = work(p, index);
                    const Scalar second = work(q, index);
                    work(p, index) = cosine * first - sine * second;
                    work(q, index) = sine * first + cosine * second;
                    const Scalar vector_first = vectors(index, p);
                    const Scalar vector_second = vectors(index, q);
                    vectors(index, p) = cosine * vector_first - sine * vector_second;
                    vectors(index, q) = sine * vector_first + cosine * vector_second;
                }
                work(p, q) = work(q, p) = Scalar{};
            }
            if (!converged)
            {
                return NoConvergence;
            }
            StaticOrDynamicVector<Index, MatrixType::RowsAtCompileTime> order(static_cast<std::size_t>(n));
            std::iota(order.begin(), order.end(), Index{});
            std::sort(order.begin(),
                      order.end(),
                      [&work](Index left, Index right) { return work(left, left) < work(right, right); });
            values.resize(n, 1);
            VectorsType sorted(n, n);
            for (Index column = 0; column < n; ++column)
            {
                values(column) = work(order[static_cast<std::size_t>(column)], order[static_cast<std::size_t>(column)]);
                for (Index row = 0; row < n; ++row)
                {
                    sorted(row, column) = vectors(row, order[static_cast<std::size_t>(column)]);
                }
            }
            vectors = std::move(sorted);
            return Success;
        }

        template <typename Complex>
        void complexQr(const std::vector<Complex>& input, Index n, std::vector<Complex>& q, std::vector<Complex>& r)
        {
            q.assign(static_cast<std::size_t>(n * n), Complex{});
            r.assign(static_cast<std::size_t>(n * n), Complex{});
            std::vector<Complex> column(static_cast<std::size_t>(n));
            const auto at = [n](Index row, Index column) { return static_cast<std::size_t>(row + column * n); };
            for (Index current = 0; current < n; ++current)
            {
                for (Index row = 0; row < n; ++row)
                    column[static_cast<std::size_t>(row)] = input[at(row, current)];
                for (Index previous = 0; previous < current; ++previous)
                {
                    Complex projection{};
                    for (Index row = 0; row < n; ++row)
                    {
                        projection += std::conj(q[at(row, previous)]) * column[static_cast<std::size_t>(row)];
                    }
                    r[at(previous, current)] = projection;
                    for (Index row = 0; row < n; ++row)
                    {
                        column[static_cast<std::size_t>(row)] -= projection * q[at(row, previous)];
                    }
                }
                // A second modified Gram-Schmidt pass keeps the accumulated Schur vectors unitary
                // when many shifted QR steps are performed.
                for (Index previous = 0; previous < current; ++previous)
                {
                    Complex correction{};
                    for (Index row = 0; row < n; ++row)
                    {
                        correction += std::conj(q[at(row, previous)]) * column[static_cast<std::size_t>(row)];
                    }
                    r[at(previous, current)] += correction;
                    for (Index row = 0; row < n; ++row)
                    {
                        column[static_cast<std::size_t>(row)] -= correction * q[at(row, previous)];
                    }
                }
                typename Complex::value_type norm{};
                for (const auto& value : column)
                    norm = std::hypot(norm, std::abs(value));
                if (norm <= std::numeric_limits<typename Complex::value_type>::epsilon())
                {
                    column.assign(static_cast<std::size_t>(n), Complex{});
                    column[static_cast<std::size_t>(current)] = Complex{1};
                    for (Index previous = 0; previous < current; ++previous)
                    {
                        Complex projection{};
                        for (Index row = 0; row < n; ++row)
                            projection += std::conj(q[at(row, previous)]) * column[static_cast<std::size_t>(row)];
                        for (Index row = 0; row < n; ++row)
                            column[static_cast<std::size_t>(row)] -= projection * q[at(row, previous)];
                    }
                    norm = {};
                    for (const auto& value : column)
                        norm = std::hypot(norm, std::abs(value));
                }
                r[at(current, current)] = Complex{norm};
                for (Index row = 0; row < n; ++row)
                    q[at(row, current)] = column[static_cast<std::size_t>(row)] / norm;
            }
        }

        template <typename Complex>
        std::vector<Complex>
        multiplyComplex(const std::vector<Complex>& left, const std::vector<Complex>& right, Index n)
        {
            std::vector<Complex> result(static_cast<std::size_t>(n * n));
            const auto at = [n](Index row, Index column) { return static_cast<std::size_t>(row + column * n); };
            for (Index column = 0; column < n; ++column)
                for (Index row = 0; row < n; ++row)
                    for (Index inner = 0; inner < n; ++inner)
                        result[at(row, column)] += left[at(row, inner)] * right[at(inner, column)];
            return result;
        }

        template <typename Complex>
        bool solveComplexSystem(std::vector<Complex> matrix, std::vector<Complex>& right, Index n)
        {
            const auto at = [n](Index row, Index column) { return static_cast<std::size_t>(row + column * n); };
            using Real = typename Complex::value_type;
            for (Index column = 0; column < n; ++column)
            {
                Index pivot = column;
                Real largest = std::abs(matrix[at(column, column)]);
                for (Index row = column + 1; row < n; ++row)
                {
                    if (std::abs(matrix[at(row, column)]) > largest)
                    {
                        largest = std::abs(matrix[at(row, column)]);
                        pivot = row;
                    }
                }
                if (!(largest > std::numeric_limits<Real>::epsilon()))
                    return false;
                if (pivot != column)
                {
                    for (Index inner = 0; inner < n; ++inner)
                        std::swap(matrix[at(column, inner)], matrix[at(pivot, inner)]);
                    std::swap(right[static_cast<std::size_t>(column)], right[static_cast<std::size_t>(pivot)]);
                }
                for (Index row = column + 1; row < n; ++row)
                {
                    const Complex factor = matrix[at(row, column)] / matrix[at(column, column)];
                    for (Index inner = column + 1; inner < n; ++inner)
                        matrix[at(row, inner)] -= factor * matrix[at(column, inner)];
                    right[static_cast<std::size_t>(row)] -= factor * right[static_cast<std::size_t>(column)];
                }
            }
            for (Index row = n; row-- > 0;)
            {
                for (Index column = row + 1; column < n; ++column)
                    right[static_cast<std::size_t>(row)] -=
                        matrix[at(row, column)] * right[static_cast<std::size_t>(column)];
                right[static_cast<std::size_t>(row)] /= matrix[at(row, row)];
            }
            return true;
        }
    } // namespace decomposition_detail

    template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    class SelfAdjointEigenSolver<Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>>
    {
    public:
        using MatrixType = Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>;
        using EigenvalueType = Matrix<Scalar, Rows, 1, ColMajor, MaxRows, 1>;
        using EigenvectorsType = Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>;
        SelfAdjointEigenSolver() = default;
        explicit SelfAdjointEigenSolver(Index size) : _values(size, 1), _vectors(size, size)
        {
        }
        explicit SelfAdjointEigenSolver(const MatrixType& input, int options = ComputeEigenvectors)
        {
            compute(input, options);
        }

        SelfAdjointEigenSolver& compute(const MatrixType& input, int options = ComputeEigenvectors)
        {
            _info = InvalidInput;
            _vectorsComputed = (options & EigenvaluesOnly) == 0;
            if (input.rows() != input.cols() || input.rows() == 0 || !input.allFinite())
                return *this;
            const auto settings = internal::currentExecutionSettings();
            const long double work = static_cast<long double>(input.rows()) * input.rows() * input.rows();
            const auto selected = decomposition_detail::selectDecompositionBackend<Scalar>(
                work, 4.0L * 1024.0L * 1024.0L, 256.0L * 1024.0L);
#ifdef PLAMATRIX_WITH_CUDA
            if (selected == internal::Backend::Cuda)
            {
                auto host = decomposition_detail::toDenseStorage<Scalar>(input);
                auto gpu = host.toGpu();
                auto [gpu_vectors, gpu_values] = internal::eighDecomposition(gpu);
                auto vectors = gpu_vectors.toCpu();
                auto values = gpu_values.toCpu();
                _values.resize(input.rows(), 1);
                _vectors.resize(input.rows(), input.cols());
                for (Index column = 0; column < input.cols(); ++column)
                {
                    _values(column) = values(column, 0);
                    for (Index row = 0; row < input.rows(); ++row)
                        _vectors(row, column) = vectors(row, column);
                }
                _info = Success;
                _backend = internal::Backend::Cuda;
                return *this;
            }
#endif
            if (selected == internal::Backend::OpenCl || selected == internal::Backend::Vulkan)
            {
                const auto factors =
                    internal::detail::GpuOps<Scalar>::selfAdjointEigen(selected, input.data(), input.rows());
                _values.resize(input.rows(), 1);
                _vectors.resize(input.rows(), input.cols());
                for (Index column = 0; column < input.cols(); ++column)
                {
                    _values(column) = factors.values[static_cast<std::size_t>(column)];
                    for (Index row = 0; row < input.rows(); ++row)
                        _vectors(row, column) = factors.vectors[static_cast<std::size_t>(row + column * input.rows())];
                }
                _info = Success;
                _backend = selected;
                return *this;
            }
            if (settings.policy == internal::ExecutionPolicy::GpuRequired)
                throw internal::Error(internal::ErrorCode::UnsupportedOperation,
                                      "SelfAdjointEigenSolver is unavailable on the selected GPU",
                                      settings.preferredGpu);
            _info = decomposition_detail::symmetricJacobi<Scalar>(input, _values, _vectors);
            _backend = internal::Backend::Cpu;
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
        internal::Backend backend() const noexcept
        {
            return _backend;
        }
        MatrixType operatorSqrt() const
        {
            if (_info != Success || !_vectorsComputed)
                throw std::logic_error("operatorSqrt requires eigenvectors");
            MatrixType result = MatrixType::Zero(_vectors.rows(), _vectors.cols());
            for (Index component = 0; component < _values.size(); ++component)
            {
                if (_values(component) < Scalar{})
                    throw std::runtime_error("operatorSqrt requires positive semidefinite input");
                const Scalar scale = std::sqrt(std::max(Scalar{}, _values(component)));
                for (Index column = 0; column < result.cols(); ++column)
                    for (Index row = 0; row < result.rows(); ++row)
                        result(row, column) += _vectors(row, component) * scale * _vectors(column, component);
            }
            return result;
        }
        MatrixType operatorInverseSqrt() const
        {
            if (_info != Success || !_vectorsComputed)
                throw std::logic_error("operatorInverseSqrt requires eigenvectors");
            MatrixType result = MatrixType::Zero(_vectors.rows(), _vectors.cols());
            for (Index component = 0; component < _values.size(); ++component)
            {
                if (!(_values(component) > Scalar{}))
                    throw std::runtime_error("operatorInverseSqrt requires positive definite input");
                const Scalar scale = Scalar{1} / std::sqrt(_values(component));
                for (Index column = 0; column < result.cols(); ++column)
                    for (Index row = 0; row < result.rows(); ++row)
                        result(row, column) += _vectors(row, component) * scale * _vectors(column, component);
            }
            return result;
        }

    protected:
        EigenvalueType _values;
        EigenvectorsType _vectors;
        ComputationInfo _info = InvalidInput;
        internal::Backend _backend = internal::Backend::Cpu;
        bool _vectorsComputed = false;
    };

    template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    class EigenSolver<Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>>
    {
    public:
        using MatrixType = Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>;
        using ComplexScalar = std::complex<Scalar>;
        using EigenvalueType = Matrix<ComplexScalar, Cols, 1, ColMajor, MaxCols, 1>;
        using EigenvectorsType = Matrix<ComplexScalar, Rows, Cols, Options, MaxRows, MaxCols>;
        EigenSolver() = default;
        explicit EigenSolver(Index size) : _values(size, 1), _vectors(size, size)
        {
        }
        explicit EigenSolver(const MatrixType& input, bool computeEigenvectors = true)
        {
            compute(input, computeEigenvectors);
        }

        EigenSolver& compute(const MatrixType& input, bool computeEigenvectors = true)
        {
            _info = InvalidInput;
            _vectorsComputed = computeEigenvectors;
            if (input.rows() != input.cols() || input.rows() == 0 || !input.allFinite())
                return *this;
            const Index n = input.rows();
            const auto at = [n](Index row, Index column) { return static_cast<std::size_t>(row + column * n); };
            std::vector<ComplexScalar> work(static_cast<std::size_t>(n * n));
            for (Index column = 0; column < n; ++column)
                for (Index row = 0; row < n; ++row)
                {
                    work[at(row, column)] = input(row, column);
                }
            const std::vector<ComplexScalar> original = work;
            const Scalar epsilon = std::numeric_limits<Scalar>::epsilon();
            bool converged = n < 2;
            Index active = n;
            for (Index iteration = 0; iteration < 4000 * std::max<Index>(n, 1); ++iteration)
            {
                while (active > 1)
                {
                    const Scalar scale =
                        std::abs(work[at(active - 2, active - 2)]) + std::abs(work[at(active - 1, active - 1)]);
                    if (std::abs(work[at(active - 1, active - 2)]) > epsilon * Scalar{64} * std::max(Scalar{1}, scale))
                        break;
                    work[at(active - 1, active - 2)] = {};
                    --active;
                }
                if (active <= 1)
                {
                    converged = true;
                    break;
                }
                if (active == 2)
                {
                    const ComplexScalar a = work[at(0, 0)];
                    const ComplexScalar b = work[at(0, 1)];
                    const ComplexScalar c = work[at(1, 0)];
                    const ComplexScalar d = work[at(1, 1)];
                    const ComplexScalar root = std::sqrt((a - d) * (a - d) + ComplexScalar{4} * b * c);
                    work[at(0, 0)] = (a + d + root) / ComplexScalar{2};
                    work[at(1, 1)] = (a + d - root) / ComplexScalar{2};
                    work[at(1, 0)] = {};
                    active = 1;
                    converged = true;
                    break;
                }
                ComplexScalar shift = work[at(active - 1, active - 1)];
                if (active > 1)
                {
                    const ComplexScalar a = work[at(active - 2, active - 2)];
                    const ComplexScalar b = work[at(active - 2, active - 1)];
                    const ComplexScalar c = work[at(active - 1, active - 2)];
                    const ComplexScalar d = work[at(active - 1, active - 1)];
                    const ComplexScalar root = std::sqrt((a - d) * (a - d) + ComplexScalar{4} * b * c);
                    const ComplexScalar first = (a + d + root) / ComplexScalar{2};
                    const ComplexScalar second = (a + d - root) / ComplexScalar{2};
                    shift = std::abs(first - d) < std::abs(second - d) ? first : second;
                }
                const auto block_at = [active](Index row, Index column)
                { return static_cast<std::size_t>(row + column * active); };
                std::vector<ComplexScalar> block(static_cast<std::size_t>(active * active));
                for (Index column = 0; column < active; ++column)
                    for (Index row = 0; row < active; ++row)
                        block[block_at(row, column)] = work[at(row, column)];
                for (Index index = 0; index < active; ++index)
                    block[block_at(index, index)] -= shift;
                std::vector<ComplexScalar> q;
                std::vector<ComplexScalar> r;
                decomposition_detail::complexQr(block, active, q, r);
                block = decomposition_detail::multiplyComplex(r, q, active);
                for (Index index = 0; index < active; ++index)
                    block[block_at(index, index)] += shift;
                for (Index column = 0; column < active; ++column)
                    for (Index row = 0; row < active; ++row)
                        work[at(row, column)] = block[block_at(row, column)];
            }
            if (!converged)
            {
                _info = NoConvergence;
                return *this;
            }
            _values.resize(n, 1);
            _vectors.resize(n, n);
            for (Index index = 0; index < n; ++index)
                _values(index) = work[at(index, index)];
            if (computeEigenvectors)
            {
                for (Index eigen = 0; eigen < n; ++eigen)
                {
                    std::vector<ComplexScalar> vector(static_cast<std::size_t>(n), ComplexScalar{1});
                    Scalar matrix_norm{};
                    for (const auto& value : original)
                        matrix_norm = std::max(matrix_norm, std::abs(value));
                    const Scalar perturbation = std::sqrt(epsilon) * std::max(Scalar{1}, matrix_norm);
                    for (int iteration = 0; iteration < 12; ++iteration)
                    {
                        auto shifted = original;
                        for (Index index = 0; index < n; ++index)
                            shifted[at(index, index)] -= _values(eigen) + ComplexScalar{perturbation};
                        if (!decomposition_detail::solveComplexSystem(shifted, vector, n))
                            break;
                        Scalar norm{};
                        for (const auto& value : vector)
                            norm = std::hypot(norm, std::abs(value));
                        for (auto& value : vector)
                            value /= norm;
                    }
                    Scalar norm{};
                    for (Index row = 0; row < n; ++row)
                    {
                        _vectors(row, eigen) = vector[static_cast<std::size_t>(row)];
                        norm = std::hypot(norm, std::abs(_vectors(row, eigen)));
                    }
                    for (Index row = 0; row < n; ++row)
                        _vectors(row, eigen) /= norm;
                }
            }
            _info = Success;
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

    template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    class GeneralizedSelfAdjointEigenSolver<Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>>
        : public SelfAdjointEigenSolver<Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>>
    {
    public:
        using MatrixType = Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>;
        using Base = SelfAdjointEigenSolver<MatrixType>;
        GeneralizedSelfAdjointEigenSolver() = default;
        explicit GeneralizedSelfAdjointEigenSolver(Index size) : Base(size)
        {
        }
        GeneralizedSelfAdjointEigenSolver(const MatrixType& first,
                                          const MatrixType& second,
                                          int options = ComputeEigenvectors | Ax_lBx)
        {
            compute(first, second, options);
        }
        GeneralizedSelfAdjointEigenSolver&
        compute(const MatrixType& first, const MatrixType& second, int options = ComputeEigenvectors | Ax_lBx)
        {
            LLT<MatrixType> cholesky(second);
            if (cholesky.info() != Success)
            {
                Base::_info = NumericalIssue;
                return *this;
            }
            const MatrixType lower = cholesky.matrixL();
            MatrixType transformed(first.rows(), first.cols());
            const int kind = (options & GenEigMask) == 0 ? Ax_lBx : options & GenEigMask;
            if (kind == Ax_lBx)
            {
                MatrixType left(first);
                for (Index column = 0; column < left.cols(); ++column)
                    for (Index row = 0; row < left.rows(); ++row)
                    {
                        for (Index inner = 0; inner < row; ++inner)
                            left(row, column) -= lower(row, inner) * left(inner, column);
                        left(row, column) /= lower(row, row);
                    }
                transformed = left.transpose().eval();
                for (Index column = 0; column < transformed.cols(); ++column)
                    for (Index row = 0; row < transformed.rows(); ++row)
                    {
                        for (Index inner = 0; inner < row; ++inner)
                            transformed(row, column) -= lower(row, inner) * transformed(inner, column);
                        transformed(row, column) /= lower(row, row);
                    }
                transformed = transformed.transpose().eval();
            }
            else
            {
                transformed = (lower.transpose() * first * lower).eval();
            }
            Base::compute(transformed, options & EigVecMask);
            if (Base::_info == Success && (options & EigenvaluesOnly) == 0)
            {
                if (kind == BAx_lx)
                    Base::_vectors = (lower * Base::_vectors).eval();
                else
                {
                    for (Index column = 0; column < Base::_vectors.cols(); ++column)
                        for (Index row = Base::_vectors.rows(); row-- > 0;)
                        {
                            for (Index inner = row + 1; inner < Base::_vectors.rows(); ++inner)
                                Base::_vectors(row, column) -= lower(inner, row) * Base::_vectors(inner, column);
                            Base::_vectors(row, column) /= lower(row, row);
                        }
                }
            }
            return *this;
        }
    };

    template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    class GeneralizedEigenSolver<Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>>
    {
    public:
        using MatrixType = Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>;
        using ComplexScalar = std::complex<Scalar>;
        using VectorType = Matrix<Scalar, Cols, 1, ColMajor, MaxCols, 1>;
        using ComplexVectorType = Matrix<ComplexScalar, Cols, 1, ColMajor, MaxCols, 1>;
        using EigenvectorsType = Matrix<ComplexScalar, Rows, Cols, Options, MaxRows, MaxCols>;
        GeneralizedEigenSolver() = default;
        explicit GeneralizedEigenSolver(Index size) : _alphas(size, 1), _betas(size, 1), _vectors(size, size)
        {
        }
        GeneralizedEigenSolver(const MatrixType& first, const MatrixType& second, bool vectors = true)
        {
            compute(first, second, vectors);
        }
        GeneralizedEigenSolver& compute(const MatrixType& first, const MatrixType& second, bool vectors = true)
        {
            _info = InvalidInput;
            _vectorsComputed = vectors;
            if (first.rows() != first.cols() || second.rows() != second.cols() || first.rows() != second.rows() ||
                first.rows() == 0 || !first.allFinite() || !second.allFinite())
            {
                return *this;
            }
            try
            {
                RealQZ<MatrixType> qz(first, second, vectors);
                _info = qz.info();
                if (_info != Success)
                    return *this;

                const Index size = first.rows();
                const MatrixType& schur = qz.matrixS();
                const MatrixType& metric = qz.matrixT();
                _alphas.resize(size, 1);
                _betas.resize(size, 1);
                _vectors.resize(size, size);
                const Scalar epsilon = std::numeric_limits<Scalar>::epsilon();
                Index index = 0;
                while (index < size)
                {
                    if (index + 1 == size || schur(index + 1, index) == Scalar{})
                    {
                        _alphas(index) = ComplexScalar{schur(index, index)};
                        _betas(index) = metric(index, index);
                        ++index;
                        continue;
                    }

                    const Scalar s00 = schur(index, index);
                    const Scalar s01 = schur(index, index + 1);
                    const Scalar s10 = schur(index + 1, index);
                    const Scalar s11 = schur(index + 1, index + 1);
                    const Scalar t00 = metric(index, index);
                    const Scalar t01 = metric(index, index + 1);
                    const Scalar t11 = metric(index + 1, index + 1);
                    const Scalar quadratic = t00 * t11;
                    const Scalar linear = -s00 * t11 - s11 * t00 + s10 * t01;
                    const Scalar constant = s00 * s11 - s01 * s10;
                    const Scalar scale = std::abs(quadratic) + std::abs(linear) + std::abs(constant);
                    const Scalar tolerance = epsilon * std::max(Scalar{1}, scale);
                    if (std::abs(quadratic) > tolerance)
                    {
                        const ComplexScalar root =
                            std::sqrt(ComplexScalar{linear * linear - Scalar{4} * quadratic * constant});
                        _alphas(index) = -linear - root;
                        _alphas(index + 1) = -linear + root;
                        _betas(index) = Scalar{2} * quadratic;
                        _betas(index + 1) = Scalar{2} * quadratic;
                    }
                    else if (std::abs(linear) > tolerance)
                    {
                        _alphas(index) = -constant;
                        _betas(index) = linear;
                        _alphas(index + 1) = ComplexScalar{1};
                        _betas(index + 1) = Scalar{};
                    }
                    else
                    {
                        _info = NumericalIssue;
                        return *this;
                    }
                    index += 2;
                }

                for (Index eigen = 0; eigen < size; ++eigen)
                {
                    if (_betas(eigen) < Scalar{})
                    {
                        _betas(eigen) = -_betas(eigen);
                        _alphas(eigen) = -_alphas(eigen);
                    }
                }

                if (vectors)
                {
                    using ComplexMatrix = Matrix<ComplexScalar, Dynamic, Dynamic>;
                    for (Index eigen = 0; eigen < size; ++eigen)
                    {
                        ComplexMatrix pencil(size, size);
                        for (Index col = 0; col < size; ++col)
                            for (Index row = 0; row < size; ++row)
                                pencil(row, col) = _betas(eigen) * first(row, col) - _alphas(eigen) * second(row, col);
                        JacobiSVD<ComplexMatrix, ComputeFullV> nullspace(pencil, ComputeFullV);
                        if (nullspace.info() != Success)
                        {
                            _info = nullspace.info();
                            return *this;
                        }
                        const Index source = nullspace.matrixV().cols() - 1;
                        Scalar norm{};
                        for (Index row = 0; row < size; ++row)
                            norm = std::hypot(norm, std::abs(nullspace.matrixV()(row, source)));
                        if (!(norm > Scalar{}))
                        {
                            _info = NumericalIssue;
                            return *this;
                        }
                        for (Index row = 0; row < size; ++row)
                            _vectors(row, eigen) = nullspace.matrixV()(row, source) / norm;
                    }
                }
                _info = Success;
            }
            catch (const internal::Error&)
            {
                throw;
            }
            catch (...)
            {
                _info = NumericalIssue;
            }
            return *this;
        }
        const ComplexVectorType& alphas() const noexcept
        {
            return _alphas;
        }
        const VectorType& betas() const noexcept
        {
            return _betas;
        }
        ComplexVectorType eigenvalues() const
        {
            ComplexVectorType result(_alphas.size(), 1);
            for (Index index = 0; index < result.size(); ++index)
                result(index) = _alphas(index) / _betas(index);
            return result;
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
        ComplexVectorType _alphas;
        VectorType _betas;
        EigenvectorsType _vectors;
        ComputationInfo _info = InvalidInput;
        bool _vectorsComputed = false;
    };
} // namespace plamatrix::v1
