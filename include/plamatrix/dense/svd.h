#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <tuple>
#include <vector>

#include "plamatrix/dense/detail/static_or_dynamic_vector.h"
#include "plamatrix/dense/detail/decomposition_backend.h"
#include "plamatrix/dense/solver_base.h"
#include "plamatrix/internal/ops/decomposition.h"

namespace plamatrix::v1
{
    namespace decomposition_detail
    {
        template <typename Scalar, typename MatrixType>
        internal::DenseStorage<Scalar, internal::Device::CPU> toDenseStorage(const MatrixType& input)
        {
            internal::DenseStorage<Scalar, internal::Device::CPU> result(input.rows(), input.cols());
            for (Index column = 0; column < input.cols(); ++column)
            {
                for (Index row = 0; row < input.rows(); ++row)
                {
                    result(row, column) = input(row, column);
                }
            }
            return result;
        }

        template <typename Scalar, int Rows, int Cols, int MaxRows, int MaxCols> class SvdStorage
        {
        public:
            using RealScalar = typename NumTraits<Scalar>::Real;
            using UType = Matrix<Scalar, Rows, Dynamic, ColMajor, MaxRows, Rows == Dynamic ? Dynamic : Rows>;
            using VType = Matrix<Scalar, Cols, Dynamic, ColMajor, MaxCols, Cols == Dynamic ? Dynamic : Cols>;
            using SingularValuesType = Matrix<RealScalar, (Rows < Cols ? Rows : Cols), 1>;

            template <typename MatrixType> void compute(const MatrixType& input, int options)
            {
                if (((options & ComputeFullU) != 0 && (options & ComputeThinU) != 0) ||
                    ((options & ComputeFullV) != 0 && (options & ComputeThinV) != 0))
                {
                    throw std::invalid_argument("SVD cannot request full and thin factors for the same side");
                }
                if (input.rows() == 0 || input.cols() == 0 || !input.allFinite())
                {
                    _info = InvalidInput;
                    _initialized = true;
                    return;
                }
                if constexpr (NumTraits<Scalar>::IsComplex)
                {
                    computeJacobi(input, options);
                }
                else
                {
                    const auto settings = internal::currentExecutionSettings();
                    const long double work =
                        static_cast<long double>(input.rows()) * input.cols() * std::min(input.rows(), input.cols());
                    const auto selected =
                        selectDecompositionBackend<Scalar>(work, 8.0L * 1024.0L * 1024.0L, 16.0L * 1024.0L);
                    if (settings.policy == internal::ExecutionPolicy::GpuRequired && selected == internal::Backend::Cpu)
                    {
                        throw internal::Error(internal::ErrorCode::UnsupportedOperation,
                                              "SVD is unavailable on the selected GPU backend",
                                              settings.preferredGpu);
                    }
                    if constexpr (MaxRows != Dynamic && MaxCols != Dynamic)
                    {
                        if (selected == internal::Backend::Cpu)
                        {
                            computeJacobi(input, options);
                            return;
                        }
                    }
                    if (selected == internal::Backend::OpenCl || selected == internal::Backend::Vulkan)
                    {
                        const auto factors =
                            internal::detail::GpuOps<Scalar>::svd(selected, input.data(), input.rows(), input.cols());
                        const Index minor = std::min(input.rows(), input.cols());
                        const bool full_u = (options & ComputeFullU) != 0;
                        const bool full_v = (options & ComputeFullV) != 0;
                        _u.resize(input.rows(), full_u ? input.rows() : minor);
                        _v.resize(input.cols(), full_v ? input.cols() : minor);
                        _singular.resize(minor, 1);
                        for (Index column = 0; column < _u.cols(); ++column)
                            for (Index row = 0; row < _u.rows(); ++row)
                                _u(row, column) = factors.u[static_cast<std::size_t>(row + column * input.rows())];
                        for (Index column = 0; column < _v.cols(); ++column)
                            for (Index row = 0; row < _v.rows(); ++row)
                                _v(row, column) = factors.v[static_cast<std::size_t>(row + column * input.cols())];
                        for (Index index = 0; index < minor; ++index)
                            _singular(index) = factors.singular[static_cast<std::size_t>(index)];
                        _rows = input.rows();
                        _cols = input.cols();
                        _options = options;
                        _info = Success;
                        _initialized = true;
                        _backend = selected;
                        return;
                    }
                    try
                    {
                        auto host = toDenseStorage<Scalar>(input);
                        internal::DenseStorage<Scalar, internal::Device::CPU> u;
                        internal::DenseStorage<Scalar, internal::Device::CPU> singular;
                        internal::DenseStorage<Scalar, internal::Device::CPU> vt;
#ifdef PLAMATRIX_WITH_CUDA
                        if (selected == internal::Backend::Cuda)
                        {
                            auto gpu = host.toGpu();
                            auto [gpu_u, gpu_singular, gpu_vt] = internal::svd(gpu);
                            u = gpu_u.toCpu();
                            singular = gpu_singular.toCpu();
                            vt = gpu_vt.toCpu();
                            _backend = internal::Backend::Cuda;
                        }
                        else
#endif
                        {
                            std::tie(u, singular, vt) = internal::svd(host);
                            _backend = internal::Backend::Cpu;
                        }
                        const Index minor = singular.rows();
                        const bool full_u = (options & ComputeFullU) != 0;
                        const bool full_v = (options & ComputeFullV) != 0;
                        _u.resize(input.rows(), full_u ? input.rows() : minor);
                        _v.resize(input.cols(), full_v ? input.cols() : minor);
                        _singular.resize(minor, 1);
                        for (Index column = 0; column < _u.cols(); ++column)
                        {
                            for (Index row = 0; row < _u.rows(); ++row)
                            {
                                _u(row, column) = u(row, column);
                            }
                        }
                        for (Index column = 0; column < _v.cols(); ++column)
                        {
                            for (Index row = 0; row < _v.rows(); ++row)
                            {
                                _v(row, column) = vt(column, row);
                            }
                        }
                        for (Index index = 0; index < minor; ++index)
                        {
                            _singular(index) = singular(index, 0);
                        }
                        _rows = input.rows();
                        _cols = input.cols();
                        _options = options;
                        _info = Success;
                        _initialized = true;
                    }
                    catch (const internal::Error&)
                    {
                        throw;
                    }
                    catch (...)
                    {
                        _info = NoConvergence;
                        _initialized = true;
                    }
                }
            }

            template <typename Rhs> auto solve(const Rhs& right) const
            {
                if (_info != Success || right.rows() != _rows)
                {
                    throw std::runtime_error("SVD solve requires a successful factorization and matching rows");
                }
                Matrix<Scalar, Cols, Rhs::ColsAtCompileTime> result(_cols, right.cols());
                result.setZero();
                const RealScalar limit = thresholdValue();
                for (Index column = 0; column < right.cols(); ++column)
                {
                    for (Index component = 0; component < _singular.size(); ++component)
                    {
                        if (!(_singular(component) > limit))
                        {
                            continue;
                        }
                        Scalar projection{};
                        for (Index row = 0; row < _rows; ++row)
                        {
                            if constexpr (NumTraits<Scalar>::IsComplex)
                                projection += std::conj(_u(row, component)) * right(row, column);
                            else
                                projection += _u(row, component) * right(row, column);
                        }
                        projection /= _singular(component);
                        for (Index row = 0; row < _cols; ++row)
                        {
                            result(row, column) += _v(row, component) * projection;
                        }
                    }
                }
                return result;
            }

            RealScalar thresholdValue() const
            {
                if (_prescribedThreshold >= RealScalar{})
                {
                    return _prescribedThreshold;
                }
                const RealScalar largest = _singular.size() == 0 ? RealScalar{} : _singular(0);
                return std::numeric_limits<RealScalar>::epsilon() * static_cast<RealScalar>(std::max(_rows, _cols)) *
                       largest;
            }

        private:
            static Scalar conjugateValue(const Scalar& value)
            {
                if constexpr (NumTraits<Scalar>::IsComplex)
                    return std::conj(value);
                return value;
            }

            template <typename MatrixType> void computeJacobi(const MatrixType& input, int options)
            {
                if (internal::currentExecutionSettings().policy == internal::ExecutionPolicy::GpuRequired)
                    throw internal::Error(internal::ErrorCode::UnsupportedOperation,
                                          "Jacobi SVD is unavailable on the selected GPU backend",
                                          internal::currentExecutionSettings().preferredGpu);

                const Index m = input.rows();
                const Index n = input.cols();
                const Index minor = std::min(m, n);
                if (m < n)
                {
                    Matrix<Scalar, Cols, Rows, ColMajor, MaxCols, MaxRows> adjoint(n, m);
                    for (Index column = 0; column < m; ++column)
                        for (Index row = 0; row < n; ++row)
                            adjoint(row, column) = conjugateValue(input(column, row));
                    int transposedOptions = 0;
                    if ((options & ComputeFullU) != 0)
                        transposedOptions |= ComputeFullV;
                    if ((options & ComputeThinU) != 0)
                        transposedOptions |= ComputeThinV;
                    if ((options & ComputeFullV) != 0)
                        transposedOptions |= ComputeFullU;
                    if ((options & ComputeThinV) != 0)
                        transposedOptions |= ComputeThinU;
                    SvdStorage<Scalar, Cols, Rows, MaxCols, MaxRows> transposed;
                    transposed.compute(adjoint, transposedOptions);
                    _u = transposed._v;
                    _v = transposed._u;
                    _singular = transposed._singular;
                    _rows = m;
                    _cols = n;
                    _options = options;
                    _info = transposed._info;
                    _backend = transposed._backend;
                    _initialized = transposed._initialized;
                    return;
                }
                MatrixType work(input);
                Matrix<Scalar, Cols, Cols, ColMajor, MaxCols, MaxCols> right =
                    Matrix<Scalar, Cols, Cols, ColMajor, MaxCols, MaxCols>::Identity(n, n);
                const RealScalar epsilon = std::numeric_limits<RealScalar>::epsilon();
                const Index maximumSweeps = std::max<Index>(30, 8 * n);
                bool converged = n <= 1;
                for (Index sweep = 0; sweep < maximumSweeps && !converged; ++sweep)
                {
                    converged = true;
                    for (Index left = 0; left < n; ++left)
                    {
                        for (Index rightColumn = left + 1; rightColumn < n; ++rightColumn)
                        {
                            RealScalar alpha{};
                            RealScalar beta{};
                            Scalar gamma{};
                            for (Index row = 0; row < m; ++row)
                            {
                                alpha += static_cast<RealScalar>(std::norm(work(row, left)));
                                beta += static_cast<RealScalar>(std::norm(work(row, rightColumn)));
                                gamma += conjugateValue(work(row, left)) * work(row, rightColumn);
                            }
                            const RealScalar gammaMagnitude = std::abs(gamma);
                            if (gammaMagnitude <= epsilon * static_cast<RealScalar>(m) * std::sqrt(alpha * beta))
                                continue;
                            converged = false;
                            const RealScalar zeta = (beta - alpha) / (RealScalar{2} * gammaMagnitude);
                            const RealScalar tangent = (zeta >= RealScalar{} ? RealScalar{1} : RealScalar{-1}) /
                                                       (std::abs(zeta) + std::sqrt(RealScalar{1} + zeta * zeta));
                            const RealScalar cosine = RealScalar{1} / std::sqrt(RealScalar{1} + tangent * tangent);
                            const RealScalar sine = cosine * tangent;
                            const Scalar phase = gamma / gammaMagnitude;
                            for (Index row = 0; row < m; ++row)
                            {
                                const Scalar leftValue = work(row, left);
                                const Scalar rightValue = work(row, rightColumn);
                                work(row, left) = cosine * leftValue - sine * conjugateValue(phase) * rightValue;
                                work(row, rightColumn) = sine * phase * leftValue + cosine * rightValue;
                            }
                            for (Index row = 0; row < n; ++row)
                            {
                                const Scalar leftValue = right(row, left);
                                const Scalar rightValue = right(row, rightColumn);
                                right(row, left) = cosine * leftValue - sine * conjugateValue(phase) * rightValue;
                                right(row, rightColumn) = sine * phase * leftValue + cosine * rightValue;
                            }
                        }
                    }
                }

                StaticOrDynamicVector<std::pair<RealScalar, Index>, Cols> order(static_cast<std::size_t>(n));
                for (Index column = 0; column < n; ++column)
                {
                    RealScalar norm{};
                    for (Index row = 0; row < m; ++row)
                        norm = std::hypot(norm, std::abs(work(row, column)));
                    order[static_cast<std::size_t>(column)] = {norm, column};
                }
                std::sort(order.begin(),
                          order.end(),
                          [](const auto& left, const auto& rightValue) {
                              return left.first != rightValue.first ? left.first > rightValue.first
                                                                    : left.second < rightValue.second;
                          });

                const bool fullU = (options & ComputeFullU) != 0;
                const bool fullV = (options & ComputeFullV) != 0;
                _u.resize(m, fullU ? m : minor);
                _v.resize(n, fullV ? n : minor);
                _singular.resize(minor, 1);
                _u.setZero();
                for (Index component = 0; component < minor; ++component)
                {
                    const auto [sigma, source] = order[static_cast<std::size_t>(component)];
                    _singular(component) = sigma;
                    if (sigma > epsilon)
                        for (Index row = 0; row < m; ++row)
                            _u(row, component) = work(row, source) / sigma;
                }
                for (Index component = 0; component < _v.cols(); ++component)
                {
                    const Index source = order[static_cast<std::size_t>(component)].second;
                    for (Index row = 0; row < n; ++row)
                        _v(row, component) = right(row, source);
                }
                for (Index component = 0; component < _u.cols(); ++component)
                {
                    RealScalar norm{};
                    for (Index row = 0; row < m; ++row)
                        norm = std::hypot(norm, std::abs(_u(row, component)));
                    if (norm > epsilon * static_cast<RealScalar>(m))
                        continue;
                    bool completed = false;
                    for (Index candidate = 0; candidate < m && !completed; ++candidate)
                    {
                        for (Index row = 0; row < m; ++row)
                            _u(row, component) = row == candidate ? Scalar{1} : Scalar{};
                        for (Index previous = 0; previous < component; ++previous)
                        {
                            Scalar projection{};
                            for (Index row = 0; row < m; ++row)
                                projection += conjugateValue(_u(row, previous)) * _u(row, component);
                            for (Index row = 0; row < m; ++row)
                                _u(row, component) -= _u(row, previous) * projection;
                        }
                        norm = RealScalar{};
                        for (Index row = 0; row < m; ++row)
                            norm = std::hypot(norm, std::abs(_u(row, component)));
                        if (norm > epsilon * static_cast<RealScalar>(m))
                        {
                            for (Index row = 0; row < m; ++row)
                                _u(row, component) /= norm;
                            completed = true;
                        }
                    }
                    if (!completed)
                    {
                        _info = NoConvergence;
                        _initialized = true;
                        return;
                    }
                }
                _rows = m;
                _cols = n;
                _options = options;
                _info = converged ? Success : NoConvergence;
                _backend = internal::Backend::Cpu;
                _initialized = true;
            }

        public:
            UType _u;
            VType _v;
            SingularValuesType _singular;
            Index _rows = 0;
            Index _cols = 0;
            int _options = 0;
            RealScalar _prescribedThreshold = RealScalar{-1};
            ComputationInfo _info = InvalidInput;
            internal::Backend _backend = internal::Backend::Cpu;
            bool _initialized = false;
        };
    } // namespace decomposition_detail

    template <typename Scalar, int Rows, int Cols, int StorageOptions, int MaxRows, int MaxCols, int Options>
    class JacobiSVD<Matrix<Scalar, Rows, Cols, StorageOptions, MaxRows, MaxCols>, Options>
        : public SolverBase<JacobiSVD<Matrix<Scalar, Rows, Cols, StorageOptions, MaxRows, MaxCols>, Options>>
    {
    public:
        using MatrixType = Matrix<Scalar, Rows, Cols, StorageOptions, MaxRows, MaxCols>;
        using Storage = decomposition_detail::SvdStorage<Scalar, Rows, Cols, MaxRows, MaxCols>;
        using RealScalar = typename Storage::RealScalar;
        using SingularValuesType = typename Storage::SingularValuesType;
        JacobiSVD() = default;
        explicit JacobiSVD(Index rows, Index columns)
        {
            _storage._rows = rows;
            _storage._cols = columns;
        }
        explicit JacobiSVD(const MatrixType& input, int computationOptions = Options)
        {
            compute(input, computationOptions);
        }
        JacobiSVD& compute(const MatrixType& input, int computationOptions = Options)
        {
            _storage.compute(input, computationOptions);
            return *this;
        }
        const SingularValuesType& singularValues() const noexcept
        {
            return _storage._singular;
        }
        const typename Storage::UType& matrixU() const noexcept
        {
            return _storage._u;
        }
        const typename Storage::VType& matrixV() const noexcept
        {
            return _storage._v;
        }
        bool computeU() const noexcept
        {
            return (_storage._options & (ComputeFullU | ComputeThinU)) != 0;
        }
        bool computeV() const noexcept
        {
            return (_storage._options & (ComputeFullV | ComputeThinV)) != 0;
        }
        JacobiSVD& setThreshold(RealScalar value)
        {
            _storage._prescribedThreshold = value;
            return *this;
        }
        RealScalar threshold() const
        {
            return _storage.thresholdValue();
        }
        Index rank() const
        {
            Index result = 0;
            for (Index index = 0; index < _storage._singular.size(); ++index)
            {
                result += _storage._singular(index) > threshold();
            }
            return result;
        }
        ComputationInfo info() const noexcept
        {
            return _storage._info;
        }
        bool isInitialized() const noexcept
        {
            return _storage._initialized;
        }
        internal::Backend backend() const noexcept
        {
            return _storage._backend;
        }
        template <typename Rhs> auto _solve(const Rhs& right) const
        {
            return _storage.solve(right);
        }
        template <typename Node> auto solve(const DenseExpression<Node>& right) const
        {
            return _solve(right.eval());
        }
        template <typename Rhs> auto solve(const MatrixBase<Rhs>& right) const
        {
            return _solve(right.derived());
        }

    private:
        Storage _storage;
    };

    template <typename Scalar, int Rows, int Cols, int StorageOptions, int MaxRows, int MaxCols, int Options>
    class BDCSVD<Matrix<Scalar, Rows, Cols, StorageOptions, MaxRows, MaxCols>, Options>
        : public SolverBase<BDCSVD<Matrix<Scalar, Rows, Cols, StorageOptions, MaxRows, MaxCols>, Options>>
    {
    public:
        using MatrixType = Matrix<Scalar, Rows, Cols, StorageOptions, MaxRows, MaxCols>;
        using Storage = decomposition_detail::SvdStorage<Scalar, Rows, Cols, MaxRows, MaxCols>;
        using RealScalar = typename Storage::RealScalar;
        BDCSVD() = default;
        explicit BDCSVD(Index rows, Index columns)
        {
            _storage._rows = rows;
            _storage._cols = columns;
        }
        explicit BDCSVD(const MatrixType& input, int computationOptions = Options)
        {
            compute(input, computationOptions);
        }
        BDCSVD& compute(const MatrixType& input, int computationOptions = Options)
        {
            _storage.compute(input, computationOptions);
            return *this;
        }
        const typename Storage::SingularValuesType& singularValues() const noexcept
        {
            return _storage._singular;
        }
        const typename Storage::UType& matrixU() const noexcept
        {
            return _storage._u;
        }
        const typename Storage::VType& matrixV() const noexcept
        {
            return _storage._v;
        }
        bool computeU() const noexcept
        {
            return (_storage._options & (ComputeFullU | ComputeThinU)) != 0;
        }
        bool computeV() const noexcept
        {
            return (_storage._options & (ComputeFullV | ComputeThinV)) != 0;
        }
        BDCSVD& setThreshold(RealScalar value)
        {
            _storage._prescribedThreshold = value;
            return *this;
        }
        RealScalar threshold() const
        {
            return _storage.thresholdValue();
        }
        Index rank() const
        {
            Index result = 0;
            for (Index index = 0; index < _storage._singular.size(); ++index)
            {
                result += _storage._singular(index) > threshold();
            }
            return result;
        }
        ComputationInfo info() const noexcept
        {
            return _storage._info;
        }
        bool isInitialized() const noexcept
        {
            return _storage._initialized;
        }
        internal::Backend backend() const noexcept
        {
            return _storage._backend;
        }
        template <typename Rhs> auto _solve(const Rhs& right) const
        {
            return _storage.solve(right);
        }
        template <typename Node> auto solve(const DenseExpression<Node>& right) const
        {
            return _solve(right.eval());
        }
        template <typename Rhs> auto solve(const MatrixBase<Rhs>& right) const
        {
            return _solve(right.derived());
        }

    private:
        Storage _storage;
    };
} // namespace plamatrix::v1
