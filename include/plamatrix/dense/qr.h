#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

#include "plamatrix/dense/permutation_matrix.h"
#include "plamatrix/dense/detail/decomposition_backend.h"
#include "plamatrix/dense/detail/static_or_dynamic_vector.h"
#include "plamatrix/dense/solver_base.h"
#include "plamatrix/internal/ops/decomposition.h"

namespace plamatrix::v1
{
    namespace decomposition_detail
    {
        template <typename Scalar> Scalar conjugateValue(const Scalar& value)
        {
            if constexpr (NumTraits<Scalar>::IsComplex)
                return std::conj(value);
            return value;
        }

        template <typename Scalar, typename MatrixType, typename TauContainer, typename PivotContainer>
        Index householderFactor(MatrixType& qr, TauContainer& tau, PivotContainer* pivots)
        {
            using RealScalar = typename NumTraits<Scalar>::Real;
            const Index m = qr.rows();
            const Index n = qr.cols();
            const Index steps = std::min(m, n);
            tau.assign(static_cast<std::size_t>(steps), Scalar{});
            RealScalar scale{};
            for (Index column = 0; column < n; ++column)
            {
                RealScalar norm{};
                for (Index row = 0; row < m; ++row)
                {
                    norm = std::hypot(norm, std::abs(qr(row, column)));
                }
                scale = std::max(scale, norm);
            }
            const RealScalar tolerance =
                std::numeric_limits<RealScalar>::epsilon() * static_cast<RealScalar>(std::max(m, n)) * scale;
            Index rank = 0;
            for (Index step = 0; step < steps; ++step)
            {
                if (pivots != nullptr)
                {
                    Index pivot = step;
                    RealScalar largest{};
                    for (Index column = step; column < n; ++column)
                    {
                        RealScalar norm{};
                        for (Index row = step; row < m; ++row)
                        {
                            norm = std::hypot(norm, std::abs(qr(row, column)));
                        }
                        if (norm > largest)
                        {
                            largest = norm;
                            pivot = column;
                        }
                    }
                    if (pivot != step)
                    {
                        for (Index row = 0; row < m; ++row)
                        {
                            std::swap(qr(row, pivot), qr(row, step));
                        }
                        std::swap((*pivots)[static_cast<std::size_t>(pivot)],
                                  (*pivots)[static_cast<std::size_t>(step)]);
                    }
                }

                RealScalar norm{};
                for (Index row = step; row < m; ++row)
                {
                    norm = std::hypot(norm, std::abs(qr(row, step)));
                }
                if (!(norm > tolerance))
                {
                    continue;
                }
                ++rank;
                const Scalar leading = qr(step, step);
                const Scalar phase = std::abs(leading) == RealScalar{} ? Scalar{1} : leading / std::abs(leading);
                const Scalar alpha = -phase * static_cast<Scalar>(norm);
                const Scalar denominator = qr(step, step) - alpha;
                qr(step, step) = alpha;
                RealScalar tail{};
                for (Index row = step + 1; row < m; ++row)
                {
                    qr(row, step) /= denominator;
                    tail += static_cast<RealScalar>(std::norm(qr(row, step)));
                }
                tau[static_cast<std::size_t>(step)] = Scalar{2} / (Scalar{1} + tail);
                for (Index column = step + 1; column < n; ++column)
                {
                    Scalar projection = qr(step, column);
                    for (Index row = step + 1; row < m; ++row)
                    {
                        projection += conjugateValue(qr(row, step)) * qr(row, column);
                    }
                    projection *= tau[static_cast<std::size_t>(step)];
                    qr(step, column) -= projection;
                    for (Index row = step + 1; row < m; ++row)
                    {
                        qr(row, column) -= qr(row, step) * projection;
                    }
                }
            }
            return rank;
        }

        template <typename Scalar, typename QrType, typename TauContainer, typename ResultType>
        void applyQt(const QrType& qr, const TauContainer& tau, ResultType& result)
        {
            for (Index step = 0; step < static_cast<Index>(tau.size()); ++step)
            {
                for (Index column = 0; column < result.cols(); ++column)
                {
                    Scalar projection = result(step, column);
                    for (Index row = step + 1; row < qr.rows(); ++row)
                    {
                        projection += conjugateValue(qr(row, step)) * result(row, column);
                    }
                    projection *= tau[static_cast<std::size_t>(step)];
                    result(step, column) -= projection;
                    for (Index row = step + 1; row < qr.rows(); ++row)
                    {
                        result(row, column) -= qr(row, step) * projection;
                    }
                }
            }
        }

        template <typename Scalar, typename QrType, typename TauContainer>
        auto explicitQ(const QrType& qr, const TauContainer& tau)
        {
            using Result = Matrix<Scalar,
                                  QrType::RowsAtCompileTime,
                                  QrType::RowsAtCompileTime,
                                  ColMajor,
                                  QrType::MaxRowsAtCompileTime,
                                  QrType::MaxRowsAtCompileTime>;
            Result result = Result::Identity(qr.rows(), qr.rows());
            for (Index offset = static_cast<Index>(tau.size()); offset-- > 0;)
            {
                for (Index column = 0; column < result.cols(); ++column)
                {
                    Scalar projection = result(offset, column);
                    for (Index row = offset + 1; row < qr.rows(); ++row)
                    {
                        projection += conjugateValue(qr(row, offset)) * result(row, column);
                    }
                    projection *= tau[static_cast<std::size_t>(offset)];
                    result(offset, column) -= projection;
                    for (Index row = offset + 1; row < qr.rows(); ++row)
                    {
                        result(row, column) -= qr(row, offset) * projection;
                    }
                }
            }
            return result;
        }
    } // namespace decomposition_detail

    template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    class HouseholderQR<Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>>
        : public SolverBase<HouseholderQR<Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>>>
    {
    public:
        using MatrixType = Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>;
        using RealScalar = typename NumTraits<Scalar>::Real;
        using QType = Matrix<Scalar, Rows, Rows, ColMajor, MaxRows, MaxRows>;
        HouseholderQR() = default;
        explicit HouseholderQR(Index rows, Index columns) : _qr(rows, columns)
        {
        }
        explicit HouseholderQR(const MatrixType& input)
        {
            compute(input);
        }

        HouseholderQR& compute(const MatrixType& input)
        {
            validate(input);
            const auto settings = internal::currentExecutionSettings();
            const long double work =
                static_cast<long double>(input.rows()) * input.cols() * std::min(input.rows(), input.cols());
            const auto selected = decomposition_detail::selectDecompositionBackend<Scalar>(
                work, 8.0L * 1024.0L * 1024.0L, 8.0L * 1024.0L * 1024.0L);
#ifdef PLAMATRIX_WITH_CUDA
            if constexpr (std::is_floating_point_v<Scalar>)
            {
                if (selected == internal::Backend::Cuda)
                {
                    internal::DenseStorage<Scalar, internal::Device::CPU> host(input.rows(), input.cols());
                    for (Index column = 0; column < input.cols(); ++column)
                        for (Index row = 0; row < input.rows(); ++row)
                            host(row, column) = input(row, column);
                    auto gpu = host.toGpu();
                    auto [gpu_q, gpu_r] = internal::qr(gpu);
                    auto q = gpu_q.toCpu();
                    auto r = gpu_r.toCpu();
                    _explicitQ.resize(input.rows(), input.rows());
                    _qr.resize(input.rows(), input.cols());
                    for (Index column = 0; column < input.rows(); ++column)
                        for (Index row = 0; row < input.rows(); ++row)
                            _explicitQ(row, column) = q(row, column);
                    for (Index column = 0; column < input.cols(); ++column)
                        for (Index row = 0; row < input.rows(); ++row)
                            _qr(row, column) = r(row, column);
                    _rank = std::min(input.rows(), input.cols());
                    _gpuFactors = true;
                    _initialized = true;
                    _info = Success;
                    _backend = internal::Backend::Cuda;
                    return *this;
                }
            }
#endif
            if constexpr (std::is_floating_point_v<Scalar>)
            {
                if (selected == internal::Backend::OpenCl || selected == internal::Backend::Vulkan)
                {
                    const auto factors =
                        internal::detail::GpuOps<Scalar>::qr(selected, input.data(), input.rows(), input.cols());
                    _explicitQ.resize(input.rows(), input.rows());
                    _qr.resize(input.rows(), input.cols());
                    std::copy(factors.q.begin(), factors.q.end(), _explicitQ.data());
                    std::copy(factors.r.begin(), factors.r.end(), _qr.data());
                    _rank = factors.rank;
                    _gpuFactors = true;
                    _initialized = true;
                    _info = Success;
                    _backend = selected;
                    return *this;
                }
            }
            if (settings.policy == internal::ExecutionPolicy::GpuRequired)
                throw internal::Error(internal::ErrorCode::UnsupportedOperation,
                                      "HouseholderQR is unavailable on the selected GPU",
                                      settings.preferredGpu);
            _qr = input;
            decomposition_detail::StaticOrDynamicVector<Index, Cols>* no_pivots = nullptr;
            _rank = decomposition_detail::householderFactor<Scalar>(_qr, _tau, no_pivots);
            _gpuFactors = false;
            _initialized = true;
            _info = Success;
            _backend = internal::Backend::Cpu;
            return *this;
        }

        const MatrixType& matrixQR() const noexcept
        {
            return _qr;
        }
        QType householderQ() const
        {
            return _gpuFactors ? _explicitQ : decomposition_detail::explicitQ<Scalar>(_qr, _tau);
        }
        ComputationInfo info() const noexcept
        {
            return _info;
        }
        internal::Backend backend() const noexcept
        {
            return _backend;
        }
        bool isInitialized() const noexcept
        {
            return _initialized;
        }
        RealScalar absDeterminant() const
        {
            if (_qr.rows() != _qr.cols())
                throw std::logic_error("absDeterminant requires a square matrix");
            RealScalar result{1};
            for (Index index = 0; index < _qr.rows(); ++index)
                result *= std::abs(_qr(index, index));
            return result;
        }
        RealScalar logAbsDeterminant() const
        {
            if (_qr.rows() != _qr.cols())
                throw std::logic_error("logAbsDeterminant requires a square matrix");
            RealScalar result{};
            for (Index index = 0; index < _qr.rows(); ++index)
                result += std::log(std::abs(_qr(index, index)));
            return result;
        }

        template <typename Rhs> auto _solve(const Rhs& right) const
        {
            if (right.rows() != _qr.rows())
            {
                throw std::invalid_argument("HouseholderQR right-hand side has incompatible rows");
            }
            if (_rank < std::min(_qr.rows(), _qr.cols()))
            {
                throw std::runtime_error("HouseholderQR solve requires full rank");
            }
            Matrix<Scalar, Rows, Rhs::ColsAtCompileTime> transformed(right);
            if (_gpuFactors)
            {
                Matrix<Scalar, Rows, Rhs::ColsAtCompileTime> projected(_qr.rows(), right.cols());
                for (Index column = 0; column < right.cols(); ++column)
                {
                    for (Index row = 0; row < _qr.rows(); ++row)
                    {
                        Scalar value{};
                        for (Index inner = 0; inner < _qr.rows(); ++inner)
                        {
                            value += decomposition_detail::conjugateValue(_explicitQ(inner, row)) *
                                     transformed(inner, column);
                        }
                        projected(row, column) = value;
                    }
                }
                transformed = std::move(projected);
            }
            else
            {
                decomposition_detail::applyQt<Scalar>(_qr, _tau, transformed);
            }
            Matrix<Scalar, Cols, Rhs::ColsAtCompileTime> result(_qr.cols(), right.cols());
            result.setZero();
            const Index solved = std::min(_qr.rows(), _qr.cols());
            for (Index column = 0; column < right.cols(); ++column)
            {
                for (Index row = solved; row-- > 0;)
                {
                    Scalar value = transformed(row, column);
                    for (Index next = row + 1; next < solved; ++next)
                    {
                        value -= _qr(row, next) * result(next, column);
                    }
                    result(row, column) = value / _qr(row, row);
                }
            }
            return result;
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
        static void validate(const MatrixType& input)
        {
            if (!input.allFinite() || input.rows() == 0 || input.cols() == 0)
            {
                throw std::invalid_argument("HouseholderQR requires a non-empty finite matrix");
            }
        }
        MatrixType _qr;
        QType _explicitQ;
        decomposition_detail::
            StaticOrDynamicVector<Scalar, (Rows == Dynamic || Cols == Dynamic) ? Dynamic : (Rows < Cols ? Rows : Cols)>
                _tau;
        Index _rank = 0;
        ComputationInfo _info = InvalidInput;
        internal::Backend _backend = internal::Backend::Cpu;
        bool _gpuFactors = false;
        bool _initialized = false;
    };

    template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols, typename PermutationIndex>
    class ColPivHouseholderQR<Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>, PermutationIndex>
        : public SolverBase<
              ColPivHouseholderQR<Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>, PermutationIndex>>
    {
    public:
        using MatrixType = Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>;
        using RealScalar = typename NumTraits<Scalar>::Real;
        using PermutationType = PermutationMatrix<Cols, MaxCols, PermutationIndex>;
        using QType = Matrix<Scalar, Rows, Rows, ColMajor, MaxRows, MaxRows>;
        ColPivHouseholderQR() = default;
        explicit ColPivHouseholderQR(Index rows, Index columns) : _qr(rows, columns)
        {
        }
        explicit ColPivHouseholderQR(const MatrixType& input)
        {
            compute(input);
        }

        ColPivHouseholderQR& compute(const MatrixType& input)
        {
            if (internal::currentExecutionSettings().policy == internal::ExecutionPolicy::GpuRequired)
            {
                throw internal::Error(internal::ErrorCode::UnsupportedOperation,
                                      "ColPivHouseholderQR is unavailable on the selected GPU");
            }
            if (!input.allFinite() || input.rows() == 0 || input.cols() == 0)
            {
                throw std::invalid_argument("ColPivHouseholderQR requires a non-empty finite matrix");
            }
            _qr = input;
            decomposition_detail::StaticOrDynamicVector<Index, Cols> pivots(static_cast<std::size_t>(input.cols()));
            std::iota(pivots.begin(), pivots.end(), Index{});
            _rank = decomposition_detail::householderFactor<Scalar>(_qr, _tau, &pivots);
            decomposition_detail::StaticOrDynamicVector<PermutationIndex, Cols> converted(pivots.size());
            std::transform(pivots.begin(),
                           pivots.end(),
                           converted.begin(),
                           [](Index value) { return static_cast<PermutationIndex>(value); });
            _permutation = PermutationType(converted);
            _initialized = true;
            _info = Success;
            return *this;
        }

        const MatrixType& matrixQR() const noexcept
        {
            return _qr;
        }
        QType householderQ() const
        {
            return decomposition_detail::explicitQ<Scalar>(_qr, _tau);
        }
        const PermutationType& colsPermutation() const noexcept
        {
            return _permutation;
        }
        ColPivHouseholderQR& setThreshold(RealScalar value)
        {
            _prescribedThreshold = value;
            return *this;
        }
        RealScalar threshold() const noexcept
        {
            return _prescribedThreshold >= RealScalar{} ? _prescribedThreshold
                                                        : std::numeric_limits<RealScalar>::epsilon() *
                                                              static_cast<RealScalar>(std::max(_qr.rows(), _qr.cols()));
        }
        Index rank() const noexcept
        {
            if (!_initialized)
                return 0;
            RealScalar largest{};
            const Index steps = std::min(_qr.rows(), _qr.cols());
            for (Index index = 0; index < steps; ++index)
                largest = std::max(largest, std::abs(_qr(index, index)));
            Index result = 0;
            for (Index index = 0; index < steps; ++index)
                result += std::abs(_qr(index, index)) > threshold() * largest;
            return result;
        }
        Index dimensionOfKernel() const noexcept
        {
            return _qr.cols() - rank();
        }
        bool isInjective() const noexcept
        {
            return rank() == _qr.cols();
        }
        bool isSurjective() const noexcept
        {
            return rank() == _qr.rows();
        }
        bool isInvertible() const noexcept
        {
            return _qr.rows() == _qr.cols() && isInjective();
        }
        ComputationInfo info() const noexcept
        {
            return _info;
        }
        bool isInitialized() const noexcept
        {
            return _initialized;
        }

        template <typename Rhs> auto _solve(const Rhs& right) const
        {
            if (right.rows() != _qr.rows())
            {
                throw std::invalid_argument("ColPivHouseholderQR right-hand side has incompatible rows");
            }
            Matrix<Scalar, Rows, Rhs::ColsAtCompileTime> transformed(right);
            decomposition_detail::applyQt<Scalar>(_qr, _tau, transformed);
            Matrix<Scalar, Cols, Rhs::ColsAtCompileTime> result(_qr.cols(), right.cols());
            const Index numerical_rank = rank();
            for (Index column = 0; column < right.cols(); ++column)
            {
                decomposition_detail::StaticOrDynamicVector<Scalar, Cols> solution(static_cast<std::size_t>(_qr.cols()),
                                                                                   Scalar{});
                for (Index row = numerical_rank; row-- > 0;)
                {
                    Scalar value = transformed(row, column);
                    for (Index next = row + 1; next < numerical_rank; ++next)
                    {
                        value -= _qr(row, next) * solution[static_cast<std::size_t>(next)];
                    }
                    solution[static_cast<std::size_t>(row)] = value / _qr(row, row);
                }
                for (Index row = 0; row < _qr.cols(); ++row)
                {
                    result(_permutation.indices()(row), column) = solution[static_cast<std::size_t>(row)];
                }
            }
            return result;
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
        MatrixType _qr;
        decomposition_detail::
            StaticOrDynamicVector<Scalar, (Rows == Dynamic || Cols == Dynamic) ? Dynamic : (Rows < Cols ? Rows : Cols)>
                _tau;
        PermutationType _permutation;
        Index _rank = 0;
        RealScalar _prescribedThreshold = RealScalar{-1};
        ComputationInfo _info = InvalidInput;
        bool _initialized = false;
    };
} // namespace plamatrix::v1
