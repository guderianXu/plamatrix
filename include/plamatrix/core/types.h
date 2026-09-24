#pragma once

#include <cstddef>
#include <complex>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace plamatrix
{
    inline namespace v1
    {
        using Index = std::int64_t;

        inline constexpr int Dynamic = -1;
        inline constexpr int DynamicIndex = 0xffffff;
        inline constexpr int Infinity = -1;

        enum NaNPropagationOptions
        {
            PropagateFast = 0,
            PropagateNaN,
            PropagateNumbers
        };

        enum StorageOptions
        {
            ColMajor = 0,
            RowMajor = 0x1,
            AutoAlign = 0,
            DontAlign = 0x2
        };

        enum AlignmentType
        {
            Unaligned = 0,
            Aligned8 = 8,
            Aligned16 = 16,
            Aligned32 = 32,
            Aligned64 = 64,
            Aligned128 = 128,
            AlignedMask = 255,
            Aligned = Aligned16
        };

        enum ComputationInfo
        {
            Success = 0,
            NumericalIssue = 1,
            NoConvergence = 2,
            InvalidInput = 3
        };

        enum TransformTraits
        {
            Isometry = 0x1,
            Affine = 0x2,
            AffineCompact = 0x10 | Affine,
            Projective = 0x20
        };

        /// Selects the scalar returned by a binary coefficient-wise operation.
        /// Applications may specialize this trait for automatic-differentiation or
        /// other user-defined scalar pairs, following Eigen's customization model.
        template <typename ScalarA, typename ScalarB, typename BinaryOp = void, typename = void>
        struct ScalarBinaryOpTraits
        {
        };

        template <typename ScalarA, typename ScalarB, typename BinaryOp>
        struct ScalarBinaryOpTraits<ScalarA, ScalarB, BinaryOp, std::void_t<std::common_type_t<ScalarA, ScalarB>>>
        {
            using ReturnType = std::common_type_t<ScalarA, ScalarB>;
        };

        template <typename T> struct NumTraits
        {
            using Real = T;
            using NonInteger = T;
            using Nested = T;
            using Literal = T;
            static constexpr bool IsComplex = false;
            static constexpr bool IsInteger = std::is_integral_v<T>;
            static constexpr bool IsSigned = std::is_signed_v<T>;
            static constexpr int RequireInitialization = !std::is_arithmetic_v<T>;
            static constexpr int ReadCost = 1;
            static constexpr int AddCost = 1;
            static constexpr int MulCost = 1;
            static constexpr Real epsilon() noexcept
            {
                return std::numeric_limits<Real>::epsilon();
            }
            static constexpr Real dummy_precision() noexcept
            {
                return std::is_same_v<Real, float>    ? Real{1.0e-5F}
                       : std::is_same_v<Real, double> ? Real{1.0e-12}
                                                      : epsilon();
            }
            static constexpr int digits10() noexcept
            {
                return std::numeric_limits<Real>::digits10;
            }
            static constexpr Real highest() noexcept
            {
                return (std::numeric_limits<Real>::max)();
            }
            static constexpr Real lowest() noexcept
            {
                return (std::numeric_limits<Real>::lowest)();
            }
            static constexpr Real infinity() noexcept
            {
                return std::numeric_limits<Real>::infinity();
            }
            static constexpr Real quiet_NaN() noexcept
            {
                return std::numeric_limits<Real>::quiet_NaN();
            }
        };

        template <typename T> struct NumTraits<std::complex<T>> : NumTraits<T>
        {
            using Real = T;
            using NonInteger = std::complex<T>;
            using Nested = std::complex<T>;
            using Literal = std::complex<T>;
            static constexpr bool IsComplex = true;
            static constexpr bool IsInteger = false;
            static constexpr bool IsSigned = true;
            static constexpr int RequireInitialization = true;
            static constexpr int ReadCost = 2;
            static constexpr int AddCost = 2;
            static constexpr int MulCost = 6;
        };
    } // namespace v1
} // namespace plamatrix
