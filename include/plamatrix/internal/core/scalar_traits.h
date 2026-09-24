#pragma once

#include <cstdint>
#include <type_traits>

#include "plamatrix/internal/core/device.h"

namespace plamatrix::internal
{

inline namespace v1
{

template <typename Scalar> struct ScalarTraits
{
    static constexpr bool supported = false;
};

template <> struct ScalarTraits<float>
{
    static constexpr bool supported = true;
    using AccumulationType = float;
    using IndexType = Index;
};

template <> struct ScalarTraits<double>
{
    static constexpr bool supported = true;
    using AccumulationType = double;
    using IndexType = Index;
};

template <> struct ScalarTraits<std::int32_t>
{
    static constexpr bool supported = true;
    using AccumulationType = std::int64_t;
    using IndexType = Index;
};

template <> struct ScalarTraits<std::int64_t>
{
    static constexpr bool supported = true;
    using AccumulationType = std::int64_t;
    using IndexType = Index;
};

template <typename Scalar>
inline constexpr bool isSupportedScalar_v = ScalarTraits<std::remove_cv_t<Scalar>>::supported;

template <typename Scalar>
using ScalarAccumulationType = typename ScalarTraits<std::remove_cv_t<Scalar>>::AccumulationType;

} // namespace v1

} // namespace plamatrix::internal
