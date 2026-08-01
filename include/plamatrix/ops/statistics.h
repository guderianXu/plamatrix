#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <type_traits>
#include <vector>

namespace plamatrix
{

/**
 * @brief Return the median of finite samples, ignoring NaN and infinities.
 *
 * The input is passed by value for in-place selection. An input without a
 * finite sample returns std::nullopt.
 */
template <typename Scalar>
std::optional<Scalar> finiteMedian(std::vector<Scalar> values)
{
    static_assert(std::is_floating_point_v<Scalar>,
                  "finiteMedian requires a floating-point scalar");
    values.erase(
        std::remove_if(
            values.begin(),
            values.end(),
            [](const Scalar value)
            {
                return !std::isfinite(value);
            }),
        values.end());
    if (values.empty())
    {
        return std::nullopt;
    }

    const std::size_t middle = values.size() / 2;
    std::nth_element(
        values.begin(),
        values.begin() + static_cast<std::ptrdiff_t>(middle),
        values.end());
    if (values.size() % 2 == 1)
    {
        return values[middle];
    }

    const Scalar lower =
        *std::max_element(
            values.begin(),
            values.begin() + static_cast<std::ptrdiff_t>(middle));
    const Scalar upper = values[middle];
    if (std::signbit(lower) != std::signbit(upper))
    {
        return lower / Scalar(2) + upper / Scalar(2);
    }
    return lower + (upper - lower) / Scalar(2);
}

} // namespace plamatrix
