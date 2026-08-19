#pragma once

#include <cmath>
#include <stdexcept>
#include <type_traits>

namespace plamatrix
{

/// Robust block-loss value and the corresponding normal-equation weight.
template <typename Scalar>
struct RobustLossEvaluation
{
    Scalar cost = Scalar(0);
    Scalar weight = Scalar(1);
};

/**
 * @brief Evaluate a Huber loss for one residual block.
 *
 * The returned cost follows the least-squares convention `0.5 * rho(s)`, where
 * `s` is the squared residual norm. `weight` is `rho'(s)` and can be applied to
 * `J^T J` and `J^T r` when building an iteratively reweighted normal equation.
 * A zero delta disables robustification. Invalid or non-finite arguments throw.
 */
template <typename Scalar>
RobustLossEvaluation<Scalar> evaluateHuberLoss(Scalar squared_residual_norm, Scalar delta)
{
    static_assert(std::is_floating_point_v<Scalar>,
                  "evaluateHuberLoss requires a floating-point scalar");
    if (!std::isfinite(squared_residual_norm) || squared_residual_norm < Scalar(0))
    {
        throw std::invalid_argument(
            "evaluateHuberLoss: squared residual norm must be finite and non-negative");
    }
    if (!std::isfinite(delta) || delta < Scalar(0))
    {
        throw std::invalid_argument(
            "evaluateHuberLoss: delta must be finite and non-negative");
    }
    if (delta == Scalar(0) || squared_residual_norm <= delta * delta)
    {
        return {Scalar(0.5) * squared_residual_norm, Scalar(1)};
    }

    const Scalar residual_norm = std::sqrt(squared_residual_norm);
    return {
        delta * residual_norm - Scalar(0.5) * delta * delta,
        delta / residual_norm,
    };
}

} // namespace plamatrix
