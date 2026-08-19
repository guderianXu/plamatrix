#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <type_traits>

namespace plamatrix
{

/// Damping controls for a Levenberg-Marquardt trust-region loop.
template <typename Scalar>
struct LevenbergMarquardtOptions
{
    Scalar initialDamping = Scalar(1e-3);
    Scalar minimumDamping = Scalar(1e-12);
    Scalar maximumDamping = Scalar(1e12);
    Scalar decreaseFactor = Scalar(0.3);
    Scalar increaseFactor = Scalar(10);
};

/**
 * @brief Track LM damping and accepted/rejected step counts.
 *
 * The caller evaluates candidate costs and invokes acceptStep() or rejectStep().
 * Construction throws when damping bounds or update factors are invalid.
 */
template <typename Scalar>
class LevenbergMarquardtStrategy
{
public:
    explicit LevenbergMarquardtStrategy(
        const LevenbergMarquardtOptions<Scalar>& options = {})
        : _options(options), _damping(options.initialDamping)
    {
        static_assert(std::is_floating_point_v<Scalar>,
                      "LevenbergMarquardtStrategy requires a floating-point scalar");
        if (!std::isfinite(_options.minimumDamping) ||
            !std::isfinite(_options.maximumDamping) ||
            !std::isfinite(_options.initialDamping) ||
            _options.minimumDamping <= Scalar(0) ||
            _options.maximumDamping < _options.minimumDamping ||
            _options.initialDamping < _options.minimumDamping ||
            _options.initialDamping > _options.maximumDamping ||
            !std::isfinite(_options.decreaseFactor) ||
            _options.decreaseFactor <= Scalar(0) ||
            _options.decreaseFactor >= Scalar(1) ||
            !std::isfinite(_options.increaseFactor) ||
            _options.increaseFactor <= Scalar(1))
        {
            throw std::invalid_argument(
                "LevenbergMarquardtStrategy: invalid damping options");
        }
    }

    /// Reduce damping after an accepted trial step.
    void acceptStep() noexcept
    {
        _damping = std::max(
            _options.minimumDamping, _damping * _options.decreaseFactor);
        ++_acceptedSteps;
    }

    /// Increase damping after a rejected or numerically invalid trial step.
    void rejectStep() noexcept
    {
        _damping = std::min(
            _options.maximumDamping, _damping * _options.increaseFactor);
        ++_rejectedSteps;
    }

    /// Return the damping for the next normal equation.
    Scalar damping() const noexcept
    {
        return _damping;
    }

    /// Return the number of accepted trial steps.
    int acceptedSteps() const noexcept
    {
        return _acceptedSteps;
    }

    /// Return the number of rejected trial steps.
    int rejectedSteps() const noexcept
    {
        return _rejectedSteps;
    }

private:
    LevenbergMarquardtOptions<Scalar> _options;
    Scalar _damping;
    int _acceptedSteps = 0;
    int _rejectedSteps = 0;
};

} // namespace plamatrix
