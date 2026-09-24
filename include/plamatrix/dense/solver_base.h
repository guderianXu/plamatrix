#pragma once

#include <stdexcept>

#include "plamatrix/dense/matrix_base.h"

namespace plamatrix::v1
{
    template <typename Derived> class SolverBase
    {
    public:
        Derived& derived() noexcept
        {
            return *static_cast<Derived*>(this);
        }

        const Derived& derived() const noexcept
        {
            return *static_cast<const Derived*>(this);
        }

        template <typename Rhs> auto solve(const MatrixBase<Rhs>& right) const
        {
            if (!derived().isInitialized())
            {
                throw std::logic_error("dense solver must be computed before solve");
            }
            return derived()._solve(right.derived());
        }
    };
} // namespace plamatrix::v1
