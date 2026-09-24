#pragma once

#include <stdexcept>

#include "plamatrix/dense/matrix.h"

namespace plamatrix::v1
{
    /// Common CRTP interface for sparse decompositions and iterative solvers.
    template <typename Derived> class SparseSolverBase
    {
    public:
        SparseSolverBase() = default;
        SparseSolverBase(const SparseSolverBase&) = delete;
        SparseSolverBase& operator=(const SparseSolverBase&) = delete;
        SparseSolverBase(SparseSolverBase&&) = default;
        SparseSolverBase& operator=(SparseSolverBase&&) = default;
        ~SparseSolverBase() = default;

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
            if (!_isInitialized)
            {
                throw std::logic_error("sparse solver must be factorized before solve");
            }
            if (derived().rows() != right.rows())
            {
                throw std::invalid_argument("sparse solver right-hand side has incompatible rows");
            }
            return derived()._solve(right.derived());
        }

    protected:
        bool _isInitialized = false;
    };
} // namespace plamatrix::v1
