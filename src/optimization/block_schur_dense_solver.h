#pragma once

#include <chrono>
#include <cmath>
#include <limits>
#include <vector>

#include "plamatrix/optimization/block_schur.h"
#include "plamatrix/sparse/csr_matrix.h"

#include "block_schur_linear_algebra.h"

#ifdef PLAMATRIX_WITH_LAPACK
#include "../ops/fortran_linalg.h"
#endif

namespace plamatrix::block_schur_detail
{

template <typename Scalar>
bool solvePositiveDefiniteFallback(Index dimension,
                                   std::vector<Scalar>* matrix,
                                   std::vector<Scalar>* solution)
{
    for (Index row = 0; row < dimension; ++row)
    {
        for (Index column = 0; column <= row; ++column)
        {
            Scalar value = (*matrix)[static_cast<std::size_t>(row * dimension + column)];
            for (Index inner = 0; inner < column; ++inner)
            {
                value -= (*matrix)[static_cast<std::size_t>(row * dimension + inner)] *
                         (*matrix)[static_cast<std::size_t>(column * dimension + inner)];
            }
            if (row == column)
            {
                if (!(value > Scalar(0)) || !std::isfinite(value))
                {
                    return false;
                }
                (*matrix)[static_cast<std::size_t>(row * dimension + column)] = std::sqrt(value);
            }
            else
            {
                (*matrix)[static_cast<std::size_t>(row * dimension + column)] =
                    value / (*matrix)[static_cast<std::size_t>(column * dimension + column)];
            }
        }
    }
    for (Index row = 0; row < dimension; ++row)
    {
        Scalar value = (*solution)[static_cast<std::size_t>(row)];
        for (Index column = 0; column < row; ++column)
        {
            value -= (*matrix)[static_cast<std::size_t>(row * dimension + column)] *
                     (*solution)[static_cast<std::size_t>(column)];
        }
        (*solution)[static_cast<std::size_t>(row)] =
            value / (*matrix)[static_cast<std::size_t>(row * dimension + row)];
    }
    for (Index row = dimension; row-- > 0;)
    {
        Scalar value = (*solution)[static_cast<std::size_t>(row)];
        for (Index column = row + 1; column < dimension; ++column)
        {
            value -= (*matrix)[static_cast<std::size_t>(column * dimension + row)] *
                     (*solution)[static_cast<std::size_t>(column)];
        }
        (*solution)[static_cast<std::size_t>(row)] =
            value / (*matrix)[static_cast<std::size_t>(row * dimension + row)];
    }
    return true;
}

template <typename Scalar>
SchurComplementSolverReport<Scalar> solveReducedSchurDense(
    const CSRMatrix<Scalar, Device::CPU>& matrix,
    const std::vector<Scalar>& rhs,
    const SchurComplementSolverOptions<Scalar>& options,
    std::vector<Scalar>* solution)
{
    SchurComplementSolverReport<Scalar> report;
    report.linearBackend = SchurComplementLinearBackend::DenseCpu;
    const auto solve_start = std::chrono::steady_clock::now();
    const Index dimension = matrix.rows();
    std::vector<Scalar> dense(
        static_cast<std::size_t>(dimension * dimension), Scalar(0));
    for (Index row = 0; row < dimension; ++row)
    {
        for (Index offset = matrix.rowOffsets()[row];
             offset < matrix.rowOffsets()[row + 1]; ++offset)
        {
            dense[static_cast<std::size_t>(row * dimension + matrix.colIndices()[offset])] =
                matrix.values()[offset];
        }
    }
    *solution = rhs;
    report.initialResidualNorm = vectorNorm(rhs);
#ifdef PLAMATRIX_WITH_LAPACK
    const bool solved = detail::fortranPositiveDefiniteSolve(
        detail::checkedLapackInt(dimension, "Schur dimension"),
        dense.data(), solution->data());
#else
    const bool solved = solvePositiveDefiniteFallback(dimension, &dense, solution);
#endif
    if (!solved)
    {
        report.message = "reduced Schur complement is not positive definite";
    }
    else
    {
        std::vector<Scalar> residual = rhs;
        for (Index row = 0; row < dimension; ++row)
        {
            Scalar product = Scalar(0);
            for (Index offset = matrix.rowOffsets()[row];
                 offset < matrix.rowOffsets()[row + 1]; ++offset)
            {
                product += matrix.values()[offset] *
                           (*solution)[static_cast<std::size_t>(matrix.colIndices()[offset])];
            }
            residual[static_cast<std::size_t>(row)] -= product;
        }
        report.finalResidualNorm = vectorNorm(residual);
        const Scalar tolerance = options.absoluteTolerance +
                                 options.relativeTolerance * report.initialResidualNorm;
        report.converged = std::isfinite(report.finalResidualNorm) &&
                           report.finalResidualNorm <= std::max(
                               tolerance, Scalar(100) * std::numeric_limits<Scalar>::epsilon() *
                                              std::max(Scalar(1), report.initialResidualNorm));
        report.iterations = 1;
        if (!report.converged)
        {
            report.message = "dense Cholesky residual exceeds tolerance";
        }
    }
    report.linearSolveSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - solve_start).count();
    return report;
}

} // namespace plamatrix::block_schur_detail
