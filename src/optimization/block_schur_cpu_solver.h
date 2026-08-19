#pragma once

#include <chrono>
#include <cmath>
#include <vector>

#include "plamatrix/optimization/block_schur.h"

#include "block_schur_linear_algebra.h"

namespace plamatrix::block_schur_detail
{

template <typename Scalar, typename ApplySchur, typename ApplyPreconditioner>
SchurComplementSolverReport<Scalar> solveReducedSchurOnCpu(
    const std::vector<Scalar>& rhs,
    const SchurComplementSolverOptions<Scalar>& options,
    ApplySchur&& apply_schur,
    ApplyPreconditioner&& apply_preconditioner,
    std::vector<Scalar>* solution)
{
    SchurComplementSolverReport<Scalar> report;
    report.linearBackend = SchurComplementLinearBackend::Cpu;
    const auto solve_start = std::chrono::steady_clock::now();
    std::vector<Scalar> residual = rhs;
    report.initialResidualNorm = vectorNorm(residual);
    report.finalResidualNorm = report.initialResidualNorm;
    const Scalar tolerance = options.absoluteTolerance +
                             options.relativeTolerance * report.initialResidualNorm;
    if (report.initialResidualNorm <= tolerance)
    {
        report.converged = true;
    }
    else
    {
        std::vector<Scalar> preconditioned(rhs.size(), Scalar(0));
        apply_preconditioner(residual, &preconditioned);
        std::vector<Scalar> direction = preconditioned;
        std::vector<Scalar> product(rhs.size(), Scalar(0));
        Scalar residual_dot = dotProduct(residual, preconditioned);
        for (int iteration = 0; iteration < options.maxIterations; ++iteration)
        {
            apply_schur(direction, &product);
            const Scalar denominator = dotProduct(direction, product);
            if (!(denominator > Scalar(0)) || !std::isfinite(denominator))
            {
                report.message = "reduced Schur complement is not positive definite";
                break;
            }
            const Scalar alpha = residual_dot / denominator;
            for (std::size_t index = 0; index < rhs.size(); ++index)
            {
                (*solution)[index] += alpha * direction[index];
                residual[index] -= alpha * product[index];
            }
            report.iterations = iteration + 1;
            report.finalResidualNorm = vectorNorm(residual);
            if (report.finalResidualNorm <= tolerance)
            {
                report.converged = true;
                break;
            }
            apply_preconditioner(residual, &preconditioned);
            const Scalar next_residual_dot = dotProduct(residual, preconditioned);
            const Scalar beta = next_residual_dot / residual_dot;
            for (std::size_t index = 0; index < rhs.size(); ++index)
            {
                direction[index] = preconditioned[index] + beta * direction[index];
            }
            residual_dot = next_residual_dot;
        }
    }
    report.linearSolveSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - solve_start).count();
    return report;
}

} // namespace plamatrix::block_schur_detail
