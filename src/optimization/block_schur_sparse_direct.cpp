#include "block_schur_sparse_direct.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "block_schur_linear_algebra.h"
#include "block_sparse_cholesky.h"

namespace plamatrix::internal
{

    bool hasSparseDirectSchurSolver() noexcept
    {
        return true;
    }

    namespace block_schur_detail
    {

        template <typename Scalar>
        SchurComplementSolverReport<Scalar>
        solveReducedSchurSparseDirect(const CsrStorage<Scalar, Device::CPU>& matrix,
                                      const std::vector<Scalar>& rhs,
                                      const SchurComplementSolverOptions<Scalar>& options,
                                      Index block_size,
                                      std::shared_ptr<void>& opaque_state,
                                      std::vector<Scalar>* solution)
        {
            SchurComplementSolverReport<Scalar> report;
            report.linearBackend = SchurComplementLinearBackend::SparseCpu;
            report.preconditionerName = "none";
            const auto solve_start = std::chrono::steady_clock::now();
            if (!solution || matrix.rows() != matrix.cols() || rhs.size() != static_cast<std::size_t>(matrix.rows()))
            {
                throw std::invalid_argument("native sparse Cholesky received incompatible dimensions");
            }
            if (!opaque_state)
            {
                opaque_state = std::make_shared<NativeBlockSparseCholesky>();
            }
            auto* state = static_cast<NativeBlockSparseCholesky*>(opaque_state.get());

            const auto symbolic_start = std::chrono::steady_clock::now();
            if (!state->ensurePattern(matrix.rows(),
                                      block_size,
                                      matrix.rowOffsets(),
                                      matrix.colIndices(),
                                      matrix.nnz(),
                                      &report.symbolicAnalysisReused,
                                      &report.message))
            {
                return report;
            }
            report.symbolicAnalysisSeconds =
                report.symbolicAnalysisReused
                    ? 0.0
                    : std::chrono::duration<double>(std::chrono::steady_clock::now() - symbolic_start).count();
            report.initialResidualNorm = vectorNorm(rhs);

            const auto factor_start = std::chrono::steady_clock::now();
            if (!state->factorize(matrix.values(), &report.message))
            {
                report.choleskyFactorizationSeconds =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - factor_start).count();
                report.linearSolveSeconds =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - solve_start).count();
                return report;
            }
            report.choleskyFactorizationSeconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - factor_start).count();

            std::vector<double> rhs_double(rhs.begin(), rhs.end());
            std::vector<double> solution_double(rhs.size());
            const auto triangular_start = std::chrono::steady_clock::now();
            if (!state->solve(rhs_double.data(), solution_double.data(), &report.message))
            {
                report.triangularSolveSeconds =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - triangular_start).count();
                report.linearSolveSeconds =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - solve_start).count();
                return report;
            }
            report.triangularSolveSeconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - triangular_start).count();
            solution->assign(solution_double.begin(), solution_double.end());

            std::vector<Scalar> residual = rhs;
            const auto evaluate_residual = [&]()
            {
                const auto residual_start = std::chrono::steady_clock::now();
                residual = rhs;
                for (Index row = 0; row < matrix.rows(); ++row)
                {
                    Scalar product = Scalar(0);
                    for (Index offset = matrix.rowOffsets()[row]; offset < matrix.rowOffsets()[row + 1]; ++offset)
                    {
                        product += matrix.values()[offset] *
                                   (*solution)[static_cast<std::size_t>(matrix.colIndices()[offset])];
                    }
                    residual[static_cast<std::size_t>(row)] -= product;
                }
                report.finalResidualNorm = vectorNorm(residual);
                report.residualCheckSeconds +=
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - residual_start).count();
            };
            evaluate_residual();
            const Scalar tolerance = options.absoluteTolerance + options.relativeTolerance * report.initialResidualNorm;
            const Scalar effective_tolerance = std::max(tolerance,
                                                        Scalar(100) * std::numeric_limits<Scalar>::epsilon() *
                                                            std::max(Scalar(1), report.initialResidualNorm));
            report.iterations = 1;
            constexpr int maximum_refinement_steps = 3;
            std::vector<double> correction_rhs(rhs.size());
            std::vector<double> correction(rhs.size());
            while (std::isfinite(report.finalResidualNorm) && report.finalResidualNorm > effective_tolerance &&
                   report.iterations <= maximum_refinement_steps)
            {
                std::copy(residual.begin(), residual.end(), correction_rhs.begin());
                const auto refinement_start = std::chrono::steady_clock::now();
                const bool refined = state->solve(correction_rhs.data(), correction.data(), &report.message);
                report.triangularSolveSeconds +=
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - refinement_start).count();
                if (!refined)
                {
                    break;
                }
                for (std::size_t index = 0; index < solution->size(); ++index)
                {
                    (*solution)[index] += static_cast<Scalar>(correction[index]);
                }
                ++report.iterations;
                evaluate_residual();
            }
            report.converged =
                std::isfinite(report.finalResidualNorm) && report.finalResidualNorm <= effective_tolerance;
            if (!report.converged)
            {
                report.message = "native sparse Cholesky residual exceeds tolerance";
            }
            report.linearSolveSeconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - solve_start).count();
            return report;
        }

        template SchurComplementSolverReport<float>
        solveReducedSchurSparseDirect(const CsrStorage<float, Device::CPU>&,
                                      const std::vector<float>&,
                                      const SchurComplementSolverOptions<float>&,
                                      Index,
                                      std::shared_ptr<void>&,
                                      std::vector<float>*);

        template SchurComplementSolverReport<double>
        solveReducedSchurSparseDirect(const CsrStorage<double, Device::CPU>&,
                                      const std::vector<double>&,
                                      const SchurComplementSolverOptions<double>&,
                                      Index,
                                      std::shared_ptr<void>&,
                                      std::vector<double>*);

    } // namespace block_schur_detail
} // namespace plamatrix::internal
