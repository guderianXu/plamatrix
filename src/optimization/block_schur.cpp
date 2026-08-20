#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "plamatrix/optimization/block_schur.h"

#include "block_schur_accelerated.h"
#include "block_schur_cpu_solver.h"
#include "block_schur_dense_solver.h"
#include "block_schur_linear_algebra.h"
#include "block_schur_sparse_assembly.h"

#ifdef PLAMATRIX_WITH_CUDA
#include <cuda_runtime_api.h>

#include "plamatrix/core/error_check.h"
#endif

namespace plamatrix
{
namespace
{

using block_schur_detail::addMatrixVector;
using block_schur_detail::addTransposeMatrixVector;
using block_schur_detail::invertPositiveDefinite;
using block_schur_detail::multiplyMatrixVector;

} // namespace
template <typename Scalar>
SchurComplementSolverReport<Scalar> solveDampedSchurComplement(
    const BlockNormalEquations<Scalar>& equations,
    Scalar damping,
    const SchurComplementSolverOptions<Scalar>& options,
    std::vector<Scalar>* primary_step,
    std::vector<Scalar>* eliminated_step)
{
    SchurComplementSolverWorkspace<Scalar> workspace;
    return solveDampedSchurComplement(
        equations,
        damping,
        options,
        workspace,
        primary_step,
        eliminated_step);
}

template <typename Scalar>
SchurComplementSolverReport<Scalar> solveDampedSchurComplement(
    const BlockNormalEquations<Scalar>& equations,
    Scalar damping,
    const SchurComplementSolverOptions<Scalar>& options,
    SchurComplementSolverWorkspace<Scalar>& workspace,
    std::vector<Scalar>* primary_step,
    std::vector<Scalar>* eliminated_step)
{
    static_assert(std::is_floating_point_v<Scalar>,
                  "solveDampedSchurComplement requires a floating-point scalar");
    if (!primary_step || !eliminated_step || primary_step == eliminated_step)
    {
        throw std::invalid_argument(
            "solveDampedSchurComplement: output vectors must be non-null and distinct");
    }
    if (!std::isfinite(damping) || damping < Scalar(0) || options.maxIterations <= 0 ||
        !std::isfinite(options.relativeTolerance) || options.relativeTolerance < Scalar(0) ||
        !std::isfinite(options.absoluteTolerance) || options.absoluteTolerance < Scalar(0) ||
        !std::isfinite(options.minimumDiagonal) || options.minimumDiagonal <= Scalar(0))
    {
        throw std::invalid_argument("solveDampedSchurComplement: invalid solver options");
    }

    SchurComplementSolverReport<Scalar> report;
    report.linearBackend = options.linearBackend;
    const Index primary_count = equations._primaryBlockCount;
    const Index eliminated_count = equations._eliminatedBlockCount;
    const Index primary_size = equations._primaryBlockSize;
    const Index eliminated_size = equations._eliminatedBlockSize;
    const Index primary_dimension = primary_count * primary_size;
    const Index eliminated_dimension = eliminated_count * eliminated_size;
    if (!options.useInitialGuess ||
        primary_step->size() != static_cast<std::size_t>(primary_dimension))
    {
        primary_step->assign(static_cast<std::size_t>(primary_dimension), Scalar(0));
    }
    eliminated_step->assign(static_cast<std::size_t>(eliminated_dimension), Scalar(0));

    auto& primary_diagonal =
        block_schur_detail::SchurComplementSolverWorkspaceAccess::hostPrimaryDiagonal(workspace);
    auto& eliminated_diagonal =
        block_schur_detail::SchurComplementSolverWorkspaceAccess::hostEliminatedDiagonal(workspace);
    auto& eliminated_inverse =
        block_schur_detail::SchurComplementSolverWorkspaceAccess::hostEliminatedInverse(workspace);
    auto& reduced_rhs =
        block_schur_detail::SchurComplementSolverWorkspaceAccess::hostReducedRhs(workspace);
    auto& host_scratch =
        block_schur_detail::SchurComplementSolverWorkspaceAccess::hostScratch(workspace);
    primary_diagonal.assign(
        equations._primaryDiagonal.begin(), equations._primaryDiagonal.end());
    eliminated_diagonal.assign(
        equations._eliminatedDiagonal.begin(), equations._eliminatedDiagonal.end());
    const auto apply_damping = [&](std::vector<Scalar>& blocks, Index count, Index size)
    {
        for (Index block = 0; block < count; ++block)
        {
            const Index offset = block * size * size;
            for (Index diagonal = 0; diagonal < size; ++diagonal)
            {
                Scalar& value = blocks[static_cast<std::size_t>(
                    offset + diagonal * size + diagonal)];
                value += damping * std::max(std::abs(value), options.minimumDiagonal);
            }
        }
    };
    apply_damping(primary_diagonal, primary_count, primary_size);
    apply_damping(eliminated_diagonal, eliminated_count, eliminated_size);

    const auto inverse_start = std::chrono::steady_clock::now();
    eliminated_inverse.resize(static_cast<std::size_t>(
        eliminated_count * eliminated_size * eliminated_size));
    host_scratch.resize(static_cast<std::size_t>(
        eliminated_size * eliminated_size + 3 * eliminated_size));
    Scalar* inverse_lower = host_scratch.data();
    Scalar* inverse_intermediate = inverse_lower + eliminated_size * eliminated_size;
    for (Index block = 0; block < eliminated_count; ++block)
    {
        const Scalar* matrix = eliminated_diagonal.data() + block * eliminated_size * eliminated_size;
        Scalar* inverse = eliminated_inverse.data() +
            static_cast<std::size_t>(block * eliminated_size * eliminated_size);
        const bool inverted = eliminated_size == 3
            ? block_schur_detail::invertPositiveDefinite3x3(matrix, inverse)
            : block_schur_detail::invertPositiveDefiniteInto(
                  matrix,
                  eliminated_size,
                  inverse,
                  inverse_lower,
                  inverse_intermediate);
        if (!inverted)
        {
            report.message = "eliminated block is not positive definite";
            return report;
        }
    }
    report.smallBlockInverseSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - inverse_start).count();

    reduced_rhs.assign(static_cast<std::size_t>(primary_dimension), Scalar(0));
    for (Index index = 0; index < primary_dimension; ++index)
    {
        reduced_rhs[static_cast<std::size_t>(index)] =
            -equations._primaryGradient[static_cast<std::size_t>(index)];
    }
    Scalar* eliminated_product = host_scratch.data();
    for (Index block = 0; block < eliminated_count; ++block)
    {
        multiplyMatrixVector(eliminated_inverse.data() + static_cast<std::size_t>(
                                 block * eliminated_size * eliminated_size),
                             eliminated_size,
                             eliminated_size,
                             equations._eliminatedGradient.data() + block * eliminated_size,
                             eliminated_product);
        for (const std::size_t cross_index :
             equations._eliminatedAdjacency[static_cast<std::size_t>(block)])
        {
            const auto& cross = equations._crossBlocks[cross_index];
            addMatrixVector(cross.values.data(),
                            primary_size,
                            eliminated_size,
                            eliminated_product,
                            Scalar(1),
                            reduced_rhs.data() + cross.primaryBlock * primary_size);
        }
    }

    std::vector<std::vector<Scalar>> preconditioner_inverse;
    if (options.linearBackend != SchurComplementLinearBackend::DenseCpu)
    {
        preconditioner_inverse.resize(static_cast<std::size_t>(primary_count));
        std::vector<std::vector<std::size_t>> primary_adjacency(
            static_cast<std::size_t>(primary_count));
        for (std::size_t index = 0; index < equations._crossBlocks.size(); ++index)
        {
            primary_adjacency[static_cast<std::size_t>(
                equations._crossBlocks[index].primaryBlock)].push_back(index);
        }
        std::vector<Scalar> schur_diagonal(
            static_cast<std::size_t>(primary_size * primary_size));
        std::vector<Scalar> temporary_matrix(
            static_cast<std::size_t>(primary_size * eliminated_size));
        const auto preconditioner_inverse_start = std::chrono::steady_clock::now();
        for (Index block = 0; block < primary_count; ++block)
        {
            const Scalar* source = primary_diagonal.data() + block * primary_size * primary_size;
            std::copy(source, source + primary_size * primary_size, schur_diagonal.begin());
            for (const std::size_t cross_index :
                 primary_adjacency[static_cast<std::size_t>(block)])
            {
                const auto& cross = equations._crossBlocks[cross_index];
                const Scalar* inverse = eliminated_inverse.data() + static_cast<std::size_t>(
                    cross.eliminatedBlock * eliminated_size * eliminated_size);
                if (primary_size == 9 && eliminated_size == 3)
                {
                    block_schur_detail::transform9x3(
                        inverse, cross.values.data(), temporary_matrix.data());
                }
                else for (Index row = 0; row < primary_size; ++row)
                {
                    multiplyMatrixVector(inverse,
                                         eliminated_size,
                                         eliminated_size,
                                         cross.values.data() + row * eliminated_size,
                                         temporary_matrix.data() + row * eliminated_size);
                }
                for (Index row = 0; row < primary_size; ++row)
                {
                    for (Index col = 0; col < primary_size; ++col)
                    {
                        const Scalar* transformed = temporary_matrix.data() +
                            static_cast<std::size_t>(row * eliminated_size);
                        const Scalar* source_row = cross.values.data() +
                            static_cast<std::size_t>(col * eliminated_size);
                        Scalar value = eliminated_size == 3
                            ? block_schur_detail::dot3(transformed, source_row)
                            : Scalar(0);
                        for (Index inner = 0;
                             inner < (eliminated_size == 3 ? 0 : eliminated_size);
                             ++inner)
                        {
                            value += transformed[inner] * source_row[inner];
                        }
                        schur_diagonal[static_cast<std::size_t>(
                            row * primary_size + col)] -= value;
                    }
                }
            }
            if (!invertPositiveDefinite(
                    schur_diagonal.data(), primary_size,
                    &preconditioner_inverse[static_cast<std::size_t>(block)]))
            {
                report.message = "Schur preconditioner block is not positive definite";
                return report;
            }
        }
        report.smallBlockInverseSeconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - preconditioner_inverse_start).count();
    }

    Scalar* point_value = host_scratch.data() + eliminated_size;
    Scalar* inverse_value = point_value + eliminated_size;
    const auto apply_schur = [&](const std::vector<Scalar>& input, std::vector<Scalar>* output)
    {
        output->assign(static_cast<std::size_t>(primary_dimension), Scalar(0));
        for (Index block = 0; block < primary_count; ++block)
        {
            multiplyMatrixVector(primary_diagonal.data() + block * primary_size * primary_size,
                                 primary_size,
                                 primary_size,
                                 input.data() + block * primary_size,
                                 output->data() + block * primary_size);
        }
        for (const auto& cross : equations._primaryCrossBlocks)
        {
            addMatrixVector(cross.values.data(),
                            primary_size,
                            primary_size,
                            input.data() + cross.columnBlock * primary_size,
                            Scalar(1),
                            output->data() + cross.rowBlock * primary_size);
            addTransposeMatrixVector(cross.values.data(),
                                     primary_size,
                                     primary_size,
                                     input.data() + cross.rowBlock * primary_size,
                                     output->data() + cross.columnBlock * primary_size);
        }
        for (Index block = 0; block < eliminated_count; ++block)
        {
            std::fill(point_value, point_value + eliminated_size, Scalar(0));
            const auto& adjacency = equations._eliminatedAdjacency[static_cast<std::size_t>(block)];
            for (const std::size_t cross_index : adjacency)
            {
                const auto& cross = equations._crossBlocks[cross_index];
                addTransposeMatrixVector(cross.values.data(),
                                         primary_size,
                                         eliminated_size,
                                         input.data() + cross.primaryBlock * primary_size,
                                         point_value);
            }
            multiplyMatrixVector(eliminated_inverse.data() + static_cast<std::size_t>(
                                     block * eliminated_size * eliminated_size),
                                 eliminated_size,
                                 eliminated_size,
                                 point_value,
                                 inverse_value);
            for (const std::size_t cross_index : adjacency)
            {
                const auto& cross = equations._crossBlocks[cross_index];
                addMatrixVector(cross.values.data(),
                                primary_size,
                                eliminated_size,
                                inverse_value,
                                Scalar(-1),
                                output->data() + cross.primaryBlock * primary_size);
            }
        }
    };

    const auto apply_preconditioner = [&](const std::vector<Scalar>& input,
                                          std::vector<Scalar>* output)
    {
        for (Index block = 0; block < primary_count; ++block)
        {
            multiplyMatrixVector(preconditioner_inverse[static_cast<std::size_t>(block)].data(),
                                 primary_size,
                                 primary_size,
                                 input.data() + block * primary_size,
                                 output->data() + block * primary_size);
        }
    };
    const double small_block_inverse_seconds = report.smallBlockInverseSeconds;

    if (primary_dimension == 0)
    {
        report.converged = true;
    }
    else if (options.linearBackend == SchurComplementLinearBackend::DenseCpu)
    {
        const auto assembly_start = std::chrono::steady_clock::now();
        bool pattern_reused = false;
        double accumulation_seconds = 0.0;
        std::vector<Scalar>* dense_schur = nullptr;
        std::vector<Scalar>* dense_reference = nullptr;
        block_schur_detail::assembleReducedSchurDenseLower(
            primary_count,
            eliminated_count,
            primary_size,
            eliminated_size,
            primary_diagonal,
            eliminated_inverse,
            equations._primaryCrossBlocks,
            equations._crossBlocks,
            equations._eliminatedAdjacency,
            workspace,
            &pattern_reused,
            &accumulation_seconds,
            &dense_schur,
            &dense_reference);
        const double assembly_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - assembly_start).count();
        report = block_schur_detail::solveReducedSchurDenseLower(
            dense_schur, *dense_reference, reduced_rhs, options, primary_step);
        report.smallBlockInverseSeconds = small_block_inverse_seconds;
        report.schurAccumulationSeconds = accumulation_seconds;
        report.schurAssemblySeconds = assembly_seconds;
        report.schurPatternReused = pattern_reused;
    }
    else if (options.linearBackend != SchurComplementLinearBackend::Cpu)
    {
#ifdef PLAMATRIX_WITH_CUDA
        if (options.linearBackend == SchurComplementLinearBackend::Cuda &&
            options.deviceIndex >= 0)
        {
            PLAMATRIX_CHECK_CUDA(cudaSetDevice(options.deviceIndex));
        }
#endif
        const auto assembly_start = std::chrono::steady_clock::now();
        double accumulation_seconds = 0.0;
        double csr_conversion_seconds = 0.0;
        auto schur_matrix = block_schur_detail::assembleReducedSchurCsr(
            primary_count,
            eliminated_count,
            primary_size,
            eliminated_size,
            primary_diagonal,
            eliminated_inverse,
            equations._primaryCrossBlocks,
            equations._crossBlocks,
            equations._eliminatedAdjacency,
            workspace,
            &report.schurPatternReused,
            options.linearBackend,
            &report.schurAssemblyOnDevice,
            &accumulation_seconds,
            &csr_conversion_seconds);
        const double assembly_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - assembly_start).count();
        auto accelerated_report = block_schur_detail::solveAcceleratedReducedSchur(
            schur_matrix,
            reduced_rhs,
            preconditioner_inverse,
            primary_size,
            options,
            workspace,
            primary_step);
        accelerated_report.schurAssemblySeconds = assembly_seconds;
        accelerated_report.smallBlockInverseSeconds = small_block_inverse_seconds;
        accelerated_report.schurAccumulationSeconds = accumulation_seconds;
        accelerated_report.csrConversionSeconds = csr_conversion_seconds;
        accelerated_report.schurPatternReused = report.schurPatternReused;
        accelerated_report.schurAssemblyOnDevice = report.schurAssemblyOnDevice;
        report = std::move(accelerated_report);
    }
    else
    {
        report = block_schur_detail::solveReducedSchurOnCpu(
            reduced_rhs, options, apply_schur, apply_preconditioner, primary_step);
        report.smallBlockInverseSeconds = small_block_inverse_seconds;
    }

    if (!report.converged)
    {
        if (report.message.empty())
        {
            report.message = "PCG iteration limit reached";
        }
        return report;
    }

    const auto back_substitution_start = std::chrono::steady_clock::now();
    Scalar* point_rhs = host_scratch.data();
    for (Index block = 0; block < eliminated_count; ++block)
    {
        for (Index index = 0; index < eliminated_size; ++index)
        {
            point_rhs[index] =
                -equations._eliminatedGradient[static_cast<std::size_t>(block * eliminated_size + index)];
        }
        for (const std::size_t cross_index :
             equations._eliminatedAdjacency[static_cast<std::size_t>(block)])
        {
            const auto& cross = equations._crossBlocks[cross_index];
            const Scalar* primary =
                primary_step->data() + cross.primaryBlock * primary_size;
            for (Index column = 0; column < eliminated_size; ++column)
            {
                Scalar value = Scalar(0);
                for (Index row = 0; row < primary_size; ++row)
                {
                    value += cross.values[static_cast<std::size_t>(
                                 row * eliminated_size + column)] * primary[row];
                }
                point_rhs[column] -= value;
            }
        }
        multiplyMatrixVector(eliminated_inverse.data() + static_cast<std::size_t>(
                                 block * eliminated_size * eliminated_size),
                             eliminated_size,
                             eliminated_size,
                             point_rhs,
                             eliminated_step->data() + block * eliminated_size);
    }
    report.backSubstitutionSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - back_substitution_start).count();
    report.message = "converged";
    return report;
}
template SchurComplementSolverReport<float> solveDampedSchurComplement(
    const BlockNormalEquations<float>&,
    float,
    const SchurComplementSolverOptions<float>&,
    std::vector<float>*,
    std::vector<float>*);
template SchurComplementSolverReport<double> solveDampedSchurComplement(
    const BlockNormalEquations<double>&,
    double,
    const SchurComplementSolverOptions<double>&,
    std::vector<double>*,
    std::vector<double>*);
template SchurComplementSolverReport<float> solveDampedSchurComplement(
    const BlockNormalEquations<float>&,
    float,
    const SchurComplementSolverOptions<float>&,
    SchurComplementSolverWorkspace<float>&,
    std::vector<float>*,
    std::vector<float>*);
template SchurComplementSolverReport<double> solveDampedSchurComplement(
    const BlockNormalEquations<double>&,
    double,
    const SchurComplementSolverOptions<double>&,
    SchurComplementSolverWorkspace<double>&,
    std::vector<double>*,
    std::vector<double>*);

} // namespace plamatrix
