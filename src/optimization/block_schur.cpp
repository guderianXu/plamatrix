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

    std::vector<Scalar> primary_diagonal = equations._primaryDiagonal;
    std::vector<Scalar> eliminated_diagonal = equations._eliminatedDiagonal;
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

    std::vector<std::vector<Scalar>> eliminated_inverse(
        static_cast<std::size_t>(eliminated_count));
    for (Index block = 0; block < eliminated_count; ++block)
    {
        const Scalar* matrix = eliminated_diagonal.data() + block * eliminated_size * eliminated_size;
        if (!invertPositiveDefinite(
                matrix, eliminated_size, &eliminated_inverse[static_cast<std::size_t>(block)]))
        {
            report.message = "eliminated block is not positive definite";
            return report;
        }
    }

    std::vector<Scalar> reduced_rhs(static_cast<std::size_t>(primary_dimension), Scalar(0));
    for (Index index = 0; index < primary_dimension; ++index)
    {
        reduced_rhs[static_cast<std::size_t>(index)] =
            -equations._primaryGradient[static_cast<std::size_t>(index)];
    }
    std::vector<Scalar> eliminated_product(static_cast<std::size_t>(eliminated_size));
    for (Index block = 0; block < eliminated_count; ++block)
    {
        multiplyMatrixVector(eliminated_inverse[static_cast<std::size_t>(block)].data(),
                             eliminated_size,
                             eliminated_size,
                             equations._eliminatedGradient.data() + block * eliminated_size,
                             eliminated_product.data());
        for (const std::size_t cross_index :
             equations._eliminatedAdjacency[static_cast<std::size_t>(block)])
        {
            const auto& cross = equations._crossBlocks[cross_index];
            addMatrixVector(cross.values.data(),
                            primary_size,
                            eliminated_size,
                            eliminated_product.data(),
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
        for (Index block = 0; block < primary_count; ++block)
        {
            const Scalar* source = primary_diagonal.data() + block * primary_size * primary_size;
            std::copy(source, source + primary_size * primary_size, schur_diagonal.begin());
            for (const std::size_t cross_index :
                 primary_adjacency[static_cast<std::size_t>(block)])
            {
                const auto& cross = equations._crossBlocks[cross_index];
                const auto& inverse = eliminated_inverse[
                    static_cast<std::size_t>(cross.eliminatedBlock)];
                for (Index row = 0; row < primary_size; ++row)
                {
                    multiplyMatrixVector(inverse.data(),
                                         eliminated_size,
                                         eliminated_size,
                                         cross.values.data() + row * eliminated_size,
                                         temporary_matrix.data() + row * eliminated_size);
                }
                for (Index row = 0; row < primary_size; ++row)
                {
                    for (Index col = 0; col < primary_size; ++col)
                    {
                        Scalar value = Scalar(0);
                        for (Index inner = 0; inner < eliminated_size; ++inner)
                        {
                            value += temporary_matrix[static_cast<std::size_t>(
                                         row * eliminated_size + inner)] *
                                     cross.values[static_cast<std::size_t>(
                                         col * eliminated_size + inner)];
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
    }

    std::vector<Scalar> point_value(static_cast<std::size_t>(eliminated_size), Scalar(0));
    std::vector<Scalar> inverse_value(static_cast<std::size_t>(eliminated_size), Scalar(0));
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
            std::fill(point_value.begin(), point_value.end(), Scalar(0));
            const auto& adjacency = equations._eliminatedAdjacency[static_cast<std::size_t>(block)];
            for (const std::size_t cross_index : adjacency)
            {
                const auto& cross = equations._crossBlocks[cross_index];
                addTransposeMatrixVector(cross.values.data(),
                                         primary_size,
                                         eliminated_size,
                                         input.data() + cross.primaryBlock * primary_size,
                                         point_value.data());
            }
            multiplyMatrixVector(eliminated_inverse[static_cast<std::size_t>(block)].data(),
                                 eliminated_size,
                                 eliminated_size,
                                 point_value.data(),
                                 inverse_value.data());
            for (const std::size_t cross_index : adjacency)
            {
                const auto& cross = equations._crossBlocks[cross_index];
                addMatrixVector(cross.values.data(),
                                primary_size,
                                eliminated_size,
                                inverse_value.data(),
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

    if (primary_dimension == 0)
    {
        report.converged = true;
    }
    else if (options.linearBackend == SchurComplementLinearBackend::DenseCpu)
    {
        const auto assembly_start = std::chrono::steady_clock::now();
        bool pattern_reused = false;
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
            &pattern_reused,
            SchurComplementLinearBackend::Cpu,
            &report.schurAssemblyOnDevice);
        const double assembly_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - assembly_start).count();
        report = block_schur_detail::solveReducedSchurDense(
            schur_matrix, reduced_rhs, options, primary_step);
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
            &report.schurAssemblyOnDevice);
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
        accelerated_report.schurPatternReused = report.schurPatternReused;
        accelerated_report.schurAssemblyOnDevice = report.schurAssemblyOnDevice;
        report = std::move(accelerated_report);
    }
    else
    {
        report = block_schur_detail::solveReducedSchurOnCpu(
            reduced_rhs, options, apply_schur, apply_preconditioner, primary_step);
    }

    if (!report.converged)
    {
        if (report.message.empty())
        {
            report.message = "PCG iteration limit reached";
        }
        return report;
    }

    std::vector<Scalar> point_rhs(static_cast<std::size_t>(eliminated_size), Scalar(0));
    for (Index block = 0; block < eliminated_count; ++block)
    {
        for (Index index = 0; index < eliminated_size; ++index)
        {
            point_rhs[static_cast<std::size_t>(index)] =
                -equations._eliminatedGradient[static_cast<std::size_t>(block * eliminated_size + index)];
        }
        for (const std::size_t cross_index :
             equations._eliminatedAdjacency[static_cast<std::size_t>(block)])
        {
            const auto& cross = equations._crossBlocks[cross_index];
            std::vector<Scalar> cross_product(static_cast<std::size_t>(eliminated_size), Scalar(0));
            addTransposeMatrixVector(cross.values.data(),
                                     primary_size,
                                     eliminated_size,
                                     primary_step->data() + cross.primaryBlock * primary_size,
                                     cross_product.data());
            for (Index index = 0; index < eliminated_size; ++index)
            {
                point_rhs[static_cast<std::size_t>(index)] -=
                    cross_product[static_cast<std::size_t>(index)];
            }
        }
        multiplyMatrixVector(eliminated_inverse[static_cast<std::size_t>(block)].data(),
                             eliminated_size,
                             eliminated_size,
                             point_rhs.data(),
                             eliminated_step->data() + block * eliminated_size);
    }
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
