#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <omp.h>

#include "plamatrix/optimization/block_schur.h"
#include "plamatrix/sparse/csr_matrix.h"

#include "block_schur_linear_algebra.h"

namespace plamatrix::block_schur_detail
{

inline Index chooseDenseCholeskyBlockSize(Index dimension)
{
    if (dimension >= 1024)
    {
        return 64;
    }
    if (dimension >= 384)
    {
        return 48;
    }
    return 32;
}

inline int chooseDenseCholeskyThreadCount(Index dimension)
{
    const int available = std::max(1, omp_get_max_threads());
    const int useful = std::max(1, static_cast<int>((dimension + 95) / 96));
    return std::min(available, useful);
}

template <typename Scalar>
void updateDenseTrailingLower(Index dimension,
                              Index block_begin,
                              Index block_end,
                              std::vector<Scalar>* matrix)
{
    constexpr Index micro_size = 4;
    const Index trailing = dimension - block_end;
    const Index tile_count = (trailing + micro_size - 1) / micro_size;
    const Index work_count = tile_count * tile_count;
    #pragma omp for schedule(static)
    for (Index work = 0; work < work_count; ++work)
    {
        const Index row_tile = work / tile_count;
        const Index column_tile = work % tile_count;
        if (column_tile > row_tile)
        {
            continue;
        }
        const Index row_begin = block_end + row_tile * micro_size;
        const Index column_begin = block_end + column_tile * micro_size;
        const Index row_end = std::min(row_begin + micro_size, dimension);
        const Index column_end = std::min(column_begin + micro_size, dimension);
        for (Index row = row_begin; row < row_end; ++row)
        {
            Scalar* current_row = matrix->data() +
                static_cast<std::size_t>(row * dimension);
            const Index active_column_end = std::min(column_end, row + 1);
            for (Index column = column_begin; column < active_column_end; ++column)
            {
                const Scalar* trailing_row = matrix->data() +
                    static_cast<std::size_t>(column * dimension);
                Scalar update = Scalar(0);
                #pragma omp simd reduction(+:update)
                for (Index inner = block_begin; inner < block_end; ++inner)
                {
                    update += current_row[inner] * trailing_row[inner];
                }
                current_row[column] -= update;
            }
        }
    }
}

template <typename Scalar>
bool factorPositiveDefiniteSerial(Index dimension, std::vector<Scalar>* matrix)
{
    for (Index row = 0; row < dimension; ++row)
    {
        for (Index column = 0; column <= row; ++column)
        {
            Scalar dot = Scalar(0);
            #pragma omp simd reduction(+:dot)
            for (Index inner = 0; inner < column; ++inner)
            {
                dot += (*matrix)[static_cast<std::size_t>(row * dimension + inner)] *
                       (*matrix)[static_cast<std::size_t>(column * dimension + inner)];
            }
            const Scalar value =
                (*matrix)[static_cast<std::size_t>(row * dimension + column)] - dot;
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
    return true;
}

template <typename Scalar>
bool factorPositiveDefiniteParallel(Index dimension, std::vector<Scalar>* matrix)
{
    const Index block_size = chooseDenseCholeskyBlockSize(dimension);
    const int thread_count = chooseDenseCholeskyThreadCount(dimension);
    bool success = true;
    #pragma omp parallel num_threads(thread_count) shared(success)
    {
        for (Index block_begin = 0; block_begin < dimension; block_begin += block_size)
        {
            const Index block_end = std::min(block_begin + block_size, dimension);
            #pragma omp single
            {
                if (success)
                {
                    for (Index column = block_begin; column < block_end; ++column)
                    {
                        Scalar* diagonal_row =
                            matrix->data() + static_cast<std::size_t>(column * dimension);
                        const Scalar pivot = diagonal_row[column];
                        if (!(pivot > Scalar(0)) || !std::isfinite(pivot))
                        {
                            success = false;
                            break;
                        }
                        diagonal_row[column] = std::sqrt(pivot);
                        for (Index row = column + 1; row < block_end; ++row)
                        {
                            (*matrix)[static_cast<std::size_t>(row * dimension + column)] /=
                                diagonal_row[column];
                        }
                        for (Index row = column + 1; row < block_end; ++row)
                        {
                            Scalar* current_row =
                                matrix->data() + static_cast<std::size_t>(row * dimension);
                            const Scalar factor = current_row[column];
                            #pragma omp simd
                            for (Index trailing_column = column + 1;
                                 trailing_column <= row; ++trailing_column)
                            {
                                current_row[trailing_column] -=
                                    factor * (*matrix)[static_cast<std::size_t>(
                                        trailing_column * dimension + column)];
                            }
                        }
                    }
                }
            }

            #pragma omp for schedule(static)
            for (Index row = block_end; row < dimension; ++row)
            {
                if (success)
                {
                    Scalar* current_row =
                        matrix->data() + static_cast<std::size_t>(row * dimension);
                    for (Index column = block_begin; column < block_end; ++column)
                    {
                        Scalar dot = Scalar(0);
                        const Scalar* diagonal_row =
                            matrix->data() + static_cast<std::size_t>(column * dimension);
                        #pragma omp simd reduction(+:dot)
                        for (Index inner = block_begin; inner < column; ++inner)
                        {
                            dot += current_row[inner] * diagonal_row[inner];
                        }
                        current_row[column] =
                            (current_row[column] - dot) / diagonal_row[column];
                    }
                }
            }

            updateDenseTrailingLower(
                dimension, block_begin, block_end, matrix);
        }
    }
    return success;
}

template <typename Scalar>
bool factorPositiveDefiniteNative(Index dimension, std::vector<Scalar>* matrix)
{
    constexpr Index parallel_dimension_threshold = 128;
    const bool use_parallel = dimension >= parallel_dimension_threshold && !omp_in_parallel();
    const bool factored = use_parallel
        ? factorPositiveDefiniteParallel(dimension, matrix)
        : factorPositiveDefiniteSerial(dimension, matrix);
    return factored;
}

template <typename Scalar>
void solvePositiveDefiniteFactoredNative(Index dimension,
                                         const std::vector<Scalar>& matrix,
                                         std::vector<Scalar>* solution)
{
    for (Index row = 0; row < dimension; ++row)
    {
        Scalar value = (*solution)[static_cast<std::size_t>(row)];
        for (Index column = 0; column < row; ++column)
        {
            value -= matrix[static_cast<std::size_t>(row * dimension + column)] *
                     (*solution)[static_cast<std::size_t>(column)];
        }
        (*solution)[static_cast<std::size_t>(row)] =
            value / matrix[static_cast<std::size_t>(row * dimension + row)];
    }
    for (Index row = dimension; row-- > 0;)
    {
        Scalar value = (*solution)[static_cast<std::size_t>(row)];
        for (Index column = row + 1; column < dimension; ++column)
        {
            value -= matrix[static_cast<std::size_t>(column * dimension + row)] *
                     (*solution)[static_cast<std::size_t>(column)];
        }
        (*solution)[static_cast<std::size_t>(row)] =
            value / matrix[static_cast<std::size_t>(row * dimension + row)];
    }
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
    const auto conversion_start = std::chrono::steady_clock::now();
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
    report.csrConversionSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - conversion_start).count();
    *solution = rhs;
    report.initialResidualNorm = vectorNorm(rhs);
    const auto factor_start = std::chrono::steady_clock::now();
    const bool factored = factorPositiveDefiniteNative(dimension, &dense);
    report.choleskyFactorizationSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - factor_start).count();
    if (!factored)
    {
        report.message = "reduced Schur complement is not positive definite";
    }
    else
    {
        const auto triangular_start = std::chrono::steady_clock::now();
        solvePositiveDefiniteFactoredNative(dimension, dense, solution);
        const bool solved = true;
        report.triangularSolveSeconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - triangular_start).count();
        if (!solved)
        {
            report.message = "dense Cholesky triangular solve failed";
            report.linearSolveSeconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - solve_start).count();
            return report;
        }
        const auto residual_start = std::chrono::steady_clock::now();
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
        report.residualCheckSeconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - residual_start).count();
        if (!report.converged)
        {
            report.message = "dense Cholesky residual exceeds tolerance";
        }
    }
    report.linearSolveSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - solve_start).count();
    return report;
}

template <typename Scalar>
SchurComplementSolverReport<Scalar> solveReducedSchurDenseLower(
    std::vector<Scalar>* matrix,
    const std::vector<Scalar>& reference,
    const std::vector<Scalar>& rhs,
    const SchurComplementSolverOptions<Scalar>& options,
    std::vector<Scalar>* solution)
{
    SchurComplementSolverReport<Scalar> report;
    report.linearBackend = SchurComplementLinearBackend::DenseCpu;
    const auto solve_start = std::chrono::steady_clock::now();
    const Index dimension = static_cast<Index>(rhs.size());
    if (matrix->size() != static_cast<std::size_t>(dimension * dimension) ||
        reference.size() != matrix->size())
    {
        throw std::invalid_argument("dense Schur storage has an invalid size");
    }
    *solution = rhs;
    report.initialResidualNorm = vectorNorm(rhs);

    const auto factor_start = std::chrono::steady_clock::now();
    const bool factored = factorPositiveDefiniteNative(dimension, matrix);
    report.choleskyFactorizationSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - factor_start).count();
    if (!factored)
    {
        report.message = "reduced Schur complement is not positive definite";
    }
    else
    {
        const auto triangular_start = std::chrono::steady_clock::now();
        solvePositiveDefiniteFactoredNative(dimension, *matrix, solution);
        const bool solved = true;
        report.triangularSolveSeconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - triangular_start).count();
        if (!solved)
        {
            report.message = "dense Cholesky triangular solve failed";
            report.linearSolveSeconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - solve_start).count();
            return report;
        }

        const auto residual_start = std::chrono::steady_clock::now();
        std::vector<Scalar> residual = rhs;
        for (Index row = 0; row < dimension; ++row)
        {
            Scalar product = Scalar(0);
            #pragma omp simd reduction(+:product)
            for (Index column = 0; column <= row; ++column)
            {
                product += reference[static_cast<std::size_t>(row * dimension + column)] *
                           (*solution)[static_cast<std::size_t>(column)];
            }
            for (Index column = row + 1; column < dimension; ++column)
            {
                product += reference[static_cast<std::size_t>(column * dimension + row)] *
                           (*solution)[static_cast<std::size_t>(column)];
            }
            residual[static_cast<std::size_t>(row)] -= product;
        }
        report.finalResidualNorm = vectorNorm(residual);
        const Scalar tolerance = options.absoluteTolerance +
                                 options.relativeTolerance * report.initialResidualNorm;
        report.converged = std::isfinite(report.finalResidualNorm) &&
                           report.finalResidualNorm <= std::max(
                               tolerance,
                               Scalar(100) * std::numeric_limits<Scalar>::epsilon() *
                                   std::max(Scalar(1), report.initialResidualNorm));
        report.iterations = 1;
        report.residualCheckSeconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - residual_start).count();
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
