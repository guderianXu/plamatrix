#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <vector>

#include <omp.h>

#include "plamatrix/optimization/block_schur.h"
#include "plamatrix/sparse/csr_matrix.h"

#include "block_schur_linear_algebra.h"

#ifdef PLAMATRIX_WITH_LAPACK
#include "../ops/fortran_linalg.h"
#endif

namespace plamatrix::block_schur_detail
{

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
    constexpr Index block_size = 32;
    bool success = true;
    #pragma omp parallel shared(success)
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

            #pragma omp for schedule(static)
            for (Index row = block_end; row < dimension; ++row)
            {
                if (success)
                {
                    Scalar* current_row =
                        matrix->data() + static_cast<std::size_t>(row * dimension);
                    for (Index column = block_end; column <= row; ++column)
                    {
                        const Scalar* trailing_row =
                            matrix->data() + static_cast<std::size_t>(column * dimension);
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
    }
    return success;
}

template <typename Scalar>
bool solvePositiveDefiniteNative(Index dimension,
                                 std::vector<Scalar>* matrix,
                                 std::vector<Scalar>* solution)
{
    constexpr Index parallel_dimension_threshold = 128;
    const bool use_parallel = dimension >= parallel_dimension_threshold &&
                              omp_get_max_threads() > 1 && !omp_in_parallel();
    const bool factored = use_parallel
        ? factorPositiveDefiniteParallel(dimension, matrix)
        : factorPositiveDefiniteSerial(dimension, matrix);
    if (!factored)
    {
        return false;
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
    const bool solved = solvePositiveDefiniteNative(dimension, &dense, solution);
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
