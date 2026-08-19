#include "block_schur_accelerated.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string>

#include "plamatrix/dense/dense_matrix.h"
#include "plamatrix/opencl/iterative_solver.h"
#include "plamatrix/opencl/runtime.h"
#include "plamatrix/sparse/iterative_solver.h"

#ifdef PLAMATRIX_WITH_CUDA
#include <cuda_runtime_api.h>

#include "plamatrix/core/error_check.h"
#endif

namespace plamatrix::block_schur_detail
{
namespace
{

template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> makeDenseVector(const std::vector<Scalar>& values)
{
    DenseMatrix<Scalar, Device::CPU> result(static_cast<Index>(values.size()), 1);
    for (std::size_t index = 0; index < values.size(); ++index)
    {
        result.data()[index] = values[index];
    }
    return result;
}

IterativeSolverOptions makeIterativeOptions(
    int max_iterations,
    double relative_tolerance,
    double absolute_tolerance)
{
    IterativeSolverOptions result;
    result.maxIterations = max_iterations;
    result.relativeTolerance = relative_tolerance;
    result.absoluteTolerance = absolute_tolerance;
    result.useJacobiPreconditioner = true;
    result.requireConvergence = false;
    return result;
}

template <typename Scalar>
void copyReport(const IterativeSolverReport& source,
                SchurComplementSolverReport<Scalar>* destination)
{
    destination->converged = source.converged;
    destination->iterations = source.iterations;
    destination->initialResidualNorm = static_cast<Scalar>(source.initialResidual);
    destination->finalResidualNorm = static_cast<Scalar>(source.finalResidual);
    destination->message = source.converged ? "converged" : "PCG iteration limit reached";
}

} // namespace

template <typename Scalar>
SchurComplementSolverReport<Scalar> solveAcceleratedReducedSchur(
    const CSRMatrix<Scalar, Device::CPU>& matrix,
    const std::vector<Scalar>& rhs,
    const std::vector<std::vector<Scalar>>& inverse_diagonal_blocks,
    Index block_size,
    const SchurComplementSolverOptions<Scalar>& options,
    std::vector<Scalar>* solution)
{
    if (!solution)
    {
        throw std::invalid_argument("solveAcceleratedReducedSchur: solution is null");
    }
    SchurComplementSolverReport<Scalar> report;
    report.linearBackend = options.linearBackend;
    if (block_size <= 0
        || static_cast<Index>(inverse_diagonal_blocks.size()) * block_size != matrix.rows())
    {
        throw std::invalid_argument(
            "solveAcceleratedReducedSchur: invalid inverse diagonal block count");
    }
    DenseMatrix<Scalar, Device::CPU> inverse_blocks_cpu(
        matrix.rows() * block_size, 1);
    std::size_t inverse_offset = 0;
    for (const auto& block : inverse_diagonal_blocks)
    {
        if (block.size() != static_cast<std::size_t>(block_size * block_size))
        {
            throw std::invalid_argument(
                "solveAcceleratedReducedSchur: invalid inverse diagonal block size");
        }
        std::copy(block.begin(), block.end(), inverse_blocks_cpu.data() + inverse_offset);
        inverse_offset += block.size();
    }
    const auto iterative_options = makeIterativeOptions(
        options.maxIterations,
        static_cast<double>(options.relativeTolerance),
        static_cast<double>(options.absoluteTolerance));
    auto rhs_cpu = makeDenseVector(rhs);
    DenseMatrix<Scalar, Device::CPU> solution_cpu(matrix.rows(), 1);
    solution_cpu.fill(Scalar(0));
    const auto solve_start = std::chrono::steady_clock::now();

    if (options.linearBackend == SchurComplementLinearBackend::Cuda)
    {
#ifdef PLAMATRIX_WITH_CUDA
        if (options.deviceIndex >= 0)
        {
            PLAMATRIX_CHECK_CUDA(cudaSetDevice(options.deviceIndex));
        }
        int active_device = 0;
        PLAMATRIX_CHECK_CUDA(cudaGetDevice(&active_device));
        cudaDeviceProp properties{};
        PLAMATRIX_CHECK_CUDA(cudaGetDeviceProperties(&properties, active_device));
        report.deviceName = properties.name;

        auto matrix_gpu = matrix.toGpu();
        auto rhs_gpu = rhs_cpu.toGpu();
        auto solution_gpu = solution_cpu.toGpu();
        auto inverse_blocks_gpu = inverse_blocks_cpu.toGpu();
        IterativeSolverWorkspace<Scalar> workspace;
        const auto iterative_report = blockPcg(
            matrix_gpu, rhs_gpu, solution_gpu, inverse_blocks_gpu,
            block_size, workspace, iterative_options);
        solution_cpu = solution_gpu.toCpu();
        copyReport(iterative_report, &report);
#else
        throw std::runtime_error(
            "CUDA Schur PCG requires PLAMATRIX_WITH_CUDA=ON");
#endif
    }
    else if (options.linearBackend == SchurComplementLinearBackend::OpenCl)
    {
        opencl::requireUsableOpenClDevice();
        const int selected_index = opencl::selectedOpenClDeviceIndex();
        if (options.deviceIndex >= 0 && options.deviceIndex != selected_index)
        {
            throw std::runtime_error(
                "OpenCL Schur PCG selected device does not match requested device index");
        }
        report.deviceName = opencl::selectedOpenClDeviceName();
        const auto iterative_report = opencl::blockPcg(
            matrix, rhs_cpu, solution_cpu, inverse_blocks_cpu,
            block_size, iterative_options);
        copyReport(iterative_report, &report);
    }
    else
    {
        throw std::invalid_argument(
            "solveAcceleratedReducedSchur requires CUDA or OpenCL backend");
    }

    report.linearSolveSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - solve_start).count();
    solution->assign(solution_cpu.data(), solution_cpu.data() + solution_cpu.rows());
    return report;
}

template SchurComplementSolverReport<float> solveAcceleratedReducedSchur(
    const CSRMatrix<float, Device::CPU>&,
    const std::vector<float>&,
    const std::vector<std::vector<float>>&,
    Index,
    const SchurComplementSolverOptions<float>&,
    std::vector<float>*);
template SchurComplementSolverReport<double> solveAcceleratedReducedSchur(
    const CSRMatrix<double, Device::CPU>&,
    const std::vector<double>&,
    const std::vector<std::vector<double>>&,
    Index,
    const SchurComplementSolverOptions<double>&,
    std::vector<double>*);

} // namespace plamatrix::block_schur_detail
