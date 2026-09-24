#include "block_schur_accelerated.h"

#include <algorithm>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "plamatrix/internal/dense/dense_storage.h"
#include "plamatrix/internal/opencl/iterative_solver.h"
#include "plamatrix/internal/opencl/runtime.h"
#include "plamatrix/internal/sparse/iterative_solver.h"
#include "plamatrix/internal/vulkan/runtime.h"
#include "plamatrix/internal/vulkan/iterative_solver.h"

#include "block_schur_sparse_assembly.h"
#include "block_schur_device_assembly.h"

#ifdef PLAMATRIX_WITH_CUDA
#include <cuda_runtime_api.h>

#include "plamatrix/internal/core/error_check.h"
#endif

namespace plamatrix::internal::block_schur_detail
{
    namespace
    {

#ifdef PLAMATRIX_WITH_CUDA
        template <typename Scalar> struct CudaSchurState
        {
            int deviceIndex = -1;
            std::unique_ptr<CsrStorage<Scalar, Device::GPU>> matrix;
            std::unique_ptr<DenseStorage<Scalar, Device::GPU>> rhs;
            std::unique_ptr<DenseStorage<Scalar, Device::GPU>> solution;
            std::unique_ptr<DenseStorage<Scalar, Device::GPU>> inverseBlocks;
            IterativeSolverWorkspace<Scalar> solverWorkspace;
        };
#endif

        template <typename Scalar> DenseStorage<Scalar, Device::CPU> makeDenseVector(const std::vector<Scalar>& values)
        {
            DenseStorage<Scalar, Device::CPU> result(static_cast<Index>(values.size()), 1);
            for (std::size_t index = 0; index < values.size(); ++index)
            {
                result.data()[index] = values[index];
            }
            return result;
        }

        IterativeSolverOptions
        makeIterativeOptions(int max_iterations, double relative_tolerance, double absolute_tolerance)
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
        void copyReport(const IterativeSolverReport& source, SchurComplementSolverReport<Scalar>* destination)
        {
            destination->converged = source.converged;
            destination->iterations = source.iterations;
            destination->initialResidualNorm = static_cast<Scalar>(source.initialResidual);
            destination->finalResidualNorm = static_cast<Scalar>(source.finalResidual);
            destination->commandSubmissions = source.commandSubmissions;
            destination->commandBufferRecordings = source.commandBufferRecordings;
            destination->gpuMilliseconds = source.gpuMilliseconds;
            destination->blockSpmvUsed = source.blockSpmv;
            destination->deviceBlockJacobiUsed = source.deviceBlockJacobi;
            destination->message = source.converged ? "converged" : "PCG iteration limit reached";
        }

    } // namespace

    template <typename Scalar>
    SchurComplementSolverReport<Scalar>
    solveAcceleratedReducedSchur(const CsrStorage<Scalar, Device::CPU>& matrix,
                                 const std::vector<Scalar>& rhs,
                                 const std::vector<std::vector<Scalar>>& inverse_diagonal_blocks,
                                 Index block_size,
                                 const SchurComplementSolverOptions<Scalar>& options,
                                 SchurComplementSolverWorkspace<Scalar>& workspace,
                                 std::vector<Scalar>* solution)
    {
        if (!solution)
        {
            throw std::invalid_argument("solveAcceleratedReducedSchur: solution is null");
        }
        SchurComplementSolverReport<Scalar> report;
        report.linearBackend = options.linearBackend;
        const bool build_device_block_jacobi = options.linearBackend == SchurComplementLinearBackend::Vulkan &&
                                               inverse_diagonal_blocks.empty() && block_size == 9;
        if (block_size <= 0 || (!build_device_block_jacobi &&
                                static_cast<Index>(inverse_diagonal_blocks.size()) * block_size != matrix.rows()))
        {
            throw std::invalid_argument("solveAcceleratedReducedSchur: invalid inverse diagonal block count");
        }
        std::unique_ptr<DenseStorage<Scalar, Device::CPU>> inverse_blocks_cpu;
        if (!build_device_block_jacobi)
            inverse_blocks_cpu = std::make_unique<DenseStorage<Scalar, Device::CPU>>(matrix.rows() * block_size, 1);
        std::size_t inverse_offset = 0;
        for (const auto& block : inverse_diagonal_blocks)
        {
            if (block.size() != static_cast<std::size_t>(block_size * block_size))
            {
                throw std::invalid_argument("solveAcceleratedReducedSchur: invalid inverse diagonal block size");
            }
            std::copy(block.begin(), block.end(), inverse_blocks_cpu->data() + inverse_offset);
            inverse_offset += block.size();
        }
        const auto iterative_options = makeIterativeOptions(options.maxIterations,
                                                            static_cast<double>(options.relativeTolerance),
                                                            static_cast<double>(options.absoluteTolerance));
        auto rhs_cpu = makeDenseVector(rhs);
        DenseStorage<Scalar, Device::CPU> solution_cpu(matrix.rows(), 1);
        if (options.useInitialGuess && solution->size() == static_cast<std::size_t>(matrix.rows()))
        {
            std::copy(solution->begin(), solution->end(), solution_cpu.data());
        }
        else
        {
            solution_cpu.fill(Scalar(0));
        }

        bool mixed_precision_used = false;
        if constexpr (std::is_same_v<Scalar, double>)
        {
            if (options.useMixedPrecision && (options.linearBackend == SchurComplementLinearBackend::Cuda ||
                                              options.linearBackend == SchurComplementLinearBackend::OpenCl ||
                                              options.linearBackend == SchurComplementLinearBackend::Vulkan))
            {
                CsrStorage<float, Device::CPU> float_matrix(matrix.rows(), matrix.cols(), matrix.nnz());
                std::copy(matrix.rowOffsets(), matrix.rowOffsets() + matrix.rows() + 1, float_matrix.rowOffsets());
                std::copy(matrix.colIndices(), matrix.colIndices() + matrix.nnz(), float_matrix.colIndices());
                for (Index index = 0; index < matrix.nnz(); ++index)
                {
                    float_matrix.values()[index] = static_cast<float>(matrix.values()[index]);
                }
                float_matrix.validateStructure();
                std::vector<float> float_rhs(rhs.size());
                std::transform(
                    rhs.begin(), rhs.end(), float_rhs.begin(), [](double value) { return static_cast<float>(value); });
                std::vector<std::vector<float>> float_inverse(inverse_diagonal_blocks.size());
                for (std::size_t block = 0; block < inverse_diagonal_blocks.size(); ++block)
                {
                    float_inverse[block].resize(inverse_diagonal_blocks[block].size());
                    std::transform(inverse_diagonal_blocks[block].begin(),
                                   inverse_diagonal_blocks[block].end(),
                                   float_inverse[block].begin(),
                                   [](double value) { return static_cast<float>(value); });
                }
                SchurComplementSolverOptions<float> float_options;
                float_options.linearBackend = options.linearBackend;
                float_options.deviceIndex = options.deviceIndex;
                float_options.maxIterations = std::min(options.maxIterations, 200);
                float_options.relativeTolerance = std::max(1.0e-3f, static_cast<float>(options.relativeTolerance));
                float_options.absoluteTolerance = std::max(1.0e-7f, static_cast<float>(options.absoluteTolerance));
                float_options.schurValuesOnDevice = false;
                float_options.useMixedPrecision = options.linearBackend == SchurComplementLinearBackend::Vulkan;
                std::vector<float> float_solution(solution_cpu.rows(), 0.0f);
                SchurComplementSolverWorkspace<float> float_workspace;
                auto& persistent_float_state = SchurComplementSolverWorkspaceAccess::mixedPrecisionState(workspace);
                SchurComplementSolverWorkspaceAccess::acceleratedState(float_workspace) = persistent_float_state;
                try
                {
                    const auto float_report = solveAcceleratedReducedSchur(float_matrix,
                                                                           float_rhs,
                                                                           float_inverse,
                                                                           block_size,
                                                                           float_options,
                                                                           float_workspace,
                                                                           &float_solution);
                    persistent_float_state = SchurComplementSolverWorkspaceAccess::acceleratedState(float_workspace);
                    if (options.linearBackend == SchurComplementLinearBackend::Vulkan)
                    {
                        solution->assign(float_solution.begin(), float_solution.end());
                        report.converged = float_report.converged;
                        report.iterations = float_report.iterations;
                        report.initialResidualNorm = static_cast<double>(float_report.initialResidualNorm);
                        report.finalResidualNorm = static_cast<double>(float_report.finalResidualNorm);
                        report.linearSolveSeconds = float_report.linearSolveSeconds;
                        report.mixedPrecisionUsed = true;
                        report.cooperativeMatrixUsed = float_report.cooperativeMatrixUsed;
                        report.blockSpmvUsed = float_report.blockSpmvUsed;
                        report.deviceBlockJacobiUsed = float_report.deviceBlockJacobiUsed;
                        report.commandSubmissions = float_report.commandSubmissions;
                        report.commandBufferRecordings = float_report.commandBufferRecordings;
                        report.gpuMilliseconds = float_report.gpuMilliseconds;
                        report.deviceName = float_report.deviceName;
                        report.message = float_report.message;
                        return report;
                    }
                    if (float_report.converged)
                    {
                        std::transform(float_solution.begin(),
                                       float_solution.end(),
                                       solution_cpu.data(),
                                       [](float value) { return static_cast<double>(value); });
                        mixed_precision_used = true;
                    }
                }
                catch (const std::runtime_error&)
                {
                    persistent_float_state.reset();
                }
            }
        }
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

            auto& opaque_state = SchurComplementSolverWorkspaceAccess::acceleratedState(workspace);
            auto state = std::static_pointer_cast<CudaSchurState<Scalar>>(opaque_state);
            const bool reusable = state && state->deviceIndex == active_device && state->matrix &&
                                  state->matrix->rows() == matrix.rows() && state->matrix->nnz() == matrix.nnz();
            if (!reusable)
            {
                state = std::make_shared<CudaSchurState<Scalar>>();
                state->deviceIndex = active_device;
                state->matrix =
                    std::make_unique<CsrStorage<Scalar, Device::GPU>>(matrix.rows(), matrix.cols(), matrix.nnz());
                state->rhs = std::make_unique<DenseStorage<Scalar, Device::GPU>>(matrix.rows(), 1);
                state->solution = std::make_unique<DenseStorage<Scalar, Device::GPU>>(matrix.rows(), 1);
                state->inverseBlocks =
                    std::make_unique<DenseStorage<Scalar, Device::GPU>>(inverse_blocks_cpu->rows(), 1);
                opaque_state = state;
            }
            if (options.schurValuesOnDevice)
            {
                copyLastCudaSchurValuesToDevice(
                    detail::CSRMatrixAccess::values(*state->matrix), static_cast<std::size_t>(matrix.nnz()), workspace);
            }
            else
            {
                PLAMATRIX_CHECK_CUDA(cudaMemcpy(detail::CSRMatrixAccess::values(*state->matrix),
                                                matrix.values(),
                                                static_cast<std::size_t>(matrix.nnz()) * sizeof(Scalar),
                                                cudaMemcpyHostToDevice));
            }
            if (!reusable)
            {
                PLAMATRIX_CHECK_CUDA(cudaMemcpy(detail::CSRMatrixAccess::colIndices(*state->matrix),
                                                matrix.colIndices(),
                                                static_cast<std::size_t>(matrix.nnz()) * sizeof(Index),
                                                cudaMemcpyHostToDevice));
                PLAMATRIX_CHECK_CUDA(cudaMemcpy(detail::CSRMatrixAccess::rowOffsets(*state->matrix),
                                                matrix.rowOffsets(),
                                                static_cast<std::size_t>(matrix.rows() + 1) * sizeof(Index),
                                                cudaMemcpyHostToDevice));
            }
            detail::CSRMatrixAccess::completeAsyncWrite(*state->matrix, true);
            PLAMATRIX_CHECK_CUDA(cudaMemcpy(state->rhs->data(),
                                            rhs_cpu.data(),
                                            static_cast<std::size_t>(matrix.rows()) * sizeof(Scalar),
                                            cudaMemcpyHostToDevice));
            PLAMATRIX_CHECK_CUDA(cudaMemcpy(state->solution->data(),
                                            solution_cpu.data(),
                                            static_cast<std::size_t>(matrix.rows()) * sizeof(Scalar),
                                            cudaMemcpyHostToDevice));
            PLAMATRIX_CHECK_CUDA(cudaMemcpy(state->inverseBlocks->data(),
                                            inverse_blocks_cpu->data(),
                                            static_cast<std::size_t>(inverse_blocks_cpu->rows()) * sizeof(Scalar),
                                            cudaMemcpyHostToDevice));
            IterativeSolverOptions cuda_iterative_options = iterative_options;
            cuda_iterative_options.convergenceCheckInterval = 8;
            const auto iterative_report = blockPcg(*state->matrix,
                                                   *state->rhs,
                                                   *state->solution,
                                                   *state->inverseBlocks,
                                                   block_size,
                                                   state->solverWorkspace,
                                                   cuda_iterative_options);
            PLAMATRIX_CHECK_CUDA(cudaMemcpy(solution_cpu.data(),
                                            state->solution->data(),
                                            static_cast<std::size_t>(matrix.rows()) * sizeof(Scalar),
                                            cudaMemcpyDeviceToHost));
            copyReport(iterative_report, &report);
#else
            throw std::runtime_error("CUDA Schur PCG requires PLAMATRIX_WITH_CUDA=ON");
#endif
        }
        else if (options.linearBackend == SchurComplementLinearBackend::OpenCl)
        {
            opencl::requireUsableOpenClDevice();
            const int selected_index = opencl::selectedOpenClDeviceIndex();
            if (options.deviceIndex >= 0 && options.deviceIndex != selected_index)
            {
                throw std::runtime_error("OpenCL Schur PCG selected device does not match requested device index");
            }
            report.deviceName = opencl::selectedOpenClDeviceName();
            const auto iterative_report =
                opencl::blockPcg(matrix, rhs_cpu, solution_cpu, *inverse_blocks_cpu, block_size, iterative_options);
            copyReport(iterative_report, &report);
        }
        else if (options.linearBackend == SchurComplementLinearBackend::Vulkan)
        {
            if constexpr (std::is_same_v<Scalar, float>)
            {
                report.deviceName = vulkan::selectedVulkanDeviceName();
                IterativeSolverOptions vulkan_iterative_options = iterative_options;
                vulkan_iterative_options.convergenceCheckInterval = 8;
                const auto iterative_report =
                    options.schurValuesOnDevice
                        ? solveLastVulkanSchurValues(matrix,
                                                     rhs_cpu,
                                                     solution_cpu,
                                                     inverse_blocks_cpu.get(),
                                                     block_size,
                                                     build_device_block_jacobi,
                                                     workspace,
                                                     vulkan_iterative_options)
                        : vulkan::blockPcg(
                              matrix, rhs_cpu, solution_cpu, *inverse_blocks_cpu, block_size, vulkan_iterative_options);
                copyReport(iterative_report, &report);
                report.cooperativeMatrixUsed = options.schurValuesOnDevice;
            }
            else
            {
                throw std::invalid_argument("Vulkan Schur PCG currently requires float equations");
            }
        }
        else
        {
            throw std::invalid_argument("solveAcceleratedReducedSchur requires CUDA, OpenCL, or Vulkan backend");
        }

        report.linearSolveSeconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - solve_start).count();
        solution->assign(solution_cpu.data(), solution_cpu.data() + solution_cpu.rows());
        report.mixedPrecisionUsed =
            mixed_precision_used || options.linearBackend == SchurComplementLinearBackend::Vulkan;
        return report;
    }

    template SchurComplementSolverReport<float> solveAcceleratedReducedSchur(const CsrStorage<float, Device::CPU>&,
                                                                             const std::vector<float>&,
                                                                             const std::vector<std::vector<float>>&,
                                                                             Index,
                                                                             const SchurComplementSolverOptions<float>&,
                                                                             SchurComplementSolverWorkspace<float>&,
                                                                             std::vector<float>*);
    template SchurComplementSolverReport<double>
    solveAcceleratedReducedSchur(const CsrStorage<double, Device::CPU>&,
                                 const std::vector<double>&,
                                 const std::vector<std::vector<double>>&,
                                 Index,
                                 const SchurComplementSolverOptions<double>&,
                                 SchurComplementSolverWorkspace<double>&,
                                 std::vector<double>*);

} // namespace plamatrix::internal::block_schur_detail
