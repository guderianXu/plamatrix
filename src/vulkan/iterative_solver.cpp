#include "plamatrix/vulkan/iterative_solver.h"

#ifdef PLAMATRIX_WITH_VULKAN

#include "plamatrix/vulkan/execution.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <stdexcept>
#include <vector>

namespace plamatrix::vulkan
{
    namespace
    {

        constexpr std::uint32_t kLocalSize = 128;
        constexpr std::size_t kFusedSolutionReadbackMaxBytes = 64 * 1024;

        void validateOptions(const IterativeSolverOptions& options)
        {
            if (options.maxIterations < 0 || !std::isfinite(options.relativeTolerance) ||
                options.relativeTolerance < 0.0 || options.relativeTolerance > 1.0 ||
                !std::isfinite(options.absoluteTolerance) || options.absoluteTolerance < 0.0 ||
                options.convergenceCheckInterval <= 0)
            {
                throw std::invalid_argument("Vulkan PCG received invalid solver options");
            }
        }

        std::vector<float> inverseDiagonal(const CSRMatrix<float, Device::CPU>& matrix, bool enabled)
        {
            std::vector<float> result(static_cast<std::size_t>(matrix.rows()), 1.0f);
            if (!enabled)
                return result;
            for (Index row = 0; row < matrix.rows(); ++row)
            {
                long double diagonal = 0.0L;
                for (Index entry = matrix.rowOffsets()[row]; entry < matrix.rowOffsets()[row + 1]; ++entry)
                {
                    if (matrix.colIndices()[entry] == row)
                        diagonal += matrix.values()[entry];
                }
                if (!std::isfinite(diagonal) || diagonal <= 0.0L)
                    throw std::runtime_error("Vulkan PCG Jacobi preconditioner requires a positive diagonal");
                const float inverse = 1.0f / static_cast<float>(diagonal);
                if (!std::isfinite(inverse) || inverse <= 0.0f)
                    throw std::runtime_error("Vulkan PCG Jacobi inverse diagonal is not finite and positive");
                result[static_cast<std::size_t>(row)] = inverse;
            }
            return result;
        }

        std::uint32_t checkedUint(Index value, const char* name)
        {
            if (value < 0 || static_cast<std::uint64_t>(value) > std::numeric_limits<std::uint32_t>::max())
                throw std::overflow_error(std::string("Vulkan ") + name + " exceeds uint32 range");
            return static_cast<std::uint32_t>(value);
        }

        std::size_t groupsFor(std::size_t count)
        {
            return std::max<std::size_t>(1, (count + kLocalSize - 1) / kLocalSize);
        }

        std::size_t reductionGroupsFor(std::size_t count)
        {
            return std::max<std::size_t>(1, (count + kLocalSize * 2 - 1) / (kLocalSize * 2));
        }

        bool shouldUseSubgroupSpmv(const Runtime& runtime, std::size_t rows, std::size_t nnz)
        {
            const char* mode = std::getenv("PLAMATRIX_VULKAN_SPMV");
            if (mode && std::strcmp(mode, "scalar") == 0)
                return false;
            if (mode && std::strcmp(mode, "subgroup") == 0)
            {
                if (!runtime.supportsSubgroupArithmetic())
                    throw std::runtime_error(
                        "PLAMATRIX_VULKAN_SPMV=subgroup requires compute subgroup arithmetic support");
                return true;
            }
            if (mode && std::strcmp(mode, "auto") != 0)
                throw std::invalid_argument("PLAMATRIX_VULKAN_SPMV must be auto, scalar, or subgroup");
            const std::size_t minimumAverageRowLength = std::max<std::size_t>(16, runtime.subgroupSize() / 2);
            return runtime.supportsSubgroupArithmetic() && rows != 0 && nnz / rows >= minimumAverageRowLength;
        }

        std::size_t spmvGroupsFor(const Runtime& runtime, std::size_t rows, bool subgroup)
        {
            if (!subgroup)
                return groupsFor(rows);
            const std::size_t rowsPerGroup = kLocalSize / runtime.subgroupSize();
            return std::max<std::size_t>(1, (rows + rowsPerGroup - 1) / rowsPerGroup);
        }

        struct CountPush
        {
            std::uint32_t count;
        };

        struct StateInitPush
        {
            float relativeToleranceSquared;
            float absoluteToleranceSquared;
        };

        struct IterationPush
        {
            std::uint32_t iterations;
        };

        struct PcgStateHost
        {
            float rho = 0.0f;
            float alpha = 0.0f;
            float beta = 0.0f;
            float residualSquared = 0.0f;
            float toleranceSquared = 0.0f;
            float initialResidualSquared = 0.0f;
            std::uint32_t flags = 0;
            std::uint32_t iterations = 0;
        };

        static_assert(sizeof(PcgStateHost) == sizeof(float) * 6 + sizeof(std::uint32_t) * 2);

        constexpr std::uint32_t kConvergedFlag = 1u;
        constexpr std::uint32_t kBreakdownFlag = 2u;

        struct PipelineSet
        {
            explicit PipelineSet(Runtime& runtime)
                : initialize(runtime,
                             "initialize",
                             6,
                             sizeof(CountPush),
                             {VK_ACCESS_SHADER_READ_BIT,
                              VK_ACCESS_SHADER_READ_BIT,
                              VK_ACCESS_SHADER_READ_BIT,
                              VK_ACCESS_SHADER_WRITE_BIT,
                              VK_ACCESS_SHADER_WRITE_BIT,
                              VK_ACCESS_SHADER_WRITE_BIT}),
                  spmv(runtime,
                       "spmv",
                       5,
                       sizeof(CountPush),
                       {VK_ACCESS_SHADER_READ_BIT,
                        VK_ACCESS_SHADER_READ_BIT,
                        VK_ACCESS_SHADER_READ_BIT,
                        VK_ACCESS_SHADER_READ_BIT,
                        VK_ACCESS_SHADER_WRITE_BIT}),
                  jacobi(runtime,
                         "jacobi",
                         3,
                         sizeof(CountPush),
                         {VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT}),
                  update(runtime,
                         "update",
                         5,
                         sizeof(CountPush),
                         {VK_ACCESS_SHADER_WRITE_BIT,
                          VK_ACCESS_SHADER_WRITE_BIT,
                          VK_ACCESS_SHADER_READ_BIT,
                          VK_ACCESS_SHADER_READ_BIT,
                          VK_ACCESS_SHADER_READ_BIT}),
                  direction(runtime,
                            "direction",
                            3,
                            sizeof(CountPush),
                            {VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_READ_BIT}),
                  reduction(runtime,
                            "dot_reduce",
                            3,
                            sizeof(CountPush),
                            {VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT}),
                  stateInit(runtime,
                            "pcg_state_init",
                            2,
                            sizeof(StateInitPush),
                            {VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT}),
                  rhoInit(runtime, "pcg_rho_init", 2, 0, {VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT}),
                  alpha(runtime, "pcg_alpha", 2, 0, {VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT}),
                  beta(runtime, "pcg_beta", 2, 0, {VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT}),
                  convergence(runtime,
                              "pcg_convergence",
                              2,
                              sizeof(IterationPush),
                              {VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT})
            {
                if (runtime.supportsSubgroupArithmetic())
                {
                    spmvSubgroup =
                        std::make_unique<ComputePipeline>(runtime,
                                                          "spmv_subgroup",
                                                          5,
                                                          sizeof(CountPush),
                                                          std::vector<VkAccessFlags>{VK_ACCESS_SHADER_READ_BIT,
                                                                                     VK_ACCESS_SHADER_READ_BIT,
                                                                                     VK_ACCESS_SHADER_READ_BIT,
                                                                                     VK_ACCESS_SHADER_READ_BIT,
                                                                                     VK_ACCESS_SHADER_WRITE_BIT});
                }
            }

            const ComputePipeline& spmvPipeline(bool subgroup) const
            {
                return subgroup ? *spmvSubgroup : spmv;
            }

            ComputePipeline initialize;
            ComputePipeline spmv;
            std::unique_ptr<ComputePipeline> spmvSubgroup;
            ComputePipeline jacobi;
            ComputePipeline update;
            ComputePipeline direction;
            ComputePipeline reduction;
            ComputePipeline stateInit;
            ComputePipeline rhoInit;
            ComputePipeline alpha;
            ComputePipeline beta;
            ComputePipeline convergence;
        };

        struct PcgBuffers
        {
            const std::size_t count;
            const std::size_t nnz;
            std::unique_ptr<Buffer> row;
            std::unique_ptr<Buffer> column;
            std::unique_ptr<Buffer> value;
            std::unique_ptr<Buffer> rhs;
            std::unique_ptr<Buffer> solution;
            std::unique_ptr<Buffer> residual;
            std::unique_ptr<Buffer> transformed;
            std::unique_ptr<Buffer> direction;
            std::unique_ptr<Buffer> matrixDirection;
            std::unique_ptr<Buffer> inverse;
            std::unique_ptr<Buffer> ones;
            std::unique_ptr<Buffer> partialA;
            std::unique_ptr<Buffer> partialB;
            std::unique_ptr<Buffer> state;
            std::unique_ptr<Buffer> stateReadback;
            std::unique_ptr<Buffer> rowUpload;
            std::unique_ptr<Buffer> columnUpload;
            std::unique_ptr<Buffer> valueUpload;
            std::unique_ptr<Buffer> rhsUpload;
            std::unique_ptr<Buffer> solutionUpload;
            std::unique_ptr<Buffer> inverseUpload;
            std::unique_ptr<Buffer> onesUpload;
            std::unique_ptr<Buffer> solutionReadback;
            std::unique_ptr<CommandContext> context;

            PcgBuffers(Runtime& runtime, std::size_t countValue, std::size_t nnzValue)
                : count(countValue), nnz(nnzValue)
            {
                const VkDeviceSize uintBytes = static_cast<VkDeviceSize>(sizeof(std::uint32_t));
                const VkDeviceSize floatBytes = static_cast<VkDeviceSize>(sizeof(float));
                const VkBufferUsageFlags storageUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
                const VkBufferUsageFlags deviceInputUsage = storageUsage | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
                const VkBufferUsageFlags deviceOutputUsage =
                    storageUsage | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
                const VkBufferUsageFlags stagingSourceUsage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
                const VkBufferUsageFlags stagingDestinationUsage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
                const VkBufferUsageFlags partialUsage =
                    storageUsage | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
                const std::size_t partialCount = reductionGroupsFor(count);

                row = std::make_unique<Buffer>(
                    runtime, (count + 1) * uintBytes, deviceInputUsage, BufferMemory::DeviceLocal);
                column = std::make_unique<Buffer>(
                    runtime, std::max<std::size_t>(1, nnz) * uintBytes, deviceInputUsage, BufferMemory::DeviceLocal);
                value = std::make_unique<Buffer>(
                    runtime, std::max<std::size_t>(1, nnz) * floatBytes, deviceInputUsage, BufferMemory::DeviceLocal);
                rhs =
                    std::make_unique<Buffer>(runtime, count * floatBytes, deviceInputUsage, BufferMemory::DeviceLocal);
                solution =
                    std::make_unique<Buffer>(runtime, count * floatBytes, deviceOutputUsage, BufferMemory::DeviceLocal);
                residual =
                    std::make_unique<Buffer>(runtime, count * floatBytes, storageUsage, BufferMemory::DeviceLocal);
                transformed =
                    std::make_unique<Buffer>(runtime, count * floatBytes, storageUsage, BufferMemory::DeviceLocal);
                direction =
                    std::make_unique<Buffer>(runtime, count * floatBytes, storageUsage, BufferMemory::DeviceLocal);
                matrixDirection =
                    std::make_unique<Buffer>(runtime, count * floatBytes, storageUsage, BufferMemory::DeviceLocal);
                inverse =
                    std::make_unique<Buffer>(runtime, count * floatBytes, deviceInputUsage, BufferMemory::DeviceLocal);
                ones =
                    std::make_unique<Buffer>(runtime, count * floatBytes, deviceInputUsage, BufferMemory::DeviceLocal);
                partialA = std::make_unique<Buffer>(
                    runtime, partialCount * floatBytes, partialUsage, BufferMemory::DeviceLocal);
                partialB = std::make_unique<Buffer>(
                    runtime, partialCount * floatBytes, partialUsage, BufferMemory::DeviceLocal);
                state = std::make_unique<Buffer>(
                    runtime, sizeof(PcgStateHost), deviceOutputUsage, BufferMemory::DeviceLocal);
                stateReadback = std::make_unique<Buffer>(runtime, sizeof(PcgStateHost), stagingDestinationUsage);

                rowUpload = std::make_unique<Buffer>(runtime, (count + 1) * uintBytes, stagingSourceUsage);
                columnUpload =
                    std::make_unique<Buffer>(runtime, std::max<std::size_t>(1, nnz) * uintBytes, stagingSourceUsage);
                valueUpload =
                    std::make_unique<Buffer>(runtime, std::max<std::size_t>(1, nnz) * floatBytes, stagingSourceUsage);
                rhsUpload = std::make_unique<Buffer>(runtime, count * floatBytes, stagingSourceUsage);
                solutionUpload = std::make_unique<Buffer>(runtime, count * floatBytes, stagingSourceUsage);
                inverseUpload = std::make_unique<Buffer>(runtime, count * floatBytes, stagingSourceUsage);
                onesUpload = std::make_unique<Buffer>(runtime, count * floatBytes, stagingSourceUsage);
                solutionReadback = std::make_unique<Buffer>(runtime, count * floatBytes, stagingDestinationUsage);
                context = std::make_unique<CommandContext>(runtime);
            }
        };

        void dispatch(CommandContext& context,
                      const ComputePipeline& pipeline,
                      const std::vector<Buffer*>& buffers,
                      std::size_t groups,
                      const void* pushData,
                      std::uint32_t pushSize)
        {
            context.dispatch(pipeline, buffers, groups, pushData, pushSize);
        }

        Buffer* dispatchDot(CommandContext& context,
                            const ComputePipeline& pipeline,
                            Buffer& left,
                            Buffer& right,
                            Buffer& partialA,
                            Buffer& partialB,
                            Buffer& ones,
                            std::size_t count)
        {
            std::size_t remaining = reductionGroupsFor(count);
            Buffer* current = &partialA;
            Buffer* next = &partialB;
            CountPush push{static_cast<std::uint32_t>(count)};
            dispatch(context, pipeline, {&left, &right, current}, remaining, &push, sizeof(push));
            while (remaining > 1)
            {
                const std::size_t nextGroups = reductionGroupsFor(remaining);
                push.count = static_cast<std::uint32_t>(remaining);
                dispatch(context, pipeline, {current, &ones, next}, nextGroups, &push, sizeof(push));
                remaining = nextGroups;
                std::swap(current, next);
            }
            return current;
        }

    } // namespace

    IterativeSolverReport pcg(const CSRMatrix<float, Device::CPU>& matrix,
                              const DenseMatrix<float, Device::CPU>& rhs,
                              DenseMatrix<float, Device::CPU>& solution,
                              const IterativeSolverOptions& options)
    {
        validateOptions(options);
        if (matrix.rows() <= 0 || matrix.rows() != matrix.cols() || rhs.rows() != matrix.rows() || rhs.cols() != 1 ||
            solution.rows() != matrix.rows() || solution.cols() != 1 || rhs.data() == solution.data())
        {
            throw std::invalid_argument("Vulkan PCG requires a non-empty square system and distinct vectors");
        }
        matrix.validateStructure();
        for (Index row = 0; row < matrix.rows(); ++row)
        {
            if (!std::isfinite(rhs.data()[row]) || !std::isfinite(solution.data()[row]))
                throw std::invalid_argument("Vulkan PCG vectors must contain finite values");
        }

        Runtime& runtime = Runtime::instance();
        static std::mutex solveMutex;
        const std::lock_guard<std::mutex> solveLock(solveMutex);
        const std::size_t count = static_cast<std::size_t>(matrix.rows());
        const std::size_t nnz = static_cast<std::size_t>(matrix.nnz());
        const std::size_t solutionBytes = count * sizeof(float);
        const bool fuseSolutionReadback = solutionBytes <= kFusedSolutionReadbackMaxBytes;
        std::vector<std::uint32_t> rowOffsets(count + 1);
        std::vector<std::uint32_t> columns(nnz);
        for (std::size_t i = 0; i < rowOffsets.size(); ++i)
            rowOffsets[i] = checkedUint(matrix.rowOffsets()[i], "row offset");
        for (std::size_t i = 0; i < columns.size(); ++i)
            columns[i] = checkedUint(matrix.colIndices()[i], "column index");
        const auto inverse = inverseDiagonal(matrix, options.useJacobiPreconditioner);
        std::vector<float> ones(count, 1.0f);

        static std::unique_ptr<PcgBuffers> workspace;
        if (!workspace || workspace->count != count || workspace->nnz != nnz)
            workspace = std::make_unique<PcgBuffers>(runtime, count, nnz);
        PcgBuffers& buffers = *workspace;
        Buffer& rowBuffer = *buffers.row;
        Buffer& columnBuffer = *buffers.column;
        Buffer& valueBuffer = *buffers.value;
        Buffer& rhsBuffer = *buffers.rhs;
        Buffer& solutionBuffer = *buffers.solution;
        Buffer& residualBuffer = *buffers.residual;
        Buffer& transformedBuffer = *buffers.transformed;
        Buffer& directionBuffer = *buffers.direction;
        Buffer& matrixDirectionBuffer = *buffers.matrixDirection;
        Buffer& inverseBuffer = *buffers.inverse;
        Buffer& onesBuffer = *buffers.ones;
        Buffer& partialABuffer = *buffers.partialA;
        Buffer& partialBBuffer = *buffers.partialB;
        Buffer& rowUpload = *buffers.rowUpload;
        Buffer& columnUpload = *buffers.columnUpload;
        Buffer& valueUpload = *buffers.valueUpload;
        Buffer& rhsUpload = *buffers.rhsUpload;
        Buffer& solutionUpload = *buffers.solutionUpload;
        Buffer& inverseUpload = *buffers.inverseUpload;
        Buffer& onesUpload = *buffers.onesUpload;
        Buffer& solutionReadback = *buffers.solutionReadback;
        auto copyToBuffer = [](Buffer& buffer, const void* source, std::size_t bytes)
        {
            void* destination = buffer.map();
            std::memcpy(destination, source, bytes);
            buffer.unmap();
        };
        auto copyFromBuffer = [](Buffer& buffer, void* destination, std::size_t bytes)
        {
            const void* source = buffer.map();
            std::memcpy(destination, source, bytes);
            buffer.unmap();
        };
        copyToBuffer(rowUpload, rowOffsets.data(), rowOffsets.size() * sizeof(std::uint32_t));
        if (nnz != 0)
        {
            copyToBuffer(columnUpload, columns.data(), columns.size() * sizeof(std::uint32_t));
            copyToBuffer(valueUpload, matrix.values(), nnz * sizeof(float));
        }
        copyToBuffer(rhsUpload, rhs.data(), count * sizeof(float));
        copyToBuffer(solutionUpload, solution.data(), count * sizeof(float));
        copyToBuffer(inverseUpload, inverse.data(), count * sizeof(float));
        copyToBuffer(onesUpload, ones.data(), count * sizeof(float));

        static PipelineSet pipelines(runtime);
        const bool useSubgroupSpmv = shouldUseSubgroupSpmv(runtime, count, nnz);
        const ComputePipeline& spmvPipeline = pipelines.spmvPipeline(useSubgroupSpmv);
        const std::size_t spmvGroups = spmvGroupsFor(runtime, count, useSubgroupSpmv);
        CommandContext& context = *buffers.context;
        const std::uint32_t descriptorSetsBefore = context.descriptorSetAllocations();
        context.resetSubmissionCount();
        context.begin();
        context.copy(rowUpload, rowBuffer, rowOffsets.size() * sizeof(std::uint32_t));
        if (nnz != 0)
        {
            context.copy(columnUpload, columnBuffer, columns.size() * sizeof(std::uint32_t));
            context.copy(valueUpload, valueBuffer, nnz * sizeof(float));
        }
        context.copy(rhsUpload, rhsBuffer, count * sizeof(float));
        context.copy(solutionUpload, solutionBuffer, count * sizeof(float));
        context.copy(inverseUpload, inverseBuffer, count * sizeof(float));
        context.copy(onesUpload, onesBuffer, count * sizeof(float));

        const CountPush countPush{static_cast<std::uint32_t>(count)};
        IterativeSolverReport report;
        dispatch(context,
                 spmvPipeline,
                 {&rowBuffer, &columnBuffer, &valueBuffer, &solutionBuffer, &matrixDirectionBuffer},
                 spmvGroups,
                 &countPush,
                 sizeof(countPush));
        dispatch(
            context,
            pipelines.initialize,
            {&rhsBuffer, &matrixDirectionBuffer, &inverseBuffer, &residualBuffer, &transformedBuffer, &directionBuffer},
            groupsFor(count),
            &countPush,
            sizeof(countPush));
        Buffer* initialResidual = dispatchDot(context,
                                              pipelines.reduction,
                                              residualBuffer,
                                              residualBuffer,
                                              partialABuffer,
                                              partialBBuffer,
                                              onesBuffer,
                                              count);
        const StateInitPush stateInitPush{static_cast<float>(options.relativeTolerance * options.relativeTolerance),
                                          static_cast<float>(options.absoluteTolerance * options.absoluteTolerance)};
        dispatch(context,
                 pipelines.stateInit,
                 {buffers.state.get(), initialResidual},
                 1,
                 &stateInitPush,
                 sizeof(stateInitPush));
        Buffer* initialRho = dispatchDot(context,
                                         pipelines.reduction,
                                         residualBuffer,
                                         transformedBuffer,
                                         partialABuffer,
                                         partialBBuffer,
                                         onesBuffer,
                                         count);
        dispatch(context, pipelines.rhoInit, {buffers.state.get(), initialRho}, 1, nullptr, 0);

        int completedIterations = 0;
        bool firstBatch = true;
        while (firstBatch || (!report.converged && completedIterations < options.maxIterations))
        {
            if (!firstBatch)
                context.begin();

            const int batchLimit =
                std::min(options.convergenceCheckInterval, options.maxIterations - completedIterations);
            for (int batchIteration = 0; batchIteration < batchLimit; ++batchIteration)
            {
                if (completedIterations > 0)
                {
                    dispatch(context,
                             pipelines.jacobi,
                             {&residualBuffer, &inverseBuffer, &transformedBuffer},
                             groupsFor(count),
                             &countPush,
                             sizeof(countPush));
                    Buffer* nextRho = dispatchDot(context,
                                                  pipelines.reduction,
                                                  residualBuffer,
                                                  transformedBuffer,
                                                  partialABuffer,
                                                  partialBBuffer,
                                                  onesBuffer,
                                                  count);
                    dispatch(context, pipelines.beta, {buffers.state.get(), nextRho}, 1, nullptr, 0);
                    dispatch(context,
                             pipelines.direction,
                             {&directionBuffer, &transformedBuffer, buffers.state.get()},
                             groupsFor(count),
                             &countPush,
                             sizeof(countPush));
                }

                dispatch(context,
                         spmvPipeline,
                         {&rowBuffer, &columnBuffer, &valueBuffer, &directionBuffer, &matrixDirectionBuffer},
                         spmvGroups,
                         &countPush,
                         sizeof(countPush));
                Buffer* denominator = dispatchDot(context,
                                                  pipelines.reduction,
                                                  directionBuffer,
                                                  matrixDirectionBuffer,
                                                  partialABuffer,
                                                  partialBBuffer,
                                                  onesBuffer,
                                                  count);
                dispatch(context, pipelines.alpha, {buffers.state.get(), denominator}, 1, nullptr, 0);
                dispatch(
                    context,
                    pipelines.update,
                    {&solutionBuffer, &residualBuffer, &directionBuffer, &matrixDirectionBuffer, buffers.state.get()},
                    groupsFor(count),
                    &countPush,
                    sizeof(countPush));
                ++completedIterations;

                Buffer* residualNow = dispatchDot(context,
                                                  pipelines.reduction,
                                                  residualBuffer,
                                                  residualBuffer,
                                                  partialABuffer,
                                                  partialBBuffer,
                                                  onesBuffer,
                                                  count);
                const IterationPush iterationPush{static_cast<std::uint32_t>(completedIterations)};
                dispatch(context,
                         pipelines.convergence,
                         {buffers.state.get(), residualNow},
                         1,
                         &iterationPush,
                         sizeof(iterationPush));

                const bool finalIteration = completedIterations >= options.maxIterations;
                const bool check = batchIteration + 1 == batchLimit || finalIteration;
                if (check)
                    break;
            }

            if (fuseSolutionReadback)
                context.copy(solutionBuffer, solutionReadback, solutionBytes);
            context.copy(*buffers.state, *buffers.stateReadback, sizeof(PcgStateHost));
            context.submitAndWait();
            PcgStateHost state{};
            copyFromBuffer(*buffers.stateReadback, &state, sizeof(state));
            if ((state.flags & kBreakdownFlag) != 0u)
                throw std::runtime_error(
                    "Vulkan PCG breakdown in device scalar recurrence: rho=" + std::to_string(state.rho) +
                    " alpha=" + std::to_string(state.alpha) + " beta=" + std::to_string(state.beta) +
                    " residual=" + std::to_string(state.residualSquared) + " flags=" + std::to_string(state.flags));
            report.initialResidual = std::sqrt(static_cast<double>(state.initialResidualSquared));
            report.iterations = static_cast<int>(state.iterations);
            report.finalResidual = std::sqrt(static_cast<double>(state.residualSquared));
            report.converged = (state.flags & kConvergedFlag) != 0u;
            firstBatch = false;
        }

        if (!fuseSolutionReadback)
        {
            context.begin();
            context.copy(solutionBuffer, solutionReadback, solutionBytes);
            context.submitAndWait();
        }
        copyFromBuffer(solutionReadback, solution.data(), solutionBytes);
        report.commandSubmissions = context.submissionCount();
        report.descriptorSetAllocations = context.descriptorSetAllocations() - descriptorSetsBefore;
        report.gpuMilliseconds = context.gpuMilliseconds();
        report.barrierCount = context.barrierCount();
        report.subgroupSpmv = useSubgroupSpmv;
        return report;
    }

} // namespace plamatrix::vulkan

#endif
