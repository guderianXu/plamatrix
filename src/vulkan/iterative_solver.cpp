#include "plamatrix/vulkan/iterative_solver.h"

#ifdef PLAMATRIX_WITH_VULKAN

#include "plamatrix/vulkan/execution.h"

#include <algorithm>
#include <array>
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

        enum class DotStateOperation : std::uint32_t
        {
            Initialize = 0,
            RhoInitialize = 1,
            Beta = 2,
            Alpha = 3,
            Convergence = 4
        };

        struct DotStatePush
        {
            std::uint32_t count;
            DotStateOperation operation;
            float relativeToleranceSquared;
            float absoluteToleranceSquared;
        };

        static_assert(sizeof(DotStatePush) == sizeof(std::uint32_t) * 4);

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
                  dotState(runtime,
                           "pcg_dot_state",
                           3,
                           sizeof(DotStatePush),
                           {VK_ACCESS_SHADER_READ_BIT,
                            VK_ACCESS_SHADER_READ_BIT,
                            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT})
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
            ComputePipeline dotState;
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
            std::unique_ptr<CommandContext> iterationContext;
            int recordedBatchIterations = 0;
            bool recordedBatchSubgroup = false;

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
                iterationContext = std::make_unique<CommandContext>(runtime);
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

        void dispatchDotState(CommandContext& context,
                              const PipelineSet& pipelines,
                              Buffer& left,
                              Buffer& right,
                              Buffer& partialA,
                              Buffer& partialB,
                              Buffer& ones,
                              Buffer& state,
                              std::size_t count,
                              DotStateOperation operation,
                              float relativeToleranceSquared = 0.0f,
                              float absoluteToleranceSquared = 0.0f)
        {
            std::size_t remaining = count;
            Buffer* currentLeft = &left;
            Buffer* currentRight = &right;
            Buffer* output = &partialA;
            while (remaining > kLocalSize * 2)
            {
                const std::size_t groups = reductionGroupsFor(remaining);
                const CountPush reductionPush{static_cast<std::uint32_t>(remaining)};
                dispatch(context,
                         pipelines.reduction,
                         {currentLeft, currentRight, output},
                         groups,
                         &reductionPush,
                         sizeof(reductionPush));
                remaining = groups;
                currentLeft = output;
                currentRight = &ones;
                output = output == &partialA ? &partialB : &partialA;
            }
            const DotStatePush push{
                static_cast<std::uint32_t>(remaining), operation, relativeToleranceSquared, absoluteToleranceSquared};
            dispatch(context, pipelines.dotState, {currentLeft, currentRight, &state}, 1, &push, sizeof(push));
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
        CommandContext& iterationContext = *buffers.iterationContext;
        const std::array<CommandContext*, 2> contexts{&context, &iterationContext};
        std::array<std::uint32_t, 2> descriptorSetsBefore{};
        for (std::size_t index = 0; index < contexts.size(); ++index)
        {
            descriptorSetsBefore[index] = contexts[index]->descriptorSetAllocations();
            contexts[index]->resetSubmissionCount();
        }
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
        dispatchDotState(context,
                         pipelines,
                         residualBuffer,
                         residualBuffer,
                         partialABuffer,
                         partialBBuffer,
                         onesBuffer,
                         *buffers.state,
                         count,
                         DotStateOperation::Initialize,
                         static_cast<float>(options.relativeTolerance * options.relativeTolerance),
                         static_cast<float>(options.absoluteTolerance * options.absoluteTolerance));
        dispatchDotState(context,
                         pipelines,
                         residualBuffer,
                         transformedBuffer,
                         partialABuffer,
                         partialBBuffer,
                         onesBuffer,
                         *buffers.state,
                         count,
                         DotStateOperation::RhoInitialize);

        auto recordIteration = [&](CommandContext& target, bool firstIteration)
        {
            if (!firstIteration)
            {
                dispatch(target,
                         pipelines.jacobi,
                         {&residualBuffer, &inverseBuffer, &transformedBuffer},
                         groupsFor(count),
                         &countPush,
                         sizeof(countPush));
                dispatchDotState(target,
                                 pipelines,
                                 residualBuffer,
                                 transformedBuffer,
                                 partialABuffer,
                                 partialBBuffer,
                                 onesBuffer,
                                 *buffers.state,
                                 count,
                                 DotStateOperation::Beta);
                dispatch(target,
                         pipelines.direction,
                         {&directionBuffer, &transformedBuffer, buffers.state.get()},
                         groupsFor(count),
                         &countPush,
                         sizeof(countPush));
            }
            dispatch(target,
                     spmvPipeline,
                     {&rowBuffer, &columnBuffer, &valueBuffer, &directionBuffer, &matrixDirectionBuffer},
                     spmvGroups,
                     &countPush,
                     sizeof(countPush));
            dispatchDotState(target,
                             pipelines,
                             directionBuffer,
                             matrixDirectionBuffer,
                             partialABuffer,
                             partialBBuffer,
                             onesBuffer,
                             *buffers.state,
                             count,
                             DotStateOperation::Alpha);
            dispatch(target,
                     pipelines.update,
                     {&solutionBuffer, &residualBuffer, &directionBuffer, &matrixDirectionBuffer, buffers.state.get()},
                     groupsFor(count),
                     &countPush,
                     sizeof(countPush));
            dispatchDotState(target,
                             pipelines,
                             residualBuffer,
                             residualBuffer,
                             partialABuffer,
                             partialBBuffer,
                             onesBuffer,
                             *buffers.state,
                             count,
                             DotStateOperation::Convergence);
        };
        auto recordReadback = [&](CommandContext& target)
        {
            if (fuseSolutionReadback)
                target.copy(solutionBuffer, solutionReadback, solutionBytes);
            target.copy(*buffers.state, *buffers.stateReadback, sizeof(PcgStateHost));
        };
        auto updateReport = [&]()
        {
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
        };

        const int firstBatchIterations = std::min(options.convergenceCheckInterval, options.maxIterations);
        for (int iteration = 0; iteration < firstBatchIterations; ++iteration)
            recordIteration(context, iteration == 0);
        recordReadback(context);
        context.submitAndWait();
        updateReport();

        int scheduledIterations = firstBatchIterations;
        while (!report.converged && scheduledIterations < options.maxIterations)
        {
            const int batchIterations =
                std::min(options.convergenceCheckInterval, options.maxIterations - scheduledIterations);
            const bool reusableBatch = batchIterations == options.convergenceCheckInterval;
            const bool reuseRecording = reusableBatch && buffers.recordedBatchIterations == batchIterations &&
                                        buffers.recordedBatchSubgroup == useSubgroupSpmv;
            if (!reuseRecording)
            {
                if (reusableBatch)
                    iterationContext.beginReusable();
                else
                    iterationContext.begin();
                iterationContext.previousSubmissionBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                                           VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
                for (int iteration = 0; iteration < batchIterations; ++iteration)
                    recordIteration(iterationContext, false);
                recordReadback(iterationContext);
                buffers.recordedBatchIterations = reusableBatch ? batchIterations : 0;
                buffers.recordedBatchSubgroup = useSubgroupSpmv;
            }
            iterationContext.submitAndWait();
            scheduledIterations += batchIterations;
            updateReport();
        }

        if (!fuseSolutionReadback)
        {
            context.begin();
            context.previousSubmissionBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
            context.copy(solutionBuffer, solutionReadback, solutionBytes);
            context.submitAndWait();
        }
        copyFromBuffer(solutionReadback, solution.data(), solutionBytes);
        for (std::size_t index = 0; index < contexts.size(); ++index)
        {
            report.commandSubmissions += contexts[index]->submissionCount();
            report.commandBufferRecordings += contexts[index]->commandBufferRecordings();
            report.descriptorSetAllocations +=
                contexts[index]->descriptorSetAllocations() - descriptorSetsBefore[index];
            report.gpuMilliseconds += contexts[index]->gpuMilliseconds();
            report.barrierCount += contexts[index]->barrierCount();
        }
        report.subgroupSpmv = useSubgroupSpmv;
        return report;
    }

} // namespace plamatrix::vulkan

#endif
