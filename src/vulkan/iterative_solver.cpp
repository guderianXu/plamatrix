#include "plamatrix/internal/vulkan/iterative_solver.h"
#include "iterative_solver_device.h"

#ifdef PLAMATRIX_WITH_VULKAN

#include "plamatrix/internal/vulkan/execution.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <stdexcept>
#include <vector>

namespace plamatrix::internal::vulkan
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

        std::vector<float> inverseDiagonal(const CsrStorage<float, Device::CPU>& matrix, bool enabled)
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
            if (mode && std::strcmp(mode, "block") == 0)
                return false;
            if (mode && std::strcmp(mode, "auto") != 0)
                throw std::invalid_argument("PLAMATRIX_VULKAN_SPMV must be auto, scalar, subgroup, or block");
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

        struct BlockSpmvPush
        {
            std::uint32_t count;
            std::uint32_t blockSize;
        };

        struct BuildBlockJacobiPush
        {
            std::uint32_t blockRows;
            std::uint32_t blockSize;
        };

        struct BlockTopology
        {
            std::vector<std::uint32_t> rowOffsets;
            std::vector<std::uint32_t> columns;
        };

        bool buildBlockTopology(const CsrStorage<float, Device::CPU>& matrix, Index blockSize, BlockTopology* topology)
        {
            if (!topology || blockSize <= 1 || matrix.rows() % blockSize != 0)
                return false;
            const Index blockRows = matrix.rows() / blockSize;
            topology->rowOffsets.assign(static_cast<std::size_t>(blockRows + 1), 0);
            topology->columns.clear();
            for (Index blockRow = 0; blockRow < blockRows; ++blockRow)
            {
                const Index firstRow = blockRow * blockSize;
                const Index first = matrix.rowOffsets()[firstRow];
                const Index entries = matrix.rowOffsets()[firstRow + 1] - first;
                if (entries < 0 || entries % blockSize != 0)
                    return false;
                const Index blocks = entries / blockSize;
                topology->rowOffsets[static_cast<std::size_t>(blockRow)] =
                    checkedUint(topology->columns.size(), "block row offset");
                for (Index block = 0; block < blocks; ++block)
                {
                    const Index firstColumn = matrix.colIndices()[first + block * blockSize];
                    if (firstColumn < 0 || firstColumn % blockSize != 0)
                        return false;
                    const Index blockColumn = firstColumn / blockSize;
                    for (Index localRow = 0; localRow < blockSize; ++localRow)
                    {
                        const Index row = firstRow + localRow;
                        if (matrix.rowOffsets()[row + 1] - matrix.rowOffsets()[row] != entries)
                            return false;
                        const Index rowStart = matrix.rowOffsets()[row] + block * blockSize;
                        for (Index localColumn = 0; localColumn < blockSize; ++localColumn)
                        {
                            if (matrix.colIndices()[rowStart + localColumn] != blockColumn * blockSize + localColumn)
                            {
                                return false;
                            }
                        }
                    }
                    topology->columns.push_back(checkedUint(blockColumn, "block column index"));
                }
            }
            topology->rowOffsets.back() = checkedUint(topology->columns.size(), "block row offset");
            return true;
        }

        struct BlockPreconditionerPush
        {
            std::uint32_t count;
            std::uint32_t blockSize;
            std::uint32_t initializeDirection;
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
                  spmvBlock(runtime,
                            "spmv_block",
                            5,
                            sizeof(BlockSpmvPush),
                            {VK_ACCESS_SHADER_READ_BIT,
                             VK_ACCESS_SHADER_READ_BIT,
                             VK_ACCESS_SHADER_READ_BIT,
                             VK_ACCESS_SHADER_READ_BIT,
                             VK_ACCESS_SHADER_WRITE_BIT}),
                  initializeResidual(
                      runtime,
                      "initialize_residual",
                      3,
                      sizeof(CountPush),
                      {VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT}),
                  jacobi(runtime,
                         "jacobi",
                         3,
                         sizeof(CountPush),
                         {VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT}),
                  blockJacobi(runtime,
                              "block_jacobi",
                              4,
                              sizeof(BlockPreconditionerPush),
                              {VK_ACCESS_SHADER_READ_BIT,
                               VK_ACCESS_SHADER_READ_BIT,
                               VK_ACCESS_SHADER_WRITE_BIT,
                               VK_ACCESS_SHADER_WRITE_BIT}),
                  buildBlockJacobi(runtime,
                                   "build_block_jacobi",
                                   4,
                                   sizeof(BuildBlockJacobiPush),
                                   {VK_ACCESS_SHADER_READ_BIT,
                                    VK_ACCESS_SHADER_READ_BIT,
                                    VK_ACCESS_SHADER_READ_BIT,
                                    VK_ACCESS_SHADER_WRITE_BIT}),
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
            ComputePipeline spmvBlock;
            std::unique_ptr<ComputePipeline> spmvSubgroup;
            ComputePipeline initializeResidual;
            ComputePipeline jacobi;
            ComputePipeline blockJacobi;
            ComputePipeline buildBlockJacobi;
            ComputePipeline update;
            ComputePipeline direction;
            ComputePipeline reduction;
            ComputePipeline dotState;
        };

        struct PcgBuffers
        {
            const std::size_t count;
            const std::size_t nnz;
            const std::size_t inverseCount;
            const VkBuffer externalValueBuffer;
            const std::uint64_t externalValueGeneration;
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
            std::uint64_t uploadedTopologyGeneration = 0;
            bool uploadedBlockTopology = false;
            bool blockSpmvAvailable = false;
            Index blockSpmvBlockSize = 0;
            int recordedInitialBatchIterations = 0;
            Index recordedInitialBlockSize = 0;
            float recordedInitialRelativeToleranceSquared = 0.0f;
            float recordedInitialAbsoluteToleranceSquared = 0.0f;
            bool initialRecordingReusable = false;
            bool recordedInitialSubgroup = false;
            bool recordedInitialBlockSpmv = false;
            bool recordedInitialDeviceBlockJacobi = false;
            bool recordedInitialFusedReadback = false;
            int recordedBatchIterations = 0;
            bool recordedBatchSubgroup = false;
            bool recordedBatchBlockSpmv = false;

            PcgBuffers(Runtime& runtime,
                       std::size_t countValue,
                       std::size_t nnzValue,
                       std::size_t inverseCountValue,
                       VkBuffer externalValueBufferValue,
                       std::uint64_t externalValueGenerationValue)
                : count(countValue), nnz(nnzValue), inverseCount(inverseCountValue),
                  externalValueBuffer(externalValueBufferValue), externalValueGeneration(externalValueGenerationValue)
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
                if (externalValueBuffer == VK_NULL_HANDLE)
                {
                    value = std::make_unique<Buffer>(runtime,
                                                     std::max<std::size_t>(1, nnz) * floatBytes,
                                                     deviceInputUsage,
                                                     BufferMemory::DeviceLocal);
                }
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
                inverse = std::make_unique<Buffer>(
                    runtime, inverseCount * floatBytes, deviceInputUsage, BufferMemory::DeviceLocal);
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
                if (externalValueBuffer == VK_NULL_HANDLE)
                {
                    valueUpload = std::make_unique<Buffer>(
                        runtime, std::max<std::size_t>(1, nnz) * floatBytes, stagingSourceUsage);
                }
                rhsUpload = std::make_unique<Buffer>(runtime, count * floatBytes, stagingSourceUsage);
                solutionUpload = std::make_unique<Buffer>(runtime, count * floatBytes, stagingSourceUsage);
                inverseUpload = std::make_unique<Buffer>(runtime, inverseCount * floatBytes, stagingSourceUsage);
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

    namespace
    {

        IterativeSolverReport pcgImpl(const CsrStorage<float, Device::CPU>& matrix,
                                      const DenseStorage<float, Device::CPU>& rhs,
                                      DenseStorage<float, Device::CPU>& solution,
                                      const DenseStorage<float, Device::CPU>* inverseBlocks,
                                      Index blockSize,
                                      Buffer* deviceValues,
                                      std::uint64_t deviceValuesGeneration,
                                      std::uint64_t topologyGeneration,
                                      const DevicePcgPrefixRecorder* prefixRecorder,
                                      bool requireDeviceBlockJacobi,
                                      const IterativeSolverOptions& options)
        {
            validateOptions(options);
            if (matrix.rows() <= 0 || matrix.rows() != matrix.cols() || rhs.rows() != matrix.rows() ||
                rhs.cols() != 1 || solution.rows() != matrix.rows() || solution.cols() != 1 ||
                rhs.data() == solution.data())
            {
                throw std::invalid_argument("Vulkan PCG requires a non-empty square system and distinct vectors");
            }
            for (Index row = 0; row < matrix.rows(); ++row)
            {
                if (!std::isfinite(rhs.data()[row]) || !std::isfinite(solution.data()[row]))
                    throw std::invalid_argument("Vulkan PCG vectors must contain finite values");
            }
            const bool useBlockPreconditioner = inverseBlocks != nullptr || requireDeviceBlockJacobi;
            if (useBlockPreconditioner && (blockSize <= 0 || matrix.rows() % blockSize != 0))
            {
                throw std::invalid_argument("Vulkan block PCG received invalid inverse diagonal blocks");
            }
            if (inverseBlocks && (inverseBlocks->cols() != 1 || inverseBlocks->rows() != matrix.rows() * blockSize))
            {
                throw std::invalid_argument("Vulkan block PCG received invalid inverse diagonal blocks");
            }
            if (deviceValues && deviceValues->size() < static_cast<VkDeviceSize>(matrix.nnz()) * sizeof(float))
            {
                throw std::invalid_argument("Vulkan PCG device values buffer is too small");
            }

            Runtime& runtime = Runtime::instance();
            static std::mutex solveMutex;
            const std::lock_guard<std::mutex> solveLock(solveMutex);
            const std::size_t count = static_cast<std::size_t>(matrix.rows());
            const std::size_t nnz = static_cast<std::size_t>(matrix.nnz());
            const std::size_t solutionBytes = count * sizeof(float);
            const bool fuseSolutionReadback = solutionBytes <= kFusedSolutionReadbackMaxBytes;
            if (!deviceValues)
                matrix.validateStructure();
            const auto scalarInverse = useBlockPreconditioner
                                           ? std::vector<float>{}
                                           : inverseDiagonal(matrix, options.useJacobiPreconditioner);
            const float* inverseData = inverseBlocks ? inverseBlocks->data() : scalarInverse.data();
            const std::size_t inverseCount =
                useBlockPreconditioner ? count * static_cast<std::size_t>(blockSize) : scalarInverse.size();
            std::vector<float> ones(count, 1.0f);

            static std::unique_ptr<PcgBuffers> workspace;
            const VkBuffer externalValueBuffer = deviceValues ? deviceValues->handle() : VK_NULL_HANDLE;
            if (!workspace || workspace->count != count || workspace->nnz != nnz ||
                workspace->inverseCount != inverseCount || workspace->externalValueBuffer != externalValueBuffer ||
                workspace->externalValueGeneration != deviceValuesGeneration)
            {
                workspace = std::make_unique<PcgBuffers>(
                    runtime, count, nnz, inverseCount, externalValueBuffer, deviceValuesGeneration);
            }
            PcgBuffers& buffers = *workspace;
            bool uploadCsrTopology = !deviceValues || buffers.uploadedTopologyGeneration != topologyGeneration;
            const char* requestedSpmvMode = std::getenv("PLAMATRIX_VULKAN_SPMV");
            const bool requestBlockSpmv = requestedSpmvMode && std::strcmp(requestedSpmvMode, "block") == 0;
            const bool selectAvailableBlockSpmv =
                buffers.blockSpmvAvailable &&
                (!requestedSpmvMode || std::strcmp(requestedSpmvMode, "auto") == 0 || requestBlockSpmv);
            if (!uploadCsrTopology && selectAvailableBlockSpmv != buffers.uploadedBlockTopology)
                uploadCsrTopology = true;
            std::vector<std::uint32_t> rowOffsets;
            std::vector<std::uint32_t> columns;
            if (uploadCsrTopology)
            {
                if (deviceValues)
                    matrix.validateStructure();
                BlockTopology blockTopology;
                buffers.blockSpmvAvailable = deviceValues && useBlockPreconditioner && blockSize == 9 &&
                                             buildBlockTopology(matrix, blockSize, &blockTopology);
                buffers.blockSpmvBlockSize = buffers.blockSpmvAvailable ? blockSize : 0;
                if (requestBlockSpmv && !buffers.blockSpmvAvailable)
                    throw std::runtime_error("PLAMATRIX_VULKAN_SPMV=block requires a dense block CSR matrix");
                const bool selectBlock =
                    buffers.blockSpmvAvailable &&
                    (!requestedSpmvMode || std::strcmp(requestedSpmvMode, "auto") == 0 || requestBlockSpmv);
                if (selectBlock)
                {
                    rowOffsets = std::move(blockTopology.rowOffsets);
                    columns = std::move(blockTopology.columns);
                }
                else
                {
                    rowOffsets.resize(count + 1);
                    columns.resize(nnz);
                    for (std::size_t index = 0; index < rowOffsets.size(); ++index)
                        rowOffsets[index] = checkedUint(matrix.rowOffsets()[index], "row offset");
                    for (std::size_t index = 0; index < columns.size(); ++index)
                        columns[index] = checkedUint(matrix.colIndices()[index], "column index");
                }
            }
            const bool useBlockSpmv =
                buffers.blockSpmvAvailable && (!requestedSpmvMode || std::strcmp(requestedSpmvMode, "auto") == 0 ||
                                               std::strcmp(requestedSpmvMode, "block") == 0);
            const char* requestedBlockJacobiMode = std::getenv("PLAMATRIX_VULKAN_BLOCK_JACOBI");
            if (requestedBlockJacobiMode && std::strcmp(requestedBlockJacobiMode, "auto") != 0 &&
                std::strcmp(requestedBlockJacobiMode, "cpu") != 0 && std::strcmp(requestedBlockJacobiMode, "gpu") != 0)
            {
                throw std::invalid_argument("PLAMATRIX_VULKAN_BLOCK_JACOBI must be auto, cpu, or gpu");
            }
            const bool requestGpuBlockJacobi =
                requestedBlockJacobiMode && std::strcmp(requestedBlockJacobiMode, "gpu") == 0;
            const bool useDeviceBlockJacobi =
                useBlockPreconditioner && useBlockSpmv &&
                (!requestedBlockJacobiMode || std::strcmp(requestedBlockJacobiMode, "auto") == 0 ||
                 requestGpuBlockJacobi);
            if (requestGpuBlockJacobi && !useDeviceBlockJacobi)
            {
                throw std::runtime_error(
                    "PLAMATRIX_VULKAN_BLOCK_JACOBI=gpu requires Vulkan device Schur values and 9x9 block SpMV");
            }
            if (requireDeviceBlockJacobi && !useDeviceBlockJacobi)
            {
                throw std::runtime_error(
                    "Vulkan Schur requested device block-Jacobi but the 9x9 block topology is unavailable");
            }
            Buffer& rowBuffer = *buffers.row;
            Buffer& columnBuffer = *buffers.column;
            Buffer& valueBuffer = deviceValues ? *deviceValues : *buffers.value;
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
            Buffer* valueUpload = buffers.valueUpload.get();
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
            if (uploadCsrTopology)
            {
                copyToBuffer(rowUpload, rowOffsets.data(), rowOffsets.size() * sizeof(std::uint32_t));
                if (nnz != 0)
                    copyToBuffer(columnUpload, columns.data(), columns.size() * sizeof(std::uint32_t));
            }
            if (!deviceValues && nnz != 0)
                copyToBuffer(*valueUpload, matrix.values(), nnz * sizeof(float));
            copyToBuffer(rhsUpload, rhs.data(), count * sizeof(float));
            copyToBuffer(solutionUpload, solution.data(), count * sizeof(float));
            if (!useDeviceBlockJacobi)
                copyToBuffer(inverseUpload, inverseData, inverseCount * sizeof(float));
            copyToBuffer(onesUpload, ones.data(), count * sizeof(float));

            static PipelineSet pipelines(runtime);
            const bool useSubgroupSpmv = !useBlockSpmv && shouldUseSubgroupSpmv(runtime, count, nnz);
            const ComputePipeline& spmvPipeline =
                useBlockSpmv ? pipelines.spmvBlock : pipelines.spmvPipeline(useSubgroupSpmv);
            const std::size_t spmvGroups =
                useBlockSpmv ? groupsFor(count) : spmvGroupsFor(runtime, count, useSubgroupSpmv);
            CommandContext& context = *buffers.context;
            CommandContext& iterationContext = *buffers.iterationContext;
            const std::array<CommandContext*, 2> contexts{&context, &iterationContext};
            std::array<std::uint32_t, 2> descriptorSetsBefore{};
            for (std::size_t index = 0; index < contexts.size(); ++index)
            {
                descriptorSetsBefore[index] = contexts[index]->descriptorSetAllocations();
                contexts[index]->resetSubmissionCount();
            }
            const CountPush countPush{static_cast<std::uint32_t>(count)};
            const BlockSpmvPush blockSpmvPush{static_cast<std::uint32_t>(count), static_cast<std::uint32_t>(blockSize)};
            const BuildBlockJacobiPush buildBlockJacobiPush{static_cast<std::uint32_t>(count / blockSize),
                                                            static_cast<std::uint32_t>(blockSize)};
            const BlockPreconditionerPush initializeBlockPush{
                static_cast<std::uint32_t>(count), static_cast<std::uint32_t>(blockSize), 1u};
            const BlockPreconditionerPush updateBlockPush{
                static_cast<std::uint32_t>(count), static_cast<std::uint32_t>(blockSize), 0u};
            const float relativeToleranceSquared =
                static_cast<float>(options.relativeTolerance * options.relativeTolerance);
            const float absoluteToleranceSquared =
                static_cast<float>(options.absoluteTolerance * options.absoluteTolerance);
            const int firstBatchIterations = std::min(options.convergenceCheckInterval, options.maxIterations);
            const bool reusableInitialRecording = deviceValues && prefixRecorder && !uploadCsrTopology;
            const bool reuseInitialRecording =
                reusableInitialRecording && buffers.initialRecordingReusable &&
                buffers.recordedInitialBatchIterations == firstBatchIterations &&
                buffers.recordedInitialBlockSize == blockSize &&
                buffers.recordedInitialRelativeToleranceSquared == relativeToleranceSquared &&
                buffers.recordedInitialAbsoluteToleranceSquared == absoluteToleranceSquared &&
                buffers.recordedInitialSubgroup == useSubgroupSpmv &&
                buffers.recordedInitialBlockSpmv == useBlockSpmv &&
                buffers.recordedInitialDeviceBlockJacobi == useDeviceBlockJacobi &&
                buffers.recordedInitialFusedReadback == fuseSolutionReadback;
            if (!reuseInitialRecording)
            {
                if (reusableInitialRecording)
                    context.beginReusable();
                else
                    context.begin();
                if (prefixRecorder)
                    (*prefixRecorder)(context);
                if (uploadCsrTopology)
                {
                    context.copy(rowUpload, rowBuffer, rowOffsets.size() * sizeof(std::uint32_t));
                    if (nnz != 0)
                        context.copy(columnUpload, columnBuffer, columns.size() * sizeof(std::uint32_t));
                }
                if (!deviceValues && nnz != 0)
                    context.copy(*valueUpload, valueBuffer, nnz * sizeof(float));
                context.copy(rhsUpload, rhsBuffer, count * sizeof(float));
                context.copy(solutionUpload, solutionBuffer, count * sizeof(float));
                if (useDeviceBlockJacobi)
                {
                    dispatch(context,
                             pipelines.buildBlockJacobi,
                             {&rowBuffer, &columnBuffer, &valueBuffer, &inverseBuffer},
                             groupsFor(count / static_cast<std::size_t>(blockSize)),
                             &buildBlockJacobiPush,
                             sizeof(buildBlockJacobiPush));
                }
                else
                {
                    context.copy(inverseUpload, inverseBuffer, inverseCount * sizeof(float));
                }
                context.copy(onesUpload, onesBuffer, count * sizeof(float));

                dispatch(context,
                         spmvPipeline,
                         {&rowBuffer, &columnBuffer, &valueBuffer, &solutionBuffer, &matrixDirectionBuffer},
                         spmvGroups,
                         useBlockSpmv ? static_cast<const void*>(&blockSpmvPush) : static_cast<const void*>(&countPush),
                         useBlockSpmv ? sizeof(blockSpmvPush) : sizeof(countPush));
                if (useBlockPreconditioner)
                {
                    dispatch(context,
                             pipelines.initializeResidual,
                             {&rhsBuffer, &matrixDirectionBuffer, &residualBuffer},
                             groupsFor(count),
                             &countPush,
                             sizeof(countPush));
                    dispatch(context,
                             pipelines.blockJacobi,
                             {&residualBuffer, &inverseBuffer, &transformedBuffer, &directionBuffer},
                             groupsFor(count),
                             &initializeBlockPush,
                             sizeof(initializeBlockPush));
                }
                else
                {
                    dispatch(context,
                             pipelines.initialize,
                             {&rhsBuffer,
                              &matrixDirectionBuffer,
                              &inverseBuffer,
                              &residualBuffer,
                              &transformedBuffer,
                              &directionBuffer},
                             groupsFor(count),
                             &countPush,
                             sizeof(countPush));
                }
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
                                 relativeToleranceSquared,
                                 absoluteToleranceSquared);
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
            }

            IterativeSolverReport report;

            auto recordIteration = [&](CommandContext& target, bool firstIteration)
            {
                if (!firstIteration)
                {
                    if (useBlockPreconditioner)
                    {
                        dispatch(target,
                                 pipelines.blockJacobi,
                                 {&residualBuffer, &inverseBuffer, &transformedBuffer, &directionBuffer},
                                 groupsFor(count),
                                 &updateBlockPush,
                                 sizeof(updateBlockPush));
                    }
                    else
                    {
                        dispatch(target,
                                 pipelines.jacobi,
                                 {&residualBuffer, &inverseBuffer, &transformedBuffer},
                                 groupsFor(count),
                                 &countPush,
                                 sizeof(countPush));
                    }
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
                         useBlockSpmv ? static_cast<const void*>(&blockSpmvPush) : static_cast<const void*>(&countPush),
                         useBlockSpmv ? sizeof(blockSpmvPush) : sizeof(countPush));
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
                dispatch(
                    target,
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

            if (!reuseInitialRecording)
            {
                for (int iteration = 0; iteration < firstBatchIterations; ++iteration)
                    recordIteration(context, iteration == 0);
                recordReadback(context);
                buffers.initialRecordingReusable = reusableInitialRecording;
                buffers.recordedInitialBatchIterations = firstBatchIterations;
                buffers.recordedInitialBlockSize = blockSize;
                buffers.recordedInitialRelativeToleranceSquared = relativeToleranceSquared;
                buffers.recordedInitialAbsoluteToleranceSquared = absoluteToleranceSquared;
                buffers.recordedInitialSubgroup = useSubgroupSpmv;
                buffers.recordedInitialBlockSpmv = useBlockSpmv;
                buffers.recordedInitialDeviceBlockJacobi = useDeviceBlockJacobi;
                buffers.recordedInitialFusedReadback = fuseSolutionReadback;
            }
            context.submitAndWait();
            if (uploadCsrTopology)
            {
                buffers.uploadedTopologyGeneration = topologyGeneration;
                buffers.uploadedBlockTopology = useBlockSpmv;
            }
            updateReport();

            int scheduledIterations = firstBatchIterations;
            while (!report.converged && scheduledIterations < options.maxIterations)
            {
                const int batchIterations =
                    std::min(options.convergenceCheckInterval, options.maxIterations - scheduledIterations);
                const bool reusableBatch = batchIterations == options.convergenceCheckInterval;
                const bool reuseRecording = reusableBatch && buffers.recordedBatchIterations == batchIterations &&
                                            buffers.recordedBatchSubgroup == useSubgroupSpmv &&
                                            buffers.recordedBatchBlockSpmv == useBlockSpmv;
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
                    buffers.recordedBatchBlockSpmv = useBlockSpmv;
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
            report.blockSpmv = useBlockSpmv;
            report.deviceBlockJacobi = useDeviceBlockJacobi;
            return report;
        }

    } // namespace

    IterativeSolverReport pcg(const CsrStorage<float, Device::CPU>& matrix,
                              const DenseStorage<float, Device::CPU>& rhs,
                              DenseStorage<float, Device::CPU>& solution,
                              const IterativeSolverOptions& options)
    {
        return pcgImpl(matrix, rhs, solution, nullptr, 1, nullptr, 0, 0, nullptr, false, options);
    }

    IterativeSolverReport blockPcg(const CsrStorage<float, Device::CPU>& matrix,
                                   const DenseStorage<float, Device::CPU>& rhs,
                                   DenseStorage<float, Device::CPU>& solution,
                                   const DenseStorage<float, Device::CPU>& inverseBlocks,
                                   Index blockSize,
                                   const IterativeSolverOptions& options)
    {
        return pcgImpl(matrix, rhs, solution, &inverseBlocks, blockSize, nullptr, 0, 0, nullptr, false, options);
    }

    IterativeSolverReport blockPcgWithDeviceValues(const CsrStorage<float, Device::CPU>& matrix,
                                                   const DenseStorage<float, Device::CPU>& rhs,
                                                   DenseStorage<float, Device::CPU>& solution,
                                                   const DenseStorage<float, Device::CPU>* inverseBlocks,
                                                   Index blockSize,
                                                   Buffer& deviceValues,
                                                   std::uint64_t deviceValuesGeneration,
                                                   std::uint64_t topologyGeneration,
                                                   const DevicePcgPrefixRecorder& prefixRecorder,
                                                   bool requireDeviceBlockJacobi,
                                                   const IterativeSolverOptions& options)
    {
        return pcgImpl(matrix,
                       rhs,
                       solution,
                       inverseBlocks,
                       blockSize,
                       &deviceValues,
                       deviceValuesGeneration,
                       topologyGeneration,
                       &prefixRecorder,
                       requireDeviceBlockJacobi,
                       options);
    }

} // namespace plamatrix::internal::vulkan

#endif
