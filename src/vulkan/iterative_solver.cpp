#include "plamatrix/vulkan/iterative_solver.h"

#ifdef PLAMATRIX_WITH_VULKAN

#include "plamatrix/vulkan/execution.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <stdexcept>
#include <vector>

namespace plamatrix::vulkan
{
    namespace
    {

        constexpr std::uint32_t kLocalSize = 128;

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

        struct CountPush
        {
            std::uint32_t count;
        };

        struct ScalePush
        {
            std::uint32_t count;
            float value;
        };

        struct PipelineSet
        {
            explicit PipelineSet(Runtime& runtime)
                : residualInit(runtime, "residual_init", 3, sizeof(CountPush)),
                  spmv(runtime, "spmv", 5, sizeof(CountPush)), jacobi(runtime, "jacobi", 3, sizeof(CountPush)),
                  update(runtime, "update", 4, sizeof(ScalePush)),
                  direction(runtime, "direction", 2, sizeof(ScalePush)),
                  reduction(runtime, "dot_reduce", 3, sizeof(CountPush))
            {
            }

            ComputePipeline residualInit;
            ComputePipeline spmv;
            ComputePipeline jacobi;
            ComputePipeline update;
            ComputePipeline direction;
            ComputePipeline reduction;
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

        double dot(CommandContext& context,
                   const ComputePipeline& pipeline,
                   Buffer& left,
                   Buffer& right,
                   Buffer& partialA,
                   Buffer& partialB,
                   Buffer& ones,
                   std::size_t count)
        {
            std::size_t remaining = groupsFor(count);
            Buffer* current = &partialA;
            Buffer* next = &partialB;
            CountPush push{static_cast<std::uint32_t>(count)};
            dispatch(context, pipeline, {&left, &right, current}, remaining, &push, sizeof(push));
            while (remaining > 1)
            {
                const std::size_t nextGroups = groupsFor(remaining);
                push.count = static_cast<std::uint32_t>(remaining);
                dispatch(context, pipeline, {current, &ones, next}, nextGroups, &push, sizeof(push));
                remaining = nextGroups;
                std::swap(current, next);
            }
            context.submitAndWait();
            const float value = *static_cast<const float*>(current->map());
            current->unmap();
            return static_cast<double>(value);
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
        const std::size_t count = static_cast<std::size_t>(matrix.rows());
        const std::size_t nnz = static_cast<std::size_t>(matrix.nnz());
        std::vector<std::uint32_t> rowOffsets(count + 1);
        std::vector<std::uint32_t> columns(nnz);
        for (std::size_t i = 0; i < rowOffsets.size(); ++i)
            rowOffsets[i] = checkedUint(matrix.rowOffsets()[i], "row offset");
        for (std::size_t i = 0; i < columns.size(); ++i)
            columns[i] = checkedUint(matrix.colIndices()[i], "column index");
        const auto inverse = inverseDiagonal(matrix, options.useJacobiPreconditioner);
        std::vector<float> ones(count, 1.0f);

        const VkDeviceSize uintBytes = static_cast<VkDeviceSize>(sizeof(std::uint32_t));
        const VkDeviceSize floatBytes = static_cast<VkDeviceSize>(sizeof(float));
        Buffer rowBuffer(runtime, (count + 1) * uintBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        Buffer columnBuffer(runtime, std::max<std::size_t>(1, nnz) * uintBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        Buffer valueBuffer(runtime, std::max<std::size_t>(1, nnz) * floatBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        Buffer rhsBuffer(runtime, count * floatBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        Buffer solutionBuffer(runtime, count * floatBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        Buffer residualBuffer(runtime, count * floatBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        Buffer transformedBuffer(runtime, count * floatBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        Buffer directionBuffer(runtime, count * floatBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        Buffer matrixDirectionBuffer(runtime, count * floatBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        Buffer inverseBuffer(runtime, count * floatBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        Buffer onesBuffer(runtime, count * floatBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        const std::size_t partialCount = groupsFor(count);
        Buffer partialABuffer(runtime, partialCount * floatBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        Buffer partialBBuffer(runtime, partialCount * floatBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

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
        copyToBuffer(rowBuffer, rowOffsets.data(), rowOffsets.size() * sizeof(std::uint32_t));
        if (nnz != 0)
        {
            copyToBuffer(columnBuffer, columns.data(), columns.size() * sizeof(std::uint32_t));
            copyToBuffer(valueBuffer, matrix.values(), nnz * sizeof(float));
        }
        copyToBuffer(rhsBuffer, rhs.data(), count * sizeof(float));
        copyToBuffer(solutionBuffer, solution.data(), count * sizeof(float));
        copyToBuffer(inverseBuffer, inverse.data(), count * sizeof(float));
        copyToBuffer(onesBuffer, ones.data(), count * sizeof(float));
        std::vector<float> zeros(count, 0.0f);
        copyToBuffer(residualBuffer, zeros.data(), count * sizeof(float));
        copyToBuffer(transformedBuffer, zeros.data(), count * sizeof(float));
        copyToBuffer(directionBuffer, zeros.data(), count * sizeof(float));
        copyToBuffer(matrixDirectionBuffer, zeros.data(), count * sizeof(float));

        static PipelineSet pipelines(runtime);
        CommandContext context(runtime);
        const CountPush countPush{static_cast<std::uint32_t>(count)};
        context.begin();
        dispatch(context,
                 pipelines.spmv,
                 {&rowBuffer, &columnBuffer, &valueBuffer, &solutionBuffer, &matrixDirectionBuffer},
                 groupsFor(count),
                 &countPush,
                 sizeof(countPush));
        dispatch(context,
                 pipelines.residualInit,
                 {&rhsBuffer, &matrixDirectionBuffer, &residualBuffer},
                 groupsFor(count),
                 &countPush,
                 sizeof(countPush));
        dispatch(context,
                 pipelines.jacobi,
                 {&residualBuffer, &inverseBuffer, &transformedBuffer},
                 groupsFor(count),
                 &countPush,
                 sizeof(countPush));
        ScalePush zeroBeta{static_cast<std::uint32_t>(count), 0.0f};
        dispatch(context,
                 pipelines.direction,
                 {&directionBuffer, &transformedBuffer},
                 groupsFor(count),
                 &zeroBeta,
                 sizeof(zeroBeta));

        IterativeSolverReport report;
        const double residualSquared = dot(context,
                                           pipelines.reduction,
                                           residualBuffer,
                                           residualBuffer,
                                           partialABuffer,
                                           partialBBuffer,
                                           onesBuffer,
                                           count);
        report.initialResidual = std::sqrt(residualSquared);
        report.finalResidual = report.initialResidual;
        const double tolerance =
            std::max(options.absoluteTolerance, options.relativeTolerance * report.initialResidual);
        report.converged = report.finalResidual <= tolerance;
        context.begin();
        double rho = dot(context,
                         pipelines.reduction,
                         residualBuffer,
                         transformedBuffer,
                         partialABuffer,
                         partialBBuffer,
                         onesBuffer,
                         count);

        for (int iteration = 0; !report.converged && iteration < options.maxIterations; ++iteration)
        {
            if (!std::isfinite(rho) || rho <= 0.0)
                throw std::runtime_error("Vulkan PCG breakdown in residual");
            context.begin();
            dispatch(context,
                     pipelines.spmv,
                     {&rowBuffer, &columnBuffer, &valueBuffer, &directionBuffer, &matrixDirectionBuffer},
                     groupsFor(count),
                     &countPush,
                     sizeof(countPush));
            const double denominator = dot(context,
                                           pipelines.reduction,
                                           directionBuffer,
                                           matrixDirectionBuffer,
                                           partialABuffer,
                                           partialBBuffer,
                                           onesBuffer,
                                           count);
            if (!std::isfinite(denominator) || denominator <= 0.0)
                throw std::runtime_error("Vulkan PCG breakdown in matrix-direction product");
            const float alpha = static_cast<float>(rho / denominator);
            ScalePush updatePush{static_cast<std::uint32_t>(count), alpha};
            context.begin();
            dispatch(context,
                     pipelines.update,
                     {&solutionBuffer, &residualBuffer, &directionBuffer, &matrixDirectionBuffer},
                     groupsFor(count),
                     &updatePush,
                     sizeof(updatePush));
            report.iterations = iteration + 1;
            const bool check =
                report.iterations % options.convergenceCheckInterval == 0 || report.iterations == options.maxIterations;
            if (check)
            {
                const double residualNow = dot(context,
                                               pipelines.reduction,
                                               residualBuffer,
                                               residualBuffer,
                                               partialABuffer,
                                               partialBBuffer,
                                               onesBuffer,
                                               count);
                report.finalResidual = std::sqrt(residualNow);
                report.converged = std::isfinite(report.finalResidual) && report.finalResidual <= tolerance;
            }
            if (report.converged)
                break;

            if (check)
                context.begin();
            dispatch(context,
                     pipelines.jacobi,
                     {&residualBuffer, &inverseBuffer, &transformedBuffer},
                     groupsFor(count),
                     &countPush,
                     sizeof(countPush));
            const double nextRho = dot(context,
                                       pipelines.reduction,
                                       residualBuffer,
                                       transformedBuffer,
                                       partialABuffer,
                                       partialBBuffer,
                                       onesBuffer,
                                       count);
            if (!std::isfinite(nextRho) || nextRho <= 0.0)
                throw std::runtime_error("Vulkan PCG breakdown in direction update");
            ScalePush directionPush{static_cast<std::uint32_t>(count), static_cast<float>(nextRho / rho)};
            context.begin();
            dispatch(context,
                     pipelines.direction,
                     {&directionBuffer, &transformedBuffer},
                     groupsFor(count),
                     &directionPush,
                     sizeof(directionPush));
            context.submitAndWait();
            rho = nextRho;
        }

        copyFromBuffer(solutionBuffer, solution.data(), count * sizeof(float));
        report.commandSubmissions = context.submissionCount();
        return report;
    }

} // namespace plamatrix::vulkan

#endif
