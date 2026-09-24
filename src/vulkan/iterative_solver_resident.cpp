#include "plamatrix/internal/sparse/iterative_solver.h"

#ifdef PLAMATRIX_WITH_VULKAN

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "plamatrix/internal/core/error.h"
#include "plamatrix/internal/vulkan/execution.h"
#include "plamatrix/internal/vulkan/native.h"

namespace plamatrix::internal::resident_solver_detail
{
namespace
{

constexpr std::size_t local_size = 128;

struct CountPush
{
    std::uint32_t count;
};

struct ScalarPush
{
    std::uint32_t count;
    float scalar;
};

std::size_t groupsFor(std::size_t count)
{
    return (count + local_size - 1) / local_size;
}

std::size_t reductionGroupsFor(std::size_t count)
{
    return (count + 2 * local_size - 1) / (2 * local_size);
}

void fill(vulkan::Buffer& buffer, std::size_t count, float value)
{
    auto* mapped = static_cast<float*>(buffer.map());
    std::fill_n(mapped, count, value);
    buffer.unmap();
}

float readScalar(vulkan::Buffer& buffer)
{
    const float value = *static_cast<const float*>(buffer.map());
    buffer.unmap();
    return value;
}

void submit(vulkan::CommandContext& commands,
            const vulkan::ComputePipeline& pipeline,
            const std::vector<vulkan::Buffer*>& buffers,
            std::size_t groups,
            const void* push,
            std::uint32_t push_size)
{
    commands.begin();
    commands.previousSubmissionBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                       VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    commands.dispatch(pipeline, buffers, groups, push, push_size);
    commands.submitAndWait();
}

double dot(vulkan::CommandContext& commands,
           const vulkan::ComputePipeline& reduction,
           vulkan::Buffer& left,
           vulkan::Buffer& right,
           vulkan::Buffer& ones,
           vulkan::Buffer& partial_a,
           vulkan::Buffer& partial_b,
           std::size_t count)
{
    vulkan::Buffer* current_left = &left;
    vulkan::Buffer* current_right = &right;
    vulkan::Buffer* output = &partial_a;
    std::size_t remaining = count;
    commands.begin();
    commands.previousSubmissionBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                       VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    do
    {
        const std::size_t groups = reductionGroupsFor(remaining);
        const CountPush push{static_cast<std::uint32_t>(remaining)};
        commands.dispatch(reduction, {current_left, current_right, output}, groups, &push, sizeof(push));
        remaining = groups;
        current_left = output;
        current_right = &ones;
        output = output == &partial_a ? &partial_b : &partial_a;
    } while (remaining > 1);
    commands.submitAndWait();
    return static_cast<double>(readScalar(*current_left));
}

} // namespace

IterativeSolverReport pcgVulkanResident(const ResidentCsrMatrix<float>& matrix,
                                        const ResidentVector<float>& rhs,
                                        ResidentVector<float>& solution,
                                        const SolveOptions& options,
                                        ExecutionContext& context)
{
    if (matrix.rows() != matrix.cols() || rhs.size() != matrix.rows() ||
        solution.size() != matrix.cols() || &rhs == &solution)
    {
        throw Error(ErrorCode::InvalidArgument,
                    "Vulkan resident PCG requires a square CSR matrix and distinct matching vectors",
                    Backend::Vulkan);
    }
    if (options.maxIterations < 0 || !std::isfinite(options.relativeTolerance) ||
        options.relativeTolerance < 0.0 || options.relativeTolerance > 1.0 ||
        !std::isfinite(options.absoluteTolerance) || options.absoluteTolerance < 0.0)
    {
        throw Error(ErrorCode::InvalidArgument, "Vulkan resident PCG options are invalid", Backend::Vulkan);
    }
    IterativeSolverReport report;
    if (matrix.rows() == 0)
    {
        report.converged = true;
        return report;
    }
    if (matrix.nnz() == 0 || matrix.rows() > std::numeric_limits<std::uint32_t>::max() ||
        vulkan::NativeAccess::rows32(matrix) == nullptr ||
        vulkan::NativeAccess::columns32(matrix) == nullptr)
    {
        throw Error(ErrorCode::InvalidState,
                    "Vulkan resident PCG requires a nonempty CSR uploaded through ResidentCsrMatrix::copyFrom",
                    Backend::Vulkan);
    }
    if (options.cancellation && options.cancellation->isCancellationRequested())
    {
        throw Error(ErrorCode::InvalidState, "Vulkan resident PCG was cancelled", Backend::Vulkan);
    }

    auto& runtime = vulkan::NativeAccess::runtime(context);
    const auto count = static_cast<std::size_t>(matrix.rows());
    const auto partial_count = reductionGroupsFor(count);
    const auto storage_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    const auto vector_bytes = static_cast<VkDeviceSize>(count * sizeof(float));
    const auto partial_bytes = static_cast<VkDeviceSize>(partial_count * sizeof(float));
    vulkan::Buffer inverse(runtime, vector_bytes, storage_usage);
    vulkan::Buffer residual(runtime, vector_bytes, storage_usage);
    vulkan::Buffer transformed(runtime, vector_bytes, storage_usage);
    vulkan::Buffer direction(runtime, vector_bytes, storage_usage);
    vulkan::Buffer matrix_direction(runtime, vector_bytes, storage_usage);
    vulkan::Buffer ones(runtime, vector_bytes, storage_usage);
    vulkan::Buffer partial_a(runtime, partial_bytes, storage_usage);
    vulkan::Buffer partial_b(runtime, partial_bytes, storage_usage);
    fill(ones, count, 1.0F);

    vulkan::ComputePipeline spmv(runtime, "spmv", 5, sizeof(CountPush));
    vulkan::ComputePipeline initialize(runtime, "initialize", 6, sizeof(CountPush));
    vulkan::ComputePipeline build_jacobi(runtime, "resident_jacobi_build", 4, sizeof(CountPush));
    vulkan::ComputePipeline precondition(runtime, "jacobi", 3, sizeof(CountPush));
    vulkan::ComputePipeline reduction(runtime, "dot_reduce", 3, sizeof(CountPush));
    vulkan::ComputePipeline update(runtime, "resident_update", 4, sizeof(ScalarPush));
    vulkan::ComputePipeline direction_update(runtime, "resident_direction", 2, sizeof(ScalarPush));
    vulkan::CommandContext commands(runtime);
    auto* rows = vulkan::NativeAccess::rows32(matrix);
    auto* columns = vulkan::NativeAccess::columns32(matrix);
    auto* values = vulkan::NativeAccess::values(matrix);
    auto* rhs_buffer = vulkan::NativeAccess::buffer(rhs);
    auto* solution_buffer = vulkan::NativeAccess::buffer(solution);
    const CountPush count_push{static_cast<std::uint32_t>(count)};
    const auto groups = groupsFor(count);
    if (options.useJacobiPreconditioner)
    {
        submit(commands, build_jacobi, {rows, columns, values, &inverse},
               groups, &count_push, sizeof(count_push));
    }
    else
    {
        fill(inverse, count, 1.0F);
    }
    commands.begin();
    commands.previousSubmissionBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                       VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    commands.dispatch(spmv, {rows, columns, values, solution_buffer, &matrix_direction},
                      groups, &count_push, sizeof(count_push));
    commands.dispatch(initialize,
                      {rhs_buffer, &matrix_direction, &inverse, &residual, &transformed, &direction},
                      groups, &count_push, sizeof(count_push));
    commands.submitAndWait();

    const auto dot_vectors = [&](vulkan::Buffer& left, vulkan::Buffer& right)
    {
        return dot(commands, reduction, left, right, ones, partial_a, partial_b, count);
    };
    const double initial_squared = dot_vectors(residual, residual);
    if (!std::isfinite(initial_squared) || initial_squared < 0.0)
    {
        throw Error(ErrorCode::NumericalFailure,
                    "Vulkan resident PCG initial residual or Jacobi diagonal is invalid",
                    Backend::Vulkan);
    }
    report.initialResidual = std::sqrt(initial_squared);
    report.finalResidual = report.initialResidual;
    if (options.recordResidualHistory)
    {
        report.residualHistory.push_back(report.finalResidual);
    }
    const double tolerance = std::max(options.absoluteTolerance,
                                      options.relativeTolerance * report.initialResidual);
    report.converged = report.finalResidual <= tolerance;
    double rho = report.converged ? 0.0 : dot_vectors(residual, transformed);
    if (!report.converged && (!std::isfinite(rho) || rho <= 0.0))
    {
        throw Error(ErrorCode::NumericalFailure,
                    "Vulkan resident PCG preconditioner is invalid",
                    Backend::Vulkan);
    }
    for (int iteration = 0; iteration < options.maxIterations && !report.converged; ++iteration)
    {
        if (options.cancellation && options.cancellation->isCancellationRequested())
        {
            throw Error(ErrorCode::InvalidState, "Vulkan resident PCG was cancelled", Backend::Vulkan);
        }
        submit(commands, spmv, {rows, columns, values, &direction, &matrix_direction},
               groups, &count_push, sizeof(count_push));
        const double denominator = dot_vectors(direction, matrix_direction);
        if (!std::isfinite(denominator) || denominator <= 0.0)
        {
            throw Error(ErrorCode::NumericalFailure,
                        "Vulkan resident PCG matrix is not numerically SPD",
                        Backend::Vulkan);
        }
        const double alpha = rho / denominator;
        const ScalarPush update_push{count_push.count, static_cast<float>(alpha)};
        submit(commands, update,
               {solution_buffer, &residual, &direction, &matrix_direction},
               groups, &update_push, sizeof(update_push));
        const double squared = dot_vectors(residual, residual);
        if (!std::isfinite(squared) || squared < 0.0)
        {
            throw Error(ErrorCode::NumericalFailure, "Vulkan resident PCG residual is invalid", Backend::Vulkan);
        }
        report.finalResidual = std::sqrt(squared);
        report.iterations = iteration + 1;
        if (options.recordResidualHistory)
        {
            report.residualHistory.push_back(report.finalResidual);
        }
        report.converged = report.finalResidual <= tolerance;
        if (report.converged)
        {
            break;
        }
        submit(commands, precondition, {&residual, &inverse, &transformed},
               groups, &count_push, sizeof(count_push));
        const double next_rho = dot_vectors(residual, transformed);
        if (!std::isfinite(next_rho) || next_rho <= 0.0)
        {
            throw Error(ErrorCode::NumericalFailure,
                        "Vulkan resident PCG direction update failed",
                        Backend::Vulkan);
        }
        const ScalarPush direction_push{count_push.count, static_cast<float>(next_rho / rho)};
        submit(commands, direction_update, {&direction, &transformed},
               groups, &direction_push, sizeof(direction_push));
        rho = next_rho;
    }
    report.commandSubmissions = commands.submissionCount();
    report.commandBufferRecordings = commands.commandBufferRecordings();
    report.descriptorSetAllocations = commands.descriptorSetAllocations();
    report.barrierCount = commands.barrierCount();
    report.gpuMilliseconds = commands.gpuMilliseconds();
    report.diagnostics.commandSubmissions = report.commandSubmissions;
    report.diagnostics.barrierCount = report.barrierCount;
    report.diagnostics.gpuMilliseconds = report.gpuMilliseconds;
    report.diagnostics.deterministic = options.deterministic;
    if (!report.converged && options.requireConvergence)
    {
        throw Error(ErrorCode::NumericalFailure, "Vulkan resident PCG did not converge", Backend::Vulkan);
    }
    return report;
}

} // namespace plamatrix::internal::resident_solver_detail

#endif
