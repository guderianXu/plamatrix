#include "plamatrix/opencl/iterative_solver.h"

#include "plamatrix/opencl/execution.h"

#include "iterative_solver_kernels.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
namespace plamatrix
{
namespace opencl
{
namespace
{
using iterative_solver_detail::kSolverSource;

void validateOptions(const IterativeSolverOptions& options)
{
    if (options.maxIterations < 0 || !std::isfinite(options.relativeTolerance)
        || options.relativeTolerance < 0.0 || options.relativeTolerance > 1.0
        || !std::isfinite(options.absoluteTolerance) || options.absoluteTolerance < 0.0)
    {
        throw std::invalid_argument("OpenCL PCG received invalid solver options");
    }
}
template <typename Scalar>
std::vector<Scalar> inverseDiagonal(
    const CSRMatrix<Scalar, Device::CPU>& matrix, bool enabled)
{
    std::vector<Scalar> result(static_cast<std::size_t>(matrix.rows()), Scalar{1});
    if (!enabled) return result;
    for (Index row = 0; row < matrix.rows(); ++row)
    {
        long double diagonal = 0.0L;
        for (Index entry = matrix.rowOffsets()[row]; entry < matrix.rowOffsets()[row + 1]; ++entry)
            if (matrix.colIndices()[entry] == row)
                diagonal += static_cast<long double>(matrix.values()[entry]);
        if (!std::isfinite(diagonal) || diagonal <= 0.0L)
            throw std::runtime_error("OpenCL PCG Jacobi preconditioner requires a positive diagonal");
        const Scalar inverse = Scalar{1} / static_cast<Scalar>(diagonal);
        if (!std::isfinite(inverse) || inverse <= Scalar{0})
            throw std::runtime_error("OpenCL PCG Jacobi inverse diagonal is not finite and positive");
        result[static_cast<std::size_t>(row)] = inverse;
    }
    return result;
}

std::size_t localSize(OpenClRuntime& runtime)
{
    std::size_t maximum = 1;
    checkOpenCl(clGetDeviceInfo(
        runtime.device(), CL_DEVICE_MAX_WORK_GROUP_SIZE,
        sizeof(maximum), &maximum, nullptr), "clGetDeviceInfo(max work-group size)");
    std::size_t result = 1;
    while (result * 2 <= std::min<std::size_t>(maximum, 256)) result *= 2;
    return result;
}

std::size_t groupsFor(std::size_t size, std::size_t local_size)
{
    return (size + local_size * 2 - 1) / (local_size * 2);
}

void launch(cl_command_queue queue, cl_kernel kernel, std::size_t size, std::size_t local_size,
            const char* operation)
{
    const std::size_t global_size = ((size + local_size - 1) / local_size) * local_size;
    checkOpenCl(clEnqueueNDRangeKernel(
        queue, kernel, 1, nullptr, &global_size, &local_size, 0, nullptr, nullptr), operation);
}

template <typename Scalar>
double dot(
    cl_command_queue queue, CompiledKernel& dot_kernel, CompiledKernel& reduce_kernel,
    const DeviceBuffer& first, const DeviceBuffer& second,
    DeviceBuffer& partial_a, DeviceBuffer& partial_b,
    Index size, std::size_t local_size, bool double_accumulation)
{
    const std::size_t accumulation_bytes = double_accumulation ? sizeof(double) : sizeof(Scalar);
    const std::size_t groups = groupsFor(static_cast<std::size_t>(size), local_size);
    kernelBufferArg(dot_kernel, 0, first);
    kernelBufferArg(dot_kernel, 1, second);
    kernelBufferArg(dot_kernel, 2, partial_a);
    kernelArg(dot_kernel, 3, static_cast<cl_long>(size));
    checkOpenCl(clSetKernelArg(dot_kernel, 4, local_size * accumulation_bytes, nullptr),
                "clSetKernelArg(dot local memory)");
    launch(queue, dot_kernel, groups * local_size, local_size, "clEnqueueNDRangeKernel(dot)");

    std::size_t remaining = groups;
    DeviceBuffer* current = &partial_a;
    DeviceBuffer* next = &partial_b;
    while (remaining > 1)
    {
        const std::size_t next_groups = groupsFor(remaining, local_size);
        kernelBufferArg(reduce_kernel, 0, *current);
        kernelBufferArg(reduce_kernel, 1, *next);
        kernelArg(reduce_kernel, 2, static_cast<cl_long>(remaining));
        checkOpenCl(clSetKernelArg(reduce_kernel, 3, local_size * accumulation_bytes, nullptr),
                    "clSetKernelArg(reduction local memory)");
        launch(queue, reduce_kernel, next_groups * local_size, local_size,
               "clEnqueueNDRangeKernel(reduction)");
        remaining = next_groups;
        std::swap(current, next);
    }
    if (double_accumulation)
    {
        double result = 0.0;
        checkOpenCl(clEnqueueReadBuffer(queue, current->get(), CL_TRUE, 0, sizeof(result),
                                        &result, 0, nullptr, nullptr), "clEnqueueReadBuffer(dot)");
        return result;
    }
    Scalar result{};
    checkOpenCl(clEnqueueReadBuffer(queue, current->get(), CL_TRUE, 0, sizeof(result),
                                    &result, 0, nullptr, nullptr), "clEnqueueReadBuffer(dot)");
    return static_cast<double>(result);
}

template <typename Scalar>
void validateBlockPreconditioner(
    const CSRMatrix<Scalar, Device::CPU>& matrix,
    const DenseMatrix<Scalar, Device::CPU>& rhs,
    const DenseMatrix<Scalar, Device::CPU>& solution,
    const DenseMatrix<Scalar, Device::CPU>& inverse_blocks,
    Index block_size)
{
    if (block_size <= 0 || matrix.rows() % block_size != 0
        || (matrix.rows() > 0
            && block_size > std::numeric_limits<Index>::max() / matrix.rows())
        || inverse_blocks.rows() != matrix.rows() * block_size
        || inverse_blocks.cols() != 1)
    {
        throw std::invalid_argument(
            "OpenCL block PCG requires complete row-major inverse diagonal blocks");
    }
    if (inverse_blocks.data() == rhs.data() || inverse_blocks.data() == solution.data())
    {
        throw std::invalid_argument(
            "OpenCL block PCG inverse blocks must not alias rhs or solution");
    }
    for (Index index = 0; index < inverse_blocks.rows(); ++index)
    {
        if (!std::isfinite(inverse_blocks.data()[index]))
        {
            throw std::invalid_argument(
                "OpenCL block PCG inverse blocks must contain finite values");
        }
    }
}

template <typename Scalar>
IterativeSolverReport solve(
    const CSRMatrix<Scalar, Device::CPU>& matrix,
    const DenseMatrix<Scalar, Device::CPU>& rhs,
    DenseMatrix<Scalar, Device::CPU>& solution,
    const IterativeSolverOptions& options,
    const DenseMatrix<Scalar, Device::CPU>* inverse_blocks,
    Index block_size)
{
    validateOptions(options);
    if (matrix.rows() <= 0 || matrix.rows() != matrix.cols()
        || rhs.rows() != matrix.rows() || rhs.cols() != 1
        || solution.rows() != matrix.rows() || solution.cols() != 1
        || rhs.data() == solution.data())
        throw std::invalid_argument("OpenCL PCG requires a non-empty square system and distinct vectors");
    matrix.validateStructure();
    for (Index row = 0; row < matrix.rows(); ++row)
        if (!std::isfinite(rhs.data()[row]) || !std::isfinite(solution.data()[row]))
            throw std::invalid_argument("OpenCL PCG vectors must contain finite values");
    if (inverse_blocks != nullptr)
    {
        validateBlockPreconditioner(
            matrix, rhs, solution, *inverse_blocks, block_size);
    }

    auto& runtime = OpenClRuntime::instance();
    requireFp64<Scalar>(runtime);
    CommandQueue queue(runtime.createQueue());
    const std::size_t count = static_cast<std::size_t>(matrix.rows());
    const std::size_t local_size = localSize(runtime);
    const std::size_t partial_count = std::max<std::size_t>(1, groupsFor(count, local_size));
    const bool double_accumulation = std::is_same_v<Scalar, double> || runtime.supportsFp64();
    const auto inverse_diagonal = inverse_blocks == nullptr
        ? inverseDiagonal(matrix, options.useJacobiPreconditioner)
        : std::vector<Scalar>{};
    const std::string build_options = std::is_same_v<Scalar, double>
        ? "-DREAL_DOUBLE=1" : (double_accumulation ? "-DACCUM_DOUBLE=1" : "");
    cl_program program = runtime.program("plamatrix_opencl_pcg_v2", kSolverSource, build_options);
    CompiledKernel spmv_kernel(program, "spmv");
    CompiledKernel initialize_kernel(program, "initialize");
    CompiledKernel initialize_residual_kernel(program, "initializeResidual");
    CompiledKernel update_kernel(program, "updateSolutionResidual");
    CompiledKernel precondition_kernel(program, "applyPreconditioner");
    CompiledKernel block_precondition_kernel(program, "applyBlockPreconditioner");
    CompiledKernel copy_kernel(program, "copyVector");
    CompiledKernel direction_kernel(program, "updateDirection");
    CompiledKernel dot_kernel(program, "dotPartial");
    CompiledKernel reduce_kernel(program, "reducePartial");

    DeviceBuffer rows(runtime.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                      byteSize<Index>(count + 1), const_cast<Index*>(matrix.rowOffsets()));
    DeviceBuffer columns(runtime.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                         byteSize<Index>(static_cast<std::size_t>(matrix.nnz())),
                         const_cast<Index*>(matrix.colIndices()));
    DeviceBuffer values(runtime.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        byteSize<Scalar>(static_cast<std::size_t>(matrix.nnz())),
                        const_cast<Scalar*>(matrix.values()));
    DeviceBuffer rhs_buffer(runtime.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            byteSize<Scalar>(count), const_cast<Scalar*>(rhs.data()));
    DeviceBuffer x(runtime.context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                   byteSize<Scalar>(count), solution.data());
    const Scalar* inverse_data = inverse_blocks != nullptr
        ? inverse_blocks->data() : inverse_diagonal.data();
    const std::size_t inverse_count = inverse_blocks != nullptr
        ? static_cast<std::size_t>(inverse_blocks->rows()) : count;
    DeviceBuffer inverse(runtime.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                         byteSize<Scalar>(inverse_count), const_cast<Scalar*>(inverse_data));
    DeviceBuffer residual(runtime.context(), CL_MEM_READ_WRITE, byteSize<Scalar>(count));
    DeviceBuffer transformed(runtime.context(), CL_MEM_READ_WRITE, byteSize<Scalar>(count));
    DeviceBuffer direction(runtime.context(), CL_MEM_READ_WRITE, byteSize<Scalar>(count));
    DeviceBuffer matrix_direction(runtime.context(), CL_MEM_READ_WRITE, byteSize<Scalar>(count));
    const std::size_t partial_bytes = double_accumulation
        ? byteSize<double>(partial_count) : byteSize<Scalar>(partial_count);
    DeviceBuffer partial_a(runtime.context(), CL_MEM_READ_WRITE, partial_bytes);
    DeviceBuffer partial_b(runtime.context(), CL_MEM_READ_WRITE, partial_bytes);
    const cl_long device_count = static_cast<cl_long>(matrix.rows());
    auto run_spmv = [&](const DeviceBuffer& input, DeviceBuffer& output)
    {
        kernelBufferArg(spmv_kernel, 0, rows); kernelBufferArg(spmv_kernel, 1, columns);
        kernelBufferArg(spmv_kernel, 2, values); kernelBufferArg(spmv_kernel, 3, input);
        kernelBufferArg(spmv_kernel, 4, output); kernelArg(spmv_kernel, 5, device_count);
        launch(queue.get(), spmv_kernel, count, local_size, "clEnqueueNDRangeKernel(spmv)");
    };
    auto run_preconditioner = [&]()
    {
        if (inverse_blocks != nullptr)
        {
            kernelBufferArg(block_precondition_kernel, 0, inverse);
            kernelBufferArg(block_precondition_kernel, 1, residual);
            kernelBufferArg(block_precondition_kernel, 2, transformed);
            kernelArg(block_precondition_kernel, 3, device_count);
            kernelArg(block_precondition_kernel, 4, static_cast<cl_long>(block_size));
            launch(queue.get(), block_precondition_kernel, count, local_size,
                   "clEnqueueNDRangeKernel(block precondition)");
        }
        else
        {
            kernelBufferArg(precondition_kernel, 0, inverse);
            kernelBufferArg(precondition_kernel, 1, residual);
            kernelBufferArg(precondition_kernel, 2, transformed);
            kernelArg(precondition_kernel, 3, device_count);
            launch(queue.get(), precondition_kernel, count, local_size,
                   "clEnqueueNDRangeKernel(precondition)");
        }
    };

    run_spmv(x, matrix_direction);
    if (inverse_blocks != nullptr)
    {
        kernelBufferArg(initialize_residual_kernel, 0, rhs_buffer);
        kernelBufferArg(initialize_residual_kernel, 1, matrix_direction);
        kernelBufferArg(initialize_residual_kernel, 2, residual);
        kernelArg(initialize_residual_kernel, 3, device_count);
        launch(queue.get(), initialize_residual_kernel, count, local_size,
               "clEnqueueNDRangeKernel(initialize residual)");
        run_preconditioner();
        kernelBufferArg(copy_kernel, 0, transformed);
        kernelBufferArg(copy_kernel, 1, direction);
        kernelArg(copy_kernel, 2, device_count);
        launch(queue.get(), copy_kernel, count, local_size,
               "clEnqueueNDRangeKernel(copy direction)");
    }
    else
    {
        kernelBufferArg(initialize_kernel, 0, rhs_buffer);
        kernelBufferArg(initialize_kernel, 1, matrix_direction);
        kernelBufferArg(initialize_kernel, 2, inverse);
        kernelBufferArg(initialize_kernel, 3, residual);
        kernelBufferArg(initialize_kernel, 4, transformed);
        kernelBufferArg(initialize_kernel, 5, direction);
        kernelArg(initialize_kernel, 6, device_count);
        launch(queue.get(), initialize_kernel, count, local_size,
               "clEnqueueNDRangeKernel(initialize)");
    }

    IterativeSolverReport report;
    double residual_squared = dot<Scalar>(queue.get(), dot_kernel, reduce_kernel, residual, residual,
        partial_a, partial_b, matrix.rows(), local_size, double_accumulation);
    if (!std::isfinite(residual_squared) || residual_squared < 0.0)
        throw std::runtime_error("OpenCL PCG initial residual is invalid");
    report.initialResidual = std::sqrt(residual_squared);
    report.finalResidual = report.initialResidual;
    const double tolerance = std::max(options.absoluteTolerance,
                                      options.relativeTolerance * report.initialResidual);
    report.converged = report.finalResidual <= tolerance;
    double rho = dot<Scalar>(queue.get(), dot_kernel, reduce_kernel, residual, transformed,
        partial_a, partial_b, matrix.rows(), local_size, double_accumulation);

    for (int iteration = 0; !report.converged && iteration < options.maxIterations; ++iteration)
    {
        if (!std::isfinite(rho) || rho <= 0.0)
            throw std::runtime_error("OpenCL PCG breakdown in preconditioned residual");
        run_spmv(direction, matrix_direction);
        const double denominator = dot<Scalar>(queue.get(), dot_kernel, reduce_kernel,
            direction, matrix_direction, partial_a, partial_b, matrix.rows(), local_size,
            double_accumulation);
        if (!std::isfinite(denominator) || denominator <= 0.0)
            throw std::runtime_error("OpenCL PCG breakdown in matrix-direction product");
        const Scalar alpha = static_cast<Scalar>(rho / denominator);
        if (!std::isfinite(alpha))
            throw std::runtime_error("OpenCL PCG breakdown: solution scale is not finite");
        kernelBufferArg(update_kernel, 0, x); kernelBufferArg(update_kernel, 1, residual);
        kernelBufferArg(update_kernel, 2, direction); kernelBufferArg(update_kernel, 3, matrix_direction);
        kernelArg(update_kernel, 4, alpha); kernelArg(update_kernel, 5, device_count);
        launch(queue.get(), update_kernel, count, local_size, "clEnqueueNDRangeKernel(update)");
        residual_squared = dot<Scalar>(queue.get(), dot_kernel, reduce_kernel, residual, residual,
            partial_a, partial_b, matrix.rows(), local_size, double_accumulation);
        if (!std::isfinite(residual_squared) || residual_squared < 0.0)
            throw std::runtime_error("OpenCL PCG residual is invalid");
        report.iterations = iteration + 1;
        report.finalResidual = std::sqrt(residual_squared);
        report.converged = std::isfinite(report.finalResidual) && report.finalResidual <= tolerance;
        if (report.converged) break;
        run_preconditioner();
        const double next_rho = dot<Scalar>(queue.get(), dot_kernel, reduce_kernel, residual, transformed,
            partial_a, partial_b, matrix.rows(), local_size, double_accumulation);
        if (!std::isfinite(next_rho) || next_rho <= 0.0)
            throw std::runtime_error("OpenCL PCG breakdown in direction update");
        const Scalar beta = static_cast<Scalar>(next_rho / rho);
        if (!std::isfinite(beta))
            throw std::runtime_error("OpenCL PCG breakdown: direction scale is not finite");
        kernelBufferArg(direction_kernel, 0, direction); kernelBufferArg(direction_kernel, 1, transformed);
        kernelArg(direction_kernel, 2, beta); kernelArg(direction_kernel, 3, device_count);
        launch(queue.get(), direction_kernel, count, local_size, "clEnqueueNDRangeKernel(direction)");
        rho = next_rho;
    }
    checkOpenCl(clEnqueueReadBuffer(queue.get(), x.get(), CL_TRUE, 0, byteSize<Scalar>(count),
                                    solution.data(), 0, nullptr, nullptr), "clEnqueueReadBuffer(solution)");
    if (!report.converged && options.requireConvergence)
    {
        std::ostringstream message;
        message << "OpenCL PCG did not converge in " << report.iterations
                << " iterations; final residual=" << report.finalResidual;
        throw std::runtime_error(message.str());
    }
    return report;
}

} // namespace

template <typename Scalar>
IterativeSolverReport pcg(
    const CSRMatrix<Scalar, Device::CPU>& matrix,
    const DenseMatrix<Scalar, Device::CPU>& rhs,
    DenseMatrix<Scalar, Device::CPU>& solution,
    const IterativeSolverOptions& options)
{
    return solve<Scalar>(matrix, rhs, solution, options, nullptr, 0);
}

template <typename Scalar>
IterativeSolverReport blockPcg(
    const CSRMatrix<Scalar, Device::CPU>& matrix,
    const DenseMatrix<Scalar, Device::CPU>& rhs,
    DenseMatrix<Scalar, Device::CPU>& solution,
    const DenseMatrix<Scalar, Device::CPU>& inverse_blocks,
    Index block_size,
    const IterativeSolverOptions& options)
{
    return solve(
        matrix, rhs, solution, options, &inverse_blocks, block_size);
}

#ifdef PLAMATRIX_USE_FLOAT
template IterativeSolverReport pcg<float>(
    const CSRMatrix<float, Device::CPU>&, const DenseMatrix<float, Device::CPU>&,
    DenseMatrix<float, Device::CPU>&, const IterativeSolverOptions&);
template IterativeSolverReport blockPcg<float>(
    const CSRMatrix<float, Device::CPU>&, const DenseMatrix<float, Device::CPU>&,
    DenseMatrix<float, Device::CPU>&, const DenseMatrix<float, Device::CPU>&,
    Index, const IterativeSolverOptions&);
#endif
#ifdef PLAMATRIX_USE_DOUBLE
template IterativeSolverReport pcg<double>(
    const CSRMatrix<double, Device::CPU>&, const DenseMatrix<double, Device::CPU>&,
    DenseMatrix<double, Device::CPU>&, const IterativeSolverOptions&);
template IterativeSolverReport blockPcg<double>(
    const CSRMatrix<double, Device::CPU>&, const DenseMatrix<double, Device::CPU>&,
    DenseMatrix<double, Device::CPU>&, const DenseMatrix<double, Device::CPU>&,
    Index, const IterativeSolverOptions&);
#endif

} // namespace opencl
} // namespace plamatrix
