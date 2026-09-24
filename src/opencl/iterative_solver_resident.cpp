#include "plamatrix/internal/sparse/iterative_solver.h"

#include "iterative_solver_kernels.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "plamatrix/internal/opencl/execution.h"
#include "plamatrix/internal/opencl/native.h"

namespace plamatrix::internal::resident_solver_detail
{

namespace
{

void checked(cl_int status, const char* operation)
{
    if (status != CL_SUCCESS)
    {
        throw Error(ErrorCode::BackendFailure,
                    std::string("OpenCL resident PCG ") + operation + " failed",
                    Backend::OpenCl,
                    status);
    }
}

class Program
{
public:
    Program(ExecutionContext& context, const char* options)
    {
        const char* source = opencl::iterative_solver_detail::kSolverSource;
        cl_int status = CL_SUCCESS;
        _program = clCreateProgramWithSource(opencl::NativeAccess::context(context), 1, &source, nullptr, &status);
        checked(status, "clCreateProgramWithSource");
        const cl_device_id device = opencl::NativeAccess::device(context);
        status = clBuildProgram(_program, 1, &device, options, nullptr, nullptr);
        if (status != CL_SUCCESS)
        {
            std::size_t size = 0;
            static_cast<void>(clGetProgramBuildInfo(_program,
                                                     opencl::NativeAccess::device(context),
                                                     CL_PROGRAM_BUILD_LOG,
                                                     0,
                                                     nullptr,
                                                     &size));
            std::string log(size, '\0');
            if (size != 0)
            {
                static_cast<void>(clGetProgramBuildInfo(_program,
                                                         opencl::NativeAccess::device(context),
                                                         CL_PROGRAM_BUILD_LOG,
                                                         size,
                                                         log.data(),
                                                         nullptr));
            }
            throw Error(ErrorCode::BackendFailure,
                        "OpenCL resident PCG program build failed: " + log,
                        Backend::OpenCl,
                        status);
        }
    }

    ~Program() noexcept
    {
        if (_program != nullptr)
        {
            static_cast<void>(clReleaseProgram(_program));
        }
    }

    Program(const Program&) = delete;
    Program& operator=(const Program&) = delete;
    cl_program get() const noexcept { return _program; }

private:
    cl_program _program = nullptr;
};

void setBuffer(cl_kernel kernel, cl_uint index, cl_mem buffer)
{
    checked(clSetKernelArg(kernel, index, sizeof(buffer), &buffer), "clSetKernelArg(buffer)");
}

template <typename Value> void setValue(cl_kernel kernel, cl_uint index, const Value& value)
{
    checked(clSetKernelArg(kernel, index, sizeof(value), &value), "clSetKernelArg(value)");
}

void launch(cl_command_queue queue, cl_kernel kernel, std::size_t count, std::size_t local_size)
{
    const std::size_t global_size = ((count + local_size - 1) / local_size) * local_size;
    checked(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, &local_size, 0, nullptr, nullptr),
            "clEnqueueNDRangeKernel");
}

std::size_t localSize(cl_device_id device)
{
    std::size_t maximum = 1;
    checked(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maximum), &maximum, nullptr),
            "clGetDeviceInfo(max work-group size)");
    std::size_t result = 1;
    while (result * 2 <= std::min<std::size_t>(maximum, 256))
    {
        result *= 2;
    }
    return result;
}

std::size_t groupsFor(std::size_t size, std::size_t local_size)
{
    return (size + local_size * 2 - 1) / (local_size * 2);
}

template <typename Scalar>
double dot(cl_command_queue queue,
           cl_kernel dot_kernel,
           cl_kernel reduce_kernel,
           cl_mem first,
           cl_mem second,
           cl_mem partial_a,
           cl_mem partial_b,
           Index size,
           std::size_t local_size,
           bool double_accumulation)
{
    const std::size_t element_bytes = double_accumulation ? sizeof(double) : sizeof(Scalar);
    std::size_t remaining = groupsFor(static_cast<std::size_t>(size), local_size);
    setBuffer(dot_kernel, 0, first);
    setBuffer(dot_kernel, 1, second);
    setBuffer(dot_kernel, 2, partial_a);
    setValue(dot_kernel, 3, static_cast<cl_long>(size));
    checked(clSetKernelArg(dot_kernel, 4, local_size * element_bytes, nullptr), "clSetKernelArg(dot scratch)");
    launch(queue, dot_kernel, remaining * local_size, local_size);

    cl_mem current = partial_a;
    cl_mem next = partial_b;
    while (remaining > 1)
    {
        const std::size_t next_groups = groupsFor(remaining, local_size);
        setBuffer(reduce_kernel, 0, current);
        setBuffer(reduce_kernel, 1, next);
        setValue(reduce_kernel, 2, static_cast<cl_long>(remaining));
        checked(clSetKernelArg(reduce_kernel, 3, local_size * element_bytes, nullptr),
                "clSetKernelArg(reduce scratch)");
        launch(queue, reduce_kernel, next_groups * local_size, local_size);
        remaining = next_groups;
        std::swap(current, next);
    }
    if (double_accumulation)
    {
        double result = 0.0;
        checked(clEnqueueReadBuffer(queue, current, CL_TRUE, 0, sizeof(result), &result, 0, nullptr, nullptr),
                "clEnqueueReadBuffer(dot)");
        return result;
    }
    Scalar result{};
    checked(clEnqueueReadBuffer(queue, current, CL_TRUE, 0, sizeof(result), &result, 0, nullptr, nullptr),
            "clEnqueueReadBuffer(dot)");
    return static_cast<double>(result);
}

} // namespace

template <typename Scalar>
IterativeSolverReport pcgOpenClResident(const ResidentCsrMatrix<Scalar>& matrix,
                                        const ResidentVector<Scalar>& rhs,
                                        ResidentVector<Scalar>& solution,
                                        const SolveOptions& options,
                                        ExecutionContext& context)
{
    matrix.validateContext(context);
    rhs.validateContext(context);
    solution.validateContext(context);
    if (context.backend() != Backend::OpenCl)
    {
        throw Error(ErrorCode::InvalidArgument, "OpenCL resident PCG requires an OpenCL context", context.backend());
    }
    if (matrix.rows() != matrix.cols() || rhs.size() != matrix.rows() || solution.size() != matrix.cols())
    {
        throw Error(ErrorCode::InvalidArgument,
                    "Resident PCG requires a square CSR matrix and matching vectors",
                    Backend::OpenCl);
    }
    if (options.maxIterations < 0 || !std::isfinite(options.relativeTolerance) ||
        options.relativeTolerance < 0.0 || options.relativeTolerance > 1.0 ||
        !std::isfinite(options.absoluteTolerance) || options.absoluteTolerance < 0.0)
    {
        throw Error(ErrorCode::InvalidArgument, "Resident PCG options are invalid", Backend::OpenCl);
    }
    if constexpr (std::is_same_v<Scalar, double>)
    {
        if (!context.capabilities().float64)
        {
            throw Error(ErrorCode::UnsupportedOperation,
                        "Selected OpenCL device does not support float64",
                        Backend::OpenCl);
        }
    }

    IterativeSolverReport report;
    const auto count = static_cast<std::size_t>(matrix.rows());
    if (count == 0)
    {
        report.converged = true;
        return report;
    }
    const cl_context native_context = opencl::NativeAccess::context(context);
    const cl_command_queue queue = opencl::NativeAccess::queue(context);
    const cl_device_id device = opencl::NativeAccess::device(context);
    const std::size_t local_size = localSize(device);
    const std::size_t partial_count = std::max<std::size_t>(1, groupsFor(count, local_size));
    const bool double_accumulation = std::is_same_v<Scalar, double> || context.capabilities().float64;
    const char* build_options = std::is_same_v<Scalar, double> ? "-DREAL_DOUBLE=1" :
                                double_accumulation ? "-DACCUM_DOUBLE=1" : "";
    Program program(context, build_options);
    opencl::detail::CompiledKernel spmv(program.get(), "spmv");
    opencl::detail::CompiledKernel initialize(program.get(), "initialize");
    opencl::detail::CompiledKernel build_jacobi(program.get(), "buildJacobi");
    opencl::detail::CompiledKernel update(program.get(), "updateSolutionResidual");
    opencl::detail::CompiledKernel precondition(program.get(), "applyPreconditioner");
    opencl::detail::CompiledKernel direction_update(program.get(), "updateDirection");
    opencl::detail::CompiledKernel dot_kernel(program.get(), "dotPartial");
    opencl::detail::CompiledKernel reduce_kernel(program.get(), "reducePartial");

    const auto scalar_bytes = opencl::detail::byteSize<Scalar>(count);
    const auto partial_bytes = partial_count * (double_accumulation ? sizeof(double) : sizeof(Scalar));
    opencl::detail::DeviceBuffer inverse(native_context, CL_MEM_READ_WRITE, scalar_bytes);
    opencl::detail::DeviceBuffer residual(native_context, CL_MEM_READ_WRITE, scalar_bytes);
    opencl::detail::DeviceBuffer transformed(native_context, CL_MEM_READ_WRITE, scalar_bytes);
    opencl::detail::DeviceBuffer direction(native_context, CL_MEM_READ_WRITE, scalar_bytes);
    opencl::detail::DeviceBuffer matrix_direction(native_context, CL_MEM_READ_WRITE, scalar_bytes);
    opencl::detail::DeviceBuffer partial_a(native_context, CL_MEM_READ_WRITE, partial_bytes);
    opencl::detail::DeviceBuffer partial_b(native_context, CL_MEM_READ_WRITE, partial_bytes);
    const cl_mem rows = opencl::NativeAccess::rows(matrix);
    const cl_mem columns = opencl::NativeAccess::columns(matrix);
    const cl_mem values = opencl::NativeAccess::values(matrix);
    const cl_mem rhs_buffer = opencl::NativeAccess::buffer(rhs);
    const cl_mem x = opencl::NativeAccess::buffer(solution);
    const cl_long size = static_cast<cl_long>(matrix.rows());

    const auto run_spmv = [&](cl_mem input, cl_mem output)
    {
        setBuffer(spmv, 0, rows);
        setBuffer(spmv, 1, columns);
        setBuffer(spmv, 2, values);
        setBuffer(spmv, 3, input);
        setBuffer(spmv, 4, output);
        setValue(spmv, 5, size);
        launch(queue, spmv, count, local_size);
    };
    const auto dot_vectors = [&](cl_mem first, cl_mem second)
    {
        return dot<Scalar>(queue,
                           dot_kernel,
                           reduce_kernel,
                           first,
                           second,
                           partial_a.get(),
                           partial_b.get(),
                           matrix.rows(),
                           local_size,
                           double_accumulation);
    };
    const auto apply_preconditioner = [&]()
    {
        setBuffer(precondition, 0, inverse.get());
        setBuffer(precondition, 1, residual.get());
        setBuffer(precondition, 2, transformed.get());
        setValue(precondition, 3, size);
        launch(queue, precondition, count, local_size);
    };

    if (options.useJacobiPreconditioner)
    {
        setBuffer(build_jacobi, 0, rows);
        setBuffer(build_jacobi, 1, columns);
        setBuffer(build_jacobi, 2, values);
        setBuffer(build_jacobi, 3, inverse.get());
        setValue(build_jacobi, 4, size);
        launch(queue, build_jacobi, count, local_size);
    }
    else
    {
        std::vector<Scalar> ones(count, Scalar(1));
        checked(clEnqueueWriteBuffer(queue,
                                     inverse.get(),
                                     CL_TRUE,
                                     0,
                                     scalar_bytes,
                                     ones.data(),
                                     0,
                                     nullptr,
                                     nullptr),
                "clEnqueueWriteBuffer(identity preconditioner)");
    }

    run_spmv(x, matrix_direction.get());
    setBuffer(initialize, 0, rhs_buffer);
    setBuffer(initialize, 1, matrix_direction.get());
    setBuffer(initialize, 2, inverse.get());
    setBuffer(initialize, 3, residual.get());
    setBuffer(initialize, 4, transformed.get());
    setBuffer(initialize, 5, direction.get());
    setValue(initialize, 6, size);
    launch(queue, initialize, count, local_size);

    const double initial_squared = dot_vectors(residual.get(), residual.get());
    if (!std::isfinite(initial_squared) || initial_squared < 0.0)
    {
        throw Error(ErrorCode::NumericalFailure, "OpenCL resident PCG initial residual is invalid", Backend::OpenCl);
    }
    report.initialResidual = std::sqrt(initial_squared);
    report.finalResidual = report.initialResidual;
    report.converged = report.finalResidual <= std::max(options.absoluteTolerance,
                                                        options.relativeTolerance * report.initialResidual);
    if (options.recordResidualHistory)
    {
        report.residualHistory.push_back(report.initialResidual);
    }
    const double tolerance = std::max(options.absoluteTolerance,
                                      options.relativeTolerance * report.initialResidual);
    double rho = report.converged ? 0.0 : dot_vectors(residual.get(), transformed.get());
    for (int iteration = 0; !report.converged && iteration < options.maxIterations; ++iteration)
    {
        if (options.cancellation != nullptr && options.cancellation->isCancellationRequested())
        {
            throw Error(ErrorCode::InvalidState, "Resident PCG was cancelled", Backend::OpenCl);
        }
        if (!std::isfinite(rho) || rho <= 0.0)
        {
            throw Error(ErrorCode::NumericalFailure,
                        "OpenCL resident PCG preconditioned residual is invalid",
                        Backend::OpenCl);
        }
        run_spmv(direction.get(), matrix_direction.get());
        const double denominator = dot_vectors(direction.get(), matrix_direction.get());
        if (!std::isfinite(denominator) || denominator <= 0.0)
        {
            throw Error(ErrorCode::NumericalFailure,
                        "OpenCL resident PCG matrix-direction product is invalid",
                        Backend::OpenCl);
        }
        const Scalar alpha = static_cast<Scalar>(rho / denominator);
        setBuffer(update, 0, x);
        setBuffer(update, 1, residual.get());
        setBuffer(update, 2, direction.get());
        setBuffer(update, 3, matrix_direction.get());
        setValue(update, 4, alpha);
        setValue(update, 5, size);
        launch(queue, update, count, local_size);
        report.iterations = iteration + 1;
        const double squared = dot_vectors(residual.get(), residual.get());
        if (!std::isfinite(squared) || squared < 0.0)
        {
            throw Error(ErrorCode::NumericalFailure,
                        "OpenCL resident PCG residual is invalid",
                        Backend::OpenCl);
        }
        report.finalResidual = std::sqrt(squared);
        if (options.recordResidualHistory)
        {
            report.residualHistory.push_back(report.finalResidual);
        }
        report.converged = report.finalResidual <= tolerance;
        ++report.diagnostics.commandSubmissions;
        if (report.converged)
        {
            break;
        }
        apply_preconditioner();
        const double next_rho = dot_vectors(residual.get(), transformed.get());
        if (!std::isfinite(next_rho) || next_rho <= 0.0)
        {
            throw Error(ErrorCode::NumericalFailure,
                        "OpenCL resident PCG direction update failed",
                        Backend::OpenCl);
        }
        const Scalar beta = static_cast<Scalar>(next_rho / rho);
        setBuffer(direction_update, 0, direction.get());
        setBuffer(direction_update, 1, transformed.get());
        setValue(direction_update, 2, beta);
        setValue(direction_update, 3, size);
        launch(queue, direction_update, count, local_size);
        rho = next_rho;
    }
    report.diagnostics.deterministic = options.deterministic;
    if (!report.converged && options.requireConvergence)
    {
        throw Error(ErrorCode::NumericalFailure, "OpenCL resident PCG did not converge", Backend::OpenCl);
    }
    return report;
}

#ifdef PLAMATRIX_USE_FLOAT
template IterativeSolverReport pcgOpenClResident<float>(const ResidentCsrMatrix<float>&,
                                                         const ResidentVector<float>&,
                                                         ResidentVector<float>&,
                                                         const SolveOptions&,
                                                         ExecutionContext&);
#endif
#ifdef PLAMATRIX_USE_DOUBLE
template IterativeSolverReport pcgOpenClResident<double>(const ResidentCsrMatrix<double>&,
                                                          const ResidentVector<double>&,
                                                          ResidentVector<double>&,
                                                          const SolveOptions&,
                                                          ExecutionContext&);
#endif

} // namespace plamatrix::internal::resident_solver_detail
