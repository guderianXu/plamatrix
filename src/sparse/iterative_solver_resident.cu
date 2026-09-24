#include "plamatrix/internal/sparse/iterative_solver.h"

#include <cmath>
#include <limits>
#include <memory>

#include "plamatrix/internal/core/allocator.h"
#include "plamatrix/internal/core/error_check.h"

namespace plamatrix::internal
{

namespace
{

constexpr int kResidentBlockSize = 256;

template <typename Scalar>
__global__ void residentSpmv(Index rows,
                             const Index* row_offsets,
                             const Index* columns,
                             const Scalar* values,
                             const Scalar* x,
                             Scalar* y)
{
    const Index row = static_cast<Index>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows)
    {
        return;
    }
    Scalar sum = Scalar(0);
    for (Index position = row_offsets[row]; position < row_offsets[row + 1]; ++position)
    {
        sum += values[position] * x[columns[position]];
    }
    y[row] = sum;
}

template <typename Scalar>
__global__ void residentResidual(Index size, const Scalar* rhs, const Scalar* ax, Scalar* residual)
{
    const Index index = static_cast<Index>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < size)
    {
        residual[index] = rhs[index] - ax[index];
    }
}

template <typename Scalar>
__global__ void residentUpdate(Index size, Scalar alpha, const Scalar* direction,
                               const Scalar* matrix_direction, Scalar* solution, Scalar* residual)
{
    const Index index = static_cast<Index>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < size)
    {
        solution[index] += alpha * direction[index];
        residual[index] -= alpha * matrix_direction[index];
    }
}

template <typename Scalar>
__global__ void residentJacobi(Index rows,
                               const Index* row_offsets,
                               const Index* columns,
                               const Scalar* values,
                               const Scalar* residual,
                               Scalar* transformed)
{
    const Index row = static_cast<Index>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows)
    {
        return;
    }
    Scalar diagonal = Scalar(0);
    for (Index position = row_offsets[row]; position < row_offsets[row + 1]; ++position)
    {
        if (columns[position] == row)
        {
            diagonal += values[position];
        }
    }
    transformed[row] = diagonal != Scalar(0) ? residual[row] / diagonal : Scalar(0);
}

template <typename Scalar>
__global__ void residentDirection(Index size, Scalar beta, const Scalar* transformed, Scalar* direction)
{
    const Index index = static_cast<Index>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < size)
    {
        direction[index] = transformed[index] + beta * direction[index];
    }
}

template <typename Scalar> struct CublasTraits;

template <> struct CublasTraits<float>
{
    static void copy(cublasHandle_t handle, int size, const float* source, float* destination)
    {
        PLAMATRIX_CHECK_CUBLAS(cublasScopy(handle, size, source, 1, destination, 1));
    }

    static float dot(cublasHandle_t handle, int size, const float* left, const float* right)
    {
        float result = 0.0F;
        PLAMATRIX_CHECK_CUBLAS(cublasSdot(handle, size, left, 1, right, 1, &result));
        return result;
    }
};

template <> struct CublasTraits<double>
{
    static void copy(cublasHandle_t handle, int size, const double* source, double* destination)
    {
        PLAMATRIX_CHECK_CUBLAS(cublasDcopy(handle, size, source, 1, destination, 1));
    }

    static double dot(cublasHandle_t handle, int size, const double* left, const double* right)
    {
        double result = 0.0;
        PLAMATRIX_CHECK_CUBLAS(cublasDdot(handle, size, left, 1, right, 1, &result));
        return result;
    }
};

template <typename Scalar> class DeviceBuffer
{
public:
    explicit DeviceBuffer(std::size_t count)
        : _count(count)
        , _data(GpuAllocator<Scalar>::allocate(count))
    {
    }

    ~DeviceBuffer() noexcept
    {
        GpuAllocator<Scalar>::deallocateNoThrow(_data, _count);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    Scalar* data() noexcept { return _data; }

private:
    std::size_t _count;
    Scalar* _data;
};

class CublasHandleGuard
{
public:
    CublasHandleGuard()
    {
        PLAMATRIX_CHECK_CUBLAS(cublasCreate(&_handle));
        PLAMATRIX_CHECK_CUBLAS(cublasSetPointerMode(_handle, CUBLAS_POINTER_MODE_HOST));
    }

    ~CublasHandleGuard() noexcept
    {
        if (_handle != nullptr)
        {
            static_cast<void>(cublasDestroy(_handle));
        }
    }

    CublasHandleGuard(const CublasHandleGuard&) = delete;
    CublasHandleGuard& operator=(const CublasHandleGuard&) = delete;

    cublasHandle_t get() const noexcept { return _handle; }

private:
    cublasHandle_t _handle = nullptr;
};

inline int residentGrid(Index size)
{
    return static_cast<int>((size + kResidentBlockSize - 1) / kResidentBlockSize);
}

template <typename Scalar>
IterativeSolverReport solveResidentPcgCuda(const ResidentCsrMatrix<Scalar>& matrix,
                                            const ResidentVector<Scalar>& rhs,
                                            ResidentVector<Scalar>& solution,
                                            const SolveOptions& options,
                                            ExecutionContext& context)
{
    matrix.validateContext(context);
    rhs.validateContext(context);
    solution.validateContext(context);
    if (context.backend() != Backend::Cuda)
    {
        throw Error(ErrorCode::InvalidState, "CUDA resident PCG requires a CUDA context", context.backend());
    }
    if (matrix.rows() != matrix.cols() || rhs.size() != matrix.rows() || solution.size() != matrix.cols())
    {
        throw Error(ErrorCode::InvalidArgument,
                    "Resident PCG requires a square CSR matrix and matching vectors",
                    context.backend());
    }
    if (options.maxIterations < 0 || !std::isfinite(options.relativeTolerance) ||
        options.relativeTolerance < 0.0 || options.relativeTolerance > 1.0 ||
        !std::isfinite(options.absoluteTolerance) || options.absoluteTolerance < 0.0)
    {
        throw Error(ErrorCode::InvalidArgument, "Resident PCG options are invalid", context.backend());
    }
    if (matrix.rows() > static_cast<Index>(std::numeric_limits<int>::max()))
    {
        throw Error(ErrorCode::InvalidArgument, "Resident PCG size exceeds cuBLAS integer range", context.backend());
    }

    try
    {
        const Index size = matrix.rows();
        IterativeSolverReport report;
        if (size == 0)
        {
            report.converged = true;
            report.diagnostics.deterministic = options.deterministic;
            return report;
        }

        DeviceBuffer<Scalar> residual(static_cast<std::size_t>(size));
        DeviceBuffer<Scalar> direction(static_cast<std::size_t>(size));
        DeviceBuffer<Scalar> transformed(static_cast<std::size_t>(size));
        DeviceBuffer<Scalar> matrix_direction(static_cast<std::size_t>(size));
        CublasHandleGuard handle_guard;
        const auto handle = handle_guard.get();

        const int grid = residentGrid(size);
        residentSpmv<<<grid, kResidentBlockSize>>>(size,
                                                    matrix.rowOffsets(),
                                                    matrix.colIndices(),
                                                    matrix.values(),
                                                    solution.data(),
                                                    matrix_direction.data());
        PLAMATRIX_CHECK_CUDA(cudaGetLastError());
        residentResidual<<<grid, kResidentBlockSize>>>(size, rhs.data(), matrix_direction.data(), residual.data());
        PLAMATRIX_CHECK_CUDA(cudaGetLastError());
        const int blas_size = static_cast<int>(size);
        const double initial_squared = static_cast<double>(CublasTraits<Scalar>::dot(
            handle, blas_size, residual.data(), residual.data()));
        report.initialResidual = std::sqrt(initial_squared);
        report.finalResidual = report.initialResidual;
        if (options.recordResidualHistory)
        {
            report.residualHistory.push_back(report.initialResidual);
        }
        const double tolerance = std::max(options.absoluteTolerance,
                                          options.relativeTolerance * report.initialResidual);
        if (report.finalResidual <= tolerance)
        {
            report.converged = true;
            return report;
        }

        if (options.useJacobiPreconditioner)
        {
            residentJacobi<<<grid, kResidentBlockSize>>>(size,
                                                          matrix.rowOffsets(),
                                                          matrix.colIndices(),
                                                          matrix.values(),
                                                          residual.data(),
                                                          transformed.data());
        }
        else
        {
            CublasTraits<Scalar>::copy(handle, blas_size, residual.data(), transformed.data());
        }
        PLAMATRIX_CHECK_CUDA(cudaGetLastError());
        CublasTraits<Scalar>::copy(handle, blas_size, transformed.data(), direction.data());
        double rho = static_cast<double>(CublasTraits<Scalar>::dot(
            handle, blas_size, residual.data(), transformed.data()));
        if (!std::isfinite(rho) || rho <= 0.0)
        {
            throw Error(ErrorCode::NumericalFailure,
                        "Resident PCG preconditioned residual is not positive",
                        context.backend());
        }

        for (int iteration = 0; iteration < options.maxIterations; ++iteration)
        {
            if (options.cancellation != nullptr && options.cancellation->isCancellationRequested())
            {
                throw Error(ErrorCode::InvalidState, "Resident PCG was cancelled", context.backend());
            }
            residentSpmv<<<grid, kResidentBlockSize>>>(size,
                                                        matrix.rowOffsets(),
                                                        matrix.colIndices(),
                                                        matrix.values(),
                                                        direction.data(),
                                                        matrix_direction.data());
            PLAMATRIX_CHECK_CUDA(cudaGetLastError());
            const double denominator = static_cast<double>(CublasTraits<Scalar>::dot(
                handle, blas_size, direction.data(), matrix_direction.data()));
            if (!std::isfinite(denominator) || denominator <= 0.0)
            {
                throw Error(ErrorCode::NumericalFailure,
                            "Resident PCG matrix is not numerically SPD",
                            context.backend());
            }
            const Scalar alpha = static_cast<Scalar>(rho / denominator);
            residentUpdate<<<grid, kResidentBlockSize>>>(size,
                                                          alpha,
                                                          direction.data(),
                                                          matrix_direction.data(),
                                                          solution.data(),
                                                          residual.data());
            PLAMATRIX_CHECK_CUDA(cudaGetLastError());
            report.iterations = iteration + 1;
            const double residual_squared = static_cast<double>(CublasTraits<Scalar>::dot(
                handle, blas_size, residual.data(), residual.data()));
            report.finalResidual = std::sqrt(residual_squared);
            if (options.recordResidualHistory)
            {
                report.residualHistory.push_back(report.finalResidual);
            }
            ++report.diagnostics.commandSubmissions;
            if (report.finalResidual <= tolerance)
            {
                report.converged = true;
                break;
            }
            if (options.useJacobiPreconditioner)
            {
                residentJacobi<<<grid, kResidentBlockSize>>>(size,
                                                              matrix.rowOffsets(),
                                                              matrix.colIndices(),
                                                              matrix.values(),
                                                              residual.data(),
                                                              transformed.data());
                PLAMATRIX_CHECK_CUDA(cudaGetLastError());
            }
            else
            {
                CublasTraits<Scalar>::copy(handle, blas_size, residual.data(), transformed.data());
            }
            const double next_rho = static_cast<double>(CublasTraits<Scalar>::dot(
                handle, blas_size, residual.data(), transformed.data()));
            const Scalar beta = static_cast<Scalar>(next_rho / rho);
            residentDirection<<<grid, kResidentBlockSize>>>(size, beta, transformed.data(), direction.data());
            PLAMATRIX_CHECK_CUDA(cudaGetLastError());
            rho = next_rho;
        }
        report.diagnostics.deterministic = options.deterministic;
        if (!report.converged && options.requireConvergence)
        {
            throw Error(ErrorCode::NumericalFailure, "Resident PCG did not converge", context.backend());
        }
        return report;
    }
    catch (const Error&)
    {
        throw;
    }
    catch (const std::exception& error)
    {
        throw Error(ErrorCode::BackendFailure, error.what(), context.backend());
    }
}

} // namespace

namespace resident_solver_detail
{

template <typename Scalar>
IterativeSolverReport pcgCudaResident(const ResidentCsrMatrix<Scalar>& matrix,
                                      const ResidentVector<Scalar>& rhs,
                                      ResidentVector<Scalar>& solution,
                                      const SolveOptions& options,
                                      ExecutionContext& context)
{
    return solveResidentPcgCuda(matrix, rhs, solution, options, context);
}

#ifdef PLAMATRIX_USE_FLOAT
template IterativeSolverReport pcgCudaResident<float>(const ResidentCsrMatrix<float>&,
                                                       const ResidentVector<float>&,
                                                       ResidentVector<float>&,
                                                       const SolveOptions&,
                                                       ExecutionContext&);
#endif

#ifdef PLAMATRIX_USE_DOUBLE
template IterativeSolverReport pcgCudaResident<double>(const ResidentCsrMatrix<double>&,
                                                        const ResidentVector<double>&,
                                                        ResidentVector<double>&,
                                                        const SolveOptions&,
                                                        ExecutionContext&);
#endif

} // namespace resident_solver_detail

} // namespace plamatrix::internal
