#include "benchmark/benchmark_cases.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "plamatrix/core/error_check.h"
#include "plamatrix/dense/dense_matrix.h"
#include "plamatrix/sparse/iterative_solver.h"
#include "plamatrix/sparse/sparse_ops.h"

namespace plamatrix
{
namespace detail
{
namespace
{

using Clock = std::chrono::high_resolution_clock;

class CudaEvent
{
public:
    CudaEvent()
    {
        PLAMATRIX_CHECK_CUDA(cudaEventCreate(&_event));
    }

    ~CudaEvent()
    {
        if (_event != nullptr)
        {
            static_cast<void>(cudaEventDestroy(_event));
        }
    }

    cudaEvent_t get() const noexcept { return _event; }

private:
    cudaEvent_t _event = nullptr;
};

struct CpuSparseFixture
{
    std::vector<Index> rows;
    std::vector<Index> columns;
    std::vector<float> values;
    CSRMatrix<float, Device::CPU> matrix;
    DenseMatrix<float, Device::CPU> vector;
    DenseMatrix<float, Device::CPU> block;
    DenseMatrix<float, Device::CPU> rhs;

    explicit CpuSparseFixture(Index size)
        : matrix(makeMatrix(size, rows, columns, values))
        , vector(DenseMatrix<float, Device::CPU>::pinned(size, 1))
        , block(DenseMatrix<float, Device::CPU>::pinned(size, 8))
        , rhs(DenseMatrix<float, Device::CPU>::pinned(size, 1))
    {
        for (Index row = 0; row < size; ++row)
        {
            vector(row, 0) = 1.0f + static_cast<float>(row % 7) * 0.01f;
            rhs(row, 0) = 1.0f;
            for (Index column = 0; column < block.cols(); ++column)
            {
                block(row, column) = static_cast<float>((row + column) % 11) * 0.02f;
            }
        }
    }

private:
    static CSRMatrix<float, Device::CPU> makeMatrix(
        Index size,
        std::vector<Index>& rows,
        std::vector<Index>& columns,
        std::vector<float>& values)
    {
        rows.reserve(static_cast<std::size_t>(size) * 3);
        columns.reserve(static_cast<std::size_t>(size) * 3);
        values.reserve(static_cast<std::size_t>(size) * 3);
        for (Index row = 0; row < size; ++row)
        {
            if (row > 0)
            {
                rows.push_back(row);
                columns.push_back(row - 1);
                values.push_back(-1.0f);
            }
            rows.push_back(row);
            columns.push_back(row);
            values.push_back(4.0f);
            if (row + 1 < size)
            {
                rows.push_back(row);
                columns.push_back(row + 1);
                values.push_back(-1.0f);
            }
        }
        return cooToCsr(size, size, rows, columns, values);
    }
};

double median(std::vector<double>& samples)
{
    std::sort(samples.begin(), samples.end());
    const std::size_t middle = samples.size() / 2;
    return (samples.size() % 2) == 0
        ? (samples[middle - 1] + samples[middle]) * 0.5
        : samples[middle];
}

template <typename Function>
double measureWall(Function&& function, int warmup = 2, int trials = 5)
{
    for (int iteration = 0; iteration < warmup; ++iteration)
    {
        function();
        PLAMATRIX_CHECK_CUDA(cudaDeviceSynchronize());
    }
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(trials));
    for (int iteration = 0; iteration < trials; ++iteration)
    {
        const auto start = Clock::now();
        function();
        PLAMATRIX_CHECK_CUDA(cudaDeviceSynchronize());
        const auto stop = Clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
    }
    return median(samples);
}

template <typename Launch, typename AfterSync>
double measureEvent(Launch&& launch, AfterSync&& after_sync, int warmup = 2, int trials = 7)
{
    CudaEvent start;
    CudaEvent stop;
    for (int iteration = 0; iteration < warmup; ++iteration)
    {
        launch();
        PLAMATRIX_CHECK_CUDA(cudaDeviceSynchronize());
        after_sync();
    }
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(trials));
    for (int iteration = 0; iteration < trials; ++iteration)
    {
        PLAMATRIX_CHECK_CUDA(cudaEventRecord(start.get()));
        launch();
        PLAMATRIX_CHECK_CUDA(cudaEventRecord(stop.get()));
        PLAMATRIX_CHECK_CUDA(cudaEventSynchronize(stop.get()));
        float elapsed_ms = 0.0f;
        PLAMATRIX_CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start.get(), stop.get()));
        samples.push_back(static_cast<double>(elapsed_ms));
        after_sync();
    }
    return median(samples);
}

template <typename Function>
double measureCold(Function&& function)
{
    PLAMATRIX_CHECK_CUDA(cudaDeviceSynchronize());
    const auto start = Clock::now();
    function();
    PLAMATRIX_CHECK_CUDA(cudaDeviceSynchronize());
    const auto stop = Clock::now();
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

template <typename Function>
double measureTransfer(Function&& function)
{
    return measureWall(std::forward<Function>(function), 1, 3);
}

void validate(const CaseResult& result)
{
    const double values[] = {
        result.time_cuda_cold_allocation_ms,
        result.time_cuda_warm_workspace_ms,
        result.time_cuda_ms,
        result.time_transfer_ms
    };
    for (double value : values)
    {
        if (!std::isfinite(value) || value < 0.0)
        {
            throw std::runtime_error(result.name + " produced an invalid CUDA timing");
        }
    }
}

template <typename Function>
void measureSynchronousCase(CaseResult& result, Function&& function)
{
    result.time_cuda_cold_allocation_ms = measureCold(function);
    result.time_cuda_warm_workspace_ms = measureWall(function);
    result.time_cuda_ms = result.time_cuda_warm_workspace_ms;
}

} // namespace

void runCooToCsrCuda(CaseResult& result, Index size)
{
    CpuSparseFixture fixture(size);
    auto rows_cpu = DenseMatrix<Index, Device::CPU>::pinned(
        static_cast<Index>(fixture.rows.size()), 1);
    auto columns_cpu = DenseMatrix<Index, Device::CPU>::pinned(rows_cpu.rows(), 1);
    auto values_cpu = DenseMatrix<float, Device::CPU>::pinned(rows_cpu.rows(), 1);
    for (Index index = 0; index < rows_cpu.rows(); ++index)
    {
        rows_cpu(index, 0) = fixture.rows[static_cast<std::size_t>(index)];
        columns_cpu(index, 0) = fixture.columns[static_cast<std::size_t>(index)];
        values_cpu(index, 0) = fixture.values[static_cast<std::size_t>(index)];
    }
    DenseMatrix<Index, Device::GPU> rows_gpu(rows_cpu.rows(), 1);
    DenseMatrix<Index, Device::GPU> columns_gpu(rows_cpu.rows(), 1);
    DenseMatrix<float, Device::GPU> values_gpu(rows_cpu.rows(), 1);
    result.time_transfer_ms = measureTransfer([&]()
    {
        rows_cpu.copyToGpuAsync(rows_gpu);
        columns_cpu.copyToGpuAsync(columns_gpu);
        values_cpu.copyToGpuAsync(values_gpu);
    });

    result.time_cuda_cold_allocation_ms = measureCold([&]()
    {
        SparseOpsWorkspace cold_workspace;
        auto output = cooToCsr(size, size, rows_gpu, columns_gpu, values_gpu, cold_workspace);
        volatile Index sink = output.nnz();
        static_cast<void>(sink);
    });

    CSRMatrix<float, Device::GPU> output(size, size, rows_cpu.rows());
    SparseOpsWorkspace workspace;
    cooToCsrAsync(size, size, rows_gpu, columns_gpu, values_gpu, output, workspace);
    PLAMATRIX_CHECK_CUDA(cudaDeviceSynchronize());
    workspace.checkStatus("coo_to_csr benchmark warmup");
    result.workspace_bytes_before = workspace.capacityBytes();
    result.time_cuda_ms = measureEvent(
        [&]() { cooToCsrAsync(size, size, rows_gpu, columns_gpu, values_gpu, output, workspace); },
        [&]() { workspace.checkStatus("coo_to_csr benchmark"); });
    result.time_cuda_warm_workspace_ms = measureWall([&]()
    {
        cooToCsrAsync(size, size, rows_gpu, columns_gpu, values_gpu, output, workspace);
        PLAMATRIX_CHECK_CUDA(cudaDeviceSynchronize());
        workspace.checkStatus("coo_to_csr benchmark");
    });
    result.workspace_bytes_after = workspace.capacityBytes();
    validate(result);
}

void runSparseProductCuda(CaseResult& result, Index size, bool matrix_product)
{
    CpuSparseFixture fixture(size);
    const auto matrix_gpu = fixture.matrix.toGpu();
    const auto input_gpu = matrix_product ? fixture.block.toGpu() : fixture.vector.toGpu();
    DenseMatrix<float, Device::GPU> output(
        size, matrix_product ? fixture.block.cols() : Index{1});
    result.time_transfer_ms = measureTransfer([&]()
    {
        auto transferred = matrix_product ? fixture.block.toGpu() : fixture.vector.toGpu();
        volatile const float* sink = transferred.data();
        static_cast<void>(sink);
    });
    result.time_cuda_cold_allocation_ms = measureCold([&]()
    {
        SparseOpsWorkspace cold_workspace;
        if (matrix_product)
        {
            spmmAsync(matrix_gpu, input_gpu, output, cold_workspace);
        }
        else
        {
            spmvAsync(matrix_gpu, input_gpu, output, cold_workspace);
        }
    });
    SparseOpsWorkspace workspace;
    if (matrix_product)
    {
        spmmAsync(matrix_gpu, input_gpu, output, workspace);
    }
    else
    {
        spmvAsync(matrix_gpu, input_gpu, output, workspace);
    }
    PLAMATRIX_CHECK_CUDA(cudaDeviceSynchronize());
    result.workspace_bytes_before = workspace.capacityBytes();
    auto launch = [&]()
    {
        if (matrix_product)
        {
            spmmAsync(matrix_gpu, input_gpu, output, workspace);
        }
        else
        {
            spmvAsync(matrix_gpu, input_gpu, output, workspace);
        }
    };
    result.time_cuda_ms = measureEvent(launch, []() {});
    result.time_cuda_warm_workspace_ms = measureWall(launch);
    result.workspace_bytes_after = workspace.capacityBytes();
    validate(result);
}

void runSpmvCuda(CaseResult& result, Index size)
{
    runSparseProductCuda(result, size, false);
}

void runSpmmCuda(CaseResult& result, Index size)
{
    runSparseProductCuda(result, size, true);
}

void runSolverCuda(CaseResult& result, Index size, bool preconditioned)
{
    CpuSparseFixture fixture(size);
    const auto matrix_gpu = fixture.matrix.toGpu();
    const auto rhs_gpu = fixture.rhs.toGpu();
    DenseMatrix<float, Device::GPU> solution(size, 1);
    IterativeSolverOptions options;
    options.maxIterations = std::max(8, static_cast<int>(size));
    options.relativeTolerance = 1.0e-5;
    result.time_transfer_ms = measureTransfer([&]()
    {
        auto transferred = fixture.rhs.toGpu();
        volatile const float* sink = transferred.data();
        static_cast<void>(sink);
    });
    auto solve = [&](IterativeSolverWorkspace<float>& workspace)
    {
        solution.fill(0.0f);
        const auto report = preconditioned
            ? pcg(matrix_gpu, rhs_gpu, solution, workspace, options)
            : cg(matrix_gpu, rhs_gpu, solution, workspace, options);
        volatile double sink = report.finalResidual;
        static_cast<void>(sink);
    };
    result.time_cuda_cold_allocation_ms = measureCold([&]()
    {
        IterativeSolverWorkspace<float> cold_workspace;
        solve(cold_workspace);
    });
    IterativeSolverWorkspace<float> workspace;
    solve(workspace);
    result.workspace_bytes_before = static_cast<std::size_t>(workspace.capacitySize())
        * sizeof(float) * 5;
    result.time_cuda_warm_workspace_ms = measureWall([&]() { solve(workspace); });
    result.time_cuda_ms = result.time_cuda_warm_workspace_ms;
    result.workspace_bytes_after = static_cast<std::size_t>(workspace.capacitySize())
        * sizeof(float) * 5;
    validate(result);
}

void runCgCuda(CaseResult& result, Index size)
{
    runSolverCuda(result, size, false);
}

void runPcgCuda(CaseResult& result, Index size)
{
    runSolverCuda(result, size, true);
}

} // namespace detail
} // namespace plamatrix
