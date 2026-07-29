#include "benchmark/benchmark_cases.h"
#include "benchmark/report_writer.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "plamatrix/dense/dense_matrix.h"
#include "plamatrix/sparse/iterative_solver.h"
#include "plamatrix/sparse/sparse_ops.h"

namespace plamatrix
{
namespace detail
{
namespace
{

struct SparseFixture
{
    std::vector<Index> rows;
    std::vector<Index> columns;
    std::vector<float> values;
    CSRMatrix<float, Device::CPU> matrix;
    DenseMatrix<float, Device::CPU> vector;
    DenseMatrix<float, Device::CPU> block;
    DenseMatrix<float, Device::CPU> rhs;

    explicit SparseFixture(Index size)
        : matrix(makeMatrix(size, rows, columns, values))
        , vector(size, 1)
        , block(size, 8)
        , rhs(size, 1)
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

bool shouldRunCase(const std::vector<std::string>& filter, const char* name)
{
    return filter.empty()
        || std::find(filter.begin(), filter.end(), std::string(name)) != filter.end();
}

template <typename Function>
double measureChecked(const char* name, Function&& function)
{
    const double elapsed = measure(std::forward<Function>(function), 2, 5);
    if (!std::isfinite(elapsed) || elapsed < 0.0)
    {
        throw std::runtime_error(std::string(name) + " produced an invalid CPU timing");
    }
    return elapsed;
}

void printCpuTimings(const CaseResult& result, bool serial, bool omp)
{
    std::cerr << "  " << result.name;
    if (serial)
    {
        std::cerr << " cpu_serial_total_ms=" << result.time_serial_ms;
    }
    if (omp)
    {
        std::cerr << " cpu_omp_total_ms=" << result.time_omp_ms;
    }
    std::cerr << std::endl;
}

void printCudaTimings(const CaseResult& result)
{
    std::cerr << "  " << result.name << " cuda"
              << " cold_allocation_ms=" << result.time_cuda_cold_allocation_ms
              << " warm_workspace_ms=" << result.time_cuda_warm_workspace_ms
              << " kernel_only_ms=" << result.time_cuda_ms
              << " transfer_ms=" << result.time_transfer_ms
              << " workspace_bytes=" << result.workspace_bytes_before
              << "->" << result.workspace_bytes_after << std::endl;
}

template <typename Function>
void runCpuModes(CaseResult& result, bool serial, bool omp, Function&& function)
{
    if (serial)
    {
        result.time_serial_ms = measureChecked(result.name.c_str(), function);
    }
    if (omp)
    {
        result.time_omp_ms = measureChecked(result.name.c_str(), function);
    }
}

void runCpuCase(CaseResult& result, Index size, bool serial, bool omp)
{
    SparseFixture fixture(size);
    if (result.name == "coo_to_csr")
    {
        runCpuModes(result, serial, omp, [&]()
        {
            auto output = cooToCsr(size, size, fixture.rows, fixture.columns, fixture.values);
            volatile Index sink = output.nnz();
            static_cast<void>(sink);
        });
        return;
    }

    if (result.name == "spmv")
    {
        DenseMatrix<float, Device::CPU> output(size, 1);
        runCpuModes(result, serial, omp, [&]() { spmv(fixture.matrix, fixture.vector, output); });
        return;
    }
    if (result.name == "spmm")
    {
        DenseMatrix<float, Device::CPU> output(size, fixture.block.cols());
        runCpuModes(result, serial, omp, [&]() { spmm(fixture.matrix, fixture.block, output); });
        return;
    }

    DenseMatrix<float, Device::CPU> solution(size, 1);
    IterativeSolverOptions options;
    options.maxIterations = std::max(8, static_cast<int>(size));
    options.relativeTolerance = 1.0e-5;
    runCpuModes(result, serial, omp, [&]()
    {
        solution.fill(0.0f);
        const auto report = result.name == "cg"
            ? cg(fixture.matrix, fixture.rhs, solution, options)
            : pcg(fixture.matrix, fixture.rhs, solution, options);
        volatile double sink = report.finalResidual;
        static_cast<void>(sink);
    });
}

using CudaRunner = void (*)(CaseResult&, Index);

void runCase(const char* name,
             Index size,
             bool serial,
             bool omp,
             bool cuda,
             BenchmarkReport& report,
             CudaRunner cuda_runner)
{
    CaseResult result;
    result.name = name;
    result.size = size;
    if (serial || omp)
    {
        runCpuCase(result, size, serial, omp);
        printCpuTimings(result, serial, omp);
    }
#ifdef PLAMATRIX_WITH_CUDA
    if (cuda)
    {
        cuda_runner(result, size);
        printCudaTimings(result);
    }
#else
    static_cast<void>(cuda);
    static_cast<void>(cuda_runner);
#endif
    report.results.push_back(std::move(result));
}

} // namespace

#ifndef PLAMATRIX_WITH_CUDA
void runCooToCsrCuda(CaseResult&, Index) {}
void runSpmvCuda(CaseResult&, Index) {}
void runSpmmCuda(CaseResult&, Index) {}
void runCgCuda(CaseResult&, Index) {}
void runPcgCuda(CaseResult&, Index) {}
#endif

void runRelease3Cases(Index size,
                      bool serial,
                      bool omp,
                      bool cuda,
                      BenchmarkReport& report,
                      const std::vector<std::string>& case_filter)
{
    const struct
    {
        const char* name;
        CudaRunner cudaRunner;
    } cases[] = {
        {"coo_to_csr", runCooToCsrCuda},
        {"spmv", runSpmvCuda},
        {"spmm", runSpmmCuda},
        {"cg", runCgCuda},
        {"pcg", runPcgCuda}
    };
    for (const auto& benchmark_case : cases)
    {
        if (shouldRunCase(case_filter, benchmark_case.name))
        {
            runCase(benchmark_case.name, size, serial, omp, cuda, report,
                    benchmark_case.cudaRunner);
        }
    }
}

} // namespace detail
} // namespace plamatrix
