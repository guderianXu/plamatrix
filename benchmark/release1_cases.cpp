#include "benchmark/benchmark_cases.h"
#include "benchmark/report_writer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <utility>

#include <omp.h>

#include "plamatrix/dense/dense_matrix.h"
#include "plamatrix/dense/elementwise.h"
#include "plamatrix/ops/indexing.h"
#include "plamatrix/ops/reduction.h"
#include "plamatrix/ops/small_matrix.h"

namespace plamatrix
{
namespace detail
{
namespace
{

using FloatMatrix = DenseMatrix<float, Device::CPU>;
using MaskMatrix = DenseMatrix<std::uint8_t, Device::CPU>;

class OmpThreadGuard
{
public:
    explicit OmpThreadGuard(int threads)
        : _previousThreads(omp_get_max_threads())
    {
        omp_set_num_threads(threads);
    }

    ~OmpThreadGuard()
    {
        omp_set_num_threads(_previousThreads);
    }

private:
    int _previousThreads;
};

FloatMatrix makeRandom(Index rows, Index cols)
{
    FloatMatrix matrix(rows, cols);
    std::mt19937 generator(42);
    std::uniform_real_distribution<float> distribution(0.25f, 1.0f);
    for (Index index = 0; index < matrix.size(); ++index)
    {
        matrix.data()[index] = distribution(generator);
    }
    return matrix;
}

MaskMatrix makeMask(Index rows)
{
    MaskMatrix mask(rows, 1);
    for (Index row = 0; row < rows; ++row)
    {
        mask(row, 0) = static_cast<std::uint8_t>((row % 3) != 0);
    }
    return mask;
}

FloatMatrix makeEighInput(Index rows)
{
    FloatMatrix matrices(rows, 6);
    for (Index row = 0; row < rows; ++row)
    {
        const float offset = static_cast<float>(row % 17) * 0.001f;
        matrices(row, 0) = 4.0f + offset;
        matrices(row, 1) = 0.1f;
        matrices(row, 2) = 0.05f;
        matrices(row, 3) = 3.0f + offset;
        matrices(row, 4) = 0.02f;
        matrices(row, 5) = 2.0f + offset;
    }
    return matrices;
}

void validateCpuTiming(double time_ms, const char* case_name)
{
    if (!std::isfinite(time_ms) || time_ms < 0.0)
    {
        throw std::runtime_error(std::string(case_name) + " produced an invalid CPU timing");
    }
}

bool shouldRunCase(const std::vector<std::string>& case_filter, const char* name)
{
    return case_filter.empty()
        || std::find(case_filter.begin(), case_filter.end(), std::string(name)) != case_filter.end();
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
double measureCpu(Function&& function)
{
    const double time_ms = measure(std::forward<Function>(function), 2, 5);
    return time_ms;
}

template <typename Function>
void runCpuModes(CaseResult& result, bool serial, bool omp, Function&& function)
{
    if (serial)
    {
        OmpThreadGuard guard(1);
        result.time_serial_ms = measureCpu(function);
        validateCpuTiming(result.time_serial_ms, result.name.c_str());
    }
    if (omp)
    {
        result.time_omp_ms = measureCpu(function);
        validateCpuTiming(result.time_omp_ms, result.name.c_str());
    }
}

} // namespace

void runElementwiseCpu(CaseResult& result, Index size, bool serial, bool omp)
{
    auto lhs = makeRandom(size, size);
    auto rhs = makeRandom(size, size);
    runCpuModes(result, serial, omp, [&]()
    {
        auto output = hadamardMultiply(lhs, rhs);
        volatile float sink = output.data()[0];
        static_cast<void>(sink);
    });
}

void runReductionCpu(CaseResult& result, Index size, bool serial, bool omp)
{
    auto input = makeRandom(size, size);
    runCpuModes(result, serial, omp, [&]()
    {
        auto output = sum(input, ReductionAxis::All);
        volatile float sink = output.data()[0];
        static_cast<void>(sink);
    });
}

void runCompactCpu(CaseResult& result, Index size, bool serial, bool omp)
{
    auto input = makeRandom(size, 3);
    auto mask = makeMask(size);
    runCpuModes(result, serial, omp, [&]()
    {
        auto output = compactRows(input, mask);
        volatile float sink = output.values.data()[0];
        static_cast<void>(sink);
    });
}

void runEigh3x3BatchCpu(CaseResult& result, Index size, bool serial, bool omp)
{
    auto input = makeEighInput(size);
    runCpuModes(result, serial, omp, [&]()
    {
        auto output = symmetricEigh3x3Batched(input);
        volatile float sink = output.eigenvalues.data()[0];
        static_cast<void>(sink);
    });
}

#ifndef PLAMATRIX_WITH_CUDA
void runElementwiseCuda(CaseResult&, Index)
{
}

void runReductionCuda(CaseResult&, Index)
{
}

void runCompactCuda(CaseResult&, Index)
{
}

void runEigh3x3BatchCuda(CaseResult&, Index)
{
}
#endif

template <typename CpuRunner, typename CudaRunner>
void runRelease1Case(const char* name,
                     Index size,
                     bool serial,
                     bool omp,
                     bool cuda,
                     BenchmarkReport& report,
                     CpuRunner&& cpu_runner,
                     CudaRunner&& cuda_runner)
{
    CaseResult result;
    result.name = name;
    result.size = size;
    if (serial || omp)
    {
        cpu_runner(result, size, serial, omp);
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

void runRelease1Cases(Index size,
                      bool serial,
                      bool omp,
                      bool cuda,
                      BenchmarkReport& report,
                      const std::vector<std::string>& case_filter)
{
    if (shouldRunCase(case_filter, "elementwise"))
    {
        runRelease1Case("elementwise", size, serial, omp, cuda, report,
                        runElementwiseCpu, runElementwiseCuda);
    }
    if (shouldRunCase(case_filter, "reduction"))
    {
        runRelease1Case("reduction", size, serial, omp, cuda, report,
                        runReductionCpu, runReductionCuda);
    }
    if (shouldRunCase(case_filter, "compact"))
    {
        runRelease1Case("compact", size, serial, omp, cuda, report,
                        runCompactCpu, runCompactCuda);
    }
    if (shouldRunCase(case_filter, "eigh3x3_batch"))
    {
        runRelease1Case("eigh3x3_batch", size, serial, omp, cuda, report,
                        runEigh3x3BatchCpu, runEigh3x3BatchCuda);
    }
}

} // namespace detail
} // namespace plamatrix
