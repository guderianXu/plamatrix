#pragma once

#include <functional>
#include <string>
#include <vector>

#include "plamatrix/core/types.h"

namespace plamatrix
{

using BenchmarkFn = std::function<void()>;

/// Result of a single benchmark case at one matrix size.
struct CaseResult
{
    std::string name;
    Index size = 0;
    double time_serial_ms = -1.0;
    double time_omp_ms = -1.0;
    double time_gpu_ms = -1.0;
    double time_transfer_ms = -1.0;
};

// Forward declaration
struct BenchmarkReport;

/// Run a benchmark function with warmup and multiple timed trials.
/// Returns the median time in milliseconds.
/// @param fn       The benchmark function to time
/// @param warmup   Number of warmup iterations (default 3)
/// @param trials   Number of timed trials (default 10)
/// @return  Median duration in milliseconds
double measure(BenchmarkFn fn, int warmup = 3, int trials = 10);

/// Run all benchmark cases for the given sizes and device mode flags.
/// @param sizes    Matrix sizes to benchmark
/// @param serial   Whether to run serial CPU benchmarks
/// @param omp      Whether to run OpenMP CPU benchmarks
/// @param gpu      Whether to run GPU benchmarks
/// @param report   Output report to populate with results
/// @param case_filter  Optional case names to run; empty means all cases
void runAllCases(const std::vector<Index>& sizes,
                 bool serial,
                 bool omp,
                 bool gpu,
                 BenchmarkReport& report,
                 const std::vector<std::string>& case_filter = {});

/// Get the list of all benchmark case names.
/// @return  Vector of case name strings
std::vector<std::string> getAllCaseNames();

// ============================================================================
// Implementation-detail GPU benchmark functions (defined by the selected GPU backend)
// ============================================================================
namespace detail
{

void runGemmGpu(CaseResult& r, Index N);
void runAddGpu(CaseResult& r, Index N);
void runSubGpu(CaseResult& r, Index N);
void runTransposeGpu(CaseResult& r, Index N);
void runSvdGpu(CaseResult& r, Index N);
void runQrGpu(CaseResult& r, Index N);
void runEighGpu(CaseResult& r, Index N);
void runSolveGpu(CaseResult& r, Index N);
void runCovarianceGpu(CaseResult& r, Index N);
void runPointTransformGpu(CaseResult& r, Index N);

} // namespace detail

} // namespace plamatrix
