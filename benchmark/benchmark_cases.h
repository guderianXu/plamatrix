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
    double time_cuda_ms = -1.0;
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
/// @param cuda     Whether to run GPU benchmarks
/// @param report   Output report to populate with results
void runAllCases(const std::vector<Index>& sizes, bool serial, bool omp, bool cuda, BenchmarkReport& report);

/// Get the list of all benchmark case names.
/// @return  Vector of case name strings
std::vector<std::string> getAllCaseNames();

// ============================================================================
// Implementation-detail GPU benchmark functions (defined in benchmark_cases.cu)
// ============================================================================
namespace detail
{

void runGemmCuda(CaseResult& r, Index N);
void runAddCuda(CaseResult& r, Index N);
void runSubCuda(CaseResult& r, Index N);
void runTransposeCuda(CaseResult& r, Index N);
void runSvdCuda(CaseResult& r, Index N);
void runQrCuda(CaseResult& r, Index N);
void runEighCuda(CaseResult& r, Index N);
void runSolveCuda(CaseResult& r, Index N);
void runCovarianceCuda(CaseResult& r, Index N);
void runPointTransformCuda(CaseResult& r, Index N);

} // namespace detail

} // namespace plamatrix
