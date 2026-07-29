#pragma once

#include <cstddef>
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
    double time_cuda_cold_allocation_ms = -1.0;
    double time_cuda_warm_workspace_ms = -1.0;
    std::size_t workspace_bytes_before = 0;
    std::size_t workspace_bytes_after = 0;
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
/// @param case_filter  Optional case names to run; empty means all cases
void runAllCases(const std::vector<Index>& sizes,
                 bool serial,
                 bool omp,
                 bool cuda,
                 BenchmarkReport& report,
                 const std::vector<std::string>& case_filter = {});

/// Get the list of all benchmark case names.
/// @return  Vector of case name strings
std::vector<std::string> getAllCaseNames();

// ============================================================================
// Implementation-detail benchmark functions (defined in benchmark case sources)
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

void runElementwiseCpu(CaseResult& r, Index N, bool serial, bool omp);
void runReductionCpu(CaseResult& r, Index N, bool serial, bool omp);
void runCompactCpu(CaseResult& r, Index N, bool serial, bool omp);
void runEigh3x3BatchCpu(CaseResult& r, Index N, bool serial, bool omp);

void runRelease1Cases(Index size,
                      bool serial,
                      bool omp,
                      bool cuda,
                      BenchmarkReport& report,
                      const std::vector<std::string>& case_filter);

void runElementwiseCuda(CaseResult& r, Index N);
void runReductionCuda(CaseResult& r, Index N);
void runCompactCuda(CaseResult& r, Index N);
void runEigh3x3BatchCuda(CaseResult& r, Index N);

void runRelease3Cases(Index size,
                      bool serial,
                      bool omp,
                      bool cuda,
                      BenchmarkReport& report,
                      const std::vector<std::string>& case_filter);

void runCooToCsrCuda(CaseResult& r, Index N);
void runSpmvCuda(CaseResult& r, Index N);
void runSpmmCuda(CaseResult& r, Index N);
void runCgCuda(CaseResult& r, Index N);
void runPcgCuda(CaseResult& r, Index N);

} // namespace detail

} // namespace plamatrix
