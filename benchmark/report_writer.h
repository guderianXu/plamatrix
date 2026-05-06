#pragma once

#include <string>
#include <vector>

#include "benchmark/benchmark_cases.h"

namespace plamatrix
{

/// Holds environment information and benchmark results.
/// Can capture system details and write a markdown report.
struct BenchmarkReport
{
    std::string cpu_model;
    int cpu_cores = 0;
    std::string gpu_model;
    std::string gpu_driver;
    std::string cuda_version;
    std::string os_info;
    std::vector<CaseResult> results;

    /// Capture CPU, GPU, and OS environment details.
    void captureEnvironment();

    /// Write results as a formatted markdown report.
    /// @param path  Output file path
    void writeMarkdown(const std::string& path) const;
};

} // namespace plamatrix
