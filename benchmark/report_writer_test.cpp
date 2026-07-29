#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

#include "benchmark/report_writer.h"

namespace plamatrix
{
namespace
{

TEST(BenchmarkReportTest, WritesRelease1CudaTimingBreakdown)
{
    BenchmarkReport report;
    CaseResult result;
    result.name = "elementwise";
    result.size = 256;
    result.time_serial_ms = 4.0;
    result.time_cuda_ms = 1.0;
    result.time_transfer_ms = 2.0;
    result.time_cuda_cold_allocation_ms = 3.0;
    result.time_cuda_warm_workspace_ms = 1.5;
    report.results.push_back(result);

    const auto path = std::filesystem::temp_directory_path() /
        "plamatrix_benchmark_report_schema_test.md";
    report.writeMarkdown(path.string());
    std::ifstream input(path);
    std::ostringstream contents;
    contents << input.rdbuf();
    input.close();
    std::filesystem::remove(path);
    const std::string markdown = contents.str();

    EXPECT_NE(markdown.find("CUDA Cold Allocation (ms)"), std::string::npos);
    EXPECT_NE(markdown.find("CUDA Warm Workspace (ms)"), std::string::npos);
    EXPECT_NE(markdown.find("CUDA Kernel Only (ms)"), std::string::npos);
    EXPECT_NE(markdown.find("Transfer (ms)"), std::string::npos);
    EXPECT_NE(markdown.find("| 256 | 4.000 | 3.000 | 1.500 | 1.000 | 2.000 |"),
              std::string::npos);
}

} // namespace
} // namespace plamatrix
