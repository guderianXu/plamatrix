#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "benchmark/benchmark_cases.h"
#include "benchmark/report_writer.h"
#include "plamatrix/core/types.h"

namespace
{

/// Print usage and exit.
void printUsage(const char* prog)
{
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --mode all|cpu|cuda     Benchmark mode (default: all)\n"
              << "  --size tiny|smoke|small|medium|large  Size preset (default: medium)\n"
              << "  --case LIST             Comma-separated cases to run (default: all)\n"
              << "  --output FILE           Output report path (default: benchmark_report.md)\n"
              << "  --list                  List all benchmark cases and exit\n"
              << "  --help                  Print this help message\n"
              << "\n"
              << "Size presets:\n"
              << "  tiny/smoke: 16, 32\n"
              << "  small:  256, 512, 1024, 2048\n"
              << "  medium: 1024, 2048, 4096, 8192\n"
              << "  large:  4096, 8192, 12288, 16384\n";
}

/// Parse the --size argument into a vector of Index values.
std::vector<plamatrix::Index> parseSize(const std::string& size_str)
{
    if (size_str == "tiny" || size_str == "smoke")
    {
        return {16, 32};
    }
    if (size_str == "small")
    {
        return {256, 512, 1024, 2048};
    }
    if (size_str == "medium")
    {
        return {1024, 2048, 4096, 8192};
    }
    if (size_str == "large")
    {
        return {4096, 8192, 12288, 16384};
    }
    std::cerr << "Unknown size preset: " << size_str << ". Using medium.\n";
    return {1024, 2048, 4096, 8192};
}

/// Parse a comma-separated --case argument into benchmark case names.
std::string trim(std::string value)
{
    auto is_not_space = [](unsigned char ch) { return !std::isspace(ch); };
    auto begin = std::find_if(value.begin(), value.end(), is_not_space);
    auto end = std::find_if(value.rbegin(), value.rend(), is_not_space).base();
    if (begin >= end)
    {
        return {};
    }
    return std::string(begin, end);
}

std::vector<std::string> parseCases(const std::string& case_str)
{
    std::vector<std::string> cases;
    std::stringstream ss(case_str);
    std::string item;
    while (std::getline(ss, item, ','))
    {
        item = trim(item);
        if (!item.empty())
        {
            cases.push_back(item);
        }
    }
    return cases;
}

bool isKnownCase(const std::string& name)
{
    auto all = plamatrix::getAllCaseNames();
    return std::find(all.begin(), all.end(), name) != all.end();
}

} // anonymous namespace

int main(int argc, char** argv)
{
    // Defaults
    std::string mode_str = "all";
    std::string size_str = "medium";
    std::string output_path = "benchmark_report.md";
    std::vector<std::string> case_filter;
    bool list_only = false;

    // Parse CLI arguments
    for (int i = 1; i < argc; ++i)
    {
        std::string arg(argv[i]);

        if (arg == "--help")
        {
            printUsage(argv[0]);
            return 0;
        }
        if (arg == "--list")
        {
            list_only = true;
        }
        else if (arg == "--mode" && i + 1 < argc)
        {
            mode_str = argv[++i];
        }
        else if (arg == "--size" && i + 1 < argc)
        {
            size_str = argv[++i];
        }
        else if (arg == "--case" && i + 1 < argc)
        {
            case_filter = parseCases(argv[++i]);
        }
        else if (arg == "--output" && i + 1 < argc)
        {
            output_path = argv[++i];
        }
        else
        {
            std::cerr << "Unknown option: " << arg << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    // --list mode
    if (list_only)
    {
        auto names = plamatrix::getAllCaseNames();
        for (const auto& name : names)
        {
            std::cout << name << "\n";
        }
        return 0;
    }

    // Validate mode
    bool run_serial = false;
    bool run_omp = false;
    bool run_cuda = false;

    if (mode_str == "all")
    {
        run_serial = true;
        run_omp = true;
        run_cuda = true;
    }
    else if (mode_str == "cpu")
    {
        run_serial = true;
        run_omp = true;
        run_cuda = false;
    }
    else if (mode_str == "cuda")
    {
        run_serial = false;
        run_omp = false;
        run_cuda = true;
    }
    else
    {
        std::cerr << "Unknown mode: " << mode_str << ". Use all, cpu, or cuda.\n";
        return 1;
    }

#ifndef PLAMATRIX_WITH_CUDA
    if (run_cuda)
    {
        if (!run_serial && !run_omp)
        {
            std::cerr << "CUDA benchmark mode requested, but this build was compiled with PLAMATRIX_WITH_CUDA=OFF.\n";
            return 1;
        }
        std::cerr << "CUDA support is not compiled in; CUDA benchmark backend will be skipped.\n";
        run_cuda = false;
    }
#endif

    for (const auto& case_name : case_filter)
    {
        if (!isKnownCase(case_name))
        {
            std::cerr << "Unknown benchmark case: " << case_name << "\n";
            std::cerr << "Use --list to show available cases.\n";
            return 1;
        }
    }

    // Parse sizes
    auto sizes = parseSize(size_str);

    std::cerr << "PlaMatrix Benchmark\n"
              << "  Mode: " << mode_str << "\n"
              << "  Sizes: ";
    for (std::size_t i = 0; i < sizes.size(); ++i)
    {
        if (i > 0) std::cerr << ", ";
        std::cerr << sizes[i];
    }
    std::cerr << "\n  Cases: ";
    if (case_filter.empty())
    {
        std::cerr << "all";
    }
    else
    {
        for (std::size_t i = 0; i < case_filter.size(); ++i)
        {
            if (i > 0) std::cerr << ", ";
            std::cerr << case_filter[i];
        }
    }
    std::cerr << "\n  Output: " << output_path << "\n" << std::endl;

    // Run benchmarks
    plamatrix::BenchmarkReport report;
    report.captureEnvironment();

    std::cerr << "Environment:\n"
              << "  CPU: " << report.cpu_model << " (" << report.cpu_cores << " cores)\n"
              << "  GPU: " << report.gpu_model << "\n"
              << "  CUDA: " << report.cuda_version << "\n"
              << "  OS: " << report.os_info << "\n" << std::endl;

    std::cerr << "Running benchmarks..." << std::endl;
    plamatrix::runAllCases(sizes, run_serial, run_omp, run_cuda, report, case_filter);

    std::cerr << "Completed " << report.results.size() << " benchmark results." << std::endl;
    if (report.results.empty())
    {
        std::cerr << "No benchmark results were produced for the selected mode, cases, and sizes.\n";
        return 1;
    }

    std::cerr << "Writing report to " << output_path << "..." << std::endl;

    report.writeMarkdown(output_path);

    std::cerr << "Done." << std::endl;
    return 0;
}
