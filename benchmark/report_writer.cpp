#include "benchmark/report_writer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include <sys/utsname.h>

#ifdef PLAMATRIX_WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace plamatrix
{

namespace
{

/// Read /proc/cpuinfo and extract the model name and core count.
void readCpuInfo(std::string& model, int& cores)
{
    std::ifstream cpuinfo("/proc/cpuinfo");
    if (!cpuinfo.is_open())
    {
        model = "Unknown";
        cores = 0;
        return;
    }

    std::string line;
    int model_count = 0;
    cores = 0;
    while (std::getline(cpuinfo, line))
    {
        if (line.rfind("model name", 0) == 0)
        {
            if (model_count == 0)
            {
                auto pos = line.find(':');
                if (pos != std::string::npos)
                {
                    model = line.substr(pos + 1);
                    // Trim leading whitespace
                    auto start = model.find_first_not_of(" \t");
                    if (start != std::string::npos)
                    {
                        model = model.substr(start);
                    }
                }
            }
            ++model_count;
        }
        if (line.rfind("processor", 0) == 0)
        {
            ++cores;
        }
    }

    if (model.empty())
    {
        model = "Unknown";
    }
}

/// Read GPU device properties via CUDA API.
void readGpuInfo(std::string& gpu_model, std::string& gpu_driver, std::string& cuda_ver)
{
#ifndef PLAMATRIX_WITH_CUDA
    gpu_model = "N/A";
    gpu_driver = "N/A";
    cuda_ver = "N/A";
#else
    int driver_version = 0;
    auto cu_err = cudaDriverGetVersion(&driver_version);
    if (cu_err != cudaSuccess)
    {
        gpu_model = "N/A";
        gpu_driver = "N/A";
    }
    else
    {
        int major = driver_version / 1000;
        int minor = (driver_version % 1000) / 10;
        gpu_driver = std::to_string(major) + "." + std::to_string(minor);
    }

    int runtime_version = 0;
    cu_err = cudaRuntimeGetVersion(&runtime_version);
    if (cu_err != cudaSuccess)
    {
        cuda_ver = "N/A";
    }
    else
    {
        int rt_major = runtime_version / 1000;
        int rt_minor = (runtime_version % 1000) / 10;
        cuda_ver = std::to_string(rt_major) + "." + std::to_string(rt_minor);
    }

    int device_count = 0;
    cu_err = cudaGetDeviceCount(&device_count);
    if (cu_err != cudaSuccess || device_count == 0)
    {
        gpu_model = "N/A";
        return;
    }

    cudaDeviceProp prop;
    cu_err = cudaGetDeviceProperties(&prop, 0);
    if (cu_err != cudaSuccess)
    {
        gpu_model = "N/A";
        return;
    }

    gpu_model = prop.name;

    // Append VRAM info
    double vram_gb = static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0);
    std::ostringstream oss;
    oss << prop.name << " (" << std::fixed << std::setprecision(1) << vram_gb << " GB)";
    gpu_model = oss.str();
#endif
}

/// Format a double in a compact way.
std::string fmtTime(double ms)
{
    if (ms < 0.0)
    {
        return "N/A";
    }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3) << ms;
    return oss.str();
}

/// Format speedup ratio.
std::string fmtSpeedup(double baseline_ms, double accelerated_ms)
{
    if (baseline_ms <= 0.0 || accelerated_ms <= 0.0)
    {
        return "N/A";
    }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << (baseline_ms / accelerated_ms) << "x";
    return oss.str();
}

} // anonymous namespace

void BenchmarkReport::captureEnvironment()
{
    readCpuInfo(cpu_model, cpu_cores);
    readGpuInfo(gpu_model, gpu_driver, cuda_version);

    // OS info via uname
    struct utsname buf;
    if (uname(&buf) == 0)
    {
        std::ostringstream oss;
        oss << buf.sysname << " " << buf.release << " " << buf.machine;
        os_info = oss.str();
    }
    else
    {
        os_info = "Unknown";
    }
}

void BenchmarkReport::writeMarkdown(const std::string& path) const
{
    std::ofstream file(path);
    if (!file.is_open())
    {
        throw std::runtime_error("Cannot open report file: " + path);
    }

    // Current timestamp
    std::time_t now = std::time(nullptr);
    char time_buf[64];
    std::strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));

    file << "# PlaMatrix Benchmark Report\n\n";
    file << "**Generated:** " << time_buf << "\n\n";

    // ---- Environment ----
    file << "## Environment\n\n";
    file << "| Item | Value |\n";
    file << "|------|-------|\n";
    file << "| CPU | " << cpu_model << " |\n";
    file << "| CPU Cores | " << cpu_cores << " |\n";
    file << "| GPU | " << gpu_model << " |\n";
    file << "| GPU Driver | " << gpu_driver << " |\n";
    file << "| CUDA Version | " << cuda_version << " |\n";
    file << "| OS | " << os_info << " |\n";
    file << "\n";

    // ---- Results ----
    file << "## Results\n\n";

    // Group results by case name
    if (results.empty())
    {
        file << "*No benchmark results.*\n\n";
    }
    else
    {
        // Collect unique case names in order of first appearance
        std::vector<std::string> case_names;
        for (const auto& r : results)
        {
            if (std::find(case_names.begin(), case_names.end(), r.name) == case_names.end())
            {
                case_names.push_back(r.name);
            }
        }

        for (const auto& case_name : case_names)
        {
            file << "### " << case_name << "\n\n";

            bool has_serial = false;
            bool has_omp = false;
            bool has_cuda = false;
            for (const auto& r : results)
            {
                if (r.name == case_name)
                {
                    if (r.time_serial_ms >= 0.0) has_serial = true;
                    if (r.time_omp_ms >= 0.0) has_omp = true;
                    if (r.time_cuda_ms >= 0.0) has_cuda = true;
                }
            }

            // Build table header dynamically
            file << "| Size |";
            if (has_serial) file << " CPU Serial (ms) |";
            if (has_omp) file << " CPU OMP (ms) |";
            if (has_cuda) file << " CUDA (ms) | Transfer (ms) |";
            if (has_serial && has_omp) file << " OMP Speedup |";
            if (has_serial && has_cuda) file << " CUDA Speedup |";
            file << "\n";

            file << "|------|";
            if (has_serial) file << "-----------------|";
            if (has_omp) file << "-------------|";
            if (has_cuda) file << "-----------|--------------|";
            if (has_serial && has_omp) file << "-------------|";
            if (has_serial && has_cuda) file << "-------------|";
            file << "\n";

            for (const auto& r : results)
            {
                if (r.name != case_name)
                {
                    continue;
                }

                file << "| " << r.size << " |";
                if (has_serial) file << " " << fmtTime(r.time_serial_ms) << " |";
                if (has_omp) file << " " << fmtTime(r.time_omp_ms) << " |";
                if (has_cuda)
                {
                    file << " " << fmtTime(r.time_cuda_ms) << " |";
                    file << " " << fmtTime(r.time_transfer_ms) << " |";
                }
                if (has_serial && has_omp)
                {
                    file << " " << fmtSpeedup(r.time_serial_ms, r.time_omp_ms) << " |";
                }
                if (has_serial && has_cuda)
                {
                    file << " " << fmtSpeedup(r.time_serial_ms, r.time_cuda_ms) << " |";
                }
                file << "\n";
            }
            file << "\n";
        }
    }

    // ---- Summary ----
    file << "## Summary\n\n";

    file << "- Times are medians from the benchmark harness timed trials.\n";
    file << "- Speedups compare CPU serial time with the accelerated path for the same case and size.\n";
    file << "- Use `--size smoke` for fast build/regression checks; use `--size medium` or `--size large` "
         << "before making CPU/GPU threshold decisions.\n";
    file << "\n";
}

} // namespace plamatrix
