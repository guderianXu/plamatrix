#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "plamatrix/dense/matrix.h"
#include "plamatrix/internal/core/execution_context.h"
#include "plamatrix/internal/dense/auto_backend.h"

namespace
{
    using plamatrix::Index;
    using plamatrix::MatrixXf;
    using plamatrix::internal::Backend;
    using plamatrix::internal::ExecutionPolicy;
    using plamatrix::internal::ScopedExecutionPolicy;

    const char* backendName(Backend backend)
    {
        switch (backend)
        {
        case Backend::Cpu:
            return "cpu";
        case Backend::Cuda:
            return "cuda";
        case Backend::Vulkan:
            return "vulkan";
        case Backend::OpenCl:
            return "opencl";
        }
        return "unknown";
    }

    Backend parseBackend(const std::string& value)
    {
        if (value == "cpu")
            return Backend::Cpu;
        if (value == "cuda")
            return Backend::Cuda;
        if (value == "vulkan")
            return Backend::Vulkan;
        if (value == "opencl")
            return Backend::OpenCl;
        throw std::invalid_argument("Unknown backend: " + value);
    }

    std::vector<Index> parseSizes(const std::string& value)
    {
        std::vector<Index> result;
        std::stringstream stream(value);
        std::string token;
        while (std::getline(stream, token, ','))
        {
            const long long parsed = std::stoll(token);
            if (parsed <= 0 || parsed > std::numeric_limits<Index>::max())
                throw std::invalid_argument("Sizes must be positive and fit PlaMatrix Index");
            result.push_back(static_cast<Index>(parsed));
        }
        if (result.empty())
            throw std::invalid_argument("At least one size is required");
        return result;
    }

    std::vector<Backend> parseBackends(const std::string& value)
    {
        std::vector<Backend> result;
        std::stringstream stream(value);
        std::string token;
        while (std::getline(stream, token, ','))
            result.push_back(parseBackend(token));
        if (result.empty())
            throw std::invalid_argument("At least one backend is required");
        return result;
    }

    double median(std::vector<double> values)
    {
        std::sort(values.begin(), values.end());
        return values[values.size() / 2];
    }

    bool available(Backend backend)
    {
        return backend == Backend::Cpu || plamatrix::internal::detail::GpuOps<float>::available(backend);
    }

    struct BenchmarkResult
    {
        double firstMilliseconds = 0.0;
        double residentMilliseconds = 0.0;
    };

    BenchmarkResult run(Backend backend, Index size, int iterations, double* checksum)
    {
        const float input = 1.0f / std::sqrt(static_cast<float>(size));
        MatrixXf left = MatrixXf::Constant(size, size, input);
        MatrixXf right = MatrixXf::Constant(size, size, input);
        const ExecutionPolicy policy =
            backend == Backend::Cpu ? ExecutionPolicy::CpuOnly : ExecutionPolicy::GpuRequired;
        ScopedExecutionPolicy selected(policy, backend == Backend::Cpu ? Backend::Cuda : backend);

        const auto first_start = std::chrono::steady_clock::now();
        MatrixXf output = left * right;
        const auto first_end = std::chrono::steady_clock::now();
        output.noalias() = left * right;

        std::vector<double> trials;
        trials.reserve(5);
        for (int trial = 0; trial < 5; ++trial)
        {
            const auto start = std::chrono::steady_clock::now();
            for (int iteration = 0; iteration < iterations; ++iteration)
                output.noalias() = left * right;
            const auto end = std::chrono::steady_clock::now();
            trials.push_back(std::chrono::duration<double, std::milli>(end - start).count() /
                             static_cast<double>(iterations));
        }
        *checksum += static_cast<const MatrixXf&>(output)(size - 1, size - 1);
        return {std::chrono::duration<double, std::milli>(first_end - first_start).count(), median(std::move(trials))};
    }
} // namespace

int main(int argc, char** argv)
{
    try
    {
        std::vector<Index> sizes{128, 256, 512};
        std::vector<Backend> backends{Backend::Cpu, Backend::Cuda, Backend::Vulkan, Backend::OpenCl};
        int iterations = 3;
        for (int index = 1; index < argc; ++index)
        {
            const std::string argument = argv[index];
            if (argument == "--help")
            {
                std::cout << "Usage: plamatrix_dense_backend_benchmark [--sizes N,N,...] "
                             "[--backends cpu,cuda,vulkan,opencl] [--iterations N]\n";
                return 0;
            }
            if (++index >= argc)
                throw std::invalid_argument("Missing value for " + argument);
            const std::string value = argv[index];
            if (argument == "--sizes")
                sizes = parseSizes(value);
            else if (argument == "--backends")
                backends = parseBackends(value);
            else if (argument == "--iterations")
            {
                iterations = std::stoi(value);
                if (iterations <= 0)
                    throw std::invalid_argument("Iterations must be positive");
            }
            else
                throw std::invalid_argument("Unknown option: " + argument);
        }

        for (const Backend backend : backends)
        {
            if (available(backend))
            {
                double warmup_checksum = 0.0;
                static_cast<void>(run(backend, 16, 1, &warmup_checksum));
            }
        }

        double checksum = 0.0;
        std::cout << "backend,size,first_ms,resident_median_ms,resident_gflops,status\n";
        for (const Index size : sizes)
        {
            for (const Backend backend : backends)
            {
                if (!available(backend))
                {
                    std::cout << backendName(backend) << ',' << size << ",0,0,0,unavailable\n";
                    continue;
                }
                const BenchmarkResult result = run(backend, size, iterations, &checksum);
                const double operations = 2.0 * static_cast<double>(size) * static_cast<double>(size) * size;
                const double gflops = operations / (result.residentMilliseconds * 1.0e6);
                std::cout << backendName(backend) << ',' << size << ',' << result.firstMilliseconds << ','
                          << result.residentMilliseconds << ',' << gflops << ",ok\n";
            }
        }
        std::cerr << "checksum=" << checksum << '\n';
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Dense backend benchmark: " << error.what() << '\n';
        return 1;
    }
}
