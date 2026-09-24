#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "plamatrix/internal/vulkan/cooperative_matrix.h"
#include "plamatrix/internal/vulkan/runtime.h"

namespace
{

    std::size_t parsePositive(const char* value, const char* name)
    {
        if (value[0] == '-')
        {
            throw std::invalid_argument(std::string(name) + " must be a positive integer");
        }
        std::size_t consumed = 0;
        const unsigned long long parsed = std::stoull(value, &consumed);
        if (consumed != std::string(value).size() || parsed == 0 || parsed > std::numeric_limits<std::size_t>::max())
        {
            throw std::invalid_argument(std::string(name) + " must be a positive integer");
        }
        return static_cast<std::size_t>(parsed);
    }

    double median(std::vector<double> values)
    {
        std::sort(values.begin(), values.end());
        return values[values.size() / 2];
    }

    template <typename Function> double measureMilliseconds(Function&& function)
    {
        const auto start = std::chrono::steady_clock::now();
        function();
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    }

} // namespace

int main(int argc, char** argv)
{
    try
    {
        const std::size_t batch_count = argc > 1 ? parsePositive(argv[1], "batch count") : 4096;
        const std::size_t trials = argc > 2 ? parsePositive(argv[2], "trial count") : 11;
        if (argc > 3)
        {
            throw std::invalid_argument("usage: plamatrix_cooperative_matrix_benchmark [batch-count] [trials]");
        }

        const auto capabilities = plamatrix::internal::vulkan::selectedVulkanCooperativeMatrixCapabilities();
        if (!capabilities.enabled)
        {
            std::cerr << "Selected Vulkan device/build does not support the required cooperative-matrix path\n";
            return 2;
        }
        if (batch_count > std::numeric_limits<std::size_t>::max() / 256)
        {
            throw std::overflow_error("batch count is too large");
        }
        std::vector<float> left(batch_count * 256);
        std::vector<float> right(batch_count * 256);
        for (std::size_t index = 0; index < left.size(); ++index)
        {
            left[index] = static_cast<float>(static_cast<int>(index % 31) - 15) * 0.037f;
            right[index] = static_cast<float>(static_cast<int>(index % 23) - 11) * 0.041f;
        }

        plamatrix::internal::vulkan::BatchedMatrixMultiplyOptions mixed_options;
        mixed_options.allowMixedPrecision = true;
        mixed_options.requireCooperativeMatrix = true;
        plamatrix::internal::vulkan::BatchedMatrixMultiplyReport warm_report;
        auto mixed_output =
            plamatrix::internal::vulkan::batchedMultiply16x16(left, right, batch_count, mixed_options, &warm_report);
        const auto cpu_reference = plamatrix::internal::vulkan::batchedMultiply16x16(left, right, batch_count);
        float maximum_absolute_error = 0.0f;
        for (std::size_t index = 0; index < mixed_output.size(); ++index)
        {
            maximum_absolute_error =
                std::max(maximum_absolute_error, std::abs(mixed_output[index] - cpu_reference[index]));
        }

        std::vector<double> cpu_times;
        std::vector<double> vulkan_times;
        std::vector<double> device_times;
        cpu_times.reserve(trials);
        vulkan_times.reserve(trials);
        device_times.reserve(trials);
        double checksum = 0.0;
        for (std::size_t trial = 0; trial < trials; ++trial)
        {
            cpu_times.push_back(measureMilliseconds(
                [&]
                {
                    const auto output = plamatrix::internal::vulkan::batchedMultiply16x16(left, right, batch_count);
                    checksum += output[trial % output.size()];
                }));
            plamatrix::internal::vulkan::BatchedMatrixMultiplyReport report;
            vulkan_times.push_back(measureMilliseconds(
                [&]
                {
                    const auto output =
                        plamatrix::internal::vulkan::batchedMultiply16x16(left, right, batch_count, mixed_options, &report);
                    checksum += output[trial % output.size()];
                }));
            device_times.push_back(report.gpuMilliseconds);
        }

        std::cout << "device=" << warm_report.deviceName << '\n'
                  << "tile=" << capabilities.mSize << 'x' << capabilities.nSize << 'x' << capabilities.kSize << '\n'
                  << "batch_count=" << batch_count << '\n'
                  << "trials=" << trials << '\n'
                  << std::fixed << std::setprecision(6) << "cpu_fp32_median_ms=" << median(cpu_times) << '\n'
                  << "vulkan_mixed_e2e_median_ms=" << median(vulkan_times) << '\n'
                  << "vulkan_gpu_median_ms=" << median(device_times) << '\n'
                  << "maximum_absolute_error=" << maximum_absolute_error << '\n'
                  << "checksum=" << checksum << '\n';
        return EXIT_SUCCESS;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Cooperative-matrix benchmark failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
