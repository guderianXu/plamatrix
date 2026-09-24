#include <algorithm>
#include <array>
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

#include "plamatrix/internal/opencl/runtime.h"
#include "plamatrix/internal/optimization/block_schur.h"
#include "plamatrix/internal/vulkan/runtime.h"

namespace
{

    using plamatrix::internal::BlockNormalEquations;
    using plamatrix::Index;
    using plamatrix::internal::SchurComplementLinearBackend;
    using plamatrix::internal::SchurComplementSolverOptions;
    using plamatrix::internal::SchurComplementSolverReport;
    using plamatrix::internal::SchurComplementSolverWorkspace;

    constexpr Index kCameraSize = 9;
    constexpr Index kPointSize = 3;

    std::size_t parsePositive(const char* value, const char* name)
    {
        if (value[0] == '-')
            throw std::invalid_argument(std::string(name) + " must be a positive integer");
        std::size_t consumed = 0;
        const unsigned long long parsed = std::stoull(value, &consumed);
        if (consumed != std::string(value).size() || parsed == 0 || parsed > std::numeric_limits<std::size_t>::max())
            throw std::invalid_argument(std::string(name) + " must be a positive integer");
        return static_cast<std::size_t>(parsed);
    }

    double median(std::vector<double> values)
    {
        std::sort(values.begin(), values.end());
        return values[values.size() / 2];
    }

    BlockNormalEquations<float> makeBaEquations(Index camera_count, Index point_count, Index observations_per_point)
    {
        BlockNormalEquations<float> equations(camera_count, point_count, kCameraSize, kPointSize);
        for (Index point = 0; point < point_count; ++point)
        {
            const Index first_camera = (point * 7 + point / 11) % camera_count;
            for (Index observation = 0; observation < observations_per_point; ++observation)
            {
                const Index camera = (first_camera + observation * 13) % camera_count;
                std::array<float, 2 * kCameraSize> camera_jacobian{};
                std::array<float, 2 * kPointSize> point_jacobian{};
                for (Index row = 0; row < 2; ++row)
                {
                    for (Index column = 0; column < kCameraSize; ++column)
                    {
                        const int pattern = static_cast<int>((camera * 3 + point * 5 + row * 7 + column) % 17) - 8;
                        camera_jacobian[static_cast<std::size_t>(row * kCameraSize + column)] =
                            static_cast<float>(pattern) * 0.0125f;
                    }
                    camera_jacobian[static_cast<std::size_t>(row * kCameraSize + row)] += 0.8f;
                    for (Index column = 0; column < kPointSize; ++column)
                    {
                        const int pattern = static_cast<int>((point * 3 + camera + row * 2 + column) % 11) - 5;
                        point_jacobian[static_cast<std::size_t>(row * kPointSize + column)] =
                            static_cast<float>(pattern) * 0.02f;
                    }
                    point_jacobian[static_cast<std::size_t>(row * kPointSize + row)] += 0.65f;
                }
                const std::array<float, 2> residual{{
                    static_cast<float>(static_cast<int>((point + camera) % 19) - 9) * 0.01f,
                    static_cast<float>(static_cast<int>((point * 2 + camera) % 23) - 11) * 0.008f,
                }};
                equations.addResidualBlock(
                    camera, point, camera_jacobian.data(), point_jacobian.data(), residual.data(), 2);
            }
        }

        std::array<float, kCameraSize * kCameraSize> camera_prior{};
        std::array<float, kCameraSize> camera_residual{};
        for (Index diagonal = 0; diagonal < kCameraSize; ++diagonal)
            camera_prior[static_cast<std::size_t>(diagonal * kCameraSize + diagonal)] = 0.1f;
        for (Index camera = 0; camera < camera_count; ++camera)
        {
            camera_residual.fill(0.0f);
            camera_residual[static_cast<std::size_t>(camera % kCameraSize)] =
                0.002f * static_cast<float>(1 + camera % 5);
            equations.addPrimaryResidualBlock(camera, camera_prior.data(), camera_residual.data(), kCameraSize);
        }
        return equations;
    }

    struct Result
    {
        std::vector<float> primary;
        std::vector<float> eliminated;
        double totalMilliseconds = 0.0;
        double assemblyMilliseconds = 0.0;
        double solveMilliseconds = 0.0;
        double gpuMilliseconds = 0.0;
        int iterations = 0;
        std::uint32_t submissions = 0;
        std::uint32_t recordings = 0;
        bool blockSpmv = false;
        bool deviceBlockJacobi = false;
    };

    Result
    measure(const BlockNormalEquations<float>& equations, SchurComplementLinearBackend backend, std::size_t trials)
    {
        SchurComplementSolverOptions<float> options;
        options.linearBackend = backend;
        options.maxIterations = 150;
        options.relativeTolerance = 2.0e-5f;
        options.absoluteTolerance = 1.0e-7f;
        options.useMixedPrecision = backend == SchurComplementLinearBackend::Vulkan;
        SchurComplementSolverWorkspace<float> workspace;
        Result result;
        auto solve = [&]()
        {
            result.primary.clear();
            result.eliminated.clear();
            return plamatrix::internal::solveDampedSchurComplement(
                equations, 0.1f, options, workspace, &result.primary, &result.eliminated);
        };
        const auto warm = solve();
        if (!warm.converged)
            throw std::runtime_error("warm Schur solve did not converge: " + warm.message);

        std::vector<double> total_times;
        std::vector<double> assembly_times;
        std::vector<double> solve_times;
        std::vector<double> gpu_times;
        total_times.reserve(trials);
        assembly_times.reserve(trials);
        solve_times.reserve(trials);
        gpu_times.reserve(trials);
        for (std::size_t trial = 0; trial < trials; ++trial)
        {
            const auto start = std::chrono::steady_clock::now();
            const auto report = solve();
            const double milliseconds =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            if (!report.converged)
                throw std::runtime_error("measured Schur solve did not converge: " + report.message);
            total_times.push_back(milliseconds);
            assembly_times.push_back(report.schurAssemblySeconds * 1000.0);
            solve_times.push_back(report.linearSolveSeconds * 1000.0);
            gpu_times.push_back(report.gpuMilliseconds);
            result.iterations = report.iterations;
            result.submissions = report.commandSubmissions;
            result.recordings = report.commandBufferRecordings;
            result.blockSpmv = report.blockSpmvUsed;
            result.deviceBlockJacobi = report.deviceBlockJacobiUsed;
        }
        result.totalMilliseconds = median(total_times);
        result.assemblyMilliseconds = median(assembly_times);
        result.solveMilliseconds = median(solve_times);
        result.gpuMilliseconds = median(gpu_times);
        return result;
    }

    float maximumDifference(const Result& left, const Result& right)
    {
        float result = 0.0f;
        for (std::size_t index = 0; index < left.primary.size(); ++index)
            result = std::max(result, std::abs(left.primary[index] - right.primary[index]));
        for (std::size_t index = 0; index < left.eliminated.size(); ++index)
            result = std::max(result, std::abs(left.eliminated[index] - right.eliminated[index]));
        return result;
    }

    void printResult(const char* name, const Result& result, float error)
    {
        std::cout << name << "_total_median_ms=" << result.totalMilliseconds << '\n'
                  << name << "_schur_assembly_median_ms=" << result.assemblyMilliseconds << '\n'
                  << name << "_linear_solve_median_ms=" << result.solveMilliseconds << '\n'
                  << name << "_gpu_median_ms=" << result.gpuMilliseconds << '\n'
                  << name << "_iterations=" << result.iterations << '\n'
                  << name << "_command_submissions=" << result.submissions << '\n'
                  << name << "_command_buffer_recordings=" << result.recordings << '\n'
                  << name << "_block_spmv=" << (result.blockSpmv ? 1 : 0) << '\n'
                  << name << "_device_block_jacobi=" << (result.deviceBlockJacobi ? 1 : 0) << '\n'
                  << name << "_maximum_absolute_error=" << error << '\n';
    }

} // namespace

int main(int argc, char** argv)
{
    try
    {
        const std::size_t camera_count = argc > 1 ? parsePositive(argv[1], "camera count") : 256;
        const std::size_t point_count = argc > 2 ? parsePositive(argv[2], "point count") : 2048;
        const std::size_t observations_per_point = argc > 3 ? parsePositive(argv[3], "observations per point") : 4;
        const std::size_t trials = argc > 4 ? parsePositive(argv[4], "trial count") : 7;
        if (argc > 5 || camera_count > static_cast<std::size_t>(std::numeric_limits<Index>::max()) ||
            point_count > static_cast<std::size_t>(std::numeric_limits<Index>::max()) ||
            observations_per_point > camera_count ||
            observations_per_point > static_cast<std::size_t>(std::numeric_limits<Index>::max()))
        {
            throw std::invalid_argument(
                "usage: plamatrix_schur_backend_benchmark [cameras] [points] [observations-per-point] [trials]");
        }
        const auto equations = makeBaEquations(static_cast<Index>(camera_count),
                                               static_cast<Index>(point_count),
                                               static_cast<Index>(observations_per_point));
        const auto cpu = measure(equations, SchurComplementLinearBackend::Cpu, trials);
        std::cout << std::fixed << std::setprecision(6) << "cameras=" << camera_count << '\n'
                  << "points=" << point_count << '\n'
                  << "observations_per_point=" << observations_per_point << '\n'
                  << "trials=" << trials << '\n';
        printResult("cpu", cpu, 0.0f);

        if (plamatrix::internal::opencl::hasUsableOpenClDevice())
        {
            const auto opencl = measure(equations, SchurComplementLinearBackend::OpenCl, trials);
            printResult("opencl", opencl, maximumDifference(cpu, opencl));
        }
        else
        {
            std::cout << "opencl_status=unavailable\n";
        }

        if (plamatrix::internal::vulkan::selectedVulkanCooperativeMatrixCapabilities().enabled)
        {
            const auto vulkan = measure(equations, SchurComplementLinearBackend::Vulkan, trials);
            printResult("vulkan", vulkan, maximumDifference(cpu, vulkan));
        }
        else
        {
            std::cout << "vulkan_status=cooperative_matrix_unavailable\n";
        }
        return EXIT_SUCCESS;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Schur backend benchmark failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
