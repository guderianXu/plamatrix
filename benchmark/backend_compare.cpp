#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "backend_scenarios.h"
#include "plamatrix/opencl/iterative_solver.h"
#include "plamatrix/opencl/runtime.h"
#include "plamatrix/vulkan/iterative_solver.h"
#include "plamatrix/vulkan/runtime.h"

using namespace plamatrix;
using namespace plamatrix::benchmark;

namespace
{

    template <typename Solve>
    void runCase(const char* backend,
                 const std::string& device,
                 const BackendFixture& fixture,
                 int convergenceCheckInterval,
                 Solve solve)
    {
        IterativeSolverOptions options;
        options.maxIterations = 200;
        options.relativeTolerance = 1.0e-5;
        options.requireConvergence = true;
        options.convergenceCheckInterval = convergenceCheckInterval;
        DenseMatrix<float, Device::CPU> solution(fixture.matrix.rows(), 1);
        solution.fill(0.0f);
        solution.fill(0.0f);
        const auto coldStart = std::chrono::steady_clock::now();
        const auto coldReport = solve(fixture.matrix, fixture.rhs, solution, options);
        const auto coldEnd = std::chrono::steady_clock::now();
        const double coldMilliseconds = std::chrono::duration<double, std::milli>(coldEnd - coldStart).count();
        for (int warmup = 0; warmup < 2; ++warmup)
        {
            solution.fill(0.0f);
            static_cast<void>(solve(fixture.matrix, fixture.rhs, solution, options));
        }
        std::vector<double> timings;
        for (int trial = 0; trial < 5; ++trial)
        {
            solution.fill(0.0f);
            const auto start = std::chrono::steady_clock::now();
            const auto report = solve(fixture.matrix, fixture.rhs, solution, options);
            const auto end = std::chrono::steady_clock::now();
            const double milliseconds = std::chrono::duration<double, std::milli>(end - start).count();
            timings.push_back(milliseconds);
            if (trial == 4)
            {
                std::sort(timings.begin(), timings.end());
                const char* spmvKernel =
                    std::string(backend) == "vulkan" ? (report.subgroupSpmv ? "subgroup" : "scalar") : "n/a";
                std::cout << fixture.scenario << ',' << backend << ',' << device << ',' << fixture.matrix.rows() << ','
                          << fixture.matrix.nnz() << ',' << report.iterations << ',' << std::setprecision(9)
                          << report.initialResidual << ',' << report.finalResidual << ',' << report.commandSubmissions
                          << ',' << report.gpuMilliseconds << ',' << report.barrierCount << ',' << spmvKernel << ','
                          << coldReport.descriptorSetAllocations << ',' << coldReport.iterations << ','
                          << coldMilliseconds << ',' << timings[timings.size() / 2] << '\n';
            }
        }
    }

} // namespace

int main(int argc, char** argv)
{
    std::string scenario = "tridiagonal";
    std::vector<Index> sizes;
    std::string matrixMarketPath;
    bool runSuite = false;
    int convergenceCheckInterval = 1;
    for (int index = 1; index < argc; ++index)
    {
        const std::string argument = argv[index];
        if (argument == "--help")
        {
            std::cout << "usage: plamatrix_backend_compare [SIZE]\n"
                         "       plamatrix_backend_compare --case NAME --sizes N,N,...\n"
                         "       plamatrix_backend_compare --suite\n"
                         "       plamatrix_backend_compare --matrix-market PATH\n"
                         "       --convergence-interval N batches PCG convergence checks.\n"
                         "cases: tridiagonal, stencil2d, stencil3d, ba_schur, mvs_visibility\n"
                         "--suite runs all representative cases; BA sizes are camera counts, other 2D/3D sizes are "
                         "side lengths.\n";
            return 0;
        }
        if (argument == "--suite")
        {
            runSuite = true;
            continue;
        }
        if (argument == "--case" || argument == "--sizes" || argument == "--matrix-market" ||
            argument == "--convergence-interval")
        {
            if (++index >= argc)
            {
                std::cerr << argument << " requires a value\n";
                return 2;
            }
            const std::string value = argv[index];
            if (argument == "--case")
            {
                scenario = value;
            }
            else if (argument == "--matrix-market")
            {
                matrixMarketPath = value;
            }
            else if (argument == "--convergence-interval")
            {
                convergenceCheckInterval = std::stoi(value);
                if (convergenceCheckInterval <= 0)
                {
                    std::cerr << "convergence interval must be positive\n";
                    return 2;
                }
            }
            else
            {
                std::stringstream values(value);
                std::string token;
                while (std::getline(values, token, ','))
                {
                    const Index size = static_cast<Index>(std::stoll(token));
                    if (size <= 0)
                    {
                        std::cerr << "sizes must be positive\n";
                        return 2;
                    }
                    sizes.push_back(size);
                }
            }
            continue;
        }
        if (!argument.empty() && argument.front() != '-')
        {
            const Index size = static_cast<Index>(std::stoll(argument));
            if (size <= 0)
            {
                std::cerr << "size must be positive\n";
                return 2;
            }
            sizes.push_back(size);
            continue;
        }
        std::cerr << "unknown argument: " << argument << '\n';
        return 2;
    }

    if (!matrixMarketPath.empty() && runSuite)
    {
        std::cerr << "--matrix-market cannot be combined with --suite\n";
        return 2;
    }
    std::cout << "scenario,backend,device,dimension,nnz,iterations,initial_residual,final_residual,command_submissions,"
                 "gpu_ms,barrier_count,spmv_kernel,cold_descriptor_set_allocations,cold_iterations,cold_ms,"
                 "warm_median_ms\n";
    auto runFixture = [convergenceCheckInterval](const BackendFixture& fixture)
    {
        if (opencl::hasUsableOpenClDevice())
        {
            runCase("opencl",
                    opencl::selectedOpenClDeviceName(),
                    fixture,
                    convergenceCheckInterval,
                    [](const auto& matrix, const auto& rhs, auto& solution, const auto& options)
                    { return opencl::pcg(matrix, rhs, solution, options); });
        }
        else
        {
            std::cerr << "opencl,skipped,no usable OpenCL device\n";
        }
        if (vulkan::hasUsableVulkanDevice())
        {
            runCase("vulkan",
                    vulkan::selectedVulkanDeviceName(),
                    fixture,
                    convergenceCheckInterval,
                    [](const auto& matrix, const auto& rhs, auto& solution, const auto& options)
                    { return vulkan::pcg(matrix, rhs, solution, options); });
        }
        else
        {
            std::cerr << "vulkan,skipped,no usable Vulkan device\n";
        }
    };
    if (!matrixMarketPath.empty())
    {
        runFixture(loadMatrixMarketFixture(matrixMarketPath));
        return 0;
    }
    if (runSuite)
    {
        for (const auto& [suiteScenario, suiteSize] : backendSuite())
            runFixture(makeBackendFixture(suiteScenario, suiteSize));
        return 0;
    }
    if (sizes.empty())
        sizes.push_back(4096);
    for (const Index size : sizes)
    {
        runFixture(makeBackendFixture(scenario, size));
    }
    return 0;
}
