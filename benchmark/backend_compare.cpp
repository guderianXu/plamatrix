#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "plamatrix/opencl/iterative_solver.h"
#include "plamatrix/opencl/runtime.h"
#include "plamatrix/vulkan/iterative_solver.h"
#include "plamatrix/vulkan/runtime.h"

using namespace plamatrix;

namespace
{

    struct Fixture
    {
        CSRMatrix<float, Device::CPU> matrix;
        DenseMatrix<float, Device::CPU> rhs;

        explicit Fixture(Index size) : matrix(size, size, size == 1 ? 1 : 3 * size - 2), rhs(size, 1)
        {
            std::vector<Index> rows(static_cast<std::size_t>(size + 1));
            std::vector<Index> columns;
            std::vector<float> values;
            columns.reserve(static_cast<std::size_t>(matrix.nnz()));
            values.reserve(static_cast<std::size_t>(matrix.nnz()));
            for (Index row = 0; row < size; ++row)
            {
                rows[static_cast<std::size_t>(row)] = static_cast<Index>(columns.size());
                if (row > 0)
                {
                    columns.push_back(row - 1);
                    values.push_back(-1.0f);
                }
                columns.push_back(row);
                values.push_back(4.0f);
                if (row + 1 < size)
                {
                    columns.push_back(row + 1);
                    values.push_back(-1.0f);
                }
                rhs(row, 0) = 1.0f;
            }
            rows.back() = static_cast<Index>(columns.size());
            std::copy(rows.begin(), rows.end(), matrix.rowOffsets());
            std::copy(columns.begin(), columns.end(), matrix.colIndices());
            std::copy(values.begin(), values.end(), matrix.values());
            matrix.validateStructure();
        }
    };

    template <typename Solve>
    void runCase(const char* backend, const std::string& device, const Fixture& fixture, Solve solve)
    {
        IterativeSolverOptions options;
        options.maxIterations = 200;
        options.relativeTolerance = 1.0e-5;
        options.requireConvergence = true;
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
                std::cout << backend << ',' << device << ',' << fixture.matrix.rows() << ',' << report.iterations << ','
                          << std::setprecision(9) << report.initialResidual << ',' << report.finalResidual << ','
                          << report.commandSubmissions << ',' << coldReport.iterations << ',' << coldMilliseconds << ','
                          << timings[timings.size() / 2] << '\n';
            }
        }
    }

} // namespace

int main(int argc, char** argv)
{
    const Index size = argc > 1 ? static_cast<Index>(std::stoll(argv[1])) : 4096;
    if (size <= 0)
    {
        std::cerr << "size must be positive\n";
        return 2;
    }
    const Fixture fixture(size);
    std::cout << "backend,device,size,iterations,initial_residual,final_residual,command_submissions,cold_iterations,"
                 "cold_ms,warm_median_ms\n";
    if (opencl::hasUsableOpenClDevice())
    {
        runCase("opencl",
                opencl::selectedOpenClDeviceName(),
                fixture,
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
                [](const auto& matrix, const auto& rhs, auto& solution, const auto& options)
                { return vulkan::pcg(matrix, rhs, solution, options); });
    }
    else
    {
        std::cerr << "vulkan,skipped,no usable Vulkan device\n";
    }
    return 0;
}
