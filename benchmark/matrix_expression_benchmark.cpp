#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "plamatrix/dense/matrix.h"

namespace
{
    volatile double checksum = 0.0;

    long parsePositive(const char* text, const char* name)
    {
        const std::string value(text);
        std::size_t consumed = 0;
        const long parsed = std::stol(value, &consumed);
        if (parsed <= 0 || consumed != value.size())
        {
            throw std::invalid_argument(std::string(name) + " must be a positive integer");
        }
        return parsed;
    }

    double measure(bool fused,
                   int iterations,
                   const plamatrix::MatrixXd& a,
                   const plamatrix::MatrixXd& b,
                   const plamatrix::MatrixXd& c,
                   const plamatrix::MatrixXd& d)
    {
        const auto start = std::chrono::steady_clock::now();
        for (int iteration = 0; iteration < iterations; ++iteration)
        {
            plamatrix::MatrixXd result = fused ? plamatrix::MatrixXd(a + b + c + d) : ((a + b).eval() + c).eval() + d;
            checksum += result(iteration % result.rows(), iteration % result.cols());
        }
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    }

    double measureArray(bool fused,
                        int iterations,
                        const plamatrix::MatrixXd& a,
                        const plamatrix::MatrixXd& b,
                        const plamatrix::MatrixXd& c,
                        const plamatrix::MatrixXd& d)
    {
        const auto start = std::chrono::steady_clock::now();
        for (int iteration = 0; iteration < iterations; ++iteration)
        {
            plamatrix::MatrixXd result;
            if (fused)
            {
                result = (a + b).array().square() + c.array() * d.array();
            }
            else
            {
                const plamatrix::MatrixXd sum = (a + b).eval();
                const plamatrix::MatrixXd square = sum.cwiseProduct(sum);
                const plamatrix::MatrixXd product = c.cwiseProduct(d);
                result = (square + product).eval();
            }
            checksum += result(iteration % result.rows(), iteration % result.cols());
        }
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    }

    double median(std::vector<double> values)
    {
        std::sort(values.begin(), values.end());
        return values[values.size() / 2];
    }
} // namespace

int main(int argc, char** argv)
{
    try
    {
        int size = 512;
        int iterations = 80;
        for (int index = 1; index < argc; ++index)
        {
            const std::string argument(argv[index]);
            if (argument == "--help")
            {
                std::cout << "Usage: plamatrix_expression_benchmark [--size N] [--iterations N]\n";
                return 0;
            }
            if ((argument == "--size" || argument == "--iterations") && index + 1 < argc)
            {
                const long value = parsePositive(argv[++index], argument.c_str());
                if (value > std::numeric_limits<int>::max())
                {
                    throw std::invalid_argument(argument + " exceeds the supported integer range");
                }
                (argument == "--size" ? size : iterations) = static_cast<int>(value);
                continue;
            }
            throw std::invalid_argument("Unknown or incomplete option: " + argument);
        }

        plamatrix::internal::ScopedExecutionPolicy cpu_only(plamatrix::internal::ExecutionPolicy::CpuOnly);
        const auto a = plamatrix::MatrixXd::Constant(size, size, 1.0);
        const auto b = plamatrix::MatrixXd::Constant(size, size, 2.0);
        const auto c = plamatrix::MatrixXd::Constant(size, size, 3.0);
        const auto d = plamatrix::MatrixXd::Constant(size, size, 4.0);
        measure(true, 2, a, b, c, d);
        measure(false, 2, a, b, c, d);
        measureArray(true, 2, a, b, c, d);
        measureArray(false, 2, a, b, c, d);
        std::vector<double> fused_times;
        std::vector<double> eager_times;
        std::vector<double> array_fused_times;
        std::vector<double> array_eager_times;
        for (int trial = 0; trial < 3; ++trial)
        {
            fused_times.push_back(measure(true, iterations, a, b, c, d));
            eager_times.push_back(measure(false, iterations, a, b, c, d));
            array_fused_times.push_back(measureArray(true, iterations, a, b, c, d));
            array_eager_times.push_back(measureArray(false, iterations, a, b, c, d));
        }
        std::cout << "size=" << size << " iterations=" << iterations << " fused_median_ms=" << median(fused_times)
                  << " eager_median_ms=" << median(eager_times)
                  << " array_fused_median_ms=" << median(array_fused_times)
                  << " array_eager_median_ms=" << median(array_eager_times) << " checksum=" << checksum << '\n';
    }
    catch (const std::exception& error)
    {
        std::cerr << "Expression benchmark: " << error.what() << '\n';
        return 1;
    }
}
