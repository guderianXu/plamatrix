#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <plamatrix/plamatrix.h>
#include <plamatrix/internal/backend.h>

namespace plamatrix::internal
{

    namespace
    {

        using Clock = std::chrono::steady_clock;
        using EigenMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

        volatile double benchmark_sink = 0.0;

        struct ComparisonResult
        {
            std::string operation;
            Index size = 0;
            double plamatrix_ms = 0.0;
            double eigen_ms = 0.0;
            double max_abs_error = 0.0;
        };

        struct Options
        {
            Index size = 256;
            int warmup = 3;
            int trials = 11;
            std::string operation = "all";
            std::string output;
        };

        DenseStorage<double, Device::CPU> makeDense(Index rows, Index cols, double diagonal = 0.0)
        {
            DenseStorage<double, Device::CPU> matrix(rows, cols);
            for (Index col = 0; col < cols; ++col)
            {
                for (Index row = 0; row < rows; ++row)
                {
                    const int seed = static_cast<int>((row * 19 + col * 31 + 7) % 43) - 21;
                    matrix(row, col) = static_cast<double>(seed) / 17.0;
                }
            }
            for (Index index = 0; index < std::min(rows, cols); ++index)
            {
                matrix(index, index) += diagonal;
            }
            return matrix;
        }

        Eigen::Map<const EigenMatrix> mapEigen(const DenseStorage<double, Device::CPU>& matrix)
        {
            return Eigen::Map<const EigenMatrix>(matrix.data(), matrix.rows(), matrix.cols());
        }

        plamatrix::MatrixXd makePublicDense(const DenseStorage<double, Device::CPU>& source)
        {
            plamatrix::MatrixXd result(source.rows(), source.cols());
            std::copy_n(source.data(), source.size(), result.data());
            return result;
        }

        double medianMilliseconds(const std::function<void()>& operation, int warmup, int trials)
        {
            for (int index = 0; index < warmup; ++index)
            {
                operation();
            }
            std::vector<double> samples;
            samples.reserve(static_cast<std::size_t>(trials));
            for (int index = 0; index < trials; ++index)
            {
                const auto start = Clock::now();
                operation();
                const auto stop = Clock::now();
                samples.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
            }
            std::sort(samples.begin(), samples.end());
            return samples[samples.size() / 2];
        }

        double maxAbsError(const DenseStorage<double, Device::CPU>& actual, const EigenMatrix& expected)
        {
            if (actual.rows() != expected.rows() || actual.cols() != expected.cols())
            {
                throw std::runtime_error("comparison result dimensions differ");
            }
            double error = 0.0;
            for (Index col = 0; col < actual.cols(); ++col)
            {
                for (Index row = 0; row < actual.rows(); ++row)
                {
                    error = std::max(error, std::abs(actual(row, col) - expected(row, col)));
                }
            }
            return error;
        }

        double maxAbsError(const plamatrix::MatrixXd& actual, const EigenMatrix& expected)
        {
            if (actual.rows() != expected.rows() || actual.cols() != expected.cols())
            {
                throw std::runtime_error("comparison result dimensions differ");
            }
            double error = 0.0;
            for (Index col = 0; col < actual.cols(); ++col)
            {
                for (Index row = 0; row < actual.rows(); ++row)
                {
                    error = std::max(error, std::abs(actual(row, col) - expected(row, col)));
                }
            }
            return error;
        }

        ComparisonResult compareGemm(const Options& options)
        {
            auto left = makeDense(options.size, options.size);
            auto right = makeDense(options.size, options.size);
            const auto eigen_left = mapEigen(left);
            const auto eigen_right = mapEigen(right);

            ComparisonResult result{"gemm", options.size};
            result.plamatrix_ms = medianMilliseconds(
                [&]()
                {
                    auto output = gemm(left, right);
                    benchmark_sink += output(0, 0);
                },
                options.warmup,
                options.trials);
            result.eigen_ms = medianMilliseconds(
                [&]()
                {
                    EigenMatrix output = (eigen_left * eigen_right).eval();
                    benchmark_sink += output(0, 0);
                },
                options.warmup,
                options.trials);

            auto actual = gemm(left, right);
            EigenMatrix expected = (eigen_left * eigen_right).eval();
            result.max_abs_error = maxAbsError(actual, expected);
            return result;
        }

        ComparisonResult compareElementwise(const Options& options, bool fused)
        {
            auto left = makeDense(options.size, options.size);
            auto right = makeDense(options.size, options.size);
            const auto public_left = makePublicDense(left);
            const auto public_right = makePublicDense(right);
            const auto eigen_left = mapEigen(left);
            const auto eigen_right = mapEigen(right);

            ComparisonResult result{fused ? "elementwise_chain_fused" : "elementwise_chain", options.size};
            result.plamatrix_ms = medianMilliseconds(
                [&]()
                {
                    if (fused)
                    {
                        const auto output = axpby(1.25, left, 1.25, right);
                        benchmark_sink += output(0, 0);
                    }
                    else
                    {
                        const plamatrix::MatrixXd output = (public_left + public_right) * 1.25;
                        benchmark_sink += output(0, 0);
                    }
                },
                options.warmup,
                options.trials);
            result.eigen_ms = medianMilliseconds(
                [&]()
                {
                    EigenMatrix output = ((eigen_left + eigen_right) * 1.25).eval();
                    benchmark_sink += output(0, 0);
                },
                options.warmup,
                options.trials);

            EigenMatrix expected = ((eigen_left + eigen_right) * 1.25).eval();
            if (fused)
            {
                result.max_abs_error = maxAbsError(axpby(1.25, left, 1.25, right), expected);
            }
            else
            {
                result.max_abs_error = maxAbsError(plamatrix::MatrixXd((public_left + public_right) * 1.25), expected);
            }
            return result;
        }

        ComparisonResult compareLinearCombination(const Options& options, bool fused)
        {
            auto first = makeDense(options.size, options.size);
            auto second = makeDense(options.size, options.size, 0.5);
            auto third = makeDense(options.size, options.size, -0.25);
            const auto public_first = makePublicDense(first);
            const auto public_second = makePublicDense(second);
            const auto public_third = makePublicDense(third);
            const auto eigen_first = mapEigen(first);
            const auto eigen_second = mapEigen(second);
            const auto eigen_third = mapEigen(third);

            ComparisonResult result{fused ? "linear_combination_fused" : "linear_combination_chain", options.size};
            result.plamatrix_ms = medianMilliseconds(
                [&]()
                {
                    if (fused)
                    {
                        const auto output = linearCombination(1.25, first, -0.75, second, 0.5, third);
                        benchmark_sink += output(0, 0);
                    }
                    else
                    {
                        const plamatrix::MatrixXd output =
                            1.25 * public_first - 0.75 * public_second + 0.5 * public_third;
                        benchmark_sink += output(0, 0);
                    }
                },
                options.warmup,
                options.trials);
            result.eigen_ms = medianMilliseconds(
                [&]()
                {
                    EigenMatrix output = (1.25 * eigen_first - 0.75 * eigen_second + 0.5 * eigen_third).eval();
                    benchmark_sink += output(0, 0);
                },
                options.warmup,
                options.trials);

            EigenMatrix expected = (1.25 * eigen_first - 0.75 * eigen_second + 0.5 * eigen_third).eval();
            if (fused)
            {
                result.max_abs_error = maxAbsError(linearCombination(1.25, first, -0.75, second, 0.5, third), expected);
            }
            else
            {
                result.max_abs_error = maxAbsError(
                    plamatrix::MatrixXd(1.25 * public_first - 0.75 * public_second + 0.5 * public_third), expected);
            }
            return result;
        }

        ComparisonResult compareDot(const Options& options, bool fused)
        {
            auto lhs = makeDense(options.size, options.size);
            auto rhs = makeDense(options.size, options.size, 0.5);
            const auto public_lhs = makePublicDense(lhs);
            const auto public_rhs = makePublicDense(rhs);
            const auto eigen_lhs = mapEigen(lhs);
            const auto eigen_rhs = mapEigen(rhs);

            ComparisonResult result{fused ? "dot_fused" : "dot_chain", options.size};
            result.plamatrix_ms = medianMilliseconds(
                [&]()
                {
                    const double output = fused ? dot(lhs, rhs) : (public_lhs.array() * public_rhs.array()).sum();
                    benchmark_sink += output;
                },
                options.warmup,
                options.trials);
            result.eigen_ms = medianMilliseconds(
                [&]() { benchmark_sink += eigen_lhs.cwiseProduct(eigen_rhs).sum(); }, options.warmup, options.trials);

            const double actual = fused ? dot(lhs, rhs) : (public_lhs.array() * public_rhs.array()).sum();
            const double expected = eigen_lhs.cwiseProduct(eigen_rhs).sum();
            result.max_abs_error = std::abs(actual - expected);
            return result;
        }

        ComparisonResult compareReductionChain(const Options& options)
        {
            auto first = makeDense(options.size, options.size);
            auto second = makeDense(options.size, options.size, 0.5);
            auto third = makeDense(options.size, options.size, -0.25);
            auto fourth = makeDense(options.size, options.size, 0.75);
            const auto public_first = makePublicDense(first);
            const auto public_second = makePublicDense(second);
            const auto public_third = makePublicDense(third);
            const auto public_fourth = makePublicDense(fourth);
            const auto eigen_first = mapEigen(first);
            const auto eigen_second = mapEigen(second);
            const auto eigen_third = mapEigen(third);
            const auto eigen_fourth = mapEigen(fourth);

            ComparisonResult result{"reduction_chain", options.size};
            result.plamatrix_ms = medianMilliseconds(
                [&]()
                {
                    benchmark_sink += (1.25 * public_first.array() - public_second.array() * 0.75 +
                                       public_third.array() * public_fourth.array())
                                          .sum();
                },
                options.warmup,
                options.trials);
            result.eigen_ms = medianMilliseconds(
                [&]()
                {
                    benchmark_sink += (1.25 * eigen_first.array() - eigen_second.array() * 0.75 +
                                       eigen_third.array() * eigen_fourth.array())
                                          .sum();
                },
                options.warmup,
                options.trials);

            const double actual = (1.25 * public_first.array() - public_second.array() * 0.75 +
                                   public_third.array() * public_fourth.array())
                                      .sum();
            const double expected =
                (1.25 * eigen_first.array() - eigen_second.array() * 0.75 + eigen_third.array() * eigen_fourth.array())
                    .sum();
            result.max_abs_error = std::abs(actual - expected);
            return result;
        }

        ComparisonResult compareSolve(const Options& options)
        {
            auto coefficients = makeDense(options.size, options.size, static_cast<double>(options.size) * 2.0);
            auto rhs = makeDense(options.size, 4);
            const auto public_coefficients = makePublicDense(coefficients);
            const auto public_rhs = makePublicDense(rhs);
            const auto eigen_coefficients = mapEigen(coefficients);
            const auto eigen_rhs = mapEigen(rhs);

            ComparisonResult result{"partial_pivot_solve", options.size};
            result.plamatrix_ms = medianMilliseconds(
                [&]()
                {
                    const auto output = public_coefficients.partialPivLu().solve(public_rhs);
                    benchmark_sink += output(0, 0);
                },
                options.warmup,
                options.trials);
            result.eigen_ms = medianMilliseconds(
                [&]()
                {
                    EigenMatrix output = eigen_coefficients.partialPivLu().solve(eigen_rhs);
                    benchmark_sink += output(0, 0);
                },
                options.warmup,
                options.trials);

            const auto actual = public_coefficients.partialPivLu().solve(public_rhs);
            EigenMatrix expected = eigen_coefficients.partialPivLu().solve(eigen_rhs);
            result.max_abs_error = maxAbsError(actual, expected);
            return result;
        }

        ComparisonResult compareSpmv(const Options& options)
        {
            std::vector<Index> rows;
            std::vector<Index> cols;
            std::vector<double> values;
            std::vector<Eigen::Triplet<double, Index>> triplets;
            for (Index row = 0; row < options.size; ++row)
            {
                const auto append = [&](Index col, double value)
                {
                    rows.push_back(row);
                    cols.push_back(col);
                    values.push_back(value);
                    triplets.emplace_back(row, col, value);
                };
                append(row, 4.0);
                if (row > 0)
                {
                    append(row - 1, -1.0);
                }
                if (row + 1 < options.size)
                {
                    append(row + 1, -1.0);
                }
            }
            auto sparse = cooToCsr(options.size, options.size, rows, cols, values);
            auto input = makeDense(options.size, 1);
            Eigen::SparseMatrix<double, Eigen::RowMajor, Index> eigen_sparse(options.size, options.size);
            eigen_sparse.setFromTriplets(triplets.begin(), triplets.end());
            const auto eigen_input = mapEigen(input);

            ComparisonResult result{"spmv", options.size};
            result.plamatrix_ms = medianMilliseconds(
                [&]()
                {
                    auto output = spmv(sparse, input);
                    benchmark_sink += output(0, 0);
                },
                options.warmup,
                options.trials);
            result.eigen_ms = medianMilliseconds(
                [&]()
                {
                    EigenMatrix output = eigen_sparse * eigen_input;
                    benchmark_sink += output(0, 0);
                },
                options.warmup,
                options.trials);

            auto actual = spmv(sparse, input);
            EigenMatrix expected = eigen_sparse * eigen_input;
            result.max_abs_error = maxAbsError(actual, expected);
            return result;
        }

        Options parseOptions(int argc, char** argv)
        {
            Options options;
            for (int index = 1; index < argc; ++index)
            {
                const std::string argument = argv[index];
                const auto requireValue = [&]() -> std::string
                {
                    if (++index >= argc)
                    {
                        throw std::invalid_argument(argument + " requires a value");
                    }
                    return argv[index];
                };
                if (argument == "--size")
                {
                    options.size = static_cast<Index>(std::stoll(requireValue()));
                }
                else if (argument == "--warmup")
                {
                    options.warmup = std::stoi(requireValue());
                }
                else if (argument == "--trials")
                {
                    options.trials = std::stoi(requireValue());
                }
                else if (argument == "--case")
                {
                    options.operation = requireValue();
                }
                else if (argument == "--output")
                {
                    options.output = requireValue();
                }
                else if (argument == "--list")
                {
                    std::cout << "gemm\nelementwise_chain\nelementwise_chain_fused\n"
                                 "linear_combination_chain\nlinear_combination_fused\n"
                                 "dot_chain\ndot_fused\nreduction_chain\npartial_pivot_solve\nspmv\n";
                    std::exit(0);
                }
                else
                {
                    throw std::invalid_argument("unknown argument: " + argument);
                }
            }
            if (options.size <= 0 || options.warmup < 0 || options.trials <= 0)
            {
                throw std::invalid_argument("size and trials must be positive; warmup must be non-negative");
            }
            return options;
        }

        void writeResults(std::ostream& output, const std::vector<ComparisonResult>& results)
        {
            output << "case,size,implementation,median_ms,max_abs_error\n";
            output << std::setprecision(10);
            for (const auto& result : results)
            {
                output << result.operation << ',' << result.size << ",plamatrix," << result.plamatrix_ms << ','
                       << result.max_abs_error << '\n';
                output << result.operation << ',' << result.size << ",eigen," << result.eigen_ms << ','
                       << result.max_abs_error << '\n';
            }
        }

    } // namespace

    int runBenchmark(int argc, char** argv)
    {
        try
        {
            const Options options = parseOptions(argc, argv);
            std::vector<ComparisonResult> results;
            const auto requested = [&](const char* name)
            { return options.operation == "all" || options.operation == name; };
            if (requested("gemm"))
            {
                results.push_back(compareGemm(options));
            }
            if (requested("elementwise_chain"))
            {
                results.push_back(compareElementwise(options, false));
            }
            if (requested("elementwise_chain_fused"))
            {
                results.push_back(compareElementwise(options, true));
            }
            if (requested("linear_combination_chain"))
            {
                results.push_back(compareLinearCombination(options, false));
            }
            if (requested("linear_combination_fused"))
            {
                results.push_back(compareLinearCombination(options, true));
            }
            if (requested("dot_chain"))
            {
                results.push_back(compareDot(options, false));
            }
            if (requested("dot_fused"))
            {
                results.push_back(compareDot(options, true));
            }
            if (requested("reduction_chain"))
            {
                results.push_back(compareReductionChain(options));
            }
            if (requested("partial_pivot_solve"))
            {
                results.push_back(compareSolve(options));
            }
            if (requested("spmv"))
            {
                results.push_back(compareSpmv(options));
            }
            if (results.empty())
            {
                throw std::invalid_argument("unknown benchmark case: " + options.operation);
            }
            for (const auto& result : results)
            {
                const double allowed_error = 1.0e-8 * static_cast<double>(std::max<Index>(1, result.size));
                if (!std::isfinite(result.max_abs_error) || result.max_abs_error > allowed_error)
                {
                    throw std::runtime_error(result.operation + " exceeds the Eigen error tolerance");
                }
            }

            writeResults(std::cout, results);
            if (!options.output.empty())
            {
                std::ofstream file(options.output);
                if (!file)
                {
                    throw std::runtime_error("cannot open benchmark output: " + options.output);
                }
                writeResults(file, results);
            }
            return 0;
        }
        catch (const std::exception& error)
        {
            std::cerr << "Eigen comparison failed: " << error.what() << '\n';
            return 1;
        }
    }

} // namespace plamatrix::internal

int main(int argc, char** argv)
{
    return plamatrix::internal::runBenchmark(argc, argv);
}
