#include "benchmark/benchmark_cases.h"

#include <algorithm>
#include <chrono>
#include <functional>
#include <random>
#include <vector>

#include "plamatrix/dense/dense_matrix.h"
#include "plamatrix/dense/dense_ops.h"
#include "plamatrix/ops/gemm.h"
#include "plamatrix/ops/point_cloud.h"
#include "plamatrix/ops/solver.h"

namespace plamatrix
{
namespace detail
{

using Clock = std::chrono::high_resolution_clock;
using FloatMatrix = DenseMatrix<float, Device::CPU>;
using GpuFloatMatrix = DenseMatrix<float, Device::GPU>;

namespace
{

template <typename T>
void doNotOptimize(T const& value)
{
    asm volatile("" : : "r,m"(value) : "memory");
}

void randomFill(FloatMatrix& mat)
{
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    Index n = mat.size();
    float* d = mat.data();
    for (Index i = 0; i < n; ++i)
    {
        d[i] = dist(rng);
    }
}

FloatMatrix makeRandom(Index rows, Index cols)
{
    FloatMatrix mat(rows, cols);
    randomFill(mat);
    return mat;
}

double median(std::vector<double>& times)
{
    std::sort(times.begin(), times.end());
    std::size_t mid = times.size() / 2;
    if (times.size() % 2 == 0)
    {
        return (times[mid - 1] + times[mid]) * 0.5;
    }
    return times[mid];
}

double measureTransferOne(const FloatMatrix& A, int trials = 5)
{
    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(trials));

    for (int i = 0; i < trials; ++i)
    {
        auto t1 = Clock::now();
        auto A_g = A.toGpu();
        auto t2 = Clock::now();
        doNotOptimize(A_g.data());
        times.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
    }

    return median(times);
}

double measureTransferTwo(const FloatMatrix& A, const FloatMatrix& B, int trials = 5)
{
    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(trials));

    for (int i = 0; i < trials; ++i)
    {
        auto t1 = Clock::now();
        auto A_g = A.toGpu();
        auto B_g = B.toGpu();
        auto t2 = Clock::now();
        doNotOptimize(A_g.data());
        doNotOptimize(B_g.data());
        times.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
    }

    return median(times);
}

double measureGpu(std::function<void()> fn, int warmup = 3, int trials = 10)
{
    for (int i = 0; i < warmup; ++i)
    {
        fn();
    }

    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(trials));

    for (int i = 0; i < trials; ++i)
    {
        auto t_start = Clock::now();
        fn();
        auto t_end = Clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(t_end - t_start).count());
    }

    return median(times);
}

} // namespace

void runGemmGpu(CaseResult& r, Index N)
{
    auto A_cpu = makeRandom(N, N);
    auto B_cpu = makeRandom(N, N);

    r.time_transfer_ms = measureTransferTwo(A_cpu, B_cpu);

    auto A_gpu = A_cpu.toGpu();
    auto B_gpu = B_cpu.toGpu();
    GpuFloatMatrix C_gpu(N, N);

    r.time_gpu_ms = measureGpu([&]()
    {
        gemm(A_gpu, B_gpu, C_gpu);
        doNotOptimize(C_gpu.data());
    });
}

void runAddGpu(CaseResult& r, Index N)
{
    auto A_cpu = makeRandom(N, N);
    auto B_cpu = makeRandom(N, N);

    r.time_transfer_ms = measureTransferTwo(A_cpu, B_cpu);

    auto A_gpu = A_cpu.toGpu();
    auto B_gpu = B_cpu.toGpu();
    GpuFloatMatrix C_gpu(N, N);

    r.time_gpu_ms = measureGpu([&]()
    {
        add(A_gpu, B_gpu, C_gpu);
        doNotOptimize(C_gpu.data());
    });
}

void runSubGpu(CaseResult& r, Index N)
{
    auto A_cpu = makeRandom(N, N);
    auto B_cpu = makeRandom(N, N);

    r.time_transfer_ms = measureTransferTwo(A_cpu, B_cpu);

    auto A_gpu = A_cpu.toGpu();
    auto B_gpu = B_cpu.toGpu();
    GpuFloatMatrix C_gpu(N, N);

    r.time_gpu_ms = measureGpu([&]()
    {
        sub(A_gpu, B_gpu, C_gpu);
        doNotOptimize(C_gpu.data());
    });
}

void runTransposeGpu(CaseResult& r, Index N)
{
    auto A_cpu = makeRandom(N, N);

    r.time_transfer_ms = measureTransferOne(A_cpu);

    auto A_gpu = A_cpu.toGpu();

    r.time_gpu_ms = measureGpu([&]()
    {
        auto C = A_gpu.transpose();
        doNotOptimize(C.data());
    });
}

void runSolveGpu(CaseResult& r, Index N)
{
    auto A_cpu = makeRandom(N, N);
    for (Index i = 0; i < N; ++i)
    {
        A_cpu(i, i) += static_cast<float>(N);
    }
    auto B_cpu = makeRandom(N, 1);

    r.time_transfer_ms = measureTransferTwo(A_cpu, B_cpu);

    auto A_gpu = A_cpu.toGpu();
    auto B_gpu = B_cpu.toGpu();

    r.time_gpu_ms = measureGpu([&]()
    {
        auto X = solve<float, Device::GPU>(A_gpu, B_gpu);
        doNotOptimize(X.data());
    });
}

void runPointTransformGpu(CaseResult& r, Index N)
{
    auto pts_cpu = makeRandom(N, 3);

    Vec3<float> axis{0.0f, 0.0f, 1.0f};
    float angle = 0.5f;
    auto R_cpu = rotationMatrix<float, Device::CPU>(axis, angle);
    Vec3<float> t{1.0f, 2.0f, 3.0f};
    auto T_cpu = rigidTransform<float, Device::CPU>(R_cpu, t);

    std::vector<double> times;
    times.reserve(5);
    for (int i = 0; i < 5; ++i)
    {
        auto tt1 = Clock::now();
        auto pts_g = pts_cpu.toGpu();
        auto T_g = T_cpu.toGpu();
        auto tt2 = Clock::now();
        doNotOptimize(pts_g.data());
        doNotOptimize(T_g.data());
        times.push_back(std::chrono::duration<double, std::milli>(tt2 - tt1).count());
    }
    r.time_transfer_ms = median(times);

    auto pts_gpu = pts_cpu.toGpu();
    auto T_gpu = T_cpu.toGpu();
    GpuFloatMatrix result(N, 3);

    r.time_gpu_ms = measureGpu([&]()
    {
        transformPoints(T_gpu, pts_gpu, result);
        doNotOptimize(result.data());
    });
}

} // namespace detail
} // namespace plamatrix
