#include "benchmark/benchmark_cases.h"

#include <chrono>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "plamatrix/dense/dense_matrix.h"
#include "plamatrix/dense/dense_ops.h"
#include "plamatrix/ops/gemm.h"
#include "plamatrix/ops/decomposition.h"
#include "plamatrix/ops/solver.h"
#include "plamatrix/ops/point_cloud.h"

namespace plamatrix
{
namespace detail
{

using Clock = std::chrono::high_resolution_clock;
using FloatMatrix = DenseMatrix<float, Device::CPU>;
using GpuFloatMatrix = DenseMatrix<float, Device::GPU>;

// ============================================================================
// Shared GPU helpers
// ============================================================================

namespace
{

/// Prevent compiler from optimizing away a value.
template <typename T>
void doNotOptimize(T const& value)
{
    asm volatile("" : : "r,m"(value) : "memory");
}

/// Fill a CPU matrix with uniform random values in [0, 1).
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

FloatMatrix makeRandomSymmetric(Index n)
{
    FloatMatrix mat(n, n);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (Index j = 0; j < n; ++j)
    {
        for (Index i = 0; i <= j; ++i)
        {
            float val = dist(rng);
            mat(i, j) = val;
            mat(j, i) = val;
        }
    }
    return mat;
}

/// Measure host-to-device transfer time for two input matrices (median over trials).
double measureTransferTwo(const FloatMatrix& A, const FloatMatrix& B, int trials = 5)
{
    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(trials));

    for (int i = 0; i < trials; ++i)
    {
        cudaDeviceSynchronize();
        auto t1 = Clock::now();
        auto A_g = A.toGpu();
        auto B_g = B.toGpu();
        cudaDeviceSynchronize();
        auto t2 = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        times.push_back(ms);
    }

    std::sort(times.begin(), times.end());
    std::size_t mid = times.size() / 2;
    if (times.size() % 2 == 0)
    {
        return (times[mid - 1] + times[mid]) * 0.5;
    }
    return times[mid];
}

/// Measure host-to-device transfer time for a single matrix (median over trials).
double measureTransferOne(const FloatMatrix& A, int trials = 5)
{
    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(trials));

    for (int i = 0; i < trials; ++i)
    {
        cudaDeviceSynchronize();
        auto t1 = Clock::now();
        auto A_g = A.toGpu();
        cudaDeviceSynchronize();
        auto t2 = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        times.push_back(ms);
    }

    std::sort(times.begin(), times.end());
    std::size_t mid = times.size() / 2;
    if (times.size() % 2 == 0)
    {
        return (times[mid - 1] + times[mid]) * 0.5;
    }
    return times[mid];
}

/// Run a benchmark function with warmup and trials, return median ms.
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
        double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        times.push_back(ms);
    }

    std::sort(times.begin(), times.end());
    std::size_t mid = times.size() / 2;
    if (times.size() % 2 == 0)
    {
        return (times[mid - 1] + times[mid]) * 0.5;
    }
    return times[mid];
}

} // anonymous namespace

// ============================================================================
// GPU benchmark implementations
// ============================================================================

void runGemmCuda(CaseResult& r, Index N)
{
    auto A_cpu = makeRandom(N, N);
    auto B_cpu = makeRandom(N, N);

    r.time_transfer_ms = measureTransferTwo(A_cpu, B_cpu);

    auto A_gpu = A_cpu.toGpu();
    auto B_gpu = B_cpu.toGpu();
    cudaDeviceSynchronize();

    r.time_cuda_ms = measureGpu([&]()
    {
        auto C = gemm(A_gpu, B_gpu);
        cudaDeviceSynchronize();
        doNotOptimize(C.data());
    });
}

void runAddCuda(CaseResult& r, Index N)
{
    auto A_cpu = makeRandom(N, N);
    auto B_cpu = makeRandom(N, N);

    r.time_transfer_ms = measureTransferTwo(A_cpu, B_cpu);

    auto A_gpu = A_cpu.toGpu();
    auto B_gpu = B_cpu.toGpu();
    cudaDeviceSynchronize();

    r.time_cuda_ms = measureGpu([&]()
    {
        auto C = add(A_gpu, B_gpu);
        cudaDeviceSynchronize();
        doNotOptimize(C.data());
    });
}

void runSubCuda(CaseResult& r, Index N)
{
    auto A_cpu = makeRandom(N, N);
    auto B_cpu = makeRandom(N, N);

    r.time_transfer_ms = measureTransferTwo(A_cpu, B_cpu);

    auto A_gpu = A_cpu.toGpu();
    auto B_gpu = B_cpu.toGpu();
    cudaDeviceSynchronize();

    r.time_cuda_ms = measureGpu([&]()
    {
        auto C = sub(A_gpu, B_gpu);
        cudaDeviceSynchronize();
        doNotOptimize(C.data());
    });
}

void runTransposeCuda(CaseResult& r, Index N)
{
    auto A_cpu = makeRandom(N, N);

    r.time_transfer_ms = measureTransferOne(A_cpu);

    auto A_gpu = A_cpu.toGpu();
    cudaDeviceSynchronize();

    r.time_cuda_ms = measureGpu([&]()
    {
        auto C = A_gpu.transpose();
        cudaDeviceSynchronize();
        doNotOptimize(C.data());
    });
}

void runSvdCuda(CaseResult& r, Index N)
{
    auto A_cpu = makeRandom(N, N);

    r.time_transfer_ms = measureTransferOne(A_cpu);

    auto A_gpu = A_cpu.toGpu();
    cudaDeviceSynchronize();

    // SVD is expensive — fewer trials
    r.time_cuda_ms = measureGpu([&]()
    {
        auto [U, S, Vt] = svd(A_gpu);
        cudaDeviceSynchronize();
        doNotOptimize(U.data());
    }, 1, 3);
}

void runQrCuda(CaseResult& r, Index N)
{
    auto A_cpu = makeRandom(N, N);

    r.time_transfer_ms = measureTransferOne(A_cpu);

    auto A_gpu = A_cpu.toGpu();
    cudaDeviceSynchronize();

    // QR is expensive — fewer trials
    r.time_cuda_ms = measureGpu([&]()
    {
        auto [Q, R] = qr(A_gpu);
        cudaDeviceSynchronize();
        doNotOptimize(Q.data());
    }, 1, 3);
}

void runEighCuda(CaseResult& r, Index N)
{
    auto A_cpu = makeRandomSymmetric(N);

    r.time_transfer_ms = measureTransferOne(A_cpu);

    auto A_gpu = A_cpu.toGpu();
    cudaDeviceSynchronize();

    // Eigh is expensive — fewer trials
    r.time_cuda_ms = measureGpu([&]()
    {
        auto eig = eigh(A_gpu);
        cudaDeviceSynchronize();
        doNotOptimize(eig.data());
    }, 1, 3);
}

void runSolveCuda(CaseResult& r, Index N)
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
    cudaDeviceSynchronize();

    r.time_cuda_ms = measureGpu([&]()
    {
        auto X = solve<float, Device::GPU>(A_gpu, B_gpu);
        cudaDeviceSynchronize();
        doNotOptimize(X.data());
    });
}

void runCovarianceCuda(CaseResult& r, Index N)
{
    auto pts_cpu = makeRandom(N, 3);

    r.time_transfer_ms = measureTransferOne(pts_cpu);

    auto pts_gpu = pts_cpu.toGpu();
    cudaDeviceSynchronize();

    r.time_cuda_ms = measureGpu([&]()
    {
        auto C = covarianceMatrix<float, Device::GPU>(pts_gpu);
        cudaDeviceSynchronize();
        doNotOptimize(C.data());
    });
}

void runPointTransformCuda(CaseResult& r, Index N)
{
    auto pts_cpu = makeRandom(N, 3);

    Vec3<float> axis{0.0f, 0.0f, 1.0f};
    float angle = 0.5f;
    auto R_cpu = rotationMatrix<float, Device::CPU>(axis, angle);
    Vec3<float> t{1.0f, 2.0f, 3.0f};
    auto T_cpu = rigidTransform<float, Device::CPU>(R_cpu, t);

    // Measure transfer time for both transform matrix and points
    {
        std::vector<double> times;
        times.reserve(5);
        for (int i = 0; i < 5; ++i)
        {
            cudaDeviceSynchronize();
            auto tt1 = Clock::now();
            auto pts_g = pts_cpu.toGpu();
            auto T_g = T_cpu.toGpu();
            cudaDeviceSynchronize();
            auto tt2 = Clock::now();
            double ms = std::chrono::duration<double, std::milli>(tt2 - tt1).count();
            times.push_back(ms);
        }
        std::sort(times.begin(), times.end());
        r.time_transfer_ms = times[2]; // median of 5
    }

    auto pts_gpu = pts_cpu.toGpu();
    auto T_gpu = T_cpu.toGpu();
    cudaDeviceSynchronize();

    r.time_cuda_ms = measureGpu([&]()
    {
        auto result = transformPoints<float, Device::GPU>(T_gpu, pts_gpu);
        cudaDeviceSynchronize();
        doNotOptimize(result.data());
    });
}

} // namespace detail
} // namespace plamatrix
