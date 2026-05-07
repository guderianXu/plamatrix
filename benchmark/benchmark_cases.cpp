#include <iostream>

#include "benchmark/benchmark_cases.h"
#include "benchmark/report_writer.h"

#include <algorithm>
#include <chrono>
#include <random>
#include <vector>

#include <omp.h>

#include "plamatrix/dense/dense_matrix.h"
#include "plamatrix/dense/dense_ops.h"
#include "plamatrix/ops/gemm.h"
#include "plamatrix/ops/decomposition.h"
#include "plamatrix/ops/solver.h"
#include "plamatrix/ops/point_cloud.h"

namespace plamatrix
{

// ============================================================================
// measure() — timing utility
// Returns median time in milliseconds across `trials` timed runs.
// ============================================================================

double measure(BenchmarkFn fn, int warmup, int trials)
{
    using Clock = std::chrono::high_resolution_clock;

    for (int i = 0; i < warmup; ++i) { fn(); }

    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(trials));

    for (int i = 0; i < trials; ++i)
    {
        cudaDeviceSynchronize();
        auto t_start = Clock::now();
        fn();
        cudaDeviceSynchronize();
        auto t_end = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        times.push_back(ms);
    }

    std::sort(times.begin(), times.end());
    std::size_t mid = times.size() / 2;
    if (times.size() % 2 == 0) { return (times[mid - 1] + times[mid]) * 0.5; }
    return times[mid];
}

// For fast element-wise operations: repeat internally to accumulate measurable wall-clock time,
// then divide by repeat count to report per-operation time. This avoids noise dominating
// measurements on medium/large matrices where a single op runs in <1 ms.
template <typename Fn>
static double measureRepeated(Fn&& fn, int warmup, int trials, int repeats)
{
    for (int i = 0; i < warmup; ++i)
    {
        for (int r = 0; r < repeats; ++r) { fn(); }
    }

    using Clock = std::chrono::high_resolution_clock;
    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(trials));

    for (int i = 0; i < trials; ++i)
    {
        cudaDeviceSynchronize();
        auto t_start = Clock::now();
        for (int r = 0; r < repeats; ++r) { fn(); }
        cudaDeviceSynchronize();
        auto t_end = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        times.push_back(ms / repeats);
    }

    std::sort(times.begin(), times.end());
    std::size_t mid = times.size() / 2;
    if (times.size() % 2 == 0) { return (times[mid - 1] + times[mid]) * 0.5; }
    return times[mid];
}

// Heuristic: for fast ops, repeat enough times so total wall-clock is ~100 ms,
// which gives reliable timing without taking forever on large matrices.
static int repeatCount(Index N)
{
    if (N <= 1024)  return 500;
    if (N <= 4096)  return 100;
    if (N <= 16384) return 20;
    return 5;
}

// ============================================================================
// Helpers
// ============================================================================

namespace
{

using FloatMatrix = DenseMatrix<float, Device::CPU>;

void randomFill(FloatMatrix& mat)
{
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    Index n = mat.size();
    float* d = mat.data();
    for (Index i = 0; i < n; ++i) { d[i] = dist(rng); }
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

class OmpThreadGuard
{
public:
    explicit OmpThreadGuard(int threads) : _prev(omp_get_max_threads()) { omp_set_num_threads(threads); }
    ~OmpThreadGuard() { omp_set_num_threads(_prev); }
    OmpThreadGuard(const OmpThreadGuard&) = delete;
    OmpThreadGuard& operator=(const OmpThreadGuard&) = delete;
private:
    int _prev;
};

template <typename T>
void doNotOptimize(T const& value)
{
    asm volatile("" : : "r,m"(value) : "memory");
}

} // anonymous namespace

// ============================================================================
// CPU benchmark case runners
// ============================================================================

namespace
{

// ---- gemm (compute-bound) ----

void runGemmSerial(CaseResult& r, Index N)
{
    OmpThreadGuard guard(1);
    auto A = makeRandom(N, N);
    auto B = makeRandom(N, N);
    r.time_serial_ms = measure([&]()
    {
        auto C = gemm(A, B);
        doNotOptimize(C.data());
    }, 2, 5);
}

void runGemmOmp(CaseResult& r, Index N)
{
    auto A = makeRandom(N, N);
    auto B = makeRandom(N, N);
    r.time_omp_ms = measure([&]()
    {
        auto C = gemm(A, B);
        doNotOptimize(C.data());
    }, 2, 5);
}

// ---- add (memory-bound) ----

void runAddSerial(CaseResult& r, Index N)
{
    OmpThreadGuard guard(1);
    auto A = makeRandom(N, N);
    auto B = makeRandom(N, N);
    int repeats = repeatCount(N);
    r.time_serial_ms = measureRepeated([&]()
    {
        auto C = add(A, B);
        doNotOptimize(C.data());
    }, 1, 5, repeats);
}

void runAddOmp(CaseResult& r, Index N)
{
    auto A = makeRandom(N, N);
    auto B = makeRandom(N, N);
    int repeats = repeatCount(N);
    r.time_omp_ms = measureRepeated([&]()
    {
        auto C = add(A, B);
        doNotOptimize(C.data());
    }, 1, 5, repeats);
}

// ---- sub ----

void runSubSerial(CaseResult& r, Index N)
{
    OmpThreadGuard guard(1);
    auto A = makeRandom(N, N);
    auto B = makeRandom(N, N);
    int repeats = repeatCount(N);
    r.time_serial_ms = measureRepeated([&]()
    {
        auto C = sub(A, B);
        doNotOptimize(C.data());
    }, 1, 5, repeats);
}

void runSubOmp(CaseResult& r, Index N)
{
    auto A = makeRandom(N, N);
    auto B = makeRandom(N, N);
    int repeats = repeatCount(N);
    r.time_omp_ms = measureRepeated([&]()
    {
        auto C = sub(A, B);
        doNotOptimize(C.data());
    }, 1, 5, repeats);
}

// ---- transpose ----

void runTransposeSerial(CaseResult& r, Index N)
{
    OmpThreadGuard guard(1);
    auto A = makeRandom(N, N);
    int repeats = repeatCount(N);
    r.time_serial_ms = measureRepeated([&]()
    {
        auto C = A.transpose();
        doNotOptimize(C.data());
    }, 1, 5, repeats);
}

void runTransposeOmp(CaseResult& r, Index N)
{
    auto A = makeRandom(N, N);
    int repeats = repeatCount(N);
    r.time_omp_ms = measureRepeated([&]()
    {
        auto C = A.transpose();
        doNotOptimize(C.data());
    }, 1, 5, repeats);
}

// ---- scalarMul ----

void runScalarMulSerial(CaseResult& r, Index N)
{
    OmpThreadGuard guard(1);
    auto A = makeRandom(N, N);
    float alpha = 2.0f;
    int repeats = repeatCount(N);
    r.time_serial_ms = measureRepeated([&]()
    {
        auto C = alpha * A;
        doNotOptimize(C.data());
    }, 1, 5, repeats);
}

void runScalarMulOmp(CaseResult& r, Index N)
{
    auto A = makeRandom(N, N);
    float alpha = 2.0f;
    int repeats = repeatCount(N);
    r.time_omp_ms = measureRepeated([&]()
    {
        auto C = alpha * A;
        doNotOptimize(C.data());
    }, 1, 5, repeats);
}

// ---- svd (very compute-heavy) ----

void runSvdSerial(CaseResult& r, Index N)
{
    OmpThreadGuard guard(1);
    auto A = makeRandom(N, N);
    r.time_serial_ms = measure([&]()
    {
        auto [U, S, Vt] = svd(A);
        doNotOptimize(U.data());
    }, 0, 3);
}

void runSvdOmp(CaseResult& r, Index N)
{
    auto A = makeRandom(N, N);
    r.time_omp_ms = measure([&]()
    {
        auto [U, S, Vt] = svd(A);
        doNotOptimize(U.data());
    }, 0, 3);
}

// ---- qr ----

void runQrSerial(CaseResult& r, Index N)
{
    OmpThreadGuard guard(1);
    auto A = makeRandom(N, N);
    r.time_serial_ms = measure([&]()
    {
        auto [Q, R] = qr(A);
        doNotOptimize(Q.data());
    }, 0, 3);
}

void runQrOmp(CaseResult& r, Index N)
{
    auto A = makeRandom(N, N);
    r.time_omp_ms = measure([&]()
    {
        auto [Q, R] = qr(A);
        doNotOptimize(Q.data());
    }, 0, 3);
}

// ---- eigh ----

void runEighSerial(CaseResult& r, Index N)
{
    OmpThreadGuard guard(1);
    auto A = makeRandomSymmetric(N);
    r.time_serial_ms = measure([&]()
    {
        auto eig = eigh(A);
        doNotOptimize(eig.data());
    }, 0, 3);
}

void runEighOmp(CaseResult& r, Index N)
{
    auto A = makeRandomSymmetric(N);
    r.time_omp_ms = measure([&]()
    {
        auto eig = eigh(A);
        doNotOptimize(eig.data());
    }, 0, 3);
}

// ---- solve ----

void runSolveSerial(CaseResult& r, Index N)
{
    OmpThreadGuard guard(1);
    auto A = makeRandom(N, N);
    for (Index i = 0; i < N; ++i) { A(i, i) += static_cast<float>(N); }
    auto B = makeRandom(N, 1);
    r.time_serial_ms = measure([&]()
    {
        auto X = solve<float, Device::CPU>(A, B);
        doNotOptimize(X.data());
    }, 0, 5);
}

void runSolveOmp(CaseResult& r, Index N)
{
    auto A = makeRandom(N, N);
    for (Index i = 0; i < N; ++i) { A(i, i) += static_cast<float>(N); }
    auto B = makeRandom(N, 1);
    r.time_omp_ms = measure([&]()
    {
        auto X = solve<float, Device::CPU>(A, B);
        doNotOptimize(X.data());
    }, 0, 5);
}

// ---- covariance ----

void runCovarianceSerial(CaseResult& r, Index N)
{
    OmpThreadGuard guard(1);
    auto points = makeRandom(N, 3);
    int repeats = repeatCount(N);
    r.time_serial_ms = measureRepeated([&]()
    {
        auto C = covarianceMatrix<float, Device::CPU>(points);
        doNotOptimize(C.data());
    }, 1, 5, repeats);
}

void runCovarianceOmp(CaseResult& r, Index N)
{
    auto points = makeRandom(N, 3);
    int repeats = repeatCount(N);
    r.time_omp_ms = measureRepeated([&]()
    {
        auto C = covarianceMatrix<float, Device::CPU>(points);
        doNotOptimize(C.data());
    }, 1, 5, repeats);
}

// ---- pointTransform ----

void runPointTransformSerial(CaseResult& r, Index N)
{
    OmpThreadGuard guard(1);
    auto points = makeRandom(N, 3);
    Vec3<float> axis{0.0f, 0.0f, 1.0f};
    float angle = 0.5f;
    auto R = rotationMatrix<float, Device::CPU>(axis, angle);
    Vec3<float> t{1.0f, 2.0f, 3.0f};
    auto T = rigidTransform<float, Device::CPU>(R, t);

    int repeats = repeatCount(N);
    r.time_serial_ms = measureRepeated([&]()
    {
        auto result = transformPoints<float, Device::CPU>(T, points);
        doNotOptimize(result.data());
    }, 1, 5, repeats);
}

void runPointTransformOmp(CaseResult& r, Index N)
{
    auto points = makeRandom(N, 3);
    Vec3<float> axis{0.0f, 0.0f, 1.0f};
    float angle = 0.5f;
    auto R = rotationMatrix<float, Device::CPU>(axis, angle);
    Vec3<float> t{1.0f, 2.0f, 3.0f};
    auto T = rigidTransform<float, Device::CPU>(R, t);

    int repeats = repeatCount(N);
    r.time_omp_ms = measureRepeated([&]()
    {
        auto result = transformPoints<float, Device::CPU>(T, points);
        doNotOptimize(result.data());
    }, 1, 5, repeats);
}

} // anonymous namespace

// ============================================================================
// runAllCases — dispatcher
// ============================================================================

void runAllCases(const std::vector<Index>& sizes, bool serial, bool omp, bool cuda, BenchmarkReport& report)
{
    for (Index N : sizes)
    {
        std::cerr << "  Benchmarking size N=" << N << "..." << std::endl;

        // gemm
        {
            CaseResult r{"gemm", N, -1.0, -1.0, -1.0, -1.0};
            if (serial) runGemmSerial(r, N);
            if (omp)    runGemmOmp(r, N);
            #ifdef PLAMATRIX_WITH_CUDA
            if (cuda)   detail::runGemmCuda(r, N);
#endif
            report.results.push_back(std::move(r));
        }

        // add
        {
            CaseResult r{"add", N, -1.0, -1.0, -1.0, -1.0};
            if (serial) runAddSerial(r, N);
            if (omp)    runAddOmp(r, N);
            #ifdef PLAMATRIX_WITH_CUDA
            if (cuda)   detail::runAddCuda(r, N);
#endif
            report.results.push_back(std::move(r));
        }

        // sub
        {
            CaseResult r{"sub", N, -1.0, -1.0, -1.0, -1.0};
            if (serial) runSubSerial(r, N);
            if (omp)    runSubOmp(r, N);
            #ifdef PLAMATRIX_WITH_CUDA
            if (cuda)   detail::runSubCuda(r, N);
#endif
            report.results.push_back(std::move(r));
        }

        // transpose
        {
            CaseResult r{"transpose", N, -1.0, -1.0, -1.0, -1.0};
            if (serial) runTransposeSerial(r, N);
            if (omp)    runTransposeOmp(r, N);
            #ifdef PLAMATRIX_WITH_CUDA
            if (cuda)   detail::runTransposeCuda(r, N);
#endif
            report.results.push_back(std::move(r));
        }

        // scalarMul
        {
            CaseResult r{"scalarMul", N, -1.0, -1.0, -1.0, -1.0};
            if (serial) runScalarMulSerial(r, N);
            if (omp)    runScalarMulOmp(r, N);
            report.results.push_back(std::move(r));
        }

        // svd
        {
            CaseResult r{"svd", N, -1.0, -1.0, -1.0, -1.0};
            if (serial) runSvdSerial(r, N);
            if (omp)    runSvdOmp(r, N);
            #ifdef PLAMATRIX_WITH_CUDA
            if (cuda)   detail::runSvdCuda(r, N);
#endif
            report.results.push_back(std::move(r));
        }

        // qr
        {
            CaseResult r{"qr", N, -1.0, -1.0, -1.0, -1.0};
            if (serial) runQrSerial(r, N);
            if (omp)    runQrOmp(r, N);
            #ifdef PLAMATRIX_WITH_CUDA
            if (cuda)   detail::runQrCuda(r, N);
#endif
            report.results.push_back(std::move(r));
        }

        // eigh
        {
            CaseResult r{"eigh", N, -1.0, -1.0, -1.0, -1.0};
            if (serial) runEighSerial(r, N);
            if (omp)    runEighOmp(r, N);
            #ifdef PLAMATRIX_WITH_CUDA
            if (cuda)   detail::runEighCuda(r, N);
#endif
            report.results.push_back(std::move(r));
        }

        // solve
        {
            CaseResult r{"solve", N, -1.0, -1.0, -1.0, -1.0};
            if (serial) runSolveSerial(r, N);
            if (omp)    runSolveOmp(r, N);
            #ifdef PLAMATRIX_WITH_CUDA
            if (cuda)   detail::runSolveCuda(r, N);
#endif
            report.results.push_back(std::move(r));
        }

        // covariance
        {
            CaseResult r{"covariance", N, -1.0, -1.0, -1.0, -1.0};
            if (serial) runCovarianceSerial(r, N);
            if (omp)    runCovarianceOmp(r, N);
            #ifdef PLAMATRIX_WITH_CUDA
            if (cuda)   detail::runCovarianceCuda(r, N);
#endif
            report.results.push_back(std::move(r));
        }

        // pointTransform
        {
            CaseResult r{"pointTransform", N, -1.0, -1.0, -1.0, -1.0};
            if (serial) runPointTransformSerial(r, N);
            if (omp)    runPointTransformOmp(r, N);
            #ifdef PLAMATRIX_WITH_CUDA
            if (cuda)   detail::runPointTransformCuda(r, N);
#endif
            report.results.push_back(std::move(r));
        }
    }
}

std::vector<std::string> getAllCaseNames()
{
    return {
        "gemm",
        "add",
        "sub",
        "transpose",
        "scalarMul",
        "svd",
        "qr",
        "eigh",
        "solve",
        "covariance",
        "pointTransform"
    };
}

} // namespace plamatrix
