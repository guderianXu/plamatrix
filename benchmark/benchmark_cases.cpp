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
#ifdef PLAMATRIX_WITH_CUDA
        cudaDeviceSynchronize();
#endif
        auto t_start = Clock::now();
        fn();
#ifdef PLAMATRIX_WITH_CUDA
        cudaDeviceSynchronize();
#endif
        auto t_end = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        times.push_back(ms);
    }

    std::sort(times.begin(), times.end());
    std::size_t mid = times.size() / 2;
    if (times.size() % 2 == 0) { return (times[mid - 1] + times[mid]) * 0.5; }
    return times[mid];
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

bool shouldRunCase(const std::vector<std::string>& case_filter, const char* name)
{
    return case_filter.empty()
        || std::find(case_filter.begin(), case_filter.end(), std::string(name)) != case_filter.end();
}

bool hasTiming(const CaseResult& r)
{
    return r.time_serial_ms >= 0.0 || r.time_omp_ms >= 0.0 || r.time_cuda_ms >= 0.0 || r.time_transfer_ms >= 0.0;
}

void appendResultIfRan(BenchmarkReport& report, CaseResult&& r)
{
    if (hasTiming(r))
    {
        report.results.push_back(std::move(r));
    }
    else
    {
        std::cerr << "  " << r.name << " (skipped — no selected backend available at this size)" << std::endl;
    }
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
    
    r.time_serial_ms = measure([&]()
    {
        auto C = add(A, B);
        doNotOptimize(C.data());
    }, 2, 5);
}

void runAddOmp(CaseResult& r, Index N)
{
    auto A = makeRandom(N, N);
    auto B = makeRandom(N, N);
    
    r.time_omp_ms = measure([&]()
    {
        auto C = add(A, B);
        doNotOptimize(C.data());
    }, 2, 5);
}

// ---- sub ----

void runSubSerial(CaseResult& r, Index N)
{
    OmpThreadGuard guard(1);
    auto A = makeRandom(N, N);
    auto B = makeRandom(N, N);
    
    r.time_serial_ms = measure([&]()
    {
        auto C = sub(A, B);
        doNotOptimize(C.data());
    }, 2, 5);
}

void runSubOmp(CaseResult& r, Index N)
{
    auto A = makeRandom(N, N);
    auto B = makeRandom(N, N);
    
    r.time_omp_ms = measure([&]()
    {
        auto C = sub(A, B);
        doNotOptimize(C.data());
    }, 2, 5);
}

// ---- transpose ----

void runTransposeSerial(CaseResult& r, Index N)
{
    OmpThreadGuard guard(1);
    auto A = makeRandom(N, N);
    
    r.time_serial_ms = measure([&]()
    {
        auto C = A.transpose();
        doNotOptimize(C.data());
    }, 2, 5);
}

void runTransposeOmp(CaseResult& r, Index N)
{
    auto A = makeRandom(N, N);
    
    r.time_omp_ms = measure([&]()
    {
        auto C = A.transpose();
        doNotOptimize(C.data());
    }, 2, 5);
}

// ---- scalarMul ----

void runScalarMulSerial(CaseResult& r, Index N)
{
    OmpThreadGuard guard(1);
    auto A = makeRandom(N, N);
    float alpha = 2.0f;
    
    r.time_serial_ms = measure([&]()
    {
        auto C = alpha * A;
        doNotOptimize(C.data());
    }, 2, 5);
}

void runScalarMulOmp(CaseResult& r, Index N)
{
    auto A = makeRandom(N, N);
    float alpha = 2.0f;
    
    r.time_omp_ms = measure([&]()
    {
        auto C = alpha * A;
        doNotOptimize(C.data());
    }, 2, 5);
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
    
    r.time_serial_ms = measure([&]()
    {
        auto C = covarianceMatrix<float, Device::CPU>(points);
        doNotOptimize(C.data());
    }, 2, 5);
}

void runCovarianceOmp(CaseResult& r, Index N)
{
    auto points = makeRandom(N, 3);
    
    r.time_omp_ms = measure([&]()
    {
        auto C = covarianceMatrix<float, Device::CPU>(points);
        doNotOptimize(C.data());
    }, 2, 5);
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

    
    r.time_serial_ms = measure([&]()
    {
        auto result = transformPoints<float, Device::CPU>(T, points);
        doNotOptimize(result.data());
    }, 2, 5);
}

void runPointTransformOmp(CaseResult& r, Index N)
{
    auto points = makeRandom(N, 3);
    Vec3<float> axis{0.0f, 0.0f, 1.0f};
    float angle = 0.5f;
    auto R = rotationMatrix<float, Device::CPU>(axis, angle);
    Vec3<float> t{1.0f, 2.0f, 3.0f};
    auto T = rigidTransform<float, Device::CPU>(R, t);

    
    r.time_omp_ms = measure([&]()
    {
        auto result = transformPoints<float, Device::CPU>(T, points);
        doNotOptimize(result.data());
    }, 2, 5);
}

} // anonymous namespace

// ============================================================================
// runAllCases — dispatcher
// ============================================================================

void runAllCases(const std::vector<Index>& sizes,
                 bool serial,
                 bool omp,
                 bool cuda,
                 BenchmarkReport& report,
                 const std::vector<std::string>& case_filter)
{
    for (Index N : sizes)
    {
        std::cerr << "--- N=" << N << " ---" << std::endl;

        // gemm
        if (shouldRunCase(case_filter, "gemm"))
        {
            CaseResult r{"gemm", N, -1.0, -1.0, -1.0, -1.0};
            if (serial) { std::cerr << "  gemm serial..." << std::endl; runGemmSerial(r, N); std::cerr << "    " << r.time_serial_ms << " ms" << std::endl; }
            if (omp)    { std::cerr << "  gemm omp..." << std::endl;    runGemmOmp(r, N); std::cerr << "    " << r.time_omp_ms << " ms" << std::endl; }
            #ifdef PLAMATRIX_WITH_CUDA
            if (cuda)   { std::cerr << "  gemm cuda..." << std::endl;   detail::runGemmCuda(r, N); std::cerr << "    " << r.time_cuda_ms << " ms" << std::endl; }
#endif
            appendResultIfRan(report, std::move(r));
        }

        // add
        if (shouldRunCase(case_filter, "add"))
        {
            CaseResult r{"add", N, -1.0, -1.0, -1.0, -1.0};
            if (serial) { std::cerr << "  add serial..." << std::endl; runAddSerial(r, N); std::cerr << "    " << r.time_serial_ms << " ms" << std::endl; }
            if (omp)    { std::cerr << "  add omp..." << std::endl;    runAddOmp(r, N); std::cerr << "    " << r.time_omp_ms << " ms" << std::endl; }
            #ifdef PLAMATRIX_WITH_CUDA
            if (cuda)   { std::cerr << "  add cuda..." << std::endl;   detail::runAddCuda(r, N); std::cerr << "    " << r.time_cuda_ms << " ms" << std::endl; }
#endif
            appendResultIfRan(report, std::move(r));
        }

        // sub
        if (shouldRunCase(case_filter, "sub"))
        {
            CaseResult r{"sub", N, -1.0, -1.0, -1.0, -1.0};
            if (serial) { std::cerr << "  sub serial..." << std::endl; runSubSerial(r, N); std::cerr << "    " << r.time_serial_ms << " ms" << std::endl; }
            if (omp)    { std::cerr << "  sub omp..." << std::endl;    runSubOmp(r, N); std::cerr << "    " << r.time_omp_ms << " ms" << std::endl; }
            #ifdef PLAMATRIX_WITH_CUDA
            if (cuda)   { std::cerr << "  sub cuda..." << std::endl;   detail::runSubCuda(r, N); std::cerr << "    " << r.time_cuda_ms << " ms" << std::endl; }
#endif
            appendResultIfRan(report, std::move(r));
        }

        // transpose
        if (shouldRunCase(case_filter, "transpose"))
        {
            CaseResult r{"transpose", N, -1.0, -1.0, -1.0, -1.0};
            if (serial) { std::cerr << "  transpose serial..." << std::endl; runTransposeSerial(r, N); std::cerr << "    " << r.time_serial_ms << " ms" << std::endl; }
            if (omp)    { std::cerr << "  transpose omp..." << std::endl;    runTransposeOmp(r, N); std::cerr << "    " << r.time_omp_ms << " ms" << std::endl; }
            #ifdef PLAMATRIX_WITH_CUDA
            if (cuda)   { std::cerr << "  transpose cuda..." << std::endl;   detail::runTransposeCuda(r, N); std::cerr << "    " << r.time_cuda_ms << " ms" << std::endl; }
#endif
            appendResultIfRan(report, std::move(r));
        }

        // scalarMul
        if (shouldRunCase(case_filter, "scalarMul"))
        {
            CaseResult r{"scalarMul", N, -1.0, -1.0, -1.0, -1.0};
            if (serial) { std::cerr << "  scalarMul serial..." << std::endl; runScalarMulSerial(r, N); std::cerr << "    " << r.time_serial_ms << " ms" << std::endl; }
            if (omp)    { std::cerr << "  scalarMul omp..." << std::endl;    runScalarMulOmp(r, N);    std::cerr << "    " << r.time_omp_ms << " ms" << std::endl; }
            appendResultIfRan(report, std::move(r));
        }

        // svd — skip large CPU sizes; combine --case svd with tiny/smoke for focused CPU runs
        if (shouldRunCase(case_filter, "svd"))
        {
            CaseResult r{"svd", N, -1.0, -1.0, -1.0, -1.0};
            const bool cpu_ok = (N <= 256);
            if (serial && cpu_ok) { std::cerr << "  svd serial..." << std::endl; runSvdSerial(r, N); std::cerr << "    " << r.time_serial_ms << " ms" << std::endl; }
            else if (serial)      { std::cerr << "  svd serial  (skipped — too large for broad CPU benchmark)" << std::endl; }
            if (omp && cpu_ok)    { std::cerr << "  svd omp..." << std::endl;    runSvdOmp(r, N);    std::cerr << "    " << r.time_omp_ms << " ms" << std::endl; }
            else if (omp)         { std::cerr << "  svd omp     (skipped — too large for broad CPU benchmark)" << std::endl; }
            #ifdef PLAMATRIX_WITH_CUDA
            if (cuda)   { std::cerr << "  svd cuda..." << std::endl;   detail::runSvdCuda(r, N); std::cerr << "    " << r.time_cuda_ms << " ms" << std::endl; }
#endif
            appendResultIfRan(report, std::move(r));
        }

        // qr — CPU Householder is O(n^3), skip large sizes
        if (shouldRunCase(case_filter, "qr"))
        {
            CaseResult r{"qr", N, -1.0, -1.0, -1.0, -1.0};
            const bool cpu_ok = (N <= 256);
            if (serial && cpu_ok) { std::cerr << "  qr serial..." << std::endl; runQrSerial(r, N); std::cerr << "    " << r.time_serial_ms << " ms" << std::endl; }
            else if (serial)      { std::cerr << "  qr serial   (skipped — too large for CPU)" << std::endl; }
            if (omp && cpu_ok)    { std::cerr << "  qr omp..." << std::endl;    runQrOmp(r, N);    std::cerr << "    " << r.time_omp_ms << " ms" << std::endl; }
            else if (omp)         { std::cerr << "  qr omp      (skipped — too large for CPU)" << std::endl; }
            #ifdef PLAMATRIX_WITH_CUDA
            if (cuda)   { std::cerr << "  qr cuda..." << std::endl;   detail::runQrCuda(r, N); std::cerr << "    " << r.time_cuda_ms << " ms" << std::endl; }
#endif
            appendResultIfRan(report, std::move(r));
        }

        // eigh — skip large CPU sizes; combine --case eigh with tiny/smoke for focused CPU runs
        if (shouldRunCase(case_filter, "eigh"))
        {
            CaseResult r{"eigh", N, -1.0, -1.0, -1.0, -1.0};
            const bool cpu_ok = (N <= 256);
            if (serial && cpu_ok) { std::cerr << "  eigh serial..." << std::endl; runEighSerial(r, N); std::cerr << "    " << r.time_serial_ms << " ms" << std::endl; }
            else if (serial)      { std::cerr << "  eigh serial  (skipped — too large for broad CPU benchmark)" << std::endl; }
            if (omp && cpu_ok)    { std::cerr << "  eigh omp..." << std::endl;    runEighOmp(r, N);    std::cerr << "    " << r.time_omp_ms << " ms" << std::endl; }
            else if (omp)         { std::cerr << "  eigh omp     (skipped — too large for broad CPU benchmark)" << std::endl; }
            #ifdef PLAMATRIX_WITH_CUDA
            if (cuda)   { std::cerr << "  eigh cuda..." << std::endl;   detail::runEighCuda(r, N); std::cerr << "    " << r.time_cuda_ms << " ms" << std::endl; }
#endif
            appendResultIfRan(report, std::move(r));
        }

        // solve
        if (shouldRunCase(case_filter, "solve"))
        {
            CaseResult r{"solve", N, -1.0, -1.0, -1.0, -1.0};
            if (serial) { std::cerr << "  solve serial..." << std::endl; runSolveSerial(r, N); std::cerr << "    " << r.time_serial_ms << " ms" << std::endl; }
            if (omp)    { std::cerr << "  solve omp..." << std::endl;    runSolveOmp(r, N);    std::cerr << "    " << r.time_omp_ms << " ms" << std::endl; }
            #ifdef PLAMATRIX_WITH_CUDA
            if (cuda)   { std::cerr << "  solve cuda..." << std::endl;   detail::runSolveCuda(r, N); std::cerr << "    " << r.time_cuda_ms << " ms" << std::endl; }
#endif
            appendResultIfRan(report, std::move(r));
        }

        // covariance
        if (shouldRunCase(case_filter, "covariance"))
        {
            CaseResult r{"covariance", N, -1.0, -1.0, -1.0, -1.0};
            if (serial) { std::cerr << "  covariance serial..." << std::endl; runCovarianceSerial(r, N); std::cerr << "    " << r.time_serial_ms << " ms" << std::endl; }
            if (omp)    { std::cerr << "  covariance omp..." << std::endl;    runCovarianceOmp(r, N);    std::cerr << "    " << r.time_omp_ms << " ms" << std::endl; }
            #ifdef PLAMATRIX_WITH_CUDA
            if (cuda)   { std::cerr << "  covariance cuda..." << std::endl;   detail::runCovarianceCuda(r, N); std::cerr << "    " << r.time_cuda_ms << " ms" << std::endl; }
#endif
            appendResultIfRan(report, std::move(r));
        }

        // pointTransform
        if (shouldRunCase(case_filter, "pointTransform"))
        {
            CaseResult r{"pointTransform", N, -1.0, -1.0, -1.0, -1.0};
            if (serial) { std::cerr << "  pointTransform serial..." << std::endl; runPointTransformSerial(r, N); std::cerr << "    " << r.time_serial_ms << " ms" << std::endl; }
            if (omp)    { std::cerr << "  pointTransform omp..." << std::endl;    runPointTransformOmp(r, N);    std::cerr << "    " << r.time_omp_ms << " ms" << std::endl; }
            #ifdef PLAMATRIX_WITH_CUDA
            if (cuda)   { std::cerr << "  pointTransform cuda..." << std::endl;   detail::runPointTransformCuda(r, N); std::cerr << "    " << r.time_cuda_ms << " ms" << std::endl; }
#endif
            appendResultIfRan(report, std::move(r));
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
