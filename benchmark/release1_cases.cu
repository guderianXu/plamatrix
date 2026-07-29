#include "benchmark/benchmark_cases.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "plamatrix/core/error_check.h"
#include "plamatrix/dense/dense_matrix.h"
#include "plamatrix/dense/elementwise.h"
#include "plamatrix/ops/indexing.h"
#include "plamatrix/ops/reduction.h"
#include "plamatrix/ops/small_matrix.h"

namespace plamatrix
{
namespace detail
{
namespace
{

using Clock = std::chrono::high_resolution_clock;
using CpuFloatMatrix = DenseMatrix<float, Device::CPU>;
using CpuMaskMatrix = DenseMatrix<std::uint8_t, Device::CPU>;
using GpuFloatMatrix = DenseMatrix<float, Device::GPU>;
using GpuMaskMatrix = DenseMatrix<std::uint8_t, Device::GPU>;
using GpuIndexMatrix = DenseMatrix<Index, Device::GPU>;

class CudaStream
{
public:
    CudaStream()
    {
        PLAMATRIX_CHECK_CUDA(cudaStreamCreate(&_stream));
    }

    ~CudaStream()
    {
        if (_stream != nullptr)
        {
            static_cast<void>(cudaStreamDestroy(_stream));
        }
    }

    cudaStream_t get() const noexcept
    {
        return _stream;
    }

private:
    cudaStream_t _stream = nullptr;
};

class CudaEvent
{
public:
    CudaEvent()
    {
        PLAMATRIX_CHECK_CUDA(cudaEventCreate(&_event));
    }

    ~CudaEvent()
    {
        if (_event != nullptr)
        {
            static_cast<void>(cudaEventDestroy(_event));
        }
    }

    cudaEvent_t get() const noexcept
    {
        return _event;
    }

private:
    cudaEvent_t _event = nullptr;
};

double median(std::vector<double>& samples)
{
    std::sort(samples.begin(), samples.end());
    const std::size_t middle = samples.size() / 2;
    if ((samples.size() % 2) == 0)
    {
        return (samples[middle - 1] + samples[middle]) * 0.5;
    }
    return samples[middle];
}

template <typename Launch, typename AfterSync>
double measureEvent(cudaStream_t stream, Launch&& launch, AfterSync&& after_sync,
                    int warmup = 2, int trials = 7)
{
    CudaEvent start;
    CudaEvent stop;
    for (int iteration = 0; iteration < warmup; ++iteration)
    {
        launch();
        PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
        after_sync();
    }

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(trials));
    for (int iteration = 0; iteration < trials; ++iteration)
    {
        PLAMATRIX_CHECK_CUDA(cudaEventRecord(start.get(), stream));
        launch();
        PLAMATRIX_CHECK_CUDA(cudaEventRecord(stop.get(), stream));
        PLAMATRIX_CHECK_CUDA(cudaEventSynchronize(stop.get()));
        float elapsed_ms = 0.0f;
        PLAMATRIX_CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start.get(), stop.get()));
        samples.push_back(static_cast<double>(elapsed_ms));
        after_sync();
    }
    return median(samples);
}

template <typename Launch, typename AfterSync>
double measureWarmTotal(cudaStream_t stream, Launch&& launch, AfterSync&& after_sync)
{
    std::vector<double> samples;
    samples.reserve(7);
    for (int iteration = 0; iteration < 7; ++iteration)
    {
        const auto start = Clock::now();
        launch();
        PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
        after_sync();
        const auto stop = Clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
    }
    return median(samples);
}

template <typename Function>
double measureColdAllocation(Function&& function)
{
    PLAMATRIX_CHECK_CUDA(cudaDeviceSynchronize());
    const auto start = Clock::now();
    function();
    PLAMATRIX_CHECK_CUDA(cudaDeviceSynchronize());
    const auto stop = Clock::now();
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

CpuFloatMatrix makeRandom(Index rows, Index cols)
{
    auto matrix = CpuFloatMatrix::pinned(rows, cols);
    std::mt19937 generator(42);
    std::uniform_real_distribution<float> distribution(0.25f, 1.0f);
    for (Index index = 0; index < matrix.size(); ++index)
    {
        matrix.data()[index] = distribution(generator);
    }
    return matrix;
}

CpuMaskMatrix makeMask(Index rows)
{
    auto mask = CpuMaskMatrix::pinned(rows, 1);
    for (Index row = 0; row < rows; ++row)
    {
        mask(row, 0) = static_cast<std::uint8_t>((row % 3) != 0);
    }
    return mask;
}

CpuFloatMatrix makeEighInput(Index rows)
{
    auto matrices = CpuFloatMatrix::pinned(rows, 6);
    for (Index row = 0; row < rows; ++row)
    {
        const float offset = static_cast<float>(row % 17) * 0.001f;
        matrices(row, 0) = 4.0f + offset;
        matrices(row, 1) = 0.1f;
        matrices(row, 2) = 0.05f;
        matrices(row, 3) = 3.0f + offset;
        matrices(row, 4) = 0.02f;
        matrices(row, 5) = 2.0f + offset;
    }
    return matrices;
}

template <typename Copies>
double measureTransfer(cudaStream_t stream, Copies&& copies)
{
    return measureEvent(stream, std::forward<Copies>(copies), []() {}, 2, 5);
}

void setWorkspaceSnapshot(CaseResult& result, std::size_t before, std::size_t after,
                          const void* before_data, const void* after_data)
{
    result.workspace_bytes_before = before;
    result.workspace_bytes_after = after;
    if (before != after || before_data != after_data)
    {
        throw std::runtime_error(result.name + " warm workspace grew during repeated benchmark calls");
    }
}

void validateTimings(const CaseResult& result)
{
    const double timings[] = {
        result.time_cuda_cold_allocation_ms,
        result.time_cuda_warm_workspace_ms,
        result.time_cuda_ms,
        result.time_transfer_ms
    };
    for (double timing : timings)
    {
        if (!std::isfinite(timing) || timing < 0.0)
        {
            throw std::runtime_error(result.name + " produced a non-finite or negative CUDA timing");
        }
    }
}

} // namespace

void runElementwiseCuda(CaseResult& result, Index size)
{
    CudaStream stream;
    auto lhs_cpu = makeRandom(size, size);
    auto rhs_cpu = makeRandom(size, size);
    GpuFloatMatrix lhs(size, size);
    GpuFloatMatrix rhs(size, size);
    result.time_transfer_ms = measureTransfer(stream.get(), [&]()
    {
        lhs_cpu.copyToGpuAsync(lhs, stream.get());
        rhs_cpu.copyToGpuAsync(rhs, stream.get());
    });
    result.time_cuda_cold_allocation_ms = measureColdAllocation([&]()
    {
        auto output = hadamardMultiply(lhs, rhs, stream.get());
        volatile const float* sink = output.data();
        static_cast<void>(sink);
    });

    GpuFloatMatrix output(size, size);
    hadamardMultiplyAsync(lhs, rhs, output, stream.get());
    PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream.get()));
    const void* const output_data = output.data();
    const std::size_t output_bytes = static_cast<std::size_t>(output.size()) * sizeof(float);
    auto launch = [&]() { hadamardMultiplyAsync(lhs, rhs, output, stream.get()); };
    result.time_cuda_warm_workspace_ms = measureWarmTotal(stream.get(), launch, []() {});
    result.time_cuda_ms = measureEvent(stream.get(), launch, []() {});
    setWorkspaceSnapshot(result, output_bytes, output_bytes, output_data, output.data());
    validateTimings(result);
}

void runReductionCuda(CaseResult& result, Index size)
{
    CudaStream stream;
    auto input_cpu = makeRandom(size, size);
    GpuFloatMatrix input(size, size);
    result.time_transfer_ms = measureTransfer(stream.get(), [&]()
    {
        input_cpu.copyToGpuAsync(input, stream.get());
    });
    result.time_cuda_cold_allocation_ms = measureColdAllocation([&]()
    {
        auto output = sum(input, ReductionAxis::All);
        volatile const float* sink = output.data();
        static_cast<void>(sink);
    });

    GpuFloatMatrix output(1, 1);
    ReductionWorkspace workspace;
    sumAsync(input, ReductionAxis::All, output, workspace, stream.get());
    PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream.get()));
    const std::size_t workspace_bytes = workspace.capacityBytes();
    const void* const workspace_data = workspace.data();
    auto launch = [&]() { sumAsync(input, ReductionAxis::All, output, workspace, stream.get()); };
    result.time_cuda_warm_workspace_ms = measureWarmTotal(stream.get(), launch, []() {});
    result.time_cuda_ms = measureEvent(stream.get(), launch, []() {});
    setWorkspaceSnapshot(result, workspace_bytes, workspace.capacityBytes(),
                         workspace_data, workspace.data());
    validateTimings(result);
}

void runCompactCuda(CaseResult& result, Index size)
{
    CudaStream stream;
    auto input_cpu = makeRandom(size, 3);
    auto mask_cpu = makeMask(size);
    GpuFloatMatrix input(size, 3);
    GpuMaskMatrix mask(size, 1);
    result.time_transfer_ms = measureTransfer(stream.get(), [&]()
    {
        input_cpu.copyToGpuAsync(input, stream.get());
        mask_cpu.copyToGpuAsync(mask, stream.get());
    });
    result.time_cuda_cold_allocation_ms = measureColdAllocation([&]()
    {
        auto output = compactRows(input, mask);
        volatile const float* sink = output.values.data();
        static_cast<void>(sink);
    });

    GpuFloatMatrix values(size, 3);
    GpuIndexMatrix indices(size, 1);
    GpuIndexMatrix selected_count(1, 1);
    IndexingWorkspace workspace;
    auto launch = [&]()
    {
        compactRowsAsync(input, mask, values, indices, selected_count, workspace, stream.get());
    };
    auto check_status = [&]() { workspace.checkStatus("compact benchmark"); };
    launch();
    PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream.get()));
    check_status();
    const std::size_t workspace_bytes = workspace.capacityBytes();
    const void* const workspace_data = workspace.data();
    result.time_cuda_warm_workspace_ms = measureWarmTotal(stream.get(), launch, check_status);
    result.time_cuda_ms = measureEvent(stream.get(), launch, check_status);
    setWorkspaceSnapshot(result, workspace_bytes, workspace.capacityBytes(),
                         workspace_data, workspace.data());
    validateTimings(result);
}

void runEigh3x3BatchCuda(CaseResult& result, Index size)
{
    CudaStream stream;
    auto input_cpu = makeEighInput(size);
    GpuFloatMatrix input(size, 6);
    result.time_transfer_ms = measureTransfer(stream.get(), [&]()
    {
        input_cpu.copyToGpuAsync(input, stream.get());
    });
    result.time_cuda_cold_allocation_ms = measureColdAllocation([&]()
    {
        auto output = symmetricEigh3x3Batched(input);
        volatile const float* sink = output.eigenvalues.data();
        static_cast<void>(sink);
    });

    GpuFloatMatrix eigenvalues(size, 3);
    GpuFloatMatrix eigenvectors(size, 9);
    SymmetricEigh3x3Workspace workspace;
    auto launch = [&]()
    {
        symmetricEigh3x3BatchedAsync(
            input, eigenvalues, eigenvectors, workspace, stream.get());
    };
    auto check_status = [&]() { workspace.checkStatus("eigh3x3_batch benchmark"); };
    launch();
    PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream.get()));
    check_status();
    const std::size_t workspace_bytes = workspace.capacityBytes();
    const void* const workspace_data = workspace.data();
    result.time_cuda_warm_workspace_ms = measureWarmTotal(stream.get(), launch, check_status);
    result.time_cuda_ms = measureEvent(stream.get(), launch, check_status);
    setWorkspaceSnapshot(result, workspace_bytes, workspace.capacityBytes(),
                         workspace_data, workspace.data());
    validateTimings(result);
}

} // namespace detail
} // namespace plamatrix
