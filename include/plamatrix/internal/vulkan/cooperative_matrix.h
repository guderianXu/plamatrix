#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace plamatrix::internal::vulkan
{

    enum class BatchedMatrixMultiplyBackend
    {
        CpuFp32,
        VulkanCooperativeMatrix,
    };

    struct BatchedMatrixMultiplyOptions
    {
        /// Opt in to FP16 input quantization with FP32 cooperative-matrix accumulation.
        bool allowMixedPrecision = false;
        /// Throw instead of falling back to CPU FP32 when the Vulkan path is unavailable or unsafe.
        bool requireCooperativeMatrix = false;
    };

    struct BatchedMatrixMultiplyReport
    {
        BatchedMatrixMultiplyBackend backend = BatchedMatrixMultiplyBackend::CpuFp32;
        std::size_t batchCount = 0;
        double gpuMilliseconds = 0.0;
        bool numericalSafetyFallback = false;
        std::string deviceName;
    };

    /**
     * Multiply equally sized batches of row-major 16x16 float matrices.
     *
     * Mixed precision is explicit. When enabled and supported, inputs are converted to FP16 and
     * multiplied by a Vulkan cooperative-matrix shader with FP32 accumulation. Unsupported or
     * numerically unsafe inputs use the CPU FP32 implementation unless `requireCooperativeMatrix`
     * is true. Input vectors must both contain `batchCount * 256` elements.
     */
    std::vector<float> batchedMultiply16x16(const std::vector<float>& left,
                                            const std::vector<float>& right,
                                            std::size_t batchCount,
                                            const BatchedMatrixMultiplyOptions& options = {},
                                            BatchedMatrixMultiplyReport* report = nullptr);

} // namespace plamatrix::internal::vulkan
