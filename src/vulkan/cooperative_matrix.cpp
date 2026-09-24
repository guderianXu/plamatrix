#include "plamatrix/internal/vulkan/cooperative_matrix.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

#ifdef PLAMATRIX_WITH_VULKAN
#include <memory>
#include <mutex>

#include "plamatrix/internal/vulkan/execution.h"
#include "plamatrix/internal/vulkan/runtime.h"
#endif

namespace plamatrix::internal::vulkan
{
    namespace
    {

        constexpr std::size_t kMatrixElements = 16 * 16;
        constexpr float kMaximumHalfValue = 65504.0f;

        std::size_t elementCount(std::size_t batch_count)
        {
            if (batch_count > std::numeric_limits<std::size_t>::max() / kMatrixElements)
            {
                throw std::overflow_error("batchedMultiply16x16 batch size overflows addressable memory");
            }
            return batch_count * kMatrixElements;
        }

        void validateInputs(const std::vector<float>& left, const std::vector<float>& right, std::size_t batch_count)
        {
            const std::size_t expected = elementCount(batch_count);
            if (left.size() != expected || right.size() != expected)
            {
                throw std::invalid_argument("batchedMultiply16x16 inputs must each contain batchCount * 256 elements");
            }
        }

        std::vector<float>
        multiplyOnCpu(const std::vector<float>& left, const std::vector<float>& right, std::size_t batch_count)
        {
            std::vector<float> output(elementCount(batch_count), 0.0f);
            for (std::size_t batch = 0; batch < batch_count; ++batch)
            {
                const std::size_t base = batch * kMatrixElements;
                for (std::size_t row = 0; row < 16; ++row)
                {
                    for (std::size_t column = 0; column < 16; ++column)
                    {
                        float value = 0.0f;
                        for (std::size_t inner = 0; inner < 16; ++inner)
                        {
                            value += left[base + row * 16 + inner] * right[base + inner * 16 + column];
                        }
                        output[base + row * 16 + column] = value;
                    }
                }
            }
            return output;
        }

        bool isSafeForHalfInputs(const std::vector<float>& left, const std::vector<float>& right)
        {
            float maximum_left = 0.0f;
            float maximum_right = 0.0f;
            for (float value : left)
            {
                if (!std::isfinite(value) || std::abs(value) > kMaximumHalfValue)
                {
                    return false;
                }
                maximum_left = std::max(maximum_left, std::abs(value));
            }
            for (float value : right)
            {
                if (!std::isfinite(value) || std::abs(value) > kMaximumHalfValue)
                {
                    return false;
                }
                maximum_right = std::max(maximum_right, std::abs(value));
            }
            const double accumulation_bound =
                16.0 * static_cast<double>(maximum_left) * static_cast<double>(maximum_right);
            return accumulation_bound <= static_cast<double>(std::numeric_limits<float>::max());
        }

#ifdef PLAMATRIX_WITH_VULKAN
        class CooperativeMatrixState
        {
        public:
            explicit CooperativeMatrixState(Runtime& runtime)
                : _runtime(runtime), _conversionPipeline(runtime,
                                                         "convert_fp32_to_fp16",
                                                         2,
                                                         sizeof(std::uint32_t),
                                                         {VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT}),
                  _pipeline(runtime,
                            "batched_gemm_cooperative",
                            2,
                            sizeof(std::uint32_t) * 2,
                            {VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT})
            {
            }

            void ensureCapacity(std::size_t element_count)
            {
                if (_capacity == element_count)
                {
                    return;
                }
                const VkDeviceSize input_float_bytes = static_cast<VkDeviceSize>(element_count * sizeof(float) * 2);
                const VkDeviceSize input_half_bytes =
                    static_cast<VkDeviceSize>(element_count * sizeof(std::uint16_t) * 2);
                const VkDeviceSize float_bytes = static_cast<VkDeviceSize>(element_count * sizeof(float));
                _inputStaging = std::make_unique<Buffer>(
                    _runtime, input_float_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, BufferMemory::HostVisible);
                _outputStaging = std::make_unique<Buffer>(
                    _runtime, float_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, BufferMemory::HostVisibleCached);
                _inputFloat =
                    std::make_unique<Buffer>(_runtime,
                                             input_float_bytes,
                                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                             BufferMemory::DeviceLocal);
                _inputHalf = std::make_unique<Buffer>(
                    _runtime, input_half_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, BufferMemory::DeviceLocal);
                _output =
                    std::make_unique<Buffer>(_runtime,
                                             float_bytes,
                                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                             BufferMemory::DeviceLocal);
                _context = std::make_unique<CommandContext>(_runtime, 2, 4);
                recordCommands(element_count);
                _capacity = element_count;
            }

            std::vector<float> multiply(const std::vector<float>& left,
                                        const std::vector<float>& right,
                                        std::size_t batch_count,
                                        double* gpu_milliseconds)
            {
                const std::size_t count = elementCount(batch_count);
                ensureCapacity(count);
                auto* mapped_input = static_cast<float*>(_inputStaging->map());
                std::copy(left.begin(), left.end(), mapped_input);
                std::copy(right.begin(), right.end(), mapped_input + count);
                _inputStaging->unmap();
                _context->resetSubmissionCount();
                _context->submitAndWait();

                std::vector<float> result(count);
                const auto* mapped = static_cast<const float*>(_outputStaging->map());
                std::copy(mapped, mapped + count, result.begin());
                _outputStaging->unmap();
                if (gpu_milliseconds)
                {
                    *gpu_milliseconds = _context->gpuMilliseconds();
                }
                return result;
            }

        private:
            void recordCommands(std::size_t element_count)
            {
                const VkDeviceSize input_float_bytes = static_cast<VkDeviceSize>(element_count * sizeof(float) * 2);
                const VkDeviceSize float_bytes = static_cast<VkDeviceSize>(element_count * sizeof(float));
                const std::uint32_t input_element_count = static_cast<std::uint32_t>(element_count * 2);
                const std::size_t batch_count = element_count / kMatrixElements;
                const struct
                {
                    std::uint32_t batchCount;
                    std::uint32_t rightOffset;
                } parameters{static_cast<std::uint32_t>(batch_count), static_cast<std::uint32_t>(element_count)};

                _context->beginReusable();
                _context->copy(*_inputStaging, *_inputFloat, input_float_bytes);
                _context->dispatch(_conversionPipeline,
                                   {_inputFloat.get(), _inputHalf.get()},
                                   (static_cast<std::size_t>(input_element_count) + 255) / 256,
                                   &input_element_count,
                                   sizeof(input_element_count));
                _context->dispatch(
                    _pipeline, {_inputHalf.get(), _output.get()}, batch_count, &parameters, sizeof(parameters));
                _context->copy(*_output, *_outputStaging, float_bytes);
            }

            Runtime& _runtime;
            ComputePipeline _conversionPipeline;
            ComputePipeline _pipeline;
            std::unique_ptr<CommandContext> _context;
            std::unique_ptr<Buffer> _inputStaging;
            std::unique_ptr<Buffer> _outputStaging;
            std::unique_ptr<Buffer> _inputFloat;
            std::unique_ptr<Buffer> _inputHalf;
            std::unique_ptr<Buffer> _output;
            std::size_t _capacity = 0;
        };
#endif

    } // namespace

    std::vector<float> batchedMultiply16x16(const std::vector<float>& left,
                                            const std::vector<float>& right,
                                            std::size_t batchCount,
                                            const BatchedMatrixMultiplyOptions& options,
                                            BatchedMatrixMultiplyReport* report)
    {
        validateInputs(left, right, batchCount);
        BatchedMatrixMultiplyReport local_report;
        local_report.batchCount = batchCount;
        if (batchCount == 0)
        {
            if (report)
            {
                *report = local_report;
            }
            return {};
        }
        if (options.requireCooperativeMatrix && !options.allowMixedPrecision)
        {
            throw std::invalid_argument(
                "batchedMultiply16x16 requires allowMixedPrecision when cooperative matrix is required");
        }

        const bool safe_for_half = isSafeForHalfInputs(left, right);
#ifdef PLAMATRIX_WITH_VULKAN
        const bool fits_vulkan_indices = batchCount <= std::numeric_limits<std::uint32_t>::max() &&
                                         elementCount(batchCount) <= std::numeric_limits<std::uint32_t>::max() / 2;
        if (options.allowMixedPrecision && safe_for_half && fits_vulkan_indices)
        {
            Runtime* runtime = nullptr;
            try
            {
                runtime = &Runtime::instance();
            }
            catch (const std::exception&)
            {
                if (options.requireCooperativeMatrix)
                {
                    throw;
                }
            }
            if (runtime && runtime->cooperativeMatrixCapabilities().enabled)
            {
                static std::mutex execution_mutex;
                std::lock_guard<std::mutex> lock(execution_mutex);
                static std::unique_ptr<CooperativeMatrixState> state;
                if (!state)
                {
                    state = std::make_unique<CooperativeMatrixState>(*runtime);
                }
                auto result = state->multiply(left, right, batchCount, &local_report.gpuMilliseconds);
                if (std::all_of(result.begin(), result.end(), [](float value) { return std::isfinite(value); }))
                {
                    local_report.backend = BatchedMatrixMultiplyBackend::VulkanCooperativeMatrix;
                    local_report.deviceName = runtime->deviceName();
                    if (report)
                    {
                        *report = local_report;
                    }
                    return result;
                }
                local_report.numericalSafetyFallback = true;
                if (options.requireCooperativeMatrix)
                {
                    throw std::runtime_error("Vulkan cooperative-matrix result contained a non-finite value");
                }
            }
            else if (options.requireCooperativeMatrix)
            {
                throw std::runtime_error(
                    "Vulkan FP16-input/FP32-accumulator 16x16 cooperative matrices are unavailable");
            }
        }
        else if (options.requireCooperativeMatrix)
        {
            if (!fits_vulkan_indices)
            {
                throw std::overflow_error("batchedMultiply16x16 input exceeds Vulkan shader index range");
            }
            throw std::runtime_error("batchedMultiply16x16 input is outside the safe FP16 mixed-precision range");
        }
#else
        if (options.requireCooperativeMatrix)
        {
            throw std::runtime_error("Vulkan cooperative matrices require PLAMATRIX_WITH_VULKAN=ON");
        }
#endif

        if (options.allowMixedPrecision && !safe_for_half)
        {
            local_report.numericalSafetyFallback = true;
        }
        auto result = multiplyOnCpu(left, right, batchCount);
        if (report)
        {
            *report = local_report;
        }
        return result;
    }

} // namespace plamatrix::internal::vulkan
