#include "block_schur_device_assembly.h"

#ifdef PLAMATRIX_WITH_VULKAN

#include "block_schur_sparse_assembly.h"
#include "../vulkan/iterative_solver_device.h"

#include "plamatrix/internal/vulkan/execution.h"
#include "plamatrix/internal/vulkan/runtime.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace plamatrix::internal::block_schur_detail
{
    namespace
    {

        constexpr std::size_t kTileElements = 16 * 16;
        constexpr float kMaximumHalfValue = 65504.0f;

        std::uint64_t nextAssemblyGeneration() noexcept
        {
            static std::atomic<std::uint64_t> generation{0};
            return generation.fetch_add(1, std::memory_order_relaxed) + 1;
        }

        std::uint32_t checkedUint(Index value, const char* name)
        {
            if (value < 0 || static_cast<std::uint64_t>(value) > std::numeric_limits<std::uint32_t>::max())
            {
                throw std::overflow_error(std::string("Vulkan Schur ") + name + " exceeds uint32 range");
            }
            return static_cast<std::uint32_t>(value);
        }

        std::vector<std::uint32_t> checkedUintVector(const std::vector<Index>& values, const char* name)
        {
            std::vector<std::uint32_t> result(values.size());
            std::transform(
                values.begin(), values.end(), result.begin(), [name](Index value) { return checkedUint(value, name); });
            return result;
        }

        std::uint32_t checkedUint(std::size_t value, const char* name)
        {
            if (value > std::numeric_limits<std::uint32_t>::max())
            {
                throw std::overflow_error(std::string("Vulkan Schur ") + name + " exceeds uint32 range");
            }
            return static_cast<std::uint32_t>(value);
        }

        float maximumMagnitude(const std::vector<float>& values, const char* name)
        {
            float result = 0.0f;
            int invalid = 0;
#pragma omp parallel for reduction(max : result) reduction(| : invalid) if (values.size() >= 65536)
            for (std::size_t index = 0; index < values.size(); ++index)
            {
                const float value = values[index];
                if (!std::isfinite(value))
                {
                    invalid = 1;
                }
                result = std::max(result, std::abs(value));
            }
            if (invalid != 0)
            {
                throw std::invalid_argument(std::string("Vulkan Schur ") + name + " contains a non-finite value");
            }
            return result;
        }

        class UploadedBuffer
        {
        public:
            UploadedBuffer(vulkan::Runtime& runtime, std::size_t bytes, VkBufferUsageFlags additional_usage = 0)
            {
                const VkDeviceSize allocation_bytes = static_cast<VkDeviceSize>(std::max<std::size_t>(1, bytes));
                _staging = std::make_unique<vulkan::Buffer>(
                    runtime, allocation_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, vulkan::BufferMemory::HostVisible);
                _device = std::make_unique<vulkan::Buffer>(runtime,
                                                           allocation_bytes,
                                                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                               VK_BUFFER_USAGE_TRANSFER_DST_BIT | additional_usage,
                                                           vulkan::BufferMemory::DeviceLocal);
            }

            void upload(const void* values, std::size_t bytes)
            {
                if (bytes > static_cast<std::size_t>(_staging->size()))
                {
                    throw std::out_of_range("Vulkan Schur upload exceeds staging buffer");
                }
                void* mapped = _staging->map();
                if (bytes != 0)
                {
                    std::memcpy(mapped, values, bytes);
                }
                else
                {
                    *static_cast<std::uint8_t*>(mapped) = 0;
                }
                _staging->unmap();
            }

            vulkan::Buffer& staging() noexcept
            {
                return *_staging;
            }

            vulkan::Buffer& device() noexcept
            {
                return *_device;
            }

        private:
            std::unique_ptr<vulkan::Buffer> _staging;
            std::unique_ptr<vulkan::Buffer> _device;
        };

        class DeviceBuffer
        {
        public:
            DeviceBuffer(vulkan::Runtime& runtime, std::size_t bytes, VkBufferUsageFlags additional_usage = 0)
            {
                _device = std::make_unique<vulkan::Buffer>(runtime,
                                                           static_cast<VkDeviceSize>(std::max<std::size_t>(1, bytes)),
                                                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | additional_usage,
                                                           vulkan::BufferMemory::DeviceLocal);
            }

            vulkan::Buffer& device() noexcept
            {
                return *_device;
            }

        private:
            std::unique_ptr<vulkan::Buffer> _device;
        };

        struct VulkanAssemblyState
        {
            VulkanAssemblyState(vulkan::Runtime& runtime,
                                std::size_t base_count,
                                std::size_t packed_count,
                                std::size_t cross_count,
                                std::size_t eliminated_count,
                                std::size_t term_count,
                                std::size_t value_count,
                                std::size_t slot_count,
                                std::uint32_t direct_offset,
                                std::uint32_t primary_size,
                                std::uint32_t eliminated_size,
                                std::size_t cross_input_count,
                                std::size_t inverse_input_count)
                : baseCount(base_count), packedCount(packed_count), crossCount(cross_count),
                  eliminatedCount(eliminated_count), termCount(term_count), valueCount(value_count),
                  slotCount(slot_count), directOffset(direct_offset), primarySize(primary_size),
                  eliminatedSize(eliminated_size), crossInputCount(cross_input_count),
                  inverseInputCount(inverse_input_count), generation(nextAssemblyGeneration()), baseHost(base_count),
                  pack(runtime,
                       "pack_schur_fp16",
                       3,
                       sizeof(std::uint32_t) * 4,
                       {VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT}),
                  conversion(runtime,
                             "convert_fp32_to_fp16",
                             2,
                             sizeof(std::uint32_t),
                             {VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT}),
                  transform(runtime,
                            "schur_transform_cooperative",
                            3,
                            sizeof(std::uint32_t) * 2,
                            {VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT}),
                  product(runtime,
                          "schur_product_cooperative",
                          6,
                          sizeof(std::uint32_t) * 2,
                          {VK_ACCESS_SHADER_READ_BIT,
                           VK_ACCESS_SHADER_READ_BIT,
                           VK_ACCESS_SHADER_READ_BIT,
                           VK_ACCESS_SHADER_READ_BIT,
                           VK_ACCESS_SHADER_READ_BIT,
                           VK_ACCESS_SHADER_WRITE_BIT}),
                  assemble(runtime,
                           "assemble_schur_products",
                           8,
                           sizeof(std::uint32_t) * 2,
                           {VK_ACCESS_SHADER_READ_BIT,
                            VK_ACCESS_SHADER_READ_BIT,
                            VK_ACCESS_SHADER_READ_BIT,
                            VK_ACCESS_SHADER_READ_BIT,
                            VK_ACCESS_SHADER_READ_BIT,
                            VK_ACCESS_SHADER_READ_BIT,
                            VK_ACCESS_SHADER_READ_BIT,
                            VK_ACCESS_SHADER_WRITE_BIT}),
                  base(runtime, base_count * sizeof(float)), crossInput(runtime, cross_input_count * sizeof(float)),
                  inverseInput(runtime, inverse_input_count * sizeof(float)),
                  crossEliminated(runtime, cross_count * sizeof(std::uint32_t)),
                  baseKinds(runtime, value_count * sizeof(std::uint32_t)),
                  baseIndices(runtime, value_count * sizeof(std::uint32_t)),
                  blockSlots(runtime, value_count * sizeof(std::uint32_t)),
                  localRows(runtime, value_count * sizeof(std::uint32_t)),
                  localColumns(runtime, value_count * sizeof(std::uint32_t)),
                  termOffsets(runtime, (slot_count + 1) * sizeof(std::uint32_t)),
                  termLeft(runtime, term_count * sizeof(std::uint32_t)),
                  termRight(runtime, term_count * sizeof(std::uint32_t)),
                  packedHalf(runtime, std::max<std::size_t>(1, packed_count) * sizeof(std::uint16_t)),
                  transformedFloat(runtime, cross_count * kTileElements * sizeof(float)),
                  transformedHalf(runtime, cross_count * kTileElements * sizeof(std::uint16_t)),
                  products(runtime, slot_count * kTileElements * sizeof(float)),
                  output(runtime, value_count * sizeof(float), VK_BUFFER_USAGE_TRANSFER_SRC_BIT)
            {
            }

            bool matches(std::size_t base_count,
                         std::size_t packed_count,
                         std::size_t cross_count,
                         std::size_t eliminated_count,
                         std::size_t term_count,
                         std::size_t value_count,
                         std::size_t slot_count,
                         std::uint32_t direct_offset,
                         std::uint32_t primary_size,
                         std::uint32_t eliminated_size,
                         std::size_t cross_input_count,
                         std::size_t inverse_input_count) const noexcept
            {
                return baseCount == base_count && packedCount == packed_count && crossCount == cross_count &&
                       eliminatedCount == eliminated_count && termCount == term_count && valueCount == value_count &&
                       slotCount == slot_count && directOffset == direct_offset && primarySize == primary_size &&
                       eliminatedSize == eliminated_size && crossInputCount == cross_input_count &&
                       inverseInputCount == inverse_input_count;
            }

            void record(vulkan::CommandContext& target, bool include_topology)
            {
                target.copy(base.staging(), base.device(), std::max<std::size_t>(1, baseCount * sizeof(float)));
                target.copy(crossInput.staging(),
                            crossInput.device(),
                            std::max<std::size_t>(1, crossInputCount * sizeof(float)));
                target.copy(inverseInput.staging(),
                            inverseInput.device(),
                            std::max<std::size_t>(1, inverseInputCount * sizeof(float)));
                if (include_topology)
                {
                    target.copy(crossEliminated.staging(),
                                crossEliminated.device(),
                                std::max<std::size_t>(1, crossCount * sizeof(std::uint32_t)));
                    target.copy(baseKinds.staging(),
                                baseKinds.device(),
                                std::max<std::size_t>(1, valueCount * sizeof(std::uint32_t)));
                    target.copy(baseIndices.staging(),
                                baseIndices.device(),
                                std::max<std::size_t>(1, valueCount * sizeof(std::uint32_t)));
                    target.copy(blockSlots.staging(),
                                blockSlots.device(),
                                std::max<std::size_t>(1, valueCount * sizeof(std::uint32_t)));
                    target.copy(localRows.staging(),
                                localRows.device(),
                                std::max<std::size_t>(1, valueCount * sizeof(std::uint32_t)));
                    target.copy(localColumns.staging(),
                                localColumns.device(),
                                std::max<std::size_t>(1, valueCount * sizeof(std::uint32_t)));
                    target.copy(termOffsets.staging(), termOffsets.device(), (slotCount + 1) * sizeof(std::uint32_t));
                    target.copy(termLeft.staging(),
                                termLeft.device(),
                                std::max<std::size_t>(1, termCount * sizeof(std::uint32_t)));
                    target.copy(termRight.staging(),
                                termRight.device(),
                                std::max<std::size_t>(1, termCount * sizeof(std::uint32_t)));
                }

                if (packedCount != 0)
                {
                    const struct
                    {
                        std::uint32_t crossCount;
                        std::uint32_t eliminatedCount;
                        std::uint32_t primarySize;
                        std::uint32_t eliminatedSize;
                    } pack_parameters{checkedUint(crossCount, "cross block count"),
                                      checkedUint(eliminatedCount, "eliminated block count"),
                                      primarySize,
                                      eliminatedSize};
                    target.dispatch(pack,
                                    {&crossInput.device(), &inverseInput.device(), &packedHalf.device()},
                                    crossCount * 2 + eliminatedCount,
                                    &pack_parameters,
                                    sizeof(pack_parameters));
                }
                if (crossCount != 0)
                {
                    const struct
                    {
                        std::uint32_t crossCount;
                        std::uint32_t inverseOffset;
                    } transform_parameters{checkedUint(crossCount, "cross block count"),
                                           checkedUint(crossCount * kTileElements * 2, "inverse offset")};
                    target.dispatch(transform,
                                    {&packedHalf.device(), &crossEliminated.device(), &transformedFloat.device()},
                                    crossCount,
                                    &transform_parameters,
                                    sizeof(transform_parameters));
                    const std::uint32_t transformed_count =
                        checkedUint(crossCount * kTileElements, "transformed element count");
                    target.dispatch(conversion,
                                    {&transformedFloat.device(), &transformedHalf.device()},
                                    (static_cast<std::size_t>(transformed_count) + 255) / 256,
                                    &transformed_count,
                                    sizeof(transformed_count));
                }
                if (slotCount != 0)
                {
                    const struct
                    {
                        std::uint32_t slotCount;
                        std::uint32_t transposeOffset;
                    } product_parameters{checkedUint(slotCount, "Schur block slot count"),
                                         checkedUint(crossCount * kTileElements, "transpose offset")};
                    target.dispatch(product,
                                    {&transformedHalf.device(),
                                     &packedHalf.device(),
                                     &termLeft.device(),
                                     &termRight.device(),
                                     &termOffsets.device(),
                                     &products.device()},
                                    slotCount,
                                    &product_parameters,
                                    sizeof(product_parameters));
                }
                const struct
                {
                    std::uint32_t valueCount;
                    std::uint32_t directOffset;
                } assembly_parameters{checkedUint(valueCount, "CSR value count"), directOffset};
                target.dispatch(assemble,
                                {&base.device(),
                                 &baseKinds.device(),
                                 &baseIndices.device(),
                                 &blockSlots.device(),
                                 &localRows.device(),
                                 &localColumns.device(),
                                 &products.device(),
                                 &output.device()},
                                (valueCount + 255) / 256,
                                &assembly_parameters,
                                sizeof(assembly_parameters));
            }

            std::size_t baseCount;
            std::size_t packedCount;
            std::size_t crossCount;
            std::size_t eliminatedCount;
            std::size_t termCount;
            std::size_t valueCount;
            std::size_t slotCount;
            std::uint32_t directOffset;
            std::uint32_t primarySize;
            std::uint32_t eliminatedSize;
            std::size_t crossInputCount;
            std::size_t inverseInputCount;
            std::uint64_t generation;
            std::uint64_t topologyGeneration = 0;
            bool uploadTopology = true;
            std::vector<float> baseHost;
            vulkan::ComputePipeline pack;
            vulkan::ComputePipeline conversion;
            vulkan::ComputePipeline transform;
            vulkan::ComputePipeline product;
            vulkan::ComputePipeline assemble;
            UploadedBuffer base;
            UploadedBuffer crossInput;
            UploadedBuffer inverseInput;
            UploadedBuffer crossEliminated;
            UploadedBuffer baseKinds;
            UploadedBuffer baseIndices;
            UploadedBuffer blockSlots;
            UploadedBuffer localRows;
            UploadedBuffer localColumns;
            UploadedBuffer termOffsets;
            UploadedBuffer termLeft;
            UploadedBuffer termRight;
            DeviceBuffer packedHalf;
            DeviceBuffer transformedFloat;
            DeviceBuffer transformedHalf;
            DeviceBuffer products;
            DeviceBuffer output;
        };

        template <typename Value> void upload(UploadedBuffer& buffer, const std::vector<Value>& values)
        {
            buffer.upload(values.data(), values.size() * sizeof(Value));
        }

    } // namespace

    std::vector<float> assembleSchurValuesOnVulkan(Index primary_size,
                                                   Index eliminated_size,
                                                   const std::vector<float>& primary_diagonal,
                                                   const std::vector<float>& eliminated_inverse,
                                                   const std::vector<float>& primary_cross_values,
                                                   const std::vector<float>& cross_values,
                                                   const std::vector<Index>& cross_eliminated_blocks,
                                                   const std::vector<Index>& base_kinds,
                                                   const std::vector<Index>& base_indices,
                                                   const std::vector<Index>& value_block_slots,
                                                   const std::vector<Index>& local_rows,
                                                   const std::vector<Index>& local_columns,
                                                   const std::vector<Index>& term_offsets,
                                                   const std::vector<Index>& term_left_cross,
                                                   const std::vector<Index>& term_right_cross,
                                                   SchurComplementSolverWorkspace<float>& workspace,
                                                   bool upload_topology)
    {
        if (primary_size <= 0 || primary_size > 16 || eliminated_size <= 0 || eliminated_size > 16)
        {
            throw std::invalid_argument("Vulkan cooperative Schur assembly requires block sizes in [1, 16]");
        }
        auto& runtime = vulkan::Runtime::instance();
        if (!runtime.cooperativeMatrixCapabilities().enabled)
        {
            throw std::runtime_error("Vulkan cooperative Schur assembly is unavailable on the selected device/build");
        }
        const std::size_t primary_block_area = static_cast<std::size_t>(primary_size * primary_size);
        const std::size_t cross_block_area = static_cast<std::size_t>(primary_size * eliminated_size);
        const std::size_t inverse_block_area = static_cast<std::size_t>(eliminated_size * eliminated_size);
        if (cross_values.size() % cross_block_area != 0 || eliminated_inverse.size() % inverse_block_area != 0 ||
            primary_cross_values.size() % primary_block_area != 0)
        {
            throw std::invalid_argument("Vulkan Schur numeric buffers do not match their block sizes");
        }
        const std::size_t cross_count = cross_values.size() / cross_block_area;
        const std::size_t eliminated_count = eliminated_inverse.size() / inverse_block_area;
        const std::size_t term_count = term_left_cross.size();
        const std::size_t value_count = base_kinds.size();
        const std::size_t slot_count = term_offsets.empty() ? 0 : term_offsets.size() - 1;
        if (cross_eliminated_blocks.size() != cross_count || term_right_cross.size() != term_count ||
            base_indices.size() != value_count || value_block_slots.size() != value_count ||
            local_rows.size() != value_count || local_columns.size() != value_count || slot_count == 0 ||
            value_count == 0)
        {
            throw std::invalid_argument("Vulkan Schur topology buffers have inconsistent sizes");
        }
        const float maximum_cross = maximumMagnitude(cross_values, "cross blocks");
        const float maximum_inverse = maximumMagnitude(eliminated_inverse, "eliminated inverses");
        const double transformed_bound =
            16.0 * static_cast<double>(maximum_cross) * static_cast<double>(maximum_inverse);
        if (maximum_cross > kMaximumHalfValue || maximum_inverse > kMaximumHalfValue ||
            transformed_bound > kMaximumHalfValue)
        {
            throw std::runtime_error("Vulkan Schur values exceed the safe FP16 cooperative-matrix range");
        }
        static_cast<void>(maximumMagnitude(primary_diagonal, "primary diagonal"));
        static_cast<void>(maximumMagnitude(primary_cross_values, "primary cross blocks"));

        if (cross_count > (std::numeric_limits<std::size_t>::max() / kTileElements - eliminated_count) / 2)
        {
            throw std::overflow_error("Vulkan Schur packed buffer size overflows size_t");
        }
        const std::size_t base_count = primary_diagonal.size() + primary_cross_values.size();
        const std::size_t packed_count = (cross_count * 2 + eliminated_count) * kTileElements;
        const std::uint32_t direct_offset = checkedUint(primary_diagonal.size(), "direct block offset");
        const std::uint32_t primary_size_uint = checkedUint(primary_size, "primary block size");
        const std::uint32_t eliminated_size_uint = checkedUint(eliminated_size, "eliminated block size");
        static_cast<void>(checkedUint(base_count, "base element count"));
        static_cast<void>(checkedUint(value_count, "CSR value count"));
        static_cast<void>(checkedUint(term_count, "product term count"));
        static_cast<void>(checkedUint(cross_count * kTileElements * 2, "inverse offset"));
        auto& opaque_state = SchurComplementSolverWorkspaceAccess::vulkanAssemblyState(workspace);
        auto state = std::static_pointer_cast<VulkanAssemblyState>(opaque_state);
        bool upload_current_topology = upload_topology;
        if (!state || !state->matches(base_count,
                                      packed_count,
                                      cross_count,
                                      eliminated_count,
                                      term_count,
                                      value_count,
                                      slot_count,
                                      direct_offset,
                                      primary_size_uint,
                                      eliminated_size_uint,
                                      cross_values.size(),
                                      eliminated_inverse.size()))
        {
            state = std::make_shared<VulkanAssemblyState>(runtime,
                                                          base_count,
                                                          packed_count,
                                                          cross_count,
                                                          eliminated_count,
                                                          term_count,
                                                          value_count,
                                                          slot_count,
                                                          direct_offset,
                                                          primary_size_uint,
                                                          eliminated_size_uint,
                                                          cross_values.size(),
                                                          eliminated_inverse.size());
            opaque_state = state;
            upload_current_topology = true;
        }

        auto& base_data = state->baseHost;
        std::copy(primary_diagonal.begin(), primary_diagonal.end(), base_data.begin());
        std::copy(primary_cross_values.begin(),
                  primary_cross_values.end(),
                  base_data.begin() + static_cast<std::ptrdiff_t>(primary_diagonal.size()));
        upload(state->base, base_data);
        upload(state->crossInput, cross_values);
        upload(state->inverseInput, eliminated_inverse);
        if (upload_current_topology)
        {
            if (term_offsets.front() != 0 || term_offsets.back() != static_cast<Index>(term_count) ||
                !std::is_sorted(term_offsets.begin(), term_offsets.end()))
            {
                throw std::invalid_argument("Vulkan Schur term offsets do not describe the product terms");
            }
            for (std::size_t cross = 0; cross < cross_count; ++cross)
            {
                if (cross_eliminated_blocks[cross] < 0 ||
                    static_cast<std::size_t>(cross_eliminated_blocks[cross]) >= eliminated_count)
                {
                    throw std::out_of_range("Vulkan Schur cross block references an invalid eliminated block");
                }
            }
            for (std::size_t term = 0; term < term_count; ++term)
            {
                if (term_left_cross[term] < 0 || term_right_cross[term] < 0 ||
                    static_cast<std::size_t>(term_left_cross[term]) >= cross_count ||
                    static_cast<std::size_t>(term_right_cross[term]) >= cross_count)
                {
                    throw std::out_of_range("Vulkan Schur product term references an invalid cross block");
                }
            }
            for (std::size_t value = 0; value < value_count; ++value)
            {
                const Index kind = base_kinds[value];
                const Index base_index = base_indices[value];
                const bool base_valid =
                    kind == 0 ||
                    (kind == 1 && base_index >= 0 && static_cast<std::size_t>(base_index) < primary_diagonal.size()) ||
                    (kind == 2 && base_index >= 0 &&
                     static_cast<std::size_t>(base_index) < primary_cross_values.size());
                if (!base_valid || value_block_slots[value] < 0 ||
                    static_cast<std::size_t>(value_block_slots[value]) >= slot_count || local_rows[value] < 0 ||
                    local_rows[value] >= primary_size || local_columns[value] < 0 ||
                    local_columns[value] >= primary_size)
                {
                    throw std::out_of_range("Vulkan Schur CSR value topology is invalid");
                }
            }
            const auto cross_eliminated = checkedUintVector(cross_eliminated_blocks, "cross eliminated index");
            const auto kinds = checkedUintVector(base_kinds, "base kind");
            const auto indices = checkedUintVector(base_indices, "base index");
            const auto slots = checkedUintVector(value_block_slots, "block slot");
            const auto rows = checkedUintVector(local_rows, "local row");
            const auto columns = checkedUintVector(local_columns, "local column");
            const auto offsets = checkedUintVector(term_offsets, "term offset");
            const auto left = checkedUintVector(term_left_cross, "left cross index");
            const auto right = checkedUintVector(term_right_cross, "right cross index");
            upload(state->crossEliminated, cross_eliminated);
            upload(state->baseKinds, kinds);
            upload(state->baseIndices, indices);
            upload(state->blockSlots, slots);
            upload(state->localRows, rows);
            upload(state->localColumns, columns);
            upload(state->termOffsets, offsets);
            upload(state->termLeft, left);
            upload(state->termRight, right);
        }
        if (upload_current_topology)
        {
            ++state->topologyGeneration;
        }
        state->uploadTopology = state->uploadTopology || upload_current_topology;
        return {};
    }

    IterativeSolverReport solveLastVulkanSchurValues(const CsrStorage<float, Device::CPU>& matrix,
                                                     const DenseStorage<float, Device::CPU>& rhs,
                                                     DenseStorage<float, Device::CPU>& solution,
                                                     const DenseStorage<float, Device::CPU>* inverse_blocks,
                                                     Index block_size,
                                                     bool build_device_block_jacobi,
                                                     SchurComplementSolverWorkspace<float>& workspace,
                                                     const IterativeSolverOptions& options)
    {
        auto state = std::static_pointer_cast<VulkanAssemblyState>(
            SchurComplementSolverWorkspaceAccess::vulkanAssemblyState(workspace));
        if (!state || state->valueCount != static_cast<std::size_t>(matrix.nnz()))
        {
            throw std::logic_error("Vulkan Schur values have not been assembled for this matrix");
        }
        const bool upload_topology = state->uploadTopology;
        const vulkan::DevicePcgPrefixRecorder record_assembly =
            [state, upload_topology](vulkan::CommandContext& context) { state->record(context, upload_topology); };
        auto report = vulkan::blockPcgWithDeviceValues(matrix,
                                                       rhs,
                                                       solution,
                                                       inverse_blocks,
                                                       block_size,
                                                       state->output.device(),
                                                       state->generation,
                                                       state->topologyGeneration,
                                                       record_assembly,
                                                       build_device_block_jacobi,
                                                       options);
        state->uploadTopology = false;
        return report;
    }

} // namespace plamatrix::internal::block_schur_detail

#endif
