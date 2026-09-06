#include "block_schur_device_assembly.h"

#include "block_schur_sparse_assembly.h"

#include "plamatrix/core/error_check.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <utility>

namespace plamatrix::block_schur_detail
{
    namespace
    {

        template <typename Value> class DeviceArray
        {
        public:
            DeviceArray() = default;

            ~DeviceArray()
            {
                cudaFree(_data);
            }

            bool ensure(std::size_t count)
            {
                const std::size_t required = std::max<std::size_t>(1, count);
                if (_count == required)
                {
                    return false;
                }
                if (_data)
                {
                    PLAMATRIX_CHECK_CUDA(cudaFree(_data));
                }
                PLAMATRIX_CHECK_CUDA(cudaMalloc(&_data, required * sizeof(Value)));
                _count = required;
                return true;
            }

            void upload(const std::vector<Value>& values)
            {
                ensure(values.size());
                if (values.empty())
                {
                    PLAMATRIX_CHECK_CUDA(cudaMemset(_data, 0, sizeof(Value)));
                }
                else
                {
                    PLAMATRIX_CHECK_CUDA(
                        cudaMemcpy(_data, values.data(), values.size() * sizeof(Value), cudaMemcpyHostToDevice));
                }
            }

            DeviceArray(const DeviceArray&) = delete;
            DeviceArray& operator=(const DeviceArray&) = delete;

            Value* data() noexcept
            {
                return _data;
            }

        private:
            Value* _data = nullptr;
            std::size_t _count = 0;
        };

        template <typename Scalar> struct CudaAssemblyState
        {
            DeviceArray<Scalar> primary;
            DeviceArray<Scalar> inverse;
            DeviceArray<Scalar> primaryCross;
            DeviceArray<Scalar> cross;
            DeviceArray<Index> crossEliminatedBlocks;
            DeviceArray<Scalar> transformedCross;
            DeviceArray<Index> baseKinds;
            DeviceArray<Index> baseIndices;
            DeviceArray<Index> blockSlots;
            DeviceArray<Index> rows;
            DeviceArray<Index> columns;
            DeviceArray<Index> termOffsets;
            DeviceArray<Index> termEliminated;
            DeviceArray<Index> termLeft;
            DeviceArray<Index> termRight;
            DeviceArray<Scalar> output;
        };

        template <typename Scalar>
        __global__ void transformCross3Kernel(Index cross_row_count,
                                              Index primary_size,
                                              const Scalar* eliminated_inverse,
                                              const Scalar* cross_values,
                                              const Index* cross_eliminated_blocks,
                                              Scalar* transformed_cross)
        {
            const Index cross_row = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (cross_row >= cross_row_count)
            {
                return;
            }
            const Index cross = cross_row / primary_size;
            const Scalar* inverse = eliminated_inverse + cross_eliminated_blocks[cross] * 9;
            const Scalar* input = cross_values + cross_row * 3;
            Scalar* output = transformed_cross + cross_row * 3;
            const Scalar first = input[0];
            const Scalar second = input[1];
            const Scalar third = input[2];
            output[0] = inverse[0] * first + inverse[1] * second + inverse[2] * third;
            output[1] = inverse[3] * first + inverse[4] * second + inverse[5] * third;
            output[2] = inverse[6] * first + inverse[7] * second + inverse[8] * third;
        }

        template <typename Scalar>
        __global__ void assembleSchurValues3Kernel(Index value_count,
                                                   Index primary_size,
                                                   const Scalar* primary_diagonal,
                                                   const Scalar* primary_cross_values,
                                                   const Scalar* cross_values,
                                                   const Scalar* transformed_cross,
                                                   const Index* base_kinds,
                                                   const Index* base_indices,
                                                   const Index* value_block_slots,
                                                   const Index* local_rows,
                                                   const Index* local_columns,
                                                   const Index* term_offsets,
                                                   const Index* term_left_cross,
                                                   const Index* term_right_cross,
                                                   Scalar* output)
        {
            const Index value_index = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (value_index >= value_count)
            {
                return;
            }
            Scalar value = Scalar(0);
            if (base_kinds[value_index] == 1)
            {
                value = primary_diagonal[base_indices[value_index]];
            }
            else if (base_kinds[value_index] == 2)
            {
                value = primary_cross_values[base_indices[value_index]];
            }
            const Index row = local_rows[value_index];
            const Index column = local_columns[value_index];
            const Index slot = value_block_slots[value_index];
            const Index cross_stride = primary_size * 3;
            for (Index term = term_offsets[slot]; term < term_offsets[slot + 1]; ++term)
            {
                const Scalar* left = transformed_cross + term_left_cross[term] * cross_stride + row * 3;
                const Scalar* right = cross_values + term_right_cross[term] * cross_stride + column * 3;
                value -= left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
            }
            output[value_index] = value;
        }

        template <typename Scalar>
        __global__ void assembleSchurValuesKernel(Index value_count,
                                                  Index primary_size,
                                                  Index eliminated_size,
                                                  const Scalar* primary_diagonal,
                                                  const Scalar* eliminated_inverse,
                                                  const Scalar* primary_cross_values,
                                                  const Scalar* cross_values,
                                                  const Index* base_kinds,
                                                  const Index* base_indices,
                                                  const Index* value_block_slots,
                                                  const Index* local_rows,
                                                  const Index* local_columns,
                                                  const Index* term_offsets,
                                                  const Index* term_eliminated,
                                                  const Index* term_left_cross,
                                                  const Index* term_right_cross,
                                                  Scalar* output)
        {
            const Index value_index = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (value_index >= value_count)
            {
                return;
            }
            Scalar value = Scalar(0);
            if (base_kinds[value_index] == 1)
            {
                value = primary_diagonal[base_indices[value_index]];
            }
            else if (base_kinds[value_index] == 2)
            {
                value = primary_cross_values[base_indices[value_index]];
            }
            const Index row = local_rows[value_index];
            const Index column = local_columns[value_index];
            const Index slot = value_block_slots[value_index];
            const Index cross_stride = primary_size * eliminated_size;
            const Index inverse_stride = eliminated_size * eliminated_size;
            for (Index term = term_offsets[slot]; term < term_offsets[slot + 1]; ++term)
            {
                const Scalar* left = cross_values + term_left_cross[term] * cross_stride;
                const Scalar* right = cross_values + term_right_cross[term] * cross_stride;
                const Scalar* inverse = eliminated_inverse + term_eliminated[term] * inverse_stride;
                Scalar product = Scalar(0);
                for (Index inner_row = 0; inner_row < eliminated_size; ++inner_row)
                {
                    for (Index inner_column = 0; inner_column < eliminated_size; ++inner_column)
                    {
                        product += left[row * eliminated_size + inner_row] *
                                   inverse[inner_row * eliminated_size + inner_column] *
                                   right[column * eliminated_size + inner_column];
                    }
                }
                value -= product;
            }
            output[value_index] = value;
        }

        template <typename Scalar>
        std::vector<Scalar> assemble(Index primary_size,
                                     Index eliminated_size,
                                     const std::vector<Scalar>& primary_diagonal,
                                     const std::vector<Scalar>& eliminated_inverse,
                                     const std::vector<Scalar>& primary_cross_values,
                                     const std::vector<Scalar>& cross_values,
                                     const std::vector<Index>& cross_eliminated_blocks,
                                     const std::vector<Index>& base_kinds,
                                     const std::vector<Index>& base_indices,
                                     const std::vector<Index>& value_block_slots,
                                     const std::vector<Index>& local_rows,
                                     const std::vector<Index>& local_columns,
                                     const std::vector<Index>& term_offsets,
                                     const std::vector<Index>& term_eliminated,
                                     const std::vector<Index>& term_left_cross,
                                     const std::vector<Index>& term_right_cross,
                                     SchurComplementSolverWorkspace<Scalar>& workspace,
                                     bool upload_topology)
        {
            const std::size_t value_count = base_kinds.size();
            auto& opaque_state = SchurComplementSolverWorkspaceAccess::deviceAssemblyState(workspace);
            auto state = std::static_pointer_cast<CudaAssemblyState<Scalar>>(opaque_state);
            if (!state)
            {
                state = std::make_shared<CudaAssemblyState<Scalar>>();
                opaque_state = state;
                upload_topology = true;
            }
            state->primary.upload(primary_diagonal);
            state->inverse.upload(eliminated_inverse);
            state->primaryCross.upload(primary_cross_values);
            state->cross.upload(cross_values);
            const bool resized_topology =
                state->baseKinds.ensure(base_kinds.size()) | state->baseIndices.ensure(base_indices.size()) |
                state->blockSlots.ensure(value_block_slots.size()) | state->rows.ensure(local_rows.size()) |
                state->columns.ensure(local_columns.size()) | state->termOffsets.ensure(term_offsets.size()) |
                state->termEliminated.ensure(term_eliminated.size()) | state->termLeft.ensure(term_left_cross.size()) |
                state->termRight.ensure(term_right_cross.size()) |
                state->crossEliminatedBlocks.ensure(cross_eliminated_blocks.size());
            if (upload_topology || resized_topology)
            {
                state->baseKinds.upload(base_kinds);
                state->baseIndices.upload(base_indices);
                state->blockSlots.upload(value_block_slots);
                state->rows.upload(local_rows);
                state->columns.upload(local_columns);
                state->termOffsets.upload(term_offsets);
                state->termEliminated.upload(term_eliminated);
                state->termLeft.upload(term_left_cross);
                state->termRight.upload(term_right_cross);
                state->crossEliminatedBlocks.upload(cross_eliminated_blocks);
            }
            state->output.ensure(value_count);

            constexpr int block_size = 256;
            const int grid_size = static_cast<int>((value_count + block_size - 1) / block_size);
            if (eliminated_size == 3)
            {
                const std::size_t transformed_count = cross_values.size();
                state->transformedCross.ensure(transformed_count);
                const Index cross_row_count = static_cast<Index>(transformed_count / 3);
                if (cross_row_count > 0)
                {
                    const int transform_grid_size = static_cast<int>((cross_row_count + block_size - 1) / block_size);
                    transformCross3Kernel<<<transform_grid_size, block_size>>>(cross_row_count,
                                                                               primary_size,
                                                                               state->inverse.data(),
                                                                               state->cross.data(),
                                                                               state->crossEliminatedBlocks.data(),
                                                                               state->transformedCross.data());
                    PLAMATRIX_CHECK_CUDA(cudaGetLastError());
                }
                assembleSchurValues3Kernel<<<grid_size, block_size>>>(static_cast<Index>(value_count),
                                                                      primary_size,
                                                                      state->primary.data(),
                                                                      state->primaryCross.data(),
                                                                      state->cross.data(),
                                                                      state->transformedCross.data(),
                                                                      state->baseKinds.data(),
                                                                      state->baseIndices.data(),
                                                                      state->blockSlots.data(),
                                                                      state->rows.data(),
                                                                      state->columns.data(),
                                                                      state->termOffsets.data(),
                                                                      state->termLeft.data(),
                                                                      state->termRight.data(),
                                                                      state->output.data());
            }
            else
            {
                assembleSchurValuesKernel<<<grid_size, block_size>>>(static_cast<Index>(value_count),
                                                                     primary_size,
                                                                     eliminated_size,
                                                                     state->primary.data(),
                                                                     state->inverse.data(),
                                                                     state->primaryCross.data(),
                                                                     state->cross.data(),
                                                                     state->baseKinds.data(),
                                                                     state->baseIndices.data(),
                                                                     state->blockSlots.data(),
                                                                     state->rows.data(),
                                                                     state->columns.data(),
                                                                     state->termOffsets.data(),
                                                                     state->termEliminated.data(),
                                                                     state->termLeft.data(),
                                                                     state->termRight.data(),
                                                                     state->output.data());
            }
            PLAMATRIX_CHECK_CUDA(cudaGetLastError());
            return {};
        }

        template <typename Scalar>
        void copyLastCudaSchurValuesToDeviceImpl(Scalar* destination,
                                                 std::size_t value_count,
                                                 SchurComplementSolverWorkspace<Scalar>& workspace)
        {
            if (!destination)
            {
                throw std::invalid_argument("CUDA Schur destination is null");
            }
            auto state = std::static_pointer_cast<CudaAssemblyState<Scalar>>(
                SchurComplementSolverWorkspaceAccess::deviceAssemblyState(workspace));
            if (!state)
            {
                throw std::logic_error("CUDA Schur values have not been assembled");
            }
            PLAMATRIX_CHECK_CUDA(
                cudaMemcpy(destination, state->output.data(), value_count * sizeof(Scalar), cudaMemcpyDeviceToDevice));
        }

    } // namespace

    template <typename Scalar>
    void copyLastCudaSchurValuesToDevice(Scalar* destination,
                                         std::size_t value_count,
                                         SchurComplementSolverWorkspace<Scalar>& workspace)
    {
        copyLastCudaSchurValuesToDeviceImpl(destination, value_count, workspace);
    }

    template <typename Scalar>
    std::vector<Scalar> assembleSchurValuesOnCuda(Index primary_size,
                                                  Index eliminated_size,
                                                  const std::vector<Scalar>& primary_diagonal,
                                                  const std::vector<Scalar>& eliminated_inverse,
                                                  const std::vector<Scalar>& primary_cross_values,
                                                  const std::vector<Scalar>& cross_values,
                                                  const std::vector<Index>& cross_eliminated_blocks,
                                                  const std::vector<Index>& base_kinds,
                                                  const std::vector<Index>& base_indices,
                                                  const std::vector<Index>& value_block_slots,
                                                  const std::vector<Index>& local_rows,
                                                  const std::vector<Index>& local_columns,
                                                  const std::vector<Index>& term_offsets,
                                                  const std::vector<Index>& term_eliminated,
                                                  const std::vector<Index>& term_left_cross,
                                                  const std::vector<Index>& term_right_cross,
                                                  SchurComplementSolverWorkspace<Scalar>& workspace,
                                                  bool upload_topology)
    {
        return assemble(primary_size,
                        eliminated_size,
                        primary_diagonal,
                        eliminated_inverse,
                        primary_cross_values,
                        cross_values,
                        cross_eliminated_blocks,
                        base_kinds,
                        base_indices,
                        value_block_slots,
                        local_rows,
                        local_columns,
                        term_offsets,
                        term_eliminated,
                        term_left_cross,
                        term_right_cross,
                        workspace,
                        upload_topology);
    }

    template std::vector<float> assembleSchurValuesOnCuda(Index,
                                                          Index,
                                                          const std::vector<float>&,
                                                          const std::vector<float>&,
                                                          const std::vector<float>&,
                                                          const std::vector<float>&,
                                                          const std::vector<Index>&,
                                                          const std::vector<Index>&,
                                                          const std::vector<Index>&,
                                                          const std::vector<Index>&,
                                                          const std::vector<Index>&,
                                                          const std::vector<Index>&,
                                                          const std::vector<Index>&,
                                                          const std::vector<Index>&,
                                                          const std::vector<Index>&,
                                                          const std::vector<Index>&,
                                                          SchurComplementSolverWorkspace<float>&,
                                                          bool);
    template std::vector<double> assembleSchurValuesOnCuda(Index,
                                                           Index,
                                                           const std::vector<double>&,
                                                           const std::vector<double>&,
                                                           const std::vector<double>&,
                                                           const std::vector<double>&,
                                                           const std::vector<Index>&,
                                                           const std::vector<Index>&,
                                                           const std::vector<Index>&,
                                                           const std::vector<Index>&,
                                                           const std::vector<Index>&,
                                                           const std::vector<Index>&,
                                                           const std::vector<Index>&,
                                                           const std::vector<Index>&,
                                                           const std::vector<Index>&,
                                                           const std::vector<Index>&,
                                                           SchurComplementSolverWorkspace<double>&,
                                                           bool);
    template void copyLastCudaSchurValuesToDevice(float*, std::size_t, SchurComplementSolverWorkspace<float>&);
    template void copyLastCudaSchurValuesToDevice(double*, std::size_t, SchurComplementSolverWorkspace<double>&);

} // namespace plamatrix::block_schur_detail
