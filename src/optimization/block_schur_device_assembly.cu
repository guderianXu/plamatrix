#include "block_schur_device_assembly.h"

#include "plamatrix/core/error_check.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <utility>

namespace plamatrix::block_schur_detail
{
namespace
{

template <typename Value>
class DeviceArray
{
public:
    explicit DeviceArray(const std::vector<Value>& values)
        : _count(std::max<std::size_t>(1, values.size()))
    {
        PLAMATRIX_CHECK_CUDA(cudaMalloc(&_data, _count * sizeof(Value)));
        if (values.empty())
        {
            PLAMATRIX_CHECK_CUDA(cudaMemset(_data, 0, sizeof(Value)));
        }
        else
        {
            PLAMATRIX_CHECK_CUDA(cudaMemcpy(
                _data, values.data(), values.size() * sizeof(Value), cudaMemcpyHostToDevice));
        }
    }

    explicit DeviceArray(std::size_t count)
        : _count(std::max<std::size_t>(1, count))
    {
        PLAMATRIX_CHECK_CUDA(cudaMalloc(&_data, _count * sizeof(Value)));
    }

    ~DeviceArray()
    {
        cudaFree(_data);
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

template <typename Scalar>
__global__ void assembleSchurValuesKernel(
    Index value_count,
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
    for (Index term = term_offsets[slot];
         term < term_offsets[slot + 1];
         ++term)
    {
        const Scalar* left = cross_values + term_left_cross[term] * cross_stride;
        const Scalar* right = cross_values + term_right_cross[term] * cross_stride;
        const Scalar* inverse = eliminated_inverse + term_eliminated[term] * inverse_stride;
        Scalar product = Scalar(0);
        for (Index inner_row = 0; inner_row < eliminated_size; ++inner_row)
        {
            for (Index inner_column = 0;
                 inner_column < eliminated_size;
                 ++inner_column)
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
std::vector<Scalar> assemble(
    Index primary_size,
    Index eliminated_size,
    const std::vector<Scalar>& primary_diagonal,
    const std::vector<Scalar>& eliminated_inverse,
    const std::vector<Scalar>& primary_cross_values,
    const std::vector<Scalar>& cross_values,
    const std::vector<Index>& base_kinds,
    const std::vector<Index>& base_indices,
    const std::vector<Index>& value_block_slots,
    const std::vector<Index>& local_rows,
    const std::vector<Index>& local_columns,
    const std::vector<Index>& term_offsets,
    const std::vector<Index>& term_eliminated,
    const std::vector<Index>& term_left_cross,
    const std::vector<Index>& term_right_cross)
{
    const std::size_t value_count = base_kinds.size();
    DeviceArray<Scalar> device_primary(primary_diagonal);
    DeviceArray<Scalar> device_inverse(eliminated_inverse);
    DeviceArray<Scalar> device_primary_cross(primary_cross_values);
    DeviceArray<Scalar> device_cross(cross_values);
    DeviceArray<Index> device_base_kinds(base_kinds);
    DeviceArray<Index> device_base_indices(base_indices);
    DeviceArray<Index> device_block_slots(value_block_slots);
    DeviceArray<Index> device_rows(local_rows);
    DeviceArray<Index> device_columns(local_columns);
    DeviceArray<Index> device_term_offsets(term_offsets);
    DeviceArray<Index> device_term_eliminated(term_eliminated);
    DeviceArray<Index> device_term_left(term_left_cross);
    DeviceArray<Index> device_term_right(term_right_cross);
    DeviceArray<Scalar> device_output(value_count);

    constexpr int block_size = 256;
    const int grid_size = static_cast<int>((value_count + block_size - 1) / block_size);
    assembleSchurValuesKernel<<<grid_size, block_size>>>(
        static_cast<Index>(value_count), primary_size, eliminated_size,
        device_primary.data(), device_inverse.data(), device_primary_cross.data(),
        device_cross.data(), device_base_kinds.data(), device_base_indices.data(),
        device_block_slots.data(), device_rows.data(), device_columns.data(),
        device_term_offsets.data(),
        device_term_eliminated.data(), device_term_left.data(), device_term_right.data(),
        device_output.data());
    PLAMATRIX_CHECK_CUDA(cudaGetLastError());
    std::vector<Scalar> result(value_count);
    PLAMATRIX_CHECK_CUDA(cudaMemcpy(
        result.data(), device_output.data(), value_count * sizeof(Scalar), cudaMemcpyDeviceToHost));
    return result;
}

} // namespace

template <typename Scalar>
std::vector<Scalar> assembleSchurValuesOnCuda(
    Index primary_size,
    Index eliminated_size,
    const std::vector<Scalar>& primary_diagonal,
    const std::vector<Scalar>& eliminated_inverse,
    const std::vector<Scalar>& primary_cross_values,
    const std::vector<Scalar>& cross_values,
    const std::vector<Index>& base_kinds,
    const std::vector<Index>& base_indices,
    const std::vector<Index>& value_block_slots,
    const std::vector<Index>& local_rows,
    const std::vector<Index>& local_columns,
    const std::vector<Index>& term_offsets,
    const std::vector<Index>& term_eliminated,
    const std::vector<Index>& term_left_cross,
    const std::vector<Index>& term_right_cross)
{
    return assemble(
        primary_size, eliminated_size, primary_diagonal, eliminated_inverse,
        primary_cross_values, cross_values, base_kinds, base_indices,
        value_block_slots, local_rows, local_columns, term_offsets, term_eliminated,
        term_left_cross, term_right_cross);
}

template std::vector<float> assembleSchurValuesOnCuda(
    Index, Index, const std::vector<float>&, const std::vector<float>&,
    const std::vector<float>&, const std::vector<float>&,
    const std::vector<Index>&, const std::vector<Index>&,
    const std::vector<Index>&,
    const std::vector<Index>&, const std::vector<Index>&,
    const std::vector<Index>&, const std::vector<Index>&,
    const std::vector<Index>&, const std::vector<Index>&);
template std::vector<double> assembleSchurValuesOnCuda(
    Index, Index, const std::vector<double>&, const std::vector<double>&,
    const std::vector<double>&, const std::vector<double>&,
    const std::vector<Index>&, const std::vector<Index>&,
    const std::vector<Index>&,
    const std::vector<Index>&, const std::vector<Index>&,
    const std::vector<Index>&, const std::vector<Index>&,
    const std::vector<Index>&, const std::vector<Index>&);

} // namespace plamatrix::block_schur_detail
